"""Genotype readers for the local-genetic-variance engine.

* ``.bed/.bim/.fam`` (PLINK 1) is read natively with the same allele coding
  as ``bigsnpr::snp_readBed`` (the reader Module 02 uses): the dosage counts
  copies of the first ``.bim`` allele (A1, column 5), missing calls are NaN.
  pandas-plink counts the other allele; using it would change GEMMA's input
  and the sign of every elastic-net coefficient.
* ``.pgen/.pvar/.psam`` (PLINK 2) is read with ``pgenlib``, counting ALT.

:class:`GenotypeSource` caches one chromosome at a time on the host and
returns window slices, so genome-wide inputs are read once per chromosome
rather than once per locus.
"""
from __future__ import annotations

import os
import threading
from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = [
    "read_bim",
    "read_fam",
    "read_bed",
    "write_bed",
    "read_pvar",
    "read_psam",
    "read_pgen",
    "GenotypeSource",
    "snp_qc_mask",
]

# 2-bit PLINK codes (00 hom A1, 01 missing, 10 het, 11 hom A2) -> A1 dosage,
# the bigsnpr convention.
_BED_LUT = np.array([2.0, np.nan, 1.0, 0.0])


def _strip(prefix: str, exts) -> str:
    for ext in exts:
        if prefix.endswith(ext):
            return prefix[: -len(ext)]
    return prefix


def read_bim(prefix: str) -> pd.DataFrame:
    prefix = _strip(prefix, (".bed", ".bim", ".fam"))
    bim = pd.read_csv(prefix + ".bim", sep=r"\s+", header=None,
                      names=["chrom", "snp", "cm", "pos", "a1", "a2"],
                      dtype={"chrom": str, "snp": str, "a1": str, "a2": str})
    return bim


def read_fam(prefix: str) -> pd.DataFrame:
    prefix = _strip(prefix, (".bed", ".bim", ".fam"))
    fam = pd.read_csv(prefix + ".fam", sep=r"\s+", header=None,
                      names=["FID", "IID", "father", "mother", "sex", "pheno"],
                      dtype=str)
    return fam


def read_bed(prefix: str, variant_index=None, dtype=np.float64, bim=None, fam=None):
    """Read a PLINK1 ``.bed`` as an ``n_samples x n_variants`` dosage matrix.

    Returns ``(genotype, bim, fam)``; ``variant_index`` (0-based) restricts
    the columns read (only those rows of the memory-mapped file are touched).
    ``bim``/``fam`` may be passed when already read.
    """
    prefix = _strip(prefix, (".bed", ".bim", ".fam"))
    bim = read_bim(prefix) if bim is None else bim
    fam = read_fam(prefix) if fam is None else fam
    n, m = len(fam), len(bim)
    bytes_per = (n + 3) // 4
    with open(prefix + ".bed", "rb") as fh:
        magic = fh.read(3)
        if magic != b"\x6c\x1b\x01":
            raise ValueError(f"{prefix}.bed is not a SNP-major PLINK1 .bed")
    if os.path.getsize(prefix + ".bed") != 3 + bytes_per * m:
        raise ValueError(f"{prefix}.bed size does not match .bim/.fam")
    raw = np.memmap(prefix + ".bed", dtype=np.uint8, mode="r", offset=3,
                    shape=(m, bytes_per))
    if variant_index is not None:
        variant_index = np.asarray(variant_index, dtype=np.int64)
        raw = raw[variant_index]
        bim = bim.iloc[variant_index].reset_index(drop=True)
    raw = np.asarray(raw)
    codes = np.empty((raw.shape[0], bytes_per * 4), dtype=np.uint8)
    for k in range(4):
        codes[:, k::4] = (raw >> (2 * k)) & 0b11
    geno = _BED_LUT[codes[:, :n]].T.astype(dtype, copy=False)
    return np.ascontiguousarray(geno), bim, fam


def write_bed(prefix: str, genotype, bim: pd.DataFrame, fam: pd.DataFrame) -> None:
    """Write ``n_samples x n_variants`` A1 dosages (NaN = missing) as PLINK1."""
    g = np.asarray(genotype, dtype=np.float64)
    n, m = g.shape
    if len(bim) != m or len(fam) != n:
        raise ValueError("genotype shape does not match bim/fam")
    code = np.full(g.shape, 1, dtype=np.uint8)          # missing
    code[g == 2] = 0
    code[g == 1] = 2
    code[g == 0] = 3
    bytes_per = (n + 3) // 4
    padded = np.zeros((m, bytes_per * 4), dtype=np.uint8)
    padded[:, :n] = code.T
    packed = np.zeros((m, bytes_per), dtype=np.uint8)
    for k in range(4):
        packed |= (padded[:, k::4] << (2 * k)).astype(np.uint8)
    with open(prefix + ".bed", "wb") as fh:
        fh.write(b"\x6c\x1b\x01")
        fh.write(packed.tobytes())
    bim[["chrom", "snp", "cm", "pos", "a1", "a2"]].to_csv(
        prefix + ".bim", sep="\t", header=False, index=False)
    fam[["FID", "IID", "father", "mother", "sex", "pheno"]].to_csv(
        prefix + ".fam", sep=" ", header=False, index=False)


def read_pvar(prefix: str) -> pd.DataFrame:
    prefix = _strip(prefix, (".pgen", ".pvar", ".psam"))
    path = prefix + ".pvar"
    header = None
    skip = 0
    with open(path) as fh:
        for line in fh:
            if line.startswith("##"):
                skip += 1
                continue
            if line.startswith("#CHROM"):
                header = line.lstrip("#").strip().split("\t")
                skip += 1
            break
    if header is None:  # .bim-like pvar without header
        tab = pd.read_csv(path, sep=r"\s+", header=None, skiprows=skip, dtype=str)
        tab.columns = ["CHROM", "ID", "CM", "POS", "ALT", "REF"][: tab.shape[1]]
    else:
        # POS parsed as integers directly: ~40% faster on a genome-wide pvar
        tab = pd.read_csv(path, sep="\t", header=None, skiprows=skip, names=header,
                          dtype={c: (np.int64 if c == "POS" else str) for c in header})
    tab["POS"] = tab["POS"].astype(np.int64)
    return tab.rename(columns={"CHROM": "chrom", "ID": "snp", "POS": "pos",
                               "REF": "ref", "ALT": "alt"})


def read_psam(prefix: str) -> pd.DataFrame:
    """``.psam`` with or without a header (headerless files are FID IID ...)."""
    prefix = _strip(prefix, (".pgen", ".pvar", ".psam"))
    path = prefix + ".psam"
    with open(path) as fh:
        first = fh.readline()
    if first.startswith("#"):
        tab = pd.read_csv(path, sep=r"\s+", dtype=str)
        tab.columns = [c.lstrip("#") for c in tab.columns]
        if "FID" not in tab.columns:
            tab.insert(0, "FID", tab["IID"])
    else:
        tab = pd.read_csv(path, sep=r"\s+", header=None, dtype=str)
        names = ["FID", "IID", "PAT", "MAT", "SEX", "PHENO"]
        tab.columns = names[: tab.shape[1]] + [f"V{i}" for i in range(tab.shape[1] - len(names))]
    return tab


def read_pgen(prefix: str, variant_index=None, sample_index=None, dtype=np.float64,
              pvar=None, psam=None):
    """Read a PLINK2 ``.pgen`` (hardcalls, ALT counts; missing = NaN).

    With ``dtype=np.int8`` the raw counts are returned (missing = -9).
    ``pvar``/``psam`` may be passed when already read.
    """
    import pgenlib

    prefix = _strip(prefix, (".pgen", ".pvar", ".psam"))
    pvar = read_pvar(prefix) if pvar is None else pvar
    psam = read_psam(prefix) if psam is None else psam
    n_all = len(psam)
    if sample_index is not None:
        sample_index = np.asarray(sample_index, dtype=np.uint32)
        sample_index.sort()
        reader = pgenlib.PgenReader((prefix + ".pgen").encode(), raw_sample_ct=n_all,
                                    sample_subset=sample_index)
        n = sample_index.size
        psam = psam.iloc[sample_index].reset_index(drop=True)
    else:
        reader = pgenlib.PgenReader((prefix + ".pgen").encode(), raw_sample_ct=n_all)
        n = n_all
    if variant_index is None:
        variant_index = np.arange(len(pvar), dtype=np.uint32)
    else:
        variant_index = np.asarray(variant_index, dtype=np.uint32)
    out = np.empty((variant_index.size, n), dtype=np.int8)
    if variant_index.size:
        reader.read_list(variant_index, out)
    reader.close()
    if np.dtype(dtype) == np.int8:
        geno = out.T
    else:
        geno = out.T.astype(dtype)
        geno[out.T < 0] = np.nan
    pvar = pvar.iloc[variant_index.astype(np.int64)].reset_index(drop=True)
    return np.ascontiguousarray(geno), pvar, psam


def snp_qc_mask(genotype, maf_min: float = 0.05, missing_max: float = 0.05):
    """Module 02 variant QC: MAF >= ``maf_min`` and missingness <= ``missing_max``."""
    g = np.asarray(genotype, dtype=np.float64)
    missing = np.isnan(g).mean(axis=0)
    with np.errstate(invalid="ignore"):
        af = np.nanmean(g, axis=0) / 2.0
    maf = np.minimum(af, 1.0 - af)
    return np.isfinite(maf) & (maf >= maf_min) & np.isfinite(missing) & (missing <= missing_max)


@dataclass
class _ChromCache:
    chrom: str
    genotype: np.ndarray        # int8 dosages, -9 = missing
    variants: pd.DataFrame


def _chrom_codes(values):
    """(codes, names): ``names[codes]`` is ``values`` without a ``chr``
    prefix. Each distinct name is normalized once: a genome-wide table has
    millions of rows but a few dozen chromosome names."""
    codes, names = pd.factorize(pd.Series(values), use_na_sentinel=False)
    names = pd.Series(names, dtype=str).str.replace("^chr", "", regex=True).to_numpy()
    return codes, names


def _norm_chrom(values) -> np.ndarray:
    codes, names = _chrom_codes(values)
    return names[codes]


class GenotypeSource:
    """Per-chromosome genotype access for PLINK files.

    ``pattern`` is either per-chromosome (contains ``{chrom}``, e.g.
    ``/path/LIBD.chr{chrom}.AA``) or one genome-wide prefix (e.g.
    ``/path/TOPMed_LIBD.AA``); ``.bed`` and ``.pgen`` filesets are detected.
    For a genome-wide file only the requested chromosome's variants are read.
    One chromosome is held in host memory at a time as int8 dosages;
    :meth:`window` returns float64 (NaN = missing) slices of a region.
    """

    def __init__(self, pattern: str, fmt: str | None = None, sample_ids=None):
        self.pattern = pattern
        self.fmt = fmt
        self.sample_ids = None if sample_ids is None else list(sample_ids)
        self.genome_wide = "{chrom}" not in pattern
        self._cache: _ChromCache | None = None
        self._tables: dict = {}
        self._chrom_rows: dict = {}
        # prep threads share one source; without the lock each of them would
        # read the same chromosome concurrently on a cold cache
        self._lock = threading.RLock()

    def _prefix(self, chrom: str) -> str:
        if self.genome_wide:
            return _strip(self.pattern, (".pgen", ".pvar", ".psam", ".bed", ".bim", ".fam"))
        return self.pattern.format(chrom=str(chrom).removeprefix("chr"))

    def _detect(self, prefix: str) -> str:
        if self.fmt:
            return self.fmt
        if os.path.exists(prefix + ".pgen"):
            return "pgen"
        if os.path.exists(prefix + ".bed"):
            return "bed"
        raise FileNotFoundError(f"No .pgen or .bed for {prefix}")

    def _read_tables(self, prefix: str):
        """(fmt, variants, samples) for a fileset, cached per prefix."""
        with self._lock:
            return self._read_tables_locked(prefix)

    def _read_tables_locked(self, prefix: str):
        if prefix not in self._tables:
            fmt = self._detect(prefix)
            if fmt == "pgen":
                tabs = (fmt, read_pvar(prefix), read_psam(prefix))
            else:
                tabs = (fmt, read_bim(prefix), read_fam(prefix))
            if not self.genome_wide:
                self._tables.clear()   # per-chromosome files: keep one
                self._chrom_rows.clear()
            self._tables[prefix] = tabs
        return self._tables[prefix]

    def _rows_of(self, prefix: str, var: pd.DataFrame, chrom: str) -> np.ndarray:
        """Variant rows of ``chrom``; the index is built once per fileset."""
        if prefix not in self._chrom_rows:
            codes, names = _chrom_codes(var["chrom"])
            self._chrom_rows[prefix] = (codes, names)
        codes, names = self._chrom_rows[prefix]
        return np.flatnonzero(np.isin(codes, np.flatnonzero(names == chrom)))

    def samples(self, chrom) -> pd.DataFrame:
        _, _, tab = self._read_tables(self._prefix(chrom))
        return tab[["FID", "IID"]].reset_index(drop=True)

    def load_chromosome(self, chrom) -> _ChromCache:
        chrom = str(chrom).removeprefix("chr")
        cache = self._cache
        if cache is not None and cache.chrom == chrom:
            return cache
        with self._lock:
            return self._load_chromosome_locked(chrom)

    def _load_chromosome_locked(self, chrom) -> _ChromCache:
        if self._cache is not None and self._cache.chrom == chrom:
            return self._cache
        self._cache = None     # release the previous chromosome first
        prefix = self._prefix(chrom)
        fmt, var, sam = self._read_tables(prefix)
        idx = self._rows_of(prefix, var, chrom)
        if fmt == "pgen":
            geno, var, _ = read_pgen(prefix, variant_index=idx, dtype=np.int8,
                                     pvar=var, psam=sam)
        else:
            geno, var, _ = read_bed(prefix, variant_index=idx, dtype=np.float32,
                                    bim=var, fam=sam)
            geno = np.where(np.isnan(geno), -9, geno).astype(np.int8)
        var = var.copy()
        var["chrom"] = _norm_chrom(var["chrom"])
        self._cache = _ChromCache(chrom, np.ascontiguousarray(geno), var)
        return self._cache

    def window(self, chrom, start: int, end: int):
        """Genotype columns with ``start <= pos <= end`` on ``chrom``."""
        cache = self.load_chromosome(chrom)
        pos = cache.variants["pos"].to_numpy()
        sel = np.flatnonzero((pos >= start) & (pos <= end))
        raw = cache.genotype[:, sel]
        geno = raw.astype(np.float64)
        geno[raw < 0] = np.nan
        return geno, cache.variants.iloc[sel].reset_index(drop=True)
