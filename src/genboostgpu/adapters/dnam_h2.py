"""Adapter for the ``dna-methylation-heritability`` analysis repository.

Reads one VMR exactly as ``00_shared/locus_io.R::load_observed_locus`` does
from an accepted ``01_vmr_catalog`` run or a ``01b_estimation_cells`` cell:

* per-VMR PLINK BED ``plink_format/chr_{c}/TOPMed_LIBD-{group}.{start}_{end}.bed``
  (a ``.no-snps`` marker means no variant in the cis window);
* phenotype ``vmr/phenotypes/{chrom}_{start}_{end}_meth.phen`` (FID IID value);
* covariates ``covs/chr_{c}/{prefix}.covar`` (sex, diagnosis) and
  ``.qcovar`` (age), plus optional ``covs/genotype_pcs.tsv`` (snpPC*);
* a +-500 kb window, MAF >= 0.05 and missingness <= 0.05 computed on every
  donor in the BED, and the ``min_cis_variants`` gate;
* donors aligned by an inner merge on (FID, IID). R's ``merge()`` sorts rows
  by the pasted key, and that row order fixes every fold assignment, so it is
  reproduced here;
* covariates ``model.matrix(~ age + factor(sex) + factor(diagnosis) + PCs)``
  without the intercept (treatment coding, levels in sorted order).

Statuses are ``ok``, ``qc_failed`` and ``excluded``, as in R; anything R would
``stop()`` on raises :class:`LocusError`.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..io.genotype import read_bed, snp_qc_mask

__all__ = ["LocusError", "ObservedLocus", "load_locus_phenotype",
           "load_observed_locus", "covariate_design", "read_run_manifest",
           "read_thresholds_min_cis_variants", "WINDOW_BP"]

WINDOW_BP = 500_000


class LocusError(RuntimeError):
    """An input condition under which the R locus reader stops."""


@dataclass
class ObservedLocus:
    status: str
    reason: str | None = None
    genotype: np.ndarray | None = None
    y: np.ndarray | None = None
    covariates: np.ndarray | None = None
    covariate_names: list = field(default_factory=list)
    metadata: pd.DataFrame | None = None
    snps_in_window: int | None = None
    variants: pd.DataFrame | None = None
    plink_source: str | None = None
    phenotype_source: str | None = None


def _read_ws(path, names):
    tab = pd.read_csv(path, sep=r"\s+", header=None, dtype=str)
    if tab.shape[1] != len(names):
        raise LocusError(f"{path} has {tab.shape[1]} columns, expected {len(names)}")
    tab.columns = names
    return tab


def _chrom_label(chrom) -> str:
    return re.sub(r"^chr", "", str(chrom), flags=re.IGNORECASE)


def load_locus_phenotype(task, vmr_run_dir, cohort=None, covar_prefix=None) -> dict:
    """Port of ``load_locus_phenotype``: unmerged phenotype/covariate tables."""
    label = _chrom_label(task["chrom"])
    chromosome_dir = f"chr_{label}"
    stem = f"{int(task['start'])}_{int(task['end'])}"
    if covar_prefix is not None:
        prefix = covar_prefix
    elif cohort == "AA":
        prefix = "TOPMed_LIBD.AA"
    else:
        prefix = "TOPMed_LIBD"
    phenotype_path = os.path.join(vmr_run_dir, "vmr", "phenotypes",
                                  f"{task['chrom']}_{stem}_meth.phen")
    covar_path = os.path.join(vmr_run_dir, "covs", chromosome_dir, prefix + ".covar")
    qcovar_path = os.path.join(vmr_run_dir, "covs", chromosome_dir, prefix + ".qcovar")
    for path in (phenotype_path, covar_path, qcovar_path):
        if not os.path.exists(path):
            raise LocusError(f"Missing observed input: {path}")
    phen = _read_ws(phenotype_path, ["FID", "IID", "phenotype"])
    covar = _read_ws(covar_path, ["FID", "IID", "sex", "diagnosis"])
    qcovar = _read_ws(qcovar_path, ["FID", "IID", "age"])
    pc_path = os.path.join(vmr_run_dir, "covs", "genotype_pcs.tsv")
    pcs = None
    pc_names: list[str] = []
    if os.path.exists(pc_path):
        pcs = pd.read_csv(pc_path, sep="\t", dtype=str)
        if not {"FID", "IID"} <= set(pcs.columns):
            raise LocusError(f"genotype_pcs.tsv lacks FID/IID: {pc_path}")
        pc_names = [c for c in pcs.columns if re.fullmatch(r"snpPC[0-9]+", c)]
        if not pc_names:
            raise LocusError(f"genotype_pcs.tsv carries no snpPC columns: {pc_path}")
        pc_names.sort(key=lambda c: int(c[5:]))
        pcs = pcs[["FID", "IID"] + pc_names].copy()
        for nm in pc_names:
            pcs[nm] = pd.to_numeric(pcs[nm], errors="coerce")
        if pcs[pc_names].isna().to_numpy().any():
            raise LocusError(f"Non-numeric or missing genotype PC values in {pc_path}")
        if (pcs["FID"] + "::" + pcs["IID"]).duplicated().any():
            raise LocusError(f"Duplicate donors in {pc_path}")
    return dict(phenotype=phen, covar=covar, qcovar=qcovar, pcs=pcs,
                pc_names=pc_names, pc_path=pc_path if pc_names else None,
                phenotype_source=os.path.realpath(phenotype_path),
                covar_source=os.path.realpath(covar_path))


def _r_merge(x: pd.DataFrame, y: pd.DataFrame) -> pd.DataFrame:
    """``merge(x, y, by = c("FID", "IID"), all = FALSE)`` row order: sorted by
    the key ``paste(FID, IID, sep = "\\r")``."""
    out = x.merge(y, on=["FID", "IID"], how="inner")
    key = (out["FID"] + "\r" + out["IID"]).to_numpy()
    order = np.argsort(key, kind="stable")
    return out.iloc[order].reset_index(drop=True)


def _factor_levels(values):
    return sorted(pd.unique(values))


def covariate_design(metadata: pd.DataFrame, pc_names=()):
    """``model.matrix(~ age + factor(sex) + factor(diagnosis) + PCs)[, -1]``."""
    cols = [metadata["age"].astype(float).to_numpy()]
    names = ["age"]
    for var in ("sex", "diagnosis"):
        levels = _factor_levels(metadata[var])
        for lev in levels[1:]:
            cols.append((metadata[var] == lev).astype(float).to_numpy())
            names.append(f"factor({var}){lev}")
    for nm in pc_names:
        cols.append(metadata[nm].astype(float).to_numpy())
        names.append(nm)
    return np.column_stack(cols), names


def load_observed_locus(task, cohort, vmr_run_dir, min_cis_variants,
                        expected_n=None, apply_snp_qc=True, covar_prefix=None,
                        estimation_group=None, window_bp=WINDOW_BP) -> ObservedLocus:
    """Port of ``load_observed_locus`` (see module docstring)."""
    if estimation_group is None:
        estimation_group = cohort.split(".", 1)[1] if "." in str(cohort) else cohort
    label = _chrom_label(task["chrom"])
    if label.upper() in ("X", "Y"):
        return ObservedLocus("excluded", "non_autosomal_vmr")
    chromosome_dir = f"chr_{label}"
    stem = f"{int(task['start'])}_{int(task['end'])}"
    bed = os.path.join(vmr_run_dir, "plink_format", chromosome_dir,
                       f"TOPMed_LIBD-{estimation_group}.{stem}.bed")
    if not os.path.exists(bed):
        if os.path.exists(bed[:-4] + ".no-snps"):
            return ObservedLocus("qc_failed", "no_snp_in_prespecified_cis_window")
        raise LocusError(f"Missing PLINK BED: {bed}")
    pheno = load_locus_phenotype(task, vmr_run_dir, cohort=cohort,
                                 covar_prefix=covar_prefix)
    genotype, bim, fam = read_bed(bed)
    window_start = max(1, int(task["start"]) - window_bp)
    window_end = int(task["end"]) + window_bp
    map_chr = bim["chrom"].astype(str).str.replace(r"^chr", "", regex=True, case=False)
    in_window = ((map_chr == label) & (bim["pos"] >= window_start)
                 & (bim["pos"] <= window_end)).to_numpy()
    if not in_window.any():
        return ObservedLocus("qc_failed", "no_snp_in_prespecified_cis_window")
    genotype = genotype[:, in_window]
    variants = bim.loc[in_window].reset_index(drop=True)
    snps_in_window = int(genotype.shape[1])
    keep_snp = snp_qc_mask(genotype)
    if int(keep_snp.sum()) < int(min_cis_variants):
        return ObservedLocus("qc_failed", "fewer_than_min_cis_variants",
                             snps_in_window=snps_in_window)
    if apply_snp_qc:
        genotype = genotype[:, keep_snp]
        variants = variants.loc[keep_snp].reset_index(drop=True)

    fam = fam[["FID", "IID"]].copy()
    tables = [fam, pheno["phenotype"], pheno["covar"], pheno["qcovar"]]
    if pheno["pcs"] is not None:
        fam_key = set(fam["FID"] + "::" + fam["IID"])
        pc_key = set(pheno["pcs"]["FID"] + "::" + pheno["pcs"]["IID"])
        missing = sorted(fam_key - pc_key)
        if missing:
            raise LocusError(
                f"Donors present in the locus BED but absent from {pheno['pc_path']}: "
                + ", ".join(missing[:10]))
        tables.append(pheno["pcs"])
    metadata = tables[0]
    for tab in tables[1:]:
        metadata = _r_merge(metadata, tab)
    meta_key = metadata["FID"] + "::" + metadata["IID"]
    fam_key = fam["FID"] + "::" + fam["IID"]
    pos = pd.Series(np.arange(len(fam)), index=fam_key.to_numpy())
    if meta_key.duplicated().any() or not meta_key.isin(pos.index).all():
        raise LocusError("Donor alignment failed or produced duplicate IDs")
    row_index = pos.loc[meta_key.to_numpy()].to_numpy()
    genotype = genotype[row_index]
    phen_num = pd.to_numeric(metadata["phenotype"], errors="coerce")
    age_num = pd.to_numeric(metadata["age"], errors="coerce")
    keep_sample = (np.isfinite(phen_num.to_numpy(dtype=float))
                   & np.isfinite(age_num.to_numpy(dtype=float))
                   & metadata["sex"].notna().to_numpy()
                   & metadata["diagnosis"].notna().to_numpy())
    metadata = metadata.loc[keep_sample].reset_index(drop=True)
    genotype = genotype[keep_sample]
    if expected_n is not None and genotype.shape[0] != int(expected_n):
        raise LocusError(
            f"Observed donor count differs from locked design: {genotype.shape[0]} "
            f"versus {expected_n}")
    y = pd.to_numeric(metadata["phenotype"]).to_numpy(dtype=float)
    covariates, names = covariate_design(metadata, pheno["pc_names"])
    return ObservedLocus("ok", None, genotype, y, covariates, names, metadata,
                         snps_in_window, variants, os.path.realpath(bed),
                         pheno["phenotype_source"])


def read_run_manifest(path) -> dict:
    """A ``field\\tvalue`` manifest as a dict."""
    tab = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    return dict(zip(tab["field"], tab["value"]))


def read_thresholds_min_cis_variants(path) -> int:
    """``min_cis_variants`` from a run's ``thresholds.yml`` (same regex as R)."""
    with open(path) as fh:
        hits = [ln for ln in fh if re.match(r"^\s+min_cis_variants:", ln)]
    if len(hits) != 1:
        raise LocusError("Cannot resolve min_cis_variants")
    return int(re.sub(r".*:\s*", "", hits[0]).split("#")[0].strip())
