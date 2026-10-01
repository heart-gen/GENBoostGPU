"""Task sources: where each locus's genotype, phenotype and covariates come from.

A task source owns a task table (one row per locus) and a ``load(task)``
method returning an :class:`~genboostgpu.adapters.dnam_h2.ObservedLocus`.

* :class:`DnamH2Source` - the ``dna-methylation-heritability`` layout
  (Module 01 run or ``01b_estimation_cells`` cell), loaded exactly like R.
* :class:`RegionSource` - generic inputs: a region table, a samples x regions
  phenotype matrix, a covariate table and genome-wide PLINK genotypes. Used
  for CpH/CpG regions built by ``scripts/build_regions.R`` and any other
  region-level phenotype.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..adapters.dnam_h2 import (
    WINDOW_BP,
    LocusError,
    ObservedLocus,
    load_observed_locus,
    read_run_manifest,
    read_thresholds_min_cis_variants,
)
from ..io.genotype import GenotypeSource, snp_qc_mask
from ..io.phenotype import CovariateSpec, read_phenotype_matrix, read_regions

__all__ = ["DnamH2Source", "RegionSource", "source_from_config"]

TASK_COLUMNS = ["task_id", "cohort", "region", "chrom", "start", "end", "n_cpgs",
                "vmr_id", "vmr_set_id"]


@dataclass
class DnamH2Source:
    """Loci from an accepted Module 01 run or 01b estimation cell."""

    vmr_run_dir: str
    cohort: str
    region: str
    tasks: pd.DataFrame
    min_cis_variants: int
    expected_n: int | None = None
    estimation_group: str | None = None
    catalog_cohort: str | None = None
    covar_prefix: str | None = None
    upstream_vmr_run_id: str = ""
    window_bp: int = WINDOW_BP
    apply_snp_qc: bool = True
    kind: str = "dnam_h2"

    def load(self, task) -> ObservedLocus:
        return load_observed_locus(
            task, self.cohort, self.vmr_run_dir, self.min_cis_variants,
            expected_n=self.expected_n, apply_snp_qc=self.apply_snp_qc,
            covar_prefix=self.covar_prefix, estimation_group=self.estimation_group,
            window_bp=self.window_bp)

    def donors(self) -> list:
        """Donor keys (``FID::IID``) of the upstream run, in file order."""
        path = os.path.join(self.vmr_run_dir, "vmr", "donors_plink.txt")
        tab = pd.read_csv(path, sep=r"\s+", header=None, dtype=str)
        return list(tab[0] + "::" + tab[1])

    def describe(self) -> dict:
        return dict(kind=self.kind, vmr_run_dir=os.path.realpath(self.vmr_run_dir),
                    cohort=self.cohort, region=self.region,
                    min_cis_variants=self.min_cis_variants, expected_n=self.expected_n,
                    estimation_group=self.estimation_group or self.cohort,
                    catalog_cohort=self.catalog_cohort or self.cohort,
                    covar_prefix=self.covar_prefix,
                    upstream_vmr_run_id=self.upstream_vmr_run_id,
                    window_bp=self.window_bp, apply_snp_qc=self.apply_snp_qc)

    @classmethod
    def from_module02_run(cls, run_dir: str, repo_root: str | None = None):
        """Rebuild the exact task universe and inputs of a Module 02 run.

        Used to replay an accepted run: the task table, donor count, covariate
        prefix and estimation group all come from the run's own manifest.
        """
        run_dir = os.path.realpath(run_dir)
        man = read_run_manifest(os.path.join(run_dir, "manifest.tsv"))
        if repo_root is None:
            repo_root = os.path.realpath(os.path.join(run_dir, "..", "..", "..", ".."))
        module = man.get("upstream_module") or "01_vmr_catalog"
        vmr_run_dir = os.path.join(repo_root, module, "_m", "runs",
                                   man["upstream_vmr_run_id"])
        tasks = pd.read_csv(os.path.join(run_dir, "config", "task-manifest.tsv"),
                            sep="\t", dtype={"chrom": str})
        min_cis = read_thresholds_min_cis_variants(
            os.path.join(run_dir, "config", "thresholds.yml"))
        cohort = man["cohort"]
        return cls(vmr_run_dir=vmr_run_dir, cohort=cohort, region=man["region"],
                   tasks=tasks, min_cis_variants=min_cis,
                   expected_n=int(man["n_donors"]),
                   estimation_group=man.get("estimation_group") or cohort,
                   catalog_cohort=man.get("catalog_cohort") or cohort,
                   covar_prefix=man.get("covar_prefix") or None,
                   upstream_vmr_run_id=man["upstream_vmr_run_id"])


@dataclass
class RegionSource:
    """Generic region phenotypes with genome-wide genotypes.

    ``regions`` has ``region_id, chrom, start, end`` (+ optional ``n_sites``);
    ``phenotypes`` is samples x regions (``sample_id`` column + one column per
    region id); ``covariates`` describes the design. Samples are matched by
    ID between phenotype, covariate and genotype tables; any donor missing
    from one table is dropped from all, and the final count is checked
    against ``expected_n`` when given.
    """

    regions: pd.DataFrame
    phenotype_path: str
    genotype_pattern: str
    covariates: CovariateSpec
    cohort: str
    region: str
    region_set_id: str = ""
    min_cis_variants: int = 100
    window_bp: int = WINDOW_BP
    maf_min: float = 0.05
    missing_max: float = 0.05
    expected_n: int | None = None
    genotype_id_column: str = "IID"
    apply_snp_qc: bool = True
    kind: str = "regions"
    _pheno: pd.DataFrame | None = field(default=None, repr=False)
    _geno: GenotypeSource | None = field(default=None, repr=False)

    @property
    def tasks(self) -> pd.DataFrame:
        r = self.regions.reset_index(drop=True)
        return pd.DataFrame(dict(
            task_id=np.arange(1, len(r) + 1), cohort=self.cohort, region=self.region,
            chrom=r["chrom"].astype(str), start=r["start"].astype(np.int64),
            end=r["end"].astype(np.int64),
            n_cpgs=r["n_sites"] if "n_sites" in r.columns else pd.NA,
            vmr_id=r["region_id"].astype(str), vmr_set_id=self.region_set_id))

    def _ensure(self):
        if self._pheno is None:
            self._pheno = read_phenotype_matrix(self.phenotype_path)
        if self._geno is None:
            self._geno = GenotypeSource(self.genotype_pattern)

    def load(self, task) -> ObservedLocus:
        self._ensure()
        chrom = str(task["chrom"]).removeprefix("chr")
        if chrom.upper() in ("X", "Y"):
            return ObservedLocus("excluded", "non_autosomal_vmr")
        start = max(1, int(task["start"]) - self.window_bp)
        end = int(task["end"]) + self.window_bp
        geno, variants = self._geno.window(chrom, start, end)
        if geno.shape[1] == 0:
            return ObservedLocus("qc_failed", "no_snp_in_prespecified_cis_window")
        snps_in_window = int(geno.shape[1])
        rid = str(task["vmr_id"])
        if rid not in self._pheno.columns:
            raise LocusError(f"Region {rid} absent from the phenotype matrix")
        samples = self._geno.samples(chrom)
        gid = samples[self.genotype_id_column].astype(str).to_numpy()
        pheno = self._pheno[["sample_id", rid]].rename(columns={rid: "phenotype"})
        design, names, meta = self.covariates.design_for(pheno, gid)
        geno = geno[meta["_geno_row"].to_numpy()]
        # Variant QC on the analysis donors (genome-wide files usually carry
        # more donors than were phenotyped).
        keep = snp_qc_mask(geno, self.maf_min, self.missing_max)
        if int(keep.sum()) < self.min_cis_variants:
            return ObservedLocus("qc_failed", "fewer_than_min_cis_variants",
                                 snps_in_window=snps_in_window)
        if self.apply_snp_qc:
            geno = geno[:, keep]
            variants = variants.loc[keep].reset_index(drop=True)
        y = meta["phenotype"].to_numpy(dtype=float)
        if self.expected_n is not None and y.size != int(self.expected_n):
            raise LocusError(f"Observed donor count differs from locked design: "
                             f"{y.size} versus {self.expected_n}")
        return ObservedLocus("ok", None, geno, y, design, names, meta,
                             snps_in_window, variants, self.genotype_pattern,
                             os.path.realpath(self.phenotype_path))

    def describe(self) -> dict:
        return dict(kind=self.kind, phenotype_path=os.path.realpath(self.phenotype_path),
                    genotype_pattern=self.genotype_pattern,
                    covariates=self.covariates.describe(), cohort=self.cohort,
                    region=self.region, region_set_id=self.region_set_id,
                    min_cis_variants=self.min_cis_variants, window_bp=self.window_bp,
                    maf_min=self.maf_min, missing_max=self.missing_max,
                    expected_n=self.expected_n,
                    genotype_id_column=self.genotype_id_column,
                    apply_snp_qc=self.apply_snp_qc)

    def donors(self) -> list:
        """Sample IDs with a phenotype, in phenotype-table order."""
        self._ensure()
        return list(self._pheno["sample_id"])


def source_from_config(cfg: dict, tasks: pd.DataFrame):
    """Rebuild a task source from a run's ``run.json`` ``source`` block."""
    kind = cfg["kind"]
    if kind == "dnam_h2":
        return DnamH2Source(
            vmr_run_dir=cfg["vmr_run_dir"], cohort=cfg["cohort"], region=cfg["region"],
            tasks=tasks, min_cis_variants=int(cfg["min_cis_variants"]),
            expected_n=cfg.get("expected_n"), estimation_group=cfg.get("estimation_group"),
            catalog_cohort=cfg.get("catalog_cohort"), covar_prefix=cfg.get("covar_prefix"),
            upstream_vmr_run_id=cfg.get("upstream_vmr_run_id", ""),
            window_bp=int(cfg.get("window_bp", WINDOW_BP)),
            apply_snp_qc=bool(cfg.get("apply_snp_qc", True)))
    if kind == "regions":
        regions = tasks.rename(columns={"vmr_id": "region_id", "n_cpgs": "n_sites"})
        return RegionSource(
            regions=regions[["region_id", "chrom", "start", "end", "n_sites"]],
            phenotype_path=cfg["phenotype_path"], genotype_pattern=cfg["genotype_pattern"],
            covariates=CovariateSpec.from_describe(cfg["covariates"]),
            cohort=cfg["cohort"], region=cfg["region"],
            region_set_id=cfg.get("region_set_id", ""),
            min_cis_variants=int(cfg["min_cis_variants"]),
            window_bp=int(cfg["window_bp"]), maf_min=float(cfg["maf_min"]),
            missing_max=float(cfg["missing_max"]), expected_n=cfg.get("expected_n"),
            genotype_id_column=cfg.get("genotype_id_column", "IID"),
            apply_snp_qc=bool(cfg.get("apply_snp_qc", True)))
    raise ValueError(f"Unknown task source kind: {kind}")


def read_regions_source(**kwargs) -> RegionSource:
    """Convenience constructor reading the region table from disk."""
    kwargs = dict(kwargs)
    kwargs["regions"] = read_regions(kwargs.pop("regions_path"))
    return RegionSource(**kwargs)
