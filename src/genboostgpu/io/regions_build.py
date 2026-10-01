"""Collect ``scripts/build_regions.R`` outputs into GENBoostGPU inputs."""
from __future__ import annotations

import glob
import os
import re

import pandas as pd

__all__ = ["merge_build"]


def _chrom_key(path):
    m = re.search(r"chr([^.]+)\.", os.path.basename(path))
    c = m.group(1) if m else ""
    return (0, int(c)) if c.isdigit() else (1, c)


def merge_build(build_dir: str, out_prefix: str | None = None) -> dict:
    """Write ``regions.tsv`` and ``phenotypes.parquet`` from per-chromosome
    ``regions/chr*.regions.tsv.gz`` / ``chr*.pheno.tsv.gz`` files."""
    out_prefix = out_prefix or build_dir
    reg_files = sorted(glob.glob(os.path.join(build_dir, "regions", "chr*.regions.tsv.gz")),
                       key=_chrom_key)
    if not reg_files:
        raise FileNotFoundError(f"No regions/chr*.regions.tsv.gz under {build_dir}")
    regions, phenos = [], []
    sample_ids = None
    for rf in reg_files:
        r = pd.read_csv(rf, sep="\t", dtype={"chrom": str})
        pf = rf.replace(".regions.tsv.gz", ".pheno.tsv.gz")
        p = pd.read_csv(pf, sep="\t", dtype={"sample_id": str}, float_precision="round_trip")
        if sample_ids is None:
            sample_ids = list(p["sample_id"])
        elif list(p["sample_id"]) != sample_ids:
            raise ValueError(f"Sample order differs in {pf}")
        if list(p.columns[1:]) != list(r["region_id"]):
            raise ValueError(f"Phenotype columns do not match regions in {pf}")
        regions.append(r)
        phenos.append(p.drop(columns=["sample_id"]))
    regions = pd.concat(regions, ignore_index=True)
    if regions["region_id"].duplicated().any():
        raise ValueError("Duplicate region_id across chromosomes")
    pheno = pd.concat([pd.DataFrame({"sample_id": sample_ids})] + phenos, axis=1)
    reg_path = out_prefix.rstrip("/") + "/regions.tsv" if os.path.isdir(out_prefix) \
        else out_prefix + ".regions.tsv"
    ph_path = reg_path.replace("regions.tsv", "phenotypes.parquet")
    regions.to_csv(reg_path, sep="\t", index=False)
    pheno.to_parquet(ph_path, index=False)
    return dict(regions=reg_path, phenotypes=ph_path, n_regions=len(regions),
                n_samples=len(sample_ids))
