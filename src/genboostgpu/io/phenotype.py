"""Generic phenotype, region and covariate inputs.

Region phenotypes (CpG/CpH VMRs, tiles, or any other region-level trait)
arrive as a region table plus a samples x regions matrix with an explicit
``sample_id`` column. Rows are matched to genotypes and covariates by ID,
never by position.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

__all__ = ["read_regions", "read_phenotype_matrix", "CovariateSpec"]


def _read_table(path: str) -> pd.DataFrame:
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    # round_trip: pandas' default fast parser can be 1 ulp off on %.17g text
    return pd.read_csv(path, sep="\t", compression="infer", float_precision="round_trip")


def read_regions(path: str) -> pd.DataFrame:
    """Region table with ``region_id, chrom, start, end`` (+ ``n_sites``)."""
    tab = _read_table(path)
    required = {"region_id", "chrom", "start", "end"}
    missing = required - set(tab.columns)
    if missing:
        raise ValueError(f"Region table lacks columns: {sorted(missing)}")
    if tab["region_id"].duplicated().any():
        raise ValueError("Duplicate region_id in region table")
    tab = tab.copy()
    tab["chrom"] = tab["chrom"].astype(str)
    if "n_sites" not in tab.columns:
        tab["n_sites"] = pd.NA
    return tab


def read_phenotype_matrix(path: str) -> pd.DataFrame:
    """Samples x regions; requires a ``sample_id`` column."""
    tab = _read_table(path)
    if "sample_id" not in tab.columns:
        raise ValueError(f"{path} lacks a sample_id column")
    tab["sample_id"] = tab["sample_id"].astype(str)
    if tab["sample_id"].duplicated().any():
        raise ValueError(f"Duplicate sample_id in {path}")
    return tab


@dataclass
class CovariateSpec:
    """Covariate design for generic inputs.

    ``path`` is a table with ``sample_id``; ``numeric`` columns enter as is
    and ``factors`` are treatment-coded with levels in sorted order (R's
    ``model.matrix`` convention). Rows with a missing value in any used
    column are dropped, as R's complete-case handling would.
    """

    path: str | None = None
    numeric: list = field(default_factory=list)
    factors: list = field(default_factory=list)
    _table: pd.DataFrame | None = field(default=None, repr=False)

    def table(self) -> pd.DataFrame | None:
        if self.path is None:
            return None
        if self._table is None:
            tab = _read_table(self.path)
            if "sample_id" not in tab.columns:
                raise ValueError(f"{self.path} lacks a sample_id column")
            tab["sample_id"] = tab["sample_id"].astype(str)
            if tab["sample_id"].duplicated().any():
                raise ValueError(f"Duplicate sample_id in {self.path}")
            self._table = tab
        return self._table

    def design_for(self, pheno: pd.DataFrame, genotype_ids):
        """Align phenotype, covariates and genotype rows by ``sample_id``.

        Returns (design matrix, column names, metadata with ``_geno_row``).
        Donor order is the phenotype table's order.
        """
        gid = pd.Series(np.arange(len(genotype_ids)), index=pd.Index(genotype_ids))
        if gid.index.duplicated().any():
            raise ValueError("Duplicate sample IDs in the genotype file")
        meta = pheno.copy()
        meta = meta[meta["sample_id"].isin(gid.index)]
        tab = self.table()
        cols = list(self.numeric) + list(self.factors)
        if tab is not None:
            missing = set(cols) - set(tab.columns)
            if missing:
                raise ValueError(f"Covariate table lacks columns: {sorted(missing)}")
            meta = meta.merge(tab[["sample_id"] + cols], on="sample_id", how="inner")
        meta["phenotype"] = pd.to_numeric(meta["phenotype"], errors="coerce")
        keep = np.isfinite(meta["phenotype"].to_numpy(dtype=float))
        for c in cols:
            keep &= meta[c].notna().to_numpy()
        meta = meta.loc[keep].reset_index(drop=True)
        meta["_geno_row"] = gid.loc[meta["sample_id"].to_numpy()].to_numpy()
        design_cols, names = [], []
        for c in self.numeric:
            design_cols.append(pd.to_numeric(meta[c]).to_numpy(dtype=float))
            names.append(c)
        for c in self.factors:
            levels = sorted(pd.unique(meta[c].astype(str)))
            for lev in levels[1:]:
                design_cols.append((meta[c].astype(str) == lev).astype(float).to_numpy())
                names.append(f"factor({c}){lev}")
        design = np.column_stack(design_cols) if design_cols else np.zeros((len(meta), 0))
        return design, names, meta

    def describe(self) -> dict:
        return dict(path=None if self.path is None else os.path.realpath(self.path),
                    numeric=list(self.numeric), factors=list(self.factors))

    @classmethod
    def from_describe(cls, d: dict) -> "CovariateSpec":
        return cls(path=d.get("path"), numeric=list(d.get("numeric", [])),
                   factors=list(d.get("factors", [])))
