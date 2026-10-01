"""Genotype-geometry features of a cis window: ``p_eff`` and ``ld_metric``.

Ports of ``effective_rank_genotype`` (``joint_pve_functions.R``) and
``adjacent_ld_metric`` (``00_functions.R``). Both depend only on genotype,
so the site-level engine computes them once per window block.
"""
from __future__ import annotations

import numpy as np

from .rstats import impute_col_means, r_col_var, r_scale, r_seq_length_out

__all__ = ["effective_rank_genotype", "adjacent_ld_metric", "polymorphic_standardized"]


def polymorphic_standardized(genotype, xp=np):
    """Mean-impute, drop columns with var <= 1e-8, and ``scale()`` the rest.

    Returns ``None`` when no column is polymorphic.
    """
    g = impute_col_means(genotype)
    v = r_col_var(g)
    keep = np.isfinite(v) & (v > 1e-8)
    if not keep.any():
        return None
    return xp.asarray(r_scale(g[:, keep]))


def effective_rank_genotype(genotype, xp=np) -> float:
    """``trace(K)^2 / trace(K^2)`` with ``K = Z Z' / p`` (Z = scaled genotype)."""
    z = polymorphic_standardized(genotype, xp=xp)
    if z is None:
        return np.nan
    k = (z @ z.T) / z.shape[1]
    trace_k = float(xp.trace(k))
    trace_k2 = float(xp.sum(k * k))
    if not np.isfinite(trace_k) or not np.isfinite(trace_k2) or trace_k2 <= 0:
        return np.nan
    return trace_k * trace_k / trace_k2


def _pairwise_r2(a, b) -> float:
    ok = ~(np.isnan(a) | np.isnan(b))
    a = a[ok]
    b = b[ok]
    if a.size < 2:
        return np.nan
    ac = a - a.mean()
    bc = b - b.mean()
    den = np.sqrt(np.sum(ac * ac) * np.sum(bc * bc))
    if den == 0 or not np.isfinite(den):
        return np.nan
    r = np.sum(ac * bc) / den
    return float(r * r) if np.isfinite(r) else np.nan


def adjacent_ld_metric(genotype, max_pairs: int = 200) -> float:
    """Median adjacent-SNP r^2 over up to ``max_pairs`` evenly spaced pairs.

    Uses the QC'd genotype *with* missing calls, like R's
    ``cor(..., use = "pairwise.complete.obs")``.
    """
    g = np.asarray(genotype, dtype=np.float64)
    p = g.shape[1]
    if p < 2:
        return 0.0
    idx = np.unique(np.round(r_seq_length_out(1, p - 1, min(max_pairs, p - 1))))
    vals = np.array([_pairwise_r2(g[:, int(j) - 1], g[:, int(j)]) for j in idx])
    vals = vals[~np.isnan(vals)]
    if vals.size == 0:
        return np.nan
    return float(np.median(vals))
