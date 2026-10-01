"""Haseman-Elston regression and its permutation-calibrated screen.

``haseman_elston`` ports ``00_functions.R::haseman_elston`` (Module 02's HE
feature). It regresses the pairwise products of the scaled covariate
residual, ``v_ij = r_i r_j``, on GRM relatedness ``u_ij`` over the upper
triangle with an intercept. Because the GRM is shared, the statistic is a
quadratic form in ``r``:

    slope = 0.5 * r' C r / sum_upper (u - ubar)^2,   C = GRM - ubar, diag(C) = 0

so many phenotypes on the same window (the site-level engine) or many
permutations (Module 03's screen) are one matrix product. The OLS standard
error uses the closed form for simple regression; both forms are tested
against R.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from .geometry import polymorphic_standardized
from .rstats import r_lm_fit

__all__ = ["HEPrep", "he_prepare", "he_from_residuals", "haseman_elston",
           "residualize_and_scale"]


@dataclass
class HEPrep:
    """Genotype-only half of HE for one window."""

    cmat: object      # n x n, GRM - ubar with zero diagonal
    denom: float      # sum over upper triangle of (u - ubar)^2
    n_snps: int
    n: int


def he_prepare(genotype, xp=np):
    """Precompute the HE design for a genotype window (``None`` if < 2 SNPs)."""
    z = polymorphic_standardized(genotype, xp=xp)
    if z is None or z.shape[1] < 2:
        return None
    n = z.shape[0]
    grm = (z @ z.T) / z.shape[1]
    iu = xp.triu_indices(n, k=1)
    u = grm[iu]
    ubar = float(u.mean())
    denom = float(xp.sum((u - ubar) ** 2))
    if not np.isfinite(denom) or denom <= 0:
        return None
    cmat = grm - ubar
    idx = xp.arange(n)
    cmat[idx, idx] = 0.0
    return HEPrep(cmat=cmat, denom=denom, n_snps=int(z.shape[1]), n=n)


def residualize_and_scale(phenotype, covariates=None):
    """``scale(residuals of lm.fit(cbind(1, covariates), y))`` as HE uses it."""
    y = np.asarray(phenotype, dtype=np.float64)
    if covariates is None or np.asarray(covariates).size == 0:
        resid = y - y.mean()
    else:
        design = np.column_stack([np.ones(y.size), np.asarray(covariates, np.float64)])
        coef, _, _ = r_lm_fit(design, y)
        coef = np.where(np.isfinite(coef), coef, 0.0)
        resid = y - design @ coef
    resid = resid - resid.mean()
    sd = np.sqrt(np.sum(resid * resid) / (resid.size - 1))
    return resid / sd


def he_from_residuals(prep: HEPrep, r, xp=np):
    """HE slope, SE, df and p-value for scaled residual column(s) ``r``.

    ``r`` is ``n`` or ``n x k``; returns arrays of length ``k``.
    """
    r = xp.asarray(r, dtype=xp.float64)
    single = r.ndim == 1
    if single:
        r = r[:, None]
    n = prep.n
    m = n * (n - 1) // 2
    slope = 0.5 * xp.sum(r * (prep.cmat @ r), axis=0) / prep.denom
    # Sums over the upper triangle of v = r_i r_j and of v^2.
    s1 = xp.sum(r, axis=0)
    s2 = xp.sum(r * r, axis=0)
    s4 = xp.sum(r ** 4, axis=0)
    sum_v = 0.5 * (s1 * s1 - s2)
    sum_v2 = 0.5 * (s2 * s2 - s4)
    syy = sum_v2 - sum_v * sum_v / m
    ssr = syy - slope * slope * prep.denom
    df = m - 2
    sigma2 = ssr / df
    se = xp.sqrt(sigma2 / prep.denom)
    slope_h = np.asarray(slope.get() if hasattr(slope, "get") else slope)
    se_h = np.asarray(se.get() if hasattr(se, "get") else se)
    with np.errstate(invalid="ignore", divide="ignore"):
        tstat = slope_h / se_h
    pval = np.where(np.isfinite(tstat), 2 * stats.t.sf(np.abs(tstat), df), np.nan)
    if single:
        return float(slope_h[0]), float(se_h[0]), int(df), float(pval[0])
    return slope_h, se_h, df, pval


def haseman_elston(genotype, phenotype, covariates=None, xp=np) -> dict:
    """Module 02 ``haseman_elston()``: he_h2, he_se, he_pvalue, he_num_snps,
    he_converged."""
    prep = he_prepare(genotype, xp=xp)
    if prep is None:
        z = polymorphic_standardized(genotype)
        return dict(he_h2=np.nan, he_se=np.nan, he_pvalue=np.nan,
                    he_num_snps=0 if z is None else int(z.shape[1]),
                    he_converged=False)
    r = residualize_and_scale(phenotype, covariates)
    slope, se, df, pval = he_from_residuals(prep, r, xp=xp)
    if not np.isfinite(slope) or df <= 0:
        return dict(he_h2=np.nan, he_se=np.nan, he_pvalue=np.nan,
                    he_num_snps=prep.n_snps, he_converged=False)
    return dict(he_h2=slope, he_se=se, he_pvalue=pval, he_num_snps=prep.n_snps,
                he_converged=bool(np.isfinite(slope) and np.isfinite(se)))
