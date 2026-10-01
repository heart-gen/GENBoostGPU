"""Small R-semantics helpers shared by the Module 02/03 ports.

Each function mirrors the R expression it is named after, including NA
behaviour, so ported code reads like the R it replaces.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "r_var",
    "r_sd",
    "r_scale",
    "r_col_var",
    "r_lm_fit",
    "r_cor",
    "safe_ratio",
    "squared_prediction_correlation",
    "impute_col_means",
    "r_seq_length_out",
]


def r_var(x) -> float:
    """``stats::var(x)`` (n - 1 denominator); NA for fewer than two values."""
    x = np.asarray(x, dtype=np.float64)
    if x.size < 2:
        return np.nan
    return float(np.var(x, ddof=1))


def r_sd(x) -> float:
    v = r_var(x)
    return float(np.sqrt(v)) if np.isfinite(v) else np.nan


def r_col_var(x):
    """``apply(x, 2, stats::var)`` for a matrix without missing values."""
    x = np.asarray(x, dtype=np.float64)
    if x.shape[0] < 2:
        return np.full(x.shape[1], np.nan)
    return np.var(x, axis=0, ddof=1)


def r_scale(x):
    """``scale(x)``: center by column means, divide by column sd (n - 1)."""
    x = np.asarray(x, dtype=np.float64)
    center = x.mean(axis=0)
    xc = x - center
    sd = np.sqrt(np.sum(xc * xc, axis=0) / (x.shape[0] - 1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return xc / sd


def r_lm_fit(design, y, tol: float = 1e-7):
    """``lm.fit(design, y)`` returning (coefficients, residuals, rank).

    Aliased columns get NA coefficients, as with R's limited-pivoting QR
    (``dqrdc2``): a column is aliased when, after removing its projection on
    the earlier retained columns, its norm falls below ``tol`` times its
    original norm. Fitted values do not depend on how the retained columns
    are solved, so the retained set is solved by least squares.
    """
    X = np.asarray(design, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).ravel()
    n, k = X.shape
    keep = []
    q = np.zeros((n, 0))
    for j in range(k):
        col = X[:, j]
        norm0 = np.linalg.norm(col)
        if norm0 == 0:
            continue
        resid = col - q @ (q.T @ col) if q.shape[1] else col.copy()
        # one re-orthogonalization pass for stability
        if q.shape[1]:
            resid = resid - q @ (q.T @ resid)
        rn = np.linalg.norm(resid)
        if rn < tol * norm0:
            continue
        keep.append(j)
        q = np.column_stack([q, resid / rn])
    coef = np.full(k, np.nan)
    if keep:
        # Householder QR, as R's dqrdc2/dqrsl (closer to R's rounding than an
        # SVD solve, which matters where downstream glmnet decisions are exact).
        qmat, rmat = np.linalg.qr(X[:, keep])
        sol = _back_substitute(rmat, qmat.T @ y)
        coef[keep] = sol
        fitted = X[:, keep] @ sol
    else:
        fitted = np.zeros(n)
    return coef, y - fitted, len(keep)


def _back_substitute(r, b):
    n = r.shape[0]
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (b[i] - r[i, i + 1:] @ x[i + 1:]) / r[i, i]
    return x


def r_cor(x, y) -> float:
    """``stats::cor(x, y)`` for two complete vectors (NA when degenerate)."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 2:
        return np.nan
    xc = x - x.mean()
    yc = y - y.mean()
    den = np.sqrt(np.sum(xc * xc) * np.sum(yc * yc))
    if den == 0 or not np.isfinite(den):
        return np.nan
    return float(np.sum(xc * yc) / den)


def safe_ratio(numerator, denominator) -> float:
    """Module 02's ``safe_ratio``: NA unless the denominator is usable."""
    if not np.isfinite(denominator) or abs(denominator) <= 1e-12:
        return np.nan
    return float(numerator / denominator)


def squared_prediction_correlation(observed, predicted, variance_tolerance=1e-12):
    """Module 02's ``squared_prediction_correlation``."""
    pv = r_var(predicted)
    if np.isfinite(pv) and pv <= variance_tolerance:
        return 0.0
    rho = r_cor(observed, predicted)
    return rho * rho if np.isfinite(rho) else np.nan


def impute_col_means(x):
    """Fill NA with column means; columns with no observed value get 0."""
    x = np.array(x, dtype=np.float64, copy=True)
    if not np.isnan(x).any():
        return x
    with np.errstate(invalid="ignore"):
        means = np.nanmean(x, axis=0)
    means[~np.isfinite(means)] = 0.0
    rows, cols = np.nonzero(np.isnan(x))
    x[rows, cols] = means[cols]
    return x


def r_seq_length_out(start: float, stop: float, length_out: int):
    """``seq(start, stop, length.out = length_out)`` with R's arithmetic."""
    length_out = int(length_out)
    if length_out <= 0:
        return np.zeros(0)
    if length_out == 1:
        return np.array([float(start)])
    if length_out == 2:
        return np.array([float(start), float(stop)])
    if start == stop:
        return np.full(length_out, float(start))
    n1 = length_out - 1
    by = (stop - start) / n1
    mid = start + np.arange(1, n1, dtype=np.float64) * by
    return np.concatenate([[float(start)], mid, [float(stop)]])
