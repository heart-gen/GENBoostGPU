"""Nested out-of-fold elastic net: port of ``crossfit_elastic_net``.

Mirrors ``02_local_genetic_variance/_h/00_functions.R`` (``impute_from_training``,
``adjust_phenotype_in_fold``, ``screen_training_snps``,
``fit_inner_elastic_net``, ``crossfit_elastic_net``) step for step. Every
data-derived quantity in an outer fold is fit on training donors and applied
to held-out donors.

The one structural difference is batching. All ``cv.glmnet`` calls of a locus
(repeats x outer folds x alphas x [full fit + inner folds]) are independent
once the folds and screens are fixed, so they are built first and solved in
one :func:`~genboostgpu.lgv.glmnet.fit_paths` call. On the GPU that is a
single kernel launch per locus.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .folds import FoldPlan, r_fold_plan
from .glmnet import GlmnetError, cv_problems, cv_summarize, fit_paths
from .rstats import (
    r_col_var,
    r_lm_fit,
    r_sd,
    r_var,
    safe_ratio,
    squared_prediction_correlation,
)

__all__ = [
    "CrossfitError",
    "CrossfitSettings",
    "CrossfitPrep",
    "prepare_crossfit",
    "finalize_crossfit",
    "crossfit_elastic_net",
    "impute_from_training",
    "adjust_phenotype_in_fold",
    "screen_training_snps",
]


class CrossfitError(RuntimeError):
    """A condition under which the R cross-fit would stop()."""


@dataclass(frozen=True)
class CrossfitSettings:
    """Locked Module 02 nested-EN settings (``joint-pve-20260820.tsv``)."""

    outer_folds: int = 5
    outer_repeats: int = 2
    inner_folds: int = 5
    alpha_grid: tuple = (0.1, 0.5, 1.0)
    lambda_rule: str = "lambda.1se"
    max_features: int = 1500


def impute_from_training(x_train, x_test):
    """Fill NA in both splits with the training column means (NaN mean -> 0)."""
    with np.errstate(invalid="ignore"):
        means = np.nanmean(x_train, axis=0)
    means[~np.isfinite(means)] = 0.0
    x_train = np.where(np.isnan(x_train), means[None, :], x_train)
    x_test = np.where(np.isnan(x_test), means[None, :], x_test)
    return x_train, x_test


def adjust_phenotype_in_fold(y_train, y_test, covar_train=None, covar_test=None):
    """Residualize on covariates fit in training donors, then scale by the
    training residual mean and sd."""
    if covar_train is None or covar_train.shape[1] == 0:
        train_adj = y_train - y_train.mean()
        test_adj = y_test - y_train.mean()
    else:
        xtr = np.column_stack([np.ones(y_train.size), covar_train])
        xte = np.column_stack([np.ones(y_test.size), covar_test])
        coef, _, _ = r_lm_fit(xtr, y_train)
        coef = np.where(np.isfinite(coef), coef, 0.0)
        train_adj = y_train - xtr @ coef
        test_adj = y_test - xte @ coef
    center = train_adj.mean()
    scale_value = r_sd(train_adj)
    if not np.isfinite(scale_value) or scale_value <= 1e-10:
        raise CrossfitError("Training phenotype has no residual variance")
    return ((train_adj - center) / scale_value, (test_adj - center) / scale_value,
            center, scale_value)


_screen_kernel = None


def _get_screen_kernel():
    global _screen_kernel
    if _screen_kernel is None:
        import numba

        # error_model="numpy": 0/0 gives NaN (mapped to -Inf) instead of raising
        @numba.njit(cache=False, nogil=True, error_model="numpy")
        def kernel(x, y, out):
            n, p = x.shape
            ym = 0.0
            for i in range(n):
                ym += y[i]
            ym /= n
            syy = 0.0
            for i in range(n):
                d = y[i] - ym
                syy += d * d
            for j in range(p):
                xm = 0.0
                for i in range(n):
                    xm += x[i, j]
                xm /= n
                sxx = 0.0
                sxy = 0.0
                for i in range(n):
                    d = x[i, j] - xm
                    sxx += d * d
                    sxy += d * (y[i] - ym)
                out[j] = abs(sxy / math.sqrt(sxx * syy))

        _screen_kernel = kernel
    return _screen_kernel


def screen_scores(x_train, y_train) -> np.ndarray:
    """|marginal correlation| of each column with ``y`` (non-finite -> -Inf).

    Sums run in a fixed sequential order, so the scores (and with them the
    order of tied SNPs) do not depend on the BLAS library, its thread count,
    the CPU or the device. SNPs with identical training genotypes therefore
    get bitwise-identical scores and keep their column order, which is what
    R's ``order()`` does in exact arithmetic. (R itself computes the
    cross-product with its BLAS, so its tie order varies at the last bit.)
    """
    x = np.asarray(x_train, dtype=np.float64)
    y = np.ascontiguousarray(y_train, dtype=np.float64)
    out = np.empty(x.shape[1])
    with np.errstate(invalid="ignore", divide="ignore"):
        _get_screen_kernel()(x, y, out)
    out[~np.isfinite(out)] = -np.inf
    return out


def screen_training_snps(x_train, y_train, max_features: int):
    """Top ``max_features`` columns by |marginal correlation|, ties by column
    index (R's ``order(score, decreasing = TRUE)``); all columns if p <= cap."""
    p = x_train.shape[1]
    if p <= max_features:
        return np.arange(p)
    score = screen_scores(x_train, y_train)
    order = np.argsort(-score, kind="stable")
    return order[:max_features]


@dataclass
class _OuterSplit:
    repeat_id: int
    fold: int
    train_index: np.ndarray
    test_index: np.ndarray
    x_train: np.ndarray
    x_test: np.ndarray
    y_train: np.ndarray
    y_test_adj: np.ndarray
    n_polymorphic: int
    inner_foldid: np.ndarray
    problem_slices: list = field(default_factory=list)  # per alpha: (start, stop)


def _prepare_split(genotype, phenotype, covariates, plan, r, f, settings):
    fold_id = plan.outer[r - 1]
    test_index = np.flatnonzero(fold_id == f)
    train_index = np.flatnonzero(fold_id != f)
    x_train, x_test = impute_from_training(genotype[train_index], genotype[test_index])
    v = r_col_var(x_train)
    keep = np.flatnonzero(np.isfinite(v) & (v > 1e-8))
    if keep.size == 0:
        raise CrossfitError("No polymorphic SNPs in outer training fold")
    x_train = x_train[:, keep]
    x_test = x_test[:, keep]
    ctr = None if covariates is None else covariates[train_index]
    cte = None if covariates is None else covariates[test_index]
    y_tr, y_te, _, _ = adjust_phenotype_in_fold(phenotype[train_index],
                                                phenotype[test_index], ctr, cte)
    selected = screen_training_snps(x_train, y_tr, settings.max_features)
    inner = plan.inner[(r, f)]
    if inner is None or inner.size != train_index.size:
        raise CrossfitError("Inner fold assignment does not match the outer training set")
    return _OuterSplit(r, f, train_index, test_index, x_train[:, selected],
                       x_test[:, selected], y_tr, y_te, int(keep.size), inner)


@dataclass
class CrossfitPrep:
    """A locus with every glmnet problem built but not yet solved."""

    n: int
    p: int
    outer_k: int
    settings: CrossfitSettings
    splits: list
    problems: list
    fold_source: str


def prepare_crossfit(genotype, phenotype, covariates=None,
                     settings: CrossfitSettings = CrossfitSettings(),
                     seed: int = 20250805,
                     fold_plan: FoldPlan | None = None) -> CrossfitPrep:
    """Preprocess every outer split and build all ``glmnet()`` problems."""
    genotype = np.asarray(genotype, dtype=np.float64)
    phenotype = np.asarray(phenotype, dtype=np.float64).ravel()
    if genotype.shape[0] != phenotype.size:
        raise CrossfitError("Genotype and phenotype sample counts differ")
    if not np.all(np.isfinite(phenotype)):
        raise CrossfitError("Phenotype contains missing or non-finite values")
    if covariates is not None:
        covariates = np.asarray(covariates, dtype=np.float64)
        if covariates.ndim == 1:
            covariates = covariates[:, None]
        if covariates.shape[0] != phenotype.size:
            raise CrossfitError("Covariate and phenotype sample counts differ")
        if not np.all(np.isfinite(covariates)):
            raise CrossfitError("Covariates contain missing or non-finite values")
    n, p = genotype.shape
    if p < 1:
        raise CrossfitError("No SNPs supplied")
    outer_k = max(2, min(int(settings.outer_folds), n))
    if fold_plan is None:
        fold_plan = r_fold_plan(n, seed, outer_k, settings.outer_repeats,
                                settings.inner_folds)
    splits = []
    problems = []
    for r in range(1, settings.outer_repeats + 1):
        for f in range(1, outer_k + 1):
            sp = _prepare_split(genotype, phenotype, covariates, fold_plan, r, f, settings)
            fold_x = {}
            for alpha in settings.alpha_grid:
                start = len(problems)
                try:
                    problems.extend(cv_problems(sp.x_train, sp.y_train,
                                                sp.inner_foldid, float(alpha),
                                                fold_cache=fold_x))
                    sp.problem_slices.append((start, len(problems), float(alpha)))
                except GlmnetError:
                    sp.problem_slices.append((start, start, float(alpha)))
            splits.append(sp)
    return CrossfitPrep(n, p, outer_k, settings, splits, problems, fold_plan.source)


def finalize_crossfit(prep: CrossfitPrep, fits, keep_predictions: bool = False) -> dict:
    """Inner tuning, held-out prediction and pooled metrics from solved fits."""
    n, p, outer_k, settings = prep.n, prep.p, prep.outer_k, prep.settings
    if len(fits) != len(prep.problems):
        raise CrossfitError("Number of fits does not match the prepared problems")
    prediction_sum = np.zeros(n)
    adjusted_sum = np.zeros(n)
    prediction_count = np.zeros(n, dtype=np.int64)
    records = []
    for sp in prep.splits:
        best = None
        best_score = np.inf
        for (start, stop, alpha) in sp.problem_slices:
            if stop == start:
                continue
            batch = fits[start:stop]
            if any(isinstance(b, Exception) for b in batch):
                continue  # R: tryCatch(cv.glmnet(...), error = NULL)
            try:
                cv = cv_summarize(sp.x_train, sp.y_train, sp.inner_foldid,
                                  batch[0], batch[1:])
            except GlmnetError:
                continue
            lam_value = cv.lambda_1se if settings.lambda_rule == "lambda.1se" \
                else cv.lambda_min
            idx = int(np.argmin(np.abs(cv.lambda_ - lam_value)))
            score = cv.cvm[idx]
            if score < best_score:  # which.min: first minimum wins
                best_score = score
                best = (cv, alpha, lam_value)
        if best is None or not np.isfinite(best_score):
            raise CrossfitError("All inner elastic-net fits failed")
        cv, alpha, lam_value = best
        pred = cv.glmnet_fit.predict_at(sp.x_test, [lam_value])[:, 0]
        _, beta = cv.glmnet_fit.coef_at([lam_value])
        prediction_sum[sp.test_index] += pred
        adjusted_sum[sp.test_index] += sp.y_test_adj
        prediction_count[sp.test_index] += 1
        records.append(dict(
            repeat_id=sp.repeat_id, fold=sp.fold, n_train=sp.train_index.size,
            n_test=sp.test_index.size, snps_polymorphic=sp.n_polymorphic,
            snps_screened=sp.x_train.shape[1],
            snps_nonzero=int(np.count_nonzero(beta[:, 0])), alpha=alpha,
            lambda_=lam_value, inner_mse=best_score,
            fold_score_variance_ratio=safe_ratio(r_var(pred), r_var(sp.y_test_adj)),
        ))
    if np.any(prediction_count != settings.outer_repeats):
        raise CrossfitError(
            "Every sample must receive exactly one held-out prediction per repeat")
    prediction = prediction_sum / prediction_count
    adjusted = adjusted_sum / prediction_count
    phen_var = r_var(adjusted)
    sst = float(np.sum((adjusted - adjusted.mean()) ** 2))
    ratio = safe_ratio(float(np.sum((adjusted - prediction) ** 2)), sst)
    r2_oof = 1.0 - ratio if np.isfinite(ratio) else np.nan
    pred_var = r_var(prediction)
    cov_ap = float(np.cov(adjusted, prediction, ddof=1)[0, 1])
    folds = pd.DataFrame.from_records(records).rename(columns={"lambda_": "lambda"})
    fsvr = folds["fold_score_variance_ratio"].to_numpy(dtype=float)
    metrics = dict(
        n=n, num_snps=p, outer_folds=outer_k, outer_repeats=settings.outer_repeats,
        inner_folds=settings.inner_folds, max_features=settings.max_features,
        r2_oof=r2_oof,
        rho2_oof=squared_prediction_correlation(adjusted, prediction),
        covariance_ratio_oof=safe_ratio(cov_ap, phen_var),
        score_variance_ratio_oof=safe_ratio(pred_var, phen_var),
        calibration_slope_oof=safe_ratio(cov_ap, pred_var),
        mean_fold_score_variance_ratio=(float(np.nanmean(fsvr))
                                        if np.any(np.isfinite(fsvr)) else np.nan),
        mean_nonzero_snps=float(folds["snps_nonzero"].mean()),
        converged=True,
    )
    out = {"metrics": metrics, "folds": folds, "fold_source": prep.fold_source}
    if keep_predictions:
        out["predictions"] = pd.DataFrame(dict(
            sample_index=np.arange(1, n + 1), adjusted_phenotype=adjusted,
            oof_prediction=prediction, prediction_repeats=prediction_count))
    return out


def crossfit_elastic_net(genotype, phenotype, covariates=None,
                         settings: CrossfitSettings = CrossfitSettings(),
                         seed: int = 20250805, fold_plan: FoldPlan | None = None,
                         device: str = "cpu", keep_predictions: bool = False) -> dict:
    """Nested out-of-fold elastic net metrics for one locus.

    ``genotype`` is n x p with NaN for missing calls; ``covariates`` n x q
    (no intercept). ``seed`` is the cross-fit seed (Stage 01 passes
    ``feature_seed + 17``). Returns ``{"metrics": dict, "folds": DataFrame}``
    (plus ``"predictions"`` when requested). For many loci at once use
    :func:`prepare_crossfit` / :func:`finalize_crossfit` around one
    :func:`~genboostgpu.lgv.glmnet.fit_paths` call.
    """
    prep = prepare_crossfit(genotype, phenotype, covariates, settings, seed, fold_plan)
    fits = fit_paths(prep.problems, device=device)
    return finalize_crossfit(prep, fits, keep_predictions=keep_predictions)
