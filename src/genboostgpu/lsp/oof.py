"""Module 03: end-to-end out-of-fold local SNP prediction.

Port of ``03_local_snp_prediction/_h/{01_prepare_folds.R, 02_fit_oof.R,
he_permutation_screen.R, 03_combine_oof.R}`` (PI-locked
``config/prediction.yml``): 5 outer folds x 5 repeats on a run-level donor
fold table, fold-internal genotype QC and scaling, a Haseman-Elston screen
calibrated with 1000 within-fold permutations, inner 5-fold ``cv.glmnet``
(``standardize = FALSE``) over alpha {0.1, 0.5, 0.9, 1} at ``lambda.min``,
and the training-mean null prediction (0 on the training-scaled scale) for
held-out donors whenever a fold's screen fails. Failed folds are never
dropped.

Every random draw (fold table, permutations, inner folds) uses R's own
generator and ``seed_for`` keys, so an accepted R run can be replayed.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..lgv.crossfit import CrossfitError, adjust_phenotype_in_fold
from ..lgv.glmnet import GlmnetError, cv_problems, cv_summarize, fit_paths
from ..lgv.he import he_prepare
from ..lgv.rrng import RRNG, r_permutations, r_sample
from ..lgv.rstats import r_cor, r_lm_fit
from .seeds import seed_for

__all__ = ["LspSettings", "donor_folds", "prep_genotypes", "he_permutation_screen",
           "prepare_locus_oof", "finalize_locus_oof", "locus_metrics",
           "PRED_COLUMNS"]

PRED_COLUMNS = ["repeat_i", "outer_fold", "donor", "y_obs", "y_pred", "screened_in",
                "n_variants", "screen_p", "screen_stat", "alpha", "lambda"]


@dataclass(frozen=True)
class LspSettings:
    """``config/prediction.yml`` (PI-locked 2026-08-23) + cis QC constants."""

    outer_folds: int = 5
    inner_folds: int = 5
    repeats: int = 5
    alpha_grid: tuple = (0.1, 0.5, 0.9, 1.0)
    lambda_rule: str = "lambda.min"
    screen_method: str = "he_permutation"
    n_permutations: int = 1000
    screen_alpha: float = 0.05
    maf_min: float = 0.05
    missing_max: float = 0.05
    min_usable_donors: int = 20
    calibration_sd_tol: float = 1e-6


def donor_folds(donors, run_id: str, region: str, settings: LspSettings = LspSettings()):
    """``01_prepare_folds.R``: one outer-fold assignment per repeat."""
    donors = list(donors)
    if len(set(donors)) != len(donors):
        raise ValueError("Duplicate donors in the donor list")
    rows = []
    for r in range(1, settings.repeats + 1):
        rng = RRNG(seed_for(run_id, region=region, repeat_i=r))
        assignment = np.resize(np.arange(1, settings.outer_folds + 1), len(donors))
        fold = r_sample(rng, assignment, len(donors))
        rows.append(pd.DataFrame(dict(repeat_i=r, donor=donors, outer_fold=fold)))
    return pd.concat(rows, ignore_index=True)


def prep_genotypes(g_train, g_test, maf_min=0.05, missing_max=0.05):
    """Fit-on-train / apply-to-test QC, imputation and scaling (or None)."""
    miss = np.isnan(g_train).mean(axis=0)
    keep = miss <= missing_max
    if not keep.any():
        return None
    g_train, g_test = g_train[:, keep], g_test[:, keep]
    with np.errstate(invalid="ignore"):
        train_mean = np.nanmean(g_train, axis=0)
    maf = np.minimum(train_mean, 2 - train_mean) / 2
    keep = ~np.isnan(maf) & (maf >= maf_min)
    if not keep.any():
        return None
    g_train, g_test, train_mean = g_train[:, keep], g_test[:, keep], train_mean[keep]
    g_train = np.where(np.isnan(g_train), train_mean[None, :], g_train)
    g_test = np.where(np.isnan(g_test), train_mean[None, :], g_test)
    train_sd = np.std(g_train, axis=0, ddof=1)
    keep = train_sd > 0
    if not keep.any():
        return None
    g_train, g_test = g_train[:, keep], g_test[:, keep]
    train_mean, train_sd = train_mean[keep], train_sd[keep]
    return dict(train=(g_train - train_mean) / train_sd,
                test=(g_test - train_mean) / train_sd, n_variants=int(g_train.shape[1]))


def _scale(v):
    v = np.asarray(v, dtype=np.float64)
    c = v - v.mean()
    return c / np.sqrt(np.sum(c * c) / (v.size - 1))


def he_permutation_screen(x_train, y_train, n_perm: int, alpha: float, seed: int,
                          xp=np) -> dict:
    """``he_permutation_screen(he_screen_prepare(x_train), y_train, ...)``."""
    prep = he_prepare(x_train, xp=xp)
    if prep is None:
        return dict(pass_=False, p=np.nan, statistic=np.nan, n_perm=0,
                    reason="insufficient_genotype_variation")
    cmat = prep.cmat
    r = xp.asarray(_scale(y_train))
    observed = 0.5 * float(xp.dot(r, cmat @ r)) / prep.denom
    if not np.isfinite(observed):
        return dict(pass_=False, p=np.nan, statistic=np.nan, n_perm=0,
                    reason="nonfinite_he_statistic")
    n = r.size
    r0 = _scale(y_train)
    perm_idx = r_permutations(RRNG(seed), n, n_perm)          # n_perm x n
    perm = r0[perm_idx.T - 1]                                 # n x n_perm
    pm = xp.asarray(perm)
    null = 0.5 * xp.sum(pm * (cmat @ pm), axis=0) / prep.denom
    null = np.asarray(null.get() if hasattr(null, "get") else null)
    finite = null[np.isfinite(null)]
    if finite.size == 0:
        return dict(pass_=False, p=np.nan, statistic=observed, n_perm=0,
                    reason="no_finite_permutation_statistic")
    p = (1 + np.sum(finite >= observed)) / (1 + finite.size)
    return dict(pass_=bool(p <= alpha), p=float(p), statistic=observed,
                n_perm=int(finite.size), reason=None)


@dataclass
class _Fold:
    repeat_i: int
    outer_fold: int
    test_ids: list
    y_te: np.ndarray
    null_rows: pd.DataFrame | None = None     # final answer when no model is fit
    gp: dict | None = None
    screen: dict | None = None
    y_tr: np.ndarray | None = None
    inner_id: np.ndarray | None = None
    slices: list = field(default_factory=list)  # (start, stop, alpha)


@dataclass
class LocusOOFPrep:
    vmr_id: str
    folds: list
    problems: list


def _null_rows(fold: _Fold, screen=None, n_variants=0):
    return pd.DataFrame(dict(
        repeat_i=fold.repeat_i, outer_fold=fold.outer_fold, donor=fold.test_ids,
        y_obs=fold.y_te, y_pred=0.0, screened_in=False, n_variants=int(n_variants),
        screen_p=np.nan if screen is None else screen["p"],
        screen_stat=np.nan if screen is None else screen["statistic"],
        alpha=np.nan, **{"lambda": np.nan}))


def prepare_locus_oof(locus, folds: pd.DataFrame, run_id: str, region: str, vmr_id: str,
                      settings: LspSettings = LspSettings(), xp=np) -> LocusOOFPrep:
    """Build every fold of one locus (screens run here; glmnet problems are
    collected for one batched solve). ``locus`` comes from a loader called
    with ``apply_snp_qc=False``."""
    meta = locus.metadata
    if "FID" in meta.columns:
        donor_key = list(meta["FID"] + "::" + meta["IID"])
    else:
        donor_key = list(meta["sample_id"].astype(str))
    fold_donors = set(folds["donor"])
    shared = [d for d in donor_key if d in fold_donors]
    if len(shared) < settings.min_usable_donors:
        raise CrossfitError(f"only {len(shared)} usable donors")
    pos = {d: i for i, d in enumerate(donor_key)}
    y_all = np.asarray(locus.y, dtype=np.float64)
    g_all = np.asarray(locus.genotype, dtype=np.float64)
    c_all = np.asarray(locus.covariates, dtype=np.float64)
    shared_set = set(shared)
    out_folds, problems = [], []
    for r in sorted(folds["repeat_i"].unique()):
        fr = folds[folds["repeat_i"] == r]
        for k in sorted(fr["outer_fold"].unique()):
            test_ids = [d for d in fr.loc[fr["outer_fold"] == k, "donor"] if d in shared_set]
            test_set = set(test_ids)
            train_ids = [d for d in shared if d not in test_set]
            if not test_ids or len(train_ids) < 10:
                continue
            tr = np.array([pos[d] for d in train_ids])
            te = np.array([pos[d] for d in test_ids])
            try:
                y_tr, y_te, _, _ = adjust_phenotype_in_fold(y_all[tr], y_all[te],
                                                            c_all[tr], c_all[te])
            except CrossfitError:
                continue  # R: tryCatch(...) -> NULL -> fold contributes no rows
            fold = _Fold(int(r), int(k), test_ids, y_te)
            gp = prep_genotypes(g_all[tr], g_all[te], settings.maf_min, settings.missing_max)
            if gp is None:
                fold.null_rows = _null_rows(fold)
                out_folds.append(fold)
                continue
            fseed = seed_for(run_id, region=region, task=vmr_id, repeat_i=int(r), fold=int(k))
            if settings.screen_method in ("none", ""):
                sc = dict(pass_=True, p=np.nan, statistic=np.nan)
            elif settings.screen_method == "he_permutation":
                sc = he_permutation_screen(gp["train"], y_tr, settings.n_permutations,
                                           settings.screen_alpha, fseed, xp=xp)
            else:
                raise ValueError(f"Unsupported screen.method {settings.screen_method!r}")
            if not sc["pass_"]:
                fold.null_rows = _null_rows(fold, sc, gp["n_variants"])
                out_folds.append(fold)
                continue
            inner_id = r_sample(RRNG(fseed), np.resize(
                np.arange(1, settings.inner_folds + 1), len(train_ids)), len(train_ids))
            fold.gp, fold.screen, fold.y_tr, fold.inner_id = gp, sc, y_tr, inner_id
            fold_x = {}
            for a in settings.alpha_grid:
                start = len(problems)
                try:
                    problems.extend(cv_problems(gp["train"], y_tr, inner_id, float(a),
                                                fold_cache=fold_x, standardize=False))
                    fold.slices.append((start, len(problems), float(a)))
                except GlmnetError:
                    fold.slices.append((start, start, float(a)))
            out_folds.append(fold)
    return LocusOOFPrep(vmr_id, out_folds, problems)


def finalize_locus_oof(prep: LocusOOFPrep, fits, settings: LspSettings = LspSettings()):
    """Per-donor held-out predictions (``PRED_COLUMNS`` + ``vmr_id``)."""
    parts = []
    for fold in prep.folds:
        if fold.null_rows is not None:
            parts.append(fold.null_rows)
            continue
        best = None
        any_fit = False
        for start, stop, alpha in fold.slices:
            if stop == start or any(isinstance(f, Exception) for f in fits[start:stop]):
                continue
            try:
                cv = cv_summarize(fold.gp["train"], fold.y_tr, fold.inner_id,
                                  fits[start], fits[start + 1:stop])
            except GlmnetError:
                continue
            any_fit = True
            lam = cv.lambda_min if settings.lambda_rule == "lambda.min" else cv.lambda_1se
            hit = np.flatnonzero(cv.lambda_ == lam)
            score = cv.cvm[hit[0]] if hit.size else np.nan
            # which.min: NA never wins, the first minimum does.
            if np.isfinite(score) and (best is None or score < best[0]):
                best = (score, cv, alpha, lam)
        if not any_fit:
            parts.append(_null_rows(fold, fold.screen, fold.gp["n_variants"]))
            continue
        if best is None:
            raise CrossfitError("every inner cv.glmnet returned NA cvm")
        _, cv, alpha, lam = best
        yhat = cv.glmnet_fit.predict_at(fold.gp["test"], [lam])[:, 0]
        parts.append(pd.DataFrame(dict(
            repeat_i=fold.repeat_i, outer_fold=fold.outer_fold, donor=fold.test_ids,
            y_obs=fold.y_te, y_pred=yhat, screened_in=True,
            n_variants=fold.gp["n_variants"], screen_p=fold.screen["p"],
            screen_stat=fold.screen["statistic"], alpha=alpha, **{"lambda": lam})))
    if not parts:
        raise CrossfitError("no fold produced predictions")
    preds = pd.concat(parts, ignore_index=True)
    preds["vmr_id"] = prep.vmr_id
    return preds


def locus_metrics(preds: pd.DataFrame, settings: LspSettings = LspSettings()) -> dict:
    """Per-locus metrics exactly as ``03_combine_oof.R``."""
    y = preds["y_obs"].to_numpy(dtype=float)
    yp = preds["y_pred"].to_numpy(dtype=float)
    sse = float(np.sum((y - yp) ** 2))
    sst = float(np.sum((y - y.mean()) ** 2))
    sd_pred = np.std(yp, ddof=1) if yp.size > 1 else np.nan
    sd_obs = np.std(y, ddof=1) if y.size > 1 else np.nan
    cal_ok = bool(np.isfinite(sd_pred) and np.isfinite(sd_obs) and sd_obs > 0
                  and sd_pred > settings.calibration_sd_tol * sd_obs)
    if cal_ok:
        coef, _, _ = r_lm_fit(np.column_stack([np.ones(yp.size), yp]), y)
        cal = (float(coef[0]), float(coef[1]))
        cor2 = r_cor(y, yp) ** 2
    else:
        cal, cor2 = (np.nan, np.nan), np.nan
    alpha = preds["alpha"].to_numpy(dtype=float)
    return dict(
        n_donors_predicted=int(preds["donor"].nunique()), n_predictions=int(len(preds)),
        r2_pred_oof=1 - sse / sst if sst > 0 else np.nan, cor2_oof=cor2,
        rmse=float(np.sqrt(np.mean((y - yp) ** 2))), mae=float(np.mean(np.abs(y - yp))),
        calibration_intercept=cal[0], calibration_slope=cal[1],
        calibration_slope_defined=cal_ok,
        screening_pass_frequency=float(preds["screened_in"].astype(float).mean()),
        median_n_variants=float(np.median(preds["n_variants"].to_numpy(dtype=float))),
        median_alpha=float(np.nanmedian(alpha)) if np.isfinite(alpha).any() else np.nan)
