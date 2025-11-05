import optuna
import cupy as cp
import numpy as np
from itertools import product
from dask import delayed, compute
from multiprocessing import cpu_count
from sklearn.model_selection import KFold
from cuml.linear_model import ElasticNet, Ridge

from .snp_processing import _corr_with_y_streaming

__all__ = [
    "boosting_elastic_net",
]

def _standardize_train_only(X, y, tr_idx):
    """
    Standardize X and y using statistics computed on the TRAIN split only.
    This prevents leakage into validation during feature screening/tuning.
    """
    x_mu = cp.mean(X[tr_idx], axis=0)
    x_sd = cp.std(X[tr_idx], axis=0) + 1e-6
    y_mu = cp.mean(y[tr_idx])
    y_sd = cp.std(y[tr_idx]) + 1e-6
    return (X - x_mu) / x_sd, (y - y_mu) / y_sd


def boosting_elastic_net(
        X, y, snp_ids, n_iter=50, batch_size=500, n_trials=20,
        alphas=(0.1, 1.0), l1_ratios=(0.1, 0.9), subsample_frac=0.7,
        ridge_grid=(1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1, 3, 10),
        cv=5, refit_each_iter=False, standardize=True, val_frac=0.2,
        random_state=13, early_stop_metric="auto", # "val_r2" | "h2" | "auto"
        patience=5, min_delta=1e-4, warmup=5, batch_corr_cols=8192,
        adaptive_trials=True, working_set=None, fixed_alpha=None, 
        fixed_l1_ratio=None, fixed_subsample_frac=None
):
    """
    Boosting ElasticNet with final Ridge refit,
    genome-wide betas, and SNP-based variance components.
    """
    # Validation split (on GPU indices), avoid leakage
    n = X.shape[0]
    use_val = 0.0 < val_frac < 0.9 and n >= 25
    if use_val:
        rng = np.random.default_rng(random_state)
        perm = rng.permutation(n).astype(np.int64)
        n_val = max(5, int(n * val_frac))
        val_idx = cp.asarray(perm[:n_val])
        tr_idx  = cp.asarray(perm[n_val:])
    else:
        tr_idx = cp.arange(n)
        val_idx = None

    # Standardization with Train only
    if standardize:
        X, y = _standardize_train_only(X, y, tr_idx)

    # Initialize working-set
    if working_set is None:
        K = int(batch_size)
        refresh = 1
    else:
        K = int(working_set.get("K", batch_size))
        refresh = int(working_set.get("refresh", 10))
    K = max(1, min(K, X.shape[1]))
    refresh = max(1, refresh)
    cached_top_idx = None

    # Pick early-stop metric
    metric_mode = early_stop_metric
    if metric_mode == "auto":
        metric_mode = "val_r2" if use_val else "h2"

    residuals = y.copy()
    cum_pred = cp.zeros_like(y)
    betas_boosting = cp.zeros(X.shape[1], dtype=cp.float32)
    h2_estimates, val_r2_hist = [], []

    # Effective subsample
    subsample_eff = float(fixed_subsample_frac) if fixed_subsample_frac is not None else float(subsample_frac)

    # Global hyperparameters (if not tuning each iter)
    def _tune_params(Xsub, ysub, trials, alpha_rng, l1_rng, fixed_a=None, fixed_l1=None):
        if fixed_a is not None and fixed_l1 is not None:
            return {"alpha": float(fixed_a), "l1_ratio": float(fixed_l1)}
        if fixed_a is not None:
            return _tune_elasticnet_optuna(
                Xsub, ysub, n_trials=max(3, trials), cv=cv, max_iter=5000, 
                subsample_frac=subsample_eff, alpha_range=(fixed_a, fixed_a),
                l1_range=l1_rng
            )
        if fixed_l1 is not None:
            return _tune_elasticnet_optuna(
                Xsub, ysub, n_trials=max(3, trials), cv=cv, max_iter=5000, 
                subsample_frac=subsample_eff, alpha_range=alpha_rng,
                l1_range=(fixed_l1, fixed_l1)
            )
        return _tune_elasticnet_optuna(
            Xsub, ysub, n_trials=trials, cv=cv, max_iter=5000, 
            subsample_frac=subsample_eff, alpha_range=alpha_rng, l1_range=l1_rng
        )
    M = int(X.shape[1])
    trials_base = _auto_trials(M if not adaptive_trials else min(M, batch_size),
                               base_max=n_trials)
    alpha_rng = alphas
    l1_rng    = l1_ratios

    if not refit_each_iter:
        best_params = _tune_params(X[tr_idx], residuals[tr_idx], trials_base,
                                  alpha_rng, l1_rng, fixed_a=fixed_alpha,
                                  fixed_l1=fixed_l1_ratio)
        best_alpha, best_l1 = best_params["alpha"], best_params["l1_ratio"]
    else:
        best_alpha = fixed_alpha
        best_l1    = fixed_l1_ratio

    best_metric = -np.inf
    bad_steps = 0

    for it in range(n_iter):
        # Refresh working set every `refresh` iterations
        if (cached_top_idx is None) or (it % refresh == 0):
            # Screen by correlation with residuals on Train only (no leakage)
            corrs = _corr_with_y_streaming(X[tr_idx], residuals[tr_idx],
                                           batch_size=batch_corr_cols)
            # Properly select by absolute correlation magnitude
            top_idx = cp.argsort(cp.abs(corrs))[-K:]
            cached_top_idx = top_idx
        else:
            top_idx = cached_top_idx

        # Tune per-iter if requested
        if refit_each_iter:
            trials_eff = _auto_trials(int(top_idx.size) if adaptive_trials else n_trials,
                                      base_max=n_trials)
            bp = _tune_params(X[tr_idx][:, top_idx], residuals[tr_idx], trials_eff,
                              alpha_rng, l1_rng, fixed_a=fixed_alpha, fixed_l1=fixed_l1_ratio)
            best_alpha, best_l1 = bp["alpha"], bp["l1_ratio"]

        # Fit on train, evaluate on val
        model = ElasticNet(alpha=best_alpha, l1_ratio=best_l1, max_iter=5_000,
                           fit_intercept=True)
        model.fit(X[tr_idx][:, top_idx], residuals[tr_idx])

        # Predict once on ALL rows and update residuals + cumulative predictions
        step_pred_all = model.predict(X[:, top_idx])
        residuals = residuals - step_pred_all
        cum_pred += step_pred_all
        betas_boosting[top_idx] += model.coef_

        # Metrics (use cumulative predictions; out-of-sample if available)
        if use_val:
            val_r2_now = _r2_gpu(y[val_idx], cum_pred[val_idx])
            val_r2_hist.append(val_r2_now)
            metric_now = val_r2_now
        else:
            # Fallback to train R^2 if no validation set (may be optimistic)
            metric_now = _r2_gpu(y[tr_idx], cum_pred[tr_idx])
        h2_estimates.append(metric_now)

        # Early stopping logic
        if it >= warmup:
            if metric_now > best_metric + min_delta:
                best_metric = metric_now
                bad_steps = 0
            else:
                bad_steps += 1
                if bad_steps >= patience:
                    break

    # Final Ridge refit on kept features
    kept_idx = cp.where(betas_boosting != 0)[0]
    ridge_betas_full = cp.zeros(X.shape[1], dtype=cp.float32)
    kept_snps, final_r2, ridge_model = [], 0.0, None

    if len(kept_idx) > 0:
        ridge_cv = min(3, cv)
        best_ridge = _tune_ridge_optuna(X[:, kept_idx], y, ridge_grid=ridge_grid,
                                        cv=ridge_cv, subsample_frac=subsample_eff)

        ridge_model = Ridge(alpha=best_ridge["alpha"])
        # Fit Ridge on TRAIN rows if we have a validation split (prevents leakage)
        Xk_fit = X[tr_idx][:, kept_idx] if use_val else X[:, kept_idx]
        yk_fit = y[tr_idx] if use_val else y
        ridge_model.fit(Xk_fit, yk_fit)
        preds_fit = ridge_model.predict(Xk_fit)
        final_r2 = _r2_gpu(yk_fit, preds_fit)
        ridge_betas_full[kept_idx] = ridge_model.coef_
        kept_snps = [snp_ids[i] for i in kept_idx.get().tolist()]

    # SNP-based variance explained (legacy)
    snp_variances = (X[tr_idx] if use_val else X).var(axis=0)
    if len(kept_idx) > 0:
        # Diagonal-only estimate on the fit-scope rows (clipped to [0,1])
        h2_diag = float(cp.clip(
            cp.sum((ridge_betas_full[kept_idx] ** 2) *
                   cp.var((X[tr_idx] if use_val else X)[:, kept_idx], axis=0)),
            0.0, 1.0
        ))
    else:
        h2_diag = 0.0
    
    # LD-aware h2 estimate via b^T Eta b using Train covariance
    if len(kept_idx) > 0:
        Xk = (X[tr_idx][:, kept_idx] if use_val else X[:, kept_idx]).astype(cp.float64)
        Xk = Xk - cp.mean(Xk, axis=0)
        Sigma = (Xk.T @ Xk) / (Xk.shape[0] - 1)
        b64 = ridge_betas_full[kept_idx].astype(cp.float64)
        vy = float(cp.var((y[tr_idx] if use_val else y).astype(cp.float64)))
        h2_ld = float(cp.clip((b64.T @ Sigma @ b64) / (vy if vy > 0 else 1.0), 0.0, 1.0))
    else:
        h2_ld = 0.0

    # Preferred: validation R^2 if we have a val set; else fall back to fit-scope R^2
    if len(kept_idx) > 0 and use_val:
        h2_val = _r2_gpu(y[val_idx], ridge_model.predict(X[val_idx][:, kept_idx]))
    else:
        h2_val = final_r2

    return {
        "betas_boosting": betas_boosting,
        "h2_estimates": h2_estimates,
        "val_r2_hist": val_r2_hist,
        "kept_snps": kept_snps,
        "ridge_betas_full": ridge_betas_full,
        "final_r2": final_r2,
        "ridge_model": ridge_model,
        "snp_ids": snp_ids,
        "snp_variances": snp_variances,
        "h2_unscaled": h2_diag, # legacy value to avoid breakage
        "h2_ld": h2_ld,
        "h2_val": h2_val,
        "best_enet": {"alpha": best_alpha, "l1_ratio": best_l1},
        "early_stop": {"metric": metric_mode, "best": best_metric,
                       "iters_run": len(h2_estimates)}
    }


def _tune_elasticnet_optuna(X, y, n_trials=20, cv=5, max_iter=5000,
                            subsample_frac=0.7, alpha_range=(1e-2, 1.0),
                            l1_range=(0.1, 0.9)):
    # Subsample
    n_samples = X.shape[0]
    idx = cp.random.choice(n_samples, int(n_samples * subsample_frac), replace=False)
    X_sub, y_sub = X[idx], y[idx]

    # Move index to CPU for sklearn's KFold
    idx_np = cp.asnumpy(cp.arange(X_sub.shape[0]))
    n_jobs = cpu_count()

    def objective(trial):
        alpha = trial.suggest_float("alpha", alpha_range[0], alpha_range[1], log=True)
        l1_ratio = trial.suggest_float("l1_ratio", l1_range[0], l1_range[1])

        kf = KFold(n_splits=cv, shuffle=True, random_state=13)
        mse_scores = []

        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(idx_np)):
            train_idx = cp.asarray(train_idx)
            val_idx = cp.asarray(val_idx)

            X_train, y_train = X_sub[train_idx], y_sub[train_idx]
            X_val, y_val = X_sub[val_idx], y_sub[val_idx]

            model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio,
                               max_iter=max_iter, fit_intercept=True)
            model.fit(X_train, y_train)

            preds = model.predict(X_val)
            mse = cp.mean((preds - y_val) ** 2).item()
            mse_scores.append(mse)
            trial.report(mse, step=fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return float(cp.mean(cp.asarray(mse_scores)))

    study = optuna.create_study(direction="minimize",
                                pruner=optuna.pruners.MedianPruner(n_warmup_steps=2))
    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs, timeout=60)

    return {
        "alpha": study.best_params["alpha"],
        "l1_ratio": study.best_params["l1_ratio"],
    }


def _tune_ridge_optuna(X, y, cv=5, subsample_frac=0.7,
                       ridge_grid=(1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1, 3, 10)):
    # Subsample
    n_samples = X.shape[0]
    idx = cp.random.choice(n_samples, int(n_samples * subsample_frac), replace=False)
    X_sub, y_sub = X[idx], y[idx]

    # Move index to CPU for sklearn's KFold
    idx_np = cp.asnumpy(cp.arange(X_sub.shape[0]))
    n_jobs = cpu_count()

    def objective(trial):
        alpha = trial.suggest_categorical("alpha", ridge_grid)

        kf = KFold(n_splits=cv, shuffle=True, random_state=13)
        tasks = []

        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(idx_np)):
            train_idx = cp.asarray(train_idx)
            val_idx = cp.asarray(val_idx)

            X_train, y_train = X_sub[train_idx], y_sub[train_idx]
            X_val, y_val = X_sub[val_idx], y_sub[val_idx]

            tasks.append(_fit_ridge_delayed(X_train, y_train, X_val, y_val, alpha))

        mses = compute(*tasks)
        mses = [float(m.item() if hasattr(m, "item") else m) for m in mses]
        return float(np.mean(mses))

    study = optuna.create_study(direction="minimize")
    n_trials = len(ridge_grid)
    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs, timeout=60)

    return {"alpha": study.best_params["alpha"]}


def _cv_elasticnet(X, y, alphas, l1_ratios, cv=5, max_iter=5000, subsample_frac=0.7):
    """
    Manual cross-validation for cuML ElasticNet.
    Evaluates all (alpha, l1_ratio) combos using CuPy batching.
    """
    # Subsample
    n_samples = X.shape[0]
    idx = cp.random.choice(n_samples, int(n_samples * subsample_frac), replace=False)
    X_sub = X[idx]
    y_sub = y[idx]

    # CPU index for KFold
    idx_np = cp.asnumpy(cp.arange(X_sub.shape[0]))

    kf = KFold(n_splits=cv, shuffle=True, random_state=13)
    param_grid = list(product(alphas, l1_ratios))
    scores_accumulator = {param: 0.0 for param in param_grid}

    for train_idx, val_idx in kf.split(idx_np):
        train_idx = cp.asarray(train_idx)
        val_idx = cp.asarray(val_idx)

        X_train, y_train = X_sub[train_idx], y_sub[train_idx]
        X_val, y_val = X_sub[val_idx], y_sub[val_idx]

        tasks = [
            _fit_score_delayed(X_train, y_train, X_val, y_val, alpha, l1,
                               max_iter, optuna=False)
            for (alpha, l1) in param_grid
        ]

        results = compute(*tasks)

        for mse, param in results:
            scores_accumulator[param] += mse

    # Average scores
    avg_scores = {param: score / cv for param, score in scores_accumulator.items()}
    best_param = min(avg_scores, key=avg_scores.get)

    return {"alpha": best_param[0], "l1_ratio": best_param[1]}


def _fit_ridge_delayed(X_train, y_train, X_val, y_val, alpha):
    @delayed
    def task():
        model = Ridge(alpha=alpha)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        mse = cp.mean((preds - y_val) ** 2)
        return mse
    return task()


def _r2_gpu(y_true, y_pred):
    """
    SSE/SST definition, clipped to [0,1].
    Avoids small-sample quirks of corrcoef-based R^2 and keeps null regions at ~0.
    """
    mask = ~cp.isnan(y_true) & ~cp.isnan(y_pred)
    if mask.sum() < 2:
        return 0.0
    yt, yp = y_true[mask], y_pred[mask]
    sst = cp.sum((yt - cp.mean(yt))**2)
    if sst <= 0:
        return 0.0
    sse = cp.sum((yt - yp)**2)
    r2 = 1.0 - (sse / sst)
    return float(cp.clip(r2, 0.0, 1.0))


def _auto_trials(M, base_max: int, min_trials: int = 5):
    # Fewer trials when feature count is small; caps at base_max
    # e.g., sqrt scaling works well in practice
    t = int(np.ceil(np.sqrt(max(1, M)) / 2))
    return max(min_trials, min(base_max, t))

## OLD FUNCTIONS
def _fit_score_delayed(X_train, y_train, X_val, y_val, alpha, l1,
                       max_iter, optuna=True):
    @delayed
    def task():
        model = ElasticNet(alpha=alpha, l1_ratio=l1,
                           max_iter=max_iter, fit_intercept=True)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        mse = cp.mean((preds - y_val) ** 2)
        if optuna:
            return mse
        else:
            return mse, (alpha, l1)
    return task()
