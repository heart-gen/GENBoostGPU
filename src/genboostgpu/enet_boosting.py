import cudf
import optuna
import cupy as cp
import numpy as np
import pandas as pd
from itertools import product
from dask import delayed, compute
from multiprocessing import cpu_count
from sklearn.model_selection import KFold
from cuml.linear_model import ElasticNet, Ridge

__all__ = [
    "boosting_elastic_net",
]

def boosting_elastic_net(
        X, y, snp_ids, n_iter=50, batch_size=500, n_trials=20,
        alphas=(0.1, 1.0), l1_ratios=(0.1, 0.9), subsample_frac=0.7,
        ridge_grid=(1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1, 3, 10),
        cv=5, refit_each_iter=False, standardize=True
):
    """
    Boosting ElasticNet with final Ridge refit,
    genome-wide betas, and SNP-based variance components.
    """
    # Standardization
    if standardize:
        X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-6)
        y = (y - y.mean()) / (y.std() + 1e-6)

    residuals = y.copy()
    betas_boosting = cp.zeros(X.shape[1])
    h2_estimates = []

    # Global hyperparameters (if not tuning each iter)
    if not refit_each_iter:
        best_params = _tune_elasticnet_optuna(X, y, n_trials=n_trials, cv=cv,
                                              alpha_range=alphas, l1_range=l1_ratios,
                                              subsample_frac=subsample_frac)
        best_alpha, best_l1 = best_params["alpha"], best_params["l1_ratio"]
    else:
        best_alpha, best_l1 = None, None

    for it in range(n_iter):
        # correlation between residuals and SNPs
        corrs = cp.corrcoef(X.T, residuals)[-1, :-1]
        top_idx = cp.argsort(cp.abs(corrs))[-batch_size:]

        # choose params
        if refit_each_iter:
            best_params = _tune_elasticnet_optuna(X[:, top_idx], residuals,
                                                  n_trials=n_trials, cv=cv)
            best_alpha, best_l1 = best_params["alpha"], best_params["l1_ratio"]

        # Fit elastic net
        model = ElasticNet(alpha=best_alpha, l1_ratio=best_l1, max_iter=5_000)
        model.fit(X[:, top_idx], residuals)
        preds = model.predict(X[:, top_idx])


        # accumulate betas
        residuals = residuals - preds
        betas_boosting[top_idx] += model.coef_

        h2_estimates.append(cp.var(preds).item())

        # early stopping
        if it > 10 and np.std(h2_estimates[-5:]) < 1e-4:
            break

    # Final Ridge refit (manual CV)
    kept_idx = cp.where(betas_boosting != 0)[0]
    ridge_betas_full = cp.zeros(X.shape[1])
    kept_snps = []
    final_r2 = None
    ridge_model = None

    if len(kept_idx) > 0:
        ridge_cv = min(3, cv)
        best_ridge = _tune_ridge_optuna(X[:, kept_idx], y, ridge_grid=ridge_grid,
                                        cv=ridge_cv, subsample_frac=subsample_frac)
        best_ridge_alpha = best_ridge["alpha"]

        ridge_model = Ridge(alpha=best_ridge_alpha)
        ridge_model.fit(X[:, kept_idx], y)

        preds = ridge_model.predict(X[:, kept_idx])
        valid_mask = ~cp.isnan(y) & ~cp.isnan(preds)
        if valid_mask.sum() > 1 and cp.var(preds[valid_mask]) > 0 and cp.var(y[valid_mask]) > 0:
            r2 = cp.corrcoef(y[valid_mask], preds[valid_mask])[0, 1] ** 2
            final_r2 = float(r2)
        else:
            final_r2 = 0.0

        # Ridge mask for all tested SNPs
        ridge_betas_full[kept_idx] = ridge_model.coef_
        kept_snps = [snp_ids[i] for i in kept_idx.tolist()]

    # SNP-based variance explained
    snp_variances = X.var(axis=0)
    h2_unscaled = float(cp.sum(ridge_betas_full ** 2 * snp_variances))

    return {
        "betas_boosting": betas_boosting,
        "h2_estimates": h2_estimates,
        "kept_snps": kept_snps,
        "ridge_betas_full": ridge_betas_full,
        "final_r2": final_r2,
        "ridge_model": ridge_model,
        "snp_ids": snp_ids, # this is the original SNP ids
        "h2_unscaled": h2_unscaled,
        "snp_variances": snp_variances,
        "best_enet": {"alpha": best_alpha, "l1_ratio": best_l1},
        "best_ridge": {"alpha": best_ridge["alpha"] if kept_idx.size > 0 else None}
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
        return cp.mean(cp.asarray(mses)).item()

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
