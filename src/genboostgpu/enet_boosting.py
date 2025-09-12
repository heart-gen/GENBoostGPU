import cudf
import cupy as cp
import numpy as np
import pandas as pd
from cuml.metrics import mean_squared_error
from cuml.linear_model import ElasticNet, Ridge

__all__ = [
    "boosting_elastic_net",
]

def boosting_elastic_net(
        X, y, snp_ids, n_iter=50, batch_size=500,
        alphas=[0.1, 0.5, 1.0], l1_ratios=[0.1, 0.5, 0.9],
        ridge_alphas=[0.1, 1.0, 10.0], cv=5, refit_each_iter=False,
        standardize=True
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
        best_params = _cv_elasticnet(X, y, alphas, l1_ratios, cv=cv)
        best_alpha, best_l1 = best_params["alpha"], best_params["l1_ratio"]
    else:
        best_alpha, best_l1 = None, None
        
    for it in range(n_iter):
        # correlation between residuals and SNPs
        corrs = cp.corrcoef(X.T, residuals)[-1, :-1]
        top_idx = cp.argsort(cp.abs(corrs))[-batch_size:]

        # choose params
        if refit_each_iter:
            best_params = _cv_elasticnet(X[:, top_idx], residuals,
                                               alphas, l1_ratios, cv=cv)
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
        best_ridge = _cv_ridge(X[:, kept_idx], y,
                                     alphas=ridge_alphas, cv=cv)
        best_ridge_alpha = best_ridge["alpha"]

        ridge_model = Ridge(alpha=best_ridge_alpha)
        ridge_model.fit(X[:, kept_idx], y)

        preds = ridge_model.predict(X[:, kept_idx])
        valid_mask = ~cp.isnan(y) & ~cp.isnan(preds)
        if valid_mask.sum() > 1:
            r2 = cp.corrcoef(y[valid_mask], preds[valid_mask])[0, 1] ** 2
            final_r2 = float(r2)

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


def _cv_elasticnet(X, y, alphas, l1_ratios, cv=5, max_iter=5000):
    """
    Manual cross-validation for cuML ElasticNet.
    Evaluates all (alpha, l1_ratio) combos using CuPy batching.
    """
    n = X.shape[0]
    fold_size = n // cv
    grid = [(a, l) for a in alphas for l in l1_ratios]
    n_grid = len(grid)

    all_scores = cp.zeros(n_grid)

    for k in range(cv):
        val_idx = slice(k * fold_size, (k + 1) * fold_size)
        train_mask = cp.ones(n, dtype=bool)
        train_mask[val_idx] = False

        X_train, y_train = X[train_mask], y[train_mask]
        X_val, y_val = X[val_idx], y[val_idx]

        fold_scores = []
        for i, (alpha, l1) in enumerate(grid):
            model = ElasticNet(alpha=alpha, l1_ratio=l1,
                               max_iter=max_iter)
            model.fit(X_train, y_train)
            preds = model.predict(X_val)

            score = mean_squared_error(y_val, preds)
            fold_scores.append(score)

        # stack scores as cupy array
        all_scores += cp.asarray(fold_scores)

    mean_scores = all_scores / cv
    best_idx = int(cp.argmin(mean_scores))
    best_alpha, best_l1 = grid[best_idx]
    
    return {"alpha": best_alpha, "l1_ratio": best_l1}


def _cv_ridge(X, y, alphas=[0.1, 1.0, 10.0], cv=5):
    """
    Manual cross-validation for cuML Ridge regression.
    Evaluates all alphas in one loop using CuPy batching.
    """
    n = X.shape[0]
    fold_size = n // cv
    n_grid = len(alphas)

    all_scores = cp.zeros(n_grid)

    for k in range(cv):
        val_idx = slice(k * fold_size, (k + 1) * fold_size)
        train_mask = cp.ones(n, dtype=bool)
        train_mask[val_idx] = False

        X_train, y_train = X[train_mask], y[train_mask]
        X_val, y_val = X[val_idx], y[val_idx]

        fold_scores = []
        for alpha in alphas:
            model = Ridge(alpha=alpha)
            model.fit(X_train, y_train)
            preds = model.predict(X_val)

            score = mean_squared_error(y_val, preds)
            fold_scores.append(score)

        all_scores += cp.asarray(fold_scores)

    mean_scores = all_scores / cv
    best_idx = int(cp.argmin(mean_scores))
    best_alpha = alphas[best_idx]

    return {"alpha": best_alpha}
