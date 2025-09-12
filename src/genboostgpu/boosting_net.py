import cudf
import cupy as cp
import numpy as np
import pandas as pd
from cuml.linear_model import ElasticNetCV, RidgeCV

def boosting_elastic_net(
        X, y, snp_ids, n_iter=50, batch_size=500,
        alphas=[0.1, 0.5, 1.0], l1_ratios=[0.1, 0.5, 0.9],
        cv=5, refit_each_iter=False,
):
    """
    Boosting ElasticNetCV with final RidgeCV refit,
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
        model_cv = ElasticNetCV(alphas=alphas, l1_ratio=l1_ratios,
                                cv=cv, max_iter=5_000)
        model_cv.fit(X, y)
        best_alpha = model_cv.alpha_
        best_l1 = model_cv.l1_ratio_
    else:
        best_alpha, best_l1 = None, None
        
    for it in range(n_iter):
        # correlation between residuals and SNPs
        corrs = cp.corrcoef(X.T, residuals)[-1, :-1]
        top_idx = cp.argsort(cp.abs(corrs))[-batch_size:]

        # run elastic net
        if refit_each_iter:
            model = ElasticNetCV(alphas=alphas, l1_ratio=l1_ratios,
                                 cv=cv, max_iter=5000)
            model.fit(X[:, top_idx], residuals)
            best_alpha = model.alpha_
            best_l1 = model.l1_ratio_
        else:
            model = ElasticNetCV(alphas=[best_alpha],
                                 l1_ratio=[best_l1],
                                 cv=cv, max_iter=5_000)
            model.fit(X[:, top_idx], residuals)

        preds = model.predict(X[:, top_idx])
        residuals = residuals - preds

        # accumulate betas
        betas_boosting[top_idx] += model.coef_

        h2_estimates.append(cp.var(preds).item())

        # early stopping
        if it > 10 and np.std(h2_estimates[-5:]) < 1e-4:
            break

    # Final RidgeCV refit
    kept_idx = cp.where(betas_boosting != 0)[0]
    ridge_betas_full = cp.zeros(X.shape[1])
    kept_snps = []
    final_r2 = None
    ridge_model = None

    if len(kept_idx) > 0:
        ridge_model = RidgeCV(alphas=(0.1, 1.0, 10.0), cv=cv)
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
        "alpha": best_alpha,
        "l1_ratio": best_l1,
        "kept_snps": kept_snps,
        "final_r2": final_r2,
        "betas_boosting": betas_boosting,
        "h2_estimates": h2_estimates,
        "ridge_betas_full": ridge_betas_full,
        "ridge_model": ridge_model,
        "snp_ids": snp_ids, # this is the original SNP ids
        "h2_unscaled": h2_unscaled,
        "snp_variances": snp_variances
    }
