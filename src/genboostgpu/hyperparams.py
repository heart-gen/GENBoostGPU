import numpy as np

__all__ = ["enet_from_targets", "alpha_l1_from_lambda"]

def enet_from_targets(M, N, c_lambda=1.0, c_ridge=1.0, eps=1e-8):
    """
    Convert target sparsity (λ1) & ridge ratio (λ2 = c_ridge*λ1) into
    ElasticNet (alpha, l1_ratio), using λ1 = c_lambda * sqrt(2 log M / N).
    This uses lasso scaling to normalize penalty strength across windows
    with different SNP counts M and sample sizes N.
    """
    M = max(int(M), 2); N = max(int(N), 2)
    lam1 = c_lambda * np.sqrt(2.0 * np.log(M) / N)
    lam2 = c_ridge * lam1
    alpha = lam1 + lam2
    l1_ratio = lam1 / (alpha + eps)
    return float(alpha), float(l1_ratio)


def alpha_l1_from_lambda(lam1, lam2, eps=1e-8):
    alpha = float(lam1 + lam2)
    l1_ratio = float(lam1 / (alpha + eps))
    return alpha, l1_ratio
