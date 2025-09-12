import cupy as cp
import pytest
from genboostgpu.boosting_net import (
    boosting_elastic_net, _cv_elasticnet, _cv_ridge
)

def test_cv_elasticnet_finds_valid_params():
    X = cp.random.randn(20, 5)
    y = cp.random.randn(20)
    result = _cv_elasticnet(X, y, alphas=[0.1, 1.0], l1_ratios=[0.1, 0.9], cv=2)
    assert "alpha" in result and "l1_ratio" in result
    assert result["alpha"] in [0.1, 1.0]

def test_cv_ridge_finds_valid_alpha():
    X = cp.random.randn(20, 5)
    y = cp.random.randn(20)
    result = _cv_ridge(X, y, alphas=[0.1, 1.0], cv=2)
    assert "alpha" in result
    assert result["alpha"] in [0.1, 1.0]

def test_boosting_elastic_net_runs_and_returns_dict():
    X = cp.random.randn(30, 10)
    y = cp.random.randn(30)
    snp_ids = [f"snp{i}" for i in range(10)]

    results = boosting_elastic_net(X, y, snp_ids, n_iter=5, batch_size=3, cv=2)
    assert isinstance(results, dict)
    assert "final_r2" in results
    assert results["betas_boosting"].shape[0] == 10
    assert len(results["snp_ids"]) == 10
    assert results["best_enet"]["alpha"] is not None
    assert results["h2_unscaled"] >= 0
