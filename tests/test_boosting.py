import cupy as cp
from genboostgpu.enet_boosting import boosting_elastic_net

def test_boosting_runs_on_small_matrix():
    X = cp.random.randint(0, 3, size=(20, 10))  # fake genotypes
    y = cp.random.randn(20)                     # fake phenotype
    snp_ids = [f"snp{i}" for i in range(10)]

    results = boosting_elastic_net(X, y, snp_ids, n_iter=5, batch_size=3)
    assert "final_r2" in results
    assert results["ridge_betas_full"].shape[0] == 10
