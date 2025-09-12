import cupy as cp
import cudf
import pandas as pd
from genboostgpu.snp_processing import (
    filter_zero_variance, impute_snps, run_ld_clumping,
    preprocess_genotypes, filter_cis_window
)

def test_filter_zero_variance_removes_constant_snps():
    X = cp.array([[1, 0], [1, 0], [1, 0]])  # first SNP constant
    snp_ids = ["snp1", "snp2"]
    X_filt, snps_filt = filter_zero_variance(X, snp_ids)
    assert "snp1" not in snps_filt
    assert X_filt.shape[1] == 1

def test_impute_snps_replaces_nans():
    X = cp.array([[1, cp.nan], [2, 1]])
    X_imputed = impute_snps(X, strategy="most_frequent")
    assert not cp.isnan(X_imputed).any()

def test_run_ld_clumping_keeps_at_least_one():
    X = cp.random.randint(0, 3, size=(20, 5))
    snp_pos = cp.arange(5) * 1000
    stat = cp.random.randn(5)
    keep = run_ld_clumping(X, snp_pos, stat, r2_thresh=0.9, kb_window=2000)
    assert keep.shape[0] >= 1

def test_preprocess_genotypes_pipeline_runs():
    X = cp.random.randint(0, 3, size=(30, 6)).astype(float)
    snp_ids = [f"snp{i}" for i in range(6)]
    snp_pos = [i*1000 for i in range(6)]
    y = cp.random.randn(30)
    X_proc, snps_proc = preprocess_genotypes(X, snp_ids, snp_pos, y)
    assert X_proc.shape[0] == 30
    assert len(snps_proc) <= 6

def test_filter_cis_window_returns_expected_snps():
    bim = cudf.DataFrame({
        "chrom": ["1","1","1"],
        "pos": [1000, 2000, 5000],
        "snp": ["snp1","snp2","snp3"]
    })
    geno_df = cudf.DataFrame({
        "snp1":[0,1], "snp2":[1,2], "snp3":[0,0]
    })
    geno_window, snps, pos = filter_cis_window(geno_df, bim, chrom=1, pos=2000, window=1000)
    assert "snp2" in snps
    assert geno_window is not None
