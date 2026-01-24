"""
Unit tests for cpg_runner module.
"""
import cudf
import cupy as cp
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
import tempfile
import shutil

from genboostgpu.cpg_runner import run_single_cpg


@pytest.fixture
def synthetic_data():
    """Create synthetic genotype and phenotype data for testing."""
    np.random.seed(42)
    N = 50  # samples
    M = 100  # SNPs

    # Create genotype matrix (samples x SNPs)
    geno_arr = cp.asarray(np.random.randint(0, 3, size=(N, M)).astype(np.float32))

    # Create BIM file
    bim = pd.DataFrame({
        "chrom": ["1"] * M,
        "snp": [f"rs{i}" for i in range(M)],
        "cm": [0.0] * M,
        "pos": np.arange(100_000, 100_000 + M * 1000, 1000),  # 1kb spacing
        "a1": ["A"] * M,
        "a2": ["G"] * M,
    })

    # Create FAM file
    fam = pd.DataFrame({
        "fid": [f"fam{i}" for i in range(N)],
        "iid": [f"sample{i}" for i in range(N)],
        "pid": ["0"] * N,
        "mid": ["0"] * N,
        "sex": [1] * N,
        "pheno": [-9] * N,
    })

    # Create phenotype DataFrame with CpG columns
    # Simulate methylation values correlated with some SNPs
    cpg_pos = 150_000  # CpG position in middle of SNPs
    cpg_id = "1_150000"

    # Create phenotype with some heritability
    beta = np.zeros(M)
    beta[45:55] = np.random.randn(10) * 0.2  # 10 causal SNPs near CpG
    genetic_component = geno_arr.get() @ beta
    noise = np.random.randn(N) * 0.5
    pheno_values = genetic_component + noise

    pheno_df = pd.DataFrame({
        cpg_id: pheno_values,
    })

    return {
        "geno_arr": geno_arr,
        "bim": bim,
        "fam": fam,
        "pheno_df": pheno_df,
        "cpg_id": cpg_id,
        "chrom": 1,
        "cpg_pos": cpg_pos,
        "N": N,
        "M": M,
    }


def test_run_single_cpg_returns_dict(synthetic_data):
    """Test that run_single_cpg returns expected dictionary structure."""
    result = run_single_cpg(
        cpg_id=synthetic_data["cpg_id"],
        chrom=synthetic_data["chrom"],
        cpg_pos=synthetic_data["cpg_pos"],
        geno_arr=synthetic_data["geno_arr"],
        bim=synthetic_data["bim"],
        fam=synthetic_data["fam"],
        pheno_df=synthetic_data["pheno_df"],
        window_size=100_000,
        n_trials=1,
        n_iter=10,
        save_full=False,
    )

    assert result is not None
    assert isinstance(result, dict)
    assert "cpg_id" in result
    assert "chrom" in result
    assert "cpg_pos" in result
    assert "num_snps" in result
    assert "N" in result
    assert "final_r2" in result
    assert "h2_unscaled" in result
    assert "n_iter" in result

    assert result["cpg_id"] == synthetic_data["cpg_id"]
    assert result["chrom"] == synthetic_data["chrom"]
    assert result["cpg_pos"] == synthetic_data["cpg_pos"]
    assert result["N"] == synthetic_data["N"]


def test_run_single_cpg_with_cudf_pheno(synthetic_data):
    """Test that run_single_cpg works with cuDF DataFrame."""
    pheno_cudf = cudf.DataFrame(synthetic_data["pheno_df"])

    result = run_single_cpg(
        cpg_id=synthetic_data["cpg_id"],
        chrom=synthetic_data["chrom"],
        cpg_pos=synthetic_data["cpg_pos"],
        geno_arr=synthetic_data["geno_arr"],
        bim=synthetic_data["bim"],
        fam=synthetic_data["fam"],
        pheno_df=pheno_cudf,
        window_size=100_000,
        n_trials=1,
        n_iter=10,
        save_full=False,
    )

    assert result is not None
    assert result["N"] == synthetic_data["N"]


def test_run_single_cpg_with_direct_pheno(synthetic_data):
    """Test that run_single_cpg works with pre-extracted phenotype array."""
    pheno_arr = cp.asarray(synthetic_data["pheno_df"][synthetic_data["cpg_id"]].values)

    result = run_single_cpg(
        cpg_id=synthetic_data["cpg_id"],
        chrom=synthetic_data["chrom"],
        cpg_pos=synthetic_data["cpg_pos"],
        geno_arr=synthetic_data["geno_arr"],
        bim=synthetic_data["bim"],
        fam=synthetic_data["fam"],
        pheno=pheno_arr,
        window_size=100_000,
        n_trials=1,
        n_iter=10,
        save_full=False,
    )

    assert result is not None
    assert result["N"] == synthetic_data["N"]


def test_run_single_cpg_fixed_params(synthetic_data):
    """Test that fixed hyperparameters are used."""
    result = run_single_cpg(
        cpg_id=synthetic_data["cpg_id"],
        chrom=synthetic_data["chrom"],
        cpg_pos=synthetic_data["cpg_pos"],
        geno_arr=synthetic_data["geno_arr"],
        bim=synthetic_data["bim"],
        fam=synthetic_data["fam"],
        pheno_df=synthetic_data["pheno_df"],
        window_size=100_000,
        n_trials=1,
        n_iter=10,
        fixed_alpha=0.5,
        fixed_l1_ratio=0.5,
        fixed_subsample=0.7,
        save_full=False,
    )

    assert result is not None


def test_run_single_cpg_early_stop(synthetic_data):
    """Test that early stopping works."""
    result = run_single_cpg(
        cpg_id=synthetic_data["cpg_id"],
        chrom=synthetic_data["chrom"],
        cpg_pos=synthetic_data["cpg_pos"],
        geno_arr=synthetic_data["geno_arr"],
        bim=synthetic_data["bim"],
        fam=synthetic_data["fam"],
        pheno_df=synthetic_data["pheno_df"],
        window_size=100_000,
        n_trials=1,
        n_iter=100,  # High n_iter, but early stopping should kick in
        early_stop={"patience": 3, "min_delta": 1e-4, "warmup": 3},
        save_full=False,
    )

    assert result is not None
    # Early stopping may reduce iterations
    assert result["n_iter"] <= 100


def test_run_single_cpg_working_set(synthetic_data):
    """Test that working set configuration works."""
    result = run_single_cpg(
        cpg_id=synthetic_data["cpg_id"],
        chrom=synthetic_data["chrom"],
        cpg_pos=synthetic_data["cpg_pos"],
        geno_arr=synthetic_data["geno_arr"],
        bim=synthetic_data["bim"],
        fam=synthetic_data["fam"],
        pheno_df=synthetic_data["pheno_df"],
        window_size=100_000,
        n_trials=1,
        n_iter=10,
        working_set={"K": 50, "refresh": 5},
        save_full=False,
    )

    assert result is not None


def test_run_single_cpg_save_full(synthetic_data):
    """Test that save_full option creates output files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_single_cpg(
            cpg_id=synthetic_data["cpg_id"],
            chrom=synthetic_data["chrom"],
            cpg_pos=synthetic_data["cpg_pos"],
            geno_arr=synthetic_data["geno_arr"],
            bim=synthetic_data["bim"],
            fam=synthetic_data["fam"],
            pheno_df=synthetic_data["pheno_df"],
            window_size=100_000,
            n_trials=1,
            n_iter=10,
            outdir=tmpdir,
            save_full=True,
        )

        assert result is not None

        # Check that output files were created
        expected_prefix = f"{synthetic_data['cpg_id']}_chr{synthetic_data['chrom']}_{synthetic_data['cpg_pos']}"
        betas_file = Path(tmpdir) / f"{expected_prefix}_betas.tsv"
        h2_file = Path(tmpdir) / f"{expected_prefix}_h2.tsv"

        assert betas_file.exists()
        assert h2_file.exists()


def test_run_single_cpg_returns_none_for_no_snps():
    """Test that run_single_cpg returns None when no SNPs in window."""
    np.random.seed(42)
    N = 50
    M = 10

    geno_arr = cp.asarray(np.random.randint(0, 3, size=(N, M)).astype(np.float32))

    # BIM with SNPs far from CpG position
    bim = pd.DataFrame({
        "chrom": ["1"] * M,
        "snp": [f"rs{i}" for i in range(M)],
        "cm": [0.0] * M,
        "pos": np.arange(1_000_000, 1_000_000 + M * 1000, 1000),  # Far away
        "a1": ["A"] * M,
        "a2": ["G"] * M,
    })

    fam = pd.DataFrame({
        "fid": [f"fam{i}" for i in range(N)],
        "iid": [f"sample{i}" for i in range(N)],
        "pid": ["0"] * N,
        "mid": ["0"] * N,
        "sex": [1] * N,
        "pheno": [-9] * N,
    })

    cpg_id = "1_100000"
    cpg_pos = 100_000  # Far from SNPs

    pheno_df = pd.DataFrame({
        cpg_id: np.random.randn(N),
    })

    result = run_single_cpg(
        cpg_id=cpg_id,
        chrom=1,
        cpg_pos=cpg_pos,
        geno_arr=geno_arr,
        bim=bim,
        fam=fam,
        pheno_df=pheno_df,
        window_size=50_000,  # Small window, no SNPs
        n_trials=1,
        n_iter=10,
        save_full=False,
    )

    assert result is None


def test_run_single_cpg_blacklist_by_cpg_id(synthetic_data):
    """Test that blacklist by cpg_id works."""
    error_regions = pd.DataFrame({
        "cpg_id": [synthetic_data["cpg_id"]],
    })

    result = run_single_cpg(
        cpg_id=synthetic_data["cpg_id"],
        chrom=synthetic_data["chrom"],
        cpg_pos=synthetic_data["cpg_pos"],
        geno_arr=synthetic_data["geno_arr"],
        bim=synthetic_data["bim"],
        fam=synthetic_data["fam"],
        pheno_df=synthetic_data["pheno_df"],
        error_regions=error_regions,
        window_size=100_000,
        n_trials=1,
        n_iter=10,
        save_full=False,
    )

    assert result is None


def test_run_single_cpg_blacklist_by_position(synthetic_data):
    """Test that blacklist by position works."""
    error_regions = pd.DataFrame({
        "Chrom": [synthetic_data["chrom"]],
        "Pos": [synthetic_data["cpg_pos"]],
    })

    result = run_single_cpg(
        cpg_id=synthetic_data["cpg_id"],
        chrom=synthetic_data["chrom"],
        cpg_pos=synthetic_data["cpg_pos"],
        geno_arr=synthetic_data["geno_arr"],
        bim=synthetic_data["bim"],
        fam=synthetic_data["fam"],
        pheno_df=synthetic_data["pheno_df"],
        error_regions=error_regions,
        window_size=100_000,
        n_trials=1,
        n_iter=10,
        save_full=False,
    )

    assert result is None


def test_run_single_cpg_h2_bounds(synthetic_data):
    """Test that h2_unscaled is bounded between 0 and 1."""
    result = run_single_cpg(
        cpg_id=synthetic_data["cpg_id"],
        chrom=synthetic_data["chrom"],
        cpg_pos=synthetic_data["cpg_pos"],
        geno_arr=synthetic_data["geno_arr"],
        bim=synthetic_data["bim"],
        fam=synthetic_data["fam"],
        pheno_df=synthetic_data["pheno_df"],
        window_size=100_000,
        n_trials=1,
        n_iter=10,
        save_full=False,
    )

    assert result is not None
    h2 = result["h2_unscaled"]
    if not np.isnan(h2):
        assert 0.0 <= h2 <= 1.0
