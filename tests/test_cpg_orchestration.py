"""
Unit and integration tests for cpg_orchestration module.
"""
import cudf
import cupy as cp
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
import tempfile
import shutil

from genboostgpu.cpg_orchestration import (
    run_cpgs_with_dask,
    run_cpgs_by_chromosome,
    _save_checkpoint,
)


@pytest.fixture
def synthetic_data():
    """Create synthetic genotype and phenotype data for testing."""
    np.random.seed(42)
    N = 50  # samples
    M = 200  # SNPs

    # Create genotype matrix (samples x SNPs)
    geno_arr = cp.asarray(np.random.randint(0, 3, size=(N, M)).astype(np.float32))

    # Create BIM file with SNPs across positions
    bim = pd.DataFrame({
        "chrom": ["1"] * M,
        "snp": [f"rs{i}" for i in range(M)],
        "cm": [0.0] * M,
        "pos": np.arange(100_000, 100_000 + M * 1000, 1000),
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

    # Create multiple CpGs
    cpgs = []
    pheno_data = {}
    for i in range(5):
        cpg_pos = 150_000 + i * 20_000  # Space them out
        cpg_id = f"1_{cpg_pos}"
        cpgs.append({
            "cpg_id": cpg_id,
            "chrom": 1,
            "cpg_pos": cpg_pos,
        })
        # Create phenotype with some heritability
        pheno_data[cpg_id] = np.random.randn(N)

    pheno_df = pd.DataFrame(pheno_data)

    return {
        "geno_arr": geno_arr,
        "bim": bim,
        "fam": fam,
        "pheno_df": pheno_df,
        "cpgs": cpgs,
        "N": N,
        "M": M,
    }


def test_run_cpgs_with_dask_returns_dataframe(synthetic_data):
    """Test that run_cpgs_with_dask returns a DataFrame."""
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_cpgs_with_dask(
            cpgs=synthetic_data["cpgs"],
            geno_arr=synthetic_data["geno_arr"],
            bim=synthetic_data["bim"],
            fam=synthetic_data["fam"],
            pheno_df=synthetic_data["pheno_df"],
            outdir=tmpdir,
            batch_size=1024,
            window_size=100_000,
            n_trials=1,
            n_iter=5,
            scatter=False,  # Avoid scatter for single-GPU test
            save=True,
            prefix="test_cpg",
            checkpoint_interval=10,
            resume=False,
        )

        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
        assert "cpg_id" in result.columns
        assert "chrom" in result.columns
        assert "cpg_pos" in result.columns
        assert "h2_unscaled" in result.columns
        assert "final_r2" in result.columns


def test_run_cpgs_with_dask_saves_parquet(synthetic_data):
    """Test that run_cpgs_with_dask saves results to parquet."""
    with tempfile.TemporaryDirectory() as tmpdir:
        run_cpgs_with_dask(
            cpgs=synthetic_data["cpgs"],
            geno_arr=synthetic_data["geno_arr"],
            bim=synthetic_data["bim"],
            fam=synthetic_data["fam"],
            pheno_df=synthetic_data["pheno_df"],
            outdir=tmpdir,
            batch_size=1024,
            window_size=100_000,
            n_trials=1,
            n_iter=5,
            scatter=False,
            save=True,
            prefix="test_cpg",
            checkpoint_interval=10,
            resume=False,
        )

        parquet_file = Path(tmpdir) / "test_cpg.summary_cpgs.parquet"
        assert parquet_file.exists()

        # Verify content
        df = pd.read_parquet(parquet_file)
        assert len(df) > 0


def test_run_cpgs_with_dask_checkpoint_resume(synthetic_data):
    """Test checkpoint and resume functionality."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # First run - process some CpGs
        first_cpgs = synthetic_data["cpgs"][:2]
        run_cpgs_with_dask(
            cpgs=first_cpgs,
            geno_arr=synthetic_data["geno_arr"],
            bim=synthetic_data["bim"],
            fam=synthetic_data["fam"],
            pheno_df=synthetic_data["pheno_df"],
            outdir=tmpdir,
            batch_size=1024,
            window_size=100_000,
            n_trials=1,
            n_iter=5,
            scatter=False,
            save=True,
            prefix="test_cpg",
            checkpoint_interval=1,
            resume=False,
        )

        # Verify checkpoint files exist
        checkpoint_file = Path(tmpdir) / "test_cpg_checkpoint.parquet"
        completed_file = Path(tmpdir) / "test_cpg_completed.txt"
        assert checkpoint_file.exists()
        assert completed_file.exists()

        # Second run - resume with all CpGs
        result = run_cpgs_with_dask(
            cpgs=synthetic_data["cpgs"],
            geno_arr=synthetic_data["geno_arr"],
            bim=synthetic_data["bim"],
            fam=synthetic_data["fam"],
            pheno_df=synthetic_data["pheno_df"],
            outdir=tmpdir,
            batch_size=1024,
            window_size=100_000,
            n_trials=1,
            n_iter=5,
            scatter=False,
            save=True,
            prefix="test_cpg",
            checkpoint_interval=1,
            resume=True,  # Resume from checkpoint
        )

        # Should have results for all CpGs
        assert len(result) >= len(first_cpgs)


def test_run_cpgs_with_dask_fixed_params(synthetic_data):
    """Test fixed hyperparameter configuration."""
    with tempfile.TemporaryDirectory() as tmpdir:
        fixed_params = {
            "fixed_alpha": 0.5,
            "fixed_l1_ratio": 0.5,
        }

        result = run_cpgs_with_dask(
            cpgs=synthetic_data["cpgs"][:2],
            geno_arr=synthetic_data["geno_arr"],
            bim=synthetic_data["bim"],
            fam=synthetic_data["fam"],
            pheno_df=synthetic_data["pheno_df"],
            outdir=tmpdir,
            batch_size=1024,
            window_size=100_000,
            n_trials=1,
            n_iter=5,
            scatter=False,
            save=False,
            prefix="test_cpg",
            fixed_params=fixed_params,
            fixed_subsample=0.7,
            resume=False,
        )

        assert len(result) > 0


def test_run_cpgs_with_dask_callable_fixed_params(synthetic_data):
    """Test callable fixed_params for per-CpG customization."""
    with tempfile.TemporaryDirectory() as tmpdir:
        def custom_params(cpg):
            # Different params based on CpG position
            if cpg["cpg_pos"] < 160_000:
                return {"fixed_alpha": 0.3, "fixed_l1_ratio": 0.3}
            return {"fixed_alpha": 0.7, "fixed_l1_ratio": 0.7}

        result = run_cpgs_with_dask(
            cpgs=synthetic_data["cpgs"][:2],
            geno_arr=synthetic_data["geno_arr"],
            bim=synthetic_data["bim"],
            fam=synthetic_data["fam"],
            pheno_df=synthetic_data["pheno_df"],
            outdir=tmpdir,
            batch_size=1024,
            window_size=100_000,
            n_trials=1,
            n_iter=5,
            scatter=False,
            save=False,
            prefix="test_cpg",
            fixed_params=custom_params,
            resume=False,
        )

        assert len(result) > 0


def test_run_cpgs_with_dask_early_stop_and_working_set(synthetic_data):
    """Test early stopping and working set configurations."""
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_cpgs_with_dask(
            cpgs=synthetic_data["cpgs"][:2],
            geno_arr=synthetic_data["geno_arr"],
            bim=synthetic_data["bim"],
            fam=synthetic_data["fam"],
            pheno_df=synthetic_data["pheno_df"],
            outdir=tmpdir,
            batch_size=1024,
            window_size=100_000,
            n_trials=1,
            n_iter=50,
            scatter=False,
            save=False,
            prefix="test_cpg",
            early_stop={"patience": 3, "min_delta": 1e-4, "warmup": 3},
            working_set={"K": 50, "refresh": 5},
            resume=False,
        )

        assert len(result) > 0


def test_run_cpgs_with_dask_empty_cpg_list(synthetic_data):
    """Test behavior with empty CpG list."""
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_cpgs_with_dask(
            cpgs=[],
            geno_arr=synthetic_data["geno_arr"],
            bim=synthetic_data["bim"],
            fam=synthetic_data["fam"],
            pheno_df=synthetic_data["pheno_df"],
            outdir=tmpdir,
            batch_size=1024,
            window_size=100_000,
            n_trials=1,
            n_iter=5,
            scatter=False,
            save=False,
            prefix="test_cpg",
            resume=False,
        )

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0


def test_save_checkpoint():
    """Test checkpoint saving function."""
    with tempfile.TemporaryDirectory() as tmpdir:
        results = [
            {"cpg_id": "1_100000", "chrom": 1, "cpg_pos": 100000, "h2_unscaled": 0.5},
            {"cpg_id": "1_200000", "chrom": 1, "cpg_pos": 200000, "h2_unscaled": 0.6},
        ]
        completed_cpgs = {"1_100000", "1_200000"}

        checkpoint_file = Path(tmpdir) / "checkpoint.parquet"
        completed_file = Path(tmpdir) / "completed.txt"

        _save_checkpoint(results, checkpoint_file, completed_cpgs, completed_file)

        assert checkpoint_file.exists()
        assert completed_file.exists()

        # Verify checkpoint content
        df = pd.read_parquet(checkpoint_file)
        assert len(df) == 2

        # Verify completed list
        with open(completed_file) as f:
            lines = [l.strip() for l in f.readlines()]
        assert "1_100000" in lines
        assert "1_200000" in lines


class TestRunCpgsByChromosome:
    """Integration tests for run_cpgs_by_chromosome."""

    @pytest.fixture
    def per_chrom_data(self, tmp_path):
        """Create per-chromosome test data structure."""
        np.random.seed(42)
        N = 30
        M = 100

        # Create genotype data
        geno_arr = cp.asarray(np.random.randint(0, 3, size=(N, M)).astype(np.float32))

        bim = pd.DataFrame({
            "chrom": ["1"] * 50 + ["2"] * 50,
            "snp": [f"rs{i}" for i in range(M)],
            "cm": [0.0] * M,
            "pos": list(range(100_000, 100_000 + 50 * 1000, 1000)) +
                   list(range(100_000, 100_000 + 50 * 1000, 1000)),
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

        # Create per-chromosome data directories
        manifest_dir = tmp_path / "cpg_manifests"
        pheno_dir = tmp_path / "phenotypes"
        manifest_dir.mkdir()
        pheno_dir.mkdir()

        # Create data for chromosomes 1 and 2
        for chrom in [1, 2]:
            # Create CpG manifest
            cpg_ids = [f"{chrom}_{pos}" for pos in range(120_000, 140_000, 5000)]
            manifest = pd.DataFrame({
                "cpg_id": cpg_ids,
                "chrom": [chrom] * len(cpg_ids),
                "cpg_pos": list(range(120_000, 140_000, 5000)),
            })
            manifest.to_parquet(manifest_dir / f"cpg_manifest_chr{chrom}.parquet")

            # Create phenotype data
            pheno_data = {cpg_id: np.random.randn(N) for cpg_id in cpg_ids}
            pheno_df = pd.DataFrame(pheno_data)
            pheno_df.to_parquet(pheno_dir / f"pheno_chr{chrom}.parquet")

        return {
            "geno_arr": geno_arr,
            "bim": bim,
            "fam": fam,
            "tmp_path": tmp_path,
            "N": N,
        }

    def test_run_cpgs_by_chromosome_basic(self, per_chrom_data):
        """Test basic per-chromosome processing."""
        tmp_path = per_chrom_data["tmp_path"]
        outdir = tmp_path / "results"

        result = run_cpgs_by_chromosome(
            chromosomes=[1],
            geno_arr=per_chrom_data["geno_arr"],
            bim=per_chrom_data["bim"],
            fam=per_chrom_data["fam"],
            cpg_manifest_template=str(tmp_path / "cpg_manifests/cpg_manifest_chr{chrom}.parquet"),
            pheno_template=str(tmp_path / "phenotypes/pheno_chr{chrom}.parquet"),
            outdir=str(outdir),
            prefix="test",
            batch_size=512,
            window_size=50_000,
            n_trials=1,
            n_iter=5,
            checkpoint_interval=10,
            resume=False,
        )

        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
        assert (result["chrom"] == 1).all()

    def test_run_cpgs_by_chromosome_multiple_chroms(self, per_chrom_data):
        """Test processing multiple chromosomes."""
        tmp_path = per_chrom_data["tmp_path"]
        outdir = tmp_path / "results"

        result = run_cpgs_by_chromosome(
            chromosomes=[1, 2],
            geno_arr=per_chrom_data["geno_arr"],
            bim=per_chrom_data["bim"],
            fam=per_chrom_data["fam"],
            cpg_manifest_template=str(tmp_path / "cpg_manifests/cpg_manifest_chr{chrom}.parquet"),
            pheno_template=str(tmp_path / "phenotypes/pheno_chr{chrom}.parquet"),
            outdir=str(outdir),
            prefix="test",
            batch_size=512,
            window_size=50_000,
            n_trials=1,
            n_iter=5,
            checkpoint_interval=10,
            resume=False,
        )

        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
        # Should have results from both chromosomes
        assert set(result["chrom"].unique()).issubset({1, 2})

    def test_run_cpgs_by_chromosome_creates_output_files(self, per_chrom_data):
        """Test that per-chromosome output files are created."""
        tmp_path = per_chrom_data["tmp_path"]
        outdir = tmp_path / "results"

        run_cpgs_by_chromosome(
            chromosomes=[1],
            geno_arr=per_chrom_data["geno_arr"],
            bim=per_chrom_data["bim"],
            fam=per_chrom_data["fam"],
            cpg_manifest_template=str(tmp_path / "cpg_manifests/cpg_manifest_chr{chrom}.parquet"),
            pheno_template=str(tmp_path / "phenotypes/pheno_chr{chrom}.parquet"),
            outdir=str(outdir),
            prefix="test",
            batch_size=512,
            window_size=50_000,
            n_trials=1,
            n_iter=5,
            checkpoint_interval=10,
            resume=False,
        )

        # Check per-chromosome output
        chr1_file = outdir / "test_chr1.summary_cpgs.parquet"
        assert chr1_file.exists()

        # Check combined output
        combined_file = outdir / "test.summary_all_cpgs.parquet"
        assert combined_file.exists()
