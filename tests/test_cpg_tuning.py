"""
Unit tests for cpg_tuning module.
"""
import cupy as cp
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
import tempfile

from genboostgpu.cpg_tuning import (
    select_tuning_cpgs,
    _load_cpgs_from_chromosomes,
    _add_snp_counts,
    _infer_N,
)


@pytest.fixture
def synthetic_data():
    """Create synthetic data for tuning tests."""
    np.random.seed(42)
    N = 50
    M = 200

    # Create genotype matrix
    geno_arr = cp.asarray(np.random.randint(0, 3, size=(N, M)).astype(np.float32))

    # Create BIM with SNPs across positions
    bim = pd.DataFrame({
        "chrom": ["1"] * 100 + ["2"] * 100,
        "snp": [f"rs{i}" for i in range(M)],
        "cm": [0.0] * M,
        "pos": list(np.arange(100_000, 100_000 + 100 * 1000, 1000)) +
               list(np.arange(100_000, 100_000 + 100 * 1000, 1000)),
        "a1": ["A"] * M,
        "a2": ["G"] * M,
    })

    # Create FAM
    fam = pd.DataFrame({
        "fid": [f"fam{i}" for i in range(N)],
        "iid": [f"sample{i}" for i in range(N)],
        "pid": ["0"] * N,
        "mid": ["0"] * N,
        "sex": [1] * N,
        "pheno": [-9] * N,
    })

    # Create CpG list
    cpgs = []
    for chrom in [1, 2]:
        for pos in range(120_000, 180_000, 10_000):
            cpgs.append({
                "cpg_id": f"{chrom}_{pos}",
                "chrom": chrom,
                "cpg_pos": pos,
            })

    return {
        "geno_arr": geno_arr,
        "bim": bim,
        "fam": fam,
        "cpgs": cpgs,
        "N": N,
        "M": M,
    }


def test_select_tuning_cpgs_returns_list(synthetic_data):
    """Test that select_tuning_cpgs returns a list of CpG dicts."""
    result = select_tuning_cpgs(
        cpgs=synthetic_data["cpgs"],
        bim=synthetic_data["bim"],
        frac=0.5,
        n_min=2,
        n_max=10,
        window_size=50_000,
    )

    assert isinstance(result, list)
    assert len(result) > 0
    assert all(isinstance(c, dict) for c in result)
    assert all("cpg_id" in c for c in result)
    assert all("chrom" in c for c in result)
    assert all("cpg_pos" in c for c in result)


def test_select_tuning_cpgs_respects_n_min_n_max(synthetic_data):
    """Test that selection respects n_min and n_max bounds."""
    # Test n_min
    result_min = select_tuning_cpgs(
        cpgs=synthetic_data["cpgs"],
        bim=synthetic_data["bim"],
        frac=0.01,  # Very small fraction
        n_min=5,
        n_max=100,
        window_size=50_000,
    )
    assert len(result_min) >= 5

    # Test n_max
    result_max = select_tuning_cpgs(
        cpgs=synthetic_data["cpgs"],
        bim=synthetic_data["bim"],
        frac=1.0,  # 100%
        n_min=1,
        n_max=3,
        window_size=50_000,
    )
    assert len(result_max) <= 3


def test_select_tuning_cpgs_stratified_by_snp_count(synthetic_data):
    """Test that selection stratifies by SNP count."""
    result = select_tuning_cpgs(
        cpgs=synthetic_data["cpgs"],
        bim=synthetic_data["bim"],
        frac=0.5,
        n_min=4,
        n_max=20,
        window_size=50_000,
        n_bins=3,
    )

    # With stratification, we should have diversity in positions
    positions = [c["cpg_pos"] for c in result]
    assert len(set(positions)) > 1


def test_select_tuning_cpgs_excludes_failed(synthetic_data):
    """Test that exclude_failed CpGs are not selected."""
    exclude = [synthetic_data["cpgs"][0]["cpg_id"]]

    result = select_tuning_cpgs(
        cpgs=synthetic_data["cpgs"],
        bim=synthetic_data["bim"],
        frac=1.0,
        n_min=1,
        n_max=100,
        window_size=50_000,
        exclude_failed=exclude,
    )

    selected_ids = [c["cpg_id"] for c in result]
    assert exclude[0] not in selected_ids


def test_select_tuning_cpgs_per_chrom_min(synthetic_data):
    """Test per-chromosome minimum selection."""
    result = select_tuning_cpgs(
        cpgs=synthetic_data["cpgs"],
        bim=synthetic_data["bim"],
        frac=0.5,
        n_min=2,
        n_max=20,
        window_size=50_000,
        per_chrom_min=2,  # At least 2 per chromosome
    )

    # Count per chromosome
    chrom_counts = {}
    for c in result:
        chrom_counts[c["chrom"]] = chrom_counts.get(c["chrom"], 0) + 1

    # Each chromosome should have at least 2
    for chrom in chrom_counts:
        assert chrom_counts[chrom] >= 2 or chrom_counts[chrom] >= len(
            [c for c in synthetic_data["cpgs"] if c["chrom"] == chrom]
        )


def test_select_tuning_cpgs_empty_raises(synthetic_data):
    """Test that empty CpG list raises ValueError."""
    with pytest.raises(ValueError, match="No CpGs with SNPs"):
        select_tuning_cpgs(
            cpgs=[],
            bim=synthetic_data["bim"],
            frac=0.5,
            n_min=1,
            n_max=10,
            window_size=50_000,
        )


def test_select_tuning_cpgs_no_snps_raises(synthetic_data):
    """Test that CpGs with no SNPs in window raises ValueError."""
    # CpGs far from any SNPs
    far_cpgs = [
        {"cpg_id": "1_999999999", "chrom": 1, "cpg_pos": 999_999_999},
    ]

    with pytest.raises(ValueError, match="No CpGs with SNPs"):
        select_tuning_cpgs(
            cpgs=far_cpgs,
            bim=synthetic_data["bim"],
            frac=0.5,
            n_min=1,
            n_max=10,
            window_size=1000,  # Small window
        )


def test_add_snp_counts(synthetic_data):
    """Test _add_snp_counts helper function."""
    result = _add_snp_counts(
        cpgs=synthetic_data["cpgs"][:3],
        bim=synthetic_data["bim"],
        window_size=50_000,
    )

    assert isinstance(result, list)
    assert len(result) == 3
    for cpg, M in result:
        assert isinstance(cpg, dict)
        assert isinstance(M, int)
        assert M >= 0


def test_infer_N_from_fam(synthetic_data):
    """Test _infer_N with FAM data."""
    N = _infer_N(fam=synthetic_data["fam"])
    assert N == synthetic_data["N"]


def test_infer_N_from_geno_arr(synthetic_data):
    """Test _infer_N with genotype array."""
    N = _infer_N(geno_arr=synthetic_data["geno_arr"])
    assert N == synthetic_data["N"]


def test_infer_N_priority(synthetic_data):
    """Test _infer_N priority (fam > geno_arr)."""
    # Create mismatched sizes
    small_fam = synthetic_data["fam"].head(10)
    N = _infer_N(fam=small_fam, geno_arr=synthetic_data["geno_arr"])
    # Should use fam (priority)
    assert N == 10


def test_infer_N_no_data():
    """Test _infer_N with no data returns 0."""
    N = _infer_N()
    assert N == 0


class TestLoadCpgsFromChromosomes:
    """Tests for _load_cpgs_from_chromosomes helper."""

    def test_load_parquet_manifests(self, tmp_path):
        """Test loading CpG manifests from parquet files."""
        # Create test manifest files
        manifest_dir = tmp_path / "manifests"
        manifest_dir.mkdir()

        for chrom in [1, 2]:
            manifest = pd.DataFrame({
                "cpg_id": [f"{chrom}_100000", f"{chrom}_200000"],
                "chrom": [chrom, chrom],
                "cpg_pos": [100_000, 200_000],
            })
            manifest.to_parquet(manifest_dir / f"cpg_manifest_chr{chrom}.parquet")

        # Create dummy pheno template
        pheno_dir = tmp_path / "pheno"
        pheno_dir.mkdir()

        result = _load_cpgs_from_chromosomes(
            chromosomes=[1, 2],
            cpg_manifest_template=str(manifest_dir / "cpg_manifest_chr{chrom}.parquet"),
            pheno_template=str(pheno_dir / "pheno_chr{chrom}.parquet"),
        )

        assert len(result) == 4  # 2 CpGs per chromosome
        assert all("cpg_id" in c for c in result)
        assert all("pheno_path" in c for c in result)

    def test_load_missing_chromosome(self, tmp_path):
        """Test that missing chromosomes are skipped gracefully."""
        manifest_dir = tmp_path / "manifests"
        manifest_dir.mkdir()

        # Only create manifest for chromosome 1
        manifest = pd.DataFrame({
            "cpg_id": ["1_100000"],
            "chrom": [1],
            "cpg_pos": [100_000],
        })
        manifest.to_parquet(manifest_dir / "cpg_manifest_chr1.parquet")

        pheno_dir = tmp_path / "pheno"
        pheno_dir.mkdir()

        # Try to load chromosomes 1 and 2 (2 is missing)
        result = _load_cpgs_from_chromosomes(
            chromosomes=[1, 2],
            cpg_manifest_template=str(manifest_dir / "cpg_manifest_chr{chrom}.parquet"),
            pheno_template=str(pheno_dir / "pheno_chr{chrom}.parquet"),
        )

        # Should only have CpGs from chromosome 1
        assert len(result) == 1
        assert result[0]["chrom"] == 1
