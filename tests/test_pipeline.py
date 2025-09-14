import os
import cudf
import pytest
import cupy as cp
import pandas as pd
import genboostgpu.pipeline as pipeline

@pytest.fixture
def toy_data(tmp_path, monkeypatch):
    """
    Create toy geno/phenotype data and patch boosting_elastic_net
    to return a simple mock result.
    """
    # Fake bim annotation
    bim = cudf.DataFrame({
        "chrom": ["1", "1", "1"],
        "pos": [1000, 2000, 3000],
        "snp": ["snp1", "snp2", "snp3"]
    })

    # Fake genotypes (2 samples x 3 SNPs)
    geno_df = cudf.DataFrame({
        "snp1": [0, 1],
        "snp2": [1, 2],
        "snp3": [0, 0],
    })

    # Fake phenotypes (2 samples x 1 CpG)
    pheno_df = cudf.DataFrame({
        "cg0001": [0.5, -0.2]
    })

    cpg_list = [("cg0001", 1, 2000)]

    # Monkeypatch boosting_elastic_net
    def fake_boosting_elastic_net(X, y, snp_ids, **kwargs):
        return {
            "betas_boosting": cp.array([0.1, 0.0, -0.2]),
            "ridge_betas_full": cp.array([0.1, 0.0, -0.2]),
            "snp_ids": snp_ids,
            "snp_variances": cp.array([1.0, 1.0, 1.0]),
            "kept_snps": snp_ids,
            "h2_estimates": [0.01, 0.02],
            "final_r2": 0.5,
            "h2_unscaled": 0.03,
            "best_enet": {"alpha": 0.1, "l1_ratio": 0.5},
            "best_ridge": {"alpha": 1.0},
        }
    monkeypatch.setattr(pipeline, "boosting_elastic_net", fake_boosting_elastic_net)

    return {
        "bim": bim,
        "geno_df": geno_df,
        "pheno_df": pheno_df,
        "cpg_list": cpg_list,
        "tmpdir": tmp_path,
    }

# Tests

def test_prepare_cpg_inputs_returns_expected(toy_data):
    inputs = pipeline.prepare_cpg_inputs(
        toy_data["cpg_list"],
        toy_data["geno_df"],
        toy_data["pheno_df"],
        toy_data["bim"],
        window=1000
    )
    assert isinstance(inputs, list)
    assert len(inputs) == 1
    cpg_id, X, y, snp_ids = inputs[0]
    assert cpg_id == "cg0001"
    assert isinstance(X, cp.ndarray)
    assert y.shape[0] == 2
    assert isinstance(snp_ids, list)


def test_run_boosting_for_cpgs_runs_and_adds_cpg_id(toy_data):
    inputs = pipeline.prepare_cpg_inputs(
        toy_data["cpg_list"],
        toy_data["geno_df"],
        toy_data["pheno_df"],
        toy_data["bim"],
        window=1000
    )
    results = pipeline.run_boosting_for_cpgs(inputs, n_iter=5)
    assert isinstance(results, list)
    assert "cpg_id" in results[0]
    assert results[0]["cpg_id"] == "cg0001"
    assert results[0]["final_r2"] == 0.5


def test_run_boosting_for_cpg_delayed_writes_files(toy_data):
    inputs = pipeline.prepare_cpg_inputs(
        toy_data["cpg_list"],
        toy_data["geno_df"],
        toy_data["pheno_df"],
        toy_data["bim"],
        window=1000
    )
    cpg_id, X, y, snp_ids = inputs[0]
    outdir = toy_data["tmpdir"]

    task = pipeline.run_boosting_for_cpg_delayed(
        cpg_id, X, y, snp_ids, outdir=outdir, save_full_betas=True, overwrite=True
    )
    # Run the delayed task synchronously
    summary_path = task.compute()

    # Check files exist
    assert os.path.exists(summary_path)
    betas_path = os.path.join(outdir, f"{cpg_id}_betas.tsv")
    assert os.path.exists(betas_path)

    # Check summary content
    df = pd.read_csv(summary_path, sep="\t")
    assert "final_r2" in df.columns
    assert df["final_r2"].iloc[0] == 0.5
