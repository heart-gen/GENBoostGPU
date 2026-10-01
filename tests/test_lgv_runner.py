"""End-to-end: generic region inputs -> lgv init/run/combine (CPU, no GEMMA)."""
import os

import numpy as np
import pandas as pd
import pytest

from genboostgpu.cli import main
from genboostgpu.io.genotype import read_bed, write_bed
from genboostgpu.lgv.crossfit import crossfit_elastic_net
from genboostgpu.lgv.folds import stable_seed


@pytest.fixture
def toy_inputs(tmp_path):
    rng = np.random.default_rng(11)
    n, m = 70, 260
    ids = [f"Br{1000 + i}" for i in range(n)]
    maf = rng.uniform(0.05, 0.5, m)
    g = rng.binomial(2, maf, size=(n, m)).astype(float)
    g[rng.random(g.shape) < 0.005] = np.nan
    pos = np.sort(rng.choice(np.arange(10_000, 1_200_000), m, replace=False))
    bim = pd.DataFrame(dict(chrom="1", snp=[f"rs{j}" for j in range(m)], cm=0, pos=pos,
                            a1="A", a2="G"))
    fam = pd.DataFrame(dict(FID=ids, IID=ids, father=0, mother=0, sex=0, pheno=-9))
    write_bed(str(tmp_path / "geno.chr1"), g, bim, fam)
    gi = np.where(np.isnan(g), np.nanmean(g, axis=0), g)
    beta = np.zeros(m)
    beta[rng.choice(m, 5, replace=False)] = 0.6
    regions = pd.DataFrame(dict(region_id=["r1", "r2", "r3"], chrom=["chr1"] * 3,
                                start=[500_000, 600_000, 9_000_000],
                                end=[500_400, 600_400, 9_000_400], n_sites=[8, 6, 9]))
    pheno = pd.DataFrame(dict(sample_id=ids,
                              r1=gi @ beta + rng.normal(size=n),
                              r2=rng.normal(size=n),
                              r3=rng.normal(size=n)))
    covs = pd.DataFrame(dict(sample_id=ids, age=rng.uniform(20, 80, n),
                             sex=rng.choice(["F", "M"], n)))
    regions.to_csv(tmp_path / "regions.tsv", sep="\t", index=False)
    pheno.to_parquet(tmp_path / "pheno.parquet", index=False)
    covs.to_csv(tmp_path / "covs.tsv", sep="\t", index=False)
    return tmp_path, g, pheno, covs


def test_bed_roundtrip(toy_inputs):
    tmp, g, _, _ = toy_inputs
    g2, bim, fam = read_bed(str(tmp / "geno.chr1"))
    np.testing.assert_array_equal(g2, g)
    assert len(bim) == g.shape[1] and len(fam) == g.shape[0]


def test_generic_region_run_features_only(toy_inputs):
    tmp, g, pheno, covs = toy_inputs
    run = tmp / "run"
    main(["lgv", "init", "--run-dir", str(run), "--run-id", "toy-run", "--bslmm", "off",
          "--regions", str(tmp / "regions.tsv"), "--phenotypes", str(tmp / "pheno.parquet"),
          "--genotypes", str(tmp / "geno.chr{chrom}"), "--covariates", str(tmp / "covs.tsv"),
          "--numeric-covariates", "age", "--factor-covariates", "sex",
          "--cohort", "toy", "--region", "brain", "--min-cis-variants", "50"])
    main(["lgv", "run", "--run-dir", str(run), "--device", "cpu", "--batch-loci", "2"])
    main(["lgv", "combine", "--run-dir", str(run), "--no-score"])
    rows = pd.read_csv(run / "results" / "combined" / "observed-joint-features.tsv", sep="\t")
    status = dict(zip(rows["vmr_id"], rows["terminal_status"]))
    assert status == {"r1": "features_only", "r2": "features_only", "r3": "qc_failed"}
    recon = pd.read_csv(run / "results" / "combined" / "task-reconciliation.tsv", sep="\t")
    assert int(recon["unaccounted"].iloc[0]) == 0
    # The runner's EN must equal a direct cross-fit with the same seed and inputs.
    from genboostgpu.io.genotype import snp_qc_mask
    bim = read_bed(str(tmp / "geno.chr1"))[1]
    in_win = ((bim["pos"] >= 1) & (bim["pos"] <= 1_000_400)).to_numpy()  # r1 +- 500 kb
    gw = g[:, in_win]
    gw = gw[:, snp_qc_mask(gw)]
    cov = np.column_stack([covs["age"], (covs["sex"] == "M").astype(float)])
    seed = stable_seed("toy-run", "brain", "r1", "joint_features")
    direct = crossfit_elastic_net(gw, pheno["r1"].to_numpy(), cov, seed=seed + 17)["metrics"]
    r1 = rows[rows["vmr_id"] == "r1"].iloc[0]
    assert r1["r2_oof"] == pytest.approx(direct["r2_oof"], abs=1e-12)
    assert r1["rho2_oof"] == pytest.approx(direct["rho2_oof"], abs=1e-12)
    assert r1["rho2_oof"] > 0.05  # planted signal is recovered out of fold


def test_environment_error_stops_shard(toy_inputs, monkeypatch):
    # A broken runtime (here: a missing CUDA library) must not be written as
    # per-locus failure rows, which a resubmitted shard would skip as done.
    import genboostgpu.lgv.runner as runner

    def broken(*args, **kwargs):
        raise ImportError("libcublas.so.12: cannot open shared object file")

    tmp, _, _, _ = toy_inputs
    run = tmp / "run"
    main(["lgv", "init", "--run-dir", str(run), "--run-id", "toy-run", "--bslmm", "off",
          "--regions", str(tmp / "regions.tsv"), "--phenotypes", str(tmp / "pheno.parquet"),
          "--genotypes", str(tmp / "geno.chr{chrom}"), "--covariates", str(tmp / "covs.tsv"),
          "--numeric-covariates", "age", "--factor-covariates", "sex",
          "--cohort", "toy", "--region", "brain", "--min-cis-variants", "50"])
    monkeypatch.setattr(runner, "haseman_elston", broken)
    with pytest.raises(ImportError):
        main(["lgv", "run", "--run-dir", str(run), "--device", "cpu", "--batch-loci", "2"])
    assert not list((run / "task_rows").glob("*.parquet"))


def test_is_environment_error():
    from genboostgpu.backend import is_environment_error

    assert is_environment_error(ImportError("x"))
    assert is_environment_error(MemoryError())
    assert not is_environment_error(ValueError("singular design"))
    # CuPy raises a missing lazily-loaded CUDA library as a plain RuntimeError.
    assert is_environment_error(RuntimeError(
        "CuPy failed to load libnvrtc.so.12: OSError: libnvrtc.so.12: cannot open "
        "shared object file: No such file or directory"))
    try:
        try:
            raise OSError("libcublas.so.12: cannot open shared object file")
        except OSError as inner:
            raise ValueError("fit failed") from inner
    except ValueError as outer:
        assert is_environment_error(outer)
