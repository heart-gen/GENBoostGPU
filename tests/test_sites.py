"""Site-level runner: window blocks + batched HE equal single-locus features."""
import numpy as np
import pandas as pd
import pytest

from genboostgpu.cli import main
from genboostgpu.io.genotype import GenotypeSource, snp_qc_mask, write_bed
from genboostgpu.lgv.geometry import effective_rank_genotype
from genboostgpu.lgv.he import haseman_elston

h5py = pytest.importorskip("h5py")


@pytest.fixture
def toy_sites(tmp_path):
    rng = np.random.default_rng(4)
    n, m, k = 50, 300, 12
    ids = [f"S{i:03d}" for i in range(n)]
    pos = np.sort(rng.choice(np.arange(1, 1_500_000), m, replace=False))
    g = rng.binomial(2, rng.uniform(0.05, 0.5, m), size=(n, m)).astype(float)
    write_bed(str(tmp_path / "g.chr7"), g,
              pd.DataFrame(dict(chrom="7", snp=[f"v{j}" for j in range(m)], cm=0, pos=pos,
                                a1="A", a2="C")),
              pd.DataFrame(dict(FID=ids, IID=ids, father=0, mother=0, sex=0, pheno=-9)))
    starts = np.sort(rng.choice(np.arange(600_000, 700_000), k, replace=False))
    units = pd.DataFrame(dict(unit_id=[f"chr7:{s}-{s}" for s in starts], chrom="chr7",
                              start=starts, end=starts, n_sites=1, var=0.01))
    (tmp_path / "units").mkdir()
    units.to_csv(tmp_path / "units" / "chr7.units.tsv.gz", sep="\t", index=False)
    beta = rng.uniform(0.2, 0.8, size=(n, k))
    beta[3, 5] = np.nan                       # one unit with a missing sample
    with h5py.File(tmp_path / "units" / "chr7.beta.h5", "w") as f:
        f["beta"] = beta                       # samples x units (h5py view)
        f["sample_id"] = np.array(ids, dtype="S")
        f["unit_id"] = np.array(units["unit_id"], dtype="S")
    pd.DataFrame(dict(sample_id=ids, age=rng.uniform(20, 80, n))).to_csv(
        tmp_path / "covs.tsv", sep="\t", index=False)
    return tmp_path, beta, units


def test_sites_blocks_match_single_locus(toy_sites):
    tmp, beta, units = toy_sites
    run = tmp / "run"
    main(["sites", "init", "--run-dir", str(run), "--run-id", "toy-sites",
          "--units-dir", str(tmp / "units"), "--genotypes", str(tmp / "g.chr{chrom}"),
          "--covariates", str(tmp / "covs.tsv"), "--numeric-covariates", "age",
          "--cohort", "toy", "--region", "x", "--features", "geometry,he",
          "--window-block-bp", "50000", "--min-cis-variants", "30",
          "--gemma-blas-coretype", "haswell"])
    import json
    assert json.load(open(run / "run.json"))["bslmm"]["blas_coretype"] == "Haswell"
    main(["sites", "run", "--run-dir", str(run), "--device", "cpu"])
    main(["lgv", "combine", "--run-dir", str(run), "--no-score"])
    rows = pd.read_csv(run / "results" / "combined" / "observed-joint-features.tsv", sep="\t")
    assert (rows["terminal_status"] == "features_only").all()
    assert rows["window_block_id"].nunique() <= 3
    covs = pd.read_csv(tmp / "covs.tsv", sep="\t")
    gs = GenotypeSource(str(tmp / "g.chr{chrom}"))
    for j in (0, 5, 11):
        r = rows.iloc[j]
        g, _ = gs.window("7", int(r["window_start"]), int(r["window_end"]))
        y = beta[:, j]
        ok = np.isfinite(y)
        G = g[ok]
        G = G[:, snp_qc_mask(G)]
        he = haseman_elston(G, y[ok], covs.loc[ok, ["age"]].to_numpy())
        assert r["n"] == ok.sum()
        assert r["he_h2"] == pytest.approx(he["he_h2"], abs=1e-12)
        assert r["p_eff"] == pytest.approx(effective_rank_genotype(G), rel=1e-12)


def test_genome_wide_fileset_matches_per_chromosome(tmp_path):
    rng = np.random.default_rng(9)
    n = 20
    ids = [f"S{i:03d}" for i in range(n)]
    fam = pd.DataFrame(dict(FID=ids, IID=ids, father=0, mother=0, sex=0, pheno=-9))
    parts = []
    for chrom, m in (("1", 40), ("2", 25), ("3", 30)):
        g = rng.binomial(2, 0.3, size=(n, m)).astype(float)
        g[rng.random(g.shape) < 0.05] = np.nan
        bim = pd.DataFrame(dict(chrom=chrom, snp=[f"c{chrom}_{j}" for j in range(m)], cm=0,
                                pos=np.sort(rng.choice(10_000, m, replace=False)) + 1,
                                a1="A", a2="G"))
        write_bed(str(tmp_path / f"per.chr{chrom}"), g, bim, fam)
        parts.append((g, bim))
    write_bed(str(tmp_path / "all"), np.hstack([p[0] for p in parts]),
              pd.concat([p[1] for p in parts], ignore_index=True), fam)
    per = GenotypeSource(str(tmp_path / "per.chr{chrom}"))
    gw = GenotypeSource(str(tmp_path / "all"))
    assert gw.genome_wide and not per.genome_wide
    for chrom in ("2", "chr1", "3"):
        a, va = per.window(chrom, 2_000, 8_000)
        b, vb = gw.window(chrom, 2_000, 8_000)
        np.testing.assert_array_equal(a, b)
        assert list(va["snp"]) == list(vb["snp"])
        assert va["snp"].str.startswith(f"c{chrom.removeprefix('chr')}_").all()
    assert list(gw.samples("2")["IID"]) == ids


def test_sites_en_batches_in_flight_match_one_batch(toy_sites):
    """Solving and finishing batches on background threads (one unit per
    batch, several finisher threads) gives the rows of one serial batch."""
    tmp, _, _ = toy_sites
    out = {}
    for tag, extra in (("one", ["--batch-units", "64"]),
                       ("many", ["--batch-units", "1", "--cpu-threads", "3"])):
        run = tmp / f"run_{tag}"
        main(["sites", "init", "--run-dir", str(run), "--run-id", "toy-sites-en",
              "--units-dir", str(tmp / "units"), "--genotypes", str(tmp / "g.chr{chrom}"),
              "--covariates", str(tmp / "covs.tsv"), "--numeric-covariates", "age",
              "--cohort", "toy", "--region", "x", "--features", "geometry,he,en",
              "--window-block-bp", "50000", "--min-cis-variants", "30"])
        main(["sites", "run", "--run-dir", str(run), "--device", "cpu"] + extra)
        main(["lgv", "combine", "--run-dir", str(run), "--no-score"])
        out[tag] = pd.read_csv(run / "results" / "combined" / "observed-joint-features.tsv",
                               sep="\t").set_index("task_id").sort_index()
    cols = ["terminal_status", "he_h2", "rho2_oof", "r2_oof", "covariance_ratio_oof",
            "score_variance_ratio_oof"]
    assert out["one"]["rho2_oof"].notna().all()
    pd.testing.assert_frame_equal(out["one"][cols], out["many"][cols], check_exact=True)


def test_read_pvar_types(tmp_path):
    from genboostgpu.io.genotype import read_pvar

    (tmp_path / "g.pvar").write_text(
        "##fileformat=PVARv1.0\n##source=test\n#CHROM\tPOS\tID\tREF\tALT\n"
        "chr1\t833068\t00123\tG\tA\n1\t1057648\trs1\tT\t.\n")
    v = read_pvar(str(tmp_path / "g"))
    assert list(v.columns) == ["chrom", "pos", "snp", "ref", "alt"]
    assert v["pos"].dtype == np.int64 and v["pos"].tolist() == [833068, 1057648]
    assert v["snp"].tolist() == ["00123", "rs1"]       # IDs stay strings
    assert v["chrom"].tolist() == ["chr1", "1"]


def _init_sites(run, base, *extra):
    main(["sites", "init", "--run-dir", str(run), "--run-id", "toy-sites-bslmm",
          "--units-dir", str(base / "units"), "--genotypes", str(base / "g.chr{chrom}"),
          "--covariates", str(base / "covs.tsv"), "--numeric-covariates", "age",
          "--cohort", "toy", "--region", "x", "--window-block-bp", "50000",
          "--min-cis-variants", "30", "--gemma-blas-coretype", "auto", *extra])


def _short_chains(run, gemma_bin=None):
    import json

    cfg = json.load(open(run / "run.json"))
    cfg["bslmm"].update(burn_in=200, sampling=2000)
    if gemma_bin is not None:
        cfg["bslmm"]["gemma_bin"] = gemma_bin
    json.dump(cfg, open(run / "run.json", "w"), indent=2)


def _other_cluster(tmp):
    """A second copy of the inputs at different paths ("the other cluster")."""
    import shutil

    other = tmp / "other_cluster"
    shutil.copytree(tmp / "units", other / "units")
    for f in tmp.glob("g.chr7.*"):
        shutil.copy(f, other / f.name)
    shutil.copy(tmp / "covs.tsv", other / "covs.tsv")
    return other


def _copy_bslmm_rows(src_run, dst_run):
    import shutil

    for f in (src_run / "bslmm_rows").glob("part-*.parquet"):
        shutil.copy(f, dst_run / "bslmm_rows" / f.name)


def _combined(run):
    return pd.read_csv(run / "results" / "combined" / "observed-joint-features.tsv",
                       sep="\t").set_index("task_id").sort_index()


def test_sites_separate_bslmm_matches_inline(toy_sites):
    """GEMMA in a CPU job run from another copy of the inputs, joined at
    combine, gives the rows of inline GEMMA."""
    import os

    from genboostgpu.lgv.bslmm import BslmmSettings

    if not os.access(BslmmSettings().gemma_bin, os.X_OK):
        pytest.skip("GEMMA binary not available")
    tmp, _, _ = toy_sites
    inline = tmp / "run_inline"
    _init_sites(inline, tmp, "--bslmm", "inline")
    _short_chains(inline)
    main(["sites", "run", "--run-dir", str(inline), "--device", "cpu",
          "--gemma-workers", "4"])
    main(["lgv", "combine", "--run-dir", str(inline), "--no-score"])

    gpu = tmp / "run_gpu"
    _init_sites(gpu, tmp, "--bslmm", "separate")
    _short_chains(gpu)
    main(["sites", "run", "--run-dir", str(gpu), "--device", "cpu"])
    other = _other_cluster(tmp)
    cpu = other / "run_cpu"
    _init_sites(cpu, other, "--bslmm", "separate")
    _short_chains(cpu)
    main(["sites", "bslmm", "--run-dir", str(cpu), "--shard", "0/2", "--gemma-workers", "2"])
    main(["sites", "bslmm", "--run-dir", str(cpu), "--shard", "1/2", "--gemma-workers", "2"])
    _copy_bslmm_rows(cpu, gpu)
    main(["lgv", "combine", "--run-dir", str(gpu), "--no-score"])

    a, b = _combined(inline), _combined(gpu)
    assert (a["terminal_status"] == "completed").sum() >= 10
    assert a["bslmm_pve"].notna().sum() >= 10
    assert list(a.columns) == list(b.columns)
    cols = [c for c in a.columns if c not in ("bslmm_elapsed_sec", "plink_source",
                                               "phenotype_source")]
    pd.testing.assert_frame_equal(a[cols], b[cols], check_exact=True)


def test_sites_separate_bslmm_join_checks(toy_sites):
    """Combine refuses GEMMA rows from a differently initialized run or from
    different inputs, and flags pending tasks that have no GEMMA row."""
    tmp, _, _ = toy_sites
    gpu = tmp / "run_gpu"
    _init_sites(gpu, tmp, "--bslmm", "separate", "--features", "geometry,he")
    _short_chains(gpu)        # the binary's path is not part of the run key
    main(["sites", "run", "--run-dir", str(gpu), "--device", "cpu"])

    # No GEMMA rows: features-only tiers keep their status; nothing is joined.
    main(["lgv", "combine", "--run-dir", str(gpu), "--no-score"])
    assert (_combined(gpu)["terminal_status"] == "features_only").all()

    other = _other_cluster(tmp)
    # GEMMA chains fail fast without a binary; the rows still carry the checks.
    differs = other / "run_maf"
    _init_sites(differs, other, "--bslmm", "separate", "--features", "geometry,he",
                "--maf-min", "0.1")
    _short_chains(differs, gemma_bin=str(tmp / "no-gemma"))
    main(["sites", "bslmm", "--run-dir", str(differs), "--gemma-workers", "1"])
    _copy_bslmm_rows(differs, gpu)
    with pytest.raises(RuntimeError, match="initialized differently"):
        main(["lgv", "combine", "--run-dir", str(gpu), "--no-score"])
    for f in (gpu / "bslmm_rows").glob("part-*.parquet"):
        f.unlink()

    with h5py.File(other / "units" / "chr7.beta.h5", "r+") as f:
        f["beta"][7, 2] += 0.01                # one donor's value of unit 3
    same = other / "run_same"
    _init_sites(same, other, "--bslmm", "separate", "--features", "geometry,he")
    _short_chains(same, gemma_bin=str(tmp / "no-gemma"))
    main(["sites", "bslmm", "--run-dir", str(same), "--gemma-workers", "1"])
    _copy_bslmm_rows(same, gpu)
    with pytest.raises(RuntimeError, match=r"different inputs .*\[3\]"):
        main(["lgv", "combine", "--run-dir", str(gpu), "--no-score"])


def test_sites_separate_missing_bslmm_rows(toy_sites):
    tmp, _, _ = toy_sites
    run = tmp / "run_sep"
    _init_sites(run, tmp, "--bslmm", "separate", "--features", "geometry,he,en")
    main(["sites", "run", "--run-dir", str(run), "--device", "cpu"])
    parts = pd.concat(pd.read_parquet(p) for p in (run / "task_rows").glob("*.parquet"))
    assert (parts["terminal_status"] == "pending").sum() >= 10
    assert parts.loc[parts["terminal_status"] == "pending", "input_digest"].notna().all()
    main(["lgv", "combine", "--run-dir", str(run), "--no-score"])
    rows = _combined(run)
    assert "input_digest" not in rows.columns
    failed = rows["terminal_status"] == "computational_failure"
    assert failed.sum() >= 10
    assert rows.loc[failed, "feature_error"].str.contains("no BSLMM row").all()
    import json

    man = json.load(open(run / "results" / "run-manifest.json"))
    assert man["bslmm_rows_missing"] == failed.sum()
