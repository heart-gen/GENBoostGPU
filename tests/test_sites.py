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
