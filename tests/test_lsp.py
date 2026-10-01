import numpy as np
import pytest

from genboostgpu.lgv.rrng import RRNG, r_permutations, r_sample_int
from genboostgpu.lsp.oof import (LspSettings, donor_folds, he_permutation_screen,
                                 locus_metrics, prep_genotypes)
from genboostgpu.lsp.seeds import seed_for, xxhash32


def test_xxhash32_matches_r_digest():
    # digest::digest(x, "xxhash32", serialize = FALSE)
    assert f"{xxhash32(b'abc'):08x}" == "32d153ff"
    assert f"{xxhash32(b'a much longer key with more than sixteen bytes|x|y'):08x}" == "fdecc370"


def test_seed_for_matches_r():
    assert seed_for("lsp-AA-caudate-20260925-a", region="caudate", repeat_i=1) == 226114056
    assert seed_for("lsp-AA-caudate-20260925-a", region="caudate",
                    task="chr1:903932-904084", repeat_i=2, fold=3) == 154512144


def test_permutations_equal_sequential_sample_int():
    a = r_permutations(RRNG(77), 23, 5)
    rng = RRNG(77)
    b = np.stack([r_sample_int(rng, 23) for _ in range(5)])
    np.testing.assert_array_equal(a, b)


def test_permutations_leave_stream_in_step():
    r1 = RRNG(5)
    r_permutations(r1, 40, 7)
    r2 = RRNG(5)
    for _ in range(7):
        r_sample_int(r2, 40)
    assert r1.unif_rand() == r2.unif_rand()


def test_donor_folds_partition():
    donors = [f"D{i}::x" for i in range(103)]
    f = donor_folds(donors, "run", "caudate", LspSettings(repeats=3))
    assert len(f) == 309
    for _, g in f.groupby("repeat_i"):
        assert sorted(g["donor"]) == sorted(donors)
        assert set(g["outer_fold"]) == {1, 2, 3, 4, 5}


def test_prep_genotypes_fits_on_training_only():
    rng = np.random.default_rng(0)
    g_tr = rng.binomial(2, 0.3, size=(40, 6)).astype(float)
    g_te = rng.binomial(2, 0.3, size=(10, 6)).astype(float)
    g_tr[0, 0] = np.nan
    g_te[:, 1] = np.nan                      # test missingness must not matter
    g_tr[:, 2] = 0.0                         # monomorphic in training -> dropped
    out = prep_genotypes(g_tr, g_te)
    assert out["n_variants"] == 5
    assert not np.isnan(out["test"]).any()
    np.testing.assert_allclose(out["train"].mean(axis=0), 0, atol=1e-12)


def test_permutation_screen_detects_signal():
    rng = np.random.default_rng(3)
    g = rng.binomial(2, 0.3, size=(80, 60)).astype(float)
    z = (g - g.mean(0)) / g.std(0, ddof=1)
    strong = he_permutation_screen(z, z[:, :20].sum(1) + rng.normal(size=80), 200, 0.05, 11)
    null = he_permutation_screen(z, rng.normal(size=80), 200, 0.05, 11)
    assert strong["pass_"] and strong["p"] <= 0.05
    assert null["n_perm"] == 200 and 0 < null["p"] <= 1


def test_locus_metrics_null_predictions():
    import pandas as pd

    preds = pd.DataFrame(dict(y_obs=[1.0, -1.0, 0.5, -0.5], y_pred=0.0, donor=list("abcd"),
                              screened_in=False, n_variants=0, alpha=np.nan))
    m = locus_metrics(preds)
    # Null predictions on mean-zero outcomes: SSE == SST, so r2_pred_oof is 0.
    assert m["r2_pred_oof"] == pytest.approx(0.0, abs=1e-12)
    assert not m["calibration_slope_defined"] and np.isnan(m["cor2_oof"])
    assert m["screening_pass_frequency"] == 0.0
