import numpy as np
import pytest

from genboostgpu.lgv.crossfit import CrossfitSettings, crossfit_elastic_net
from genboostgpu.lgv.geometry import adjacent_ld_metric, effective_rank_genotype
from genboostgpu.lgv.he import haseman_elston, he_from_residuals, he_prepare


def _locus(fx):
    g = np.loadtxt(fx("locus_g.tsv"))
    cov = np.loadtxt(fx("locus_cov.tsv"))
    y = np.loadtxt(fx("locus_y.txt"))
    ref = dict(line.rstrip("\n").split("\t") for line in open(fx("locus_features.tsv")))
    return g, cov, y, {k: float(v) for k, v in ref.items()}


def test_crossfit_matches_r_module02(fx):
    g, cov, y, ref = _locus(fx)
    settings = CrossfitSettings(max_features=150)
    out = crossfit_elastic_net(g, y, cov, settings, seed=1234567, keep_predictions=True)
    m = out["metrics"]
    for key in ("r2_oof", "rho2_oof", "covariance_ratio_oof", "score_variance_ratio_oof",
                "calibration_slope_oof", "mean_fold_score_variance_ratio",
                "mean_nonzero_snps"):
        assert m[key] == pytest.approx(ref[key], abs=1e-10), key
    oof = np.loadtxt(fx("locus_oof_prediction.txt"))
    np.testing.assert_allclose(out["predictions"]["oof_prediction"], oof, atol=1e-10)


def test_he_and_geometry_match_r(fx):
    g, cov, y, ref = _locus(fx)
    he = haseman_elston(g, y, cov)
    assert he["he_h2"] == pytest.approx(ref["he_h2"], abs=1e-12)
    assert he["he_se"] == pytest.approx(ref["he_se"], abs=1e-12)
    assert he["he_pvalue"] == pytest.approx(ref["he_pvalue"], abs=1e-12)
    assert effective_rank_genotype(g) == pytest.approx(ref["p_eff"], rel=1e-12)
    assert adjacent_ld_metric(g) == pytest.approx(ref["ld_metric"], abs=1e-14)


def test_he_quadratic_form_batches_columns(fx):
    g, cov, y, _ = _locus(fx)
    prep = he_prepare(g)
    rng = np.random.default_rng(1)
    r = rng.normal(size=(g.shape[0], 4))
    r = (r - r.mean(0)) / r.std(0, ddof=1)
    slope, se, _, _ = he_from_residuals(prep, r)
    for k in range(4):
        s1, se1, _, _ = he_from_residuals(prep, r[:, k])
        assert slope[k] == pytest.approx(s1, rel=1e-12)
        assert se[k] == pytest.approx(se1, rel=1e-12)


@pytest.mark.gpu
def test_crossfit_gpu_matches_cpu(fx):
    g, cov, y, _ = _locus(fx)
    settings = CrossfitSettings(max_features=150)
    c = crossfit_elastic_net(g, y, cov, settings, seed=1234567)["metrics"]
    gg = crossfit_elastic_net(g, y, cov, settings, seed=1234567, device="gpu")["metrics"]
    for key in ("r2_oof", "rho2_oof"):
        assert gg[key] == pytest.approx(c[key], abs=1e-8)


def test_bslmm_blas_coretype(monkeypatch):
    import genboostgpu.lgv.bslmm as bslmm

    monkeypatch.setenv("OPENBLAS_CORETYPE", "Prescott")
    env = bslmm.BslmmSettings().gemma_env()
    assert env["OPENBLAS_CORETYPE"] == "SkylakeX" and env["OPENBLAS_NUM_THREADS"] == "1"
    assert "OPENBLAS_CORETYPE" not in bslmm.BslmmSettings(blas_coretype="auto").gemma_env()
    # Old run.json files have no blas_coretype and get the pinned default.
    old = {k: v for k, v in bslmm.BslmmSettings().as_dict().items() if k != "blas_coretype"}
    assert bslmm.BslmmSettings(**old).blas_coretype == "SkylakeX"

    monkeypatch.setattr(bslmm, "_cpu_flags", lambda: frozenset({"avx", "avx2", "fma"}))
    with pytest.raises(RuntimeError, match="avx512"):
        bslmm.BslmmSettings().check_cpu()
    bslmm.BslmmSettings(blas_coretype="Haswell").check_cpu()
    bslmm.BslmmSettings(blas_coretype="auto").check_cpu()
