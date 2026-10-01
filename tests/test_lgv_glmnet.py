import numpy as np
import pytest

from genboostgpu.lgv.glmnet import GlmnetError, cv_glmnet_gaussian, glmnet_gaussian


def _load(fx, tag):
    x = np.loadtxt(fx(f"glmnet_{tag}_x.tsv"))
    y = np.loadtxt(fx(f"glmnet_{tag}_y.txt"))
    foldid = np.loadtxt(fx(f"glmnet_{tag}_foldid.txt")).astype(int)
    return x, y, foldid


@pytest.mark.parametrize("tag", ["cov", "naive"])
@pytest.mark.parametrize("alpha", ["0.1", "1"])
def test_path_matches_r(fx, tag, alpha):
    x, y, _ = _load(fx, tag)
    ref = np.loadtxt(fx(f"glmnet_{tag}_a{alpha}_path.tsv"))
    npasses = int(np.loadtxt(fx(f"glmnet_{tag}_a{alpha}_npasses.txt")))
    fit = glmnet_gaussian(x, y, alpha=float(alpha))
    assert fit.lambda_.size == ref.shape[0]
    assert fit.npasses == npasses
    # Reductions follow Eigen's summation order, so the path is bitwise R's.
    np.testing.assert_array_equal(fit.lambda_, ref[:, 0])
    np.testing.assert_array_equal(fit.a0, ref[:, 1])
    np.testing.assert_array_equal(fit.dev_ratio, ref[:, 2])
    np.testing.assert_array_equal(fit.beta.T, ref[:, 3:])


@pytest.mark.parametrize("tag", ["cov", "naive"])
@pytest.mark.parametrize("alpha", ["0.1", "1"])
def test_cv_matches_r(fx, tag, alpha):
    x, y, foldid = _load(fx, tag)
    ref = np.loadtxt(fx(f"glmnet_{tag}_a{alpha}_cv.tsv"))
    sel = np.loadtxt(fx(f"glmnet_{tag}_a{alpha}_cvsel.txt"))
    cv = cv_glmnet_gaussian(x, y, foldid, alpha=float(alpha))
    np.testing.assert_allclose(cv.lambda_, ref[:, 0], rtol=1e-12)
    np.testing.assert_allclose(cv.cvm, ref[:, 1], atol=1e-12)
    np.testing.assert_allclose(cv.cvsd, ref[:, 2], atol=1e-12)
    assert cv.lambda_min == sel[0]
    assert cv.lambda_1se == sel[1]


def test_r_errors_are_reproduced():
    x = np.random.default_rng(0).normal(size=(20, 1))
    with pytest.raises(GlmnetError, match="2 or more columns"):
        glmnet_gaussian(x, np.arange(20.0))
    with pytest.raises(GlmnetError, match="y is constant"):
        glmnet_gaussian(np.random.default_rng(0).normal(size=(20, 3)), np.ones(20))


def test_threaded_cpu_matches_serial(fx):
    from genboostgpu.lgv.glmnet import GlmnetProblem, fit_paths

    x, y, _ = _load(fx, "cov")
    probs = [GlmnetProblem(x, y, a) for a in (0.1, 0.5, 1.0)]
    serial = fit_paths(probs, threads=1)
    threaded = fit_paths(probs, threads=3)
    for s, t in zip(serial, threaded):
        np.testing.assert_array_equal(s.beta, t.beta)


@pytest.mark.gpu
@pytest.mark.parametrize("tag", ["cov", "naive"])
def test_gpu_matches_cpu(fx, tag):
    from genboostgpu.lgv.glmnet import GlmnetProblem, fit_paths

    x, y, foldid = _load(fx, tag)
    probs = [GlmnetProblem(x, y, a) for a in (0.1, 0.5, 1.0)]
    probs += [GlmnetProblem(x[foldid != k], y[foldid != k], 0.5) for k in range(1, 6)]
    cpu = fit_paths(probs, device="cpu")
    gpu = fit_paths(probs, device="gpu")
    for c, g in zip(cpu, gpu):
        assert c.lambda_.size == g.lambda_.size
        np.testing.assert_allclose(g.beta, c.beta, atol=1e-9)
        np.testing.assert_allclose(g.a0, c.a0, atol=1e-9)
