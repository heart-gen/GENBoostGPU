"""R-equivalent ``glmnet()`` / ``cv.glmnet()`` for the Gaussian family.

The numerical core is :func:`genboostgpu.lgv.glmnet_core.elnet_core`, a port
of glmnet 4.1-10. This module reproduces the R wrappers around it:

* ``glmnet()`` defaults: ``nlambda = 100``,
  ``lambda.min.ratio = ifelse(nobs < nvars, 0.01, 1e-4)``, ``thresh = 1e-7``,
  ``maxit = 1e5``, ``dfmax = nvars + 1``, ``pmax = min(2 * dfmax + 20, nvars)``,
  ``type.gaussian = ifelse(nvars < 500, "covariance", "naive")``, and
  ``fix.lam()`` on the first lambda;
* ``cv.glmnet()`` with a supplied ``foldid``: full-data path, per-fold refits
  on that lambda sequence, grouped ``cvm``/``cvsd`` weighted by fold size, and
  ``lambda.min``/``lambda.1se`` exactly as ``getOptcv.glmnet``.

Problems can be solved one at a time on the CPU (``numba.njit``) or many at
once on the GPU (one CUDA thread per problem) through :func:`fit_paths`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import cached_property

import numpy as np

from . import glmnet_core

__all__ = [
    "GlmnetError",
    "GlmnetProblem",
    "GlmnetFit",
    "CvGlmnetFit",
    "glmnet_gaussian",
    "cv_glmnet_gaussian",
    "fit_paths",
]

# glmnet.control() defaults.
_EPS = 1e-6
_BIG = 9.9e35
_MNLAM = 5
_FDEV = 1e-5
_DEVMAX = 0.999


class GlmnetError(RuntimeError):
    """A condition under which R's glmnet() would stop()."""


@dataclass
class GlmnetProblem:
    """One ``glmnet(x, y, alpha = alpha, lambda = lambda_)`` call."""

    x: np.ndarray
    y: np.ndarray
    alpha: float
    lambda_: np.ndarray | None = None
    nlambda: int = 100
    lambda_min_ratio: float | None = None
    standardize: bool = True
    intercept: bool = True
    thresh: float = 1e-7
    maxit: int = 100000
    type_gaussian: str | None = None

    def resolved(self):
        x = np.asarray(self.x, dtype=np.float64)
        y = np.asarray(self.y, dtype=np.float64).ravel()
        if x.ndim != 2:
            raise GlmnetError("x must be a matrix")
        n, p = x.shape
        if p <= 1:
            raise GlmnetError("x should be a matrix with 2 or more columns")
        if y.size != n:
            raise GlmnetError("number of observations in y not equal to rows of x")
        if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
            raise GlmnetError("x or y has missing or infinite values")
        ybar = y.mean() if self.intercept else 0.0
        if np.sum((y - ybar) ** 2) == 0:
            raise GlmnetError(
                "y is constant; gaussian glmnet fails at standardization step"
            )
        tg = self.type_gaussian or ("covariance" if p < 500 else "naive")
        if self.lambda_ is None:
            lmr = self.lambda_min_ratio
            if lmr is None:
                lmr = 1e-2 if n < p else 1e-4
            if lmr >= 1:
                raise GlmnetError("lambda.min.ratio should be less than 1")
            flmin = float(lmr)
            ulam = np.zeros(1)
            nlam = int(self.nlambda)
        else:
            lam = np.asarray(self.lambda_, dtype=np.float64).ravel()
            if np.any(lam < 0):
                raise GlmnetError("lambdas should be non-negative")
            flmin = 1.0
            ulam = np.sort(lam)[::-1].copy()
            nlam = ulam.size
        ne = p + 1
        nx = min(ne * 2 + 20, p)
        return dict(x=x, y=y, n=n, p=p, naive=int(tg == "naive"), flmin=flmin,
                    ulam=ulam, nlam=nlam, ne=ne, nx=nx,
                    user_lambda=self.lambda_ is not None)


@dataclass
class GlmnetFit:
    """The parts of an R ``elnet`` object the pipeline uses."""

    a0: np.ndarray
    beta: np.ndarray          # p x L (dense)
    lambda_: np.ndarray
    dev_ratio: np.ndarray
    df: np.ndarray
    nulldev: float
    npasses: int
    jerr: int
    nobs: int

    def predict(self, newx, index=None):
        """``predict(fit, newx)`` at every lambda (or at column(s) ``index``)."""
        newx = np.asarray(newx, dtype=np.float64)
        if index is None:
            return self.a0[None, :] + newx @ self.beta
        return self.a0[index] + newx @ self.beta[:, index]

    def coef(self, index):
        return self.a0[index], self.beta[:, index]

    @cached_property
    def _active_rows(self):
        # SNPs nonzero anywhere on the path; every other row of beta is zero
        return np.flatnonzero(self.beta.any(axis=1))

    def coef_at(self, s):
        """``coef(fit, s = s)``: intercepts (len(s),) and betas (p, len(s)),
        linearly interpolated exactly as ``predict.glmnet``/``lambda.interp``."""
        s = np.atleast_1d(np.asarray(s, dtype=np.float64))
        left, right, frac = lambda_interp(self.lambda_, s)
        a0 = self.a0[left] * frac + self.a0[right] * (1.0 - frac)
        # Interpolate only the active rows: a zero row stays exactly zero, so
        # the result is identical to interpolating all p rows. Column-major,
        # like the fancy-indexed full interpolation: the BLAS product in
        # predict_at rounds differently for the two layouts.
        beta = np.zeros((self.beta.shape[0], s.size), order="F")
        rows = self._active_rows
        if rows.size:
            b = self.beta[rows]
            beta[rows] = b[:, left] * frac + b[:, right] * (1.0 - frac)
        return a0, beta

    def predict_at(self, newx, s):
        """``predict(fit, newx, s = s)``: n x len(s)."""
        a0, beta = self.coef_at(s)
        newx = np.asarray(newx, dtype=np.float64)
        return a0[None, :] + newx @ beta


def _r_approx_linear(x, y, v):
    """``approx(x, y, v)$y`` for one point (R's approx1, method "linear")."""
    n = x.size
    i, j = 0, n - 1
    if v < x[i] or v > x[j]:
        return np.nan
    while i < j - 1:
        ij = (i + j) // 2
        if v < x[ij]:
            j = ij
        else:
            i = ij
    if v == x[j]:
        return y[j]
    if v == x[i]:
        return y[i]
    return y[i] + (y[j] - y[i]) * ((v - x[i]) / (x[j] - x[i]))


def _r_approx_seq(x, v):
    """Vectorized ``approx(x, seq_along(x), v)$y`` for increasing ``x``.

    Same arithmetic as R's approx1 bisection: exact hits return the index;
    otherwise ``y[i] + (y[j] - y[i]) * ((v - x[i]) / (x[j] - x[i]))`` with
    ``y = 1..k`` (so ``y[j] - y[i] = 1``).
    """
    x = np.asarray(x, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    k = x.size
    i = np.clip(np.searchsorted(x, v, side="right") - 1, 0, k - 1)
    j = np.minimum(i + 1, k - 1)
    yi = (i + 1).astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        interp = yi + 1.0 * ((v - x[i]) / (x[j] - x[i]))
    out = np.where(v == x[i], yi, np.where(v == x[j], j + 1.0, interp))
    return np.where((v < x[0]) | (v > x[-1]), np.nan, out)


def lambda_interp(lam, s):
    """R's ``lambda.interp(lambda, s)``; returns 0-based left/right and frac."""
    lam = np.asarray(lam, dtype=np.float64)
    s = np.asarray(s, dtype=np.float64)
    if lam.size == 1:
        z = np.zeros(s.size, dtype=np.int64)
        return z, z.copy(), np.ones(s.size)
    k = lam.size
    sfrac = (lam[0] - s) / (lam[0] - lam[k - 1])
    lnorm = (lam[0] - lam) / (lam[0] - lam[k - 1])
    lo, hi = lnorm.min(), lnorm.max()
    sfrac = np.where(sfrac < lo, lo, sfrac)
    sfrac = np.where(sfrac > hi, hi, sfrac)
    coord = _r_approx_seq(lnorm, sfrac)
    left = np.floor(coord).astype(np.int64)
    right = np.ceil(coord).astype(np.int64)
    with np.errstate(invalid="ignore", divide="ignore"):
        frac = (sfrac - lnorm[right - 1]) / (lnorm[left - 1] - lnorm[right - 1])
    frac = np.where(left == right, 1.0, frac)
    frac = np.where(np.abs(lnorm[left - 1] - lnorm[right - 1])
                    < np.finfo(np.float64).eps, 1.0, frac)
    return left - 1, right - 1, frac


@dataclass
class CvGlmnetFit:
    lambda_: np.ndarray
    cvm: np.ndarray
    cvsd: np.ndarray
    lambda_min: float
    lambda_1se: float
    index_min: int
    index_1se: int
    glmnet_fit: GlmnetFit
    fold_fits: list = field(default_factory=list)

    def index_for(self, rule: str) -> int:
        if rule == "lambda.min":
            return self.index_min
        if rule == "lambda.1se":
            return self.index_1se
        raise ValueError(f"Unknown lambda rule: {rule}")


# --------------------------------------------------------------------------
# CPU solver
# --------------------------------------------------------------------------

_cpu_core = None


def _get_cpu_core():
    global _cpu_core
    if _cpu_core is None:
        import numba

        _cpu_core = numba.njit(cache=False, nogil=True, fastmath=False)(
            glmnet_core.elnet_core)
    return _cpu_core


def _alloc(spec):
    n, p, nlam, nx = spec["n"], spec["p"], spec["nlam"], spec["nx"]
    naive = spec["naive"]
    return dict(
        a0=np.zeros(nlam), ca=np.zeros(nx * nlam), ia=np.zeros(nx, np.int64),
        kin=np.zeros(nlam, np.int64), rsqo=np.zeros(nlam), almo=np.zeros(nlam),
        ints_out=np.zeros(3, np.int64), dbl_out=np.zeros(2),
        xm=np.zeros(p), xs=np.zeros(p), xv=np.zeros(p), ju=np.zeros(p, np.int8),
        g=np.zeros(p), a=np.zeros(p), mm=np.zeros(p, np.int64),
        ix=np.zeros(p, np.int8), da=np.zeros(nx),
        cgram=np.zeros(1 if naive else p * nx), vq=np.zeros(p),
    )


def _finish(spec, prob, out) -> GlmnetFit:
    lmu, nlp, jerr = (int(v) for v in out["ints_out"])
    if jerr > 0:
        raise GlmnetError(f"glmnet fatal error code {jerr}")
    p, nx = spec["p"], spec["nx"]
    y = spec["y_orig"]
    ybar = y.mean() if prob.intercept else 0.0
    nulldev = float(np.sum((y - ybar) ** 2) / y.size)
    if lmu < 1:
        raise GlmnetError("an empty model has been returned; probably a convergence issue")
    beta = np.zeros((p, lmu))
    ia = out["ia"]
    for m in range(lmu):
        nk = int(out["kin"][m])
        col = out["ca"][m * nx:m * nx + nk]
        beta[ia[:nk], m] = col
    lam = out["almo"][:lmu].copy()
    if not spec["user_lambda"] and lam.size > 2:
        llam = np.log(lam)
        lam[0] = math.exp(2 * llam[1] - llam[2])
    df = np.count_nonzero(beta, axis=0)
    return GlmnetFit(a0=out["a0"][:lmu].copy(), beta=beta, lambda_=lam,
                     dev_ratio=out["rsqo"][:lmu].copy(), df=df, nulldev=nulldev,
                     npasses=nlp, jerr=jerr, nobs=spec["n"])


def _solve_cpu(prob: GlmnetProblem) -> GlmnetFit:
    spec = prob.resolved()
    core = _get_cpu_core()
    out = _alloc(spec)
    x = np.asfortranarray(spec["x"]).ravel(order="F").copy()
    y = spec["y"].copy()
    spec["y_orig"] = spec["y"]
    vp = np.ones(spec["p"])
    core(x, spec["n"], spec["p"], y,
         float(prob.alpha), spec["flmin"], spec["ulam"], spec["nlam"],
         int(prob.standardize), int(prob.intercept), float(prob.thresh),
         int(prob.maxit), spec["naive"], spec["ne"], spec["nx"], vp,
         _EPS, _BIG, _MNLAM, _FDEV, _DEVMAX,
         out["a0"], out["ca"], out["ia"], out["kin"], out["rsqo"], out["almo"],
         out["ints_out"], out["dbl_out"],
         out["xm"], out["xs"], out["xv"], out["ju"], out["g"], out["a"],
         out["mm"], out["ix"], out["da"], out["cgram"], out["vq"])
    return _finish(spec, prob, out)


def _solve_cpu_safe(prob):
    try:
        return _solve_cpu(prob)
    except GlmnetError as err:
        return err


def fit_paths(problems, device: str = "cpu", threads: int = 1):
    """Solve many glmnet problems. Returns a list of ``GlmnetFit`` or
    ``GlmnetError`` (one per problem, in order).

    ``device="gpu"`` solves the whole list in one CUDA launch. On the CPU the
    numba core releases the GIL, so ``threads > 1`` solves problems in
    parallel threads.
    """
    if device == "gpu":
        from .glmnet_gpu import solve_batch_gpu

        return solve_batch_gpu(problems)
    if threads > 1 and len(problems) > 1:
        from concurrent.futures import ThreadPoolExecutor

        _get_cpu_core()
        with ThreadPoolExecutor(max_workers=threads) as pool:
            return list(pool.map(_solve_cpu_safe, problems))
    return [_solve_cpu_safe(prob) for prob in problems]


def glmnet_gaussian(x, y, alpha=1.0, lambda_=None, device="cpu", **kwargs) -> GlmnetFit:
    """``glmnet(x, y, family = "gaussian", alpha = alpha, lambda = lambda_)``."""
    res = fit_paths([GlmnetProblem(x, y, alpha, lambda_, **kwargs)], device=device)[0]
    if isinstance(res, Exception):
        raise res
    return res


# --------------------------------------------------------------------------
# cv.glmnet
# --------------------------------------------------------------------------

def cv_problems(x, y, foldid, alpha, fold_cache=None, **kwargs):
    """All ``glmnet()`` calls one ``cv.glmnet(..., foldid)`` makes.

    The full-data fit comes first, then one fit per fold. As in glmnet 4.1-10
    (``cv.glmnet.raw``), each fold fit computes its *own* default lambda path:
    the fold calls receive cv.glmnet's ``lambda`` argument (NULL), because
    ``lambda <- glmnet.object$lambda`` is only assigned after the fold loop.
    Fold predictions at the full-data lambdas are then interpolated with
    ``lambda.interp``.

    ``fold_cache`` (a dict) lets calls for several alphas on the same ``x`` and
    ``foldid`` share the fold matrices, so the GPU uploads each one once.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).ravel()
    foldid = np.asarray(foldid)
    nfolds = int(foldid.max())
    if nfolds < 3:
        raise GlmnetError("nfolds must be bigger than 3; nfolds=10 recommended")
    probs = [GlmnetProblem(x, y, alpha, None, **kwargs)]
    for i in range(1, nfolds + 1):
        keep = foldid != i
        if fold_cache is None:
            xk = x[keep]
        else:
            xk = fold_cache.get(i)
            if xk is None:
                xk = fold_cache[i] = x[keep]
        probs.append(GlmnetProblem(xk, y[keep], alpha, None, **kwargs))
    return probs


def cv_summarize(x, y, foldid, full_fit: GlmnetFit, fold_fits) -> CvGlmnetFit:
    """Assemble ``cv.glmnet`` statistics from the full fit and fold fits."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).ravel()
    foldid = np.asarray(foldid)
    nfolds = int(foldid.max())
    lam = full_fit.lambda_
    nlambda = lam.size
    predmat = np.full((y.size, nlambda), np.nan)
    for i in range(1, nfolds + 1):
        fit_i = fold_fits[i - 1]
        if isinstance(fit_i, Exception):
            raise fit_i
        which = foldid == i
        preds = fit_i.predict_at(x[which], lam)
        nlami = min(preds.shape[1], nlambda)
        predmat[np.ix_(which, np.arange(nlami))] = preds[:, :nlami]
        if nlami < nlambda:
            predmat[np.ix_(which, np.arange(nlami - 1, nlambda))] = \
                preds[:, [nlami - 1]]
    cvraw = (y[:, None] - predmat) ** 2
    cvraw[np.isinf(cvraw)] = np.nan
    outmat = np.empty((nfolds, nlambda))
    wisum = np.empty(nfolds)
    for i in range(1, nfolds + 1):
        mati = cvraw[foldid == i]
        wisum[i - 1] = mati.shape[0]
        with np.errstate(invalid="ignore"):
            outmat[i - 1] = np.nanmean(mati, axis=0)
    good_n = float(nfolds)
    cvm = _wmean_cols(outmat, wisum)
    finite = np.isfinite(cvm)
    with np.errstate(invalid="ignore"):
        cvsd = np.sqrt(_wmean_cols((outmat - np.where(finite, cvm, 0.0)) ** 2, wisum)
                       / (good_n - 1))
    cvsd[~finite] = np.nan
    keep = ~np.isnan(cvsd)
    lam_k, cvm_k, cvsd_k = lam[keep], cvm[keep], cvsd[keep]
    if lam_k.size == 0:
        raise GlmnetError("cv.glmnet produced no finite cross-validation error")
    cvmin = np.nanmin(cvm_k)
    lambda_min = float(np.max(lam_k[cvm_k <= cvmin]))
    idmin = int(np.flatnonzero(lam_k == lambda_min)[0])
    semin = cvm_k[idmin] + cvsd_k[idmin]
    lambda_1se = float(np.max(lam_k[cvm_k <= semin]))
    id1se = int(np.flatnonzero(lam_k == lambda_1se)[0])
    # Indices into the full path (cv.glmnet drops NA-cvsd lambdas first;
    # the selected lambda is always on the full path).
    full_idx = np.flatnonzero(keep)
    return CvGlmnetFit(lambda_=lam_k, cvm=cvm_k, cvsd=cvsd_k,
                       lambda_min=lambda_min, lambda_1se=lambda_1se,
                       index_min=int(full_idx[idmin]), index_1se=int(full_idx[id1se]),
                       glmnet_fit=full_fit, fold_fits=list(fold_fits))


def _wmean_cols(m, w):
    """``weighted.mean(m[, j], w, na.rm = TRUE)`` for every column of ``m``
    (folds x lambdas); NaN where a column has no finite entry.

    NA entries contribute an exact 0.0 to both sums, and the column sums run
    down the folds in order, so each value is bitwise what summing only the
    non-NA entries of that column gives.
    """
    ok = ~np.isnan(m)
    with np.errstate(invalid="ignore"):
        num = np.where(ok, m * w[:, None], 0.0).sum(axis=0)
        den = np.where(ok, w[:, None], 0.0).sum(axis=0)
        out = num / den
    out[~ok.any(axis=0)] = np.nan
    return out


def cv_glmnet_gaussian(x, y, foldid, alpha=1.0, device="cpu", **kwargs) -> CvGlmnetFit:
    """``cv.glmnet(x, y, foldid = foldid, alpha = alpha, type.measure = "mse")``."""
    probs = cv_problems(x, y, foldid, alpha, **kwargs)
    fits = fit_paths(probs, device=device)
    if isinstance(fits[0], Exception):
        raise fits[0]
    return cv_summarize(x, y, foldid, fits[0], fits[1:])
