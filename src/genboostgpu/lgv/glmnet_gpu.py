"""Warp-cooperative CUDA port of the glmnet Gaussian path solver.

Same algorithm as :mod:`genboostgpu.lgv.glmnet_core` (glmnet 4.1-10's
``ElnetDriver<gaussian>``), mapped onto the GPU as **one warp per glmnet
problem**:

* control flow (lambda path, passes, strong set, KKT) is executed uniformly
  by the 32 lanes, which hold identical scalar state in registers;
* the length-``n`` loops (dot products, residual and standardization
  updates) are split across lanes with sample ``i`` owned by lane ``i % 32``,
  so memory access is coalesced and lanes never read each other's samples;
* reductions use an xor butterfly, which leaves the bitwise-identical sum in
  every lane (floating-point addition is commutative), so lanes never
  disagree on a scalar.
* problems that share a design matrix (the alphas of one fold) share one
  device copy: ``prep_kernel`` (one warp per distinct matrix) runs Chkvars
  and standardizes it in place once, and the path kernel only reads it;
* global read-modify-write updates (``-=``, ``*=``) are made only by the
  lane that owns the element, never redundantly by all 32 lanes, since lanes
  of a warp are not guaranteed to run in lockstep.

Reductions sum in a different order than the CPU core, so results agree with
the CPU/R path to rounding (~1e-12) rather than bit for bit.

Only the configurations the pipeline uses are supported on the GPU:
``intercept = TRUE``, either ``standardize`` setting, unit penalty factors.
"""
from __future__ import annotations

import math

import numpy as np

from .glmnet import GlmnetError, GlmnetProblem, _BIG, _DEVMAX, _EPS, _FDEV, _MNLAM, _finish

__all__ = ["solve_batch_gpu", "WARPS_PER_BLOCK"]

WARPS_PER_BLOCK = 4
_FULL = 0xFFFFFFFF
_kernel = None
cuda = None  # numba.cuda, bound on first use (a module global so the CUDA
             # simulator can substitute its own namespace in tests)


def _build_kernel():
    global cuda
    from numba import cuda as _cuda

    cuda = _cuda

    @cuda.jit(device=True, inline=True)
    def wsum(v):
        v += cuda.shfl_xor_sync(_FULL, v, 16)
        v += cuda.shfl_xor_sync(_FULL, v, 8)
        v += cuda.shfl_xor_sync(_FULL, v, 4)
        v += cuda.shfl_xor_sync(_FULL, v, 2)
        v += cuda.shfl_xor_sync(_FULL, v, 1)
        return v

    @cuda.jit(device=True, inline=True)
    def wdot(X, xa, Y, ya, n, lane):
        s = 0.0
        for i in range(lane, n, 32):
            s += X[xa + i] * Y[ya + i]
        return wsum(s)

    @cuda.jit
    def prep_kernel(U, X, n_arr, p_arr, isd_arr, x_off, s_off, JU, XM, XS, XV):
        """Chkvars + Standardize1's x part, one warp per distinct matrix.

        Problems that share a matrix (the alphas of one fold) share its
        standardized copy and column statistics, which depend only on x.
        """
        tid = cuda.threadIdx.x
        lane = tid % 32
        u = cuda.blockIdx.x * (cuda.blockDim.x // 32) + tid // 32
        if u >= U:
            return
        n = n_arr[u]
        p = p_arr[u]
        isd = isd_arr[u]
        xo = x_off[u]
        po = s_off[u]
        for j in range(p):
            t = X[xo + j * n]
            d = False
            for i in range(1 + lane, n, 32):
                if X[xo + j * n + i] != t:
                    d = True
            ok = wsum(1.0 if d else 0.0) > 0.0
            if lane == 0:
                JU[po + j] = 1 if ok else 0
        w = 1.0 / n
        vsq = math.sqrt(w)
        for j in range(p):
            if lane == 0:
                XM[po + j] = 0.0
                XS[po + j] = 1.0
                XV[po + j] = 1.0
        cuda.syncwarp(_FULL)
        for j in range(p):
            if JU[po + j] == 0:
                continue
            s = 0.0
            for i in range(lane, n, 32):
                s += X[xo + j * n + i] * w
            s = wsum(s)
            ss = 0.0
            for i in range(lane, n, 32):
                v = vsq * (X[xo + j * n + i] - s)
                X[xo + j * n + i] = v
                ss += v * v
            ss = wsum(ss)
            if isd:
                sd = math.sqrt(ss)
                for i in range(lane, n, 32):
                    X[xo + j * n + i] = X[xo + j * n + i] / sd
            if lane == 0:
                XM[po + j] = s
                if isd:
                    XS[po + j] = sd
                else:
                    XV[po + j] = ss
            cuda.syncwarp(_FULL)

    @cuda.jit
    def kernel(B, X, Y, n_arr, p_arr, nlam_arr, nx_arr, naive_arr, alpha_arr,
               flmin_arr, isd_arr, x_off, y_off, p_off, lam_off, nx_off, ca_off, c_off,
               s_off, ULAM, thr, maxit, eps, big, mnlam, sml, rsqmax,
               A0, CA, IA, KIN, RSQO, ALMO, INTS, DBL,
               XM, XS, XV, JU, G, A, MM, IX, DA, CG):
        tid = cuda.threadIdx.x
        lane = tid % 32
        b = cuda.blockIdx.x * (cuda.blockDim.x // 32) + tid // 32
        if b >= B:
            return
        n = n_arr[b]
        p = p_arr[b]
        nlam = nlam_arr[b]
        nx = nx_arr[b]
        naive = naive_arr[b]
        beta = alpha_arr[b]
        flmin = flmin_arr[b]
        isd = isd_arr[b]
        xo = x_off[b]
        yo = y_off[b]
        po = p_off[b]
        lo = lam_off[b]
        nxo = nx_off[b]
        cao = ca_off[b]
        co = c_off[b]
        so = s_off[b]
        ne = p + 1

        lmu = 0
        nlp = 0
        jerr = 0

        # ---- Chkvars / Standardize1 x part: done by prep_kernel ----
        any_ju = False
        for j in range(p):
            if JU[so + j] != 0:
                any_ju = True
        if not any_ju:
            if lane == 0:
                INTS[3 * b] = 0
                INTS[3 * b + 1] = 0
                INTS[3 * b + 2] = 7777
            return

        # ---- Standardize1, y part ----
        w = 1.0 / n
        vsq = math.sqrt(w)
        s = 0.0
        for i in range(lane, n, 32):
            s += Y[yo + i] * w
        ym = wsum(s)
        ss = 0.0
        for i in range(lane, n, 32):
            v = vsq * (Y[yo + i] - ym)
            Y[yo + i] = v
            ss += v * v
        ys = math.sqrt(wsum(ss))
        for i in range(lane, n, 32):
            Y[yo + i] = Y[yo + i] / ys
        DBL[2 * b] = ym
        DBL[2 * b + 1] = ys

        for j in range(p):
            G[po + j] = 0.0
            A[po + j] = 0.0
            MM[po + j] = 0
            IX[po + j] = 0
        cuda.syncwarp(_FULL)
        for j in range(p):
            if JU[so + j] == 0:
                continue
            s = wdot(X, xo + j * n, Y, yo, n, lane)
            G[po + j] = abs(s) if naive else s
        cuda.syncwarp(_FULL)

        # ---- path ----
        omb = 1.0 - beta
        alf = 1.0
        if flmin < 1.0:
            eqs = eps if eps > flmin else flmin
            alf = math.pow(eqs, 1.0 / (nlam - 1.0))
        mnl = mnlam if mnlam < nlam else nlam
        nin = 0
        rsq = 0.0
        rsq_prev = 0.0
        iz = False
        lmda_curr = 0.0

        for m in range(nlam):
            alm0 = lmda_curr
            alm = alm0
            if flmin >= 1.0:
                alm = ULAM[lo + m] / ys
            elif m > 1:
                alm = alm * alf
            elif m == 0:
                alm = big
            else:
                alm0 = 0.0
                for j in range(p):
                    if JU[so + j] == 0:
                        continue
                    gj = abs(G[po + j])
                    if gj > alm0:
                        alm0 = gj
                bb = beta if beta > 1e-3 else 1e-3
                alm0 = alm0 / bb
                alm = alm0 * alf
            lmda_curr = alm
            dem = alm * omb
            ab = alm * beta

            rsq_prev = rsq
            if naive:
                tlam = beta * (2.0 * alm - alm0)
                for k in range(lane, p, 32):
                    if IX[po + k] == 0 and JU[so + k] != 0:
                        if G[po + k] > tlam:
                            IX[po + k] = 1
                cuda.syncwarp(_FULL)

            maxit_hit = False
            maxact_hit = False
            do_partial = iz
            while True:
                if do_partial:
                    if not naive:
                        for l in range(lane, nin, 32):
                            DA[nxo + l] = A[po + IA[nxo + l]]
                        cuda.syncwarp(_FULL)
                    iz = True
                    while True:
                        nlp += 1
                        dlx = 0.0
                        for l in range(nin):
                            k = IA[nxo + l]
                            old = A[po + k]
                            gk = 0.0
                            if naive:
                                gk = wdot(X, xo + k * n, Y, yo, n, lane)
                            else:
                                gk = G[po + k]
                            xvk = XV[so + k]
                            u = gk + old * xvk
                            v = abs(u) - ab
                            new = 0.0
                            if v > 0.0:
                                lim = big / ys * XS[so + k]
                                t = math.copysign(v, u) / (xvk + dem)
                                if t > lim:
                                    t = lim
                                if t < -lim:
                                    t = -lim
                                new = t
                            cuda.syncwarp(_FULL)
                            A[po + k] = new
                            if old == new:
                                continue
                            diff = new - old
                            d2 = xvk * diff * diff
                            if d2 > dlx:
                                dlx = d2
                            rsq += diff * (2.0 * gk - diff * xvk)
                            if naive:
                                for i in range(lane, n, 32):
                                    Y[yo + i] -= diff * X[xo + k * n + i]
                            else:
                                ck = MM[po + k] - 1
                                for l2 in range(lane, nin, 32):
                                    j = IA[nxo + l2]
                                    G[po + j] -= CG[co + ck * p + j] * diff
                                cuda.syncwarp(_FULL)
                        if dlx < thr:
                            break
                        if nlp > maxit:
                            maxit_hit = True
                            break
                    if maxit_hit:
                        break
                    if not naive:
                        cuda.syncwarp(_FULL)
                        for l in range(lane, nin, 32):
                            DA[nxo + l] -= A[po + IA[nxo + l]]
                        cuda.syncwarp(_FULL)
                        for j in range(lane, p, 32):
                            if MM[po + j] != 0 or JU[so + j] == 0:
                                continue
                            s = 0.0
                            for l in range(nin):
                                s += DA[nxo + l] * CG[co + l * p + j]
                            G[po + j] += s
                        cuda.syncwarp(_FULL)

                done = False
                while True:
                    if naive and nlp > maxit:
                        maxit_hit = True
                        break
                    nlp += 1
                    dlx = 0.0
                    for k in range(p):
                        if naive:
                            if IX[po + k] == 0:
                                continue
                        else:
                            if JU[so + k] == 0:
                                continue
                        old = A[po + k]
                        gk = 0.0
                        if naive:
                            gk = wdot(X, xo + k * n, Y, yo, n, lane)
                        else:
                            gk = G[po + k]
                        xvk = XV[so + k]
                        u = gk + old * xvk
                        v = abs(u) - ab
                        new = 0.0
                        if v > 0.0:
                            lim = big / ys * XS[so + k]
                            t = math.copysign(v, u) / (xvk + dem)
                            if t > lim:
                                t = lim
                            if t < -lim:
                                t = -lim
                            new = t
                        cuda.syncwarp(_FULL)
                        A[po + k] = new
                        if old == new:
                            continue
                        if MM[po + k] == 0:
                            nin += 1
                            if nin > nx:
                                maxact_hit = True
                                break
                            cuda.syncwarp(_FULL)
                            MM[po + k] = nin
                            IA[nxo + nin - 1] = k
                            cuda.syncwarp(_FULL)
                            if not naive:
                                col = nin - 1
                                for j in range(p):
                                    if JU[so + j] == 0:
                                        continue
                                    cv = 0.0
                                    if j == k:
                                        cv = XV[so + j]
                                    elif MM[po + j] != 0:
                                        cv = CG[co + (MM[po + j] - 1) * p + k]
                                    else:
                                        cv = wdot(X, xo + j * n, X, xo + k * n, n, lane)
                                    CG[co + col * p + j] = cv
                                cuda.syncwarp(_FULL)
                        diff = new - old
                        d2 = xvk * diff * diff
                        if d2 > dlx:
                            dlx = d2
                        rsq += diff * (2.0 * gk - diff * xvk)
                        if naive:
                            for i in range(lane, n, 32):
                                Y[yo + i] -= diff * X[xo + k * n + i]
                        else:
                            ck = MM[po + k] - 1
                            for j in range(lane, p, 32):
                                if JU[so + j] != 0:
                                    G[po + j] -= CG[co + ck * p + j] * diff
                            cuda.syncwarp(_FULL)
                    if maxact_hit:
                        break
                    if dlx < thr:
                        if not naive:
                            done = True
                            break
                        for k in range(p):
                            if IX[po + k] != 0 or JU[so + k] == 0:
                                continue
                            s = wdot(X, xo + k * n, Y, yo, n, lane)
                            G[po + k] = abs(s)
                        cuda.syncwarp(_FULL)
                        updated = False
                        for k in range(p):
                            if IX[po + k] != 0 or JU[so + k] == 0:
                                continue
                            if G[po + k] > ab:
                                updated = True
                                cuda.syncwarp(_FULL)
                                IX[po + k] = 1
                        cuda.syncwarp(_FULL)
                        if not updated:
                            done = True
                            break
                        continue
                    if nlp > maxit:
                        maxit_hit = True
                    break
                if maxit_hit or maxact_hit or done:
                    break
                do_partial = True

            if maxit_hit:
                jerr = -m - 1
                break
            if maxact_hit:
                jerr = -10001 - m
                break

            me = 0
            for l in range(nin):
                val = A[po + IA[nxo + l]]
                CA[cao + m * nx + l] = val
                if val != 0.0:
                    me += 1
            RSQO[lo + m] = rsq
            KIN[lo + m] = nin
            ALMO[lo + m] = alm
            lmu = m + 1
            prop = math.inf
            if rsq != 0.0:
                prop = (rsq - rsq_prev) / rsq
            if lmu < mnl or flmin >= 1.0:
                continue
            if me > ne or prop < sml or rsq > rsqmax:
                break

        cuda.syncwarp(_FULL)
        if jerr <= 0:
            # read-modify-write: each lambda is owned by one lane
            for k in range(lane, lmu, 32):
                ALMO[lo + k] = ALMO[lo + k] * ys
                nk = KIN[lo + k]
                for l in range(nk):
                    CA[cao + k * nx + l] = CA[cao + k * nx + l] * (ys / XS[so + IA[nxo + l]])
                s = 0.0
                for l in range(nk):
                    s -= CA[cao + k * nx + l] * XM[so + IA[nxo + l]]
                s += ym
                A0[lo + k] = s
        if lane == 0:
            INTS[3 * b] = lmu
            INTS[3 * b + 1] = nlp
            INTS[3 * b + 2] = jerr

    return prep_kernel, kernel


def _get_kernel():
    global _kernel
    if _kernel is None:
        _kernel = _build_kernel()
    return _kernel


def solve_batch_gpu(problems, warps_per_block: int = WARPS_PER_BLOCK, xp=None):
    """Solve ``problems`` (list of GlmnetProblem) on the current CUDA device.

    Returns a list of ``GlmnetFit`` / ``GlmnetError`` in input order.
    ``xp`` defaults to CuPy; tests pass NumPy together with numba's CUDA
    simulator (``NUMBA_ENABLE_CUDASIM=1``) to exercise the kernel on a CPU.
    """
    if xp is None:
        from ..backend import import_cupy
        xp = import_cupy()
    cp = xp

    def host(a):
        return a.get() if hasattr(a, "get") else a

    results = [None] * len(problems)
    specs = []
    idx = []
    for i, prob in enumerate(problems):
        if not prob.intercept:
            raise ValueError("GPU solver supports intercept=TRUE only")
        try:
            spec = prob.resolved()
        except GlmnetError as err:
            results[i] = err
            continue
        spec["y_orig"] = spec["y"]
        specs.append(spec)
        idx.append(i)
    if not specs:
        return results

    B = len(specs)
    # Distinct design matrices: the alphas of one fold share an x object
    # (cv_problems fold_cache); each is uploaded and standardized once.
    umap = np.empty(B, np.int64)
    ukeys, umats, uisd = {}, [], []
    for b, (s, i) in enumerate(zip(specs, idx)):
        key = (id(s["x"]), int(problems[i].standardize))
        if key not in ukeys:
            ukeys[key] = len(umats)
            umats.append(s["x"])
            uisd.append(key[1])
        umap[b] = ukeys[key]
    U = len(umats)
    un_arr = np.array([m.shape[0] for m in umats], np.int64)
    up_arr = np.array([m.shape[1] for m in umats], np.int64)

    n_arr = np.array([s["n"] for s in specs], np.int64)
    p_arr = np.array([s["p"] for s in specs], np.int64)
    nlam_arr = np.array([s["nlam"] for s in specs], np.int64)
    nx_arr = np.array([s["nx"] for s in specs], np.int64)
    naive_arr = np.array([s["naive"] for s in specs], np.int64)
    alpha_arr = np.array([problems[i].alpha for i in idx], np.float64)
    flmin_arr = np.array([s["flmin"] for s in specs], np.float64)
    isd_arr = np.array([int(problems[i].standardize) for i in idx], np.int64)
    thr = float(problems[idx[0]].thresh)
    maxit = int(problems[idx[0]].maxit)

    def offsets(sizes):
        off = np.zeros(len(sizes) + 1, np.int64)
        np.cumsum(sizes, out=off[1:])
        return off

    ux_off = offsets(un_arr * up_arr)
    us_off = offsets(up_arr)
    x_off = ux_off[umap]
    s_off = us_off[umap]
    y_off = offsets(n_arr)
    p_off = offsets(p_arr)
    lam_off = offsets(nlam_arr)
    nx_off = offsets(nx_arr)
    ca_off = offsets(nx_arr * nlam_arr)
    c_off = offsets(np.where(naive_arr == 1, 1, p_arr * nx_arr))

    Xh = np.empty(int(ux_off[-1]))
    for u, m in enumerate(umats):
        Xh[ux_off[u]:ux_off[u + 1]] = m.ravel(order="F")
    Yh = np.empty(int(y_off[-1]))
    ULAMh = np.zeros(int(lam_off[-1]))
    for b, s in enumerate(specs):
        Yh[y_off[b]:y_off[b + 1]] = s["y"]
        if s["user_lambda"]:
            ULAMh[lam_off[b]:lam_off[b] + s["nlam"]] = s["ulam"]

    d = cp.asarray
    X = d(Xh)
    del Xh
    Y = d(Yh)
    ULAM = d(ULAMh)
    nl = int(lam_off[-1])
    np_ = int(p_off[-1])
    nus = int(us_off[-1])
    nxt = int(nx_off[-1])
    A0 = cp.zeros(nl)
    CA = cp.zeros(int(ca_off[-1]))
    IA = cp.zeros(nxt, cp.int64)
    KIN = cp.zeros(nl, cp.int64)
    RSQO = cp.zeros(nl)
    ALMO = cp.zeros(nl)
    INTS = cp.zeros(3 * B, cp.int64)
    DBL = cp.zeros(2 * B)
    XM = cp.zeros(nus)
    XS = cp.zeros(nus)
    XV = cp.zeros(nus)
    JU = cp.zeros(nus, cp.int8)
    G = cp.zeros(np_)
    A = cp.zeros(np_)
    MM = cp.zeros(np_, cp.int64)
    IX = cp.zeros(np_, cp.int8)
    DA = cp.zeros(nxt)
    CG = cp.zeros(int(c_off[-1]))

    prep_kernel, kernel = _get_kernel()
    tpb = 32 * warps_per_block
    prep_kernel[(U + warps_per_block - 1) // warps_per_block, tpb](
        U, X, d(un_arr), d(up_arr), d(np.array(uisd, np.int64)), d(ux_off), d(us_off),
        JU, XM, XS, XV)
    blocks = (B + warps_per_block - 1) // warps_per_block
    kernel[blocks, tpb](
        B, X, Y, d(n_arr), d(p_arr), d(nlam_arr), d(nx_arr), d(naive_arr),
        d(alpha_arr), d(flmin_arr), d(isd_arr), d(x_off), d(y_off), d(p_off), d(lam_off),
        d(nx_off), d(ca_off), d(c_off), d(s_off), ULAM, thr, maxit, _EPS, _BIG, _MNLAM,
        _FDEV, _DEVMAX, A0, CA, IA, KIN, RSQO, ALMO, INTS, DBL,
        XM, XS, XV, JU, G, A, MM, IX, DA, CG)
    if hasattr(cp, "cuda"):
        cp.cuda.runtime.deviceSynchronize()

    A0h, CAh, IAh, KINh = host(A0), host(CA), host(IA), host(KIN)
    RSQOh, ALMOh, INTSh = host(RSQO), host(ALMO), host(INTS)
    for b, s in enumerate(specs):
        out = dict(
            a0=A0h[lam_off[b]:lam_off[b + 1]], ca=CAh[ca_off[b]:ca_off[b + 1]],
            ia=IAh[nx_off[b]:nx_off[b + 1]], kin=KINh[lam_off[b]:lam_off[b + 1]],
            rsqo=RSQOh[lam_off[b]:lam_off[b + 1]], almo=ALMOh[lam_off[b]:lam_off[b + 1]].copy(),
            ints_out=INTSh[3 * b:3 * b + 3],
        )
        try:
            results[idx[b]] = _finish(s, problems[idx[b]], out)
        except GlmnetError as err:
            results[idx[b]] = err
    return results
