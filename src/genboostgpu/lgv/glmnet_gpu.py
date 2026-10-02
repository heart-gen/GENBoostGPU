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
* the coordinate loop is latency-bound (one warp per problem leaves few warps
  per SM to hide memory latency), so in naive mode the residual lives in
  shared memory and each lane holds its slice of the current ``x`` column in
  registers (``NC`` = ceil(max n / 32) values, a compile-time constant), used
  for both the gradient and the residual update. The next coordinate's column
  and scalars (``A``, ``XV``, ``XS``) are loaded while the current one is
  processed, and the index after it one step earlier (the GPU issues
  instructions in order, so a load is only hidden if nothing uses it until
  later): active variables in partial passes, a compact ascending
  strong-set list ``SL`` in full passes;
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

from .glmnet import (GlmnetError, GlmnetProblem, _BIG, _DEVMAX, _EPS, _FDEV, _MNLAM, _finish,
                     _solve_cpu_safe)

__all__ = ["solve_batch_gpu", "WARPS_PER_BLOCK"]

WARPS_PER_BLOCK = 1   # a block's slots are held until its slowest warp ends
_SHARED_BYTES = 48 * 1024   # static shared memory per block
_FULL = 0xFFFFFFFF
_kernels = {}
cuda = None  # numba.cuda, bound on first use (a module global so the CUDA
             # simulator can substitute its own namespace in tests)


def _build_kernel(NC, WPB):
    """Kernels for problems with n <= 32 * NC, launched with WPB warps per block."""
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
    def cload(R, X, xa, n, lane):
        """Lane's slice of column ``X[xa:xa + n]`` (samples lane + 32c) into R."""
        for c in range(NC):
            i = lane + 32 * c
            R[c] = X[xa + i] if i < n else 0.0

    @cuda.jit(device=True, inline=True)
    def cdot(R, Y, ya, n, lane):
        """Warp dot product of a register-held column with ``Y``: each lane
        sums its samples in order, then the butterfly."""
        s = 0.0
        for c in range(NC):
            i = lane + 32 * c
            if i < n:
                s += R[c] * Y[ya + i]
        return wsum(s)

    @cuda.jit(device=True, inline=True)
    def cupd(R, Y, ya, diff, n, lane):
        for c in range(NC):
            i = lane + 32 * c
            if i < n:
                Y[ya + i] -= diff * R[c]

    @cuda.jit(device=True, inline=True)
    def strong_list(IX, po, p, SL, lane):
        """Compact ascending list of strong variables (IX != 0); returns its length."""
        ns = 0
        for base in range(0, p, 32):
            k = base + lane
            f = k < p and IX[po + k] != 0
            m = cuda.ballot_sync(_FULL, f)
            if f:
                SL[po + ns + cuda.popc(m & ((1 << lane) - 1))] = k
            ns += cuda.popc(m)
        cuda.syncwarp(_FULL)
        return ns

    @cuda.jit(device=True, inline=True)
    def ldot(X, xa, Y, ya, n):
        """One lane's sum_i X[xa + i] * Y[ya + i] in Eigen's order (as
        :func:`~genboostgpu.lgv.glmnet_core.edot`). Used where the warp has
        many independent dot products: each lane takes its own column, which
        avoids a butterfly reduction per column."""
        if n <= 0:
            return 0.0
        if n < 2:
            return X[xa] * Y[ya]
        al = (n // 2) * 2
        al2 = (n // 4) * 4
        s0 = X[xa] * Y[ya]
        s1 = X[xa + 1] * Y[ya + 1]
        if al > 2:
            s2 = X[xa + 2] * Y[ya + 2]
            s3 = X[xa + 3] * Y[ya + 3]
            for i in range(4, al2, 4):
                s0 += X[xa + i] * Y[ya + i]
                s1 += X[xa + i + 1] * Y[ya + i + 1]
                s2 += X[xa + i + 2] * Y[ya + i + 2]
                s3 += X[xa + i + 3] * Y[ya + i + 3]
            s0 = s0 + s2
            s1 = s1 + s3
            if al > al2:
                s0 += X[xa + al2] * Y[ya + al2]
                s1 += X[xa + al2 + 1] * Y[ya + al2 + 1]
        res = s0 + s1
        for i in range(al, n):
            res += X[xa + i] * Y[ya + i]
        return res

    @cuda.jit
    def prep_kernel(U, XR, X, n_arr, p_arr, isd_arr, x_off, s_off, JU, XM, XS, XV):
        """Chkvars + Standardize1's x part, one warp per distinct matrix.

        Problems that share a matrix (the alphas of one fold) share its
        standardized copy and column statistics, which depend only on x.
        Matrices arrive row-major in ``XR`` (a plain copy on the host) and are
        transposed here into the column-major ``X`` the path kernel reads.
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
            for i in range(lane, n, 32):
                X[xo + j * n + i] = XR[xo + i * p + j]
        cuda.syncwarp(_FULL)
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

    NMAX = 32 * NC
    NSH = WPB * NMAX   # shared-array shapes must be compile-time constants

    @cuda.jit
    def kernel(B, X, Y, n_arr, p_arr, nlam_arr, nx_arr, naive_arr, alpha_arr,
               flmin_arr, isd_arr, x_off, y_off, p_off, lam_off, nx_off, ca_off, c_off,
               s_off, ULAM, thr, maxit, eps, big, mnlam, sml, rsqmax,
               A0, CA, IA, KIN, RSQO, ALMO, INTS, DBL,
               XM, XS, XV, JU, G, A, MM, IX, DA, CG, SL):
        tid = cuda.threadIdx.x
        lane = tid % 32
        Ysh = cuda.shared.array(NSH, np.float64)
        cur = cuda.local.array(NC, np.float64)
        nxt = cuda.local.array(NC, np.float64)
        ysb = (tid // 32) * NMAX
        b = cuda.blockIdx.x * WPB + tid // 32
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
            Ysh[ysb + i] = Y[yo + i] / ys
        DBL[2 * b] = ym
        DBL[2 * b + 1] = ys

        for j in range(p):
            G[po + j] = 0.0
            A[po + j] = 0.0
            MM[po + j] = 0
            IX[po + j] = 0
        cuda.syncwarp(_FULL)
        for j in range(lane, p, 32):
            if JU[so + j] != 0:
                s = ldot(X, xo + j * n, Ysh, ysb, n)
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
        ns = 0

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
                ns = strong_list(IX, po, p, SL, lane)

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
                        kn = 0
                        knn = 0
                        an = 0.0
                        xvn = 0.0
                        xsn = 0.0
                        if naive and nin > 0:
                            kn = IA[nxo]
                            cload(nxt, X, xo + kn * n, n, lane)
                            an = A[po + kn]
                            xvn = XV[so + kn]
                            xsn = XS[so + kn]
                            if nin > 1:
                                knn = IA[nxo + 1]
                        for l in range(nin):
                            gk = 0.0
                            if naive:
                                # the next variable's column and scalars are
                                # loaded now and its index one step earlier,
                                # so no load is waited on in the loop body
                                k = kn
                                old = an
                                xvk = xvn
                                xsk = xsn
                                for c in range(NC):
                                    cur[c] = nxt[c]
                                if l + 1 < nin:
                                    kn = knn
                                    if l + 2 < nin:
                                        knn = IA[nxo + l + 2]
                                    cload(nxt, X, xo + kn * n, n, lane)
                                    an = A[po + kn]
                                    xvn = XV[so + kn]
                                    xsn = XS[so + kn]
                                gk = cdot(cur, Ysh, ysb, n, lane)
                            else:
                                k = IA[nxo + l]
                                old = A[po + k]
                                gk = G[po + k]
                                xvk = XV[so + k]
                                xsk = XS[so + k]
                            u = gk + old * xvk
                            v = abs(u) - ab
                            new = 0.0
                            if v > 0.0:
                                lim = big / ys * xsk
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
                                cupd(cur, Ysh, ysb, diff, n, lane)
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
                    kn = 0
                    knn = 0
                    an = 0.0
                    xvn = 0.0
                    xsn = 0.0
                    if naive and ns > 0:
                        kn = SL[po]
                        cload(nxt, X, xo + kn * n, n, lane)
                        an = A[po + kn]
                        xvn = XV[so + kn]
                        xsn = XS[so + kn]
                        if ns > 1:
                            knn = SL[po + 1]
                    for q in range(ns if naive else p):
                        gk = 0.0
                        if naive:
                            # the strong set in ascending order, as the full
                            # scan over IX; prefetched as in partial passes
                            k = kn
                            old = an
                            xvk = xvn
                            xsk = xsn
                            for c in range(NC):
                                cur[c] = nxt[c]
                            if q + 1 < ns:
                                kn = knn
                                if q + 2 < ns:
                                    knn = SL[po + q + 2]
                                cload(nxt, X, xo + kn * n, n, lane)
                                an = A[po + kn]
                                xvn = XV[so + kn]
                                xsn = XS[so + kn]
                            gk = cdot(cur, Ysh, ysb, n, lane)
                        else:
                            k = q
                            if JU[so + k] == 0:
                                continue
                            gk = G[po + k]
                            old = A[po + k]
                            xvk = XV[so + k]
                            xsk = XS[so + k]
                        u = gk + old * xvk
                        v = abs(u) - ab
                        new = 0.0
                        if v > 0.0:
                            lim = big / ys * xsk
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
                                for j in range(lane, p, 32):
                                    if JU[so + j] == 0:
                                        continue
                                    cv = 0.0
                                    if j == k:
                                        cv = XV[so + j]
                                    elif MM[po + j] != 0:
                                        cv = CG[co + (MM[po + j] - 1) * p + k]
                                    else:
                                        cv = ldot(X, xo + j * n, X, xo + k * n, n)
                                    CG[co + col * p + j] = cv
                                cuda.syncwarp(_FULL)
                        diff = new - old
                        d2 = xvk * diff * diff
                        if d2 > dlx:
                            dlx = d2
                        rsq += diff * (2.0 * gk - diff * xvk)
                        if naive:
                            cupd(cur, Ysh, ysb, diff, n, lane)
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
                        cuda.syncwarp(_FULL)   # every lane reads the whole residual
                        for k in range(lane, p, 32):
                            if IX[po + k] != 0 or JU[so + k] == 0:
                                continue
                            G[po + k] = abs(ldot(X, xo + k * n, Ysh, ysb, n))
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
                        ns = strong_list(IX, po, p, SL, lane)
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


_compact_kernel = None


def _get_compact_kernel():
    """Copies each problem's used block of CA (lmu rows x W columns, W = its
    largest active-set size) into a dense buffer, so only that is downloaded:
    CA is sized for nx = p active variables per lambda, which is ~10x what a
    path uses."""
    global _compact_kernel, cuda
    if _compact_kernel is None:
        from numba import cuda as _cuda

        cuda = _cuda

        @cuda.jit
        def compact(B, CA, ca_off, nx_arr, INTS, W, cc_off, OUT):
            tid = cuda.threadIdx.x
            lane = tid % 32
            b = cuda.blockIdx.x * (cuda.blockDim.x // 32) + tid // 32
            if b >= B:
                return
            w = W[b]
            nx = nx_arr[b]
            cao = ca_off[b]
            o = cc_off[b]
            for t in range(lane, INTS[3 * b] * w, 32):
                m = t // w
                OUT[o + t] = CA[cao + m * nx + (t - m * w)]

        _compact_kernel = compact
    return _compact_kernel


def _get_kernel(nc: int, wpb: int):
    key = (nc, wpb)
    if key not in _kernels:
        _kernels[key] = _build_kernel(nc, wpb)
    return _kernels[key]


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
    nmax_gpu = _SHARED_BYTES // 8   # one warp's residual must fit in shared memory
    x_finite = {}   # the alphas of a fold share x: check it once
    for i, prob in enumerate(problems):
        if not prob.intercept:
            raise ValueError("GPU solver supports intercept=TRUE only")
        try:
            fin = x_finite.get(id(prob.x))
            if fin is None:
                fin = x_finite[id(prob.x)] = bool(np.all(np.isfinite(prob.x)))
            spec = prob.resolved(x_finite=fin)
        except GlmnetError as err:
            results[i] = err
            continue
        if spec["n"] > nmax_gpu:
            results[i] = _solve_cpu_safe(prob)
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

    Yh = np.empty(int(y_off[-1]))
    ULAMh = np.zeros(int(lam_off[-1]))
    for b, s in enumerate(specs):
        Yh[y_off[b]:y_off[b + 1]] = s["y"]
        if s["user_lambda"]:
            ULAMh[lam_off[b]:lam_off[b] + s["nlam"]] = s["ulam"]

    d = cp.asarray
    nc = int(-(-n_arr.max() // 32))
    wpb = max(1, min(warps_per_block, _SHARED_BYTES // (32 * nc * 8)))
    prep_kernel, kernel = _get_kernel(nc, wpb)
    tpb = 32 * wpb
    # each matrix is copied row-major straight into its slice of the device
    # buffer (a host staging buffer this size costs seconds in page faults)
    XR = cp.empty(int(ux_off[-1]))
    for u, m in enumerate(umats):
        m = np.ascontiguousarray(m).ravel()
        if hasattr(XR, "set"):
            XR[ux_off[u]:ux_off[u + 1]].set(m)
        else:
            XR[ux_off[u]:ux_off[u + 1]] = m
    X = cp.empty_like(XR)
    nus = int(us_off[-1])
    XM = cp.zeros(nus)
    XS = cp.zeros(nus)
    XV = cp.zeros(nus)
    JU = cp.zeros(nus, cp.int8)
    prep_kernel[(U + wpb - 1) // wpb, tpb](
        U, XR, X, d(un_arr), d(up_arr), d(np.array(uisd, np.int64)), d(ux_off), d(us_off),
        JU, XM, XS, XV)
    if hasattr(cp, "cuda"):
        cp.cuda.runtime.deviceSynchronize()
    del XR   # before the path outputs are allocated: no rise in peak memory
    Y = d(Yh)
    ULAM = d(ULAMh)
    nl = int(lam_off[-1])
    np_ = int(p_off[-1])
    nxt = int(nx_off[-1])
    A0 = cp.zeros(nl)
    CA = cp.zeros(int(ca_off[-1]))
    IA = cp.zeros(nxt, cp.int64)
    KIN = cp.zeros(nl, cp.int64)
    RSQO = cp.zeros(nl)
    ALMO = cp.zeros(nl)
    INTS = cp.zeros(3 * B, cp.int64)
    DBL = cp.zeros(2 * B)
    G = cp.zeros(np_)
    A = cp.zeros(np_)
    MM = cp.zeros(np_, cp.int64)
    IX = cp.zeros(np_, cp.int8)
    DA = cp.zeros(nxt)
    CG = cp.zeros(int(c_off[-1]))
    SL = cp.zeros(np_, cp.int64)

    blocks = (B + wpb - 1) // wpb
    kernel[blocks, tpb](
        B, X, Y, d(n_arr), d(p_arr), d(nlam_arr), d(nx_arr), d(naive_arr),
        d(alpha_arr), d(flmin_arr), d(isd_arr), d(x_off), d(y_off), d(p_off), d(lam_off),
        d(nx_off), d(ca_off), d(c_off), d(s_off), ULAM, thr, maxit, _EPS, _BIG, _MNLAM,
        _FDEV, _DEVMAX, A0, CA, IA, KIN, RSQO, ALMO, INTS, DBL,
        XM, XS, XV, JU, G, A, MM, IX, DA, CG, SL)
    if hasattr(cp, "cuda"):
        cp.cuda.runtime.deviceSynchronize()

    INTSh, KINh = host(INTS), host(KIN)
    lmu_arr = np.maximum(INTSh[0::3], 0)
    W = np.zeros(B, np.int64)
    for b in range(B):
        if lmu_arr[b] > 0:
            W[b] = KINh[lam_off[b]:lam_off[b] + lmu_arr[b]].max()
    cc_off = offsets(lmu_arr * W)
    CC = cp.zeros(max(int(cc_off[-1]), 1))
    _get_compact_kernel()[(B + 3) // 4, 128](B, CA, d(ca_off), d(nx_arr), INTS, d(W), d(cc_off), CC)
    if hasattr(cp, "cuda"):
        cp.cuda.runtime.deviceSynchronize()
    del CA
    A0h, CCh, IAh = host(A0), host(CC), host(IA)
    RSQOh, ALMOh = host(RSQO), host(ALMO)
    for b, s in enumerate(specs):
        out = dict(
            a0=A0h[lam_off[b]:lam_off[b + 1]], ca=CCh[cc_off[b]:cc_off[b + 1]], ca_stride=W[b],
            ia=IAh[nx_off[b]:nx_off[b + 1]], kin=KINh[lam_off[b]:lam_off[b + 1]],
            rsqo=RSQOh[lam_off[b]:lam_off[b + 1]], almo=ALMOh[lam_off[b]:lam_off[b + 1]].copy(),
            ints_out=INTSh[3 * b:3 * b + 3],
        )
        try:
            results[idx[b]] = _finish(s, problems[idx[b]], out)
        except GlmnetError as err:
            results[idx[b]] = err
    return results
