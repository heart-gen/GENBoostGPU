"""Port of glmnet's dense Gaussian elastic-net path solver (glmnet 4.1-10).

This is a line-for-line translation of glmnetpp's ``ElnetDriver<gaussian>``:
``Chkvars``, ``Standardize1``/``Standardize``, the path driver
(``ElnetPathBase``/``ElnetPathGaussianBase``) and both point solvers
(covariance and naive, ``ElnetPointGaussianBase``). R chooses covariance mode
when ``nvars < 500`` and naive mode otherwise; both are ported because the
iterates, and therefore the final digits, differ between them.

``elnet_core`` is written with scalar loops over flat, preallocated arrays and
no allocation or exceptions, and is compiled by ``numba.njit`` (see
:mod:`genboostgpu.lgv.glmnet`; the GPU kernel in
:mod:`genboostgpu.lgv.glmnet_gpu` follows the same algorithm). Reductions use
Eigen's summation order (:func:`edot`), so paths are bitwise identical to R's.

Array layout: ``x`` is column-major, ``x[j * n + i]``; ``ca`` is ``nx x nlam``
column-major, ``ca[m * nx + l]``; ``cgram`` is ``p x nx`` column-major.
``ia`` holds 0-based variable indices (glmnet's is 1-based).
"""
from __future__ import annotations

import math

from numba import njit

# Status codes written to ints_out[2] (glmnet's jerr).
JERR_OK = 0
JERR_NON_POSITIVE_PENALTY = 10000
JERR_ALL_EXCLUDED = 7777


# ---- reductions in Eigen's order -------------------------------------------
# glmnetpp computes every dot product and squared norm with Eigen 3.4, whose
# dynamic-size sum (redux_impl, LinearVectorizedTraversal) on SSE2 keeps two
# 2-double packet accumulators over blocks of 4, folds them, adds a leftover
# packet, takes the horizontal sum and adds the scalar tail. Summing in that
# order (instead of left to right) makes the solver's iterates bitwise
# identical to R's glmnet (conda-forge/CRAN x86-64 builds use SSE2 packets),
# so near-tied convergence and KKT decisions resolve the same way.


@njit(cache=False, nogil=True)
def edot(a, ao, b, bo, n):
    """sum_i a[ao + i] * b[bo + i] in Eigen's order."""
    if n <= 0:
        return 0.0
    if n < 2:
        return a[ao] * b[bo]
    al = (n // 2) * 2
    al2 = (n // 4) * 4
    s0 = a[ao] * b[bo]
    s1 = a[ao + 1] * b[bo + 1]
    if al > 2:
        s2 = a[ao + 2] * b[bo + 2]
        s3 = a[ao + 3] * b[bo + 3]
        for i in range(4, al2, 4):
            s0 += a[ao + i] * b[bo + i]
            s1 += a[ao + i + 1] * b[bo + i + 1]
            s2 += a[ao + i + 2] * b[bo + i + 2]
            s3 += a[ao + i + 3] * b[bo + i + 3]
        s0 = s0 + s2
        s1 = s1 + s3
        if al > al2:
            s0 += a[ao + al2] * b[bo + al2]
            s1 += a[ao + al2 + 1] * b[bo + al2 + 1]
    res = s0 + s1
    for i in range(al, n):
        res += a[ao + i] * b[bo + i]
    return res


@njit(cache=False, nogil=True)
def edot_c(a, ao, c, n):
    """sum_i a[ao + i] * c (a dot with a constant weight vector)."""
    if n <= 0:
        return 0.0
    if n < 2:
        return a[ao] * c
    al = (n // 2) * 2
    al2 = (n // 4) * 4
    s0 = a[ao] * c
    s1 = a[ao + 1] * c
    if al > 2:
        s2 = a[ao + 2] * c
        s3 = a[ao + 3] * c
        for i in range(4, al2, 4):
            s0 += a[ao + i] * c
            s1 += a[ao + i + 1] * c
            s2 += a[ao + i + 2] * c
            s3 += a[ao + i + 3] * c
        s0 = s0 + s2
        s1 = s1 + s3
        if al > al2:
            s0 += a[ao + al2] * c
            s1 += a[ao + al2 + 1] * c
    res = s0 + s1
    for i in range(al, n):
        res += a[ao + i] * c
    return res


@njit(cache=False, nogil=True)
def esumsq(a, ao, n):
    """sum_i a[ao + i]^2 (Eigen squaredNorm)."""
    return edot(a, ao, a, ao, n)



def elnet_core(
    x, n, p, y,
    beta, flmin, ulam, nlam, isd, intr, thr, maxit, naive, ne, nx, vp,
    eps, big, mnlam, sml, rsqmax,
    a0, ca, ia, kin, rsqo, almo, ints_out, dbl_out,
    xm, xs, xv, ju, g, a, mm, ix, da, cgram, vq,
):
    """Fit one glmnet gaussian path in place. See module docstring.

    ``ints_out`` receives ``(lmu, nlp, jerr)``; ``dbl_out`` receives
    ``(ym, ys)``. ``x`` and ``y`` are overwritten (standardized / residual).
    """
    lmu = 0
    nlp = 0
    jerr = 0

    # ---- normalize_penalty ------------------------------------------------
    vmax = vp[0]
    for j in range(p):
        if vp[j] > vmax:
            vmax = vp[j]
    if vmax <= 0.0:
        ints_out[0] = 0
        ints_out[1] = 0
        ints_out[2] = JERR_NON_POSITIVE_PENALTY
        return
    vsum = 0.0
    for j in range(p):
        v = vp[j]
        if v < 0.0:
            v = 0.0
        vq[j] = v
        vsum += v
    for j in range(p):
        vq[j] = vq[j] * (p / vsum)

    # ---- Chkvars: a column is usable unless constant ----------------------
    any_ju = False
    for j in range(p):
        t = x[j * n]
        ok = 0
        for i in range(1, n):
            if x[j * n + i] != t:
                ok = 1
                break
        ju[j] = ok
        if ok:
            any_ju = True
    if not any_ju:
        ints_out[0] = 0
        ints_out[1] = 0
        ints_out[2] = JERR_ALL_EXCLUDED
        return

    # ---- Standardize1 (unit weights) --------------------------------------
    for j in range(p):
        xm[j] = 0.0
        xs[j] = 0.0
        xv[j] = 0.0
    w = 1.0 / n
    vsq = math.sqrt(w)
    ym = 0.0
    ys = 0.0
    if not intr:
        ym = 0.0
        for i in range(n):
            y[i] = y[i] * vsq
        ys = math.sqrt(esumsq(y, 0, n))
        for i in range(n):
            y[i] = y[i] / ys
        for j in range(p):
            if not ju[j]:
                continue
            xm[j] = 0.0
            for i in range(n):
                x[j * n + i] = x[j * n + i] * vsq
            xv[j] = esumsq(x, j * n, n)
            if isd:
                xbq = edot_c(x, j * n, vsq, n)
                xbq = xbq * xbq
                vc = xv[j] - xbq
                xs[j] = math.sqrt(vc)
                for i in range(n):
                    x[j * n + i] = x[j * n + i] / xs[j]
                xv[j] = 1.0 + xbq / vc
            else:
                xs[j] = 1.0
    else:
        for j in range(p):
            if not ju[j]:
                continue
            s = edot_c(x, j * n, w, n)
            xm[j] = s
            for i in range(n):
                x[j * n + i] = vsq * (x[j * n + i] - s)
            xv[j] = esumsq(x, j * n, n)
            if isd:
                xs[j] = math.sqrt(xv[j])
        if not isd:
            for j in range(p):
                xs[j] = 1.0
        else:
            for j in range(p):
                if not ju[j]:
                    continue
                for i in range(n):
                    x[j * n + i] = x[j * n + i] / xs[j]
            for j in range(p):
                xv[j] = 1.0
        ym = edot_c(y, 0, w, n)
        for i in range(n):
            y[i] = vsq * (y[i] - ym)
        ys = math.sqrt(esumsq(y, 0, n))
        for i in range(n):
            y[i] = y[i] / ys
    dbl_out[0] = ym
    dbl_out[1] = ys

    # Gradient (covariance mode: signed g; naive mode: |x_j . r|).
    for j in range(p):
        g[j] = 0.0
        a[j] = 0.0
        mm[j] = 0
        ix[j] = 0
    for j in range(p):
        if not ju[j]:
            continue
        s = edot(x, j * n, y, 0, n)
        g[j] = math.fabs(s) if naive else s

    # Box constraints: glmnet passes +-big, divided by ys and times xs.
    # cl(0,j) = -big / ys * xs(j), cl(1,j) = big / ys * xs(j).

    # ---- path -----------------------------------------------------------
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
        # initialize_point
        alm0 = lmda_curr
        alm = alm0
        if flmin >= 1.0:
            alm = ulam[m] / ys
        elif m > 1:
            alm = alm * alf
        elif m == 0:
            alm = big
        else:
            alm0 = 0.0
            for j in range(p):
                if ju[j] == 0 or vq[j] <= 0.0:
                    continue
                gj = math.fabs(g[j]) / vq[j]
                if gj > alm0:
                    alm0 = gj
            bb = beta if beta > 1e-3 else 1e-3
            alm0 = alm0 / bb
            alm = alm0 * alf
        lmda_curr = alm
        dem = alm * omb
        ab = alm * beta

        # ---- point fit: initialize ----
        rsq_prev = rsq
        if naive:
            tlam = beta * (2.0 * alm - alm0)
            for k in range(p):
                if ix[k] or not ju[k]:
                    continue
                if g[k] > tlam * vq[k]:
                    ix[k] = 1

        maxit_hit = False
        maxact_hit = False

        # Two phases repeat: a partial fit over the active set (skipped on the
        # first lambda that never warmed), then full passes until converged
        # (and, in naive mode, until the KKT check adds no strong variable).
        do_partial = iz
        while True:
            if do_partial:
                # ---- partial_fit ----
                if not naive:
                    for l in range(nin):
                        da[l] = a[ia[l]]
                iz = True
                while True:
                    nlp += 1
                    dlx = 0.0
                    for l in range(nin):
                        k = ia[l]
                        old = a[k]
                        gk = 0.0
                        if naive:
                            gk = edot(y, 0, x, k * n, n)
                        else:
                            gk = g[k]
                        u = gk + old * xv[k]
                        v = math.fabs(u) - vq[k] * ab
                        new = 0.0
                        if v > 0.0:
                            lo = -big / ys * xs[k]
                            hi = big / ys * xs[k]
                            t = math.copysign(v, u) / (xv[k] + vq[k] * dem)
                            if t > hi:
                                t = hi
                            if t < lo:
                                t = lo
                            new = t
                        a[k] = new
                        if old == new:
                            continue
                        diff = new - old
                        d2 = xv[k] * diff * diff
                        if d2 > dlx:
                            dlx = d2
                        rsq += diff * (2.0 * gk - diff * xv[k])
                        if naive:
                            for i in range(n):
                                y[i] -= diff * x[k * n + i]
                        else:
                            ck = mm[k] - 1
                            for l2 in range(nin):
                                j = ia[l2]
                                g[j] -= cgram[ck * p + j] * diff
                    if dlx < thr:
                        break
                    if nlp > maxit:
                        maxit_hit = True
                        break
                if maxit_hit:
                    break
                if not naive:
                    for l in range(nin):
                        da[l] -= a[ia[l]]
                    for j in range(p):
                        if mm[j] != 0 or not ju[j]:
                            continue
                        s = 0.0
                        for l in range(nin):
                            s += da[l] * cgram[l * p + j]
                        g[j] += s

            # ---- initial_fit: full passes ----
            done = False
            while True:
                if naive and nlp > maxit:
                    maxit_hit = True
                    break
                nlp += 1
                dlx = 0.0
                for k in range(p):
                    if naive:
                        if not ix[k]:
                            continue
                    else:
                        if not ju[k]:
                            continue
                    old = a[k]
                    gk = 0.0
                    if naive:
                        gk = edot(y, 0, x, k * n, n)
                    else:
                        gk = g[k]
                    u = gk + old * xv[k]
                    v = math.fabs(u) - vq[k] * ab
                    new = 0.0
                    if v > 0.0:
                        lo = -big / ys * xs[k]
                        hi = big / ys * xs[k]
                        t = math.copysign(v, u) / (xv[k] + vq[k] * dem)
                        if t > hi:
                            t = hi
                        if t < lo:
                            t = lo
                        new = t
                    a[k] = new
                    if old == new:
                        continue
                    if mm[k] == 0:
                        # update_active
                        nin += 1
                        if nin > nx:
                            maxact_hit = True
                            break
                        mm[k] = nin
                        ia[nin - 1] = k
                        if not naive:
                            col = nin - 1
                            for j in range(p):
                                if not ju[j]:
                                    continue
                                if j == k:
                                    cgram[col * p + j] = xv[j]
                                elif mm[j] != 0:
                                    cgram[col * p + j] = cgram[(mm[j] - 1) * p + k]
                                else:
                                    cgram[col * p + j] = edot(x, j * n, x, k * n, n)
                    diff = new - old
                    d2 = xv[k] * diff * diff
                    if d2 > dlx:
                        dlx = d2
                    rsq += diff * (2.0 * gk - diff * xv[k])
                    if naive:
                        for i in range(n):
                            y[i] -= diff * x[k * n + i]
                    else:
                        ck = mm[k] - 1
                        for j in range(p):
                            if ju[j]:
                                g[j] -= cgram[ck * p + j] * diff
                if maxact_hit:
                    break
                if dlx < thr:
                    if not naive:
                        done = True
                        break
                    # naive check_kkt: refresh |grad| off the strong set and
                    # add violators; converged only if none were added.
                    for k in range(p):
                        if ix[k] or not ju[k]:
                            continue
                        g[k] = math.fabs(edot(y, 0, x, k * n, n))
                    updated = False
                    for k in range(p):
                        if ix[k] or not ju[k]:
                            continue
                        if g[k] > ab * vq[k]:
                            ix[k] = 1
                            updated = True
                    if not updated:
                        done = True
                        break
                    continue
                if nlp > maxit:
                    maxit_hit = True
                    break
                break  # not converged: fall back to a partial fit
            if maxit_hit or maxact_hit or done:
                break
            do_partial = True

        if maxit_hit:
            jerr = -m - 1
            break
        if maxact_hit:
            jerr = -10001 - m
            break

        # ---- process_point_fit ----
        me = 0
        for l in range(nin):
            val = a[ia[l]]
            ca[m * nx + l] = val
            if val != 0.0:
                me += 1
        rsqo[m] = rsq
        kin[m] = nin
        almo[m] = alm
        lmu = m + 1
        if rsq == 0.0:
            prop = math.inf
        else:
            prop = (rsq - rsq_prev) / rsq
        if lmu < mnl or flmin >= 1.0:
            continue
        if me > ne or prop < sml or rsq > rsqmax:
            break

    # ---- unstandardize (ElnetDriver::fit) ----
    if jerr <= 0:
        for k in range(lmu):
            almo[k] *= ys
            nk = kin[k]
            for l in range(nk):
                ca[k * nx + l] *= ys / xs[ia[l]]
            s = 0.0
            if intr:
                for l in range(nk):
                    s -= ca[k * nx + l] * xm[ia[l]]
                s += ym
            a0[k] = s
    ints_out[0] = lmu
    ints_out[1] = nlp
    ints_out[2] = jerr
