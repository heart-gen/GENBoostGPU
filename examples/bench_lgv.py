"""Benchmark the Module 02 nested elastic net on CPU and GPU.

    python examples/bench_lgv.py --n 153 --p 2500 --loci 32 [--device cpu,gpu] [--threads 8]

Builds synthetic loci (AR(1) LD, MAF 0.05-0.5, 1% missing calls, one planted
sparse signal), prepares every glmnet problem of the locked cross-fit and
times the batched solve. Prints loci/s and glmnet paths/s per device, so the
cheapest allocation (GPU batch size, CPU threads, or no GPU) can be chosen for
a cohort. GEMMA BSLMM is not included: it runs on CPU cores alongside.
"""
import argparse
import time

import numpy as np

from genboostgpu.lgv.crossfit import CrossfitSettings, finalize_crossfit, prepare_crossfit
from genboostgpu.lgv.glmnet import fit_paths


def synthetic_locus(rng, n, p):
    lat = rng.normal(size=(n, p))
    for j in range(1, p):
        lat[:, j] = 0.7 * lat[:, j - 1] + np.sqrt(0.51) * lat[:, j]
    maf = rng.uniform(0.05, 0.5, p)
    q = 1 - maf
    from scipy.stats import norm
    t0, t1 = norm.ppf(q ** 2), norm.ppf(q ** 2 + 2 * maf * q)
    g = np.where(lat <= t0, 0.0, np.where(lat <= t1, 1.0, 2.0))
    g[rng.random(g.shape) < 0.01] = np.nan
    gi = np.where(np.isnan(g), np.nanmean(g, axis=0), g)
    beta = np.zeros(p)
    beta[rng.choice(p, 5, replace=False)] = rng.normal(0, 0.5, 5)
    cov = np.column_stack([rng.uniform(20, 80, n), rng.integers(0, 2, n), rng.integers(0, 2, n)])
    y = gi @ beta + 0.01 * cov[:, 0] + rng.normal(size=n)
    return g, y, cov


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=153)
    ap.add_argument("--p", type=int, default=2500)
    ap.add_argument("--loci", type=int, default=16)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    preps = [prepare_crossfit(*synthetic_locus(rng, a.n, a.p), CrossfitSettings(), seed=1000 + i)
             for i in range(a.loci)]
    problems = [q for p in preps for q in p.problems]
    for dev in a.device.split(","):
        fit_paths(problems[:4], device=dev, threads=a.threads)  # compile / warm up
        t = time.time()
        fits = fit_paths(problems, device=dev, threads=a.threads)
        el = time.time() - t
        k = 0
        r2 = []
        for p in preps:
            m = len(p.problems)
            r2.append(finalize_crossfit(p, fits[k:k + m])["metrics"]["rho2_oof"])
            k += m
        print(f"{dev:4s} threads={a.threads}: {a.loci} loci, {len(problems)} paths in {el:.2f}s "
              f"-> {a.loci / el:.2f} loci/s, {len(problems) / el:.0f} paths/s "
              f"(median rho2_oof {np.median(r2):.3f})")


if __name__ == "__main__":
    main()
