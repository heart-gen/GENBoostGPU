"""GEMMA BSLMM for the joint-model ``bslmm_pve`` feature.

Ports ``bslmm_pilot_functions.R::{residualize_phenotype, write_bimbam_inputs,
fit_bslmm_pve}``. The BIMBAM files are written byte-identically to R's
(``%.6f`` dosages, ``%.10g`` phenotype, ``snp_{j}`` names, ``snp,pos,chr``
annotation) and GEMMA receives the same arguments and seed, so the MCMC is
the same computation as in the R pipeline.

BSLMM is CPU-bound. :class:`BslmmPool` runs it in a process pool so a GPU
worker can keep computing EN/HE features while GEMMA chains run on the
node's otherwise idle cores.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from .rstats import r_lm_fit

__all__ = ["residualize_phenotype", "write_bimbam_inputs", "fit_bslmm_pve",
           "BslmmPool", "BslmmSettings", "file_sha256", "mean_impute_for_bslmm"]


class BslmmSettings:
    """Locked GEMMA settings (``joint-pve-20260820.tsv``)."""

    def __init__(self, gemma_bin="/projects/p32505/opt/bin/gemma", bslmm_mode=1,
                 burn_in=10000, sampling=100000, rpace=10, threads=1):
        self.gemma_bin = gemma_bin
        self.bslmm_mode = int(bslmm_mode)
        self.burn_in = int(burn_in)
        self.sampling = int(sampling)
        self.rpace = int(rpace)
        self.threads = int(threads)

    def as_dict(self):
        return dict(gemma_bin=self.gemma_bin, bslmm_mode=self.bslmm_mode,
                    burn_in=self.burn_in, sampling=self.sampling, rpace=self.rpace,
                    threads=self.threads)


def residualize_phenotype(phenotype, covariates=None):
    """``lm.fit(cbind(1, covariates), y)$residuals`` (or centering)."""
    y = np.asarray(phenotype, dtype=np.float64)
    if covariates is None or np.asarray(covariates).size == 0:
        return y - y.mean()
    design = np.column_stack([np.ones(y.size), np.asarray(covariates, np.float64)])
    _, resid, _ = r_lm_fit(design, y)
    return resid


def mean_impute_for_bslmm(genotype):
    """Stage 01's full-data mean imputation for GEMMA (no NaN-mean -> 0 rule:
    QC'd variants always have observed calls)."""
    g = np.array(genotype, dtype=np.float64, copy=True)
    if np.isnan(g).any():
        means = np.nanmean(g, axis=0)
        r, c = np.nonzero(np.isnan(g))
        g[r, c] = means[c]
    return g


def write_bimbam_inputs(outdir, genotype, phenotype, prefix="locus", snp_names=None):
    os.makedirs(outdir, exist_ok=True)
    geno_path = os.path.join(outdir, f"{prefix}.geno.txt")
    pheno_path = os.path.join(outdir, f"{prefix}.pheno.txt")
    anno_path = os.path.join(outdir, f"{prefix}.anno.txt")
    p = genotype.shape[1]
    if snp_names is None:
        snp_names = [f"snp_{j}" for j in range(1, p + 1)]
    with open(geno_path, "w") as fh:
        for j in range(p):
            doses = ",".join("%.6f" % v for v in genotype[:, j])
            fh.write(f"{snp_names[j]},A,T,{doses}\n")
    with open(pheno_path, "w") as fh:
        for v in phenotype:
            fh.write("%.10g\n" % v)
    with open(anno_path, "w") as fh:
        for j in range(p):
            fh.write(f"{snp_names[j]},{j + 1},1\n")
    return dict(geno=geno_path, pheno=pheno_path, anno=anno_path)


def _failure(status, elapsed, hyp_path, error, n_mcmc=0):
    return dict(converged=False, pve_mean=np.nan, pve_median=np.nan, pve_q025=np.nan,
                pve_q975=np.nan, h_mean=np.nan, n_mcmc=int(n_mcmc),
                exit_status=status, elapsed_sec=elapsed, hyp_path=hyp_path, error=error)


def fit_bslmm_pve(genotype, phenotype, work_dir, settings: BslmmSettings, seed: int,
                  keep_work: bool = False) -> dict:
    """Run GEMMA ``-bslmm`` and summarize the PVE chain (R-identical)."""
    if not os.path.exists(settings.gemma_bin):
        raise FileNotFoundError(f"GEMMA binary not found: {settings.gemma_bin}")
    os.makedirs(work_dir, exist_ok=True)
    inputs = write_bimbam_inputs(work_dir, genotype, phenotype, prefix="locus")
    prefix = "bslmm"
    args = [settings.gemma_bin, "-g", inputs["geno"], "-p", inputs["pheno"],
            "-a", inputs["anno"], "-bslmm", str(settings.bslmm_mode),
            "-w", str(settings.burn_in), "-s", str(settings.sampling),
            "-rpace", str(settings.rpace), "-seed", str(int(seed)),
            "-outdir", work_dir, "-o", prefix]
    log_path = os.path.join(work_dir, "gemma.log")
    # One BLAS thread per chain, as in the R pipeline's 1-CPU array tasks:
    # parallelism comes from running chains side by side, and oversubscribed
    # BLAS threads make each chain many times slower.
    env = dict(os.environ)
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "GOTO_NUM_THREADS", "BLIS_NUM_THREADS"):
        env[var] = str(settings.threads)
    started = time.time()
    with open(log_path, "w") as log:
        status = subprocess.call(args, stdout=log, stderr=subprocess.STDOUT,
                                 cwd=work_dir, env=env)
    elapsed = time.time() - started
    hyp_path = os.path.join(work_dir, prefix + ".hyp.txt")
    try:
        if status != 0 or not os.path.exists(hyp_path):
            err = (open(log_path).read().replace("\n", " | ")
                   if os.path.exists(log_path) else "GEMMA failed without a log")
            return _failure(status, elapsed, hyp_path, err)
        hyp = pd.read_csv(hyp_path, sep=r"\s+", float_precision="round_trip")
        hyp.columns = [
            "".join(ch for ch in c if ch.isalnum() or ch == "_").lower()
            for c in hyp.columns
        ]
        if "pve" not in hyp.columns:
            return _failure(status, elapsed, hyp_path, "hyp.txt lacks a pve column",
                            n_mcmc=len(hyp))
        pve = pd.to_numeric(hyp["pve"], errors="coerce").to_numpy(dtype=float)
        pve = pve[np.isfinite(pve)]
        h = (pd.to_numeric(hyp["h"], errors="coerce").to_numpy(dtype=float)
             if "h" in hyp.columns else np.array([]))
        h = h[np.isfinite(h)]
        return dict(
            converged=pve.size > 0,
            pve_mean=float(pve.mean()) if pve.size else np.nan,
            pve_median=float(np.median(pve)) if pve.size else np.nan,
            pve_q025=float(np.quantile(pve, 0.025)) if pve.size else np.nan,
            pve_q975=float(np.quantile(pve, 0.975)) if pve.size else np.nan,
            h_mean=float(h.mean()) if h.size else np.nan,
            n_mcmc=int(pve.size), exit_status=int(status), elapsed_sec=elapsed,
            hyp_path=hyp_path, error=None)
    finally:
        if not keep_work:
            shutil.rmtree(work_dir, ignore_errors=True)


def _bslmm_job(args):
    genotype, residual_y, work_dir, settings_dict, seed, keep_work = args
    try:
        return fit_bslmm_pve(genotype, residual_y, work_dir,
                             BslmmSettings(**settings_dict), seed, keep_work)
    except Exception as err:  # reported as a computational failure row
        return _failure(-1, 0.0, None, f"{type(err).__name__}: {err}")


class BslmmPool:
    """Process pool for GEMMA chains (``max_workers`` CPU processes)."""

    def __init__(self, max_workers: int, settings: BslmmSettings):
        self.settings = settings
        self._pool = ProcessPoolExecutor(max_workers=max(1, int(max_workers)))

    def submit(self, genotype, residual_y, work_dir, seed, keep_work=False):
        return self._pool.submit(
            _bslmm_job, (genotype, residual_y, work_dir, self.settings.as_dict(),
                         int(seed), keep_work))

    def shutdown(self):
        self._pool.shutdown(wait=True)


def file_sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
