"""Site-level (million-scale) Module 02 features for CpG/CpH units.

Units are the per-chromosome matrices written by ``scripts/build_regions.R
--step qc`` (``units/chr{c}.beta.h5`` + ``units/chr{c}.units.tsv.gz``): single
CpG sites, CpH tiles, or CpH sites from ``scripts/prepare_site_inputs.R``.

Sites are grouped into **window blocks**: with ``window_block_bp = B`` every
unit whose start falls in the same ``B``-bp bin shares one cis window,
``[min(start) - window_bp, max(end) + window_bp]``, recorded per row
(``window_start``, ``window_end``). ``B = 0`` gives each unit its exact own
window (Module 02 semantics; no sharing). Per block, genotype I/O, variant QC,
the GRM, ``p_eff`` and LD are computed once, and HE is one batched quadratic
form over every unit with complete data. Units with missing values use their
own donor subset. Elastic-net problems of many units are solved in one
batch (one CUDA launch on the GPU).

Rows use the Stage 01 schema (``vmr_id`` holds the unit id) plus
``window_block_id``, ``window_start``, ``window_end``; ``genboostgpu lgv
combine`` reconciles and, when a frozen model and support are configured and
the units fall inside the characterized domain, scores them.
"""
from __future__ import annotations

import functools
import glob
import os
import time
from dataclasses import asdict

import numpy as np
import pandas as pd

from ..backend import is_environment_error
from ..io.genotype import GenotypeSource, snp_qc_mask
from ..io.phenotype import CovariateSpec
from ..lgv.bslmm import BslmmPool, BslmmSettings, mean_impute_for_bslmm, residualize_phenotype
from ..lgv.crossfit import CrossfitSettings, finalize_crossfit, prepare_crossfit
from ..lgv.folds import stable_seed
from ..lgv.geometry import adjacent_ld_metric, effective_rank_genotype
from ..lgv.glmnet import fit_paths
from ..lgv.he import he_from_residuals, he_prepare
from ..lgv.rstats import r_lm_fit
from ..lgv.runner import (
    ROW_COLUMNS, RunConfig, _apply_bslmm, _blank_row, _done_task_ids, _write_part,
    finalize_rows, load_run, shard_tasks, write_run,
)

__all__ = ["SITE_EXTRA_COLUMNS", "site_tasks", "init_sites_run", "run_sites_shard"]

_log = functools.partial(print, flush=True)

SITE_EXTRA_COLUMNS = ["window_block_id", "window_start", "window_end"]


def site_tasks(units_dir: str, cohort: str, region: str, unit_set_id: str,
               window_block_bp: int, chroms=None) -> pd.DataFrame:
    """Task table over every unit (one row per site/tile) with block ids."""
    files = sorted(glob.glob(os.path.join(units_dir, "chr*.units.tsv.gz")))
    if chroms:
        want = {str(c).removeprefix("chr") for c in chroms}
        files = [f for f in files
                 if os.path.basename(f)[3:].split(".")[0] in want]
    if not files:
        raise FileNotFoundError(f"No units/chr*.units.tsv.gz under {units_dir}")
    parts = []
    for f in files:
        u = pd.read_csv(f, sep="\t", dtype={"chrom": str})
        u["unit_index"] = np.arange(len(u))
        parts.append(u)
    u = pd.concat(parts, ignore_index=True)
    if window_block_bp and window_block_bp > 0:
        block = (u["start"].to_numpy() - 1) // int(window_block_bp)
    else:
        block = np.arange(len(u))
    u["window_block_id"] = u["chrom"].astype(str) + ":" + pd.Series(block).astype(str)
    return pd.DataFrame(dict(
        task_id=np.arange(1, len(u) + 1), cohort=cohort, region=region,
        chrom=u["chrom"].astype(str), start=u["start"].astype(np.int64),
        end=u["end"].astype(np.int64), n_cpgs=u.get("n_sites", pd.Series(1, index=u.index)),
        vmr_id=u["unit_id"].astype(str), vmr_set_id=unit_set_id,
        unit_index=u["unit_index"], window_block_id=u["window_block_id"]))


def init_sites_run(run_dir, run_id, cohort, region, units_dir, genotype_pattern,
                   covariates: CovariateSpec, features=("geometry", "he", "en"),
                   window_bp=500_000, window_block_bp=0, maf_min=0.05, missing_max=0.05,
                   min_cis_variants=100, genotype_id_column="IID", unit_set_id="",
                   chroms=None, bslmm_mode="off", seed_run_id=None, joint_model_path=None,
                   joint_model_sha256=None, support_path=None, support_cell=None,
                   smoke_run=False, bslmm_settings=None):
    tasks = site_tasks(units_dir, cohort, region, unit_set_id, window_block_bp, chroms)
    source = dict(kind="sites", units_dir=os.path.realpath(units_dir),
                  genotype_pattern=genotype_pattern, covariates=covariates.describe(),
                  cohort=cohort, region=region, unit_set_id=unit_set_id,
                  window_bp=int(window_bp), window_block_bp=int(window_block_bp),
                  maf_min=float(maf_min), missing_max=float(missing_max),
                  min_cis_variants=int(min_cis_variants),
                  genotype_id_column=genotype_id_column, features=list(features))
    cfg = RunConfig(run_id=run_id, cohort=cohort, region=region, source=source,
                    seed_run_id=seed_run_id, bslmm_mode=bslmm_mode,
                    bslmm=bslmm_settings or BslmmSettings().as_dict(),
                    joint_model_path=joint_model_path,
                    joint_model_source_sha256=joint_model_sha256,
                    support_path=support_path, support_cell=support_cell or cohort,
                    smoke_run=smoke_run)
    write_run(run_dir, cfg, tasks)


class _UnitStore:
    """Lazy per-chromosome access to ``units/chr{c}.beta.h5``."""

    def __init__(self, units_dir):
        self.units_dir = units_dir
        self._chrom = None
        self._beta = None
        self._samples = None

    def load(self, chrom):
        import h5py

        c = str(chrom).removeprefix("chr")
        if c != self._chrom:
            with h5py.File(os.path.join(self.units_dir, f"chr{c}.beta.h5"), "r") as fh:
                self._beta = fh["beta"][()]           # samples x units
                self._samples = [s.decode() if isinstance(s, bytes) else str(s)
                                 for s in fh["sample_id"][()]]
            self._chrom = c
        return self._beta, self._samples


def _residualize_scale(Y, design):
    """``scale(lm.fit(cbind(1, design), y)$residuals)`` for every column of Y."""
    X = np.column_stack([np.ones(Y.shape[0]), design]) if design.size else \
        np.ones((Y.shape[0], 1))
    coef_mask = np.isfinite(r_lm_fit(X, Y[:, 0])[0])
    q, _ = np.linalg.qr(X[:, coef_mask])
    R = Y - q @ (q.T @ Y)
    R = R - R.mean(axis=0)
    return R / np.sqrt(np.sum(R * R, axis=0) / (R.shape[0] - 1))


def _finite_features(features):
    req = []
    if "geometry" in features:
        req += ["p_eff", "ld_metric"]
    if "he" in features:
        req += ["he_h2"]
    if "en" in features:
        req += ["rho2_oof", "r2_oof"]
    return req


def run_sites_shard(run_dir, shard=0, n_shards=1, device="auto", batch_units=64,
                    cpu_threads=1, gemma_workers=None, log=_log) -> dict:
    from ..backend import get_backend

    cfg, tasks = load_run(run_dir)
    src = cfg.source
    features = tuple(src.get("features", ("geometry", "he", "en")))
    backend = get_backend(device)
    dev = backend.name
    settings = CrossfitSettings(**{k: (tuple(v) if isinstance(v, list) else v)
                                   for k, v in cfg.crossfit.items()})
    cov = CovariateSpec.from_describe(src["covariates"])
    geno_src = GenotypeSource(src["genotype_pattern"])
    store = _UnitStore(src["units_dir"])
    mine = shard_tasks(tasks, shard, n_shards)
    done = _done_task_ids(run_dir, "task_rows", shard)
    mine = mine[~mine["task_id"].isin(done)]
    pool = BslmmPool(gemma_workers or max(1, (os.cpu_count() or 2) - 2),
                     BslmmSettings(**cfg.bslmm)) if cfg.bslmm_mode == "inline" else None
    log(f"[sites shard {shard}/{n_shards}] {len(mine)} units in "
        f"{mine['window_block_id'].nunique()} window blocks on {dev}; features={features}")
    started = time.time()
    out_rows, en_queue, bs_pending = [], [], []

    def flush_en(force=False):
        nonlocal en_queue
        if not en_queue or (not force and sum(len(p.problems) for _, p in en_queue)
                            < batch_units * 180):
            return
        problems = [q for _, p in en_queue for q in p.problems]
        fits = fit_paths(problems, device=dev, threads=cpu_threads)
        k = 0
        for row, prep in en_queue:
            m = len(prep.problems)
            try:
                en = finalize_crossfit(prep, fits[k:k + m])["metrics"]
                row.update(rho2_oof=en["rho2_oof"], r2_oof=en["r2_oof"],
                           covariance_ratio_oof=en["covariance_ratio_oof"],
                           score_variance_ratio_oof=en["score_variance_ratio_oof"],
                           en_converged=True)
            except Exception as err:
                if is_environment_error(err):
                    raise
                row.update(en_converged=False, feature_error=f"EN: {err}")
            k += m
        en_queue = []

    def emit(force=False):
        nonlocal out_rows, bs_pending
        flush_en(force)
        still = []
        for row, fut in bs_pending:
            if force or fut.done():
                _apply_bslmm(row, fut.result())
            else:
                still.append((row, fut))
        bs_pending = still
        pending_ids = {id(r) for r, _ in bs_pending} | {id(r) for r, _ in en_queue}
        ready = [r for r in out_rows if id(r) not in pending_ids]
        if ready and (force or len(ready) >= 256):
            frame = pd.DataFrame(ready)
            for col in ROW_COLUMNS + SITE_EXTRA_COLUMNS:
                if col not in frame.columns:
                    frame[col] = None
            frame = _finalize_site_rows(frame, cfg.bslmm_mode, features)
            _write_part(run_dir, "task_rows", shard,
                        frame[ROW_COLUMNS + SITE_EXTRA_COLUMNS].to_dict("records"))
            ready_ids = {id(r) for r in ready}
            out_rows = [r for r in out_rows if id(r) not in ready_ids]

    for block_id, block in mine.groupby("window_block_id", sort=False):
        rows = []
        for _, t in block.iterrows():
            seed = stable_seed(cfg.effective_seed_run_id, cfg.region, t["vmr_id"],
                               "joint_features")
            r = _blank_row(cfg, t, dict(src, estimation_group=cfg.cohort), seed)
            r["window_block_id"] = block_id
            rows.append(r)
        out_rows.extend(rows)
        try:
            _process_block(block, rows, src, features, cov, geno_src, store, settings,
                           pool, en_queue, bs_pending, run_dir)
        except Exception as err:
            if is_environment_error(err):
                raise
            for r in rows:
                r.update(terminal_status="computational_failure", computational_failure=True,
                         feature_error=f"{type(err).__name__}: {err}")
        emit()
    emit(force=True)
    if pool is not None:
        pool.shutdown()
    log(f"[sites shard {shard}] done in {time.time() - started:.1f}s")
    return dict(units=len(mine), wall_sec=time.time() - started)


def _process_block(block, rows, src, features, cov, geno_src, store, settings, pool,
                   en_queue, bs_pending, run_dir):
    chrom = str(block["chrom"].iloc[0]).removeprefix("chr")
    if chrom.upper() in ("X", "Y"):
        for r in rows:
            r.update(terminal_status="excluded", exclusion_reason="non_autosomal_vmr")
        return
    w_start = max(1, int(block["start"].min()) - int(src["window_bp"]))
    w_end = int(block["end"].max()) + int(src["window_bp"])
    for r in rows:
        r.update(window_start=w_start, window_end=w_end)
    geno, _ = geno_src.window(chrom, w_start, w_end)
    if geno.shape[1] == 0:
        for r in rows:
            r.update(terminal_status="qc_failed",
                     exclusion_reason="no_snp_in_prespecified_cis_window")
        return
    beta, samples = store.load(chrom)
    gid = geno_src.samples(chrom)[src.get("genotype_id_column", "IID")].astype(str).to_numpy()
    # Donor base set: phenotype store order, genotyped, complete covariates.
    probe = pd.DataFrame({"sample_id": samples, "phenotype": 0.0})
    design, _, meta = cov.design_for(probe, gid)
    store_pos = pd.Series(np.arange(len(samples)), index=samples)
    srow = store_pos.loc[meta["sample_id"].to_numpy()].to_numpy()
    G = geno[meta["_geno_row"].to_numpy()]
    snps_in_window = int(G.shape[1])
    keep = snp_qc_mask(G, src["maf_min"], src["missing_max"])
    if int(keep.sum()) < int(src["min_cis_variants"]):
        for r in rows:
            r.update(terminal_status="qc_failed", exclusion_reason="fewer_than_min_cis_variants",
                     snps_in_window=snps_in_window)
        return
    G = G[:, keep]
    Y = beta[np.ix_(srow, block["unit_index"].to_numpy())]
    complete = np.all(np.isfinite(Y), axis=0)
    base_geom = None
    if "geometry" in features and complete.any():
        base_geom = (effective_rank_genotype(G), adjacent_ld_metric(G))
    prep_base = he_prepare(G) if ("he" in features and complete.any()) else None
    he_vals = None
    if prep_base is not None:
        R = _residualize_scale(Y[:, complete], design)
        he_vals = he_from_residuals(prep_base, R)
    ci = 0
    for j, r in enumerate(rows):
        y = Y[:, j]
        ok = np.isfinite(y)
        g_j, d_j = (G, design) if complete[j] else (G[ok], design[ok])
        y_j = y if complete[j] else y[ok]
        r.update(snps_in_window=snps_in_window, num_snps=int(G.shape[1]),
                 n_variants=int(G.shape[1]), n=int(y_j.size), samples=int(y_j.size),
                 mean_methylation=float(np.mean(y_j)),
                 methylation_variance=float(np.var(y_j, ddof=1)),
                 plink_source=src["genotype_pattern"], phenotype_source=src["units_dir"])
        if "geometry" in features:
            if complete[j]:
                r.update(p_eff=base_geom[0], ld_metric=base_geom[1])
            else:
                r.update(p_eff=effective_rank_genotype(g_j), ld_metric=adjacent_ld_metric(g_j))
        if "he" in features:
            if complete[j]:
                slope, se, _, p = he_vals
                r.update(he_h2=float(slope[ci]), he_se=float(se[ci]), he_pvalue=float(p[ci]),
                         he_converged=bool(np.isfinite(slope[ci]) and np.isfinite(se[ci])))
            else:
                prep = he_prepare(g_j)
                if prep is not None:
                    s1, se1, _, p1 = he_from_residuals(prep, _residualize_scale(
                        y_j[:, None], d_j)[:, 0])
                    r.update(he_h2=s1, he_se=se1, he_pvalue=p1,
                             he_converged=bool(np.isfinite(s1) and np.isfinite(se1)))
        if complete[j]:
            ci += 1
        r["terminal_status"] = "pending"
        if "en" in features:
            prep = prepare_crossfit(g_j, y_j, d_j if d_j.size else None, settings,
                                    seed=r["feature_seed"] + 17)
            en_queue.append((r, prep))
        if pool is not None:
            wd = os.path.join(run_dir, "work", f"unit-{int(r['task_id']):09d}")
            bs_pending.append((r, pool.submit(mean_impute_for_bslmm(g_j),
                                              residualize_phenotype(y_j, d_j if d_j.size else None),
                                              wd, r["feature_seed"])))


def _finalize_site_rows(frame, bslmm_mode, features):
    if bslmm_mode != "off" and set(features) >= {"geometry", "he", "en"}:
        return finalize_rows(frame, bslmm_mode)
    pending = frame["terminal_status"].astype(str) == "pending"
    if not pending.any():
        return frame.drop(columns=[c for c in frame.columns if c.startswith("_")])
    sub = frame.loc[pending]
    req = _finite_features(features)
    finite = np.isfinite(sub[req].apply(pd.to_numeric, errors="coerce")
                         .to_numpy(dtype=float)).all(axis=1) if req else np.ones(len(sub), bool)
    ok = finite.copy()
    if "he" in features:
        ok &= sub["he_converged"].fillna(False).astype(bool).to_numpy()
    if "en" in features:
        ok &= sub["en_converged"].fillna(False).astype(bool).to_numpy()
    frame.loc[pending, "feature_complete"] = False
    frame.loc[pending, "computational_failure"] = ~ok
    frame.loc[pending, "terminal_status"] = np.where(ok, "features_only", "computational_failure")
    frame.loc[pending, "feature_error"] = np.where(ok, None, "requested features incomplete")
    return frame.drop(columns=[c for c in frame.columns if c.startswith("_")])
