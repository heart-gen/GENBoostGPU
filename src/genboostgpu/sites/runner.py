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
batch (one CUDA launch on the GPU). Batches are solved and their units'
cross-fits finished on background threads while the main thread prepares
the next batch.

GEMMA BSLMM runs inline on the shard's spare CPU cores (``bslmm_mode=
"inline"``), or in a CPU-only job (``"separate"``, ``genboostgpu sites
bslmm``) that may run on another cluster; combine joins its rows after
checking that both jobs come from identically initialized runs and saw the
same inputs.

Rows use the Stage 01 schema (``vmr_id`` holds the unit id) plus
``window_block_id``, ``window_start``, ``window_end``; ``genboostgpu lgv
combine`` reconciles and, when a frozen model and support are configured and
the units fall inside the characterized domain, scores them.
"""
from __future__ import annotations

import functools
import glob
import hashlib
import os
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

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
    finalize_rows, load_run, run_key, shard_tasks, write_run,
)

__all__ = ["SITE_EXTRA_COLUMNS", "site_tasks", "init_sites_run", "run_sites_shard",
           "run_sites_bslmm_shard"]

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
    separate = cfg.bslmm_mode == "separate"
    # The input digest lets combine check that the GEMMA job saw the same data.
    columns = ROW_COLUMNS + SITE_EXTRA_COLUMNS + (["input_digest"] if separate else [])
    started = time.time()
    out_rows, en_queue, bs_pending = [], [], []
    # Batches in flight, oldest first: (rows, future of the solve). A solve
    # returns one future per unit, finished on the finisher thread. The GPU
    # starts the next batch while the previous one is being finished; at most
    # two batches are in flight.
    en_inflight = []
    solver = ThreadPoolExecutor(1)
    # One thread: finishing a unit is mostly Python, so more threads only
    # contend for the GIL (measured slower at 4 and 16).
    finisher = ThreadPoolExecutor(1)

    def finalize_one(prep, fits):
        try:
            return finalize_crossfit(prep, fits)["metrics"]
        except Exception as err:
            return err

    def solve(batch):
        problems = [q for _, p in batch for q in p.problems]
        fits = fit_paths(problems, device=dev, threads=cpu_threads)
        futures, k = [], 0
        for _, prep in batch:
            m = len(prep.problems)
            futures.append(finisher.submit(finalize_one, prep, fits[k:k + m]))
            k += m
        return futures         # each unit's fits are freed once it is finished

    def finished(entry):
        return entry[1].done() and all(f.done() for f in entry[1].result())

    def harvest_oldest():
        rows, future = en_inflight.pop(0)
        for row, unit in zip(rows, future.result()):
            en = unit.result()
            if isinstance(en, Exception):
                if is_environment_error(en):
                    raise en
                row.update(en_converged=False, feature_error=f"EN: {en}")
            else:
                row.update(rho2_oof=en["rho2_oof"], r2_oof=en["r2_oof"],
                           covariance_ratio_oof=en["covariance_ratio_oof"],
                           score_variance_ratio_oof=en["score_variance_ratio_oof"],
                           en_converged=True)

    def flush_en(force=False):
        nonlocal en_queue
        while en_inflight and finished(en_inflight[0]):
            harvest_oldest()
        if en_queue and (force or sum(len(p.problems) for _, p in en_queue)
                         >= batch_units * 180):
            while len(en_inflight) >= 2:
                harvest_oldest()
            en_inflight.append(([r for r, _ in en_queue], solver.submit(solve, en_queue)))
            en_queue = []
        while force and en_inflight:
            harvest_oldest()

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
        for rows, _ in en_inflight:
            pending_ids |= {id(r) for r in rows}
        ready = [r for r in out_rows if id(r) not in pending_ids]
        if ready and (force or len(ready) >= 256):
            frame = pd.DataFrame(ready)
            for col in columns:
                if col not in frame.columns:
                    frame[col] = None
            frame = _finalize_site_rows(frame, cfg.bslmm_mode, features)
            _write_part(run_dir, "task_rows", shard, frame[columns].to_dict("records"))
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
                           pool, en_queue, bs_pending, run_dir, digest=separate)
        except Exception as err:
            if is_environment_error(err):
                raise
            for r in rows:
                r.update(terminal_status="computational_failure", computational_failure=True,
                         feature_error=f"{type(err).__name__}: {err}")
        emit()
    emit(force=True)
    solver.shutdown()
    finisher.shutdown()
    if pool is not None:
        pool.shutdown()
    log(f"[sites shard {shard}] done in {time.time() - started:.1f}s")
    return dict(units=len(mine), wall_sec=time.time() - started)


def _load_block(block, src, cov, geno_src, store):
    """Window, donors, variant QC and phenotypes shared by a block's units.

    Returns ``(info, data)``: ``info`` holds the fields every unit row of the
    block gets; ``data`` is ``None`` when the block stops here (``info`` then
    carries its terminal status), else ``(G, design, Y, sample_ids)``.
    """
    chrom = str(block["chrom"].iloc[0]).removeprefix("chr")
    if chrom.upper() in ("X", "Y"):
        return dict(terminal_status="excluded", exclusion_reason="non_autosomal_vmr"), None
    w_start = max(1, int(block["start"].min()) - int(src["window_bp"]))
    w_end = int(block["end"].max()) + int(src["window_bp"])
    info = dict(window_start=w_start, window_end=w_end)
    geno, _ = geno_src.window(chrom, w_start, w_end)
    if geno.shape[1] == 0:
        info.update(terminal_status="qc_failed",
                    exclusion_reason="no_snp_in_prespecified_cis_window")
        return info, None
    beta, samples = store.load(chrom)
    gid = geno_src.samples(chrom)[src.get("genotype_id_column", "IID")].astype(str).to_numpy()
    # Donor base set: phenotype store order, genotyped, complete covariates.
    probe = pd.DataFrame({"sample_id": samples, "phenotype": 0.0})
    design, _, meta = cov.design_for(probe, gid)
    store_pos = pd.Series(np.arange(len(samples)), index=samples)
    srow = store_pos.loc[meta["sample_id"].to_numpy()].to_numpy()
    G = geno[meta["_geno_row"].to_numpy()]
    info["snps_in_window"] = int(G.shape[1])
    keep = snp_qc_mask(G, src["maf_min"], src["missing_max"])
    if int(keep.sum()) < int(src["min_cis_variants"]):
        info.update(terminal_status="qc_failed", exclusion_reason="fewer_than_min_cis_variants")
        return info, None
    Y = beta[np.ix_(srow, block["unit_index"].to_numpy())]
    return info, (G[:, keep], design, Y, meta["sample_id"].astype(str).to_numpy())


def _unit_inputs(G, design, Y, complete, j):
    """Genotypes, covariates and phenotype of unit ``j`` (its own donor subset
    when it has missing values)."""
    y = Y[:, j]
    if complete[j]:
        return G, design, y, None
    ok = np.isfinite(y)
    return G[ok], design[ok], y[ok], ok


def _block_digest(G, design, sample_ids) -> bytes:
    h = hashlib.sha256()          # fastest of hashlib's digests here (SHA extensions)
    for a in (np.ascontiguousarray(G, np.float64), np.ascontiguousarray(design, np.float64)):
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    h.update("\t".join(sample_ids).encode())
    return h.digest()


def _unit_digest(block_digest: bytes, ok, y) -> str:
    """Digest of one unit's GEMMA inputs (genotypes, covariates, donors and
    phenotype), so rows computed on two clusters are joined only when both
    saw the same data."""
    h = hashlib.sha256(block_digest)
    h.update(b"all" if ok is None else np.packbits(ok).tobytes())
    h.update(np.ascontiguousarray(y, np.float64).tobytes())
    return h.hexdigest()[:32]


def _bslmm_inputs(g_j, d_j, y_j):
    return mean_impute_for_bslmm(g_j), residualize_phenotype(y_j, d_j if d_j.size else None)


def _work_dir(run_dir, task_id):
    return os.path.join(run_dir, "work", f"unit-{int(task_id):09d}")


def _process_block(block, rows, src, features, cov, geno_src, store, settings, pool,
                   en_queue, bs_pending, run_dir, digest=False):
    info, data = _load_block(block, src, cov, geno_src, store)
    for r in rows:
        r.update(info)
    if data is None:
        return
    G, design, Y, sample_ids = data
    snps_in_window = info["snps_in_window"]
    complete = np.all(np.isfinite(Y), axis=0)
    base_geom = None
    if "geometry" in features and complete.any():
        base_geom = (effective_rank_genotype(G), adjacent_ld_metric(G))
    prep_base = he_prepare(G) if ("he" in features and complete.any()) else None
    he_vals = None
    if prep_base is not None:
        R = _residualize_scale(Y[:, complete], design)
        he_vals = he_from_residuals(prep_base, R)
    bd = _block_digest(G, design, sample_ids) if digest else None
    ci = 0
    for j, r in enumerate(rows):
        g_j, d_j, y_j, ok = _unit_inputs(G, design, Y, complete, j)
        r.update(snps_in_window=snps_in_window, num_snps=int(G.shape[1]),
                 n_variants=int(G.shape[1]), n=int(y_j.size), samples=int(y_j.size),
                 mean_methylation=float(np.mean(y_j)),
                 methylation_variance=float(np.var(y_j, ddof=1)),
                 plink_source=src["genotype_pattern"], phenotype_source=src["units_dir"])
        if digest:
            r["input_digest"] = _unit_digest(bd, ok, y_j)
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
            bs_pending.append((r, pool.submit(*_bslmm_inputs(g_j, d_j, y_j),
                                              _work_dir(run_dir, r["task_id"]),
                                              r["feature_seed"])))


def run_sites_bslmm_shard(run_dir, shard=0, n_shards=1, gemma_workers=None,
                          log=_log) -> dict:
    """``bslmm_mode="separate"``: GEMMA only, for a CPU-only job.

    Units are loaded, QC'd and residualized by the same code as ``sites run``
    with inline GEMMA, with the same seeds, so the BSLMM columns equal an
    inline run's. Each row carries the run key and a digest of the unit's
    inputs; ``lgv combine`` checks both against the feature rows before it
    joins them, which lets this job run on a different cluster than the GPU
    shards (see ``docs/user-guide/sites_and_cost.rst``).
    """
    cfg, tasks = load_run(run_dir)
    if cfg.bslmm_mode != "separate":
        raise ValueError(f"run {cfg.run_id} has bslmm_mode={cfg.bslmm_mode!r}; "
                         "sites bslmm needs a run initialized with --bslmm separate")
    src = cfg.source
    key = run_key(cfg, tasks)
    cov = CovariateSpec.from_describe(src["covariates"])
    geno_src = GenotypeSource(src["genotype_pattern"])
    store = _UnitStore(src["units_dir"])
    mine = shard_tasks(tasks, shard, n_shards)
    done = _done_task_ids(run_dir, "bslmm_rows", shard)
    mine = mine[~mine["task_id"].isin(done)]
    workers = gemma_workers or max(1, (os.cpu_count() or 2) - 1)
    pool = BslmmPool(workers, BslmmSettings(**cfg.bslmm))
    log(f"[sites bslmm shard {shard}/{n_shards}] {len(mine)} units in "
        f"{mine['window_block_id'].nunique()} window blocks; {workers} GEMMA workers")
    started = time.time()
    inflight, rows, chains = {}, [], 0

    def harvest(limit):
        # Waiting once the queue is full bounds the genotype copies held in memory.
        nonlocal rows
        while inflight:
            ready = [f for f in inflight if f.done()]
            if not ready:
                if len(inflight) <= limit:
                    break
                ready = list(wait(inflight, return_when=FIRST_COMPLETED).done)
            for fut in ready:
                r = inflight.pop(fut)
                _apply_bslmm(r, fut.result())
                r["bslmm_error"] = r.pop("_bslmm_error", None)
                rows.append(r)
        if len(rows) >= 256 or (limit == 0 and rows):
            _write_part(run_dir, "bslmm_rows", shard, rows)
            rows = []

    for _, block in mine.groupby("window_block_id", sort=False):
        submitted = set()
        try:
            info, data = _load_block(block, src, cov, geno_src, store)
            if data is None:
                continue
            G, design, Y, sample_ids = data
            complete = np.all(np.isfinite(Y), axis=0)
            bd = _block_digest(G, design, sample_ids)
            for j, (_, t) in enumerate(block.iterrows()):
                g_j, d_j, y_j, ok = _unit_inputs(G, design, Y, complete, j)
                tid = int(t["task_id"])
                seed = stable_seed(cfg.effective_seed_run_id, cfg.region, t["vmr_id"],
                                   "joint_features")
                fut = pool.submit(*_bslmm_inputs(g_j, d_j, y_j), _work_dir(run_dir, tid), seed)
                inflight[fut] = dict(task_id=tid, run_key=key,
                                     input_digest=_unit_digest(bd, ok, y_j))
                submitted.add(tid)
                chains += 1
                harvest(4 * workers)
        except Exception as err:
            if is_environment_error(err):
                raise
            rows.extend(dict(task_id=int(tid), run_key=key, bslmm_converged=False,
                             bslmm_error=f"{type(err).__name__}: {err}")
                        for tid in block["task_id"] if int(tid) not in submitted)
    harvest(0)
    pool.shutdown()
    log(f"[sites bslmm shard {shard}] {chains} chains in {time.time() - started:.1f}s")
    return dict(units=len(mine), chains=chains, wall_sec=time.time() - started)


def _finalize_site_rows(frame, bslmm_mode, features):
    if bslmm_mode == "separate" and set(features) >= {"geometry", "he", "en"}:
        # BSLMM arrives from the GEMMA job; terminal status is set at combine.
        return frame.drop(columns=[c for c in frame.columns if c.startswith("_")])
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
