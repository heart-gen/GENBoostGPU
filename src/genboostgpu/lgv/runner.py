"""Module 02 feature runner: shards, batching, GEMMA overlap, terminal rows.

One process handles one shard of the task table (``--shard i/N``, one per
GPU or per CPU allocation). Within a shard:

1. a thread pool loads and QCs loci and builds every glmnet problem of the
   nested cross-fit (``prepare_crossfit``), plus HE and genotype geometry;
2. GEMMA BSLMM chains run in a CPU process pool (``bslmm_mode="inline"``),
   or are deferred to a CPU-only job (``"separate"``), or skipped (``"off"``);
3. the glmnet problems of ``batch_loci`` loci are solved in one call, a
   single CUDA launch on the GPU, while the next batch is being prepared.

Every task writes exactly one terminal row with the Stage 01 schema
(``completed``, ``qc_failed``, ``excluded`` or ``computational_failure``), in
append-only parquet parts, so a killed shard resumes where it stopped and the
combine step can reconcile the full task universe.
"""
from __future__ import annotations

import functools
import glob
import hashlib
import json
import os
import socket
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from ..backend import is_environment_error
from .bslmm import BslmmPool, BslmmSettings, mean_impute_for_bslmm, residualize_phenotype
from .crossfit import CrossfitSettings, finalize_crossfit, prepare_crossfit
from .folds import stable_seed
from .geometry import adjacent_ld_metric, effective_rank_genotype
from .glmnet import fit_paths
from .he import haseman_elston
from .tasks import source_from_config

__all__ = ["ROW_COLUMNS", "BSLMM_COLUMNS", "RunConfig", "write_run", "load_run",
           "run_shard", "run_bslmm_shard", "finalize_rows", "shard_tasks", "run_key"]

_log = functools.partial(print, flush=True)

ROW_COLUMNS = [
    "task_id", "cohort", "region", "population", "catalog_cohort", "estimation_group",
    "chrom", "start", "end", "vmr_id", "vmr_set_id", "upstream_vmr_run_id", "n_cpgs",
    "n", "samples", "num_snps", "n_variants", "snps_in_window", "p_eff", "ld_metric",
    "mean_methylation", "methylation_variance",
    "bslmm_pve", "bslmm_pve_median", "bslmm_pve_q025", "bslmm_pve_q975",
    "bslmm_h_mean", "bslmm_converged", "bslmm_exit_status", "bslmm_elapsed_sec",
    "bslmm_n_mcmc", "he_h2", "he_se", "he_pvalue", "he_converged",
    "rho2_oof", "r2_oof", "covariance_ratio_oof", "score_variance_ratio_oof",
    "en_converged", "feature_complete", "computational_failure", "terminal_status",
    "exclusion_reason", "feature_error", "feature_seed", "plink_source",
    "phenotype_source",
]
BSLMM_COLUMNS = ["bslmm_pve", "bslmm_pve_median", "bslmm_pve_q025", "bslmm_pve_q975",
                 "bslmm_h_mean", "bslmm_converged", "bslmm_exit_status",
                 "bslmm_elapsed_sec", "bslmm_n_mcmc"]


@dataclass
class RunConfig:
    """Everything that determines a run's numbers (written to ``run.json``)."""

    run_id: str
    cohort: str
    region: str
    source: dict
    seed_run_id: str | None = None
    crossfit: dict = field(default_factory=lambda: asdict(CrossfitSettings()))
    bslmm: dict = field(default_factory=lambda: BslmmSettings().as_dict())
    bslmm_mode: str = "inline"
    joint_model_path: str | None = None
    joint_model_source_sha256: str | None = None
    joint_model_run_id: str = ""
    support_path: str | None = None
    support_cell: str | None = None
    max_outside_calibration_domain: float = 0.10
    smoke_run: bool = False
    created_at: str = ""
    software: dict = field(default_factory=dict)
    notes: str = ""

    @property
    def effective_seed_run_id(self) -> str:
        return self.seed_run_id or self.run_id


def _software() -> dict:
    import platform

    import numba

    try:
        from importlib.metadata import version

        ver = version("genboostgpu")
    except Exception:
        ver = "unknown"
    git = None
    here = os.path.dirname(os.path.abspath(__file__))
    try:
        import subprocess

        git = subprocess.run(["git", "-C", here, "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=10).stdout.strip() or None
    except Exception:
        pass
    return dict(package="genboostgpu", version=ver, git_commit=git, python=platform.python_version(),
                numpy=np.__version__, pandas=pd.__version__, numba=numba.__version__,
                host=socket.gethostname())


def write_run(run_dir: str, config: RunConfig, tasks: pd.DataFrame) -> None:
    """Create a run directory (refuses to overwrite an existing run)."""
    if os.path.exists(os.path.join(run_dir, "run.json")):
        raise FileExistsError(f"Run already exists: {run_dir}")
    for sub in ("task_rows", "bslmm_rows", "results/combined", "logs"):
        os.makedirs(os.path.join(run_dir, sub), exist_ok=True)
    if not config.created_at:
        config.created_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    if not config.software:
        config.software = _software()
    tasks = tasks.sort_values("task_id").reset_index(drop=True)
    if tasks["task_id"].duplicated().any():
        raise ValueError("Duplicate task_id in task table")
    tasks.to_csv(os.path.join(run_dir, "tasks.tsv"), sep="\t", index=False)
    with open(os.path.join(run_dir, "run.json"), "w") as fh:
        json.dump(asdict(config), fh, indent=2, default=str)


def load_run(run_dir: str):
    with open(os.path.join(run_dir, "run.json")) as fh:
        cfg = RunConfig(**json.load(fh))
    tasks = pd.read_csv(os.path.join(run_dir, "tasks.tsv"), sep="\t",
                        dtype={"chrom": str, "vmr_id": str, "vmr_set_id": str})
    return cfg, tasks


# Source fields that name files; they differ between clusters and do not
# change the numbers, so the run key leaves them out.
_PATH_KEYS = frozenset({"units_dir", "genotype_pattern", "phenotype_path", "vmr_run_dir",
                        "path"})


def _without_paths(value):
    if isinstance(value, dict):
        return {k: _without_paths(v) for k, v in value.items() if k not in _PATH_KEYS}
    return value


def run_key(cfg: RunConfig, tasks: pd.DataFrame) -> str:
    """Digest of everything that determines a run's numbers except file paths.

    Two run directories initialized with the same arguments on different
    clusters (only the input paths differ) share a key, so BSLMM rows from a
    CPU-only job on one cluster can be joined with feature rows from the
    other. The task table, seeds, GEMMA settings (not the binary's path),
    cross-fit settings and source options all enter the key.
    """
    bslmm = {k: v for k, v in cfg.bslmm.items() if k != "gemma_bin"}
    t = tasks.sort_values("task_id")
    spec = dict(run_id=cfg.run_id, seed_run_id=cfg.effective_seed_run_id,
                cohort=cfg.cohort, region=cfg.region, bslmm_mode=cfg.bslmm_mode,
                bslmm=bslmm, crossfit=cfg.crossfit, source=_without_paths(cfg.source),
                tasks=[t[c].astype(str).tolist()
                       for c in ("task_id", "vmr_id", "chrom", "start", "end")])
    text = json.dumps(spec, sort_keys=True, default=str)
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def shard_tasks(tasks: pd.DataFrame, shard: int, n_shards: int) -> pd.DataFrame:
    """Contiguous blocks keep each shard on few chromosomes (genotype cache)."""
    if not 0 <= shard < n_shards:
        raise ValueError("shard must be in [0, n_shards)")
    blocks = np.array_split(np.arange(len(tasks)), n_shards)
    return tasks.iloc[blocks[shard]].reset_index(drop=True)


def _done_task_ids(run_dir: str, subdir: str, shard: int) -> set:
    done = set()
    for path in glob.glob(os.path.join(run_dir, subdir, f"part-{shard:05d}-*.parquet")):
        done.update(pd.read_parquet(path, columns=["task_id"])["task_id"].tolist())
    return done


def _write_part(run_dir: str, subdir: str, shard: int, rows: list) -> None:
    if not rows:
        return
    stamp = f"{int(time.time() * 1e6):x}"
    path = os.path.join(run_dir, subdir, f"part-{shard:05d}-{stamp}.parquet")
    tmp = path + ".tmp"
    pd.DataFrame(rows).to_parquet(tmp, index=False)
    os.replace(tmp, path)


def _blank_row(cfg: RunConfig, task, src: dict, seed: int) -> dict:
    row = {c: None for c in ROW_COLUMNS}
    est_group = src.get("estimation_group") or cfg.cohort
    row.update(
        task_id=int(task["task_id"]), cohort=cfg.cohort, region=cfg.region,
        population=est_group, catalog_cohort=src.get("catalog_cohort") or cfg.cohort,
        estimation_group=est_group, chrom=str(task["chrom"]), start=int(task["start"]),
        end=int(task["end"]), vmr_id=str(task["vmr_id"]),
        vmr_set_id=str(task.get("vmr_set_id", "")),
        upstream_vmr_run_id=src.get("upstream_vmr_run_id", ""),
        n_cpgs=None if pd.isna(task.get("n_cpgs")) else int(task["n_cpgs"]),
        bslmm_converged=False, he_converged=False, en_converged=False,
        feature_complete=False, computational_failure=False, feature_seed=int(seed))
    return row


def finalize_rows(rows: pd.DataFrame, bslmm_mode: str = "inline") -> pd.DataFrame:
    """Set ``feature_complete``/``terminal_status`` exactly as Stage 01 does.

    Rows that already carry a terminal status other than ``pending`` are
    left alone (qc_failed, excluded, or failures raised during estimation).
    With ``bslmm_mode="off"`` the joint model cannot be applied: rows whose
    other features are complete get ``terminal_status = "features_only"``,
    ``feature_complete = FALSE`` and no computational-failure flag.
    """
    rows = rows.copy()
    if "_bslmm_error" not in rows.columns and "bslmm_error" in rows.columns:
        rows = rows.rename(columns={"bslmm_error": "_bslmm_error"})
    pending = rows["terminal_status"].astype(str) == "pending"
    if not pending.any():
        return rows.drop(columns=[c for c in rows.columns if c.startswith("_")])
    sub = rows.loc[pending]
    if bslmm_mode == "off":
        req = sub[["he_h2", "rho2_oof", "r2_oof", "p_eff", "ld_metric"]]
        finite = np.isfinite(req.apply(pd.to_numeric, errors="coerce")
                             .to_numpy(dtype=float)).all(axis=1)
        ok = (finite & sub["he_converged"].fillna(False).astype(bool).to_numpy()
              & sub["en_converged"].fillna(False).astype(bool).to_numpy())
        rows.loc[pending, "feature_complete"] = False
        rows.loc[pending, "computational_failure"] = ~ok
        rows.loc[pending, "terminal_status"] = np.where(ok, "features_only",
                                                        "computational_failure")
        rows.loc[pending, "feature_error"] = np.where(
            ok, None, "HE/EN/geometry features incomplete")
        return rows.drop(columns=[c for c in rows.columns if c.startswith("_")])
    req = sub[["bslmm_pve", "he_h2", "rho2_oof", "r2_oof", "p_eff", "ld_metric"]]
    finite = np.isfinite(req.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)).all(axis=1)
    b_ok = sub["bslmm_converged"].fillna(False).astype(bool).to_numpy()
    h_ok = sub["he_converged"].fillna(False).astype(bool).to_numpy()
    e_ok = sub["en_converged"].fillna(False).astype(bool).to_numpy()
    complete = finite & b_ok & h_ok & e_ok
    errors = []
    for k, (_, r) in enumerate(sub.iterrows()):
        if complete[k]:
            errors.append(None)
            continue
        parts = []
        if not b_ok[k]:
            parts.append(f"BSLMM: {r.get('_bslmm_error') or 'not converged'}")
        if not h_ok[k]:
            parts.append("HE did not converge")
        if not e_ok[k]:
            parts.append("nested EN did not converge")
        if not finite[k]:
            parts.append("nonfinite joint feature")
        errors.append(" | ".join(parts))
    rows.loc[pending, "feature_complete"] = complete
    rows.loc[pending, "computational_failure"] = ~complete
    rows.loc[pending, "terminal_status"] = np.where(complete, "completed",
                                                    "computational_failure")
    rows.loc[pending, "feature_error"] = errors
    return rows.drop(columns=[c for c in rows.columns if c.startswith("_")])


@dataclass
class _Prepared:
    task: object
    row: dict
    prep: object = None
    locus: object = None
    error: str | None = None


def _prepare_one(cfg: RunConfig, source, task, settings: CrossfitSettings) -> _Prepared:
    src = cfg.source
    seed = stable_seed(cfg.effective_seed_run_id, cfg.region, task["vmr_id"], "joint_features")
    row = _blank_row(cfg, task, src, seed)
    try:
        locus = source.load(task)
        if locus.status != "ok":
            if locus.snps_in_window is not None:
                row["snps_in_window"] = int(locus.snps_in_window)
            row["terminal_status"] = locus.status
            row["exclusion_reason"] = locus.reason
            return _Prepared(task, row)
        g, y, cov = locus.genotype, locus.y, locus.covariates
        row.update(snps_in_window=int(locus.snps_in_window), num_snps=int(g.shape[1]),
                   n_variants=int(g.shape[1]), plink_source=locus.plink_source,
                   phenotype_source=locus.phenotype_source, n=int(g.shape[0]),
                   samples=int(g.shape[0]), mean_methylation=float(np.mean(y)),
                   methylation_variance=float(np.var(y, ddof=1)))
        prep = prepare_crossfit(g, y, cov, settings, seed=seed + 17)
        he = haseman_elston(g, y, cov)
        row.update(he_h2=he["he_h2"], he_se=he["he_se"], he_pvalue=he["he_pvalue"],
                   he_converged=bool(he["he_converged"]),
                   p_eff=effective_rank_genotype(g), ld_metric=adjacent_ld_metric(g))
        row["terminal_status"] = "pending"
        return _Prepared(task, row, prep, locus)
    except Exception as err:  # R: tryCatch(estimate_task(), error = ...)
        if is_environment_error(err):
            raise
        blank = _blank_row(cfg, task, src, seed)
        blank.update(terminal_status="computational_failure", computational_failure=True,
                     feature_error=f"{type(err).__name__}: {err}")
        return _Prepared(task, blank, error=traceback.format_exc())


def _apply_bslmm(row: dict, res: dict) -> None:
    row.update(bslmm_pve=res["pve_mean"], bslmm_pve_median=res["pve_median"],
               bslmm_pve_q025=res["pve_q025"], bslmm_pve_q975=res["pve_q975"],
               bslmm_h_mean=res["h_mean"], bslmm_converged=bool(res["converged"]),
               bslmm_exit_status=res["exit_status"], bslmm_elapsed_sec=res["elapsed_sec"],
               bslmm_n_mcmc=res["n_mcmc"], _bslmm_error=res.get("error"))


def run_shard(run_dir: str, shard: int = 0, n_shards: int = 1, device: str = "auto",
              batch_loci: int = 16, prep_threads: int = 4, cpu_threads: int = 1,
              gemma_workers: int | None = None, flush_every: int = 64,
              keep_bslmm_work: bool = False, log=_log) -> dict:
    """Compute feature rows for one shard (see module docstring)."""
    from ..backend import get_backend

    cfg, tasks = load_run(run_dir)
    backend = get_backend(device)
    dev = backend.name
    source = source_from_config(cfg.source, tasks)
    settings = CrossfitSettings(**{k: (tuple(v) if isinstance(v, list) else v)
                                   for k, v in cfg.crossfit.items()})
    mine = shard_tasks(tasks, shard, n_shards)
    done = _done_task_ids(run_dir, "task_rows", shard)
    todo = [r for _, r in mine.iterrows() if int(r["task_id"]) not in done]
    log(f"[shard {shard}/{n_shards}] {len(mine)} tasks, {len(done)} already done, "
        f"{len(todo)} to run on {dev}")
    if cfg.bslmm_mode == "inline":
        if gemma_workers is None:
            gemma_workers = max(1, (os.cpu_count() or 2) - prep_threads - 1)
        pool = BslmmPool(gemma_workers, BslmmSettings(**cfg.bslmm))
    else:
        pool = None
    work_root = os.path.join(run_dir, "work")
    pending_bslmm = {}  # task_id -> (row, future)
    out_rows = []
    stats = dict(tasks=len(todo), completed=0, failed=0, other=0, en_sec=0.0,
                 started=time.time())

    def flush(force=False):
        nonlocal out_rows
        if out_rows and (force or len(out_rows) >= flush_every):
            frame = pd.DataFrame(out_rows)
            for col in ROW_COLUMNS:
                if col not in frame.columns:
                    frame[col] = None
            if cfg.bslmm_mode == "separate":
                # BSLMM arrives from the CPU job; terminal status is set at combine.
                frame = frame.drop(columns=[c for c in frame.columns if c.startswith("_")])
            else:
                frame = finalize_rows(frame, cfg.bslmm_mode)
            for s in frame["terminal_status"]:
                key = "completed" if s == "completed" else (
                    "failed" if s == "computational_failure" else "other")
                stats[key] += 1
            _write_part(run_dir, "task_rows", shard, frame[ROW_COLUMNS].to_dict("records"))
            out_rows = []

    def drain_bslmm(block=False):
        for tid in list(pending_bslmm):
            row, fut = pending_bslmm[tid]
            if block or fut.done():
                _apply_bslmm(row, fut.result())
                out_rows.append(row)
                del pending_bslmm[tid]

    batches = [todo[i:i + batch_loci] for i in range(0, len(todo), batch_loci)]
    with ThreadPoolExecutor(max_workers=max(1, prep_threads)) as tp:
        def submit_batch(b):
            return [tp.submit(_prepare_one, cfg, source, t, settings) for t in b]

        next_futs = submit_batch(batches[0]) if batches else []
        for bi in range(len(batches)):
            prepared = [f.result() for f in next_futs]
            next_futs = submit_batch(batches[bi + 1]) if bi + 1 < len(batches) else []
            ready = [p for p in prepared if p.prep is not None]
            for p in prepared:
                if p.prep is None:
                    out_rows.append(p.row)
            # GEMMA chains start now so they overlap with the EN solve.
            futures = {}
            if pool is not None:
                for p in ready:
                    g = mean_impute_for_bslmm(p.locus.genotype)
                    ry = residualize_phenotype(p.locus.y, p.locus.covariates)
                    wd = os.path.join(work_root, f"vmr-{int(p.row['task_id']):07d}")
                    futures[p.row["task_id"]] = pool.submit(
                        g, ry, wd, p.row["feature_seed"], keep_bslmm_work)
            problems = [q for p in ready for q in p.prep.problems]
            t0 = time.time()
            fits = fit_paths(problems, device=dev, threads=cpu_threads) if problems else []
            stats["en_sec"] += time.time() - t0
            k = 0
            for p in ready:
                m = len(p.prep.problems)
                tid = p.row["task_id"]
                try:
                    en = finalize_crossfit(p.prep, fits[k:k + m])["metrics"]
                    p.row.update(rho2_oof=en["rho2_oof"], r2_oof=en["r2_oof"],
                                 covariance_ratio_oof=en["covariance_ratio_oof"],
                                 score_variance_ratio_oof=en["score_variance_ratio_oof"],
                                 en_converged=bool(en["converged"]))
                    ok = True
                except Exception as err:
                    if is_environment_error(err):
                        raise
                    blank = _blank_row(cfg, p.task, cfg.source, p.row["feature_seed"])
                    blank.update(terminal_status="computational_failure",
                                 computational_failure=True,
                                 feature_error=f"{type(err).__name__}: {err}")
                    p.row = blank
                    ok = False
                k += m
                if ok and tid in futures:
                    pending_bslmm[tid] = (p.row, futures[tid])
                else:
                    if tid in futures:
                        futures[tid].cancel()
                    out_rows.append(p.row)
            drain_bslmm(block=False)
            flush()
            elapsed = time.time() - stats["started"]
            log(f"[shard {shard}] batch {bi + 1}/{len(batches)}: {len(ready)} loci solved "
                f"({len(problems)} glmnet paths); EN {stats['en_sec']:.1f}s, "
                f"wall {elapsed:.1f}s, GEMMA pending {len(pending_bslmm)}")
    drain_bslmm(block=True)
    flush(force=True)
    if pool is not None:
        pool.shutdown()
    stats["wall_sec"] = time.time() - stats["started"]
    log(f"[shard {shard}] done: {stats}")
    return stats


def run_bslmm_shard(run_dir: str, shard: int = 0, n_shards: int = 1,
                    gemma_workers: int | None = None, log=_log) -> dict:
    """``bslmm_mode="separate"``: run only GEMMA for a shard (CPU-only job)."""
    cfg, tasks = load_run(run_dir)
    source = source_from_config(cfg.source, tasks)
    key = run_key(cfg, tasks)
    mine = shard_tasks(tasks, shard, n_shards)
    done = _done_task_ids(run_dir, "bslmm_rows", shard)
    gemma_workers = gemma_workers or max(1, (os.cpu_count() or 2) - 1)
    pool = BslmmPool(gemma_workers, BslmmSettings(**cfg.bslmm))
    futs = {}
    rows = []
    for _, task in mine.iterrows():
        tid = int(task["task_id"])
        if tid in done:
            continue
        seed = stable_seed(cfg.effective_seed_run_id, cfg.region, task["vmr_id"],
                           "joint_features")
        try:
            locus = source.load(task)
        except Exception as err:
            if is_environment_error(err):
                raise
            rows.append(dict(task_id=tid, run_key=key, bslmm_converged=False,
                             bslmm_error=f"{type(err).__name__}: {err}"))
            continue
        if locus.status != "ok":
            continue
        wd = os.path.join(run_dir, "work", f"vmr-{tid:07d}")
        futs[tid] = pool.submit(mean_impute_for_bslmm(locus.genotype),
                                residualize_phenotype(locus.y, locus.covariates), wd, seed)
    for tid, fut in futs.items():
        r = {"task_id": tid, "run_key": key}
        _apply_bslmm(r, fut.result())
        r["bslmm_error"] = r.pop("_bslmm_error", None)
        rows.append(r)
        if len(rows) >= 256:
            _write_part(run_dir, "bslmm_rows", shard, rows)
            rows = []
    _write_part(run_dir, "bslmm_rows", shard, rows)
    pool.shutdown()
    log(f"[bslmm shard {shard}] wrote {len(futs)} chains")
    return dict(chains=len(futs))
