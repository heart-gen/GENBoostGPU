"""Module 03 runner: shards of loci -> per-donor OOF predictions -> metrics.

Same run-directory contract as :mod:`genboostgpu.lgv.runner` (``run.json``,
``tasks.tsv``, append-only parquet parts, resumable shards), plus the
run-level ``donor-folds.tsv``. ``combine`` writes the R module's tables:
``oof-prediction-{cohort}-{region}-vmrs.tsv``, ``fold-repeat-diagnostics.tsv``,
``fold-diagnostics.tsv``, ``screen-bh-diagnostic.tsv``,
``predictions-per-donor.tsv`` and ``prediction-run-qc.tsv``.
"""
from __future__ import annotations

import functools
import glob
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict

import numpy as np
import pandas as pd

from ..backend import is_environment_error
from ..lgv.glmnet import fit_paths
from ..lgv.runner import _done_task_ids, _write_part, shard_tasks
from ..lgv.tasks import source_from_config
from .oof import LspSettings, donor_folds, finalize_locus_oof, locus_metrics, prepare_locus_oof

__all__ = ["init_lsp_run", "run_lsp_shard", "combine_lsp_run", "LEAKAGE_TRIPWIRE"]

_log = functools.partial(print, flush=True)

LEAKAGE_TRIPWIRE = 0.5  # config/prediction.yml leakage_tripwire.max_median_r2_pred_oof


def init_lsp_run(run_dir, run_id, cohort, region, source, tasks, settings=LspSettings(),
                 seed_run_id=None, folds=None, smoke_run=False, notes=""):
    if os.path.exists(os.path.join(run_dir, "run.json")):
        raise FileExistsError(f"Run already exists: {run_dir}")
    for sub in ("oof_rows", "status_rows", "results/combined", "logs"):
        os.makedirs(os.path.join(run_dir, sub), exist_ok=True)
    seed_run_id = seed_run_id or run_id
    if folds is None:
        folds = donor_folds(source.donors(), seed_run_id, region, settings)
    folds.to_csv(os.path.join(run_dir, "donor-folds.tsv"), sep="\t", index=False)
    tasks.sort_values("task_id").to_csv(os.path.join(run_dir, "tasks.tsv"), sep="\t",
                                        index=False)
    src = source.describe()
    src["apply_snp_qc"] = False
    cfg = dict(kind="lsp", run_id=run_id, seed_run_id=seed_run_id, cohort=cohort,
               region=region, source=src, settings=asdict(settings), smoke_run=smoke_run,
               created_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"), notes=notes)
    with open(os.path.join(run_dir, "run.json"), "w") as fh:
        json.dump(cfg, fh, indent=2, default=str)


def _load(run_dir):
    cfg = json.load(open(os.path.join(run_dir, "run.json")))
    tasks = pd.read_csv(os.path.join(run_dir, "tasks.tsv"), sep="\t",
                        dtype={"chrom": str, "vmr_id": str})
    folds = pd.read_csv(os.path.join(run_dir, "donor-folds.tsv"), sep="\t",
                        dtype={"donor": str})
    settings = LspSettings(**{k: (tuple(v) if isinstance(v, list) else v)
                              for k, v in cfg["settings"].items()})
    return cfg, tasks, folds, settings


def run_lsp_shard(run_dir, shard=0, n_shards=1, device="auto", batch_loci=8,
                  prep_threads=4, cpu_threads=1, log=_log):
    from ..backend import get_backend

    cfg, tasks, folds, settings = _load(run_dir)
    backend = get_backend(device)
    source = source_from_config(cfg["source"], tasks)
    mine = shard_tasks(tasks, shard, n_shards)
    done = _done_task_ids(run_dir, "status_rows", shard)
    todo = [r for _, r in mine.iterrows() if int(r["task_id"]) not in done]
    log(f"[lsp shard {shard}/{n_shards}] {len(todo)} loci to run on {backend.name}")

    def prepare(task):
        try:
            locus = source.load(task)
            if locus.status != "ok":
                return task, None, dict(status=locus.status, detail=locus.reason)
            prep = prepare_locus_oof(locus, folds, cfg["seed_run_id"], cfg["region"],
                                     str(task["vmr_id"]), settings, xp=backend.xp)
            return task, prep, None
        except Exception as err:
            if is_environment_error(err):
                raise
            return task, None, dict(status="failed", detail=f"{type(err).__name__}: {err}")

    started = time.time()
    with ThreadPoolExecutor(max_workers=max(1, prep_threads)) as tp:
        for b0 in range(0, len(todo), batch_loci):
            batch = list(tp.map(prepare, todo[b0:b0 + batch_loci]))
            ready = [(t, p) for t, p, s in batch if p is not None]
            problems = [q for _, p in ready for q in p.problems]
            fits = fit_paths(problems, device=backend.name, threads=cpu_threads) \
                if problems else []
            status_rows, pred_parts, k = [], [], 0
            for task, prep, st in batch:
                tid = int(task["task_id"])
                if prep is None:
                    status_rows.append(dict(task_id=tid, vmr_id=str(task["vmr_id"]), **st))
                    continue
                m = len(prep.problems)
                try:
                    preds = finalize_locus_oof(prep, fits[k:k + m], settings)
                    preds["task_id"] = tid
                    pred_parts.append(preds)
                    status_rows.append(dict(task_id=tid, vmr_id=prep.vmr_id, status="ok",
                                            detail=None))
                except Exception as err:
                    if is_environment_error(err):
                        raise
                    status_rows.append(dict(task_id=tid, vmr_id=prep.vmr_id, status="failed",
                                            detail=f"{type(err).__name__}: {err}"))
                k += m
            if pred_parts:
                _write_part(run_dir, "oof_rows", shard,
                            pd.concat(pred_parts, ignore_index=True).to_dict("records"))
            _write_part(run_dir, "status_rows", shard, status_rows)
            log(f"[lsp shard {shard}] {min(b0 + batch_loci, len(todo))}/{len(todo)} loci, "
                f"{len(problems)} glmnet paths in batch, wall {time.time() - started:.1f}s")


def _parts(run_dir, sub):
    paths = sorted(glob.glob(os.path.join(run_dir, sub, "part-*.parquet")))
    return pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True) if paths else None


def combine_lsp_run(run_dir, log=_log) -> dict:
    from ..lgv.combine import _write_tsv

    cfg, tasks, _, settings = _load(run_dir)
    comb = os.path.join(run_dir, "results", "combined")
    status = _parts(run_dir, "status_rows")
    preds = _parts(run_dir, "oof_rows")
    if status is None or preds is None:
        raise RuntimeError("No Module 03 shard output to combine")
    if status["task_id"].duplicated().any():
        raise RuntimeError("Duplicate task IDs among Module 03 status rows")
    expected = set(tasks["task_id"].astype(int))
    seen = set(status["task_id"].astype(int))
    recon = dict(expected=len(expected), completed=int((status["status"] == "ok").sum()),
                 qc_failed=int(status["status"].isin(["qc_failed", "excluded"]).sum()),
                 failed=int((status["status"] == "failed").sum()),
                 unaccounted=len(expected - seen), unexpected=len(seen - expected))
    _write_tsv(pd.DataFrame([recon]), os.path.join(comb, "task-reconciliation.tsv"))
    rows = []
    for vmr_id, g in preds.groupby("vmr_id", sort=True):
        rows.append(dict(vmr_id=vmr_id, **locus_metrics(g, settings)))
    metrics = pd.DataFrame(rows)
    by_repeat = preds.groupby(["vmr_id", "repeat_i"]).apply(
        lambda g: pd.Series(dict(n_predictions=len(g),
                                 r2_pred_oof=locus_metrics(g, settings)["r2_pred_oof"],
                                 screening_pass_frequency=g["screened_in"].mean())),
        include_groups=False).reset_index()
    spread = by_repeat.groupby("vmr_id")["r2_pred_oof"].agg(
        r2_repeat_min="min", r2_repeat_max="max", r2_repeat_sd="std").reset_index()
    metrics = metrics.merge(spread, on="vmr_id", how="left").merge(
        tasks.drop(columns=["task_id"], errors="ignore"), on="vmr_id", how="left")
    metrics["region"] = cfg["region"]
    metrics["cohort"] = cfg["cohort"]
    _write_tsv(by_repeat, os.path.join(comb, "fold-repeat-diagnostics.tsv"))
    _write_tsv(metrics, os.path.join(
        comb, f"oof-prediction-{cfg['cohort']}-{cfg['region']}-vmrs.tsv"))
    screen = preds[preds["screen_p"].notna()].groupby("vmr_id")["screen_p"].min()
    if len(screen):
        from scipy.stats import false_discovery_control

        bh = false_discovery_control(screen.to_numpy(), method="bh")
        _write_tsv(pd.DataFrame(dict(vmr_id=screen.index, min_screen_p=screen.values,
                                     min_screen_p_bh=bh)),
                   os.path.join(comb, "screen-bh-diagnostic.tsv"))
    per_donor = preds.groupby("donor")["vmr_id"].nunique().rename("n_vmrs_predicted")
    _write_tsv(per_donor.reset_index(), os.path.join(comb, "predictions-per-donor.tsv"))
    median_r2 = float(np.nanmedian(metrics["r2_pred_oof"]))
    qc = dict(region=cfg["region"], cohort=cfg["cohort"], expected_vmrs=len(tasks),
              scored_vmrs=len(metrics), failed_vmrs=recon["failed"],
              median_r2_pred_oof=median_r2,
              mean_r2_pred_oof=float(np.nanmean(metrics["r2_pred_oof"])),
              frac_r2_positive=float(np.nanmean(metrics["r2_pred_oof"] > 0)),
              median_cor2_oof=float(np.nanmedian(metrics["cor2_oof"])),
              median_calibration_slope=float(np.nanmedian(metrics["calibration_slope"])),
              mean_screening_pass=float(np.nanmean(metrics["screening_pass_frequency"])),
              donor_prediction_counts_equal=bool(per_donor.nunique() == 1),
              complete=recon["failed"] == 0 and recon["unaccounted"] == 0,
              leakage_tripwire_passed=median_r2 <= LEAKAGE_TRIPWIRE)
    _write_tsv(pd.DataFrame([qc]), os.path.join(comb, "prediction-run-qc.tsv"))
    log(f"[lsp combine] {recon}; median r2_pred_oof {median_r2:.3g}")
    if not qc["leakage_tripwire_passed"]:
        raise RuntimeError(f"Leakage tripwire: median r2_pred_oof {median_r2:.3f} > "
                           f"{LEAKAGE_TRIPWIRE}")
    return dict(reconciliation=recon, qc=qc)
