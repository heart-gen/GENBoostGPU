"""Combine shard outputs: reconcile, apply the frozen model, score, check.

Equivalent to Module 02 Stages 02-05 for a GENBoostGPU run directory:

* reconcile every expected task (missing tasks become explicit
  computational-failure rows; duplicate or unexpected task IDs stop the run),
  clearing stale diagnostics when their condition no longer holds;
* in ``bslmm_mode="separate"``, join the CPU job's BSLMM rows first;
* apply the frozen joint model and the per-cell domain gate, derive the
  within-cell score, and evaluate the six Stage 05 criteria.

Outputs under ``results/combined/`` use the R pipeline's file names and
columns. ``--write-r-task-rows`` additionally writes ``results/task_rows/
vmr-%07d.tsv`` so R Stage 02 can consume the run directly.
"""
from __future__ import annotations

import functools
import glob
import hashlib
import json
import os
import time

import numpy as np
import pandas as pd

from .joint_model import load_joint_model
from .runner import BSLMM_COLUMNS, ROW_COLUMNS, finalize_rows, load_run
from .score import apply_domain_gate, check_score, derive_score, read_support

__all__ = ["combine_run", "read_parts", "reconcile"]

_log = functools.partial(print, flush=True)


def read_parts(run_dir: str, subdir: str) -> pd.DataFrame | None:
    paths = sorted(glob.glob(os.path.join(run_dir, subdir, "part-*.parquet")))
    if not paths:
        return None
    return pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)


def _write_tsv(frame: pd.DataFrame, path: str) -> None:
    """R-style TSV: NA for missing, TRUE/FALSE for logicals."""
    out = frame.copy()
    for col in out.columns:
        if out[col].dtype == bool:
            out[col] = np.where(out[col], "TRUE", "FALSE")
        elif out[col].dtype == object:
            vals = out[col]
            is_bool = vals.map(lambda v: isinstance(v, (bool, np.bool_)))
            if is_bool.any():
                out[col] = vals.map(lambda v: ("TRUE" if v else "FALSE")
                                    if isinstance(v, (bool, np.bool_)) else v)
    tmp = path + ".tmp"
    out.to_csv(tmp, sep="\t", index=False, na_rep="NA", float_format="%.15g")
    os.replace(tmp, path)


def _write_or_clear(ids, path):
    if len(ids):
        pd.DataFrame({"task_id": sorted(ids)}).to_csv(path, sep="\t", index=False)
    elif os.path.exists(path):
        os.remove(path)


def reconcile(rows: pd.DataFrame | None, tasks: pd.DataFrame, combined_dir: str,
              cfg) -> tuple[pd.DataFrame, dict]:
    expected_ids = tasks["task_id"].astype(int).tolist()
    if rows is None or rows.empty:
        rows = pd.DataFrame(columns=ROW_COLUMNS)
    ids = rows["task_id"].astype(int)
    duplicate_ids = sorted(set(ids[ids.duplicated(keep=False)]))
    unexpected_ids = sorted(set(ids) - set(expected_ids))
    observed = set(ids)
    missing_ids = sorted(set(expected_ids) - observed)
    status = rows["terminal_status"].astype(str)
    recon = dict(
        expected=len(expected_ids), task_files=len(rows), unique_task_rows=len(observed),
        completed=int((status == "completed").sum()),
        excluded=int((status == "excluded").sum()),
        qc_failed=int((status == "qc_failed").sum()),
        computational_failure=int((status == "computational_failure").sum()),
        features_only=int((status == "features_only").sum()),
        unaccounted=len(missing_ids), duplicate_task_ids=len(duplicate_ids),
        unexpected_task_ids=len(unexpected_ids))
    _write_tsv(pd.DataFrame([recon]), os.path.join(combined_dir, "task-reconciliation.tsv"))
    _write_or_clear(missing_ids, os.path.join(combined_dir, "missing-task-ids.tsv"))
    _write_or_clear(duplicate_ids, os.path.join(combined_dir, "duplicate-task-ids.tsv"))
    _write_or_clear(unexpected_ids, os.path.join(combined_dir, "unexpected-task-ids.tsv"))
    if duplicate_ids or unexpected_ids:
        raise RuntimeError("Task reconciliation found duplicate or unexpected task IDs")
    if missing_ids:
        src = cfg.source
        miss = tasks[tasks["task_id"].isin(missing_ids)].copy()
        fill = pd.DataFrame({c: [None] * len(miss) for c in ROW_COLUMNS})
        for c in set(ROW_COLUMNS) & set(miss.columns):
            fill[c] = miss[c].to_numpy()
        group = src.get("estimation_group") or cfg.cohort
        fill["population"] = group
        fill["estimation_group"] = group
        fill["catalog_cohort"] = src.get("catalog_cohort") or cfg.cohort
        fill["upstream_vmr_run_id"] = src.get("upstream_vmr_run_id", "")
        fill["feature_complete"] = False
        fill["computational_failure"] = True
        fill["terminal_status"] = "computational_failure"
        fill["feature_error"] = "missing task output"
        rows = pd.concat([rows, fill], ignore_index=True)
    rows = rows.sort_values("task_id").reset_index(drop=True)
    if rows["task_id"].astype(int).tolist() != expected_ids:
        raise RuntimeError("Combined rows do not reconcile exactly to the task table")
    return rows, recon


def _sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def combine_run(run_dir: str, write_r_task_rows: bool = False, score: bool = True,
                log=_log) -> dict:
    cfg, tasks = load_run(run_dir)
    combined = os.path.join(run_dir, "results", "combined")
    os.makedirs(combined, exist_ok=True)
    rows = read_parts(run_dir, "task_rows")
    if cfg.bslmm_mode == "separate" and rows is not None:
        bs = read_parts(run_dir, "bslmm_rows")
        rows = rows.drop(columns=[c for c in BSLMM_COLUMNS if c in rows.columns])
        if bs is not None:
            if bs["task_id"].duplicated().any():
                raise RuntimeError("Duplicate task IDs among BSLMM rows")
            rows = rows.merge(bs, on="task_id", how="left")
        for c in BSLMM_COLUMNS:
            if c not in rows.columns:
                rows[c] = None
        rows["bslmm_converged"] = rows["bslmm_converged"].fillna(False).astype(bool)
        rows = finalize_rows(rows, "inline")
    rows, recon = reconcile(rows, tasks, combined, cfg)
    rows = rows[ROW_COLUMNS + [c for c in rows.columns if c not in ROW_COLUMNS]]
    _write_tsv(rows, os.path.join(combined, "observed-joint-features.tsv"))
    if write_r_task_rows:
        tr = os.path.join(run_dir, "results", "task_rows")
        os.makedirs(tr, exist_ok=True)
        for _, r in rows.iterrows():
            _write_tsv(pd.DataFrame([r]), os.path.join(tr, f"vmr-{int(r['task_id']):07d}.tsv"))
    log(f"[combine] reconciliation: {recon}")
    result = dict(reconciliation=recon)
    if score and cfg.bslmm_mode != "off":
        if not (cfg.joint_model_path and cfg.support_path):
            raise RuntimeError("run.json lacks joint_model_path/support_path; "
                               "cannot score (use --no-score for features only)")
        model = load_joint_model(cfg.joint_model_path, cfg.joint_model_source_sha256)
        support = read_support(cfg.support_path, cfg.support_cell or cfg.cohort)
        est = apply_domain_gate(rows, model, support, cfg.joint_model_run_id)
        _write_tsv(est, os.path.join(combined, "observed-joint-estimates.tsv"))
        sc = derive_score(est)
        score_path = os.path.join(combined,
                                  f"local-genetic-control-{cfg.cohort}-{cfg.region}-vmrs.tsv")
        _write_tsv(sc, score_path)
        checks, decision, extras = check_score(
            sc, recon, cfg.max_outside_calibration_domain, cfg.smoke_run)
        _write_tsv(checks, os.path.join(combined, "observed-score-qc.tsv"))
        result.update(decision=decision, **extras)
        log(f"[combine] decision: {decision}  ({extras})")
        result["joint_model_json_sha256"] = model.json_sha256
        result["joint_model_source_sha256"] = model.source_sha256
        result["support_sha256"] = support["sha256"]
    outputs = {os.path.relpath(p, run_dir): _sha(p)
               for p in sorted(glob.glob(os.path.join(combined, "*.tsv")))}
    manifest = dict(run_id=cfg.run_id, cohort=cfg.cohort, region=cfg.region,
                    seed_run_id=cfg.effective_seed_run_id, bslmm_mode=cfg.bslmm_mode,
                    combined_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    run_json_sha256=_sha(os.path.join(run_dir, "run.json")),
                    tasks_sha256=_sha(os.path.join(run_dir, "tasks.tsv")),
                    absolute_pve_interpretation_allowed=False,
                    outputs=outputs, **result)
    with open(os.path.join(run_dir, "results", "run-manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2, default=str)
    return manifest
