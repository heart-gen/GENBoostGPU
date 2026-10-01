"""Domain gate, relative local-SNP-contribution score, and score QC.

Ports Stages 03-05 of ``02_local_genetic_variance``:

* :func:`apply_domain_gate` - Stage 03's per-cell characterized-support gate
  (``n`` in ``allowed_n``; ``num_snps``, ``p_eff``, ``ld_metric`` inside the
  cell's support; ``p_eff`` inside ``[1, n]``) plus the frozen model;
* :func:`derive_score` - Stage 04's within-cell midrank percentile on
  ``pve_cis_joint_unbounded``, its z-score and quartile labels;
* :func:`check_score` - Stage 05's six acceptance criteria and decision.

Absolute PVE interpretation is never authorised: every row carries
``absolute_pve_interpretation_allowed = FALSE``.
"""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from .joint_model import JointModel

__all__ = ["read_support", "apply_domain_gate", "derive_score", "check_score",
           "DECISION_LOCKED", "SCORE_BASIS"]

DECISION_LOCKED = "PASS_RELATIVE_GENETIC_CONTROL_FAIL_ABSOLUTE_LOCUS_PVE"
SCORE_BASIS = "pve_cis_joint_unbounded"


def read_support(path: str, cell: str) -> dict:
    """Read one cell's rows of ``joint-pve-characterized-support.tsv``."""
    sha = hashlib.sha256(open(path, "rb").read()).hexdigest()
    sup = pd.read_csv(path, sep="\t", dtype=str)
    required = {"cell", "feature", "support_min", "support_max", "allowed_n"}
    if not required <= set(sup.columns):
        raise ValueError("Characterized-support table lacks required columns "
                         "(needs a per-cell table with a `cell` column)")
    sup = sup[sup["cell"] == cell]
    if sup.empty:
        raise ValueError(f"Characterized-support table has no rows for cell: {cell}")
    domain = {}
    for feature in ("num_snps", "p_eff", "ld_metric"):
        row = sup[sup["feature"] == feature]
        if len(row) != 1:
            raise ValueError(f"Characterized support is not unique for {feature}")
        lo, hi = float(row["support_min"].iloc[0]), float(row["support_max"].iloc[0])
        if not (np.isfinite(lo) and np.isfinite(hi)):
            raise ValueError(f"Nonfinite characterized support: {feature}")
        domain[feature] = (lo, hi)
    allowed_n = sorted({int(v) for v in str(sup["allowed_n"].iloc[0]).split(",") if v})
    if not allowed_n:
        raise ValueError("Characterized support lacks allowed_n")
    return dict(domain=domain, allowed_n=allowed_n, sha256=sha, cell=cell)


def _flag(series) -> np.ndarray:
    if series.dtype == bool:
        return series.to_numpy()
    v = series.astype(str).str.strip().str.lower()
    ok = v.isin(["true", "false", "t", "f", "1", "0"])
    if not ok.all():
        raise ValueError("Invalid or missing logical flag")
    return v.isin(["true", "t", "1"]).to_numpy()


def apply_domain_gate(features: pd.DataFrame, model: JointModel, support: dict,
                      joint_model_run_id: str = "") -> pd.DataFrame:
    """Stage 03: domain status/reason and frozen-model estimates."""
    d = features.copy()
    complete_flag = _flag(d["feature_complete"])
    failure_flag = _flag(d["computational_failure"])
    d["joint_pve_domain_status"] = "feature_incomplete"
    d["joint_pve_domain_reason"] = np.where(complete_flag, None, "joint features incomplete")
    complete = np.flatnonzero(complete_flag & ~failure_flag)
    if complete.size == 0:
        raise ValueError("No complete observed joint-feature rows")
    dom = support["domain"]
    reasons = []
    for i in complete:
        r = []
        n = int(float(d["n"].iloc[i]))
        if n not in support["allowed_n"]:
            r.append("n outside characterized values")
        p = float(d["num_snps"].iloc[i])
        if not np.isfinite(p) or p < dom["num_snps"][0] or p > dom["num_snps"][1]:
            r.append("num_snps outside characterized support")
        pe = float(d["p_eff"].iloc[i])
        if not np.isfinite(pe) or pe < 1 or pe > float(d["n"].iloc[i]):
            r.append("p_eff outside mathematical range [1,n]")
        elif pe < dom["p_eff"][0] or pe > dom["p_eff"][1]:
            r.append("p_eff outside characterized support")
        ld = float(d["ld_metric"].iloc[i])
        if not np.isfinite(ld) or ld < dom["ld_metric"][0] or ld > dom["ld_metric"][1]:
            r.append("ld_metric outside characterized support")
        reasons.append("; ".join(r) if r else None)
    reasons = np.array(reasons, dtype=object)
    status = d["joint_pve_domain_status"].to_numpy(dtype=object)
    status[complete] = np.where(pd.isna(reasons), "within_domain", "outside_domain")
    d["joint_pve_domain_status"] = status
    reason_col = d["joint_pve_domain_reason"].to_numpy(dtype=object)
    reason_col[complete] = reasons
    d["joint_pve_domain_reason"] = reason_col
    pred = model.predict(d.iloc[complete])
    for col in pred.columns:
        out_col = "positive_signal_audit_only" if col == "positive_signal" else col
        d[out_col] = pd.NA
        d.loc[d.index[complete], out_col] = pred[col].to_numpy()
    d["joint_model_run_id"] = joint_model_run_id
    d["joint_model_sha256"] = model.source_sha256
    d["config_characterized_support_sha256"] = support["sha256"]
    d["absolute_pve_interpretation_allowed"] = False
    return d


def derive_score(estimates: pd.DataFrame) -> pd.DataFrame:
    """Stage 04: within-cell midrank score on the unbounded estimate."""
    d = estimates.copy()
    banned = {"h2_unscaled", "r_squared_cv"} & set(d.columns)
    if banned:
        raise ValueError(f"Banned legacy columns in score input: {sorted(banned)}")
    if d["vmr_id"].duplicated().any():
        raise ValueError("Duplicate vmr_id in score input")
    for col in ("cohort", "region", "vmr_set_id"):
        if d[col].nunique(dropna=False) != 1:
            raise ValueError("Score input must describe exactly one cohort, region, "
                             "and vmr_set_id")
    complete = _flag(d["feature_complete"])
    failure = _flag(d["computational_failure"])
    raw_all = pd.to_numeric(d[SCORE_BASIS], errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(raw_all)
    status = d["joint_pve_domain_status"].astype(object)
    within = (status.notna() & (status == "within_domain")).to_numpy()
    eligible = complete & ~failure & finite & within
    reason = np.full(len(d), None, dtype=object)
    reason[~complete] = "incomplete_joint_features"
    reason[complete & failure] = "computational_failure"
    reason[complete & ~failure & ~finite] = "nonfinite_joint_estimate"
    reason[complete & ~failure & finite & status.isna().to_numpy()] = \
        "missing_joint_pve_domain_status"
    reason[complete & ~failure & finite & status.notna().to_numpy() & ~within] = \
        "outside_expanded_simulation_domain"
    n_eligible = int(eligible.sum())
    if n_eligible < 2:
        raise ValueError("Fewer than two eligible VMRs; score is undefined")
    raw = raw_all[eligible]
    midrank = rankdata(raw, method="average")
    score = (midrank - 0.5) / n_eligible
    sd = np.std(score, ddof=1)
    score_z = (score - score.mean()) / sd
    if not np.all(np.isfinite(score_z)):
        raise ValueError("Eligible score has zero or nonfinite variance")
    d["local_snp_contribution_score_basis"] = SCORE_BASIS
    d["local_genetic_control_eligible"] = eligible
    d["local_genetic_control_exclusion_reason"] = reason
    d["local_snp_contribution_score"] = np.nan
    d["local_snp_contribution_score_z"] = np.nan
    d["local_snp_contribution_quartile"] = None
    d.loc[eligible, "local_snp_contribution_score"] = score
    d.loc[eligible, "local_snp_contribution_score_z"] = score_z
    d.loc[eligible, "local_snp_contribution_quartile"] = np.where(
        score <= 0.25, "bottom_quartile",
        np.where(score >= 0.75, "top_quartile", "middle_50_percent"))
    d["absolute_pve_interpretation_allowed"] = False
    d["local_genetic_control_decision"] = DECISION_LOCKED
    return d


def check_score(score: pd.DataFrame, recon: dict, max_outside_calibration_domain: float,
                smoke_run: bool = False):
    """Stage 05: six criteria and the run decision. Returns (checks, decision)."""
    complete = _flag(score["feature_complete"])
    failure = _flag(score["computational_failure"])
    eligible = _flag(score["local_genetic_control_eligible"])
    absolute = _flag(score["absolute_pve_interpretation_allowed"])
    complete_n = int((complete & ~failure).sum())
    eligible_n = int(eligible.sum())
    eligible_rate = eligible_n / complete_n if complete_n else 0.0
    elig_score = score.loc[eligible, "local_snp_contribution_score"]
    raw = score.loc[eligible, SCORE_BASIS]
    max_tie = (raw.value_counts().max() / eligible_n) if eligible_n else np.nan
    lb = _flag(score.loc[eligible, "pve_lower_boundary_hit"]) if eligible_n else np.array([])
    ub = _flag(score.loc[eligible, "pve_upper_boundary_hit"]) if eligible_n else np.array([])
    boundary_rate = float(np.mean(lb | ub)) if eligible_n else np.nan
    recon_ok = (recon["unique_task_rows"] == recon["expected"] and recon["unaccounted"] == 0
                and recon["duplicate_task_ids"] == 0 and recon["unexpected_task_ids"] == 0)
    rows = [
        ("task_reconciliation_complete", float(recon_ok), "equal", 1.0,
         f"{recon['expected']} expected; {recon['unique_task_rows']} unique; "
         f"{recon['unaccounted']} unaccounted"),
        ("zero_computational_failures", float(failure.sum()), "equal", 0.0,
         f"{int(failure.sum())} rows flagged"),
        ("within_domain_rate", eligible_rate, "greater_or_equal",
         1 - max_outside_calibration_domain,
         f"{eligible_n}/{complete_n} complete-feature VMRs eligible"),
        ("score_is_nondegenerate", float(elig_score.nunique()), "greater_or_equal", 2.0,
         f"{elig_score.nunique()} unique values on {SCORE_BASIS}; max tie fraction "
         f"{max_tie:.4f}; boundary rate {boundary_rate:.4f}"),
        ("absolute_pve_is_prohibited", float(absolute.sum()), "equal", 0.0,
         "absolute_pve_interpretation_allowed must be FALSE for every row"),
        ("interpretation_decision_is_locked",
         float((score["local_genetic_control_decision"] == DECISION_LOCKED).all()),
         "equal", 1.0, ",".join(score["local_genetic_control_decision"].unique())),
    ]
    checks = pd.DataFrame(rows, columns=["criterion", "observed", "rule", "threshold",
                                         "detail"])
    checks["pass"] = np.where(checks["rule"] == "equal",
                              checks["observed"] == checks["threshold"],
                              checks["observed"] >= checks["threshold"])
    checks = checks[["criterion", "observed", "rule", "threshold", "pass", "detail"]]
    all_pass = bool(checks["pass"].all())
    if not all_pass:
        decision = "FAIL_OBSERVED_RELATIVE_SCORE_QC"
    elif smoke_run:
        decision = "PASS_SMOKE_ONLY_NOT_ACCEPTABLE"
    else:
        decision = "PASS_RELATIVE_SCORE_OBSERVED_QC"
    extras = dict(max_tie_fraction=max_tie, boundary_rate=boundary_rate,
                  eligible_n=eligible_n, complete_n=complete_n)
    return checks, decision, extras
