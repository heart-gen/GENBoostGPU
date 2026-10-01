"""Apply the frozen Module 02 joint PVE model.

The model (``monotone_hinge_ridge_v1``, gate ``final_joint_pve_v1``) is
exported from its R ``.rds`` with ``scripts/export_frozen_joint_model.R``; this
module reproduces ``joint_pve_functions.R::{make_joint_matrix,
predict_joint_pve}``. The model is never refit or modified here.

The output columns are the same as R's, including ``positive_signal``, which
Stage 03 renames to ``positive_signal_audit_only``: it is not a biological
class and absolute-PVE interpretation stays prohibited.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = ["JointModel", "load_joint_model", "SIGNAL_FEATURES", "DESIGN_TERMS"]

SIGNAL_FEATURES = ("bslmm_pve", "he_h2", "rho2_oof", "r2_oof")
DESIGN_TERMS = ("design__log_n", "design__log_p", "design__log_p_eff", "design__ld")


@dataclass
class JointModel:
    spec: dict
    json_sha256: str

    @property
    def source_sha256(self) -> str:
        return self.spec["source_rds_sha256"]

    def _matrix(self, data: pd.DataFrame):
        cols = []
        names = []
        for term in self.spec["signal_terms"]:
            feature = term["feature"]
            lo, hi = self.spec["signal_spec"][feature]["clip"]
            x = pd.to_numeric(data[feature], errors="coerce").to_numpy(dtype=float)
            if not np.all(np.isfinite(x)):
                raise ValueError(f"Nonfinite joint feature: {feature}")
            x = np.minimum(hi, np.maximum(lo, x))
            cols.append(np.maximum(x - float(term["knot"]), 0.0))
            names.append(term["name"])
        raw = np.column_stack([
            np.log(pd.to_numeric(data["n"]).to_numpy(dtype=float)),
            np.log(pd.to_numeric(data["num_snps"]).to_numpy(dtype=float)),
            np.log(pd.to_numeric(data["p_eff"]).to_numpy(dtype=float)),
            pd.to_numeric(data["ld_metric"]).to_numpy(dtype=float),
        ])
        if not np.all(np.isfinite(raw)):
            raise ValueError("Nonfinite joint design feature")
        scaler = self.spec["design_scaler"]
        if list(scaler["names"]) != list(DESIGN_TERMS):
            raise ValueError("Unexpected design scaler terms")
        center = np.asarray(scaler["center"], dtype=float)
        scale = np.asarray(scaler["scale"], dtype=float)
        design = (raw - center) / scale
        matrix = np.column_stack(cols + [design[:, k] for k in range(design.shape[1])])
        return matrix, names + list(DESIGN_TERMS)

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """``predict_joint_pve(model, data)`` for rows with complete features."""
        x, names = self._matrix(data)
        coef_names = list(self.spec["coefficients"]["names"])
        coef_values = np.asarray(self.spec["coefficients"]["values"], dtype=float)
        if coef_names[0] != "(Intercept)" or coef_names[1:] != names:
            raise ValueError("Joint-model coefficients do not match the design matrix")
        unbounded = coef_values[0] + x @ coef_values[1:]
        lo = float(self.spec["output_lower"])
        hi = float(self.spec["output_upper"])
        estimate = np.minimum(hi, np.maximum(lo, unbounded))
        q = self.spec.get("conformal_q")
        cutoff = self.spec.get("null_cutoff")
        if q is not None and np.isfinite(q):
            lower = np.maximum(lo, estimate - q)
            upper = np.minimum(hi, estimate + q)
        else:
            lower = upper = np.full(estimate.size, np.nan)
        positive = (estimate > cutoff) if cutoff is not None and np.isfinite(cutoff) \
            else np.full(estimate.size, pd.NA)
        return pd.DataFrame(dict(
            pve_cis_joint_unbounded=unbounded,
            pve_cis_joint_calibrated=estimate,
            pve_simulation_reference_lower=lower,
            pve_simulation_reference_upper=upper,
            pve_lower_boundary_hit=unbounded <= lo,
            pve_upper_boundary_hit=unbounded >= hi,
            positive_signal=positive,
        ), index=data.index)


def load_joint_model(path: str, expected_source_sha256: str | None = None) -> JointModel:
    """Load an exported model JSON; optionally pin the source ``.rds`` sha."""
    raw = open(path, "rb").read()
    spec = json.loads(raw)
    if spec.get("format") != "genboostgpu-joint-pve-model-v1":
        raise ValueError(f"{path} is not a genboostgpu joint-PVE model export")
    if spec.get("gate_version") != "final_joint_pve_v1":
        raise ValueError(f"Unexpected joint-model gate version: {spec.get('gate_version')}")
    if expected_source_sha256 and spec["source_rds_sha256"].lower() != \
            expected_source_sha256.lower():
        raise ValueError("Joint-model source checksum does not match the pinned value")
    return JointModel(spec=spec, json_sha256=hashlib.sha256(raw).hexdigest())
