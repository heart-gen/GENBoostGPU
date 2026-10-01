import json

import numpy as np
import pandas as pd
import pytest

from genboostgpu.lgv.joint_model import DESIGN_TERMS, load_joint_model
from genboostgpu.lgv.score import (DECISION_LOCKED, apply_domain_gate, check_score,
                                   derive_score, read_support)


@pytest.fixture
def toy_model(tmp_path):
    terms, names, values = [], ["(Intercept)"], [0.01]
    spec = {}
    for f, clip in [("bslmm_pve", [0, 1]), ("he_h2", [-1, 2]), ("rho2_oof", [0, 1]),
                    ("r2_oof", [-1, 1])]:
        spec[f] = {"clip": clip, "knots": [0, 0.1]}
        for k in (0, 0.1):
            name = f"signal__{f}__k{str(k).replace('.', 'p')}"
            terms.append({"name": name, "feature": f, "knot": k})
            names.append(name)
            values.append(0.2)
    names += list(DESIGN_TERMS)
    values += [0.01, -0.01, 0.02, 0.0]
    model = {"format": "genboostgpu-joint-pve-model-v1", "source_rds_sha256": "ab" * 32,
             "family": "monotone_hinge_ridge_v1", "gate_version": "final_joint_pve_v1",
             "lambda": 1e-4, "output_lower": 0, "output_upper": 1, "conformal_q": 0.5,
             "null_cutoff": 0.2, "signal_spec": spec, "signal_terms": terms,
             "design_scaler": {"names": list(DESIGN_TERMS), "center": [5, 7, 4, 0.2],
                               "scale": [0.3, 1.5, 0.5, 0.15]},
             "coefficients": {"names": names, "values": values}}
    path = tmp_path / "model.json"
    path.write_text(json.dumps(model))
    return load_joint_model(str(path), "ab" * 32), model


def _features(n_rows=40, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(dict(
        task_id=np.arange(1, n_rows + 1), vmr_id=[f"v{i}" for i in range(n_rows)],
        cohort="AA", region="caudate", vmr_set_id="set", chrom="chr1", start=1, end=2,
        n_cpgs=6, n_variants=500, mean_methylation=0.5, methylation_variance=0.01,
        n=153, num_snps=rng.integers(200, 3000, n_rows), p_eff=rng.uniform(10, 60, n_rows),
        ld_metric=rng.uniform(0.05, 0.3, n_rows), bslmm_pve=rng.uniform(0, 0.6, n_rows),
        he_h2=rng.normal(0.1, 0.2, n_rows), rho2_oof=rng.uniform(0, 0.3, n_rows),
        r2_oof=rng.normal(0, 0.1, n_rows), feature_complete=True,
        computational_failure=False))


def test_joint_model_linear_predictor(toy_model):
    model, spec = toy_model
    d = _features(5)
    out = model.predict(d)
    row = d.iloc[0]
    x = []
    for t in spec["signal_terms"]:
        lo, hi = spec["signal_spec"][t["feature"]]["clip"]
        v = min(hi, max(lo, row[t["feature"]]))
        x.append(max(v - t["knot"], 0))
    raw = [np.log(row.n), np.log(row.num_snps), np.log(row.p_eff), row.ld_metric]
    sc = spec["design_scaler"]
    x += [(r - c) / s for r, c, s in zip(raw, sc["center"], sc["scale"])]
    expected = spec["coefficients"]["values"][0] + np.dot(spec["coefficients"]["values"][1:], x)
    assert out["pve_cis_joint_unbounded"].iloc[0] == pytest.approx(expected, rel=1e-12)


def test_gate_score_and_check(tmp_path, toy_model):
    model, _ = toy_model
    sup = tmp_path / "support.tsv"
    rows = [("AA", f, lo, hi, "117,118,153") for f, lo, hi in
            [("num_snps", 100, 12000), ("p_eff", 7, 274), ("ld_metric", 0.01, 0.41)]]
    pd.DataFrame(rows, columns=["cell", "feature", "support_min", "support_max",
                                "allowed_n"]).to_csv(sup, sep="\t", index=False)
    d = _features(40)
    d.loc[0, "n"] = 200                      # outside allowed_n
    d.loc[1, "feature_complete"] = False
    est = apply_domain_gate(d, model, read_support(str(sup), "AA"))
    assert est.loc[0, "joint_pve_domain_status"] == "outside_domain"
    assert est.loc[1, "joint_pve_domain_status"] == "feature_incomplete"
    sc = derive_score(est)
    elig = sc["local_genetic_control_eligible"]
    assert elig.sum() == 38
    s = sc.loc[elig, "local_snp_contribution_score"]
    assert s.min() == pytest.approx(0.5 / 38) and s.max() == pytest.approx(37.5 / 38)
    assert (sc["local_genetic_control_decision"] == DECISION_LOCKED).all()
    recon = dict(expected=40, unique_task_rows=40, unaccounted=0, duplicate_task_ids=0,
                 unexpected_task_ids=0)
    checks, decision, _ = check_score(sc, recon, 0.10)
    assert decision == "PASS_RELATIVE_SCORE_OBSERVED_QC", checks
    _, decision_smoke, _ = check_score(sc, recon, 0.10, smoke_run=True)
    assert decision_smoke == "PASS_SMOKE_ONLY_NOT_ACCEPTABLE"
    recon_bad = dict(recon, unaccounted=1, unique_task_rows=39)
    _, decision_bad, _ = check_score(sc, recon_bad, 0.10)
    assert decision_bad == "FAIL_OBSERVED_RELATIVE_SCORE_QC"
