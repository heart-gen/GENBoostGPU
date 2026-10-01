"""Seeds and fold assignments for the nested cross-fit.

The defaults reproduce ``02_local_genetic_variance/_h/00_functions.R`` exactly:
``stable_seed`` (Stage 01), ``offset_seed`` and ``make_balanced_folds`` use
R's own generator (see :mod:`genboostgpu.lgv.rrng`). A run can instead import
an explicit fold table, which is recorded in the run manifest.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .rrng import RRNG, r_sample

__all__ = [
    "SEED_MODULUS",
    "stable_seed",
    "offset_seed",
    "make_balanced_folds",
    "FoldPlan",
    "r_fold_plan",
    "fold_plan_to_frame",
    "read_fold_table",
]

SEED_MODULUS = 2147483629


def stable_seed(*parts) -> int:
    """Stage 01's ``stable_seed(...)``: a string hash reduced mod 2147483629.

    R calls ``paste(..., collapse = "|")`` on length-one arguments, so they
    are joined with ``paste``'s default ``sep = " "`` (``collapse`` never
    applies). The hash walks ``utf8ToInt`` code points in double precision;
    integers are exact here, so this matches bit for bit.
    """
    text = " ".join(str(p) for p in parts)
    value = 104729
    for ch in text:
        value = (value * 131 + ord(ch)) % SEED_MODULUS
    return int(max(1, value))


def offset_seed(seed: int, offset: int) -> int:
    """``offset_seed()``: addition in double precision, then one modulo."""
    value = (float(seed) + float(offset)) % SEED_MODULUS
    return int(max(1, value))


def make_balanced_folds(n: int, k: int, seed: int) -> np.ndarray:
    """``make_balanced_folds(n, k, seed)``: 1-based fold ids, R-identical."""
    n = int(n)
    if n < 6:
        raise ValueError("At least six samples are required")
    k = max(2, min(int(k), n))
    assignment = np.resize(np.arange(1, k + 1), n)
    return r_sample(RRNG(seed), assignment, n).astype(np.int64)


@dataclass
class FoldPlan:
    """Outer and inner fold ids for one locus.

    ``outer[r]`` holds the 1-based outer fold of each sample in repeat ``r``.
    ``inner[(r, f)]`` holds the 1-based inner fold of each *outer-training*
    sample (in outer-training order) for repeat ``r``, outer fold ``f``.
    """

    outer: list[np.ndarray]
    inner: dict[tuple[int, int], np.ndarray] = field(default_factory=dict)
    source: str = "r_rng"

    @property
    def n_repeats(self) -> int:
        return len(self.outer)


def r_fold_plan(n: int, seed: int, outer_folds: int, outer_repeats: int,
                inner_folds: int, inner_seed_fn=None) -> FoldPlan:
    """Fold plan exactly as ``crossfit_elastic_net`` draws it.

    Outer repeat ``r`` (1-based) uses ``offset_seed(seed, r * 1009)``; inner
    folds for (``r``, ``f``) use ``offset_seed(seed, r * 1009 + f * 9173)``
    over the outer-training samples, with ``inner_folds`` clamped to
    ``[3, n_train]`` as in ``fit_inner_elastic_net``.
    """
    outer_k = max(2, min(int(outer_folds), int(n)))
    outer = []
    inner = {}
    for r in range(1, int(outer_repeats) + 1):
        fold_id = make_balanced_folds(n, outer_k, offset_seed(seed, r * 1009))
        outer.append(fold_id)
        for f in range(1, outer_k + 1):
            n_train = int(np.sum(fold_id != f))
            k_inner = max(3, min(int(inner_folds), n_train))
            s = (inner_seed_fn(r, f) if inner_seed_fn is not None
                 else offset_seed(seed, r * 1009 + f * 9173))
            inner[(r, f)] = make_balanced_folds(n_train, k_inner, s)
    return FoldPlan(outer=outer, inner=inner, source="r_rng")


FOLD_TABLE_COLUMNS = ["task_id", "level", "repeat_id", "outer_fold",
                      "sample_index", "fold"]


def fold_plan_to_frame(plan: FoldPlan, task_id) -> pd.DataFrame:
    """Long table of a fold plan (the format :func:`read_fold_table` reads).

    ``level == "outer"`` rows give each sample's outer fold for a repeat
    (``outer_fold`` is blank). ``level == "inner"`` rows give the inner fold
    of each outer-training sample for (``repeat_id``, ``outer_fold``).
    ``sample_index`` is the 1-based position of the donor in the locus.
    """
    rows = []
    for r, fold_id in enumerate(plan.outer, start=1):
        for i, f in enumerate(fold_id, start=1):
            rows.append((str(task_id), "outer", r, pd.NA, i, int(f)))
        for f in np.unique(fold_id):
            train_index = np.flatnonzero(fold_id != f) + 1
            inner = plan.inner[(r, int(f))]
            for i, g in zip(train_index, inner):
                rows.append((str(task_id), "inner", r, int(f), int(i), int(g)))
    return pd.DataFrame(rows, columns=FOLD_TABLE_COLUMNS)


def read_fold_table(path: str, task_id) -> FoldPlan:
    """Load an imported fold plan for one task (see :func:`fold_plan_to_frame`)."""
    tab = pd.read_csv(path, sep="\t", dtype={"task_id": str})
    missing = set(FOLD_TABLE_COLUMNS) - set(tab.columns)
    if missing:
        raise ValueError(f"Fold table lacks columns: {sorted(missing)}")
    tab = tab[tab["task_id"] == str(task_id)]
    if tab.empty:
        raise KeyError(f"No fold rows for task {task_id} in {path}")
    outer_tab = tab[tab["level"] == "outer"]
    inner_tab = tab[tab["level"] == "inner"]
    outer = []
    inner = {}
    for r, sub in outer_tab.groupby("repeat_id", sort=True):
        sub = sub.sort_values("sample_index")
        idx = sub["sample_index"].to_numpy(np.int64)
        if not np.array_equal(idx, np.arange(1, idx.size + 1)):
            raise ValueError(f"Outer folds for repeat {r} do not cover samples 1..n")
        fold_id = sub["fold"].to_numpy(np.int64)
        outer.append(fold_id)
        for f in np.unique(fold_id):
            expected = np.flatnonzero(fold_id != f) + 1
            part = inner_tab[(inner_tab["repeat_id"] == r)
                             & (inner_tab["outer_fold"].astype("Int64") == int(f))]
            part = part.sort_values("sample_index")
            if not np.array_equal(part["sample_index"].to_numpy(np.int64), expected):
                raise ValueError(
                    f"Inner folds for repeat {r}, outer fold {f} do not match "
                    "the outer-training samples"
                )
            inner[(int(r), int(f))] = part["fold"].to_numpy(np.int64)
    return FoldPlan(outer=outer, inner=inner, source=f"table:{path}")
