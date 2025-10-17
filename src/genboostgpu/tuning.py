import warnings
import numpy as np
import pandas as pd
import itertools as it
from .vmr_runner import run_single_window
from .hyperparams import enet_from_targets
from .snp_processing import count_snps_in_window

__all__ = ["select_tuning_windows", "global_tune_params"]

def select_tuning_windows(
    windows, bim, frac=0.02, n_min=150, n_max=1000, window_size=500_000, use_window=True,
    n_bins=3, per_chrom_min=0, seed=13, exclude_failed=None
):
    """
    Select a stratified subset of windows for global hyperparameter tuning.
    """
    rng = np.random.default_rng(seed)
    # Optional exclusion set
    excl = set(exclude_failed or [])

    # Compute SNP counts per window
    rows = []
    for w in windows:
        key = (str(w["chrom"]), int(w["start"]), int(w.get("end", w["start"])))
        if key in excl:
            continue
        M_raw = count_snps_in_window(
            bim, w["chrom"], w["start"], w.get("end", w["start"]),
            window_size=window_size, use_window=use_window
        )
        if M_raw > 0:
            rows.append({**w, "M_raw": M_raw})

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("No windows with SNPs for tuning selection.")

    # Decide target
    target = int(np.clip(round(frac * len(df)), n_min, n_max))
    
    # Bin by M_raw (quantiles)
    n_unique = df["M_raw"].nunique()
    bins = min(n_bins, n_unique)
    if bins > 1:
        df["bin"] = pd.qcut(df["M_raw"], q=bins, duplicates="drop")
    else:
        df["bin"] = "all"

    # Optional: ensure some chrom diversity
    picked_idx = []
    if per_chrom_min > 0:
        for ch, dch in df.groupby("chrom"):
            k = min(per_chrom_min, len(dch))
            if k > 0:
                picked_idx.extend(dch.sample(n=k, random_state=seed).index.tolist())
                
    remaining = df.drop(index=picked_idx)

    # Allocate evenly
    need = max(0, target - len(picked_idx))
    if need > 0:
        bin_groups = remaining.groupby("bin")
        B = max(1, len(bin_groups))
        base = need // B
        extra = need % B

        # Order bins
        ordered_bins = [b for b, _ in sorted(bin_groups, key=lambda x: str(x[0]))]
        for i, b in enumerate(ordered_bins):
            g = bin_groups.get_group(b)
            take = min(len(g), base + (1 if i < extra else 0))
            if take > 0:
                picked_idx.extend(g.sample(n=take, random_state=seed).index.tolist())

    sel = df.loc[picked_idx].drop_duplicates()
    if len(sel) < target:
        short = target - len(sel)
        fill = df.drop(index=sel.index).sort_values("M_raw", ascending=False).head(short)
        sel = pd.concat([sel, fill], axis=0).drop_duplicates()

    return sel.drop(columns=["bin"] if "bin" in sel.columns else []).to_dict(orient="records")


def global_tune_params(
    tuning_windows, geno_arr=None, bim=None, fam=None,
    error_regions=None, outdir="tuning_tmp",
    window_size=500_000, by_hand=False,
    grid=None, early_stop=None, use_window=True, batch_size=4096
):
    """
    One-time global hyperparam search on a small, stratified subset.
    """
    # Default grid: |grid| * |tuning_windows|
    if grid is None:
        grid = {
            "c_lambda": [0.7, 1.0, 1.4],
            "c_ridge": [0.5, 1.0, 2.0],
            "subsample_frac": [0.5, 0.7, 0.9],
            "batch_size": [2048, 4096, 8192],
        }
    if early_stop is None:
        early_stop = {"patience": 5, "min_delta": 1e-4, "min_rel_gain": 1e-3}

    # build param combos
    N = _infer_N_global(fam=fam, geno_arr=geno_arr, tuning_windows=tuning_windows)
    if N < 2:
        raise ValueError("Could not infer a valid sample size, N, for tuning.")
    
    combos = list(it.product(
        grid["c_lambda"], grid["c_ridge"], grid["subsample_frac"], grid["batch_size"], 
    ))

    best, best_score = None, -np.inf
    for (c_lam, c_ridge, sub, bs) in combos:
        scores = []
        for w in tuning_windows:
            # Lightweight call: n_trials=1, rely on early stopping
            M = int(w.get("M_raw") or count_snps_in_window(
                bim, w["chrom"], w["start"], w.get("end", w["start"]),
                window_size=window_size, use_window=use_window
            ))
            if M <= 1:
                continue

            # Calculate fixed alpha using lasso scaling
            alpha, l1r = enet_from_targets(M, N, c_lambda=c_lam, c_ridge=c_ridge)
            
            res = run_single_window(
                chrom=w["chrom"], start=w["start"], end=w.get("end", w["start"]),
                geno_arr=geno_arr, bim=bim, fam=fam,
                geno_path=w.get("geno_path"), pheno=w.get("pheno"),
                pheno_path=w.get("pheno_path"), pheno_id=w.get("pheno_id"),
                has_header=w.get("has_header", True), y_pos=w.get("y_pos"),
                error_regions=error_regions, outdir=outdir,
                window_size=window_size, by_hand=by_hand,
                n_trials=1, n_iter=100, use_window=use_window,
                batch_size=min(bs, M), fixed_alpha=alpha,
                fixed_l1_ratio=l1r, fixed_subsample=sub,
                early_stop=early_stop, save_full=False
            )
            if res is not None and np.isfinite(res.get("final_r2", np.nan)):
                scores.append(float(res["final_r2"]))
        if scores:
            score = float(np.nanmedian(scores))  # robust across windows
            if score > best_score:
                best_score = score
                best = dict(
                    c_lambda=float(c_lam), 
                    c_ridge=float(c_ridge),
                    subsample_frac=float(sub), 
                    batch_size=int(bs)
                )

    if best is None:
        raise RuntimeError("Global tuning failed to evaluate any window.")
    best["score"] = best_score
    return best


def _infer_N_global(fam=None, geno_arr=None, tuning_windows=None):
    """
    Try to infer N (number of individuals) once:
      1) len(fam) if provided
      2) geno_arr.shape[0] if provided
      3) first window's 'pheno' length if available (fallback)
    """
    if fam is not None:
        try:
            return int(len(fam))
        except Exception:
            pass
    if geno_arr is not None:
        try:
            return int(getattr(geno_arr, "shape", [0])[0])
        except Exception:
            pass
    if tuning_windows:
        for w in tuning_windows:
            p = w.get("pheno", None)
            if p is not None:
                try:
                    return int(getattr(p, "shape", [0])[0])
                except Exception:
                    continue
    warnings.warn("Unable to infer sample size N from any source.")
    return 0
