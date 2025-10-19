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
            rows.append({**w, "M_raw": int(M_raw)})

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
        df["bin"] = pd.Series(["all"] * len(df), dtype="category")

    # Optional: ensure some chrom diversity
    picked_idx = []
    if per_chrom_min > 0:
        for _, dch in df.groupby("chrom"):
            k = min(per_chrom_min, len(dch))
            if k > 0:
                picked_idx.extend(dch.sample(n=k, random_state=seed).index.tolist())
                
    remaining = df.drop(index=picked_idx)

    # Allocate evenly
    need = max(0, target - len(picked_idx))
    if need > 0:
        groups = list(remaining.groupby("bin", observed=True))
        B = max(1, len(groups))
        base = need // B
        extra = need % B

        # Order bins
        ordered_bins = sorted(groups, key=lambda t: str(t[0]))
        for i, (_, g) in enumerate(ordered_bins):
            take = min(len(g), base + (1 if i < extra else 0))
            if take > 0:
                picked_idx.extend(g.sample(n=take, random_state=seed).index.tolist())

    sel = df.loc[picked_idx].drop_duplicates()
    if len(sel) < target:
        short = target - len(sel)
        fill = df.drop(index=sel.index).sort_values("M_raw", ascending=False).head(short)
        sel = pd.concat([sel, fill], axis=0).drop_duplicates()

    # Drop the helper column if present
    if "bin" in sel.columns:
        sel = sel.drop(columns=["bin"])

    return sel.to_dict(orient="records")


def global_tune_params(
    tuning_windows, geno_arr=None, bim=None, fam=None,
    error_regions=None, outdir="tuning_tmp", window_size=500_000, 
    by_hand=False, grid=None, early_stop=None, use_window=True, 
    batch_size=4096, use_multi_gpu=True, dask_client=None, 
    max_in_flight=None, rmm_pool_size="12GB"
):
    """
    One-time global hyperparam search. If multiple GPUs are present (and either
    dask_client is provided or use_multi_gpu=True), evaluate windows in parallel.
    """
    if grid is None:
        grid = {
            "c_lambda": [0.7, 1.0, 1.4],
            "c_ridge":  [0.5, 1.0, 2.0],
            "subsample_frac": [0.5, 0.7, 0.9],
            "batch_size": [2048, 4096, 8192],
        }
    if early_stop is None:
        early_stop = {"patience": 5, "min_delta": 1e-4, "warmup": 5}

    # Infer sample size once
    N = _infer_N_global(fam=fam, geno_arr=geno_arr, tuning_windows=tuning_windows)
    if N < 2:
        raise ValueError("Could not infer a valid sample size N for tuning.")

    # Precompute per-window M (use cached M_raw if present)
    def _M_for(w):
        if "M_raw" in w and w["M_raw"] is not None:
            return int(w["M_raw"])
        return int(count_snps_in_window(
            bim, w["chrom"], w["start"], w.get("end", w["start"]),
            window_size=window_size, use_window=use_window
        ))
    windows_M = [(w, _M_for(w)) for w in tuning_windows]
    windows_M = [(w, M) for (w, M) in windows_M if M > 1]
    if not windows_M:
        raise ValueError("No tuning windows with M > 1.")

    combos = list(it.product(
        grid["c_lambda"], grid["c_ridge"], grid["subsample_frac"], grid["batch_size"]
    ))

    # Decide execution mode
    created_cluster = False
    client = dask_client
    if client is None and use_multi_gpu:
        try:
            from numba import cuda
            n_gpus = len(cuda.gpus)
        except Exception:
            n_gpus = 0
        if n_gpus > 1:
            from dask_cuda import LocalCUDACluster
            from dask.distributed import Client
            cluster = LocalCUDACluster(rmm_pool_size=rmm_pool_size, 
                                       threads_per_worker=1, 
                                       rmm_async=True)
            client = Client(cluster)
            created_cluster = True
            if max_in_flight is None:
                max_in_flight = 2 * n_gpus

    def _eval_combo_serial(c_lam, c_ridge, sub, bs):
        scores = []
        for w, M in windows_M:
            alpha, l1r = enet_from_targets(M, N, c_lambda=c_lam, c_ridge=c_ridge)
            res = run_single_window(
                chrom=w["chrom"], start=w["start"], end=w.get("end", w["start"]),
                geno_arr=geno_arr, bim=bim, fam=fam,
                geno_path=w.get("geno_path"), pheno=w.get("pheno"),
                pheno_path=w.get("pheno_path"), pheno_id=w.get("pheno_id"),
                has_header=w.get("has_header", True), y_pos=w.get("y_pos"),
                error_regions=error_regions, outdir=outdir,
                window_size=window_size, by_hand=by_hand, use_window=use_window,
                n_trials=1, n_iter=100, batch_size=min(bs, M),
                fixed_alpha=alpha, fixed_l1_ratio=l1r, fixed_subsample=sub,
                early_stop=early_stop, save_full=False
            )
            if res is not None and np.isfinite(res.get("final_r2", np.nan)):
                scores.append(float(res["final_r2"]))
        return float(np.nanmedian(scores)) if scores else -np.inf

    def _eval_combo_dask(c_lam, c_ridge, sub, bs):
        from dask.distributed import as_completed, Future
        # Scatter big read-only objects once
        geno_f = _maybe_scatter(client, geno_arr)
        bim_f  = _maybe_scatter(client, bim)
        fam_f  = _maybe_scatter(client, fam)

        futs = []
        scores = []
        for w, M in windows_M:
            # basic back-pressure
            if max_in_flight and len(futs) >= max_in_flight:
                f, r = next(as_completed(futs, with_results=True))
                futs.remove(f)
                if r is not None and np.isfinite(r.get("final_r2", np.nan)):
                    scores.append(float(r["final_r2"]))

            alpha, l1r = enet_from_targets(M, N, c_lambda=c_lam, c_ridge=c_ridge)
            futs.append(client.submit(
                run_single_window,
                chrom=w["chrom"], start=w["start"], end=w.get("end", w["start"]),
                geno_arr=geno_f, bim=bim_f, fam=fam_f,
                geno_path=w.get("geno_path"), pheno=w.get("pheno"),
                pheno_path=w.get("pheno_path"), pheno_id=w.get("pheno_id"),
                has_header=w.get("has_header", True), y_pos=w.get("y_pos"),
                error_regions=error_regions, outdir=outdir,
                window_size=window_size, by_hand=by_hand, use_window=use_window,
                n_trials=1, n_iter=100, batch_size=min(bs, M),
                fixed_alpha=alpha, fixed_l1_ratio=l1r, fixed_subsample=sub,
                early_stop=early_stop, save_full=False,
                pure=False
            ))

        # Drain remaining
        from dask.distributed import as_completed
        for f, r in as_completed(futs, with_results=True):
            if r is not None and np.isfinite(r.get("final_r2", np.nan)):
                scores.append(float(r["final_r2"]))

        return float(np.nanmedian(scores)) if scores else -np.inf

    # Main loop
    best, best_score = None, -np.inf
    for (c_lam, c_ridge, sub, bs) in combos:
        score = _eval_combo_dask(c_lam, c_ridge, sub, bs) if client else _eval_combo_serial(c_lam, c_ridge, sub, bs)
        if score > best_score:
            best_score = score
            best = dict(
                c_lambda=float(c_lam),
                c_ridge=float(c_ridge),
                subsample_frac=float(sub),
                batch_size=int(bs)
            )

    if created_cluster:
        client.close() # Should also close cluster

    if best is None:
        raise RuntimeError("Global tuning failed—no valid scores.")
    best["score"] = float(best_score)
    return best


def _maybe_scatter(client, x, broadcast=True):
    if client is None or x is None:
        return x
    from dask.distributed import Future
    return x if isinstance(x, Future) else client.scatter(x, broadcast=broadcast)


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
