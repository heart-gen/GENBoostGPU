import cupy as cp
import numpy as np
from pathlib import Path
from .data_io import load_genotypes, load_phenotypes, save_results
from .snp_processing import (
    filter_zero_variance, impute_snps,
    run_ld_clumping, filter_cis_window,
    preprocess_genotypes, _corr_with_y_streaming
)
from .enet_boosting import boosting_elastic_net

__all__ = [
    "run_single_window",
]

def run_single_window(chrom, start, end, has_header=True, y_pos=None,
                      geno_arr=None, bim=None, fam=None, geno_path=None, pheno=None,
                      pheno_path=None, pheno_id=None, batch_size=8192,
                      error_regions=None, outdir="results", window_size=500_000,
                      by_hand=False, n_trials=20, n_iter=100, use_window=True,
                      fixed_alpha=None, fixed_l1_ratio=None, fixed_subsample=None,
                      early_stop=None, working_set=None, save_full=True):
    """
    Run boosting elastic net for one genomic window.

    Supports either:
      - Pre-loaded genotype and phenotype arrays (geno_arr, pheno)
      - File paths (geno_path, pheno_path + pheno_id)

    Returns:
        dict with window summary metrics (or None if skipped)
    """
    # Load genotypes if not passed in
    if geno_arr is None or bim is None or fam is None:
        if geno_path is None:
            raise ValueError("Either geno_arr+bim+fam or geno_path must be provided")
        geno_arr, bim, fam = load_genotypes(str(geno_path))

    # Load phenotype if not passed in
    if pheno is None:
        if pheno_path is None:
            raise ValueError("Either pheno array or pheno_path must be provided")
        df = load_phenotypes(str(pheno_path), header=has_header)
        if pheno_id is None:
            raise ValueError("pheno_id required if using pheno_path")
        pheno = (df.iloc[:, y_pos].to_cupy() if y_pos is not None else df[pheno_id].to_cupy())

    y = pheno.astype(cp.float32) # No standardization here

    # Skip blacklist if provided
    if error_regions is not None:
        mask = (error_regions["Chrom"] == chrom) & \
               (error_regions["Start"] == start) & \
               (error_regions["End"] == end)
        if mask.any():
            print(f"Skipping blacklisted region: {chrom}:{start}-{end}")
            return None

    # Filter cis window
    X, snps, snp_pos = filter_cis_window(
        geno_arr, bim, chrom, start, end,
        window_size=window_size, use_window=use_window
    )
    if X is None or len(snps) == 0:
        return None

    # Preprocess
    if by_hand:
        X, snps, snp_pos = filter_zero_variance(X, snps, snp_pos)
        X = impute_snps(X)
        stat = _corr_with_y_streaming(X, y, batch_size)
        # stat = cp.abs(cp.corrcoef(X.T, y)[-1, :-1])
        keep_idx = run_ld_clumping(X, snp_pos, stat, r2_thresh=0.2)
        if keep_idx.size == 0:
            return None
        X = X[:, keep_idx]
        snps = [snps[i] for i in keep_idx.tolist()]
    else:
        X, snps = preprocess_genotypes(
            X, snps, snp_pos, y, r2_thresh=0.2, batch_size=batch_size
        )
    
    N, M = X.shape
    # Use fixed hyperparameters if provided
    use_fixed = (fixed_alpha is not None) or (fixed_l1_ratio is not None) or (fixed_subsample is not None)
    local_n_trials = 1 if use_fixed else n_trials

    # Booster kwargs (only include keys that are set)
    enet_kwargs = {}
    if fixed_alpha is not None:       enet_kwargs["fixed_alpha"] = float(fixed_alpha)
    if fixed_l1_ratio is not None:    enet_kwargs["fixed_l1_ratio"] = float(fixed_l1_ratio)
    if fixed_subsample is not None:   enet_kwargs["fixed_subsample_frac"] = float(fixed_subsample)
    if working_set is not None:       enet_kwargs["working_set"] = working_set

    if early_stop is not None:
        if "patience"    in early_stop: enet_kwargs["patience"]    = int(early_stop["patience"])
        if "min_delta"   in early_stop: enet_kwargs["min_delta"]   = float(early_stop["min_delta"])
        if "warmup"      in early_stop: enet_kwargs["warmup"]      = int(early_stop["warmup"])
        if "metric"      in early_stop: enet_kwargs["early_stop_metric"] = str(early_stop["metric"])

    # Default calibration
    if early_stop is None or "metric" not in early_stop:
        enet_kwargs["early_stop_metric"] = "val_r2"
    enet_kwargs.setdefault("val_frac", 0.20)

    # Run boosting EN
    results = boosting_elastic_net(
        X, y, snp_ids=snps, n_iter=n_iter, 
        n_trials=local_n_trials,
        batch_size=min(int(batch_size), M),
        **enet_kwargs
    )

    # Choose calibrated h2 to report under legacy key
    h2_legacy = results.get("h2_val", None)
    if h2_legacy is None or (isinstance(h2_legacy, float) and np.isnan(h2_legacy)):
        h2_legacy = results.get("h2_ld", results.get("h2_unscaled", np.nan))

    try:
        h2_legacy = float(h2_legacy)
    except Exception:
        h2_legacy = np.nan

    if h2_legacy is not np.nan:
        h2_legacy = float(max(0.0, min(1.0, h2_legacy)))

    # Save + return
    if save_full:
        Path(outdir).mkdir(parents=True, exist_ok=True)
        out_prefix = Path(outdir) / f"{pheno_id}_chr{chrom}_{start}_{end}"
        save_results(
            results["ridge_betas_full"], results["h2_estimates"], 
            str(out_prefix), snp_ids=results["snp_ids"]
        )

    return {
        "chrom": chrom, "start": start, "end": end, "num_snps": M,
        "N": N, "final_r2": results.get("final_r2"),
        "h2_unscaled": h2_legacy,
        "n_iter": len(results.get("h2_estimates", [])),
    }
