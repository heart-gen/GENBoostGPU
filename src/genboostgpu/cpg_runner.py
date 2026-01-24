"""
CpG-level heritability estimation runner.

This module mirrors vmr_runner.py but with CpG-specific semantics:
- Input: (cpg_id, chrom, cpg_pos) instead of (chrom, start, end)
- Phenotype lookup: by cpg_id from pheno_df or pheno_path
- Window: cis-window centered on cpg_pos
"""
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
    "run_single_cpg",
]


def run_single_cpg(
    cpg_id, chrom, cpg_pos,
    # Data sources (same pattern as vmr_runner)
    geno_arr=None, bim=None, fam=None, geno_path=None,
    pheno=None, pheno_df=None, pheno_path=None,
    # Error filtering
    error_regions=None,
    # Window/preprocessing
    window_size=500_000,
    by_hand=False,
    batch_size=8192,
    # Hyperparameters
    n_trials=20, n_iter=100,
    fixed_alpha=None, fixed_l1_ratio=None, fixed_subsample=None,
    # Early stopping
    early_stop=None,
    # Working set
    working_set=None,
    # Output
    outdir="results", save_full=False
):
    """
    Run boosting elastic net for one CpG site.

    Parameters
    ----------
    cpg_id : str
        Unique identifier for the CpG site
    chrom : int
        Chromosome number (1-22)
    cpg_pos : int
        Genomic position of the CpG site

    Data sources (provide one of each group):
        geno_arr, bim, fam : pre-loaded genotype data
        geno_path : str, PLINK prefix to load genotypes

        pheno : cupy array, pre-extracted phenotype vector for this CpG
        pheno_df : cudf/pandas DataFrame with cpg_id as column
        pheno_path : str, path to phenotype file (will load and extract cpg_id column)

    error_regions : pd.DataFrame, optional
        Blacklist regions with columns: cpg_id or (Chrom, Pos)
    window_size : int
        Size of cis-window around CpG position (default 500kb)
    by_hand : bool
        If True, use manual preprocessing steps (like vmr_runner)
    batch_size : int
        Batch size for correlation computation
    n_trials : int
        Number of Optuna trials for hyperparameter tuning
    n_iter : int
        Number of boosting iterations
    fixed_alpha, fixed_l1_ratio, fixed_subsample : float, optional
        Fixed hyperparameters (skip tuning if provided)
    early_stop : dict, optional
        Early stopping config: {patience, min_delta, warmup, metric}
    working_set : dict, optional
        Working set config: {K, refresh}
    outdir : str
        Output directory for results
    save_full : bool
        If True, save full beta coefficients and h2 history

    Returns
    -------
    dict or None
        Dictionary with CpG summary metrics:
        - cpg_id: CpG identifier
        - chrom: chromosome
        - cpg_pos: genomic position
        - num_snps: number of SNPs after preprocessing
        - N: number of samples
        - final_r2: final R-squared value
        - h2_unscaled: calibrated heritability estimate
        - n_iter: number of boosting iterations run
        Returns None if CpG is skipped (blacklisted or no SNPs)
    """
    # Load genotypes if not passed in
    if geno_arr is None or bim is None or fam is None:
        if geno_path is None:
            raise ValueError("Either geno_arr+bim+fam or geno_path must be provided")
        geno_arr, bim, fam = load_genotypes(str(geno_path))

    # Load/extract phenotype
    if pheno is None:
        if pheno_df is not None:
            # Extract from DataFrame by cpg_id column
            if hasattr(pheno_df, 'to_cupy'):
                # cudf DataFrame
                pheno = pheno_df[cpg_id].to_cupy()
            else:
                # pandas DataFrame
                pheno = cp.asarray(pheno_df[cpg_id].values)
        elif pheno_path is not None:
            # Load from file and extract cpg_id column
            df = load_phenotypes(str(pheno_path), header=True)
            if cpg_id not in df.columns:
                raise ValueError(f"cpg_id '{cpg_id}' not found in phenotype file")
            pheno = df[cpg_id].to_cupy()
        else:
            raise ValueError("Either pheno array, pheno_df, or pheno_path must be provided")

    y = pheno.astype(cp.float32)

    # Skip blacklist if provided
    if error_regions is not None:
        # Check if cpg_id is in blacklist
        if "cpg_id" in error_regions.columns:
            if cpg_id in error_regions["cpg_id"].values:
                print(f"Skipping blacklisted CpG: {cpg_id}")
                return None
        # Or check by position
        elif "Chrom" in error_regions.columns and "Pos" in error_regions.columns:
            mask = (error_regions["Chrom"] == chrom) & (error_regions["Pos"] == cpg_pos)
            if mask.any():
                print(f"Skipping blacklisted CpG: {cpg_id} ({chrom}:{cpg_pos})")
                return None

    # Filter cis window around CpG position
    # Note: for CpG, we pass cpg_pos as both start and end (single position)
    result = filter_cis_window(
        geno_arr, bim, chrom, cpg_pos,
        window_size=window_size, use_window=True
    )
    if result is None:
        return None
    X, snps, snp_pos = result
    if X is None or len(snps) == 0:
        return None

    # Preprocess genotypes
    if by_hand:
        X, snps, snp_pos = filter_zero_variance(X, snps, snp_pos)
        if X is None or X.shape[1] == 0:
            return None
        X = impute_snps(X)
        stat = _corr_with_y_streaming(X, y, batch_size)
        keep_idx = run_ld_clumping(X, snp_pos, stat, r2_thresh=0.2)
        if keep_idx.size == 0:
            return None
        X = X[:, keep_idx]
        snps = [snps[i] for i in keep_idx.tolist()]
    else:
        result = preprocess_genotypes(
            X, snps, snp_pos, y, r2_thresh=0.2, batch_size=batch_size
        )
        if result is None:
            return None
        X, snps = result

    if X is None or X.shape[1] == 0:
        return None

    N, M = X.shape

    # Use fixed hyperparameters if provided
    use_fixed = (fixed_alpha is not None) or (fixed_l1_ratio is not None) or (fixed_subsample is not None)
    local_n_trials = 1 if use_fixed else n_trials

    # Booster kwargs (only include keys that are set)
    enet_kwargs = {}
    if fixed_alpha is not None:
        enet_kwargs["fixed_alpha"] = float(fixed_alpha)
    if fixed_l1_ratio is not None:
        enet_kwargs["fixed_l1_ratio"] = float(fixed_l1_ratio)
    if fixed_subsample is not None:
        enet_kwargs["fixed_subsample_frac"] = float(fixed_subsample)
    if working_set is not None:
        enet_kwargs["working_set"] = working_set

    if early_stop is not None:
        if "patience" in early_stop:
            enet_kwargs["patience"] = int(early_stop["patience"])
        if "min_delta" in early_stop:
            enet_kwargs["min_delta"] = float(early_stop["min_delta"])
        if "warmup" in early_stop:
            enet_kwargs["warmup"] = int(early_stop["warmup"])
        if "metric" in early_stop:
            enet_kwargs["early_stop_metric"] = str(early_stop["metric"])

    # Default calibration
    if early_stop is None or "metric" not in early_stop:
        enet_kwargs["early_stop_metric"] = "val_r2"
    enet_kwargs.setdefault("val_frac", 0.20)

    # Run boosting elastic net
    results = boosting_elastic_net(
        X, y, snp_ids=snps, n_iter=n_iter,
        n_trials=local_n_trials,
        batch_size=min(int(batch_size), M),
        **enet_kwargs
    )

    # Choose calibrated h2 to report under legacy key
    # Priority: h2_val > h2_ld > h2_unscaled
    h2_legacy = results.get("h2_val", None)
    if h2_legacy is None or (isinstance(h2_legacy, float) and np.isnan(h2_legacy)):
        h2_legacy = results.get("h2_ld", results.get("h2_unscaled", np.nan))

    try:
        h2_legacy = float(h2_legacy)
    except Exception:
        h2_legacy = np.nan

    if h2_legacy is not np.nan:
        h2_legacy = float(max(0.0, min(1.0, h2_legacy)))

    # Save results if requested
    if save_full:
        Path(outdir).mkdir(parents=True, exist_ok=True)
        out_prefix = Path(outdir) / f"{cpg_id}_chr{chrom}_{cpg_pos}"
        save_results(
            results["ridge_betas_full"], results["h2_estimates"],
            str(out_prefix), snp_ids=results["snp_ids"]
        )

    return {
        "cpg_id": cpg_id,
        "chrom": chrom,
        "cpg_pos": cpg_pos,
        "num_snps": M,
        "N": N,
        "final_r2": results.get("final_r2"),
        "h2_unscaled": h2_legacy,
        "n_iter": len(results.get("h2_estimates", [])),
    }
