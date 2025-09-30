import cupy as cp
import numpy as np
from pathlib import Path
from .data_io import load_genotypes, load_phenotypes, save_results
from .snp_processing import (
    filter_zero_variance, impute_snps,
    run_ld_clumping, filter_cis_window,
    preprocess_genotypes
)
from .enet_boosting import boosting_elastic_net

__all__ = [
    "run_single_window",
]

def run_single_window(chrom, start, end, has_header=True, y_pos=None,
                      geno_arr=None, bim=None, fam=None, geno_path=None,
                      pheno=None, pheno_path=None, pheno_id=None,
                      error_regions=None, outdir="results", window_size=500_000,
                      by_hand=False, n_trials=20, n_iter=100, use_window=True):
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
        if y_pos is not None:
            pheno = df.iloc[:, y_pos].to_cupy()
        else:
            pheno = df[pheno_id].to_cupy()

    y = (pheno - pheno.mean()) / (pheno.std() + 1e-6)

    # Skip blacklist if provided
    if error_regions is not None:
        mask = (error_regions["Chrom"] == chrom) & \
               (error_regions["Start"] == start) & \
               (error_regions["End"] == end)
        if mask.any():
            print(f"Skipping blacklisted region: {chrom}:{start}-{end}")
            return None

    # Filter cis window
    X, snps, snp_pos = filter_cis_window(geno_arr, bim, chrom, start, end,
                                         window_size=window_size,
                                         use_window=use_window)
    if X is None or len(snps) == 0:
        return None

    # Preprocess
    if by_hand:
        X, snps = filter_zero_variance(X, snps)
        X = impute_snps(X)
        snp_pos = [snp_pos[i] for i, sid in enumerate(snps)]
        stat = cp.abs(cp.corrcoef(X.T, y)[-1, :-1])
        keep_idx = run_ld_clumping(X, snp_pos, stat, r2_thresh=0.2)
        if keep_idx.size == 0:
            return None
        X = X[:, keep_idx]
        snps = [snps[i] for i in keep_idx.tolist()]
    else:
        X, snps = preprocess_genotypes(X, snps, snp_pos, y, r2_thresh=0.2)

    # Run boosting EN
    results = boosting_elastic_net(
        X, y, snps, n_iter=n_iter, n_trials=n_trials,
        batch_size=min(1000, X.shape[1])
    )

    # Save + return
    Path(outdir).mkdir(parents=True, exist_ok=True)
    out_prefix = Path(outdir) / f"{pheno_id}_chr{chrom}_{start}_{end}"
    save_results(results["ridge_betas_full"],
                 results["h2_estimates"], str(out_prefix),
                 snp_ids=results["snp_ids"])

    return {
        "chrom": chrom,
        "start": start,
        "end": end,
        "num_snps": X.shape[1],
        "final_r2": results["final_r2"],
        "h2_unscaled": results["h2_unscaled"],
        "n_iter": len(results["h2_estimates"]),
    }
