import os
import cupy as cp
import pandas as pd
from dask import delayed

from .enet_boosting import boosting_elastic_net
from .snp_processing import preprocess_genotypes, filter_cis_window

__all__ = [
    "prepare_cpg_inputs",
    "run_boosting_for_cpgs",
    "run_boosting_for_cpg_delayed",
]

def prepare_cpg_inputs(cpg_list, geno_arr, pheno_df, bim,
                       window_size=20_000, var_thresh=1e-6,
                       impute_strategy="most_frequent",
                       r2_thresh=0.1):
    """
    Prepare genotype (Xs) and phenotype (ys) matrices for a list of CpGs.
    """
    inputs = []

    for cpg_id, chrom, cpg_pos in cpg_list:
        # Phenotype vector
        y = pheno_df[cpg_id].to_cupy()

        # Extract cis SNPs
        geno_arr, snp_ids, snp_pos = filter_cis_window(geno_arr, bim,
                                                       chrom, cpg_pos,
                                                       window_size)
        if geno_arr is None or len(snp_ids) == 0:
            continue

        # Filter zero-variance SNPs
        X, snp_ids = preprocess_genotypes(geno_arr, snp_ids, snp_pos, y,
                                          var_thresh=var_thresh,
                                          impute_strategy=impute_strategy,
                                          r2_thresh=r2_thresh)

        if X.shape[1] == 0:
            continue

        inputs.append((cpg_id, X, y, snp_ids))

    return inputs


def run_boosting_for_cpgs(inputs, n_iter=50, batch_size=500,
                          alphas=[0.1, 0.5, 1.0],
                          l1_ratios=[0.1, 0.5, 0.9],
                          ridge_alphas=[0.1, 1.0, 10.0],
                          cv=5, refit_each_iter=False,
                          standardize=True):
    """
    Run boosting_elastic_net for many CpGs.

    Parameters
    ----------
    inputs : list
        Output from prepare_cpg_inputs -> [(cpg_id, X, y, snp_ids), ...].
    (other args)
        Passed to boosting_elastic_net.

    Returns
    -------
    results : list
        Each element is a dict from boosting_elastic_net with an extra key "cpg_id".
    """
    results = []
    for cpg_id, X, y, snp_ids in inputs:
        res = boosting_elastic_net(
            X, y, snp_ids,
            n_iter=n_iter, batch_size=batch_size,
            alphas=alphas, l1_ratios=l1_ratios,
            ridge_alphas=ridge_alphas, cv=cv,
            refit_each_iter=refit_each_iter,
            standardize=standardize
        )
        res["cpg_id"] = cpg_id
        results.append(res)
    return results


@delayed
def run_boosting_for_cpg_delayed(cpg_id, X, y, snp_ids,
                                 outdir="cpg_results",
                                 n_iter=50, batch_size=500,
                                 alphas=[0.1, 0.5, 1.0],
                                 l1_ratios=[0.1, 0.5, 0.9],
                                 ridge_alphas=[0.1, 1.0, 10.0],
                                 cv=5, refit_each_iter=False,
                                 standardize=True,
                                 save_full_betas=False,
                                 overwrite=False):
    """
    Dask-delayed task for one CpG boosting analysis with checkpointing.
    """
    os.makedirs(outdir, exist_ok=True)
    summary_path = os.path.join(outdir, f"{cpg_id}_summary.tsv")
    betas_path = os.path.join(outdir, f"{cpg_id}_betas.tsv")

    if not overwrite and os.path.exists(summary_path):
        return delayed(lambda x: x)(summary_path)

    # Run boosting model
    res = boosting_elastic_net(
        X, y, snp_ids,
        n_iter=n_iter, batch_size=batch_size,
        alphas=alphas, l1_ratios=l1_ratios,
        ridge_alphas=ridge_alphas, cv=cv,
        refit_each_iter=refit_each_iter,
        standardize=standardize
    )
    res["cpg_id"] = cpg_id

    # Save summary
    summary_df = pd.DataFrame([{
        "cpg_id": cpg_id,
        "final_r2": res["final_r2"],
        "h2_unscaled": res["h2_unscaled"],
        "num_snps_kept": len(res["kept_snps"]),
        "best_enet_alpha": res["best_enet"]["alpha"],
        "best_enet_l1_ratio": res["best_enet"]["l1_ratio"],
        "best_ridge_alpha": res["best_ridge"]["alpha"],
    }])
    summary_df.to_csv(summary_path, sep="\t", index=False)

    # Save betas if requested
    if save_full_betas:
        kept_idx = [i for i, snp in enumerate(res["snp_ids"])]

        betas_df = pd.DataFrame({
            "snp_id": res["snp_ids"], # only retained
            "beta_boosting": cp.asnumpy(res["betas_boosting"])[kept_idx],
            "beta_ridge": cp.asnumpy(res["ridge_betas_full"])[kept_idx],
            "snp_variance": cp.asnumpy(res["snp_variances"][kept_idx])
        })
        betas_df["variance_contrib"] = betas_df["beta_ridge"]**2 * betas_df["snp_variance"]
        betas_df.to_csv(betas_path, sep="\t", index=False)

    return summary_path
