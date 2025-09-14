import os
import cudf
import cupy as cp
import numpy as np
import pandas as pd
from pyhere import here
from pathlib import Path
from genboostgpu.data_io import load_genotypes, load_phenotypes, save_results
from genboostgpu.snp_processing import (
    filter_zero_variance, impute_snps,
    run_ld_clumping, filter_cis_window,
    preprocess_genotypes
)
from genboostgpu.enet_boosting import boosting_elastic_net

def get_error_list(error_file="../_h/snp-error-window.tsv"):
    if Path(error_file).exists():
        df = pd.read_csv(error_file, sep='\t')
        df["Chrom"] = df["Chr"].str.replace("chr", "", regex=False).astype(int)
        return df[["Chrom", "Start", "End"]]
    else:
        print(f"Warning: Error regions file not found: {error_file}")
        return pd.DataFrame(columns=["Chrom", "Start", "End"])


def check_if_blacklisted(chrom, start, end, error_regions):
    if error_regions.empty:
        return False
    mask = (error_regions["Chrom"] == chrom) & \
           (error_regions["Start"] == start) & \
           (error_regions["End"] == end)
    if mask.any():
        print(f"Skipping blacklisted region: {chrom}:{start}-{end}")
        return True
    return False


def get_vmr_list(region):
    base_dir = Path(here("heritability/gcta")) / f"{region.lower()}" / "_m"
    vmr_file = base_dir / "vmr_list.txt"
    if not vmr_file.exists():
        raise FileNotFoundError(f"VMR list file not found: {vmr_file}")
    return pd.read_csv(vmr_file, sep="\t", header=None)


def construct_data_path(chrom, start, end, region, dtype="plink"):
    chrom_dir = f"chr_{chrom}"
    base_dir = Path(here("heritability/gcta")) / f"{region.lower()}" / "_m"
    if dtype.lower() == "plink":
        fn = f"subset_TOPMed_LIBD.AA.{start}_{end}.bed"
        return base_dir / "plink_format" / chrom_dir / fn
    elif dtype.lower() == "vmr":
        fn = f"{start}_{end}_meth.phen"
        return base_dir / "vmr" / chrom_dir / fn
    else:
        raise ValueError(f"Unknown dtype: {dtype}")


def run_single_vmr(region, chrom, start, end, error_regions,
                   outdir="results", window=20_000, by_hand=False):
    # skip blacklist
    if check_if_blacklisted(chrom, start, end, error_regions):
        return None

    print(f"Running VMR: chr{chrom}:{start}-{end}")

    # load genotype + bim/fam
    geno_path = construct_data_path(chrom, start, end, region, "plink")
    geno_df, bim, fam = load_genotypes(str(geno_path))

    # load phenotype
    pheno_path = construct_data_path(chrom, start, end, region, "vmr")
    y = load_phenotypes(str(pheno_path), header=False).iloc[:, 2].to_cupy()
    y = (y - y.mean()) / (y.std() + 1e-6)

    # filter cis window
    geno_window, snps, snp_pos = filter_cis_window(geno_df, bim, chrom, start,
                                                   window=window)
    if geno_window is None or len(snps) == 0:
        print("No SNPs in window")
        return None

    # Convert so NaNs are perserved
    if hasattr(geno_window, "to_numpy"):
        geno_arr = geno_window.to_arrow().to_pandas().to_numpy(dtype="float32")
    else:
        geno_arr = geno_window.to_numpy(dtype="float32")

    X = cp.asarray(geno_arr)

    # preprocess
    if by_hand:
        X, snps = filter_zero_variance(X, snps)
        X = impute_snps(X)

        # subset SNPs based on filtering
        snp_pos = [snp_pos[i] for i, sid in enumerate(snps)]

        # LD clumping (phenotype-informed)
        stat = cp.abs(cp.corrcoef(X.T, y)[-1, :-1])
        keep_idx = run_ld_clumping(X, snp_pos, stat, r2_thresh=0.2,
                                   kb_window=window)
        if keep_idx.size == 0:
            print("No SNPs left after clumping")
            return None

        X = X[:, keep_idx]
        snps = [snps[i] for i in keep_idx.tolist()]
    else:
        X, snps = preprocess_genotypes(X, snps, snp_pos, y, kb_window=window)

    # boosting elastic net
    results = boosting_elastic_net(X, y, snps, n_iter=100,
                                   alphas=np.arange(0.05, 1.0 + 0.05, 0.05),
                                   batch_size=min(1000, X.shape[1]))

    # write results
    out_prefix = Path(outdir) / f"chr{chrom}_{start}_{end}"
    Path(outdir).mkdir(parents=True, exist_ok=True)
    save_results(results["ridge_betas_full"],
                 results["h2_estimates"], str(out_prefix),
                 snp_ids=results["snp_ids"])

    # summary
    summary = {
        'chrom': chrom,
        'start': start,
        'end': end,
        'num_snps': X.shape[1],
        'final_r2': results["final_r2"],
        'h2_unscaled': results["h2_unscaled"],
        "n_iter": len(results["h2_estimates"]),
        }
    return summary


def main():
    region = os.environ.get("REGION")
    if not region:
        raise ValueError("REGION environment variable must be set")

    error_regions = get_error_list()
    vmr_list = get_vmr_list(region)

    all_summaries = []
    for _, row in vmr_list.iterrows():
        chrom, start, end = row[0], row[1], row[2]
        summary = run_single_vmr(region, chrom, start, end,
                                 error_regions, window=500_000)
        if summary:
            all_summaries.append(summary)

    if all_summaries:
        df = pd.DataFrame(all_summaries)
        df.to_csv("results/summary_all_vmrs.tsv", sep="\t", index=False)


if __name__ == "__main__":
    main()
