import os
import pandas as pd
from pyhere import here
from pathlib import Path

from genboostgpu.data_io import load_genotypes
from genboostgpu.orchestration import run_windows_with_dask

def construct_data_path(num_samples, dtype="phen"):
    base_dir = Path(here("inputs/simulated-data/_m")) / f"sim_{num_samples}_indiv"
    if dtype.lower() == "plink":
        return base_dir / "plink_sim" / "simulated.bed"
    elif dtype.lower() == "phen":
        return base_dir / "simulated.phen"
    else:
        raise ValueError(f"Unknown dtype: {dtype}")


def get_pheno_loc(num_samples):
    base_dir = Path(here("inputs/simulated-data/_m")) / f"sim_{num_samples}_indiv"
    mapping_file = os.path.join(base_dir, "snp_phenotype_mapping.tsv")

    if not os.path.exists(mapping_file):
        raise FileNotFoundError(f"Mapping file not found: {mapping_file}")

    mapped_df = pd.read_csv(mapping_file, sep="\t", header=0)

    return mapped_df


def main():
    num_samples = os.environ.get("NUM_SAMPLES")

    if not num_samples:
        raise ValueError("NUM_SAMPLES environment variable must be set")

    # load genotype + bim/fam
    geno_path = construct_data_path(num_samples, "plink")
    geno_arr, bim, fam = load_genotypes(str(geno_path))
    print("Loaded genotype matrix")

    # Build windows config list
    pheno_path = construct_data_path(chrom, start, end, region, "vmr")
    windows = []
    for _, row in pheno_loc_df.iterrows():
        windows.append({
            "chrom": row["chrom"],
            "start": row["start"],
            "end": row["end"],
            "pheno_id": row["phenotype_id"],
            "geno_arr": geno_arr,
            "pheno_path": pheno_path,
        })

    # Run with dask orchestration
    df = run_windows_with_dask(
        windows, outdir="results", window_size=500_000,
        n_iter=100, n_trials=15, use_window=False,
        save=True, prefix="simu_100"
    )
    print(f"Completed {len(df)} VMRs")
    print(df.head())


if __name__ == "__main__":
    main()
