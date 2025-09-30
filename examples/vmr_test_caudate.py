import os
import pandas as pd
from pyhere import here
from pathlib import Path
from genboostgpu.orchestration import run_windows_with_dask

def get_error_list(error_file="../_h/snp-error-window.tsv"):
    if Path(error_file).exists():
        df = pd.read_csv(error_file, sep='\t')
        df["Chrom"] = df["Chr"].str.replace("chr", "", regex=False).astype(int)
        return df[["Chrom", "Start", "End"]]
    else:
        print(f"Warning: Error regions file not found: {error_file}")
        return pd.DataFrame(columns=["Chrom", "Start", "End"])


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


def build_windows(vmr_list):
    # Build windows config list
    windows = []
    for i, row in vmr_list.iterrows():
        chrom, start, end = row[0], row[1], row[2]
        geno_path = construct_data_path(chrom, start, end, region, "plink")
        pheno_path = construct_data_path(chrom, start, end, region, "vmr")
        windows.append({
            "chrom": chrom,
            "start": start,
            "end": end,
            "pheno_id": f"vmr_{i+1}",
            "geno_path": geno_path,
            "pheno_path": pheno_path,
            "has_header": False,
            "y_pos": 2,
        })
    return windows


def main():
    region = os.environ.get("REGION")
    if not region:
        raise ValueError("REGION environment variable must be set")

    error_regions = get_error_list()
    vmr_list = get_vmr_list(region)
    windows = build_windows(vmr_list)

    # Run with dask orchestration
    df = run_windows_with_dask(
        windows, error_regions=error_regions,
        outdir="results", window_size=500_000,
        n_iter=100, n_trials=20, use_window=True, ## To decrease time lower n_trials to 10
        save=True, prefix="vmr"
    )
    print(f"Completed {len(df)} VMRs")
    print(df.head())


if __name__ == "__main__":
    main()
