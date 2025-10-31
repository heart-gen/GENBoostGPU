import random, os
from pathlib import Path

import cupy as cp
import numpy as np
import pandas as pd
from pyhere import here

from genboostgpu.hyperparams import enet_from_targets
from genboostgpu.orchestration import run_windows_with_dask
from genboostgpu.snp_processing import count_snps_in_window
from genboostgpu.tuning import global_tune_params, select_tuning_windows

random.seed(42)
np.random.seed(42)
cp.random.seed(42)

WINDOW_SIZE = 500_000
EARLY_STOP = {"patience": 5, "min_delta": 1e-4, "warmup": 5}

def get_error_list(error_file="../_h/snp-error-window.tsv"):
    error_path = Path(error_file)
    if error_path.exists():
        df = pd.read_csv(error_path, sep="\t")
        df["Chrom"] = df["Chr"].str.replace("chr", "", regex=False).astype(int)
        return df[["Chrom", "Start", "End"]]
    print(f"Warning: Error regions file not found: {error_file}")
    return pd.DataFrame(columns=["Chrom", "Start", "End"])


def get_vmr_list(region: str) -> pd.DataFrame:
    base_dir = Path(here("heritability/gcta")) / region.lower() / "_m"
    vmr_file = base_dir / "vmr_list.txt"
    if not vmr_file.exists():
        raise FileNotFoundError(f"VMR list file not found: {vmr_file}")
    return pd.read_csv(vmr_file, sep="\t", header=None)


def load_bim(geno_path):
    bim_path = Path(geno_path).with_suffix(".bim")
    if not bim_path.exists():
        raise FileNotFoundError(f"Missing BIM file for window: {bim_path}")
    cols = ["chrom", "snp", "cm", "pos", "a1", "a2"]
    bim = pd.read_csv(bim_path, sep=r"\s+", header=None, names=cols)
    bim["chrom"] = bim["chrom"].astype(str)
    bim["pos"] = bim["pos"].astype(int)
    return bim[["chrom", "snp", "pos"]]


def load_fam(geno_path):
    fam_path = Path(geno_path).with_suffix(".fam")
    if not fam_path.exists():
        raise FileNotFoundError(f"Missing FAM file for window: {fam_path}")
    cols = ["fid", "iid", "father", "mother", "sex", "phenotype"]
    fam = pd.read_csv(fam_path, sep=r"\s+", header=None, names=cols)
    return fam


def construct_data_path(chrom, start, end, region, dtype="plink"):
    chrom_dir = f"chr_{chrom}"
    base_dir = Path(here("heritability/gcta")) / region.lower() / "_m"
    if dtype.lower() == "plink":
        fn = f"subset_TOPMed_LIBD.AA.{start}_{end}.bed"
        return base_dir / "plink_format" / chrom_dir / fn
    if dtype.lower() == "vmr":
        fn = f"{start}_{end}_meth.phen"
        return base_dir / "vmr" / chrom_dir / fn
    raise ValueError(f"Unknown dtype: {dtype}")


def read_bim(geno_path: Path) -> pd.DataFrame:
    bim_path = geno_path.with_suffix(".bim")
    if not bim_path.exists():
        raise FileNotFoundError(f"Missing BIM file for {geno_path}")
    cols = ["chrom", "snp", "cm", "pos", "a1", "a2"]
    bim = pd.read_csv(bim_path, sep=r"\s+", header=None, names=cols)
    bim["chrom"] = bim["chrom"].astype(int)
    bim["pos"] = bim["pos"].astype(int)
    return bim


def read_fam(geno_path: Path) -> pd.DataFrame:
    fam_path = geno_path.with_suffix(".fam")
    if not fam_path.exists():
        raise FileNotFoundError(f"Missing FAM file for {geno_path}")
    cols = ["family_id", "individual_id", "paternal_id", "maternal_id", "sex", "phenotype"]
    fam = pd.read_csv(fam_path, sep=r"\s+", header=None, names=cols)
    return fam


def build_windows(region: str, vmr_list: pd.DataFrame):
    windows = []
    bim_tables = []
    fam = None

    for i, row in vmr_list.iterrows():
        chrom, start, end = int(row[0]), int(row[1]), int(row[2])
        geno_path = Path(construct_data_path(chrom, start, end, region, "plink"))
        pheno_path = Path(construct_data_path(chrom, start, end, region, "vmr"))

        bim = read_bim(geno_path)
        bim_tables.append(bim[["chrom", "snp", "pos"]])

        if fam is None:
            fam = read_fam(geno_path)

        windows.append({
            "chrom": chrom,
            "start": start,
            "end": end,
            "pheno_id": f"vmr_{i + 1}",
            "geno_path": str(geno_path),
            "pheno_path": str(pheno_path),
            "has_header": False,
            "y_pos": 2,
            "M_raw": int(len(bim)),
        })

    if not windows:
        raise ValueError("No VMR windows were discovered—check input files.")

    bim_df = pd.concat(bim_tables, axis=0, ignore_index=True).drop_duplicates()
    return windows, bim_df, fam


def make_fixed_fn(*, bim, N, c_lambda, c_ridge, window_size=500_000, use_window=True):
    if N <= 0:
        raise ValueError("Unable to infer sample size from the FAM file.")

    def _fn(window):
        M = int(window.get("M_raw") or count_snps_in_window(
            bim,
            window["chrom"],
            window["start"],
            window.get("end", window["start"]),
            window_size=window_size,
            use_window=use_window,
        ))
        if M <= 1:
            return {}
        alpha, l1_ratio = enet_from_targets(M, N, c_lambda=c_lambda, c_ridge=c_ridge)
        return {"fixed_alpha": alpha, "fixed_l1_ratio": l1_ratio}

    return _fn


def main():
    region = os.environ.get("REGION")
    if not region:
        raise ValueError("REGION environment variable must be set")

    error_regions = get_error_list()
    vmr_list = get_vmr_list(region)
    windows, bim, fam = build_windows(region, vmr_list)

    tuning_windows = select_tuning_windows(
        windows, bim, frac=0.05, n_min=40, n_max=200,
        window_size=500_000, use_window=True,
        n_bins=3, per_chrom_min=1, seed=13,
    )

    best = global_tune_params(
        tuning_windows=tuning_windows, bim=bim, fam=fam,
        window_size=500_000, use_window=True,
        grid={
            "c_lambda": [0.5, 0.7, 1.0, 1.4, 2.0],
            "c_ridge": [0.5, 1.0, 2.0],
            "subsample_frac": [0.6, 0.7, 0.8],
            "batch_size": [4096, 8192],
        },
        early_stop={"patience": 5, "min_delta": 1e-4, "warmup": 5},
        use_window=True,
    )

    print("Best global parameters:", best)

    fixed_fn = make_fixed_fn(
        bim=bim,
        N=len(fam) if fam is not None else 0,
        c_lambda=best["c_lambda"],
        c_ridge=best["c_ridge"],
        window_size=500_000,
        use_window=True,
    )

    df = run_windows_with_dask(
        windows, error_regions=error_regions, outdir="results",
        window_size=500_000, use_window=True,
        batch_size=best["batch_size"], fixed_params=fixed_fn,
        fixed_subsample=best["subsample_frac"],
        early_stop={"patience": 5, "min_delta": 1e-4, "warmup": 5},
        working_set={"K": 1024, "refresh": 10}, save=True,
        prefix="vmr",
    )

    print(f"Completed {len(df)} VMRs")
    print(df.head())


if __name__ == "__main__":
    main()
