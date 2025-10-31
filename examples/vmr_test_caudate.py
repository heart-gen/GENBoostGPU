import os
import random
from pathlib import Path

import cupy as cp
import numpy as np
import pandas as pd
from pyhere import here

from genboostgpu.hyperparams import enet_from_targets
from genboostgpu.orchestration import run_windows_with_dask
from genboostgpu.tuning import select_tuning_windows, global_tune_params

random.seed(42)
np.random.seed(42)
cp.random.seed(42)

WINDOW_SIZE = 500_000
EARLY_STOP = {"patience": 5, "min_delta": 1e-4, "warmup": 5}


def get_error_list(error_file="../_h/snp-error-window.tsv"):
    if Path(error_file).exists():
        df = pd.read_csv(error_file, sep="\t")
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
    base_dir = Path(here("heritability/gcta")) / f"{region.lower()}" / "_m"
    if dtype.lower() == "plink":
        fn = f"subset_TOPMed_LIBD.AA.{start}_{end}.bed"
        return base_dir / "plink_format" / chrom_dir / fn
    elif dtype.lower() == "vmr":
        fn = f"{start}_{end}_meth.phen"
        return base_dir / "vmr" / chrom_dir / fn
    else:
        raise ValueError(f"Unknown dtype: {dtype}")


def build_windows(vmr_list, region):
    windows = []
    bim_frames = []
    fam = None
    for i, row in vmr_list.iterrows():
        chrom, start, end = int(row[0]), int(row[1]), int(row[2])
        geno_path = construct_data_path(chrom, start, end, region, "plink")
        pheno_path = construct_data_path(chrom, start, end, region, "vmr")

        bim = load_bim(geno_path)
        bim_frames.append(bim)
        if fam is None:
            fam = load_fam(geno_path)

        windows.append(
            {
                "chrom": chrom,
                "start": start,
                "end": end,
                "pheno_id": f"vmr_{i + 1}",
                "geno_path": geno_path,
                "pheno_path": pheno_path,
                "has_header": False,
                "y_pos": 2,
                "M_raw": int(len(bim)),
            }
        )

    if not windows:
        raise ValueError("No VMR windows were constructed.")

    bim_all = pd.concat(bim_frames, axis=0, ignore_index=True)
    return windows, bim_all, fam


def make_fixed_fn(*, N, c_lambda, c_ridge):
    def _fn(w):
        M = int(w.get("M_raw", 0))
        if M <= 1:
            return {}
        alpha, l1r = enet_from_targets(M, N, c_lambda=c_lambda, c_ridge=c_ridge)
        return {"fixed_alpha": alpha, "fixed_l1_ratio": l1r}

    return _fn


def main():
    region = os.environ.get("REGION")
    if not region:
        raise ValueError("REGION environment variable must be set")

    error_regions = get_error_list()
    vmr_list = get_vmr_list(region)
    windows, bim, fam = build_windows(vmr_list, region)

    tuning_windows = select_tuning_windows(
        windows,
        bim,
        frac=0.05,
        n_min=30,
        n_max=120,
        window_size=WINDOW_SIZE,
        use_window=True,
        n_bins=4,
        per_chrom_min=1,
        seed=13,
    )
    best = global_tune_params(
        tuning_windows=tuning_windows,
        bim=bim,
        fam=fam,
        window_size=WINDOW_SIZE,
        use_window=True,
        grid={
            "c_lambda": [0.5, 0.7, 1.0, 1.4, 2.0],
            "c_ridge": [0.7, 1.0, 1.4],
            "subsample_frac": [0.6, 0.7, 0.8],
            "batch_size": [4096, 6144],
        },
        early_stop=EARLY_STOP,
        batch_size=4096,
    )

    print("Best global hyperparameters:", best)

    N = len(fam)
    fixed_fn = make_fixed_fn(N=N, c_lambda=best["c_lambda"], c_ridge=best["c_ridge"])

    df = run_windows_with_dask(
        windows,
        error_regions=error_regions,
        outdir="results",
        window_size=WINDOW_SIZE,
        batch_size=best["batch_size"],
        use_window=True,
        fixed_params=fixed_fn,
        fixed_subsample=best["subsample_frac"],
        early_stop=EARLY_STOP,
        working_set={"K": 1024, "refresh": 10},
        prefix=f"vmr_{region.lower()}",
    )
    print(f"Completed {len(df)} VMRs")
    print(df.head())


if __name__ == "__main__":
    main()
