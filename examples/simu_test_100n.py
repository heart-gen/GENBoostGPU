import os
import session_info
import pandas as pd
from pyhere import here
from pathlib import Path

from genboostgpu.vmr_runner import run_single_window
from genboostgpu.hyperparams import enet_from_targets
from genboostgpu.orchestration import run_windows_with_dask
from genboostgpu.snp_processing import count_snps_in_window
from genboostgpu.data_io import load_genotypes, load_phenotypes
from genboostgpu.tuning import select_tuning_windows, global_tune_params

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


def build_windows(num_samples):
    # Load phenotypes
    pheno_path = construct_data_path(num_samples, "phen")

    # Build windows config list
    pheno_loc_df = get_pheno_loc(num_samples)
    windows = []
    for _, row in pheno_loc_df[0:10].iterrows(): ## Test the first 10
        windows.append({
            "chrom": row["chrom"],
            "start": row["start"],
            "end": row["end"],
            "pheno_id": row["phenotype_id"],
            "pheno_path": pheno_path,
        })
    return windows


def tune_windows(windows, geno_arr, bim, fam):
    tw = select_tuning_windows(
        windows, bim, frac=0.05, n_min=60, n_max=300, window_size=500_000,
        use_window=False, n_bins=3, per_chrom_min=1, seed=13
    )
    # Tune alpha scale only
    best = global_tune_params(
        tuning_windows=tw,
        geno_arr=geno_arr, bim=bim, fam=fam,
        window_size=500_000, by_hand=False, use_window=False,
        grid={
            "c_lambda":       [0.5, 0.7, 1.0, 1.4, 2.0],
            "c_ridge":        [1.0],      # keep EN balance fixed here
            "subsample_frac": [0.7],      # keep fixed for now
            "batch_size":     [4096],     # keep fixed for now
        },
        early_stop={"patience": 5, "min_delta": 1e-4, "warmup": 5},
        batch_size=4096
    )
    print("Best alpha scale:", best)

    return best


def fixed_params_for_window(w, bim, N, c_lambda=None, c_ridge=None):
    ## Needs bim move M
    M = count_snps_in_window(
        bim, w["chrom"], w["start"], w.get("end", w["start"]),
        window_size=500_000, use_window=True
    )
    # Convert global (c_lambda, c_ridge) -> window-specific (alpha, l1_ratio)
    alpha, l1r = enet_from_targets(M, N, c_lambda=c_lambda, c_ridge=c_ridge)
    return alpha, l1r


def make_fixed_fn(*, bim, N, c_lambda, c_ridge, window_size=500_000, use_window=True):
    def _fn(w):
        M = int(w.get("M_raw") or count_snps_in_window(
            bim, w["chrom"], w["start"], w.get("end", w["start"]),
            window_size=window_size, use_window=use_window
        ))
        if M <= 1:
            return {}  # skip fixing if no SNPs
        alpha, l1r = enet_from_targets(M, N, c_lambda=c_lambda, c_ridge=c_ridge)
        return {"fixed_alpha": alpha, "fixed_l1_ratio": l1r}
    return _fn


def main(SINGLE=False):
    num_samples = os.environ.get("NUM_SAMPLES")

    if not num_samples:
        raise ValueError("NUM_SAMPLES environment variable must be set")

    # Load genotypes
    geno_path = construct_data_path(num_samples, "plink")
    geno_arr, bim, fam = load_genotypes(str(geno_path))
    N = len(fam) if fam is not None else int(getattr(geno_arr, "shape", [0])[0])
    
    # Build and tune windows
    windows = build_windows(num_samples)
    best = tune_windows(windows, geno_arr, bim, fam)

    if SINGLE:
        # Test run_single_window
        results = []
        for w in windows:
            alpha, l1r = fixed_params_for_window(w, bim, N,
                                                 c_lambda=best["c_lambda"],
                                                 c_ridge=best["c_ridge"])
            r = run_single_window(
                chrom=w["chrom"], start=w["start"], end=w.get("end", w["start"]),
                geno_arr=geno_arr, bim=bim, fam=fam,
                pheno_path=w["pheno_path"], pheno_id=w["pheno_id"],
                window_size=500_000, use_window=True,
                # fixed hyperparams -> skip per-window tuning
                fixed_alpha=alpha, fixed_l1_ratio=l1r,
                fixed_subsample=best["subsample_frac"],
                batch_size=best["batch_size"], n_trials=1, n_iter=100,
                early_stop={"patience": 5, "min_delta": 1e-4, "warmup": 5},
                save_full=True
            )
            if r is not None:
                results.append(r)
        df = pd.DataFrame([r for r in results if r is not None])
    else:
        # Run with dask orchestration
        fixed_fn = make_fixed_fn(
        bim=bim, N=N, c_lambda=best["c_lambda"],
        c_ridge=best["c_ridge"], window_size=500_000, use_window=False
        )

        df = run_windows_with_dask(
            windows, geno_arr=geno_arr, bim=bim, fam=fam,
            outdir="results", batch_size=best["batch_size"],
            window_size=500_000, use_window=False,
            fixed_params=fixed_fn, fixed_subsample=best["subsample_frac"],
            early_stop={"patience": 5, "min_delta": 1e-4, "warmup": 5},
            working_set={"K": 1024, "refresh": 10},
            prefix="simu_100"
        )

    print(f"Completed {len(df)} phenotypes")
    print(df.head())

    # Session information
    session_info.show()

if __name__ == "__main__":
    main()
