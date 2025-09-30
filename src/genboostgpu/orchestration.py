import pandas as pd
from numba import cuda
from dask import delayed, compute
from dask.distributed import Client
from dask_cuda import LocalCUDACluster
from .vmr_runner import run_single_window

__all__ = [
    "run_windows_with_dask",
]

def run_windows_with_dask(windows, error_regions=None,
                          outdir="results", window_size=500_000,
                          by_hand=False, n_trials=20, n_iter=100,
                          use_window=True, save=True, prefix="vmr"):
    """
    Orchestrate boosting_elastic_net across genomic windows.

    Parameters
    ----------
    windows : list of dicts
        Each dict should contain at least:
          chrom, start, end, pheno_id
        Optionally:
          geno_arr, bim, fam (for preloaded genotypes)
          geno_path (if loading from file)
          pheno (cupy array) or pheno_path + pheno_id
    """
    cluster, client = None, None
    if len(cuda.gpus) > 1:
        cluster = LocalCUDACluster()
        client = Client(cluster)
    else:
        print("Running on single GPU / CPU without Dask cluster.")

    if not windows:
        raise ValueError("No windows provided to run_windows_with_dask().")

    tasks = [
        delayed(run_single_window)(
            chrom=w["chrom"], start=w["start"], end=w["end"],
            geno_arr=w.get("geno_arr", None), bim=w.get("bim", None),
            fam=w.get("fam", None), geno_path=w.get("geno_path", None),
            pheno=w.get("pheno", None), pheno_path=w.get("pheno_path", None),
            pheno_id=w.get("pheno_id", None), has_header=w.get("has_header", True),
            y_pos=w.get("y_pos", None), error_regions=error_regions, outdir=outdir,
            window_size=window_size, by_hand=by_hand, n_trials=n_trials,
            n_iter=n_iter, use_window=use_window
        )
        for w in windows
    ]

    results = compute(*tasks)

    if client:
        client.close()
        cluster.close()

    df = pd.DataFrame([r for r in results if r is not None])
    if save:
        df.to_parquet(f"{outdir}/{prefix}.summary_windows.parquet", index=False)

    return df
