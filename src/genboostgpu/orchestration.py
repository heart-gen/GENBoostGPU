import pandas as pd
from numba import cuda
from dask_cuda import LocalCUDACluster
from .vmr_runner import run_single_window
from dask.distributed import Client, Future, as_completed

__all__ = [
    "run_windows_with_dask",
]

def run_windows_with_dask(
        windows, geno_arr=None, bim=None, fam=None, error_regions=None,
        outdir="results", batch_size=2048, window_size=500_000,
        by_hand=False, n_trials=20, n_iter=100, scatter=True,
        use_window=True, save=True, prefix="vmr", max_in_flight=None
):
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
    n_gpus = len(cuda.gpus)
    if n_gpus > 1:
        cluster = LocalCUDACluster(
            rmm_pool_size="12GB",
            threads_per_worker=1,
            rmm_async=True
        )
        client = Client(cluster)
        if max_in_flight is None:
            max_in_flight = 2 * n_gpus
    else:
        ##print("Running on single GPU / CPU without Dask cluster.")
        return _run_serial(
            windows, geno_arr, bim, fam, error_regions, outdir, batch_size,
            window_size, by_hand, n_trials, n_iter, use_window, save, prefix
        )

    # Optional scatter big, read-only arrays once
    if scatter:
        geno_f = _ensure_future(geno_arr, client)
        bim_f  = _ensure_future(bim,      client)
        fam_f  = _ensure_future(fam,      client)
    else:
        geno_f, bim_f, fam_f = geno_arr, bim, fam

    if not windows:
        raise ValueError("No windows provided to run_windows_with_dask().")

    futures, results = [], []
    for w in windows:
        if max_in_flight and len(futures) >= max_in_flight:
            f, r = next(as_completed(futures, with_results=True))
            futures.remove(f)
            if r is not None:
                results.append(r)

        futures.append(client.submit(
            run_single_window, chrom=w["chrom"], start=w["start"], end=w["end"],
            geno_arr=geno_f, bim=bim_f, fam=fam_f, geno_path=w.get("geno_path", None),
            pheno=w.get("pheno", None), pheno_path=w.get("pheno_path", None),
            pheno_id=w.get("pheno_id", None), has_header=w.get("has_header", True),
            y_pos=w.get("y_pos", None), error_regions=error_regions, outdir=outdir,
            window_size=window_size, by_hand=by_hand, n_trials=n_trials, n_iter=n_iter,
            use_window=use_window, batch_size=batch_size, pure=False
        ))

    for f, r in as_completed(futures, with_results=True):
        if r is not None:
            results.append(r)

    client.close(); cluster.close()

    df = pd.DataFrame([r for r in results if r is not None])
    if save:
        df.to_parquet(f"{outdir}/{prefix}.summary_windows.parquet", index=False)
    return df


def _run_serial(windows, geno_arr, bim, fam, error_regions, outdir, n_batch,
                window_size, by_hand, n_trials, n_iter, use_window, save, prefix):
    out = []
    for w in windows:
        r = run_single_window(
            chrom=w["chrom"], start=w["start"], end=w["end"],
            geno_arr=geno_arr, bim=bim, fam=fam,
            geno_path=w.get("geno_path"), pheno=w.get("pheno"),
            pheno_path=w.get("pheno_path"), pheno_id=w.get("pheno_id"),
            has_header=w.get("has_header", True), y_pos=w.get("y_pos"),
            error_regions=error_regions, outdir=outdir,
            window_size=window_size, by_hand=by_hand, n_trials=n_trials,
            n_iter=n_iter, use_window=use_window, batch_size=n_batch
        )
        if r is not None:
            out.append(r)
    df = pd.DataFrame(out)
    if save:
        df.to_parquet(f"{outdir}/{prefix}.summary_windows.parquet", index=False)
    return df


def _ensure_future(x, client, broadcast=True):
    if x is None or isinstance(x, Future):
        return x
    return client.scatter(x, broadcast=broadcast)
