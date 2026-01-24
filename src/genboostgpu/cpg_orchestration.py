"""
CpG-level orchestration for multi-GPU processing at million-scale.

This module provides:
- run_cpgs_with_dask(): Single-chromosome CpG processing (mirrors orchestration.py)
- run_cpgs_by_chromosome(): Per-chromosome iterator for million-scale data

Features for million-scale processing:
- Checkpoint/resume support
- Batched submission to avoid scheduler overwhelm
- Progress logging
- Error isolation
"""
import os
import time
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Union, Callable
from numba import cuda
from contextlib import ExitStack
from dask_cuda import LocalCUDACluster
from dask.distributed import Client, Future, as_completed

from .cpg_runner import run_single_cpg
from .hyperparams import enet_from_targets
from .snp_processing import count_snps_in_window

__all__ = [
    "run_cpgs_with_dask",
    "run_cpgs_by_chromosome",
]

# Configure logging
logger = logging.getLogger(__name__)


def run_cpgs_with_dask(
    cpgs: List[Dict],
    # Data sources
    geno_arr=None, bim=None, fam=None,
    pheno_df=None,
    # Settings
    error_regions=None,
    outdir: str = "results",
    batch_size: int = 8192,
    window_size: int = 500_000,
    n_trials: int = 20,
    n_iter: int = 100,
    scatter: bool = True,
    save: bool = True,
    prefix: str = "cpg",
    max_in_flight: Optional[int] = None,
    fixed_params: Optional[Union[Dict, Callable]] = None,
    fixed_subsample: Optional[float] = None,
    early_stop: Optional[Dict] = None,
    working_set: Optional[Dict] = None,
    # Million-scale features
    checkpoint_interval: int = 10000,
    resume: bool = True,
    progress_interval: int = 1000,
) -> pd.DataFrame:
    """
    Orchestrate CpG processing across multiple GPUs using Dask.

    Parameters
    ----------
    cpgs : List[dict]
        List of CpG specifications, each with keys:
        - cpg_id: str (required)
        - chrom: int (required)
        - cpg_pos: int (required)
        - pheno: cupy array (optional, pre-extracted phenotype)
        - pheno_path: str (optional, path to phenotype file)

    geno_arr, bim, fam : pre-loaded genotype data (scattered to workers)
    pheno_df : cudf/pandas DataFrame with CpG columns (scattered to workers)

    error_regions : pd.DataFrame, optional
        Blacklist CpGs (cpg_id column or Chrom/Pos columns)
    outdir : str
        Output directory for results
    batch_size : int
        Batch size for correlation computation
    window_size : int
        Cis-window size around CpG position
    n_trials : int
        Optuna trials for hyperparameter tuning
    n_iter : int
        Boosting iterations
    scatter : bool
        Whether to scatter genotypes to workers (recommended)
    save : bool
        Whether to save summary parquet
    prefix : str
        Prefix for output files
    max_in_flight : int, optional
        Max concurrent tasks (default: 2 * n_gpus)
    fixed_params : dict or callable, optional
        Fixed hyperparameters or function(cpg_dict) -> params
    fixed_subsample : float, optional
        Fixed subsample fraction
    early_stop : dict, optional
        Early stopping config: {patience, min_delta, warmup, metric}
    working_set : dict, optional
        Working set config: {K, refresh}
    checkpoint_interval : int
        Save checkpoint every N CpGs (default: 10000)
    resume : bool
        Resume from checkpoint if available (default: True)
    progress_interval : int
        Log progress every N CpGs (default: 1000)

    Returns
    -------
    pd.DataFrame
        Summary DataFrame with columns: cpg_id, chrom, cpg_pos, num_snps,
        N, final_r2, h2_unscaled, n_iter
    """
    n_gpus = len(cuda.gpus)
    Path(outdir).mkdir(parents=True, exist_ok=True)

    # Setup checkpoint/resume
    completed_file = Path(outdir) / f"{prefix}_completed.txt"
    checkpoint_file = Path(outdir) / f"{prefix}_checkpoint.parquet"
    error_log = Path(outdir) / f"{prefix}_errors.log"

    completed_cpgs = set()
    results = []

    if resume and checkpoint_file.exists():
        try:
            checkpoint_df = pd.read_parquet(checkpoint_file)
            results = checkpoint_df.to_dict(orient="records")
            completed_cpgs = set(checkpoint_df["cpg_id"].tolist())
            logger.info(f"Resuming from checkpoint: {len(completed_cpgs)} CpGs completed")
        except Exception as e:
            logger.warning(f"Could not load checkpoint: {e}")

    if resume and completed_file.exists():
        try:
            with open(completed_file, "r") as f:
                completed_cpgs.update(line.strip() for line in f if line.strip())
        except Exception as e:
            logger.warning(f"Could not load completed list: {e}")

    # Filter out already-completed CpGs
    pending_cpgs = [c for c in cpgs if c["cpg_id"] not in completed_cpgs]
    if not pending_cpgs:
        logger.info("All CpGs already completed")
        if results:
            return pd.DataFrame(results)
        return pd.DataFrame()

    logger.info(f"Processing {len(pending_cpgs)} CpGs ({len(completed_cpgs)} already done)")

    if n_gpus > 1:
        logging.getLogger("distributed.worker").setLevel(logging.WARNING)

        with ExitStack() as stack:
            cluster = stack.enter_context(
                LocalCUDACluster(
                    rmm_pool_size="12GB",
                    threads_per_worker=1,
                    rmm_async=True,
                    dashboard_address=None,
                    local_directory=os.environ.get("TMPDIR", "/tmp")
                )
            )
            client = stack.enter_context(Client(cluster))

            if max_in_flight is None:
                max_in_flight = 2 * n_gpus

            # Scatter data to workers if requested
            if scatter:
                geno_f = _ensure_future(geno_arr, client)
                bim_f = _ensure_future(bim, client)
                fam_f = _ensure_future(fam, client)
                pheno_f = _ensure_future(pheno_df, client)
            else:
                geno_f, bim_f, fam_f, pheno_f = geno_arr, bim, fam, pheno_df

            def _fixed_dict_for(cpg):
                if callable(fixed_params):
                    return fixed_params(cpg) or {}
                return fixed_params or {}

            futures = []
            start_time = time.time()
            processed = 0

            for i, cpg in enumerate(pending_cpgs):
                # Backpressure: wait for a slot
                if max_in_flight and len(futures) >= max_in_flight:
                    f, r = next(as_completed(futures, with_results=True))
                    futures.remove(f)
                    if r is not None:
                        results.append(r)
                        completed_cpgs.add(r["cpg_id"])
                    processed += 1

                    # Progress logging
                    if processed % progress_interval == 0:
                        elapsed = time.time() - start_time
                        rate = processed / elapsed if elapsed > 0 else 0
                        remaining = len(pending_cpgs) - processed
                        logger.info(
                            f"Progress: {processed}/{len(pending_cpgs)} CpGs "
                            f"({rate:.1f}/s, ~{remaining/rate/60:.1f}m remaining)"
                        )

                    # Checkpoint
                    if processed % checkpoint_interval == 0:
                        _save_checkpoint(results, checkpoint_file, completed_cpgs, completed_file)

                fp = _fixed_dict_for(cpg)
                submit_kwargs = dict(
                    cpg_id=cpg["cpg_id"],
                    chrom=cpg["chrom"],
                    cpg_pos=cpg["cpg_pos"],
                    geno_arr=geno_f,
                    bim=bim_f,
                    fam=fam_f,
                    pheno=cpg.get("pheno"),
                    pheno_df=pheno_f,
                    pheno_path=cpg.get("pheno_path"),
                    error_regions=error_regions,
                    outdir=outdir,
                    window_size=window_size,
                    batch_size=batch_size,
                    n_trials=n_trials,
                    n_iter=n_iter,
                    save_full=False,
                    pure=False,
                )

                if "fixed_alpha" in fp:
                    submit_kwargs["fixed_alpha"] = fp["fixed_alpha"]
                if "fixed_l1_ratio" in fp:
                    submit_kwargs["fixed_l1_ratio"] = fp["fixed_l1_ratio"]
                if fixed_subsample is not None:
                    submit_kwargs["fixed_subsample"] = fixed_subsample
                if early_stop is not None:
                    submit_kwargs["early_stop"] = early_stop
                if working_set is not None:
                    submit_kwargs["working_set"] = working_set

                futures.append(client.submit(run_single_cpg, **submit_kwargs))

            # Drain remaining futures
            for f, r in as_completed(futures, with_results=True):
                if r is not None:
                    results.append(r)
                    completed_cpgs.add(r["cpg_id"])
                processed += 1

                if processed % progress_interval == 0:
                    elapsed = time.time() - start_time
                    rate = processed / elapsed if elapsed > 0 else 0
                    logger.info(f"Progress: {processed}/{len(pending_cpgs)} CpGs ({rate:.1f}/s)")

                if processed % checkpoint_interval == 0:
                    _save_checkpoint(results, checkpoint_file, completed_cpgs, completed_file)

            # Graceful shutdown
            client.cancel([])
            client.shutdown()

        # Final save
        df = pd.DataFrame([r for r in results if r is not None])
        if save:
            df.to_parquet(f"{outdir}/{prefix}.summary_cpgs.parquet", index=False)
            _save_checkpoint(results, checkpoint_file, completed_cpgs, completed_file)
        return df
    else:
        return _run_serial_cpgs(
            pending_cpgs, geno_arr, bim, fam, pheno_df, error_regions, outdir,
            batch_size, window_size, n_trials, n_iter, save, prefix,
            fixed_params, fixed_subsample, early_stop, working_set,
            checkpoint_interval, checkpoint_file, completed_cpgs, completed_file,
            progress_interval, results
        )


def _run_serial_cpgs(
    cpgs, geno_arr, bim, fam, pheno_df, error_regions, outdir,
    batch_size, window_size, n_trials, n_iter, save, prefix,
    fixed_params, fixed_subsample, early_stop, working_set,
    checkpoint_interval, checkpoint_file, completed_cpgs, completed_file,
    progress_interval, results
):
    """Serial fallback for single-GPU execution."""

    def _fixed_dict_for(cpg):
        if callable(fixed_params):
            return fixed_params(cpg) or {}
        return fixed_params or {}

    start_time = time.time()

    for i, cpg in enumerate(cpgs):
        try:
            fp = _fixed_dict_for(cpg)
            r = run_single_cpg(
                cpg_id=cpg["cpg_id"],
                chrom=cpg["chrom"],
                cpg_pos=cpg["cpg_pos"],
                geno_arr=geno_arr,
                bim=bim,
                fam=fam,
                pheno=cpg.get("pheno"),
                pheno_df=pheno_df,
                pheno_path=cpg.get("pheno_path"),
                error_regions=error_regions,
                outdir=outdir,
                window_size=window_size,
                batch_size=batch_size,
                n_trials=n_trials,
                n_iter=n_iter,
                fixed_alpha=fp.get("fixed_alpha"),
                fixed_l1_ratio=fp.get("fixed_l1_ratio"),
                fixed_subsample=fixed_subsample,
                early_stop=early_stop,
                working_set=working_set,
                save_full=False,
            )
            if r is not None:
                results.append(r)
                completed_cpgs.add(r["cpg_id"])
        except Exception as e:
            logger.error(f"Error processing CpG {cpg['cpg_id']}: {e}")

        # Progress logging
        if (i + 1) % progress_interval == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            remaining = len(cpgs) - (i + 1)
            logger.info(
                f"Progress: {i + 1}/{len(cpgs)} CpGs "
                f"({rate:.1f}/s, ~{remaining/rate/60:.1f}m remaining)"
            )

        # Checkpoint
        if (i + 1) % checkpoint_interval == 0:
            _save_checkpoint(results, checkpoint_file, completed_cpgs, completed_file)

    df = pd.DataFrame([r for r in results if r is not None])
    if save:
        df.to_parquet(f"{outdir}/{prefix}.summary_cpgs.parquet", index=False)
        _save_checkpoint(results, checkpoint_file, completed_cpgs, completed_file)
    return df


def run_cpgs_by_chromosome(
    chromosomes: List[int],
    # Shared genotype data (loaded once)
    geno_path: Optional[str] = None,
    geno_arr=None, bim=None, fam=None,
    # Per-chromosome CpG data (use {chrom} placeholder)
    cpg_manifest_template: str = "data/cpg_manifests/cpg_manifest_chr{chrom}.parquet",
    pheno_template: str = "data/phenotypes/pheno_chr{chrom}.parquet",
    # Settings
    outdir: str = "results",
    prefix: str = "cpg",
    batch_size: int = 8192,
    window_size: int = 500_000,
    n_trials: int = 20,
    n_iter: int = 100,
    fixed_params: Optional[Union[Dict, Callable]] = None,
    fixed_subsample: Optional[float] = None,
    early_stop: Optional[Dict] = None,
    working_set: Optional[Dict] = None,
    checkpoint_interval: int = 10000,
    resume: bool = True,
    progress_interval: int = 1000,
    error_regions: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Process CpGs chromosome-by-chromosome to manage memory at million-scale.

    This function:
    1. Loads genotypes ONCE (shared across all chromosomes)
    2. Iterates through chromosomes:
       a. Loads CpG manifest for chromosome
       b. Loads phenotype matrix for chromosome
       c. Runs run_cpgs_with_dask for that chromosome
       d. Frees phenotype memory before next chromosome
    3. Aggregates results across chromosomes

    Parameters
    ----------
    chromosomes : List[int]
        Chromosome numbers to process (e.g., [1, 2, ..., 22])
    geno_path : str, optional
        PLINK prefix for genotypes (loaded once if geno_arr not provided)
    geno_arr, bim, fam : optional
        Pre-loaded genotype data
    cpg_manifest_template : str
        Template path for CpG manifests. Use {chrom} as placeholder.
        Expected columns: cpg_id, chrom, cpg_pos
    pheno_template : str
        Template path for phenotype files. Use {chrom} as placeholder.
        Supports .parquet (preferred) or .tsv.gz
    outdir : str
        Output directory
    prefix : str
        Prefix for output files
    [other params same as run_cpgs_with_dask]

    Returns
    -------
    pd.DataFrame
        Combined results from all chromosomes
    """
    from .data_io import load_genotypes

    Path(outdir).mkdir(parents=True, exist_ok=True)

    # Load genotypes ONCE (shared across all chromosomes)
    if geno_arr is None or bim is None or fam is None:
        if geno_path is None:
            raise ValueError("Either geno_arr+bim+fam or geno_path must be provided")
        logger.info(f"Loading genotypes from {geno_path}...")
        geno_arr, bim, fam = load_genotypes(geno_path)
        logger.info(f"Loaded genotypes: {geno_arr.shape[0]} samples, {geno_arr.shape[1]} SNPs")

    N = geno_arr.shape[0]
    all_results = []

    # Track global checkpoint for cross-chromosome resume
    global_completed_file = Path(outdir) / f"{prefix}_global_completed.txt"
    completed_cpgs = set()

    if resume and global_completed_file.exists():
        try:
            with open(global_completed_file, "r") as f:
                completed_cpgs = set(line.strip() for line in f if line.strip())
            logger.info(f"Resuming: {len(completed_cpgs)} CpGs already completed globally")
        except Exception as e:
            logger.warning(f"Could not load global completed list: {e}")

    # Also load any existing chromosome results
    for chrom in chromosomes:
        chrom_parquet = Path(outdir) / f"{prefix}_chr{chrom}.summary_cpgs.parquet"
        if chrom_parquet.exists():
            try:
                chrom_df = pd.read_parquet(chrom_parquet)
                completed_cpgs.update(chrom_df["cpg_id"].tolist())
                all_results.append(chrom_df)
            except Exception:
                pass

    # Process each chromosome
    for chrom in chromosomes:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing chromosome {chrom}")
        logger.info(f"{'='*60}")

        # Load CpG manifest for this chromosome
        manifest_path = cpg_manifest_template.format(chrom=chrom)
        try:
            if manifest_path.endswith(".parquet"):
                manifest_df = pd.read_parquet(manifest_path)
            elif manifest_path.endswith(".gz"):
                manifest_df = pd.read_csv(manifest_path, sep="\t", compression="gzip")
            else:
                manifest_df = pd.read_csv(manifest_path, sep="\t")
        except FileNotFoundError:
            logger.warning(f"Manifest not found for chr{chrom}: {manifest_path}")
            continue
        except Exception as e:
            logger.error(f"Error loading manifest for chr{chrom}: {e}")
            continue

        # Build CpG list from manifest
        cpgs = []
        for _, row in manifest_df.iterrows():
            cpg_id = row["cpg_id"]
            if cpg_id in completed_cpgs:
                continue
            cpgs.append({
                "cpg_id": cpg_id,
                "chrom": int(row.get("chrom", chrom)),
                "cpg_pos": int(row["cpg_pos"]),
            })

        if not cpgs:
            logger.info(f"All CpGs on chr{chrom} already completed, skipping")
            continue

        logger.info(f"Chromosome {chrom}: {len(cpgs)} CpGs to process")

        # Load phenotype data for this chromosome
        pheno_path = pheno_template.format(chrom=chrom)
        try:
            import cudf
            if pheno_path.endswith(".parquet"):
                pheno_df = cudf.read_parquet(pheno_path)
            elif pheno_path.endswith(".gz"):
                pheno_df = cudf.read_csv(pheno_path, sep="\t", compression="gzip")
            else:
                pheno_df = cudf.read_csv(pheno_path, sep="\t")
        except ImportError:
            # Fallback to pandas if cudf not available
            if pheno_path.endswith(".parquet"):
                pheno_df = pd.read_parquet(pheno_path)
            elif pheno_path.endswith(".gz"):
                pheno_df = pd.read_csv(pheno_path, sep="\t", compression="gzip")
            else:
                pheno_df = pd.read_csv(pheno_path, sep="\t")
        except FileNotFoundError:
            logger.error(f"Phenotype file not found for chr{chrom}: {pheno_path}")
            continue
        except Exception as e:
            logger.error(f"Error loading phenotypes for chr{chrom}: {e}")
            continue

        logger.info(f"Loaded phenotypes: {pheno_df.shape[0]} samples, {pheno_df.shape[1]} CpGs")

        # Run CpGs for this chromosome
        chrom_prefix = f"{prefix}_chr{chrom}"
        chrom_df = run_cpgs_with_dask(
            cpgs=cpgs,
            geno_arr=geno_arr,
            bim=bim,
            fam=fam,
            pheno_df=pheno_df,
            error_regions=error_regions,
            outdir=outdir,
            batch_size=batch_size,
            window_size=window_size,
            n_trials=n_trials,
            n_iter=n_iter,
            scatter=True,
            save=True,
            prefix=chrom_prefix,
            fixed_params=fixed_params,
            fixed_subsample=fixed_subsample,
            early_stop=early_stop,
            working_set=working_set,
            checkpoint_interval=checkpoint_interval,
            resume=resume,
            progress_interval=progress_interval,
        )

        if len(chrom_df) > 0:
            all_results.append(chrom_df)
            completed_cpgs.update(chrom_df["cpg_id"].tolist())

            # Update global completed list
            with open(global_completed_file, "w") as f:
                for cpg_id in sorted(completed_cpgs):
                    f.write(f"{cpg_id}\n")

        # Free phenotype memory before next chromosome
        del pheno_df
        try:
            import gc
            gc.collect()
        except Exception:
            pass

        logger.info(f"Completed chr{chrom}: {len(chrom_df)} CpGs processed")

    # Combine all results
    if all_results:
        combined_df = pd.concat(all_results, ignore_index=True)
        combined_df.to_parquet(f"{outdir}/{prefix}.summary_all_cpgs.parquet", index=False)
        logger.info(f"\nTotal: {len(combined_df)} CpGs processed across {len(chromosomes)} chromosomes")
        return combined_df

    return pd.DataFrame()


def _ensure_future(x, client, broadcast=True):
    """Scatter data to workers if not already a Future."""
    if x is None or isinstance(x, Future):
        return x
    return client.scatter(x, broadcast=broadcast)


def _save_checkpoint(results, checkpoint_file, completed_cpgs, completed_file):
    """Save checkpoint to disk."""
    try:
        if results:
            df = pd.DataFrame([r for r in results if r is not None])
            df.to_parquet(checkpoint_file, index=False)

        with open(completed_file, "w") as f:
            for cpg_id in sorted(completed_cpgs):
                f.write(f"{cpg_id}\n")

        logger.debug(f"Checkpoint saved: {len(results)} results, {len(completed_cpgs)} completed")
    except Exception as e:
        logger.warning(f"Failed to save checkpoint: {e}")
