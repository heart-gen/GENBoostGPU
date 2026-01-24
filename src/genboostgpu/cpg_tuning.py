"""
CpG-specific hyperparameter tuning with cross-chromosome validation.

This module provides:
- select_tuning_cpgs(): Stratified CpG sampling for tuning
- global_tune_cpg_params(): Cross-chromosome validated hyperparameter search
"""
import warnings
import logging
import numpy as np
import pandas as pd
import itertools as it
from typing import List, Dict, Optional, Tuple

from .cpg_runner import run_single_cpg
from .hyperparams import enet_from_targets
from .snp_processing import count_snps_in_window

__all__ = [
    "select_tuning_cpgs",
    "global_tune_cpg_params",
]

logger = logging.getLogger(__name__)


def select_tuning_cpgs(
    cpgs: List[Dict],
    bim,
    frac: float = 0.001,
    n_min: int = 50,
    n_max: int = 500,
    window_size: int = 500_000,
    n_bins: int = 3,
    per_chrom_min: int = 0,
    seed: int = 13,
    exclude_failed: Optional[List[str]] = None,
) -> List[Dict]:
    """
    Select a stratified subset of CpGs for global hyperparameter tuning.

    For million-scale CpG data, we typically want 0.1% (frac=0.001) which
    gives 500-1000 tuning CpGs from millions.

    Parameters
    ----------
    cpgs : List[dict]
        CpG specifications with keys: cpg_id, chrom, cpg_pos
    bim : DataFrame
        BIM file with SNP positions
    frac : float
        Fraction of CpGs to select (default 0.001 = 0.1%)
    n_min : int
        Minimum number of tuning CpGs (default 50)
    n_max : int
        Maximum number of tuning CpGs (default 500)
    window_size : int
        Cis-window size for counting SNPs
    n_bins : int
        Number of SNP count bins for stratification
    per_chrom_min : int
        Minimum CpGs per chromosome (for diversity)
    seed : int
        Random seed
    exclude_failed : List[str], optional
        CpG IDs to exclude (e.g., previously failed)

    Returns
    -------
    List[dict]
        Selected CpG specifications for tuning
    """
    rng = np.random.default_rng(seed)
    excl = set(exclude_failed or [])

    # Compute SNP counts per CpG
    rows = []
    for cpg in cpgs:
        cpg_id = cpg["cpg_id"]
        if cpg_id in excl:
            continue

        # Count SNPs in cis-window around CpG
        M_raw = count_snps_in_window(
            bim, cpg["chrom"], cpg["cpg_pos"], cpg["cpg_pos"],
            window_size=window_size, use_window=True
        )
        if M_raw > 0:
            rows.append({**cpg, "M_raw": int(M_raw)})

    if not rows:
        raise ValueError("No CpGs with SNPs for tuning selection.")

    df = pd.DataFrame(rows)
    logger.info(f"Selecting tuning CpGs from {len(df)} with SNPs in cis-window")

    # Decide target count
    target = int(np.clip(round(frac * len(df)), n_min, n_max))
    logger.info(f"Target tuning set size: {target} CpGs")

    # Bin by M_raw (SNP count quantiles)
    n_unique = df["M_raw"].nunique()
    bins = min(n_bins, n_unique)
    if bins > 1:
        df["bin"] = pd.qcut(df["M_raw"], q=bins, duplicates="drop")
    else:
        df["bin"] = pd.Series(["all"] * len(df), dtype="category")

    # Optional: ensure chromosome diversity
    picked_idx = []
    if per_chrom_min > 0:
        for _, dch in df.groupby("chrom"):
            k = min(per_chrom_min, len(dch))
            if k > 0:
                picked_idx.extend(dch.sample(n=k, random_state=seed).index.tolist())

    remaining = df.drop(index=picked_idx)

    # Allocate evenly across bins
    need = max(0, target - len(picked_idx))
    if need > 0:
        groups = list(remaining.groupby("bin", observed=True))
        B = max(1, len(groups))
        base = need // B
        extra = need % B

        ordered_bins = sorted(groups, key=lambda t: str(t[0]))
        for i, (_, g) in enumerate(ordered_bins):
            take = min(len(g), base + (1 if i < extra else 0))
            if take > 0:
                picked_idx.extend(g.sample(n=take, random_state=seed).index.tolist())

    sel = df.loc[picked_idx].drop_duplicates()

    # Fill if short
    if len(sel) < target:
        short = target - len(sel)
        fill = df.drop(index=sel.index).sort_values("M_raw", ascending=False).head(short)
        sel = pd.concat([sel, fill], axis=0).drop_duplicates()

    # Drop helper columns
    for col in ["bin", "M_raw"]:
        if col in sel.columns:
            sel = sel.drop(columns=[col])

    logger.info(f"Selected {len(sel)} tuning CpGs")
    return sel.to_dict(orient="records")


def global_tune_cpg_params(
    train_chromosomes: List[int],
    val_chromosomes: List[int],
    geno_arr,
    bim,
    fam,
    cpg_manifest_template: str,
    pheno_template: str,
    grid: Optional[Dict] = None,
    frac: float = 0.05,
    n_min: int = 50,
    n_max: int = 200,
    window_size: int = 500_000,
    early_stop: Optional[Dict] = None,
    working_set: Optional[Dict] = None,
    batch_size: int = 4096,
    n_iter: int = 100,
    use_multi_gpu: bool = True,
    rmm_pool_size: str = "12GB",
    max_in_flight: Optional[int] = None,
) -> Dict:
    """
    Cross-chromosome validation for hyperparameter tuning.

    This function:
    1. Selects tuning CpGs from TRAIN chromosomes only
    2. For each parameter combination:
       a. Runs on train CpGs -> computes median R2 (train score)
       b. Runs on hold-out CpGs from VAL chromosomes -> computes median R2 (val score)
    3. Selects parameters with best validation score (not train score)

    This prevents overfitting hyperparameters to specific chromosomes.

    Parameters
    ----------
    train_chromosomes : List[int]
        Chromosomes for training/tuning (e.g., [1, 3, 5, ..., 21])
    val_chromosomes : List[int]
        Chromosomes for validation (e.g., [2, 4, 6, ..., 22])
    geno_arr, bim, fam : pre-loaded genotype data
    cpg_manifest_template : str
        Template path for CpG manifests (use {chrom} placeholder)
    pheno_template : str
        Template path for phenotype files (use {chrom} placeholder)
    grid : dict, optional
        Hyperparameter grid. Default:
        {
            "c_lambda": [0.5, 0.7, 1.0, 1.4, 2.0],
            "c_ridge": [0.5, 1.0, 2.0],
            "subsample_frac": [0.6, 0.7, 0.8],
            "batch_size": [4096, 8192],
        }
    frac : float
        Fraction of CpGs to use for tuning per chromosome set
    n_min, n_max : int
        Min/max tuning CpGs per set
    window_size : int
        Cis-window size
    early_stop : dict, optional
        Early stopping config
    working_set : dict, optional
        Working set config
    batch_size : int
        Default batch size
    n_iter : int
        Boosting iterations
    use_multi_gpu : bool
        Whether to use multi-GPU via Dask
    rmm_pool_size : str
        RMM pool size for Dask workers
    max_in_flight : int, optional
        Max concurrent tasks

    Returns
    -------
    dict
        Best parameters: {c_lambda, c_ridge, subsample_frac, batch_size,
                         train_score, val_score}
    """
    if grid is None:
        grid = {
            "c_lambda": [0.5, 0.7, 1.0, 1.4, 2.0],
            "c_ridge": [0.5, 1.0, 2.0],
            "subsample_frac": [0.6, 0.7, 0.8],
            "batch_size": [4096, 8192],
        }

    if early_stop is None:
        early_stop = {"patience": 5, "min_delta": 1e-4, "warmup": 5}

    # Infer N
    N = _infer_N(fam, geno_arr)
    if N < 2:
        raise ValueError("Could not infer sample size N")

    logger.info(f"Starting cross-chromosome hyperparameter tuning")
    logger.info(f"Train chromosomes: {train_chromosomes}")
    logger.info(f"Validation chromosomes: {val_chromosomes}")

    # Load CpGs from train and val chromosomes
    train_cpgs = _load_cpgs_from_chromosomes(
        train_chromosomes, cpg_manifest_template, pheno_template
    )
    val_cpgs = _load_cpgs_from_chromosomes(
        val_chromosomes, cpg_manifest_template, pheno_template
    )

    logger.info(f"Loaded {len(train_cpgs)} train CpGs, {len(val_cpgs)} val CpGs")

    # Select tuning subsets
    train_tuning = select_tuning_cpgs(
        train_cpgs, bim, frac=frac, n_min=n_min, n_max=n_max,
        window_size=window_size, seed=13
    )
    val_tuning = select_tuning_cpgs(
        val_cpgs, bim, frac=frac, n_min=n_min, n_max=n_max,
        window_size=window_size, seed=42  # Different seed for validation
    )

    logger.info(f"Selected {len(train_tuning)} train tuning CpGs, {len(val_tuning)} val tuning CpGs")

    # Precompute M for each tuning CpG
    train_with_M = _add_snp_counts(train_tuning, bim, window_size)
    val_with_M = _add_snp_counts(val_tuning, bim, window_size)

    # Filter to CpGs with sufficient SNPs
    train_with_M = [(c, M) for c, M in train_with_M if M > 1]
    val_with_M = [(c, M) for c, M in val_with_M if M > 1]

    if not train_with_M:
        raise ValueError("No train tuning CpGs with M > 1")
    if not val_with_M:
        raise ValueError("No validation tuning CpGs with M > 1")

    # Setup Dask if multi-GPU
    client = None
    created_cluster = False
    if use_multi_gpu:
        try:
            from numba import cuda
            n_gpus = len(cuda.gpus)
        except Exception:
            n_gpus = 0
        if n_gpus > 1:
            from dask_cuda import LocalCUDACluster
            from dask.distributed import Client
            cluster = LocalCUDACluster(
                rmm_pool_size=rmm_pool_size,
                threads_per_worker=1,
                rmm_async=True
            )
            client = Client(cluster)
            created_cluster = True
            if max_in_flight is None:
                max_in_flight = 2 * n_gpus

    # Grid search
    combos = list(it.product(
        grid["c_lambda"], grid["c_ridge"],
        grid["subsample_frac"], grid["batch_size"]
    ))

    best = None
    best_val_score = -np.inf

    logger.info(f"Evaluating {len(combos)} parameter combinations")

    for i, (c_lam, c_ridge, sub, bs) in enumerate(combos):
        logger.info(f"Combo {i+1}/{len(combos)}: c_lambda={c_lam}, c_ridge={c_ridge}, "
                   f"subsample={sub}, batch_size={bs}")

        # Evaluate on train CpGs
        train_scores = _evaluate_params_on_cpgs(
            train_with_M, geno_arr, bim, fam, pheno_template,
            c_lam, c_ridge, sub, bs, N, n_iter, early_stop, working_set,
            window_size, client, max_in_flight
        )
        train_score = float(np.nanmedian(train_scores)) if train_scores else -np.inf

        # Evaluate on validation CpGs
        val_scores = _evaluate_params_on_cpgs(
            val_with_M, geno_arr, bim, fam, pheno_template,
            c_lam, c_ridge, sub, bs, N, n_iter, early_stop, working_set,
            window_size, client, max_in_flight
        )
        val_score = float(np.nanmedian(val_scores)) if val_scores else -np.inf

        logger.info(f"  Train score: {train_score:.4f}, Val score: {val_score:.4f}")

        if val_score > best_val_score:
            best_val_score = val_score
            best = {
                "c_lambda": float(c_lam),
                "c_ridge": float(c_ridge),
                "subsample_frac": float(sub),
                "batch_size": int(bs),
                "train_score": float(train_score),
                "val_score": float(val_score),
            }

    if created_cluster and client:
        client.close()

    if best is None:
        raise RuntimeError("Global tuning failed - no valid scores")

    logger.info(f"Best parameters (cross-chromosome validated):")
    logger.info(f"  c_lambda={best['c_lambda']}, c_ridge={best['c_ridge']}")
    logger.info(f"  subsample_frac={best['subsample_frac']}, batch_size={best['batch_size']}")
    logger.info(f"  Train score={best['train_score']:.4f}, Val score={best['val_score']:.4f}")

    return best


def _load_cpgs_from_chromosomes(
    chromosomes: List[int],
    cpg_manifest_template: str,
    pheno_template: str,
) -> List[Dict]:
    """Load CpG specifications from chromosome manifest files."""
    cpgs = []
    for chrom in chromosomes:
        manifest_path = cpg_manifest_template.format(chrom=chrom)
        try:
            if manifest_path.endswith(".parquet"):
                df = pd.read_parquet(manifest_path)
            elif manifest_path.endswith(".gz"):
                df = pd.read_csv(manifest_path, sep="\t", compression="gzip")
            else:
                df = pd.read_csv(manifest_path, sep="\t")

            for _, row in df.iterrows():
                cpgs.append({
                    "cpg_id": row["cpg_id"],
                    "chrom": int(row.get("chrom", chrom)),
                    "cpg_pos": int(row["cpg_pos"]),
                    "pheno_path": pheno_template.format(chrom=chrom),
                })
        except FileNotFoundError:
            logger.warning(f"Manifest not found for chr{chrom}: {manifest_path}")
        except Exception as e:
            logger.warning(f"Error loading manifest for chr{chrom}: {e}")

    return cpgs


def _add_snp_counts(
    cpgs: List[Dict],
    bim,
    window_size: int,
) -> List[Tuple[Dict, int]]:
    """Add SNP counts to CpG list."""
    result = []
    for cpg in cpgs:
        M = count_snps_in_window(
            bim, cpg["chrom"], cpg["cpg_pos"], cpg["cpg_pos"],
            window_size=window_size, use_window=True
        )
        result.append((cpg, int(M)))
    return result


def _evaluate_params_on_cpgs(
    cpgs_with_M: List[Tuple[Dict, int]],
    geno_arr, bim, fam,
    pheno_template: str,
    c_lam: float, c_ridge: float,
    sub: float, bs: int,
    N: int, n_iter: int,
    early_stop: Dict,
    working_set: Optional[Dict],
    window_size: int,
    client,
    max_in_flight: Optional[int],
) -> List[float]:
    """Evaluate parameters on a set of CpGs."""
    scores = []

    # Group CpGs by chromosome for efficient phenotype loading
    by_chrom = {}
    for cpg, M in cpgs_with_M:
        chrom = cpg["chrom"]
        if chrom not in by_chrom:
            by_chrom[chrom] = []
        by_chrom[chrom].append((cpg, M))

    for chrom, chrom_cpgs in by_chrom.items():
        # Load phenotype data for this chromosome
        pheno_path = pheno_template.format(chrom=chrom)
        try:
            import cudf
            if pheno_path.endswith(".parquet"):
                pheno_df = cudf.read_parquet(pheno_path)
            else:
                pheno_df = cudf.read_csv(pheno_path, sep="\t")
        except (ImportError, Exception):
            if pheno_path.endswith(".parquet"):
                pheno_df = pd.read_parquet(pheno_path)
            else:
                pheno_df = pd.read_csv(pheno_path, sep="\t")

        for cpg, M in chrom_cpgs:
            alpha, l1r = enet_from_targets(M, N, c_lambda=c_lam, c_ridge=c_ridge)

            try:
                res = run_single_cpg(
                    cpg_id=cpg["cpg_id"],
                    chrom=cpg["chrom"],
                    cpg_pos=cpg["cpg_pos"],
                    geno_arr=geno_arr,
                    bim=bim,
                    fam=fam,
                    pheno_df=pheno_df,
                    window_size=window_size,
                    n_trials=1,
                    n_iter=n_iter,
                    batch_size=min(bs, M),
                    fixed_alpha=alpha,
                    fixed_l1_ratio=l1r,
                    fixed_subsample=sub,
                    early_stop=early_stop,
                    working_set=working_set,
                    save_full=False,
                )
                if res is not None and np.isfinite(res.get("final_r2", np.nan)):
                    scores.append(float(res["final_r2"]))
            except Exception as e:
                logger.debug(f"Error evaluating CpG {cpg['cpg_id']}: {e}")

        # Free memory
        del pheno_df

    return scores


def _infer_N(fam=None, geno_arr=None) -> int:
    """Infer sample size N."""
    if fam is not None:
        try:
            return int(len(fam))
        except Exception:
            pass
    if geno_arr is not None:
        try:
            return int(getattr(geno_arr, "shape", [0])[0])
        except Exception:
            pass
    warnings.warn("Unable to infer sample size N")
    return 0


def leave_one_chromosome_out_tune(
    chromosomes: List[int],
    val_chromosomes: List[int],
    geno_arr, bim, fam,
    cpg_manifest_template: str,
    pheno_template: str,
    grid: Optional[Dict] = None,
    **kwargs
) -> Dict:
    """
    Leave-one-chromosome-out cross-validation for more robust tuning.

    Uses the specified val_chromosomes as hold-out, rest as train.

    Example:
        # Use chr22, chr20, chr18 as validation (smaller chromosomes)
        best = leave_one_chromosome_out_tune(
            chromosomes=list(range(1, 23)),
            val_chromosomes=[22, 20, 18],
            ...
        )
    """
    train_chromosomes = [c for c in chromosomes if c not in val_chromosomes]

    return global_tune_cpg_params(
        train_chromosomes=train_chromosomes,
        val_chromosomes=val_chromosomes,
        geno_arr=geno_arr,
        bim=bim,
        fam=fam,
        cpg_manifest_template=cpg_manifest_template,
        pheno_template=pheno_template,
        grid=grid,
        **kwargs
    )
