#!/usr/bin/env python3
"""
Example: Million-scale CpG heritability estimation with GENBoostGPU

This script demonstrates the full workflow for processing millions of CpGs
with cross-chromosome validated hyperparameter tuning and checkpoint/resume.

Required directory structure:
    data/
    ├── cpg_manifests/              # Per-chromosome (parquet or compressed TSV)
    │   ├── cpg_manifest_chr1.parquet
    │   ├── cpg_manifest_chr2.parquet
    │   └── ...
    ├── phenotypes/                 # Per-chromosome (parquet preferred)
    │   ├── pheno_chr1.parquet
    │   ├── pheno_chr2.parquet
    │   └── ...
    └── genotypes/                  # Single shared PLINK set
        ├── genotypes.bed
        ├── genotypes.bim
        └── genotypes.fam

Usage:
    python cpg_test_million.py

Notes:
    - Genotypes are loaded ONCE and shared across all chromosomes
    - Per-chromosome phenotype data is loaded/unloaded to manage memory
    - Checkpoint/resume support allows interruption and continuation
    - Cross-chromosome validation prevents hyperparameter overfitting
"""
import logging
import argparse
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Million-scale CpG heritability estimation"
    )
    parser.add_argument(
        "--geno-path", type=str, required=True,
        help="PLINK prefix for genotypes (e.g., data/genotypes/genotypes)"
    )
    parser.add_argument(
        "--cpg-manifest-template", type=str,
        default="data/cpg_manifests/cpg_manifest_chr{chrom}.parquet",
        help="Template for CpG manifests (use {chrom} placeholder)"
    )
    parser.add_argument(
        "--pheno-template", type=str,
        default="data/phenotypes/pheno_chr{chrom}.parquet",
        help="Template for phenotype files (use {chrom} placeholder)"
    )
    parser.add_argument(
        "--outdir", type=str, default="results",
        help="Output directory"
    )
    parser.add_argument(
        "--prefix", type=str, default="cpg_full",
        help="Output prefix"
    )
    parser.add_argument(
        "--chromosomes", type=str, default="1-22",
        help="Chromosomes to process (e.g., '1-22' or '1,2,3')"
    )
    parser.add_argument(
        "--skip-tuning", action="store_true",
        help="Skip hyperparameter tuning (use defaults)"
    )
    parser.add_argument(
        "--checkpoint-interval", type=int, default=10000,
        help="Checkpoint every N CpGs"
    )
    parser.add_argument(
        "--no-resume", action="store_true",
        help="Don't resume from checkpoint (start fresh)"
    )
    args = parser.parse_args()

    # Parse chromosomes
    if "-" in args.chromosomes:
        start, end = map(int, args.chromosomes.split("-"))
        chromosomes = list(range(start, end + 1))
    else:
        chromosomes = [int(c) for c in args.chromosomes.split(",")]

    logger.info(f"Processing chromosomes: {chromosomes}")

    # Import here to avoid slow startup if just checking --help
    from genboostgpu import (
        run_cpgs_by_chromosome,
        global_tune_cpg_params,
        load_genotypes,
    )

    # Create output directory
    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # Step 1: Load genotypes ONCE (shared across all chromosomes)
    # -------------------------------------------------------------------------
    logger.info(f"Loading genotypes from {args.geno_path}...")
    geno_arr, bim, fam = load_genotypes(args.geno_path)
    N = len(fam)
    M_total = geno_arr.shape[1]
    logger.info(f"Loaded genotypes: {N} samples, {M_total} SNPs")

    # -------------------------------------------------------------------------
    # Step 2: Cross-chromosome hyperparameter tuning (optional)
    # -------------------------------------------------------------------------
    if not args.skip_tuning:
        logger.info("Starting cross-chromosome hyperparameter tuning...")

        # Split chromosomes: odd for training, even for validation
        train_chroms = [c for c in chromosomes if c % 2 == 1]
        val_chroms = [c for c in chromosomes if c % 2 == 0]

        # If only processing one chromosome or all same parity, adjust
        if not train_chroms:
            train_chroms = chromosomes[:len(chromosomes)//2]
            val_chroms = chromosomes[len(chromosomes)//2:]
        if not val_chroms:
            val_chroms = [chromosomes[-1]]
            train_chroms = chromosomes[:-1]

        logger.info(f"Train chromosomes: {train_chroms}")
        logger.info(f"Validation chromosomes: {val_chroms}")

        best_params = global_tune_cpg_params(
            train_chromosomes=train_chroms,
            val_chromosomes=val_chroms,
            geno_arr=geno_arr,
            bim=bim,
            fam=fam,
            cpg_manifest_template=args.cpg_manifest_template,
            pheno_template=args.pheno_template,
            grid={
                "c_lambda": [0.5, 0.7, 1.0, 1.4, 2.0],
                "c_ridge": [0.5, 1.0, 2.0],
                "subsample_frac": [0.6, 0.7, 0.8],
                "batch_size": [4096, 8192],
            },
            frac=0.05,  # 5% of tuning CpGs
            n_min=50,
            n_max=200,
            early_stop={"patience": 5, "min_delta": 1e-4, "warmup": 5},
            working_set={"K": 1024, "refresh": 10},
        )

        logger.info(f"Best parameters (cross-chromosome validated):")
        logger.info(f"  c_lambda: {best_params['c_lambda']}")
        logger.info(f"  c_ridge: {best_params['c_ridge']}")
        logger.info(f"  subsample_frac: {best_params['subsample_frac']}")
        logger.info(f"  batch_size: {best_params['batch_size']}")
        logger.info(f"  Train score: {best_params['train_score']:.4f}")
        logger.info(f"  Val score: {best_params['val_score']:.4f}")

        # Create fixed_params function from tuned parameters
        def fixed_params(cpg):
            from genboostgpu.hyperparams import enet_from_targets
            from genboostgpu.snp_processing import count_snps_in_window
            M = count_snps_in_window(
                bim, cpg["chrom"], cpg["cpg_pos"], cpg["cpg_pos"],
                window_size=500_000, use_window=True
            )
            if M < 2:
                return {}
            alpha, l1r = enet_from_targets(
                M, N,
                c_lambda=best_params["c_lambda"],
                c_ridge=best_params["c_ridge"]
            )
            return {"fixed_alpha": alpha, "fixed_l1_ratio": l1r}

        fixed_subsample = best_params["subsample_frac"]
        batch_size = best_params["batch_size"]
    else:
        # Use defaults
        fixed_params = None
        fixed_subsample = 0.7
        batch_size = 8192

    # -------------------------------------------------------------------------
    # Step 3: Full processing across all chromosomes
    # -------------------------------------------------------------------------
    logger.info("Starting full CpG processing...")

    df = run_cpgs_by_chromosome(
        chromosomes=chromosomes,
        geno_arr=geno_arr,
        bim=bim,
        fam=fam,
        cpg_manifest_template=args.cpg_manifest_template,
        pheno_template=args.pheno_template,
        outdir=args.outdir,
        prefix=args.prefix,
        batch_size=batch_size,
        window_size=500_000,
        n_trials=20,
        n_iter=100,
        fixed_params=fixed_params,
        fixed_subsample=fixed_subsample,
        early_stop={"patience": 5, "min_delta": 1e-4, "warmup": 5},
        working_set={"K": 1024, "refresh": 10},
        checkpoint_interval=args.checkpoint_interval,
        resume=not args.no_resume,
        progress_interval=1000,
    )

    # -------------------------------------------------------------------------
    # Step 4: Summary
    # -------------------------------------------------------------------------
    logger.info("\n" + "=" * 60)
    logger.info("Processing complete!")
    logger.info("=" * 60)
    logger.info(f"Total CpGs processed: {len(df)}")

    if len(df) > 0:
        logger.info(f"Median h2: {df['h2_unscaled'].median():.4f}")
        logger.info(f"Mean h2: {df['h2_unscaled'].mean():.4f}")
        logger.info(f"Median R2: {df['final_r2'].median():.4f}")
        logger.info(f"Mean SNPs per CpG: {df['num_snps'].mean():.1f}")

        # Save summary statistics
        summary_stats = {
            "total_cpgs": len(df),
            "median_h2": float(df["h2_unscaled"].median()),
            "mean_h2": float(df["h2_unscaled"].mean()),
            "median_r2": float(df["final_r2"].median()),
            "mean_snps": float(df["num_snps"].mean()),
        }

        import json
        summary_path = Path(args.outdir) / f"{args.prefix}_summary_stats.json"
        with open(summary_path, "w") as f:
            json.dump(summary_stats, f, indent=2)
        logger.info(f"Summary statistics saved to: {summary_path}")

    logger.info(f"Results saved to: {args.outdir}/{args.prefix}.summary_all_cpgs.parquet")


if __name__ == "__main__":
    main()
