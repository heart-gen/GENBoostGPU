#!/usr/bin/env Rscript
#' Prepare CpG Input Files for GENBoostGPU
#'
#' This script processes BSseq objects from whole-genome bisulfite sequencing (WGBS)
#' data and exports per-chromosome files for use with GENBoostGPU's CpG processing
#' pipeline.
#'
#' Usage:
#'   Rscript prepare_cpg_inputs.R --bsseq path/to/bsseq.rds --output data/cpg_inputs
#'
#' Required packages:
#'   - bsseq (Bioconductor)
#'   - arrow (for parquet output)
#'   - GenomicRanges (Bioconductor)
#'   - optparse (for command line arguments)

suppressPackageStartupMessages({
  library(bsseq)
  library(arrow)
  library(GenomicRanges)
  library(optparse)
})

#' Prepare CpG data from a BSseq object
#'
#' @param bsseq_rds Path to saved BSseq object (.rds file)
#' @param output_dir Output directory for generated files
#' @param chromosomes Chromosomes to process (default: chr1-chr22)
#' @param smooth Whether to apply BSmooth smoothing
#' @param smooth_ns BSmooth ns parameter (number of CpGs in smoothing window)
#' @param smooth_h BSmooth h parameter (minimum window half-width)
#' @param min_cov Minimum coverage filter (sites with coverage below this are excluded)
#' @param sample_id_col Column name in pData for sample IDs (NULL = use colnames)
#'
#' @return NULL (files are written to disk)
prepare_cpg_data <- function(
    bsseq_rds,
    output_dir,
    chromosomes = paste0("chr", 1:22),
    smooth = TRUE,
    smooth_ns = 70,
    smooth_h = 1000,
    min_cov = 1,
    sample_id_col = NULL
) {
  # Create output directories
  manifest_dir <- file.path(output_dir, "cpg_manifests")
  pheno_dir <- file.path(output_dir, "phenotypes")
  dir.create(manifest_dir, recursive = TRUE, showWarnings = FALSE)
  dir.create(pheno_dir, recursive = TRUE, showWarnings = FALSE)

  # Load BSseq object
  message("Loading BSseq object from: ", bsseq_rds)
  bs <- readRDS(bsseq_rds)
  message("  Samples: ", ncol(bs))
  message("  CpG sites: ", nrow(bs))

  # Get sample IDs
  if (!is.null(sample_id_col) && sample_id_col %in% colnames(pData(bs))) {
    sample_ids <- pData(bs)[[sample_id_col]]
  } else {
    sample_ids <- colnames(bs)
  }
  message("  Sample IDs: ", paste(head(sample_ids, 3), collapse = ", "), ", ...")

  # Smooth if requested and not already smoothed
  if (smooth && !hasBeenSmoothed(bs)) {
    message("Applying BSmooth smoothing (ns=", smooth_ns, ", h=", smooth_h, ")...")
    bs <- BSmooth(
      bs,
      ns = smooth_ns,
      h = smooth_h,
      verbose = TRUE,
      parallelBy = "chromosome"
    )
    message("  Smoothing complete")
  } else if (smooth && hasBeenSmoothed(bs)) {
    message("BSseq object already smoothed, skipping smoothing step")
  }

  # Get methylation values
  message("Extracting methylation values...")
  if (hasBeenSmoothed(bs)) {
    beta <- getMeth(bs, type = "smooth", what = "perBase")
  } else {
    beta <- getMeth(bs, type = "raw", what = "perBase")
  }
  message("  Beta matrix dimensions: ", nrow(beta), " x ", ncol(beta))

  # Get CpG locations
  gr <- granges(bs)

  # Track summary statistics
  total_cpgs <- 0
  total_samples <- ncol(bs)

  # Process each chromosome
  for (chrom in chromosomes) {
    message("\nProcessing ", chrom, "...")

    # Subset to chromosome
    idx <- seqnames(gr) == chrom
    n_cpgs_chrom <- sum(idx)

    if (n_cpgs_chrom == 0) {
      warning("  No CpGs found on ", chrom, " - skipping")
      next
    }

    gr_chrom <- gr[idx]
    beta_chrom <- beta[idx, , drop = FALSE]

    # Apply minimum coverage filter if coverage data available
    if (min_cov > 0) {
      cov_mat <- getCoverage(bs)[idx, , drop = FALSE]
      # Keep CpGs with median coverage >= min_cov
      med_cov <- apply(cov_mat, 1, median, na.rm = TRUE)
      cov_filter <- med_cov >= min_cov
      gr_chrom <- gr_chrom[cov_filter]
      beta_chrom <- beta_chrom[cov_filter, , drop = FALSE]
      n_filtered <- sum(!cov_filter)
      if (n_filtered > 0) {
        message("  Filtered ", n_filtered, " CpGs with median coverage < ", min_cov)
      }
    }

    n_cpgs_final <- nrow(beta_chrom)
    if (n_cpgs_final == 0) {
      warning("  No CpGs remaining after filtering on ", chrom, " - skipping")
      next
    }

    # Create CpG IDs (format: chr_position)
    cpg_ids <- paste0(
      gsub("chr", "", as.character(seqnames(gr_chrom))),
      "_",
      start(gr_chrom)
    )

    # Create CpG manifest (annotation)
    manifest <- data.frame(
      cpg_id = cpg_ids,
      chrom = as.integer(gsub("chr", "", as.character(seqnames(gr_chrom)))),
      cpg_pos = start(gr_chrom),
      stringsAsFactors = FALSE
    )

    # Extract chromosome number for filename
    chrom_num <- gsub("chr", "", chrom)

    # Save manifest as parquet
    manifest_path <- file.path(manifest_dir, sprintf("cpg_manifest_chr%s.parquet", chrom_num))
    write_parquet(manifest, manifest_path)
    message("  Saved manifest: ", manifest_path)

    # Create phenotype matrix (samples as rows, CpGs as columns)
    # This is the format expected by GENBoostGPU
    pheno_mat <- t(beta_chrom)
    colnames(pheno_mat) <- cpg_ids
    rownames(pheno_mat) <- sample_ids

    # Convert to data frame for parquet
    pheno_df <- as.data.frame(pheno_mat)

    # Handle any remaining NAs (impute with column mean)
    na_counts <- colSums(is.na(pheno_df))
    if (any(na_counts > 0)) {
      n_na_cols <- sum(na_counts > 0)
      message("  Imputing NAs in ", n_na_cols, " CpG columns with column means")
      for (col in names(pheno_df)[na_counts > 0]) {
        col_mean <- mean(pheno_df[[col]], na.rm = TRUE)
        pheno_df[[col]][is.na(pheno_df[[col]])] <- col_mean
      }
    }

    # Save phenotypes as parquet
    pheno_path <- file.path(pheno_dir, sprintf("pheno_chr%s.parquet", chrom_num))
    write_parquet(pheno_df, pheno_path)
    message("  Saved phenotypes: ", pheno_path)

    message("  ", chrom, ": ", n_cpgs_final, " CpGs, ", total_samples, " samples")
    total_cpgs <- total_cpgs + n_cpgs_final
  }

  message("\n", strrep("=", 60))
  message("Processing complete!")
  message(strrep("=", 60))
  message("Total CpGs processed: ", total_cpgs)
  message("Total samples: ", total_samples)
  message("Output directory: ", output_dir)
  message("\nDirectory structure:")
  message("  ", output_dir, "/")
  message("  ├── cpg_manifests/")
  message("  │   ├── cpg_manifest_chr1.parquet")
  message("  │   ├── cpg_manifest_chr2.parquet")
  message("  │   └── ...")
  message("  └── phenotypes/")
  message("      ├── pheno_chr1.parquet")
  message("      ├── pheno_chr2.parquet")
  message("      └── ...")

  invisible(NULL)
}


#' Validate sample alignment between BSseq and genotype data
#'
#' @param bsseq_rds Path to BSseq object
#' @param fam_file Path to PLINK .fam file
#' @param sample_id_col Column in BSseq pData containing sample IDs
#'
#' @return Data frame with alignment information
validate_sample_alignment <- function(bsseq_rds, fam_file, sample_id_col = NULL) {
  message("Validating sample alignment...")

  # Load BSseq sample IDs
  bs <- readRDS(bsseq_rds)
  if (!is.null(sample_id_col) && sample_id_col %in% colnames(pData(bs))) {
    bs_samples <- pData(bs)[[sample_id_col]]
  } else {
    bs_samples <- colnames(bs)
  }

  # Load FAM file
  fam <- read.table(fam_file, header = FALSE, stringsAsFactors = FALSE)
  colnames(fam) <- c("FID", "IID", "PID", "MID", "Sex", "Pheno")
  geno_samples <- fam$IID

  # Check alignment
  common <- intersect(bs_samples, geno_samples)
  bs_only <- setdiff(bs_samples, geno_samples)
  geno_only <- setdiff(geno_samples, bs_samples)

  message("  BSseq samples: ", length(bs_samples))
  message("  Genotype samples: ", length(geno_samples))
  message("  Common samples: ", length(common))
  message("  BSseq-only samples: ", length(bs_only))
  message("  Genotype-only samples: ", length(geno_only))

  if (length(common) == 0) {
    stop("No common samples found! Check sample ID formats.")
  }

  if (length(bs_only) > 0) {
    message("  WARNING: Some BSseq samples not in genotypes")
    message("    First few: ", paste(head(bs_only, 5), collapse = ", "))
  }

  if (length(geno_only) > 0) {
    message("  WARNING: Some genotype samples not in BSseq")
    message("    First few: ", paste(head(geno_only, 5), collapse = ", "))
  }

  # Check order
  bs_order <- match(common, bs_samples)
  geno_order <- match(common, geno_samples)
  order_match <- all(diff(bs_order) > 0) && all(diff(geno_order) > 0)

  if (order_match) {
    message("  Sample order: ALIGNED (no reordering needed)")
  } else {
    message("  Sample order: MISALIGNED (will need reordering)")
  }

  invisible(data.frame(
    sample_id = common,
    bs_index = match(common, bs_samples),
    geno_index = match(common, geno_samples),
    stringsAsFactors = FALSE
  ))
}


# -----------------------------------------------------------------------------
# Command Line Interface
# -----------------------------------------------------------------------------
if (sys.nframe() == 0) {
  option_list <- list(
    make_option(c("-b", "--bsseq"), type = "character", default = NULL,
                help = "Path to BSseq RDS file", metavar = "FILE"),
    make_option(c("-o", "--output"), type = "character", default = "data/cpg_inputs",
                help = "Output directory [default: %default]", metavar = "DIR"),
    make_option(c("-c", "--chromosomes"), type = "character", default = "1-22",
                help = "Chromosomes to process (e.g., '1-22' or '1,2,3') [default: %default]"),
    make_option("--no-smooth", action = "store_true", default = FALSE,
                help = "Skip BSmooth smoothing"),
    make_option("--smooth-ns", type = "integer", default = 70,
                help = "BSmooth ns parameter [default: %default]"),
    make_option("--smooth-h", type = "integer", default = 1000,
                help = "BSmooth h parameter [default: %default]"),
    make_option("--min-cov", type = "integer", default = 1,
                help = "Minimum median coverage filter [default: %default]"),
    make_option("--sample-id-col", type = "character", default = NULL,
                help = "Column in pData for sample IDs [default: colnames]"),
    make_option("--validate-fam", type = "character", default = NULL,
                help = "Path to PLINK .fam file for sample validation", metavar = "FILE")
  )

  parser <- OptionParser(
    usage = "%prog [options]",
    description = "Prepare CpG input files for GENBoostGPU from BSseq objects",
    option_list = option_list
  )

  args <- parse_args(parser)

  # Validate required arguments
  if (is.null(args$bsseq)) {
    stop("--bsseq argument is required. Use --help for usage.")
  }

  if (!file.exists(args$bsseq)) {
    stop("BSseq file not found: ", args$bsseq)
  }

  # Parse chromosomes
  if (grepl("-", args$chromosomes)) {
    range_parts <- strsplit(args$chromosomes, "-")[[1]]
    chrom_nums <- seq(as.integer(range_parts[1]), as.integer(range_parts[2]))
  } else {
    chrom_nums <- as.integer(strsplit(args$chromosomes, ",")[[1]])
  }
  chromosomes <- paste0("chr", chrom_nums)

  # Validate sample alignment if requested
  if (!is.null(args$validate_fam)) {
    if (!file.exists(args$validate_fam)) {
      stop("FAM file not found: ", args$validate_fam)
    }
    validate_sample_alignment(args$bsseq, args$validate_fam, args$sample_id_col)
    message("")
  }

  # Run preparation
  prepare_cpg_data(
    bsseq_rds = args$bsseq,
    output_dir = args$output,
    chromosomes = chromosomes,
    smooth = !args$no_smooth,
    smooth_ns = args$smooth_ns,
    smooth_h = args$smooth_h,
    min_cov = args$min_cov,
    sample_id_col = args$sample_id_col
  )
}
