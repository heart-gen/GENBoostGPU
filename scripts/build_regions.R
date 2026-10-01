#!/usr/bin/env Rscript
## Build CpG or CpH region phenotypes for GENBoostGPU from bsseq/HDF5 stores.
##
## Units are methylation measurements per sample: single CpG sites
## (--unit site) or fixed tiles that aggregate many cytosines
## (--unit tile, sum(M) / sum(Cov) per sample; the default for CpH, whose
## per-site methylation is too sparse and too noisy to call regions on).
##
## Steps (one SLURM task per chromosome except `pca`):
##
##   qc     per chromosome: sample selection, context filter (CG, CHG, CHH, CA,
##          CC, CT, CH), --exclude-bed regions, coverage QC, unit methylation
##          matrix -> units/chr{c}.*
##   pca    genome-wide: top-N variable units, optional regression on genotype
##          PCs, prcomp(scale = TRUE) -> pca/meth_pcs.tsv
##   call   per chromosome: residualize units on methylation PCs, residual SD,
##          per-chromosome quantile cutoff, bsseq regionFinder3(maxGap), keep
##          regions with n_units > min_units, region phenotype = mean unit
##          methylation -> regions/chr{c}.*
##   tiles  per chromosome: use QC'd units directly as regions (no calling)
##
## Site-level inputs for `genboostgpu sites` are the `qc` outputs themselves
## (units/chr{c}.beta.h5 + units/chr{c}.units.tsv.gz). For CpH sites use
## --unit site with --min-sd and/or --top-per-chrom to keep a tractable,
## variable subset (scripts/prepare_site_inputs.sh wraps this).
##
## `call` mirrors the Module 01 VMR definition (dna-methylation-heritability
## 01_vmr_catalog: snpPC1-3 removed before PCA, meth PC1-5 removed to form
## residual methylation, 99th percentile residual-SD cutoff per chromosome,
## regionFinder3 maxGap 1000, n > 5) with every constant a parameter.
##
## Then: `genboostgpu regions merge --build-dir <out>` writes regions.tsv and
## phenotypes.parquet for `genboostgpu lgv init --regions ...`.
##
## Inputs: a bsseq/SummarizedExperiment .rds (e.g. nonCpGse.rds), an HDF5 SE
## directory (se.rds + assays.h5), or per-chromosome .rda files (pattern with
## {chrom}, e.g. bs_chr{chrom}_DLPFC_CpH.rda). Stale HDF5 seed paths are
## repointed to --hdf5-dir (same basename), as DNAm-biomarkers-SCZ's
## cph_clocks/multiregion/_h/utils_hdf5.R::repoint_hdf5 does.
##
## Environment: R with bsseq, HDF5Array, DelayedArray, rhdf5, data.table
## (e.g. /projects/p32505/opt/envs/epigenomics). Whole-genome CpH stores need a
## large-memory node for `qc` (the row ranges are held in memory).

suppressPackageStartupMessages({
    library(data.table)
    library(SummarizedExperiment)
    library(DelayedArray)
    library(rhdf5)
})

## ------------------------------------------------------------------ arguments
parse_args <- function(defaults) {
    args <- commandArgs(trailingOnly = TRUE)
    out <- defaults
    i <- 1L
    while (i <= length(args)) {
        a <- args[[i]]
        if (!startsWith(a, "--")) stop("Unexpected argument: ", a)
        key <- sub("^--", "", a)
        if (grepl("=", key, fixed = TRUE)) {
            val <- sub("^[^=]*=", "", key); key <- sub("=.*$", "", key)
        } else {
            val <- if (i < length(args) && !startsWith(args[[i + 1L]], "--")) {
                i <- i + 1L; args[[i]]
            } else "TRUE"
        }
        key <- gsub("-", "_", key)
        if (!key %in% names(defaults)) stop("Unknown option --", gsub("_", "-", key))
        out[[key]] <- val
        i <- i + 1L
    }
    out
}

opt <- parse_args(list(
    step = "", input = "", rda_object = "", hdf5_dir = "", samples = "",
    coldata_id = "", chrom = "", context = "CG", unit = "", tile_width = "10000",
    min_cov = "5", min_covered_frac = "0.8", min_tile_cov = "20",
    out = "", top_n = "1000000", n_meth_pcs = "5", snp_pcs = "", n_snp_pcs = "3",
    sd_quantile = "0.99", max_gap = "", min_units = "5", block_rows = "1000000",
    m_assay = "M", cov_assay = "Cov", min_sd = "0", top_per_chrom = "0",
    exclude_bed = ""
))
num <- function(x) as.numeric(x)
if (!nzchar(opt$out)) stop("--out is required")
if (!opt$step %in% c("qc", "pca", "call", "tiles")) stop("--step must be qc, pca, call or tiles")
opt$context <- toupper(opt$context)
if (!nzchar(opt$unit)) opt$unit <- if (opt$context == "CG") "site" else "tile"
if (!opt$unit %in% c("site", "tile")) stop("--unit must be site or tile")
tile_width <- as.integer(opt$tile_width)
max_gap <- if (nzchar(opt$max_gap)) as.integer(opt$max_gap) else
    if (opt$unit == "tile") tile_width else 1000L
for (d in c("units", "pca", "regions", "manifest")) {
    dir.create(file.path(opt$out, d), recursive = TRUE, showWarnings = FALSE)
}
chrom_label <- sub("^chr", "", opt$chrom, ignore.case = TRUE)

log_step <- function(...) message(format(Sys.time(), "[%H:%M:%S] "), ...)

write_manifest <- function(step, extra = list()) {
    rec <- c(list(step = step, chrom = opt$chrom, time = format(Sys.time(), "%Y-%m-%dT%H:%M:%S%z"),
                  r = R.version.string,
                  bsseq = tryCatch(as.character(packageVersion("bsseq")), error = function(e) NA)),
             opt, extra)
    f <- file.path(opt$out, "manifest",
                   sprintf("%s%s.tsv", step, if (nzchar(opt$chrom)) paste0("-chr", chrom_label) else ""))
    fwrite(data.table(field = names(rec), value = vapply(rec, function(v) paste(v, collapse = ","), "")),
           f, sep = "\t")
}

## ---------------------------------------------------------------------- input
repoint_hdf5 <- function(obj, new_dir) {
    for (an in assayNames(obj)) {
        a <- assay(obj, an, withDimnames = FALSE)
        if (!is(a, "DelayedArray")) next
        s <- seed(a)
        if (!("filepath" %in% slotNames(s))) next
        old <- slot(s, "filepath")
        if (file.exists(old)) next
        cand <- file.path(new_dir, basename(old))
        if (!file.exists(cand)) stop("No local copy of ", basename(old), " in ", new_dir)
        slot(s, "filepath") <- cand
        assay(obj, an, withDimnames = FALSE) <- DelayedArray(s)
        log_step("repointed ", an, " -> ", cand)
    }
    obj
}

load_store <- function() {
    path <- opt$input
    if (grepl("{chrom}", path, fixed = TRUE)) path <- gsub("{chrom}", chrom_label, path, fixed = TRUE)
    obj <- if (dir.exists(path)) {
        HDF5Array::loadHDF5SummarizedExperiment(path)
    } else if (grepl("\\.rds$", path, ignore.case = TRUE)) {
        readRDS(path)
    } else {
        e <- new.env(); nm <- load(path, envir = e)
        get(if (nzchar(opt$rda_object)) opt$rda_object else nm[[1L]], envir = e)
    }
    if (nzchar(opt$hdf5_dir)) obj <- repoint_hdf5(obj, opt$hdf5_dir)
    obj
}

select_samples <- function(obj) {
    cd <- as.data.frame(colData(obj))
    ids <- if (nzchar(opt$coldata_id)) as.character(cd[[opt$coldata_id]]) else colnames(obj)
    if (is.null(ids)) stop("Samples have no IDs; pass --coldata-id")
    if (!nzchar(opt$samples)) return(list(obj = obj, sample_id = ids))
    sheet <- fread(opt$samples, colClasses = "character")
    if (!all(c("sample_id", "store_id") %in% names(sheet))) {
        stop("--samples needs columns sample_id (output ID) and store_id (",
             "matched against ", if (nzchar(opt$coldata_id)) opt$coldata_id else "colnames", ")")
    }
    idx <- match(sheet$store_id, ids)
    if (anyNA(idx)) stop("Samples absent from the store: ", paste(head(sheet$store_id[is.na(idx)], 10), collapse = ", "))
    if (anyDuplicated(sheet$sample_id)) stop("Duplicate sample_id in --samples")
    list(obj = obj[, idx], sample_id = sheet$sample_id)
}

context_filter <- function(obj) {
    mc <- mcols(rowRanges(obj))
    ctx <- opt$context
    if (ctx %in% c("CG", "CHG", "CHH")) {
        if ("c_context" %in% names(mc)) return(as.character(mc$c_context) == ctx)
        if (ctx == "CG") return(rep(TRUE, nrow(obj)))  # CpG-only stores
        stop("Store lacks a c_context column")
    }
    if (!"trinucleotide_context" %in% names(mc)) stop("Store lacks trinucleotide_context")
    tri <- toupper(as.character(mc$trinucleotide_context))
    if (ctx == "CH") return(substr(tri, 2, 2) != "G")
    if (ctx %in% c("CA", "CC", "CT")) return(substr(tri, 1, 2) == ctx)
    stop("Unknown --context ", ctx)
}

## Cytosines inside any interval of the comma-separated BED files are dropped
## before units are formed (e.g. the ENCODE hg38 blacklist, which removes the
## poorly mapped acrocentric short arms where CpH "variability" is mapping
## artefact). BED starts are 0-based; chromosome names match with or without
## a "chr" prefix; gzipped files are read directly.
read_exclude <- function(paths, chr_label) {
    parts <- lapply(strsplit(paths, ",", fixed = TRUE)[[1]], function(f) {
        if (!file.exists(f)) stop("--exclude-bed file not found: ", f)
        con <- gzfile(f); ln <- readLines(con); close(con)   # plain or gzipped
        ln <- ln[nzchar(ln) & !grepl("^(#|track|browser)", ln)]
        f3 <- do.call(rbind, lapply(strsplit(ln, "[ \t]+"), `[`, 1:3))
        if (is.null(f3) || anyNA(f3)) stop("--exclude-bed needs chrom, start, end columns: ", f)
        keep <- sub("^chr", "", f3[, 1]) == chr_label
        data.table(start = as.integer(f3[keep, 2]) + 1L, end = as.integer(f3[keep, 3]))
    })
    ex <- rbindlist(parts)
    IRanges::reduce(IRanges::IRanges(ex$start, ex$end))
}

h5_write_matrix <- function(file, mat, sample_id, unit_id) {
    if (file.exists(file)) file.remove(file)
    h5createFile(file)
    h5createDataset(file, "beta", dims = dim(mat), storage.mode = "double",
                    chunk = c(min(nrow(mat), 4096L), ncol(mat)), level = 4, fillValue = NaN)
    h5write(mat, file, "beta")
    h5write(sample_id, file, "sample_id")
    h5write(unit_id, file, "unit_id")
    h5closeAll()
}

read_units <- function(chrom) {
    f <- file.path(opt$out, "units", sprintf("chr%s.beta.h5", chrom))
    list(beta = h5read(f, "beta"), sample_id = as.character(h5read(f, "sample_id")),
         units = fread(file.path(opt$out, "units", sprintf("chr%s.units.tsv.gz", chrom))))
}

impute_rows <- function(m) {
    mu <- rowMeans(m, na.rm = TRUE)
    w <- which(is.na(m), arr.ind = TRUE)
    if (nrow(w)) m[w] <- mu[w[, 1L]]
    m
}

## ---------------------------------------------------------------------- steps
if (opt$step == "qc") {
    if (!nzchar(opt$chrom)) stop("--chrom is required for qc")
    obj <- load_store()
    sel <- select_samples(obj); obj <- sel$obj
    ## %in% on the seqnames Rle stays run-length encoded (a genome-wide CpH
    ## store has ~10^9 rows; as.character() would expand them all).
    keep_chr <- as.logical(seqnames(obj) %in% c(chrom_label, paste0("chr", chrom_label)))
    obj <- obj[keep_chr, ]
    obj <- obj[context_filter(obj), ]
    n_excluded <- 0L
    if (nzchar(opt$exclude_bed)) {
        ex <- read_exclude(opt$exclude_bed, chrom_label)
        drop <- IRanges::overlapsAny(IRanges::IRanges(start(rowRanges(obj)), width = 1L), ex)
        n_excluded <- sum(drop)
        log_step("chr", chrom_label, ": --exclude-bed drops ", n_excluded, " of ", length(drop),
                 " cytosines (", length(ex), " intervals, ", sum(IRanges::width(ex)), " bp)")
        obj <- obj[!drop, ]
    }
    pos <- start(rowRanges(obj))
    log_step("chr", chrom_label, ": ", nrow(obj), " ", opt$context, " cytosines x ", ncol(obj), " samples")
    M <- assay(obj, opt$m_assay, withDimnames = FALSE)
    C <- assay(obj, opt$cov_assay, withDimnames = FALSE)
    n <- nrow(obj); block <- as.integer(num(opt$block_rows))
    starts <- seq.int(1L, max(1L, n), by = block)
    min_cov <- num(opt$min_cov); frac <- num(opt$min_covered_frac)
    if (opt$unit == "site") {
        parts <- list(); bparts <- list()
        for (s in starts) {
            e <- min(n, s + block - 1L)
            m <- as.matrix(M[s:e, , drop = FALSE]); cv <- as.matrix(C[s:e, , drop = FALSE])
            covered <- cv >= min_cov
            ok <- rowMeans(covered) >= frac
            b <- m[ok, , drop = FALSE] / cv[ok, , drop = FALSE]
            b[!covered[ok, , drop = FALSE]] <- NA
            parts[[length(parts) + 1L]] <- data.table(pos = pos[s:e][ok], n_sites = 1L)
            bparts[[length(bparts) + 1L]] <- b
        }
        units <- rbindlist(parts); beta <- do.call(rbind, bparts)
        units[, `:=`(chrom = paste0("chr", chrom_label), start = pos, end = pos)]
    } else {
        tile <- (pos - 1L) %/% tile_width
        ut <- sort(unique(tile))
        sumM <- matrix(0, length(ut), ncol(obj)); sumC <- sumM; nsite <- integer(length(ut))
        for (s in starts) {
            e <- min(n, s + block - 1L)
            g <- match(tile[s:e], ut)
            sumM_b <- rowsum(as.matrix(M[s:e, , drop = FALSE]), g, reorder = TRUE)
            sumC_b <- rowsum(as.matrix(C[s:e, , drop = FALSE]), g, reorder = TRUE)
            gi <- as.integer(rownames(sumM_b))
            sumM[gi, ] <- sumM[gi, ] + sumM_b; sumC[gi, ] <- sumC[gi, ] + sumC_b
            nsite <- nsite + tabulate(g, nbins = length(ut))
        }
        covered <- sumC >= num(opt$min_tile_cov)
        ok <- rowMeans(covered) >= frac
        beta <- sumM[ok, , drop = FALSE] / sumC[ok, , drop = FALSE]
        beta[!covered[ok, , drop = FALSE]] <- NA
        st <- ut[ok] * tile_width + 1L
        units <- data.table(chrom = paste0("chr", chrom_label), start = st,
                            end = st + tile_width - 1L, n_sites = nsite[ok], pos = st)
    }
    units[, unit_id := sprintf("%s:%d-%d", chrom, start, end)]
    units[, var := apply(beta, 1L, stats::var, na.rm = TRUE)]
    ## Variance prefilters for site-level runs (CpH has ~10^9 cytosines):
    ## --min-sd drops near-invariant units, --top-per-chrom keeps the K most
    ## variable. Both are recorded in the manifest; neither is used by `call`
    ## by default, which needs the full background to set its cutoff.
    keep_u <- rep(TRUE, nrow(units))
    if (num(opt$min_sd) > 0) keep_u <- keep_u & !is.na(units$var) & sqrt(units$var) >= num(opt$min_sd)
    k_top <- as.integer(num(opt$top_per_chrom))
    if (k_top > 0 && sum(keep_u) > k_top) {
        thr <- sort(units$var[keep_u], decreasing = TRUE)[k_top]
        keep_u <- keep_u & units$var >= thr
    }
    if (!all(keep_u)) {
        log_step("variance prefilter keeps ", sum(keep_u), " of ", nrow(units), " units")
        units <- units[keep_u]; beta <- beta[keep_u, , drop = FALSE]
    }
    fwrite(units[, .(unit_id, chrom, start, end, n_sites, var)],
           file.path(opt$out, "units", sprintf("chr%s.units.tsv.gz", chrom_label)), sep = "\t")
    h5_write_matrix(file.path(opt$out, "units", sprintf("chr%s.beta.h5", chrom_label)),
                    beta, sel$sample_id, units$unit_id)
    write_manifest("qc", list(n_units = nrow(units), n_samples = ncol(beta),
                              n_excluded_cytosines = n_excluded,
                              exclude_bed_md5 = if (nzchar(opt$exclude_bed)) paste(tools::md5sum(
                                  strsplit(opt$exclude_bed, ",", fixed = TRUE)[[1]]), collapse = ",") else ""))
    log_step("chr", chrom_label, ": wrote ", nrow(units), " ", opt$unit, " units")
}

if (opt$step == "pca") {
    files <- list.files(file.path(opt$out, "units"), pattern = "\\.units\\.tsv\\.gz$", full.names = TRUE)
    if (!length(files)) stop("No QC'd units under ", file.path(opt$out, "units"))
    allu <- rbindlist(lapply(files, function(f) fread(f)[, file := f]))
    top_n <- min(as.integer(num(opt$top_n)), nrow(allu))
    allu <- allu[order(-var)][seq_len(top_n)]
    mats <- list(); sample_id <- NULL
    for (f in unique(allu$file)) {
        chrom <- sub("^chr", "", sub("\\.units\\.tsv\\.gz$", "", basename(f)))
        u <- read_units(chrom)
        if (is.null(sample_id)) sample_id <- u$sample_id
        if (!identical(sample_id, u$sample_id)) stop("Sample order differs across chromosomes")
        idx <- match(allu[file == f]$unit_id, u$units$unit_id)
        mats[[f]] <- u$beta[idx, , drop = FALSE]
    }
    x <- impute_rows(do.call(rbind, mats))     # units x samples
    if (nzchar(opt$snp_pcs)) {
        pcs <- fread(opt$snp_pcs, colClasses = list(character = "sample_id"))
        k <- as.integer(num(opt$n_snp_pcs))
        cols <- paste0("snpPC", seq_len(k))
        idx <- match(sample_id, pcs$sample_id)
        if (anyNA(idx)) stop("Samples missing from --snp-pcs: ", paste(head(sample_id[is.na(idx)]), collapse = ", "))
        design <- cbind(1, as.matrix(pcs[idx, ..cols]))
        x <- t(qr.resid(qr(design), t(x)))
    }
    pc <- prcomp(t(x), center = TRUE, scale. = TRUE)
    k <- min(as.integer(num(opt$n_meth_pcs)), ncol(pc$x))
    out <- data.table(sample_id = sample_id, pc$x[, seq_len(k), drop = FALSE])
    fwrite(out, file.path(opt$out, "pca", "meth_pcs.tsv"), sep = "\t")
    fwrite(data.table(pc = seq_along(pc$sdev), var_explained = pc$sdev^2 / sum(pc$sdev^2)),
           file.path(opt$out, "pca", "variance_explained.tsv"), sep = "\t")
    write_manifest("pca", list(n_units_used = nrow(x), n_samples = ncol(x)))
    log_step("PCA on ", nrow(x), " units x ", ncol(x), " samples")
}

region_pheno <- function(beta, idx_list) {
    t(vapply(idx_list, function(ix) colMeans(beta[ix, , drop = FALSE], na.rm = TRUE),
             numeric(ncol(beta))))
}

write_gz <- function(dt, path) {
    ## fwrite(compress = "gzip") of a zero-row table can leave an empty file;
    ## write through a gzfile connection so the header always survives.
    con <- gzfile(path, "w")
    on.exit(close(con))
    utils::write.table(dt, con, sep = "\t", quote = FALSE, row.names = FALSE, na = "NA")
}

write_regions <- function(regions, pheno, sample_id) {
    write_gz(regions, file.path(opt$out, "regions", sprintf("chr%s.regions.tsv.gz", chrom_label)))
    ph <- data.table(sample_id = sample_id)
    if (nrow(regions)) {
        ph <- cbind(ph, as.data.table(t(pheno)))
        setnames(ph, c("sample_id", regions$region_id))
    }
    write_gz(ph, file.path(opt$out, "regions", sprintf("chr%s.pheno.tsv.gz", chrom_label)))
}

if (opt$step == "call") {
    if (!nzchar(opt$chrom)) stop("--chrom is required for call")
    u <- read_units(chrom_label)
    pcs <- fread(file.path(opt$out, "pca", "meth_pcs.tsv"), colClasses = list(character = "sample_id"))
    idx <- match(u$sample_id, pcs$sample_id)
    if (anyNA(idx)) stop("Samples missing from meth_pcs.tsv")
    k <- as.integer(num(opt$n_meth_pcs))
    design <- cbind(1, as.matrix(pcs[idx, paste0("PC", seq_len(k)), with = FALSE]))
    resid <- t(qr.resid(qr(design), t(impute_rows(u$beta))))
    sd_res <- apply(resid, 1L, stats::sd)
    units <- u$units[, sd := sd_res]
    ord <- order(units$start)
    units <- units[ord]; beta <- u$beta[ord, , drop = FALSE]
    sd_cut <- stats::quantile(units$sd, probs = num(opt$sd_quantile), na.rm = TRUE)
    is_high <- as.integer(!is.na(units$sd) & units$sd > sd_cut)
    empty <- data.table(region_id = character(), chrom = character(), start = integer(),
                        end = integer(), n_units = integer(), n_sites = integer())
    if (sum(is_high) == 0) {
        write_regions(empty, matrix(numeric(), 0, ncol(beta)), u$sample_id)
    } else {
        found <- bsseq:::regionFinder3(is_high, rep(paste0("chr", chrom_label), nrow(units)),
                                       units$start, maxGap = max_gap, verbose = FALSE)$up
        found <- as.data.table(found)[n > as.integer(num(opt$min_units))]
        idx_list <- lapply(seq_len(nrow(found)), function(i) found$idxStart[i]:found$idxEnd[i])
        regions <- data.table(
            chrom = paste0("chr", chrom_label), start = as.integer(found$start),
            end = as.integer(units$end[found$idxEnd]), n_units = as.integer(found$n),
            n_sites = vapply(idx_list, function(ix) sum(units$n_sites[ix]), 0))
        regions[, region_id := sprintf("%s:%d-%d", chrom, start, end)]
        setcolorder(regions, c("region_id", "chrom", "start", "end", "n_units", "n_sites"))
        write_regions(regions, region_pheno(beta, idx_list), u$sample_id)
    }
    fwrite(data.table(chrom = paste0("chr", chrom_label), sd_cut = sd_cut,
                      n_units = nrow(units), n_high = sum(is_high)),
           file.path(opt$out, "regions", sprintf("chr%s.cutoff.tsv", chrom_label)), sep = "\t")
    write_manifest("call", list(sd_cut = sd_cut, max_gap = max_gap))
    log_step("chr", chrom_label, ": cutoff ", signif(sd_cut, 4), "; ", sum(is_high), " high units")
}

if (opt$step == "tiles") {
    if (!nzchar(opt$chrom)) stop("--chrom is required for tiles")
    u <- read_units(chrom_label)
    regions <- u$units[, .(region_id = unit_id, chrom, start, end, n_units = 1L, n_sites)]
    write_regions(regions, u$beta, u$sample_id)
    write_manifest("tiles", list(n_regions = nrow(regions)))
}
