#!/usr/bin/env Rscript
## Regenerate the R reference outputs used by tests/test_lgv_*.py.
##
##   Rscript tests/fixtures/make_r_fixtures.R <dna-methylation-heritability repo root>
##
## Run in the environment that produced the Module 02 runs (R 4.4.3,
## glmnet 4.1-10; /projects/p32505/opt/envs/calibrated-local-h2). Outputs are
## small TSV/TXT files in tests/fixtures/r/ and are committed, so the Python
## test suite needs neither R nor the analysis repository.
args <- commandArgs(TRUE)
repo <- if (length(args)) args[[1L]] else
    "/gpfs/projects/b1213/users/kynon/projects/dna-methylation-heritability"
script_dir <- dirname(normalizePath(sub("^--file=", "", grep("^--file=",
    commandArgs(FALSE), value = TRUE)[[1L]])))
out <- file.path(script_dir, "r")
dir.create(out, showWarnings = FALSE, recursive = TRUE)
H <- file.path(repo, "02_local_genetic_variance", "_h")
source(file.path(H, "00_functions.R"))
source(file.path(H, "joint_pve_functions.R"))
suppressPackageStartupMessages(library(glmnet))
## Numbers are written with 17 significant digits so they round-trip exactly
## (write.table's default of 15 would perturb inputs at ~1e-15).
w <- function(x, f) {
    x <- as.data.frame(x)
    for (j in seq_along(x)) if (is.double(x[[j]]))
        x[[j]] <- ifelse(is.na(x[[j]]), "NaN", sprintf("%.17g", x[[j]]))
    write.table(x, file.path(out, f), sep = "\t", row.names = FALSE,
                col.names = FALSE, na = "NaN", quote = FALSE)
}
wl <- function(x, f) writeLines(sprintf("%.17g", as.numeric(x)), file.path(out, f))

## ---- R RNG: set.seed + sample ---------------------------------------------
set.seed(20250805); wl(runif(5), "rng_runif_20250805.txt")
set.seed(123); wl(sample(rep(1:5, length.out = 23)), "rng_sample_123.txt")
set.seed(2147483628); wl(sample.int(153), "rng_sampleint_2147483628.txt")
wl(make_balanced_folds(117, 5, 987654321L), "folds_117_5_987654321.txt")

## ---- glmnet / cv.glmnet on SNP-like data ------------------------------------
set.seed(1)
snp <- function(n, p) sapply(runif(p, 0.05, 0.5), function(f) rbinom(n, 2, f))
for (case in list(list(tag = "cov", n = 80L, p = 60L),
                  list(tag = "naive", n = 70L, p = 520L))) {
    x <- snp(case$n, case$p)
    b <- rep(0, case$p); b[sample(case$p, 4)] <- rnorm(4, 0, 0.6)
    y <- drop(x %*% b) + rnorm(case$n); y <- (y - mean(y)) / sd(y)
    foldid <- sample(rep(1:5, length.out = case$n))
    w(x, sprintf("glmnet_%s_x.tsv", case$tag)); wl(y, sprintf("glmnet_%s_y.txt", case$tag))
    wl(foldid, sprintf("glmnet_%s_foldid.txt", case$tag))
    for (a in c(0.1, 1)) {
        f <- glmnet(x, y, alpha = a)
        w(cbind(f$lambda, f$a0, f$dev.ratio, t(as.matrix(f$beta))),
          sprintf("glmnet_%s_a%s_path.tsv", case$tag, a))
        wl(f$npasses, sprintf("glmnet_%s_a%s_npasses.txt", case$tag, a))
        cv <- cv.glmnet(x, y, alpha = a, foldid = foldid)
        w(cbind(cv$lambda, cv$cvm, cv$cvsd), sprintf("glmnet_%s_a%s_cv.tsv", case$tag, a))
        wl(c(cv$lambda.min, cv$lambda.1se), sprintf("glmnet_%s_a%s_cvsel.txt", case$tag, a))
    }
}

## ---- Module 02 locus features -----------------------------------------------
set.seed(42)
n <- 90L; p <- 400L
maf <- runif(p, 0.05, 0.5)
lat <- matrix(rnorm(n * p), n, p)
for (j in 2:p) lat[, j] <- 0.7 * lat[, j - 1] + sqrt(0.51) * lat[, j]
g <- sapply(seq_len(p), function(j) {
    q <- 1 - maf[j]; t0 <- qnorm(q^2); t1 <- qnorm(q^2 + 2 * maf[j] * q)
    ifelse(lat[, j] <= t0, 0, ifelse(lat[, j] <= t1, 1, 2))
})
g[sample(length(g), round(0.01 * length(g)))] <- NA
gi <- g; for (j in 1:p) gi[is.na(gi[, j]), j] <- mean(gi[, j], na.rm = TRUE)
b <- rep(0, p); b[sample(p, 6)] <- rnorm(6, 0, 1)
age <- runif(n, 20, 80); sex <- sample(c("F", "M"), n, TRUE); dx <- sample(c("Control", "Schizo"), n, TRUE)
cov <- model.matrix(~ age + factor(sex) + factor(dx))[, -1, drop = FALSE]
y <- drop(gi %*% b) + 0.02 * age + rnorm(n)
w(g, "locus_g.tsv"); w(cov, "locus_cov.tsv"); wl(y, "locus_y.txt")
en <- crossfit_elastic_net(g, y, cov, outer_folds = 5L, outer_repeats = 2L,
                           inner_folds = 5L, alpha_grid = c(0.1, 0.5, 1),
                           lambda_rule = "lambda.1se", max_features = 150L,
                           seed = 1234567L, keep_predictions = TRUE)
he <- haseman_elston(g, y, cov)
res <- c(en$metrics[c("r2_oof", "rho2_oof", "covariance_ratio_oof",
                      "score_variance_ratio_oof", "calibration_slope_oof",
                      "mean_fold_score_variance_ratio", "mean_nonzero_snps")],
         he[c("he_h2", "he_se", "he_pvalue")],
         p_eff = effective_rank_genotype(g), ld_metric = adjacent_ld_metric(g))
writeLines(paste(names(res), sprintf("%.17g", as.numeric(unlist(res))), sep = "\t"),
           file.path(out, "locus_features.tsv"))
wl(en$predictions$oof_prediction, "locus_oof_prediction.txt")
writeLines(c(R.version.string, paste("glmnet", packageVersion("glmnet"))),
           file.path(out, "VERSIONS.txt"))
cat("Wrote fixtures to", out, "\n")
