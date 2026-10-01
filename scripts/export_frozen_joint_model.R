#!/usr/bin/env Rscript
## Export the frozen Module 02 joint PVE model (glmnet RDS) to portable JSON.
##
## Usage:
##   Rscript scripts/export_frozen_joint_model.R \
##       --model <joint-pve-calibrator.rds> --sha256 <expected sha> --out <model.json>
##
## The model is a single-lambda glmnet ridge on hinge-expanded signal features
## plus centered/scaled design terms (FINAL_JOINT_PVE_STRATEGY.md section 7).
## Its prediction is the linear predictor a0 + x'beta at that lambda, so the
## coefficients, the hinge specification and the design scaler are all that
## genboostgpu.lgv.joint_model needs. Numbers are written with 17 significant
## digits, which round-trips IEEE doubles exactly.
##
## Requires only glmnet (to extract coef()); no JSON package is used, so it runs
## in the calibrated-local-h2 environment that produced the model.

args <- commandArgs(trailingOnly = TRUE)
get_arg <- function(name) {
    hit <- grep(paste0("^--", name, "="), args, value = TRUE)
    if (length(hit)) return(sub(paste0("^--", name, "="), "", hit[[1L]]))
    i <- match(paste0("--", name), args)
    if (!is.na(i) && i < length(args)) return(args[[i + 1L]])
    ""
}
model_path <- get_arg("model")
expected <- tolower(get_arg("sha256"))
out_path <- get_arg("out")
if (!nzchar(model_path) || !nzchar(out_path)) stop("--model and --out are required")

observed <- tolower(sub(" .*$", "", system2("sha256sum", normalizePath(model_path),
                                            stdout = TRUE)[[1L]]))
if (nzchar(expected) && !identical(observed, expected)) {
    stop("Model checksum mismatch: expected ", expected, " observed ", observed)
}
suppressPackageStartupMessages(library(glmnet))
model <- readRDS(model_path)
if (!identical(model$gate_version, "final_joint_pve_v1")) {
    stop("Unexpected joint-model gate version: ", model$gate_version)
}
cf <- coef(model$glmnet_fit, s = model$lambda)
coef_names <- rownames(cf)
coef_values <- as.numeric(cf[, 1L])

num <- function(x) {
    x <- as.numeric(x)
    vapply(x, function(v) {
        if (is.na(v)) "null" else if (is.infinite(v)) {
            if (v > 0) "\"Inf\"" else "\"-Inf\""
        } else sprintf("%.17g", v)
    }, character(1L))
}
str_q <- function(x) paste0("\"", gsub("\"", "\\\\\"", x), "\"")
arr <- function(x) paste0("[", paste(x, collapse = ", "), "]")

## Signal terms in model-matrix order, with their (feature, knot) pairs.
signal_terms <- character()
for (feature in names(model$signal_spec)) {
    spec <- model$signal_spec[[feature]]
    for (knot in spec$knots) {
        label <- gsub("-", "m", format(knot, scientific = FALSE, trim = TRUE))
        label <- gsub("\\.", "p", label)
        name <- paste0("signal__", feature, "__k", label)
        signal_terms <- c(signal_terms, sprintf(
            "{\"name\": %s, \"feature\": %s, \"knot\": %s}",
            str_q(name), str_q(feature), num(knot)))
    }
}
signal_spec <- vapply(names(model$signal_spec), function(feature) {
    spec <- model$signal_spec[[feature]]
    sprintf("%s: {\"clip\": %s, \"knots\": %s}", str_q(feature),
            arr(num(spec$clip)), arr(num(spec$knots)))
}, character(1L))
scaler <- model$design_scaler
design <- sprintf("{\"names\": %s, \"center\": %s, \"scale\": %s}",
                  arr(str_q(names(scaler$center))), arr(num(scaler$center)),
                  arr(num(scaler$scale)))

lines <- c(
    "{",
    sprintf("  \"format\": %s,", str_q("genboostgpu-joint-pve-model-v1")),
    sprintf("  \"source_rds_sha256\": %s,", str_q(observed)),
    sprintf("  \"family\": %s,", str_q(model$family)),
    sprintf("  \"gate_version\": %s,", str_q(model$gate_version)),
    sprintf("  \"lambda\": %s,", num(model$lambda)),
    sprintf("  \"output_lower\": %s,", num(model$output_lower)),
    sprintf("  \"output_upper\": %s,", num(model$output_upper)),
    sprintf("  \"conformal_q\": %s,", num(model$conformal_q)),
    sprintf("  \"null_cutoff\": %s,", num(model$null_cutoff)),
    sprintf("  \"signal_spec\": {%s},", paste(signal_spec, collapse = ", ")),
    sprintf("  \"signal_terms\": [%s],", paste(signal_terms, collapse = ", ")),
    sprintf("  \"design_scaler\": %s,", design),
    sprintf("  \"coefficients\": {\"names\": %s, \"values\": %s},",
            arr(str_q(coef_names)), arr(num(coef_values))),
    sprintf("  \"exported_by\": %s,", str_q(R.version.string)),
    sprintf("  \"glmnet_version\": %s", str_q(as.character(packageVersion("glmnet")))),
    "}"
)
writeLines(lines, out_path)
cat("Wrote", out_path, "from", model_path, "(sha256", observed, ")\n")
