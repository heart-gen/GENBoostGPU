#!/bin/bash
# Site-level inputs for `genboostgpu sites` (replaces prepare_cpg_inputs.R).
#
#   scripts/prepare_site_inputs.sh --input <store> --chrom 21 --context CA \
#       --out <build dir> [--samples sheet.tsv --coldata-id brnum] \
#       [--min-cov 5 --min-covered-frac 0.8 --min-sd 0.02 --top-per-chrom 200000]
#
# Writes <out>/units/chr{c}.beta.h5 and chr{c}.units.tsv.gz: one unit per
# cytosine passing coverage QC and the variance prefilter, with an explicit
# sample_id vector, read chunk by chunk from HDF5-backed stores (the legacy
# prepare_cpg_inputs.R built a genome-wide coverage matrix per chromosome and
# wrote no sample IDs). Any extra options are passed to build_regions.R.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
RSCRIPT=${RSCRIPT:-Rscript}
exec "${RSCRIPT}" "${HERE}/build_regions.R" --step qc --unit site "$@"
