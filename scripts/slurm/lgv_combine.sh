#!/bin/bash
# Reconcile shards, apply the frozen joint model, score and run the QC gate.
#SBATCH --job-name=gbg_lgv_combine
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=01:00:00
set -euo pipefail
export PYTHONUNBUFFERED=1
RUN_DIR=${GBG_RUN_DIR:?GBG_RUN_DIR is required}
PYTHON=${GBG_PYTHON:-python}
"${PYTHON}" -m genboostgpu.cli lgv combine --run-dir "${RUN_DIR}" ${GBG_COMBINE_ARGS:-}
