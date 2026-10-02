#!/bin/bash
# GEMMA-only shard for site runs initialized with --bslmm separate (no GPU).
# May run on another cluster than the GPU shards; see docs/user-guide/sites_and_cost.rst.
#SBATCH --job-name=gbg_sites_bslmm
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=04:00:00
set -euo pipefail
export PYTHONUNBUFFERED=1
RUN_DIR=${GBG_RUN_DIR:?GBG_RUN_DIR is required}
N_SHARDS=${GBG_N_SHARDS:?GBG_N_SHARDS is required}
PYTHON=${GBG_PYTHON:-python}
SHARD=${SLURM_ARRAY_TASK_ID:-0}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
"${PYTHON}" -m genboostgpu.cli sites bslmm --run-dir "${RUN_DIR}" \
    --shard "${SHARD}/${N_SHARDS}" --gemma-workers "${SLURM_CPUS_PER_TASK:-8}"
