#!/bin/bash
# One CPU-only shard (no GPU): glmnet in numba threads, GEMMA in processes.
#SBATCH --job-name=gbg_lgv_cpu
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
CPUS=${SLURM_CPUS_PER_TASK:-8}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
"${PYTHON}" -m genboostgpu.cli lgv run --run-dir "${RUN_DIR}" \
    --shard "${SHARD}/${N_SHARDS}" --device cpu --batch-loci 4 \
    --prep-threads 1 --cpu-threads "$(( CPUS / 2 ))" --gemma-workers "$(( CPUS / 2 ))"
