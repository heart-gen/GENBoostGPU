#!/bin/bash
# One GPU shard of a GENBoostGPU Module 02 run (array task = shard index).
# Edit the #SBATCH account/partition for your allocation, or pass them to sbatch.
#SBATCH --job-name=gbg_lgv_gpu
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
set -euo pipefail
export PYTHONUNBUFFERED=1
RUN_DIR=${GBG_RUN_DIR:?GBG_RUN_DIR is required}
N_SHARDS=${GBG_N_SHARDS:?GBG_N_SHARDS is required}
PYTHON=${GBG_PYTHON:-python}
SHARD=${SLURM_ARRAY_TASK_ID:-0}
# GEMMA (bslmm inline) uses the cores the GPU work leaves idle.
GEMMA_WORKERS=${GBG_GEMMA_WORKERS:-$(( ${SLURM_CPUS_PER_TASK:-16} - 4 ))}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
"${PYTHON}" -m genboostgpu.cli lgv run --run-dir "${RUN_DIR}" \
    --shard "${SHARD}/${N_SHARDS}" --device gpu \
    --batch-loci "${GBG_BATCH_LOCI:-32}" --prep-threads 3 \
    --gemma-workers "${GEMMA_WORKERS}"
