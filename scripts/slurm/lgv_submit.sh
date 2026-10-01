#!/bin/bash
# Submit a GENBoostGPU Module 02 run: feature shards -> (BSLMM shards) -> combine.
#
#   GBG_RUN_DIR=<run dir> GBG_N_SHARDS=8 GBG_ACCOUNT=<acct> scripts/slurm/lgv_submit.sh [gpu|cpu]
#
# The run directory must exist (genboostgpu lgv init). Combine is chained with
# afterany so a cancelled shard is reconciled (and fails the QC gate) instead
# of silently blocking; rerunning a shard resumes from its written rows.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
MODE=${1:-gpu}
: "${GBG_RUN_DIR:?}" "${GBG_N_SHARDS:?}" "${GBG_ACCOUNT:?}"
LOGS=${GBG_RUN_DIR}/logs
mkdir -p "${LOGS}"
LAST=$(( GBG_N_SHARDS - 1 ))
THROTTLE=${GBG_THROTTLE:-200}
COMMON=(--account="${GBG_ACCOUNT}" --export=ALL)
BSLMM_MODE=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['bslmm_mode'])" "${GBG_RUN_DIR}/run.json")
if [[ "${MODE}" == "gpu" ]]; then
  STEP=${HERE}/lgv_gpu.sh
else
  STEP=${HERE}/lgv_cpu.sh
fi
FEAT=$(sbatch --parsable "${COMMON[@]}" --array=0-${LAST}%${THROTTLE} \
       --output="${LOGS}/features_%A_%a.log" "${STEP}")
DEPS="afterany:${FEAT}"
if [[ "${BSLMM_MODE}" == "separate" ]]; then
  BS=$(sbatch --parsable "${COMMON[@]}" --array=0-${LAST}%${THROTTLE} \
       --output="${LOGS}/bslmm_%A_%a.log" "${HERE}/lgv_bslmm_cpu.sh")
  DEPS="${DEPS}:${BS}"
fi
COMB=$(sbatch --parsable "${COMMON[@]}" --dependency="${DEPS}" \
       --output="${LOGS}/combine_%j.log" "${HERE}/lgv_combine.sh")
printf 'step\tjob_id\tscript\n' > "${GBG_RUN_DIR}/submitted-jobs.tsv.tmp"
printf 'features\t%s\t%s\n' "${FEAT}" "${STEP}" >> "${GBG_RUN_DIR}/submitted-jobs.tsv.tmp"
[[ -n "${BS:-}" ]] && printf 'bslmm\t%s\t%s\n' "${BS}" "${HERE}/lgv_bslmm_cpu.sh" >> "${GBG_RUN_DIR}/submitted-jobs.tsv.tmp"
printf 'combine\t%s\t%s\n' "${COMB}" "${HERE}/lgv_combine.sh" >> "${GBG_RUN_DIR}/submitted-jobs.tsv.tmp"
cat "${GBG_RUN_DIR}/submitted-jobs.tsv.tmp" >> "${GBG_RUN_DIR}/submitted-jobs.tsv"
rm -f "${GBG_RUN_DIR}/submitted-jobs.tsv.tmp"
cat "${GBG_RUN_DIR}/submitted-jobs.tsv"
