#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

usage() {
    cat <<'EOF'
Run one recoverable phase of the KIRC Event-Gated regularization search.

Usage:
  bash scripts/launch_kirc_regularization_search.sh PHASE

Phases must be run in order:
  reference -> optimizer -> dropout -> combine -> refine -> final

Important overrides:
  GPU_IDS="0 1 2"                     exactly three GPUs
  STATE_DIR=./hypersearch/regularization_search
  RESULTS_DIR=./hypersearch/results
  DATA_ROOT_DIR=/data0/lfy_data/Pathology/CONCH/kirc/pt_files
  EVENT_BANK_PATH=assets/event_bank/tcga_kirc/conch_event_bank.pt
  MAX_EPOCHS=30 BATCH_SIZE=32 EVAL_SLOT_SEED=100000
  WAIT_FOR_GPU_FREE_MIB=18000
  DRY_RUN=1                            generate tasks without starting tmux

Each completed summary.csv is skipped when a phase is relaunched. Later phases
select their base configuration directly from earlier completed summaries.
EOF
}

result_summary_for_tag() {
    local tag="$1"
    local result_root="${RESULTS_DIR}/kirc/SlotSPE"
    local summary
    [[ -d "${result_root}" ]] || return 1
    summary="$(find "${result_root}" -mindepth 2 -maxdepth 2 -type f \
        -path "*_sp_${tag}/summary.csv" -print -quit)"
    [[ -n "${summary}" ]] || return 1
    printf '%s\n' "${summary}"
}

wait_for_gpu_memory() {
    local gpu_id="$1"
    local free_mib
    if (( WAIT_FOR_GPU_FREE_MIB <= 0 )); then
        return
    fi
    while :; do
        free_mib="$(nvidia-smi --id="${gpu_id}" --query-gpu=memory.free \
            --format=csv,noheader,nounits 2>/dev/null | head -n 1 | tr -dc '0-9')"
        if [[ -n "${free_mib}" ]] && (( free_mib >= WAIT_FOR_GPU_FREE_MIB )); then
            echo "GPU ${gpu_id} is ready: ${free_mib} MiB free."
            return
        fi
        echo "$(date '+%F %T') GPU ${gpu_id} has ${free_mib:-unknown} MiB free; waiting."
        sleep 60
    done
}

run_worker() {
    local worker_id="$1"
    local gpu_id="$2"
    local run_dir="$3"
    local task_file="$4"
    local failures=0
    local worker_count=3

    mkdir -p "${run_dir}/logs" "${run_dir}/status"
    trap 'touch "'"${run_dir}"'/status/worker_'"${worker_id}"'.done"' EXIT

    while IFS=$'\t' read -r task_id phase variant tag source_tag seed opt lr \
        weight_decay tau_event lambda_event topk_ratio wsi_dropout fusion_dropout \
        event_dropout lambda_decoder lambda_recon event_dim slot_iters; do
        [[ "${task_id}" == "task_id" ]] && continue
        (( task_id % worker_count == worker_id )) || continue

        local completed_summary=""
        completed_summary="$(result_summary_for_tag "${tag}" || true)"
        if [[ -n "${completed_summary}" ]]; then
            echo "[worker ${worker_id} | GPU ${gpu_id}] SKIP ${tag}: ${completed_summary}"
            continue
        fi

        wait_for_gpu_memory "${gpu_id}"
        echo "[worker ${worker_id} | GPU ${gpu_id}] START ${tag}"
        set +e
        SPECIFIC_SIMPLE="${tag}" \
        MAX_EPOCHS="${MAX_EPOCHS}" \
        BATCH_SIZE="${BATCH_SIZE}" \
        SEED="${seed}" \
        EVAL_SLOT_SEED="${EVAL_SLOT_SEED}" \
        DATA_ROOT_DIR="${DATA_ROOT_DIR}" \
        DATA_PATH="${DATA_PATH}" \
        RESULTS_DIR="${RESULTS_DIR}" \
        EVENT_BANK_PATH="${EVENT_BANK_PATH}" \
        PYTHON="${PYTHON_BIN}" \
        bash scripts/run_kirc_event_gated.sh \
            --gpu "${gpu_id}" \
            --opt "${opt}" \
            --lr "${lr}" \
            --weight_decay "${weight_decay}" \
            --tau_event "${tau_event}" \
            --lambda_event "${lambda_event}" \
            --topk_ratio "${topk_ratio}" \
            --wsi_projection_dropout "${wsi_dropout}" \
            --fusion_dropout "${fusion_dropout}" \
            --event_residual_dropout "${event_dropout}" \
            --lambda_decoder_loss "${lambda_decoder}" \
            --lambda_recon_loss "${lambda_recon}" \
            --event_projection_dim "${event_dim}" \
            --slot_iters "${slot_iters}" \
            2>&1 | tee "${run_dir}/logs/${tag}.log"
        local exit_code="${PIPESTATUS[0]}"
        set -e

        if (( exit_code == 0 )); then
            echo "[worker ${worker_id} | GPU ${gpu_id}] DONE ${tag}"
        else
            echo "[worker ${worker_id} | GPU ${gpu_id}] FAILED(${exit_code}) ${tag}" >&2
            printf '%s\t%s\t%s\n' "${task_id}" "${tag}" "${exit_code}" \
                >> "${run_dir}/status/failures.tsv"
            failures=$((failures + 1))
        fi
    done < "${task_file}"

    (( failures == 0 ))
}

count_completed() {
    local task_file="$1"
    local completed=0
    local task_id phase variant tag remainder
    while IFS=$'\t' read -r task_id phase variant tag remainder; do
        [[ "${task_id}" == "task_id" ]] && continue
        if result_summary_for_tag "${tag}" >/dev/null 2>&1; then
            completed=$((completed + 1))
        fi
    done < "${task_file}"
    printf '%s' "${completed}"
}

write_ranking() {
    local task_file="$1"
    local output_file="$2"
    "${PYTHON_BIN}" scripts/regularization_search.py rank \
        --tasks "${task_file}" \
        --results-dir "${RESULTS_DIR}" \
        --output "${output_file}"
}

run_monitor() {
    local phase="$1"
    local run_dir="$2"
    local task_file="$3"
    local total_groups
    local last_completed=-1
    total_groups=$(( $(wc -l < "${task_file}") - 1 ))

    echo "Waiting for three GPU workers for phase ${phase}..."
    while :; do
        local finished=0
        local completed
        local worker_id
        for worker_id in 0 1 2; do
            [[ -f "${run_dir}/status/worker_${worker_id}.done" ]] && finished=$((finished + 1))
        done
        completed="$(count_completed "${task_file}")"
        if (( completed != last_completed && completed > 0 )); then
            write_ranking "${task_file}" "${run_dir}/live_ranking.csv"
            last_completed="${completed}"
        fi
        echo "$(date '+%F %T') completed: ${completed}/${total_groups}; workers: ${finished}/3"
        (( finished == 3 )) && break
        sleep 20
    done

    write_ranking "${task_file}" "${STATE_DIR}/rankings/${phase}.csv"
    if [[ "${phase}" == "final" ]]; then
        "${PYTHON_BIN}" scripts/regularization_search.py report \
            --state-dir "${STATE_DIR}" \
            --results-dir "${RESULTS_DIR}" \
            --output "${STATE_DIR}/final_3seed_report.csv"
    fi
    if [[ -s "${run_dir}/status/failures.tsv" ]]; then
        echo "Some experiments failed: ${run_dir}/status/failures.tsv" >&2
    else
        echo "Phase ${phase} completed successfully."
    fi
}

if [[ "${1:-}" == "--worker" ]]; then
    shift
    run_worker "$@"
    exit
elif [[ "${1:-}" == "--monitor" ]]; then
    shift
    run_monitor "$@"
    exit
elif [[ "${1:-}" == "--help" || "${1:-}" == "-h" || $# -ne 1 ]]; then
    usage
    [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]] && exit 0
    exit 2
fi

PHASE="$1"
case "${PHASE}" in
    reference|optimizer|dropout|combine|refine|final) ;;
    *) usage >&2; exit 2 ;;
esac

GPU_IDS="${GPU_IDS:-0 1 2}"
GPU_IDS="${GPU_IDS//,/ }"
read -r -a GPUS <<< "${GPU_IDS}"
if (( ${#GPUS[@]} != 3 )); then
    echo "ERROR: GPU_IDS must contain exactly three IDs; got ${GPU_IDS}" >&2
    exit 2
fi

STATE_DIR="${STATE_DIR:-./hypersearch/regularization_search}"
RESULTS_DIR="${RESULTS_DIR:-./hypersearch/results}"
DATA_ROOT_DIR="${DATA_ROOT_DIR:-/data0/lfy_data/Pathology/CONCH/kirc/pt_files}"
DATA_PATH="${DATA_PATH:-./dataset_csv}"
EVENT_BANK_PATH="${EVENT_BANK_PATH:-assets/event_bank/tcga_kirc/conch_event_bank.pt}"
MAX_EPOCHS="${MAX_EPOCHS:-30}"
BATCH_SIZE="${BATCH_SIZE:-32}"
EVAL_SLOT_SEED="${EVAL_SLOT_SEED:-100000}"
WAIT_FOR_GPU_FREE_MIB="${WAIT_FOR_GPU_FREE_MIB:-18000}"
PYTHON_BIN="${PYTHON:-python}"
SESSION_NAME="${SESSION_NAME:-slotspe_reg_${PHASE}}"
RUN_ID="${RUN_ID:-$(date '+%Y%m%d_%H%M%S')}"
RUN_DIR="${STATE_DIR}/runs/${PHASE}_${RUN_ID}"
TASK_FILE="${STATE_DIR}/tasks/${PHASE}.tsv"

mkdir -p "${STATE_DIR}/tasks" "${STATE_DIR}/rankings" "${RESULTS_DIR}"
"${PYTHON_BIN}" scripts/regularization_search.py plan \
    --phase "${PHASE}" \
    --state-dir "${STATE_DIR}" \
    --results-dir "${RESULTS_DIR}" \
    --output "${TASK_FILE}"

task_count=$(( $(wc -l < "${TASK_FILE}") - 1 ))
echo "Generated ${task_count} task(s) for ${PHASE}: ${TASK_FILE}"
column -t -s $'\t' "${TASK_FILE}" 2>/dev/null || sed -n '1,12p' "${TASK_FILE}"
if (( task_count == 0 )); then
    echo "No experiment is needed for phase ${PHASE}; continue with the next phase."
    exit 0
fi
if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "DRY_RUN=1: tmux was not started."
    exit 0
fi

for required_path in "${DATA_ROOT_DIR}" "${DATA_PATH}" "${EVENT_BANK_PATH}"; do
    if [[ ! -e "${required_path}" ]]; then
        echo "ERROR: required path does not exist: ${required_path}" >&2
        exit 1
    fi
done
command -v tmux >/dev/null || { echo "ERROR: tmux is not installed" >&2; exit 1; }
command -v "${PYTHON_BIN}" >/dev/null || { echo "ERROR: Python not found: ${PYTHON_BIN}" >&2; exit 1; }
if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
    echo "ERROR: tmux session already exists: ${SESSION_NAME}" >&2
    exit 1
fi

mkdir -p "${RUN_DIR}/logs" "${RUN_DIR}/status"
export STATE_DIR RESULTS_DIR DATA_ROOT_DIR DATA_PATH EVENT_BANK_PATH
export MAX_EPOCHS BATCH_SIZE EVAL_SLOT_SEED WAIT_FOR_GPU_FREE_MIB PYTHON_BIN

runtime_env=(
    "STATE_DIR=${STATE_DIR}"
    "RESULTS_DIR=${RESULTS_DIR}"
    "DATA_ROOT_DIR=${DATA_ROOT_DIR}"
    "DATA_PATH=${DATA_PATH}"
    "EVENT_BANK_PATH=${EVENT_BANK_PATH}"
    "MAX_EPOCHS=${MAX_EPOCHS}"
    "BATCH_SIZE=${BATCH_SIZE}"
    "EVAL_SLOT_SEED=${EVAL_SLOT_SEED}"
    "WAIT_FOR_GPU_FREE_MIB=${WAIT_FOR_GPU_FREE_MIB}"
    "PYTHON_BIN=${PYTHON_BIN}"
)

printf -v monitor_cmd '%q ' env "${runtime_env[@]}" bash "$0" \
    --monitor "${PHASE}" "${RUN_DIR}" "${TASK_FILE}"
tmux new-session -d -s "${SESSION_NAME}" -n monitor "${monitor_cmd}"
tmux set-window-option -t "${SESSION_NAME}:monitor" remain-on-exit on

for worker_id in 0 1 2; do
    printf -v worker_cmd '%q ' env "${runtime_env[@]}" bash "$0" --worker \
        "${worker_id}" "${GPUS[${worker_id}]}" "${RUN_DIR}" "${TASK_FILE}"
    tmux new-window -d -t "${SESSION_NAME}" -n "gpu${GPUS[${worker_id}]}" "${worker_cmd}"
done

echo "Started phase ${PHASE} in tmux session ${SESSION_NAME}."
echo "Attach with: tmux attach -t ${SESSION_NAME}"
