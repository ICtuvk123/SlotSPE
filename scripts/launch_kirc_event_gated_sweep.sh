#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

usage() {
    cat <<'EOF'
Launch a KIRC event-gated hyperparameter sweep in tmux with exactly three GPUs.

Usage:
  bash scripts/launch_kirc_event_gated_sweep.sh

Common overrides:
  GPU_IDS="0 1 2"                         Three physical GPU IDs (exactly three)
  LRS="0.0003 0.0004 0.0005 0.0006 0.0007"
  TAU_EVENTS="0.05 0.10"
  LAMBDA_EVENTS="0.5 1.0 2.0"
  TOPK_RATIOS="0.50"
  DELTA_PATCH_EVENTS="0.20"
  DELTA_SEMS="0.20"
  DELTA_VIS="0.30"
  MAX_EPOCHS=30 BATCH_SIZE=32 SEED=3
  WAIT_FOR_GPU_FREE_MIB=18000              Wait instead of starting on a busy GPU
  HYPERSEARCH_ROOT=./hypersearch
  SESSION_NAME=slotspe_event_sweep
  TAG_PREFIX=eg_sweep
  DRY_RUN=1                              Only generate and print the task table

Each GPU runs one worker and consumes its assigned experiments sequentially.
Completed experiments (those with summary.csv) are skipped when relaunched.
The monitor refreshes live_ranking.csv whenever another group finishes.
EOF
}

safe_token() {
    local value="$1"
    value="${value//./p}"
    value="${value//-/m}"
    value="${value//\//_}"
    printf '%s' "${value}"
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
        echo "$(date '+%F %T') GPU ${gpu_id} has ${free_mib:-unknown} MiB free; waiting for ${WAIT_FOR_GPU_FREE_MIB} MiB."
        sleep 60
    done
}

run_worker() {
    local worker_id="$1"
    local gpu_id="$2"
    local run_dir="$3"
    local task_file="$4"
    local worker_count=3
    local failures=0

    mkdir -p "${run_dir}/logs" "${run_dir}/status"
    trap 'touch "${run_dir}/status/worker_'"${worker_id}"'.done"' EXIT

    while IFS=$'\t' read -r task_id tag lr tau_event lambda_event topk \
        delta_patch delta_sem delta_vis; do
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
        SEED="${SEED}" \
        DATA_ROOT_DIR="${DATA_ROOT_DIR}" \
        DATA_PATH="${DATA_PATH}" \
        RESULTS_DIR="${RESULTS_DIR}" \
        EVENT_BANK_PATH="${EVENT_BANK_PATH}" \
        PYTHON="${PYTHON_BIN}" \
        bash scripts/run_kirc_event_gated.sh \
            --gpu "${gpu_id}" \
            --lr "${lr}" \
            --topk_ratio "${topk}" \
            --lambda_recon_loss "${LAMBDA_RECON_LOSS}" \
            --tau_event "${tau_event}" \
            --lambda_event "${lambda_event}" \
            --delta_patch_event "${delta_patch}" \
            --delta_sem "${delta_sem}" \
            --delta_vis "${delta_vis}" \
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

write_ranking() {
    local task_file="$1"
    local output_file="$2"

    "${PYTHON_BIN}" - "${task_file}" "${RESULTS_DIR}" "${output_file}" <<'PY'
import csv
import sys
from pathlib import Path

import pandas as pd

task_file, results_dir, output_file = map(Path, sys.argv[1:])
with task_file.open(newline="") as handle:
    tasks = list(csv.DictReader(handle, delimiter="\t"))

root = results_dir / "kirc" / "SlotSPE"
rows = []
for task in tasks:
    matches = list(root.glob(f"*_sp_{task['tag']}/summary.csv"))
    if not matches:
        continue
    summary = matches[0]
    frame = pd.read_csv(summary)
    means = frame[frame["folds"].astype(str) == "mean"]
    stds = frame[frame["folds"].astype(str) == "std"]
    if means.empty:
        continue
    row = dict(task)
    row.update(
        mean_cindex=float(means.iloc[0]["val_cindex"]),
        std_cindex=(float(stds.iloc[0]["val_cindex"]) if not stds.empty else float("nan")),
        mean_ipcw=float(means.iloc[0]["val_cindex_ipcw"]),
        mean_ibs=float(means.iloc[0]["val_IBS"]),
        mean_iauc=float(means.iloc[0]["val_iauc"]),
        summary=str(summary),
    )
    rows.append(row)

if not rows:
    raise SystemExit(0)

ranking = pd.DataFrame(rows).sort_values("mean_cindex", ascending=False)
temporary_file = output_file.with_suffix(output_file.suffix + ".tmp")
ranking.to_csv(temporary_file, index=False)
temporary_file.replace(output_file)
print(f"\nCompleted groups: {len(ranking)}; current top results:\n")
print(ranking[[
    "tag", "lr", "tau_event", "lambda_event", "mean_cindex", "std_cindex",
]].head(10).to_string(index=False))
print(f"\nLive ranking: {output_file}")
PY
}

count_completed() {
    local task_file="$1"
    local completed=0
    local task_id tag remainder
    while IFS=$'\t' read -r task_id tag remainder; do
        [[ "${task_id}" == "task_id" ]] && continue
        if result_summary_for_tag "${tag}" >/dev/null 2>&1; then
            completed=$((completed + 1))
        fi
    done < "${task_file}"
    printf '%s' "${completed}"
}

run_monitor() {
    local run_dir="$1"
    local task_file="$2"
    local total_groups
    local last_completed=-1
    total_groups=$(( $(wc -l < "${task_file}") - 1 ))

    echo "Waiting for three GPU workers..."
    while :; do
        local finished=0
        local completed
        local i
        for i in 0 1 2; do
            [[ -f "${run_dir}/status/worker_${i}.done" ]] && finished=$((finished + 1))
        done
        completed="$(count_completed "${task_file}")"
        if (( completed != last_completed && completed > 0 )); then
            write_ranking "${task_file}" "${run_dir}/live_ranking.csv"
            last_completed="${completed}"
        fi
        echo "$(date '+%F %T') completed groups: ${completed}/${total_groups}; workers finished: ${finished}/3"
        (( finished == 3 )) && break
        sleep 20
    done

    write_ranking "${task_file}" "${run_dir}/ranking.csv"

    if [[ -s "${run_dir}/status/failures.tsv" ]]; then
        echo "Some experiments failed; see ${run_dir}/status/failures.tsv" >&2
    else
        echo "All experiments completed successfully."
    fi
    echo "This tmux session is retained for inspection; kill it when done."
}

if [[ "${1:-}" == "--worker" ]]; then
    shift
    run_worker "$@"
    exit
elif [[ "${1:-}" == "--monitor" ]]; then
    shift
    run_monitor "$@"
    exit
elif [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    usage
    exit
elif (( $# != 0 )); then
    usage >&2
    exit 2
fi

GPU_IDS="${GPU_IDS:-0 1 2}"
GPU_IDS="${GPU_IDS//,/ }"
read -r -a GPUS <<< "${GPU_IDS}"
if (( ${#GPUS[@]} != 3 )); then
    echo "ERROR: GPU_IDS must contain exactly three IDs; got: ${GPU_IDS}" >&2
    exit 2
fi

LRS="${LRS:-0.0003 0.0004 0.0005 0.0006 0.0007}"
TAU_EVENTS="${TAU_EVENTS:-0.05 0.10}"
LAMBDA_EVENTS="${LAMBDA_EVENTS:-0.5 1.0 2.0}"
TOPK_RATIOS="${TOPK_RATIOS:-0.50}"
DELTA_PATCH_EVENTS="${DELTA_PATCH_EVENTS:-0.20}"
DELTA_SEMS="${DELTA_SEMS:-0.20}"
DELTA_VIS="${DELTA_VIS:-0.30}"
MAX_EPOCHS="${MAX_EPOCHS:-30}"
BATCH_SIZE="${BATCH_SIZE:-32}"
SEED="${SEED:-3}"
LAMBDA_RECON_LOSS="${LAMBDA_RECON_LOSS:-0.01}"
WAIT_FOR_GPU_FREE_MIB="${WAIT_FOR_GPU_FREE_MIB:-18000}"
DATA_ROOT_DIR="${DATA_ROOT_DIR:-/data0/lfy_data/Pathology/CONCH/kirc/pt_files}"
DATA_PATH="${DATA_PATH:-./dataset_csv}"
HYPERSEARCH_ROOT="${HYPERSEARCH_ROOT:-./hypersearch}"
RESULTS_DIR="${RESULTS_DIR:-${HYPERSEARCH_ROOT}/results}"
EVENT_BANK_PATH="${EVENT_BANK_PATH:-assets/event_bank/tcga_kirc/conch_event_bank.pt}"
PYTHON_BIN="${PYTHON:-python}"
SESSION_NAME="${SESSION_NAME:-slotspe_event_sweep}"
TAG_PREFIX="${TAG_PREFIX:-eg_sweep}"
RUN_ID="${RUN_ID:-$(date '+%Y%m%d_%H%M%S')}"
RUN_DIR="${SWEEP_DIR:-${HYPERSEARCH_ROOT}/runs/${TAG_PREFIX}_${RUN_ID}}"
TASK_FILE="${RUN_DIR}/tasks.tsv"

for required_path in "${DATA_ROOT_DIR}" "${DATA_PATH}" "${EVENT_BANK_PATH}"; do
    if [[ ! -e "${required_path}" ]]; then
        echo "ERROR: required path does not exist: ${required_path}" >&2
        exit 1
    fi
done
command -v tmux >/dev/null || { echo "ERROR: tmux is not installed" >&2; exit 1; }
command -v "${PYTHON_BIN}" >/dev/null || { echo "ERROR: Python not found: ${PYTHON_BIN}" >&2; exit 1; }

mkdir -p "${RUN_DIR}/logs" "${RUN_DIR}/status" "${RESULTS_DIR}"
printf 'task_id\ttag\tlr\ttau_event\tlambda_event\ttopk_ratio\tdelta_patch_event\tdelta_sem\tdelta_vis\n' \
    > "${TASK_FILE}"

task_id=0
for lr in ${LRS}; do
    for tau_event in ${TAU_EVENTS}; do
        for lambda_event in ${LAMBDA_EVENTS}; do
            for topk in ${TOPK_RATIOS}; do
                for delta_patch in ${DELTA_PATCH_EVENTS}; do
                    for delta_sem in ${DELTA_SEMS}; do
                        for delta_vis in ${DELTA_VIS}; do
                            tag="${TAG_PREFIX}_lr$(safe_token "${lr}")_tau$(safe_token "${tau_event}")_lam$(safe_token "${lambda_event}")_topk$(safe_token "${topk}")_dp$(safe_token "${delta_patch}")_ds$(safe_token "${delta_sem}")_dv$(safe_token "${delta_vis}")_s${SEED}"
                            printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
                                "${task_id}" "${tag}" "${lr}" "${tau_event}" \
                                "${lambda_event}" "${topk}" "${delta_patch}" \
                                "${delta_sem}" "${delta_vis}" >> "${TASK_FILE}"
                            task_id=$((task_id + 1))
                        done
                    done
                done
            done
        done
    done
done

echo "Generated ${task_id} experiments in ${TASK_FILE}"
echo "Budget estimate: 30-epoch 5-fold historical runs took 3.46-4.72 h/group."
echo "With three GPUs, ${task_id} groups are approximately $(( (task_id + 2) / 3 )) groups/GPU."
column -t -s $'\t' "${TASK_FILE}" 2>/dev/null || sed -n '1,12p' "${TASK_FILE}"
if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "DRY_RUN=1: tmux was not started."
    exit
fi
if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
    echo "ERROR: tmux session already exists: ${SESSION_NAME}" >&2
    echo "Attach with: tmux attach -t ${SESSION_NAME}" >&2
    exit 1
fi

export MAX_EPOCHS BATCH_SIZE SEED LAMBDA_RECON_LOSS WAIT_FOR_GPU_FREE_MIB
export DATA_ROOT_DIR DATA_PATH
export RESULTS_DIR EVENT_BANK_PATH PYTHON_BIN

tmux_env=(
    "MAX_EPOCHS=${MAX_EPOCHS}"
    "BATCH_SIZE=${BATCH_SIZE}"
    "SEED=${SEED}"
    "LAMBDA_RECON_LOSS=${LAMBDA_RECON_LOSS}"
    "WAIT_FOR_GPU_FREE_MIB=${WAIT_FOR_GPU_FREE_MIB}"
    "DATA_ROOT_DIR=${DATA_ROOT_DIR}"
    "DATA_PATH=${DATA_PATH}"
    "RESULTS_DIR=${RESULTS_DIR}"
    "EVENT_BANK_PATH=${EVENT_BANK_PATH}"
    "PYTHON_BIN=${PYTHON_BIN}"
)

printf -v monitor_cmd '%q ' env "${tmux_env[@]}" bash "$0" \
    --monitor "${RUN_DIR}" "${TASK_FILE}"
tmux new-session -d -s "${SESSION_NAME}" -n monitor "${monitor_cmd}"
tmux set-window-option -t "${SESSION_NAME}:monitor" remain-on-exit on

for worker_id in 0 1 2; do
    printf -v worker_cmd '%q ' env "${tmux_env[@]}" bash "$0" \
        --worker "${worker_id}" "${GPUS[${worker_id}]}" "${RUN_DIR}" "${TASK_FILE}"
    tmux new-window -d -t "${SESSION_NAME}" -n "gpu${GPUS[${worker_id}]}" "${worker_cmd}"
done

echo "Started tmux session '${SESSION_NAME}' with GPUs: ${GPUS[*]}"
echo "Attach:  tmux attach -t ${SESSION_NAME}"
echo "Logs:    ${RUN_DIR}/logs"
echo "Live:    ${RUN_DIR}/live_ranking.csv (refreshed after each completed group)"
echo "Ranking: ${RUN_DIR}/ranking.csv (created after all workers finish)"
