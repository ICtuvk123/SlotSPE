#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

GPU_IDS="${GPU_IDS:-5 6 7}"
STATE_DIR="${STATE_DIR:-./hypersearch/regularization_search}"
RESULTS_DIR="${RESULTS_DIR:-./hypersearch/results}"
POLL_SECONDS="${POLL_SECONDS:-30}"
PHASES=(reference optimizer dropout combine refine final)

result_summary_for_tag() {
    local tag="$1"
    local result_root="${RESULTS_DIR}/kirc/SlotSPE"
    local summary
    [[ -d "${result_root}" ]] || return 1
    summary="$(find "${result_root}" -mindepth 2 -maxdepth 2 -type f \
        -path "*_sp_${tag}/summary.csv" -print -quit)"
    [[ -n "${summary}" ]] || return 1
}

phase_counts() {
    local task_file="$1"
    local total=0
    local completed=0
    local task_id phase variant tag remainder
    while IFS=$'\t' read -r task_id phase variant tag remainder; do
        [[ "${task_id}" == "task_id" ]] && continue
        total=$((total + 1))
        if result_summary_for_tag "${tag}"; then
            completed=$((completed + 1))
        fi
    done < "${task_file}"
    printf '%s %s\n' "${completed}" "${total}"
}

latest_run_for_phase() {
    local phase="$1"
    find "${STATE_DIR}/runs" -mindepth 1 -maxdepth 1 -type d \
        -name "${phase}_*" -print 2>/dev/null | sort | tail -n 1
}

for phase in "${PHASES[@]}"; do
    session_name="slotspe_reg_${phase}"
    task_file="${STATE_DIR}/tasks/${phase}.tsv"

    if ! tmux has-session -t "${session_name}" 2>/dev/null; then
        echo "$(date '+%F %T') launching phase ${phase} on GPUs ${GPU_IDS}"
        GPU_IDS="${GPU_IDS}" STATE_DIR="${STATE_DIR}" RESULTS_DIR="${RESULTS_DIR}" \
            SESSION_NAME="${session_name}" \
            bash scripts/launch_kirc_regularization_search.sh "${phase}"
    else
        echo "$(date '+%F %T') phase ${phase} already has session ${session_name}"
    fi

    if [[ ! -f "${task_file}" ]]; then
        echo "ERROR: phase ${phase} did not create ${task_file}" >&2
        exit 1
    fi

    while :; do
        read -r completed total < <(phase_counts "${task_file}")
        echo "$(date '+%F %T') phase ${phase}: ${completed}/${total} completed"
        if (( completed == total )); then
            break
        fi

        latest_run="$(latest_run_for_phase "${phase}")"
        if [[ -n "${latest_run}" ]]; then
            finished_workers=0
            for worker_id in 0 1 2; do
                [[ -f "${latest_run}/status/worker_${worker_id}.done" ]] \
                    && finished_workers=$((finished_workers + 1))
            done
            if (( finished_workers == 3 )); then
                echo "ERROR: phase ${phase} workers exited with incomplete results." >&2
                [[ -f "${latest_run}/status/failures.tsv" ]] \
                    && sed -n '1,120p' "${latest_run}/status/failures.tsv" >&2
                exit 1
            fi
        fi
        sleep "${POLL_SECONDS}"
    done
done

python scripts/regularization_search.py report \
    --state-dir "${STATE_DIR}" \
    --results-dir "${RESULTS_DIR}" \
    --output "${STATE_DIR}/final_3seed_report.csv"
echo "$(date '+%F %T') all regularization phases completed"
