#!/usr/bin/env bash
set -euo pipefail

# Read-only storage audit. This script never deletes files.

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
UNI_DIR=${UNI_DIR:-"${HOME}/Data/Pathology/UNI/kirc"}
CONCH_DIR=${CONCH_DIR:-"/data0/lfy_data/Pathology/CONCH/kirc"}
FAILED_RAW_DIR=${FAILED_RAW_DIR:-"${HOME}/Data/Pathology/Slides/kirc_failed3_work"}

show_path() {
    local label="$1"
    local path="$2"
    if [[ -e "${path}" ]]; then
        printf '%-28s ' "${label}"
        du -sh "${path}"
    else
        printf '%-28s %s\n' "${label}" "missing: ${path}"
    fi
}

echo "KIRC storage audit (read-only)"
df -h "${ROOT_DIR}"
echo
show_path "Legacy UNI features" "${UNI_DIR}"
show_path "New CONCH features" "${CONCH_DIR}"
show_path "Failed-slide raw work" "${FAILED_RAW_DIR}"
show_path "Training results" "${ROOT_DIR}/results_train"
show_path "Test results" "${ROOT_DIR}/results_test"
show_path "UNI checkpoint" "${HOME}/Data/Pathology/checkpoints/vit_large_patch16_224.dinov2.uni_mass100k"

if [[ -d "${UNI_DIR}/pt_files" ]]; then
    printf 'Legacy UNI PT count: '
    find "${UNI_DIR}/pt_files" -maxdepth 1 -type f -name '*.pt' | wc -l
fi
if [[ -d "${CONCH_DIR}/pt_files" ]]; then
    printf 'CONCH PT count: '
    find "${CONCH_DIR}/pt_files" -maxdepth 1 -type f -name '*.pt' | wc -l
fi

echo
echo "Deletion policy: retain UNI until CONCH validation_report.json and both training smoke tests pass."
echo "This audit intentionally performs no cleanup."
