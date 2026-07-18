#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON:-python}"
V2_DIR="assets/event_bank/tcga_kirc/v2"
PROMPT="assets/event_bank/prompts/generate_pathology_events_v2.md"
CANDIDATES="${V2_DIR}/events_candidates.json"
CURATION="${V2_DIR}/curation_decisions.json"
PATCH_CORE="${V2_DIR}/events_patch_core.json"
CONTEXT="${V2_DIR}/events_context.json"
RESERVE="${V2_DIR}/events_reserve.json"
BANK="${V2_DIR}/conch_event_bank_v2.pt"
V0_BANK="assets/event_bank/tcga_kirc/conch_event_bank.pt"
AUDIT="${V2_DIR}/analysis/v0_v2_audit.json"
CHECKPOINT="${CONCH_CHECKPOINT:-/data0/lfy_data/Pathology/checkpoints/conch/pytorch_model.bin}"

bash scripts/setup_conch.sh

if [[ ! -f "${CANDIDATES}" ]]; then
    if ! "${PYTHON_BIN}" tools/generate_event_candidates.py \
        --cancer-type "kidney renal clear cell carcinoma" \
        --dataset "TCGA-KIRC" \
        --num-events 38 \
        --prompt "${PROMPT}" \
        --output "${CANDIDATES}"; then
        echo "V2 generation paused. Save valid JSON at ${CANDIDATES}, then rerun." >&2
        exit 2
    fi
else
    echo "Using existing v2 candidate file: ${CANDIDATES}"
fi

"${PYTHON_BIN}" tools/review_event_candidates.py \
    --input "${CANDIDATES}" \
    --output "${PATCH_CORE}" \
    --curation-file "${CURATION}" \
    --context-output "${CONTEXT}" \
    --reserve-output "${RESERVE}"

if [[ "${CONCH_FROM_HF:-0}" == "1" ]]; then
    "${PYTHON_BIN}" tools/build_conch_event_bank.py \
        --events "${PATCH_CORE}" --output "${BANK}" --from-hf \
        --device "${CONCH_DEVICE:-cuda}" --batch-size "${CONCH_BATCH_SIZE:-32}"
else
    "${PYTHON_BIN}" tools/build_conch_event_bank.py \
        --events "${PATCH_CORE}" --output "${BANK}" --checkpoint "${CHECKPOINT}" \
        --device "${CONCH_DEVICE:-cuda}" --batch-size "${CONCH_BATCH_SIZE:-32}"
fi

if [[ -f "${V0_BANK}" ]]; then
    ANALYSIS_ARGS=(
        --v0-bank "${V0_BANK}"
        --v2-bank "${BANK}"
        --output "${AUDIT}"
        --device "${CONCH_DEVICE:-cuda}"
    )
    if [[ -n "${V2_PATCH_FEATURE_DIR:-}" ]]; then
        if [[ ! -d "${V2_PATCH_FEATURE_DIR}" ]]; then
            echo "ERROR: V2_PATCH_FEATURE_DIR is not a directory: ${V2_PATCH_FEATURE_DIR}" >&2
            exit 1
        fi
        ANALYSIS_ARGS+=(--patch-feature-dir "${V2_PATCH_FEATURE_DIR}")
    fi
    "${PYTHON_BIN}" tools/analyze_event_banks.py "${ANALYSIS_ARGS[@]}"
else
    echo "Skipping v0/v2 audit because the v0 bank is missing: ${V0_BANK}" >&2
fi

echo "V2 patch-local bank: ${BANK}"
echo "Context-only catalog: ${CONTEXT}"
echo "Reserve catalog: ${RESERVE}"
