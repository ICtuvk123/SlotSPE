#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"
PYTHON_BIN="${PYTHON:-python}"
RAW="assets/event_bank/tcga_kirc/events_raw.json"
REVIEWED="assets/event_bank/tcga_kirc/events_reviewed.json"
BANK="assets/event_bank/tcga_kirc/conch_event_bank.pt"
CHECKPOINT="${CONCH_CHECKPOINT:-/data0/lfy_data/Pathology/checkpoints/conch/pytorch_model.bin}"

bash scripts/setup_conch.sh

if [[ ! -f "${RAW}" ]]; then
    if ! "${PYTHON_BIN}" tools/generate_event_candidates.py \
        --cancer-type "kidney renal clear cell carcinoma" \
        --dataset "TCGA-KIRC" \
        --num-events 30 \
        --output "${RAW}"; then
        echo "Event generation paused. Save valid JSON at ${RAW}, then rerun this script." >&2
        exit 2
    fi
else
    echo "Using existing candidate file: ${RAW}"
fi

"${PYTHON_BIN}" tools/review_event_candidates.py \
    --input "${RAW}" \
    --output "${REVIEWED}"

if [[ "${CONCH_FROM_HF:-0}" == "1" ]]; then
    "${PYTHON_BIN}" tools/build_conch_event_bank.py \
        --events "${REVIEWED}" --output "${BANK}" --from-hf \
        --device "${CONCH_DEVICE:-cuda}" --batch-size "${CONCH_BATCH_SIZE:-32}"
else
    "${PYTHON_BIN}" tools/build_conch_event_bank.py \
        --events "${REVIEWED}" --output "${BANK}" --checkpoint "${CHECKPOINT}" \
        --device "${CONCH_DEVICE:-cuda}" --batch-size "${CONCH_BATCH_SIZE:-32}"
fi
