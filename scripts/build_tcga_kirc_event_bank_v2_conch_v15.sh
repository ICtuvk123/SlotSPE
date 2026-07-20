#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON:-python}"
EVENTS="${EVENTS:-assets/event_bank/tcga_kirc/v2/events_patch_core.json}"
OUTPUT="${OUTPUT:-assets/event_bank/tcga_kirc/v2/titan_text_event_bank_v2.pt}"
TITAN_MODEL_DIR="${TITAN_MODEL_DIR:-/data0/lfy_data/Pathology/checkpoints/TITAN}"

if [[ ! -d "${TITAN_MODEL_DIR}" ]]; then
    echo "ERROR: local TITAN model directory is missing: ${TITAN_MODEL_DIR}" >&2
    echo "Request access at https://huggingface.co/MahmoodLab/TITAN and download the repository first." >&2
    exit 1
fi
for required_file in config.json modeling_titan.py model.safetensors text_transformer.py conch_tokenizer.py tokenizer.json; do
    if [[ ! -f "${TITAN_MODEL_DIR}/${required_file}" ]]; then
        echo "ERROR: incomplete TITAN download; missing ${TITAN_MODEL_DIR}/${required_file}" >&2
        echo "The current Hugging Face account must first be approved at https://huggingface.co/MahmoodLab/TITAN" >&2
        exit 1
    fi
done

"${PYTHON_BIN}" tools/build_titan_event_bank.py \
    --events "${EVENTS}" \
    --output "${OUTPUT}" \
    --model "${TITAN_MODEL_DIR}" \
    --device "${TITAN_DEVICE:-cuda}" \
    --batch-size "${TITAN_BATCH_SIZE:-32}" \
    --expected-dim 768 \
    --source-patch-size 512 \
    --encoder-input-size 448

echo "TITAN text v2 event bank: ${OUTPUT}"
