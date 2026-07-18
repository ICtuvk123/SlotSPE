#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON:-python}"
DATA_ROOT_DIR="${DATA_ROOT_DIR:-/data0/lfy_data/Pathology/CONCH/kirc/pt_files}"
DATA_PATH="${DATA_PATH:-./dataset_csv}"
RESULTS_DIR="${RESULTS_DIR:-./results_train}"
EVENT_BANK_PATH="${EVENT_BANK_PATH:-assets/event_bank/tcga_kirc/conch_event_bank.pt}"

if [[ ! -f "${EVENT_BANK_PATH}" ]]; then
    echo "ERROR: missing event bank: ${EVENT_BANK_PATH}" >&2
    echo "Run: bash scripts/build_tcga_kirc_event_bank.sh" >&2
    exit 1
fi
if [[ ! -d "${DATA_ROOT_DIR}" ]]; then
    echo "ERROR: missing normalized CONCH patch tensors: ${DATA_ROOT_DIR}" >&2
    echo "Run: bash scripts/run_kirc_conch_preprocess.sh" >&2
    exit 1
fi

"${PYTHON_BIN}" survival.py \
    --data_root_dir "${DATA_ROOT_DIR}" \
    --data_path "${DATA_PATH}" \
    --results_dir "${RESULTS_DIR}" \
    --study kirc \
    --n_classes 4 \
    --num_patches 4096 \
    --encoding_dim 512 \
    --max_epochs "${MAX_EPOCHS:-30}" \
    --batch_size "${BATCH_SIZE:-32}" \
    --seed "${SEED:-3}" \
    --specific_simple "${SPECIFIC_SIMPLE:-conch_event_gated}" \
    --method SlotSPE \
    --rna_format Pathways \
    --label_col survival_months_dss \
    --bag_loss nll_surv \
    --signature combine \
    --slot_num_omics 8 \
    --slot_num_wsi 8 \
    --slot_iters 10 \
    --temperature 0.01 \
    --topk_ratio 0.25 \
    --top_k_method parallel_topk_st \
    --slot_attention_type event_gated \
    --event_bank_path "${EVENT_BANK_PATH}" \
    --slot_feature_encoder conch \
    --reuse_slot_features_as_conch \
    --event_gate_start_iter 1 \
    --event_projection_dim 256 \
    --tau_event 0.10 \
    --patch_event_support_mode calibrated_sigmoid \
    --delta_patch_event "${DELTA_PATCH_EVENT:-0.20}" \
    --beta_patch_event 0.10 \
    --delta_sem 0.20 \
    --beta_sem 0.10 \
    --delta_vis 0.30 \
    --beta_vis 0.10 \
    --lambda_js 1.0 \
    --lambda_event 1.0 \
    "$@"
