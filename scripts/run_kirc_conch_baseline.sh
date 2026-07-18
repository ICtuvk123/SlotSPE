#!/usr/bin/env bash
set -euo pipefail

# Fair baseline for Event-Gated SlotSPE: exact same normalized CONCH PT files,
# splits, survival head, and training hyperparameters, with event gating off.

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON:-python}"
DATA_ROOT_DIR="${DATA_ROOT_DIR:-/data0/lfy_data/Pathology/CONCH/kirc/pt_files}"
DATA_PATH="${DATA_PATH:-./dataset_csv}"
RESULTS_DIR="${RESULTS_DIR:-./results_train}"

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
    --specific_simple "${SPECIFIC_SIMPLE:-conch_original}" \
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
    --slot_attention_type original \
    --slot_feature_encoder conch \
    "$@"
