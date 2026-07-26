#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
DATA_REPO=${DATA_REPO:-/home/dzdx/19G242/tianic/SlotSPE}
EXPERIMENT_ROOT=${EXPERIMENT_ROOT:-/home/dzdx/19G242/tianic/SlotSPE_experiments}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-gc_rsm_conch_qv_online_fold0}

K_START=${K_START:-0}
K_END=${K_END:-1}
MAX_EPOCHS=${MAX_EPOCHS:-10}
NUM_PATCHES=${NUM_PATCHES:-32}
PATCH_BATCH_SIZE=${PATCH_BATCH_SIZE:-8}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-8}
CONCH_LORA_LR=${CONCH_LORA_LR:-0.0001}
SLOTSPE_LR=${SLOTSPE_LR:-0.0005}
SEED=${SEED:-3}

RESULTS_ROOT="${EXPERIMENT_ROOT}/results/${EXPERIMENT_NAME}"
LOG_DIR="${EXPERIMENT_ROOT}/logs"
LOG_FILE="${LOG_DIR}/${EXPERIMENT_NAME}.log"
mkdir -p "${RESULTS_ROOT}" "${LOG_DIR}"

echo "[$(date -Is)] START ${EXPERIMENT_NAME}" | tee -a "${LOG_FILE}"
echo "[info] branch=$(git -C "${ROOT_DIR}" branch --show-current)" | tee -a "${LOG_FILE}"
echo "[info] commit=$(git -C "${ROOT_DIR}" rev-parse HEAD)" | tee -a "${LOG_FILE}"

python -u "${ROOT_DIR}/survival.py" \
    --data_root_dir "${DATA_REPO}/Pathology/CONCH_v1.5/kirc/pt_files" \
    --results_dir "${RESULTS_ROOT}" \
    --data_path "${DATA_REPO}/dataset_csv" \
    --study kirc \
    --k_start "${K_START}" \
    --k_end "${K_END}" \
    --n_classes 4 \
    --num_patches "${NUM_PATCHES}" \
    --encoding_dim 768 \
    --max_epochs "${MAX_EPOCHS}" \
    --batch_size 1 \
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
    --gradient_clip_norm 1.0 \
    --lr "${SLOTSPE_LR}" \
    --conch_lora_lr "${CONCH_LORA_LR}" \
    --seed "${SEED}" \
    --eval_slot_seed 100000 \
    --specific_simple "${EXPERIMENT_NAME}" \
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
    --raw_wsi_dir "${DATA_REPO}/Pathology/RawWSI/kirc_gdc" \
    --patch_coords_dir "${DATA_REPO}/Pathology/Slides/kirc_conch_v15_work" \
    --online_conch_model_dir "${DATA_REPO}/checkpoints/TITAN" \
    --online_target_patch_size 448 \
    --online_patch_batch_size "${PATCH_BATCH_SIZE}" \
    --conch_qv_lora_mode gene \
    --conch_qv_lora_layers 2 \
    --conch_qv_lora_rank 8 \
    --conch_gene_hidden_dim 128 \
    --conch_gradient_checkpointing \
    2>&1 | tee -a "${LOG_FILE}"

echo "[$(date -Is)] DONE ${EXPERIMENT_NAME}" | tee -a "${LOG_FILE}"
