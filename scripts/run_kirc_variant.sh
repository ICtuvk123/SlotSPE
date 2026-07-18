#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 11 ]; then
  echo "Usage: $0 GPU TAG SEED LR MAX_EPOCHS BATCH_SIZE SLOT_WSI SLOT_OMICS SLOT_ITERS TOPK LAMBDA_RECON" >&2
  exit 2
fi

GPU="$1"
TAG="$2"
SEED="$3"
LR="$4"
MAX_EPOCHS="$5"
BATCH_SIZE="$6"
SLOT_WSI="$7"
SLOT_OMICS="$8"
SLOT_ITERS="$9"
TOPK="${10}"
LAMBDA_RECON="${11}"

cd /home/liufangyi/proj/SLotSPE/SlotSPE

mkdir -p ./results_train /tmp/mpl_slotspe
export MPLCONFIGDIR=/tmp/mpl_slotspe

PYTHON_BIN="${PYTHON_BIN:-/home/liufangyi/anaconda3/envs/slotspe/bin/python}"

"${PYTHON_BIN}" survival.py \
  --gpu "${GPU}" \
  --data_root_dir /home/liufangyi/Data/Pathology/UNI/kirc/pt_files \
  --results_dir ./results_train \
  --data_path ./dataset_csv \
  --study kirc \
  --k_start 0 \
  --k_end 5 \
  --n_classes 4 \
  --num_patches 4096 \
  --encoding_dim 1024 \
  --max_epochs "${MAX_EPOCHS}" \
  --batch_size "${BATCH_SIZE}" \
  --seed "${SEED}" \
  --lr "${LR}" \
  --specific_simple "${TAG}" \
  --method SlotSPE \
  --rna_format Pathways \
  --label_col survival_months_dss \
  --bag_loss nll_surv \
  --signature combine \
  --slot_num_omics "${SLOT_OMICS}" \
  --slot_num_wsi "${SLOT_WSI}" \
  --slot_iters "${SLOT_ITERS}" \
  --temperature 0.01 \
  --topk_ratio "${TOPK}" \
  --lambda_recon_loss "${LAMBDA_RECON}" \
  --top_k_method parallel_topk_st
