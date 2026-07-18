#!/usr/bin/env bash
set -euo pipefail

cd /home/liufangyi/proj/SLotSPE/SlotSPE

mkdir -p ./results_train /tmp/mpl_slotspe
export MPLCONFIGDIR=/tmp/mpl_slotspe

python survival.py \
  --data_root_dir /home/liufangyi/Data/Pathology/UNI/kirc/pt_files \
  --results_dir ./results_train \
  --data_path ./dataset_csv \
  --study kirc \
  --k_start 0 \
  --k_end 5 \
  --n_classes 4 \
  --num_patches 4096 \
  --encoding_dim 1024 \
  --max_epochs 30 \
  --batch_size 32 \
  --seed 3 \
  --specific_simple local_full_20260702 \
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
  --top_k_method parallel_topk_st
