#!/usr/bin/env bash
set -uo pipefail

SOURCE_DIR="/data0/lfy_data/Pathology/CONCH/kirc/pt_files"
REMOTE_DIR="SlotSPE-transfer/kirc_conch_v2/pt_files"
BYPY_BIN="/home/liufangyi/anaconda3/bin/bypy"
LOG_FILE="/home/liufangyi/proj/SLotSPE/SlotSPE/bypy_kirc_missing_upload.log"
FILES=(
  "TCGA-T7-A92I-01Z-00-DX1.3B036C1D-F8A7-475F-9830-C0972AD3889F.pt"
  "TCGA-T7-A92I-01Z-00-DX2.87A6A24E-42E3-4107-AE91-369C0D168556.pt"
  "TCGA-T7-A92I-01Z-00-DX3.FC717D92-13BD-4968-8E82-0CC53EB6D5D1.pt"
)

unset http_proxy https_proxy all_proxy no_proxy
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY NO_PROXY

{
  echo "[$(date '+%F %T')] targeted upload started"
  if env | grep -qi proxy; then
    echo "ERROR: proxy variables remain in tmux environment"
    exit 2
  fi
  echo "proxy check: clear"
  for file in "${FILES[@]}"; do
    echo "uploading: ${file}"
    "${BYPY_BIN}" -s 64MB -t 7200 -r 5 upload \
      "${SOURCE_DIR}/${file}" "${REMOTE_DIR}/${file}" overwrite || exit $?
  done
  echo "[$(date '+%F %T')] targeted upload complete"
} 2>&1 | tee -a "${LOG_FILE}"
