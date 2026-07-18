#!/usr/bin/env bash
set -uo pipefail

SOURCE_DIR="/data0/lfy_data/Pathology/CONCH/kirc"
REMOTE_DIR="SlotSPE-transfer/kirc_conch_v2"
BYPY_BIN="/home/liufangyi/anaconda3/bin/bypy"
LOG_FILE="/home/liufangyi/proj/SLotSPE/SlotSPE/bypy_kirc_upload.log"

unset http_proxy https_proxy all_proxy no_proxy
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY NO_PROXY

{
    echo "[$(date '+%F %T')] tmux upload started"
    echo "source: ${SOURCE_DIR}"
    echo "remote: /apps/bypy/${REMOTE_DIR}"
    if env | grep -qi proxy; then
        echo "ERROR: proxy variables remain in tmux environment"
        env | grep -i proxy
        exit 2
    fi
    echo "proxy check: clear"
    "${BYPY_BIN}" upload "${SOURCE_DIR}" "${REMOTE_DIR}"
    status=$?
    echo "[$(date '+%F %T')] bypy exit code: ${status}"
    exit "${status}"
} 2>&1 | tee -a "${LOG_FILE}"
