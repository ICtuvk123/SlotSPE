#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${1:-deploy/qwen2_5_omni/models/Qwen2.5-Omni-3B}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-7860}"

if [ "$#" -gt 0 ]; then
  shift
fi

python deploy/qwen2_5_omni/web_app.py \
  --model-path "${MODEL_PATH}" \
  --host "${HOST}" \
  --port "${PORT}" \
  "$@"
