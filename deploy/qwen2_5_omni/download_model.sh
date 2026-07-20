#!/usr/bin/env bash
set -euo pipefail

MODEL_ID="${1:-Qwen/Qwen2.5-Omni-3B}"
TARGET_ROOT="${2:-deploy/qwen2_5_omni/models}"
MODEL_DIR="${TARGET_ROOT}/${MODEL_ID##*/}"

python -m pip show huggingface_hub >/dev/null 2>&1 || {
  echo "huggingface_hub is not installed. Run setup_env.sh first." >&2
  exit 1
}

mkdir -p "${TARGET_ROOT}"

huggingface-cli download "${MODEL_ID}" \
  --local-dir "${MODEL_DIR}" \
  --local-dir-use-symlinks False

echo "Model downloaded to ${MODEL_DIR}"

