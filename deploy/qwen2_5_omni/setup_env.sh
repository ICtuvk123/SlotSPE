#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-qwen-omni}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is required but was not found on PATH." >&2
  exit 1
fi

eval "$(conda shell.bash hook)"

if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  conda create -y -n "${ENV_NAME}" "python=${PYTHON_VERSION}"
fi

conda activate "${ENV_NAME}"

python -m pip install --upgrade pip
python -m pip install torch torchvision torchaudio --index-url "${PYTORCH_INDEX_URL}"
python -m pip uninstall -y transformers || true
python -m pip install "git+https://github.com/huggingface/transformers@v4.51.3-Qwen2.5-Omni-preview"
python -m pip install \
  accelerate \
  "qwen-omni-utils[decord]" \
  soundfile \
  gradio \
  huggingface_hub

conda install -y -c conda-forge ffmpeg

python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("cuda device:", torch.cuda.get_device_name(0))
PY

echo "Environment ready. Activate it with: conda activate ${ENV_NAME}"

