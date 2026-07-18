#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONCH_DIR="${ROOT_DIR}/third_party/CONCH"
PYTHON_BIN="${PYTHON:-python}"

"${PYTHON_BIN}" - <<'PY'
import sys
if sys.version_info < (3, 10):
    raise SystemExit(f"Python >=3.10 is required; found {sys.version.split()[0]}")
print("Python:", sys.version.split()[0])
try:
    import torch
except ImportError as exc:
    raise SystemExit("PyTorch must be installed before CONCH") from exc
print("torch:", torch.__version__)
print("torch CUDA build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
PY

if "${PYTHON_BIN}" -c 'from conch.open_clip_custom import create_model_from_pretrained, get_tokenizer, tokenize' 2>/dev/null; then
    echo "CONCH is already importable."
else
    if [[ -e "${CONCH_DIR}" && ! -d "${CONCH_DIR}/.git" ]]; then
        echo "ERROR: ${CONCH_DIR} exists but is not a CONCH git checkout." >&2
        exit 1
    fi
    if [[ ! -d "${CONCH_DIR}/.git" ]]; then
        mkdir -p "${ROOT_DIR}/third_party"
        git clone https://github.com/mahmoodlab/CONCH.git "${CONCH_DIR}"
    else
        echo "Using existing checkout: ${CONCH_DIR}"
    fi
    "${PYTHON_BIN}" -m pip install -e "${CONCH_DIR}"
fi

"${PYTHON_BIN}" - <<'PY'
from conch.open_clip_custom import create_model_from_pretrained, get_tokenizer, tokenize
print("CONCH import smoke test passed")
PY
