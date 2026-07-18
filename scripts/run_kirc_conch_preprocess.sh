#!/usr/bin/env bash
set -euo pipefail

# Launch the disk-bounded KIRC CONCH pipeline in tmux.

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PIPELINE="${ROOT_DIR}/feature_extract/tools/scripts/S06-Stream-KIRC-CONCH.sh"

SESSION_NAME=${SESSION_NAME:-kirc_conch}
CONDA_ENV=${CONDA_ENV:-slotspe}
PATHOLOGY_DATA_ROOT=${PATHOLOGY_DATA_ROOT:-"/data0/lfy_data/Pathology"}
MODEL_CKPT=${MODEL_CKPT:-"${PATHOLOGY_DATA_ROOT}/checkpoints/conch/pytorch_model.bin"}
MANIFEST=${MANIFEST:-"${ROOT_DIR}/feature_extract/tools/gdc/gdc_manifest_rcc.txt"}
WORK_DIR=${WORK_DIR:-"${PATHOLOGY_DATA_ROOT}/Slides/kirc_conch_work"}
FEAT_DIR=${FEAT_DIR:-"${PATHOLOGY_DATA_ROOT}/CONCH/kirc"}
LOG_DIR=${LOG_DIR:-"${WORK_DIR}/logs"}
RUN_LOG=${RUN_LOG:-"${LOG_DIR}/run.log"}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-6}
EXTRACT_BATCH_SIZE=${EXTRACT_BATCH_SIZE:-16}
EXTRACT_NUM_WORKERS=${EXTRACT_NUM_WORKERS:-1}
MIN_FREE_GIB=${MIN_FREE_GIB:-25}

die() {
    echo "[error] $*" >&2
    exit 1
}

[[ -f "${PIPELINE}" ]] || die "pipeline not found: ${PIPELINE}"
[[ -f "${MODEL_CKPT}" ]] || die "CONCH checkpoint not found: ${MODEL_CKPT}"
[[ -f "${MANIFEST}" ]] || die "manifest not found: ${MANIFEST}"
python -c 'import cv2, h5py, openslide; from conch.open_clip_custom import create_model_from_pretrained' \
    >/dev/null 2>&1 || die "missing preprocessing dependencies: cv2, h5py, openslide, or conch"
command -v tmux >/dev/null 2>&1 || die "tmux is not installed"
if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
    die "tmux session already exists: ${SESSION_NAME}"
fi

mkdir -p "${LOG_DIR}"
RUNNER="${LOG_DIR}/${SESSION_NAME}.run.sh"
cat > "${RUNNER}" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "${ROOT_DIR}"
source "${HOME}/anaconda3/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export PYTHONPATH="${ROOT_DIR}/feature_extract/tools/CLAM/_vendor:${ROOT_DIR}/feature_extract/tools/CLAM\${PYTHONPATH:+:\${PYTHONPATH}}"
MODEL_CKPT="${MODEL_CKPT}" \\
MANIFEST="${MANIFEST}" \\
WORK_DIR="${WORK_DIR}" \\
FEAT_DIR="${FEAT_DIR}" \\
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \\
EXTRACT_BATCH_SIZE="${EXTRACT_BATCH_SIZE}" \\
EXTRACT_NUM_WORKERS="${EXTRACT_NUM_WORKERS}" \\
MIN_FREE_GIB="${MIN_FREE_GIB}" \\
bash "${PIPELINE}" 2>&1 | tee -a "${RUN_LOG}"
EOF
chmod +x "${RUNNER}"

tmux new-session -d -s "${SESSION_NAME}" "bash ${RUNNER}"
echo "[info] started tmux session: ${SESSION_NAME}"
echo "[info] attach: tmux attach -t ${SESSION_NAME}"
echo "[info] log: ${RUN_LOG}"
