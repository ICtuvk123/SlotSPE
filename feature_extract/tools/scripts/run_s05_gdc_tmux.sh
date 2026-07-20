#!/usr/bin/env bash
set -euo pipefail

SESSION_NAME=${SESSION_NAME:-gdc_kirc}
STUDY=${STUDY:-kirc}
ARCH=${ARCH:-UNI}
CONDA_ENV=${CONDA_ENV:-slotspe}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
BATCH_SIZE=${BATCH_SIZE:-50}
START_BATCH=${START_BATCH:-1}
ALLOW_FAILED_SLIDES=${ALLOW_FAILED_SLIDES:-0}
MAG=${MAG:-20}
SIZE=${SIZE:-256}
TARGET_PATCH_SIZE=${TARGET_PATCH_SIZE:-${SIZE}}
EXTRACT_BATCH_SIZE=${EXTRACT_BATCH_SIZE:-64}
EXTRACT_NUM_WORKERS=${EXTRACT_NUM_WORKERS:-1}
SAVE_DTYPE=${SAVE_DTYPE:-float32}
KEEP_RAW=${KEEP_RAW:-0}
KEEP_PATCH_COORDS=${KEEP_PATCH_COORDS:-0}
KEEP_MASKS=${KEEP_MASKS:-0}
KEEP_STITCHES=${KEEP_STITCHES:-0}
SAVE_MASK=${SAVE_MASK:-1}
STITCH=${STITCH:-0}
GPU_WAIT_CANDIDATES=${GPU_WAIT_CANDIDATES:-""}
GPU_MIN_FREE_MIB=${GPU_MIN_FREE_MIB:-14000}
GPU_POLL_SECONDS=${GPU_POLL_SECONDS:-60}

REPO_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
S05_SCRIPT="${REPO_DIR}/feature_extract/tools/scripts/S05-Batch-GDC-Download-Preprocess.sh"
GDC_CLIENT=${GDC_CLIENT:-"${REPO_DIR}/feature_extract/tools/gdc/gdc-client"}
WORK_DIR=${WORK_DIR:-"${HOME}/Data/Pathology/Slides/${STUDY}_batch_work"}
FEAT_DIR=${FEAT_DIR:-"${HOME}/Data/Pathology/${ARCH}/${STUDY}"}
LOG_DIR=${LOG_DIR:-"${WORK_DIR}/logs"}
RUN_LOG=${RUN_LOG:-"${LOG_DIR}/run.log"}
TMUX_RUN_SCRIPT="${LOG_DIR}/${SESSION_NAME}.run.sh"

die() {
    echo "[error] $*" >&2
    exit 1
}

[[ -n "${MANIFEST:-}" ]] || die "set MANIFEST=/path/to/gdc_manifest.txt"
[[ -n "${MODEL_CKPT:-}" ]] || die "set MODEL_CKPT=/path/to/pytorch_model.bin"
[[ -f "${MANIFEST}" ]] || die "manifest not found: ${MANIFEST}"
[[ -e "${MODEL_CKPT}" ]] || die "model checkpoint not found: ${MODEL_CKPT}"
[[ -x "${GDC_CLIENT}" ]] || die "gdc-client not executable: ${GDC_CLIENT}"
[[ -f "${S05_SCRIPT}" ]] || die "S05 script not found: ${S05_SCRIPT}"

if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
    die "tmux session already exists: ${SESSION_NAME}"
fi

mkdir -p "${LOG_DIR}" "${FEAT_DIR}/pt_files"

cat > "${TMUX_RUN_SCRIPT}" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "${REPO_DIR}"
source "${HOME}/anaconda3/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export PYTHONPATH="${REPO_DIR}/feature_extract/tools/CLAM/_vendor:${REPO_DIR}/feature_extract/tools/CLAM\${PYTHONPATH:+:\${PYTHONPATH}}"
echo "[info] started at \$(date)"
echo "[info] session: ${SESSION_NAME}"
echo "[info] conda env: ${CONDA_ENV}"
echo "[info] log: ${RUN_LOG}"
GDC_CLIENT="${GDC_CLIENT}" \\
MANIFEST="${MANIFEST}" \\
STUDY="${STUDY}" \\
MODEL_CKPT="${MODEL_CKPT}" \\
ARCH="${ARCH}" \\
WORK_DIR="${WORK_DIR}" \\
FEAT_DIR="${FEAT_DIR}" \\
LOG_DIR="${LOG_DIR}" \\
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \\
BATCH_SIZE="${BATCH_SIZE}" \\
START_BATCH="${START_BATCH}" \\
ALLOW_FAILED_SLIDES="${ALLOW_FAILED_SLIDES}" \\
MAG="${MAG}" \\
SIZE="${SIZE}" \\
TARGET_PATCH_SIZE="${TARGET_PATCH_SIZE}" \\
EXTRACT_BATCH_SIZE="${EXTRACT_BATCH_SIZE}" \\
EXTRACT_NUM_WORKERS="${EXTRACT_NUM_WORKERS}" \\
SAVE_DTYPE="${SAVE_DTYPE}" \\
KEEP_RAW="${KEEP_RAW}" \\
KEEP_PATCH_COORDS="${KEEP_PATCH_COORDS}" \\
KEEP_MASKS="${KEEP_MASKS}" \\
KEEP_STITCHES="${KEEP_STITCHES}" \\
SAVE_MASK="${SAVE_MASK}" \\
STITCH="${STITCH}" \\
GPU_WAIT_CANDIDATES="${GPU_WAIT_CANDIDATES}" \\
GPU_MIN_FREE_MIB="${GPU_MIN_FREE_MIB}" \\
GPU_POLL_SECONDS="${GPU_POLL_SECONDS}" \\
bash "${S05_SCRIPT}" 2>&1 | tee -a "${RUN_LOG}"
echo "[info] finished at \$(date)"
EOF

chmod +x "${TMUX_RUN_SCRIPT}"

tmux new-session -d -s "${SESSION_NAME}" "bash ${TMUX_RUN_SCRIPT}"

echo "[info] started tmux session: ${SESSION_NAME}"
echo "[info] attach: tmux attach -t ${SESSION_NAME}"
echo "[info] log: ${RUN_LOG}"
echo "[info] runner: ${TMUX_RUN_SCRIPT}"
