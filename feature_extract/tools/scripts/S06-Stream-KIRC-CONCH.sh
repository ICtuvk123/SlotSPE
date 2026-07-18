#!/usr/bin/env bash
set -euo pipefail

# Disk-bounded TCGA-KIRC CONCH preprocessing.
#
# Each manifest entry is handled independently:
#   download one WSI -> tissue segmentation/448px coordinates -> CONCH ->
#   validate [N,512] normalized float16 PT -> delete the raw WSI.
#
# The final PT is intentionally a plain Tensor so the exact same files can be
# used by Original SlotSPE and Event-Gated SlotSPE (with reuse enabled).

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd "${SCRIPT_DIR}/../../.." && pwd)

MANIFEST=${MANIFEST:-"${REPO_DIR}/feature_extract/tools/gdc/gdc_manifest_rcc.txt"}
GDC_CLIENT=${GDC_CLIENT:-"${REPO_DIR}/feature_extract/tools/gdc/gdc-client"}

STUDY=${STUDY:-kirc}
PATHOLOGY_DATA_ROOT=${PATHOLOGY_DATA_ROOT:-"/data0/lfy_data/Pathology"}
MODEL_CKPT=${MODEL_CKPT:-"${PATHOLOGY_DATA_ROOT}/checkpoints/conch/pytorch_model.bin"}
WORK_DIR=${WORK_DIR:-"${PATHOLOGY_DATA_ROOT}/Slides/${STUDY}_conch_work"}
FEAT_DIR=${FEAT_DIR:-"${PATHOLOGY_DATA_ROOT}/CONCH/${STUDY}"}
LOG_DIR=${LOG_DIR:-"${WORK_DIR}/logs"}

# One WSI at a time is required here: the 939-slide manifest describes about
# 873 GiB of raw data and the largest single WSI is about 3.8 GiB.
BATCH_SIZE=${BATCH_SIZE:-1}
START_BATCH=${START_BATCH:-1}
EXTRACT_BATCH_SIZE=${EXTRACT_BATCH_SIZE:-16}
EXTRACT_NUM_WORKERS=${EXTRACT_NUM_WORKERS:-1}
# GPU 6 is the currently reserved/default preprocessing device on this host;
# callers can override it without editing the script.
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-6}
MIN_FREE_GIB=${MIN_FREE_GIB:-25}
DRY_RUN=${DRY_RUN:-0}

die() {
    echo "[error] $*" >&2
    exit 1
}

[[ -f "${MANIFEST}" ]] || die "KIRC GDC manifest not found: ${MANIFEST}"
[[ -f "${MODEL_CKPT}" ]] || die "CONCH checkpoint not found: ${MODEL_CKPT}. Request access at https://huggingface.co/MahmoodLab/conch"
[[ -x "${GDC_CLIENT}" ]] || die "gdc-client is not executable: ${GDC_CLIENT}"
[[ "${BATCH_SIZE}" == "1" ]] || die "BATCH_SIZE must remain 1 on this disk-constrained host"

mkdir -p "${FEAT_DIR}/pt_files" "${LOG_DIR}"
available_kib=$(df --output=avail "${FEAT_DIR}" | tail -n 1 | tr -d ' ')
required_kib=$((MIN_FREE_GIB * 1024 * 1024))
if (( available_kib < required_kib )); then
    available_gib=$((available_kib / 1024 / 1024))
    die "only ${available_gib} GiB free; require at least ${MIN_FREE_GIB} GiB before starting"
fi

echo "[info] KIRC manifest: ${MANIFEST}"
echo "[info] CONCH checkpoint: ${MODEL_CKPT}"
echo "[info] final tensors: ${FEAT_DIR}/pt_files"
echo "[info] temporary work: ${WORK_DIR}"
echo "[info] one WSI per batch; 448x448 at 20x; normalized 512-D float16"

MANIFEST="${MANIFEST}" \
STUDY="${STUDY}" \
ARCH=CONCH \
MODEL_CKPT="${MODEL_CKPT}" \
GDC_CLIENT="${GDC_CLIENT}" \
WORK_DIR="${WORK_DIR}" \
FEAT_DIR="${FEAT_DIR}" \
LOG_DIR="${LOG_DIR}" \
BATCH_SIZE="${BATCH_SIZE}" \
START_BATCH="${START_BATCH}" \
MAG=20 \
SIZE=448 \
TARGET_PATCH_SIZE=448 \
SLIDE_EXT=.svs \
EXTRACT_BATCH_SIZE="${EXTRACT_BATCH_SIZE}" \
EXTRACT_NUM_WORKERS="${EXTRACT_NUM_WORKERS}" \
PROJ_TO_CONTRAST=Y \
SAVE_DTYPE=float16 \
VALIDATE_CONCH=1 \
CONCH_EXPECTED_DIM=512 \
KEEP_RAW=0 \
KEEP_PATCH_COORDS=1 \
KEEP_MASKS=0 \
KEEP_STITCHES=0 \
SAVE_MASK=0 \
STITCH=0 \
ALLOW_FAILED_SLIDES=1 \
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
DRY_RUN="${DRY_RUN}" \
bash "${SCRIPT_DIR}/S05-Batch-GDC-Download-Preprocess.sh"

if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[dry-run] skipping final feature validation"
    exit 0
fi

python3 "${REPO_DIR}/tools/validate_conch_features.py" \
    --feature-dir "${FEAT_DIR}/pt_files" \
    --manifest "${MANIFEST}" \
    --expected-dim 512 \
    --expected-dtype float16 \
    --allow-missing 3 \
    --report "${FEAT_DIR}/validation_report.json"

echo "[info] completed. Do not delete prior features until the report and training smoke tests pass."
