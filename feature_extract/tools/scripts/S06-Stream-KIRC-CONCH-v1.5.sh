#!/usr/bin/env bash
set -euo pipefail

# Disk-bounded TCGA-KIRC CONCH v1.5 preprocessing.
#
# One WSI at a time:
#   download -> 512px non-overlapping coordinates at 20x -> resize to the
#   encoder's 448px input -> normalized 768-D CONCH v1.5 tensor -> validate
#   -> delete the raw WSI.

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd "${SCRIPT_DIR}/../../.." && pwd)

MANIFEST=${MANIFEST:-"${REPO_DIR}/feature_extract/tools/gdc/gdc_manifest_kirc_slotspe_494.txt"}
GDC_CLIENT=${GDC_CLIENT:-"${REPO_DIR}/feature_extract/tools/gdc/gdc-client"}

STUDY=${STUDY:-kirc}
PATHOLOGY_DATA_ROOT=${PATHOLOGY_DATA_ROOT:-"/data0/lfy_data/Pathology"}
TITAN_MODEL_DIR=${TITAN_MODEL_DIR:-"${PATHOLOGY_DATA_ROOT}/checkpoints/TITAN"}
WORK_DIR=${WORK_DIR:-"${PATHOLOGY_DATA_ROOT}/Slides/${STUDY}_conch_v15_work"}
FEAT_DIR=${FEAT_DIR:-"${PATHOLOGY_DATA_ROOT}/CONCH_v1.5/${STUDY}"}
LOG_DIR=${LOG_DIR:-"${WORK_DIR}/logs"}

BATCH_SIZE=${BATCH_SIZE:-1}
START_BATCH=${START_BATCH:-1}
END_BATCH=${END_BATCH:-0}
EXTRACT_BATCH_SIZE=${EXTRACT_BATCH_SIZE:-16}
EXTRACT_NUM_WORKERS=${EXTRACT_NUM_WORKERS:-1}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-6}
GPU_WAIT_CANDIDATES=${GPU_WAIT_CANDIDATES:-"0,1,2,3,4"}
GPU_MIN_FREE_MIB=${GPU_MIN_FREE_MIB:-14000}
GPU_POLL_SECONDS=${GPU_POLL_SECONDS:-60}
MIN_FREE_GIB=${MIN_FREE_GIB:-25}
ALLOW_MISSING=${ALLOW_MISSING:-0}
DRY_RUN=${DRY_RUN:-0}
GDC_DIRECT=${GDC_DIRECT:-1}
GDC_DOWNLOAD_RETRIES=${GDC_DOWNLOAD_RETRIES:-5}
GDC_RETRY_SECONDS=${GDC_RETRY_SECONDS:-30}

die() {
    echo "[error] $*" >&2
    exit 1
}

[[ -f "${MANIFEST}" ]] || die "KIRC GDC manifest not found: ${MANIFEST}"
[[ -d "${TITAN_MODEL_DIR}" ]] || die "TITAN model directory not found: ${TITAN_MODEL_DIR}. Request access at https://huggingface.co/MahmoodLab/TITAN"
for required_file in config.json modeling_titan.py model.safetensors conch_v1_5.py conch_v1_5_pytorch_model.bin; do
    [[ -f "${TITAN_MODEL_DIR}/${required_file}" ]] || die "incomplete TITAN download; missing ${TITAN_MODEL_DIR}/${required_file}. Obtain gated access and download the full repository first"
done
[[ -x "${GDC_CLIENT}" ]] || die "gdc-client is not executable: ${GDC_CLIENT}"
[[ "${BATCH_SIZE}" == "1" ]] || die "BATCH_SIZE must remain 1 on this disk-constrained host"
if [[ "${DRY_RUN}" != "1" ]]; then
    python3 -c 'import transformers; assert int(transformers.__version__.split(".", 1)[0]) < 5, transformers.__version__' \
        >/dev/null 2>&1 || die "TITAN requires transformers<5 (official: 4.46.0); use a separate preprocessing environment"
fi

mkdir -p "${FEAT_DIR}/pt_files" "${LOG_DIR}"
available_kib=$(df --output=avail "${FEAT_DIR}" | tail -n 1 | tr -d ' ')
required_kib=$((MIN_FREE_GIB * 1024 * 1024))
if (( available_kib < required_kib )); then
    available_gib=$((available_kib / 1024 / 1024))
    die "only ${available_gib} GiB free; require at least ${MIN_FREE_GIB} GiB before starting"
fi

echo "[info] KIRC manifest: ${MANIFEST}"
echo "[info] TITAN/CONCH v1.5 model: ${TITAN_MODEL_DIR}"
echo "[info] final tensors: ${FEAT_DIR}/pt_files"
echo "[info] one WSI per batch; 512x512 field at 20x -> 448 input; normalized 768-D float16"
echo "[info] GPU wait candidates: ${GPU_WAIT_CANDIDATES:-disabled}"

MANIFEST="${MANIFEST}" \
STUDY="${STUDY}" \
ARCH=CONCH_v1.5 \
MODEL_CKPT="${TITAN_MODEL_DIR}" \
GDC_CLIENT="${GDC_CLIENT}" \
WORK_DIR="${WORK_DIR}" \
FEAT_DIR="${FEAT_DIR}" \
LOG_DIR="${LOG_DIR}" \
BATCH_SIZE="${BATCH_SIZE}" \
START_BATCH="${START_BATCH}" \
END_BATCH="${END_BATCH}" \
MAG=20 \
SIZE=512 \
TARGET_PATCH_SIZE=448 \
SLIDE_EXT=.svs \
EXTRACT_BATCH_SIZE="${EXTRACT_BATCH_SIZE}" \
EXTRACT_NUM_WORKERS="${EXTRACT_NUM_WORKERS}" \
PROJ_TO_CONTRAST=Y \
SAVE_DTYPE=float16 \
VALIDATE_CONCH=1 \
CONCH_EXPECTED_DIM=768 \
KEEP_RAW=0 \
KEEP_PATCH_COORDS=1 \
KEEP_MASKS=0 \
KEEP_STITCHES=0 \
SAVE_MASK=0 \
STITCH=0 \
ALLOW_FAILED_SLIDES=1 \
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
GPU_WAIT_CANDIDATES="${GPU_WAIT_CANDIDATES}" \
GPU_MIN_FREE_MIB="${GPU_MIN_FREE_MIB}" \
GPU_POLL_SECONDS="${GPU_POLL_SECONDS}" \
DRY_RUN="${DRY_RUN}" \
GDC_DIRECT="${GDC_DIRECT}" \
GDC_DOWNLOAD_RETRIES="${GDC_DOWNLOAD_RETRIES}" \
GDC_RETRY_SECONDS="${GDC_RETRY_SECONDS}" \
bash "${SCRIPT_DIR}/S05-Batch-GDC-Download-Preprocess.sh"

if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[dry-run] skipping final feature validation"
    exit 0
fi

if [[ "${END_BATCH}" == "0" ]]; then
    python3 "${REPO_DIR}/tools/validate_conch_features.py" \
        --feature-dir "${FEAT_DIR}/pt_files" \
        --manifest "${MANIFEST}" \
        --expected-dim 768 \
        --expected-dtype float16 \
        --allow-missing "${ALLOW_MISSING}" \
        --report "${FEAT_DIR}/validation_report.json"
else
    echo "[info] partial run through batch ${END_BATCH}; skipping full-manifest validation"
fi

echo "[info] completed. Keep the old CONCH v1 features until the paired comparison finishes."
