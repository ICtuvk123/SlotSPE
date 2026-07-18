#!/bin/bash
set -euo pipefail

# Batch TCGA WSI processing:
# 1. split a GDC manifest into batches
# 2. download one batch of raw .svs files
# 3. run CLAM S03 segmentation/patch-coordinate generation
# 4. run CLAM S04 feature extraction
# 5. keep final pt_files only, then remove raw slides and intermediate files
#
# Usage:
#   MANIFEST=/path/to/gdc_manifest.txt \
#   STUDY=kirc \
#   MODEL_CKPT=/path/to/uni_or_conch_checkpoint.bin \
#   ARCH=UNI \
#   bash S05-Batch-GDC-Download-Preprocess.sh

#######################################
# Required user settings
#######################################

# GDC manifest downloaded from the GDC data portal.
MANIFEST=${MANIFEST:-""}

# Study name used only for output directory naming.
STUDY=${STUDY:-"kirc"}

# Feature extractor. Keep this aligned with --encoding_dim used in training.
ARCH=${ARCH:-"UNI"}
MODEL_CKPT=${MODEL_CKPT:-""}

#######################################
# Path settings
#######################################

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CLAM_DIR=${CLAM_DIR:-"${SCRIPT_DIR}/../CLAM"}

# Working area for temporary raw slides and patch coordinates.
WORK_DIR=${WORK_DIR:-"/Data/Pathology/Slides/${STUDY}_batch_work"}

# Final feature directory. Training should point to:
#   ${FEAT_DIR}/pt_files
FEAT_DIR=${FEAT_DIR:-"/Data/Pathology/${ARCH}/${STUDY}"}

# Optional persistent log/status directory.
LOG_DIR=${LOG_DIR:-"${WORK_DIR}/logs"}

#######################################
# Batch and preprocessing settings
#######################################

BATCH_SIZE=${BATCH_SIZE:-50}
START_BATCH=${START_BATCH:-1}

MAG=${MAG:-20}
SIZE=${SIZE:-256}
TARGET_PATCH_SIZE=${TARGET_PATCH_SIZE:-${SIZE}}
SLIDE_EXT=${SLIDE_EXT:-".svs"}

# Start conservative. Increase only after confirming GPU memory is enough.
EXTRACT_BATCH_SIZE=${EXTRACT_BATCH_SIZE:-64}
EXTRACT_NUM_WORKERS=${EXTRACT_NUM_WORKERS:-1}
PROJ_TO_CONTRAST=${PROJ_TO_CONTRAST:-"N"}
SAVE_DTYPE=${SAVE_DTYPE:-"float32"}

# Validate every CONCH batch before deleting its source WSI. Projected public
# CONCH features have dimension 512 and must be normalized for text matching.
VALIDATE_CONCH=${VALIDATE_CONCH:-1}
CONCH_EXPECTED_DIM=${CONCH_EXPECTED_DIM:-512}
VALIDATOR=${VALIDATOR:-"${SCRIPT_DIR}/../../../tools/validate_conch_features.py"}

# S03 options.
SAVE_MASK=${SAVE_MASK:-1}
STITCH=${STITCH:-0}

# Deletion policy after a batch succeeds.
KEEP_RAW=${KEEP_RAW:-0}
KEEP_PATCH_COORDS=${KEEP_PATCH_COORDS:-0}
KEEP_MASKS=${KEEP_MASKS:-0}
KEEP_STITCHES=${KEEP_STITCHES:-0}

# If set to 1, slides explicitly marked failed by S03 are logged and do not
# stop later batches. Keep 0 for strict all-slides-must-succeed runs.
ALLOW_FAILED_SLIDES=${ALLOW_FAILED_SLIDES:-0}

# Set DRY_RUN=1 to print commands without running them.
DRY_RUN=${DRY_RUN:-0}

GDC_CLIENT=${GDC_CLIENT:-"gdc-client"}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

#######################################
# Helpers
#######################################

die() {
    echo "[error] $*" >&2
    exit 1
}

run_cmd() {
    echo "[cmd] $*"
    if [[ "${DRY_RUN}" != "1" ]]; then
        "$@"
    fi
}

require_file() {
    [[ -f "$1" ]] || die "file not found: $1"
}

require_dir() {
    [[ -d "$1" ]] || die "directory not found: $1"
}

safe_rm_rf() {
    local path="$1"
    [[ -n "${path}" ]] || die "refusing to remove an empty path"
    [[ "${path}" == "${WORK_DIR}"/* ]] || die "refusing to remove path outside WORK_DIR: ${path}"
    if [[ -e "${path}" ]]; then
        run_cmd rm -rf "${path}"
    fi
}

count_expected_pt() {
    local batch_manifest="$1"
    local count=0
    local filename slide_id pt_path

    while IFS=$'\t' read -r _id filename _rest; do
        [[ -n "${filename:-}" ]] || continue
        [[ "${filename}" == *"${SLIDE_EXT}" ]] || continue
        slide_id="${filename%"${SLIDE_EXT}"}"
        pt_path="${FEAT_DIR}/pt_files/${slide_id}.pt"
        if [[ -f "${pt_path}" ]]; then
            count=$((count + 1))
        fi
    done < <(tail -n +2 "${batch_manifest}")

    echo "${count}"
}

count_manifest_slides() {
    local batch_manifest="$1"
    local count=0
    local filename

    while IFS=$'\t' read -r _id filename _rest; do
        [[ -n "${filename:-}" ]] || continue
        [[ "${filename}" == *"${SLIDE_EXT}" ]] || continue
        count=$((count + 1))
    done < <(tail -n +2 "${batch_manifest}")

    echo "${count}"
}

count_raw_slides() {
    local batch_manifest="$1"
    local raw_dir="$2"
    local count=0
    local filename

    while IFS=$'\t' read -r _id filename _rest; do
        [[ -n "${filename:-}" ]] || continue
        [[ "${filename}" == *"${SLIDE_EXT}" ]] || continue
        if find "${raw_dir}" -type f -name "${filename}" -print -quit | grep -q .; then
            count=$((count + 1))
        fi
    done < <(tail -n +2 "${batch_manifest}")

    echo "${count}"
}

count_failed_patch_slides() {
    local process_csv="$1"
    if [[ ! -f "${process_csv}" ]]; then
        echo 0
        return
    fi

    awk -F',' 'NR > 1 && $3 ~ /^failed/ { count++ } END { print count + 0 }' "${process_csv}"
}

append_failed_patch_slides() {
    local batch_name="$1"
    local process_csv="$2"
    local failed_file="$3"

    [[ -f "${process_csv}" ]] || return
    awk -F',' -v batch="${batch_name}" 'NR > 1 && $3 ~ /^failed/ {
        print batch "\t" $3 "\t" $1
    }' "${process_csv}" >> "${failed_file}"
}

#######################################
# Validation
#######################################

[[ -n "${MANIFEST}" ]] || die "set MANIFEST=/path/to/gdc_manifest.txt"
require_file "${MANIFEST}"
require_dir "${CLAM_DIR}"
[[ -n "${MODEL_CKPT}" ]] || die "set MODEL_CKPT=/path/to/model/checkpoint"

if [[ "${DRY_RUN}" != "1" ]]; then
    command -v "${GDC_CLIENT}" >/dev/null 2>&1 || die "gdc-client not found; set GDC_CLIENT=/path/to/gdc-client"
fi

mkdir -p "${WORK_DIR}" "${LOG_DIR}" "${FEAT_DIR}/pt_files"

SPLIT_DIR="${WORK_DIR}/manifest_batches"
mkdir -p "${SPLIT_DIR}"

HEADER_FILE="${SPLIT_DIR}/manifest_header.tsv"
head -n 1 "${MANIFEST}" > "${HEADER_FILE}"

if ! compgen -G "${SPLIT_DIR}/batch_*.tsv" >/dev/null; then
    TMP_SPLIT_DIR="${SPLIT_DIR}/chunks"
    rm -rf "${TMP_SPLIT_DIR}"
    mkdir -p "${TMP_SPLIT_DIR}"
    tail -n +2 "${MANIFEST}" | split -l "${BATCH_SIZE}" - "${TMP_SPLIT_DIR}/chunk_"

    batch_idx=1
    for chunk in "${TMP_SPLIT_DIR}"/chunk_*; do
        batch_manifest=$(printf "%s/batch_%04d.tsv" "${SPLIT_DIR}" "${batch_idx}")
        cat "${HEADER_FILE}" "${chunk}" > "${batch_manifest}"
        batch_idx=$((batch_idx + 1))
    done
    rm -rf "${TMP_SPLIT_DIR}"
fi

STATUS_FILE="${LOG_DIR}/batch_status.tsv"
if [[ ! -f "${STATUS_FILE}" ]]; then
    echo -e "batch\tstatus\texpected_pt\tfound_pt" > "${STATUS_FILE}"
fi

FAILED_SLIDES_FILE="${LOG_DIR}/failed_slides.tsv"
if [[ ! -f "${FAILED_SLIDES_FILE}" ]]; then
    echo -e "batch\tstatus\tslide_id" > "${FAILED_SLIDES_FILE}"
fi

echo "[info] manifest: ${MANIFEST}"
echo "[info] study: ${STUDY}"
echo "[info] arch: ${ARCH}"
echo "[info] work dir: ${WORK_DIR}"
echo "[info] final features: ${FEAT_DIR}/pt_files"
echo "[info] batch size: ${BATCH_SIZE}"
echo "[info] allow failed slides: ${ALLOW_FAILED_SLIDES}"
echo "[info] source/target patch size: ${SIZE}/${TARGET_PATCH_SIZE}"
echo "[info] extraction batch/workers/dtype: ${EXTRACT_BATCH_SIZE}/${EXTRACT_NUM_WORKERS}/${SAVE_DTYPE}"

#######################################
# Main loop
#######################################

for batch_manifest in "${SPLIT_DIR}"/batch_*.tsv; do
    batch_name=$(basename "${batch_manifest}" .tsv)
    batch_num=$((10#${batch_name#batch_}))

    if (( batch_num < START_BATCH )); then
        continue
    fi

    expected_pt=$(count_manifest_slides "${batch_manifest}")
    found_pt_before=$(count_expected_pt "${batch_manifest}")

    echo
    echo "[info] ===== ${batch_name}: expected ${expected_pt}, already found ${found_pt_before} pt files ====="

    if [[ "${expected_pt}" == "${found_pt_before}" ]]; then
        if [[ "${ARCH}" == "CONCH" && "${PROJ_TO_CONTRAST}" == "Y" && "${VALIDATE_CONCH}" == "1" ]]; then
            require_file "${VALIDATOR}"
            run_cmd python3 "${VALIDATOR}" \
                --feature-dir "${FEAT_DIR}/pt_files" \
                --manifest "${batch_manifest}" \
                --slide-ext "${SLIDE_EXT}" \
                --expected-dim "${CONCH_EXPECTED_DIM}" \
                --expected-dtype "${SAVE_DTYPE}"
        fi
        echo "[info] ${batch_name} already completed; skipping."
        echo -e "${batch_name}\tskipped_existing\t${expected_pt}\t${found_pt_before}" >> "${STATUS_FILE}"
        continue
    fi

    RAW_BATCH_DIR="${WORK_DIR}/raw/${batch_name}"
    PATCH_BATCH_DIR="${WORK_DIR}/patch/${batch_name}/tiles-${MAG}x-s${SIZE}"
    mkdir -p "${RAW_BATCH_DIR}" "${PATCH_BATCH_DIR}"

    found_raw_before=$(count_raw_slides "${batch_manifest}" "${RAW_BATCH_DIR}")
    if [[ "${expected_pt}" == "${found_raw_before}" ]]; then
        echo "[info] ${batch_name} raw slides already present; skipping download."
    else
        echo "[info] downloading ${batch_name}: found ${found_raw_before}/${expected_pt} raw slides before download"
        run_cmd "${GDC_CLIENT}" download -m "${batch_manifest}" -d "${RAW_BATCH_DIR}"
    fi

    cd "${CLAM_DIR}"

    patch_args=(
        python3 create_patches_fp.py
        --source "${RAW_BATCH_DIR}"
        --save_dir "${PATCH_BATCH_DIR}"
        --patch_size "${SIZE}"
        --step_size "${SIZE}"
        --preset tcga.csv
        --patch_magnification "${MAG}"
        --seg
        --patch
        --auto_skip
        --in_child_dir
    )

    if [[ "${SAVE_MASK}" == "1" ]]; then
        patch_args+=(--save_mask)
    fi
    if [[ "${STITCH}" == "1" ]]; then
        patch_args+=(--stitch)
    fi

    echo "[info] running S03 patch-coordinate generation for ${batch_name}"
    run_cmd env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" "${patch_args[@]}"

    echo "[info] running S04 feature extraction for ${batch_name}"
    run_cmd env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" python3 extract_features_fp.py \
        --arch "${ARCH}" \
        --ckpt_path "${MODEL_CKPT}" \
        --data_h5_dir "${PATCH_BATCH_DIR}" \
        --data_slide_dir "${RAW_BATCH_DIR}" \
        --csv_path "${PATCH_BATCH_DIR}/process_list_autogen.csv" \
        --feat_dir "${FEAT_DIR}" \
        --target_patch_size "${TARGET_PATCH_SIZE}" \
        --batch_size "${EXTRACT_BATCH_SIZE}" \
        --num_workers "${EXTRACT_NUM_WORKERS}" \
        --save_dtype "${SAVE_DTYPE}" \
        --slide_ext "${SLIDE_EXT}" \
        --slide_in_child_dir \
        --auto_skip \
        --proj_to_contrast "${PROJ_TO_CONTRAST}"

    if [[ "${DRY_RUN}" == "1" ]]; then
        echo "[dry-run] would validate output, then clean raw/intermediate files according to KEEP_* settings"
        continue
    fi

    found_pt_after=$(count_expected_pt "${batch_manifest}")
    echo "[info] ${batch_name}: found ${found_pt_after}/${expected_pt} pt files after extraction"

    failed_patch_count=$(count_failed_patch_slides "${PATCH_BATCH_DIR}/process_list_autogen.csv")
    if [[ "${found_pt_after}" != "${expected_pt}" ]]; then
        if [[ "${ALLOW_FAILED_SLIDES}" == "1" ]] && (( found_pt_after + failed_patch_count == expected_pt )); then
            echo "[warning] ${batch_name}: accepting ${failed_patch_count} S03-failed slide(s); see ${FAILED_SLIDES_FILE}"
            append_failed_patch_slides "${batch_name}" "${PATCH_BATCH_DIR}/process_list_autogen.csv" "${FAILED_SLIDES_FILE}"
            echo -e "${batch_name}\tcompleted_with_failed_slides\t${expected_pt}\t${found_pt_after}" >> "${STATUS_FILE}"
        else
            echo -e "${batch_name}\tincomplete\t${expected_pt}\t${found_pt_after}" >> "${STATUS_FILE}"
            die "${batch_name} incomplete; keeping raw/intermediate files for inspection"
        fi
    else
        echo -e "${batch_name}\tcompleted\t${expected_pt}\t${found_pt_after}" >> "${STATUS_FILE}"
    fi

    if [[ "${ARCH}" == "CONCH" && "${PROJ_TO_CONTRAST}" == "Y" && "${VALIDATE_CONCH}" == "1" ]]; then
        require_file "${VALIDATOR}"
        echo "[info] validating normalized CONCH tensors before cleanup"
        run_cmd python3 "${VALIDATOR}" \
            --feature-dir "${FEAT_DIR}/pt_files" \
            --manifest "${batch_manifest}" \
            --slide-ext "${SLIDE_EXT}" \
            --expected-dim "${CONCH_EXPECTED_DIM}" \
            --expected-dtype "${SAVE_DTYPE}" \
            --allow-missing "${failed_patch_count}"
    fi

    if [[ "${KEEP_RAW}" != "1" ]]; then
        echo "[info] removing raw slides for ${batch_name}"
        safe_rm_rf "${RAW_BATCH_DIR}"
    fi

    if [[ "${KEEP_PATCH_COORDS}" != "1" ]]; then
        echo "[info] removing patch coordinates for ${batch_name}"
        safe_rm_rf "${PATCH_BATCH_DIR}/patches"
    fi

    if [[ "${KEEP_MASKS}" != "1" ]]; then
        safe_rm_rf "${PATCH_BATCH_DIR}/masks"
    fi

    if [[ "${KEEP_STITCHES}" != "1" ]]; then
        safe_rm_rf "${PATCH_BATCH_DIR}/stitches"
    fi

    echo "[info] ${batch_name} finished"
done

echo
echo "[info] all requested batches finished"
echo "[info] final features are in: ${FEAT_DIR}/pt_files"
echo "[info] status file: ${STATUS_FILE}"
