#!/usr/bin/env bash
set -euo pipefail

# Download TCGA-KIRC raw WSI only.
#
# This script intentionally does not run CLAM patching, CONCH/TITAN feature
# extraction, validation, or cleanup.  It keeps GDC partial files so rerunning
# the same command can resume after an interrupted download.

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

STUDY=${STUDY:-kirc}
PATHOLOGY_DATA_ROOT=${PATHOLOGY_DATA_ROOT:-"${ROOT_DIR}/Pathology"}
MANIFEST=${MANIFEST:-"${ROOT_DIR}/feature_extract/tools/gdc/gdc_manifest_kirc_slotspe_494.txt"}
GDC_CLIENT=${GDC_CLIENT:-"${ROOT_DIR}/feature_extract/tools/gdc/gdc-client"}
RAW_DIR=${RAW_DIR:-"${PATHOLOGY_DATA_ROOT}/RawWSI/${STUDY}_gdc"}
LOG_DIR=${LOG_DIR:-"${PATHOLOGY_DATA_ROOT}/RawWSI/logs"}
LOG_FILE=${LOG_FILE:-"${LOG_DIR}/${STUDY}_gdc_download.log"}

SLIDE_EXT=${SLIDE_EXT:-".svs"}
SLIDE_EXT_UPPER=$(printf '%s' "${SLIDE_EXT}" | tr '[:lower:]' '[:upper:]')
MIN_FREE_GIB=${MIN_FREE_GIB:-600}
GDC_DIRECT=${GDC_DIRECT:-1}
GDC_PREFLIGHT_URL=${GDC_PREFLIGHT_URL:-"https://api.gdc.cancer.gov/status"}
DRY_RUN=${DRY_RUN:-0}

die() {
    echo "[error] $*" >&2
    exit 1
}

timestamp() {
    date '+%Y-%m-%dT%H:%M:%S%z'
}

run_cmd() {
    echo "[cmd] $*"
    if [[ "${DRY_RUN}" != "1" ]]; then
        "$@"
    fi
}

gdc_direct_env() {
    env \
        -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
        -u http_proxy -u https_proxy -u all_proxy \
        NO_PROXY='*' no_proxy='*' "$@"
}

count_manifest_slides() {
    awk -v ext="${SLIDE_EXT}" -F'\t' 'NR > 1 && $2 ~ ext "$" { count++ } END { print count + 0 }' "${MANIFEST}"
}

count_downloaded_slides() {
    find "${RAW_DIR}" -type f \( -name "*${SLIDE_EXT}" -o -name "*${SLIDE_EXT_UPPER}" \) | wc -l
}

check_gdc_direct_connectivity() {
    if [[ "${DRY_RUN}" == "1" ]]; then
        echo "[dry-run] would verify direct GDC connectivity: ${GDC_PREFLIGHT_URL}"
        return
    fi
    echo "[info] checking direct GDC connectivity with proxy variables removed"
    gdc_direct_env python3 -c \
        'import sys, urllib.request; response = urllib.request.urlopen(sys.argv[1], timeout=20); print("[info] GDC direct preflight HTTP", response.status)' \
        "${GDC_PREFLIGHT_URL}" \
        || die "cannot reach GDC directly; fix DNS/network or set GDC_DIRECT=0 if you intentionally need the current proxy"
}

[[ -f "${MANIFEST}" ]] || die "manifest not found: ${MANIFEST}"
if [[ "${DRY_RUN}" != "1" ]]; then
    [[ -x "${GDC_CLIENT}" ]] || die "gdc-client is not executable: ${GDC_CLIENT}"
fi

mkdir -p "${RAW_DIR}" "${LOG_DIR}"

available_kib=$(df -Pk "${RAW_DIR}" | awk 'NR == 2 { print $4 }')
required_kib=$((MIN_FREE_GIB * 1024 * 1024))
if (( available_kib < required_kib )); then
    available_gib=$((available_kib / 1024 / 1024))
    die "only ${available_gib} GiB free at ${RAW_DIR}; require at least ${MIN_FREE_GIB} GiB"
fi

expected=$(count_manifest_slides)
downloaded=$(count_downloaded_slides)

{
    echo "[$(timestamp)] START raw ${STUDY} WSI download"
    echo "[info] repo: ${ROOT_DIR}"
    echo "[info] manifest: ${MANIFEST}"
    echo "[info] gdc-client: ${GDC_CLIENT}"
    echo "[info] raw output: ${RAW_DIR}"
    echo "[info] expected slides: ${expected}"
    echo "[info] already downloaded slides: ${downloaded}"
    echo "[info] dry run: ${DRY_RUN}"
} | tee -a "${LOG_FILE}"

if [[ "${GDC_DIRECT}" == "1" ]]; then
    check_gdc_direct_connectivity 2>&1 | tee -a "${LOG_FILE}"
    run_cmd gdc_direct_env "${GDC_CLIENT}" download -m "${MANIFEST}" -d "${RAW_DIR}" 2>&1 | tee -a "${LOG_FILE}"
else
    run_cmd "${GDC_CLIENT}" download -m "${MANIFEST}" -d "${RAW_DIR}" 2>&1 | tee -a "${LOG_FILE}"
fi

downloaded=$(count_downloaded_slides)
{
    echo "[$(timestamp)] DONE raw ${STUDY} WSI download"
    echo "[info] downloaded slides: ${downloaded}/${expected}"
    du -sh "${RAW_DIR}" || true
} | tee -a "${LOG_FILE}"
