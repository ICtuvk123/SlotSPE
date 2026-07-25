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
DOWNLOAD_WORKERS=${DOWNLOAD_WORKERS:-10}
GDC_PROCESSES_PER_WORKER=${GDC_PROCESSES_PER_WORKER:-2}
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

validate_positive_integer() {
    local name=$1
    local value=$2
    [[ "${value}" =~ ^[1-9][0-9]*$ ]] || die "${name} must be a positive integer, got: ${value}"
}

build_manifest_shards() {
    local shard_dir=$1

    mkdir -p "${shard_dir}"
    find "${shard_dir}" -maxdepth 1 -type f -name 'manifest_*.tsv' -delete
    awk -v workers="${DOWNLOAD_WORKERS}" -v output_dir="${shard_dir}" -F'\t' '
        NR == 1 {
            header = $0
            next
        }
        {
            worker = (NR - 2) % workers
            output = sprintf("%s/manifest_%02d.tsv", output_dir, worker)
            if (!(output in initialized)) {
                print header > output
                initialized[output] = 1
            }
            print $0 >> output
        }
    ' "${MANIFEST}"
}

download_shard() {
    local shard=$1
    local worker_log=$2
    local -a command=(
        "${GDC_CLIENT}" download
        -m "${shard}"
        -d "${RAW_DIR}"
        -n "${GDC_PROCESSES_PER_WORKER}"
    )

    if [[ "${GDC_DIRECT}" == "1" ]]; then
        run_cmd gdc_direct_env "${command[@]}" >"${worker_log}" 2>&1
    else
        run_cmd "${command[@]}" >"${worker_log}" 2>&1
    fi
}

[[ -f "${MANIFEST}" ]] || die "manifest not found: ${MANIFEST}"
validate_positive_integer DOWNLOAD_WORKERS "${DOWNLOAD_WORKERS}"
validate_positive_integer GDC_PROCESSES_PER_WORKER "${GDC_PROCESSES_PER_WORKER}"
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
    echo "[info] parallel files: ${DOWNLOAD_WORKERS}"
    echo "[info] connections per file: ${GDC_PROCESSES_PER_WORKER}"
    echo "[info] maximum connections: $((DOWNLOAD_WORKERS * GDC_PROCESSES_PER_WORKER))"
    echo "[info] dry run: ${DRY_RUN}"
} | tee -a "${LOG_FILE}"

if [[ "${GDC_DIRECT}" == "1" ]]; then
    check_gdc_direct_connectivity 2>&1 | tee -a "${LOG_FILE}"
fi

SHARD_DIR="${LOG_DIR}/${STUDY}_manifest_shards_${DOWNLOAD_WORKERS}"
build_manifest_shards "${SHARD_DIR}"

pids=()
worker_logs=()
for shard in "${SHARD_DIR}"/manifest_*.tsv; do
    worker_name=$(basename "${shard}" .tsv)
    worker_log="${LOG_DIR}/${STUDY}_gdc_${worker_name}.log"
    echo "[$(timestamp)] START ${worker_name}: ${shard}" | tee -a "${worker_log}"
    download_shard "${shard}" "${worker_log}" &
    pids+=("$!")
    worker_logs+=("${worker_log}")
done

stop_workers() {
    if (( ${#pids[@]} > 0 )); then
        kill "${pids[@]}" 2>/dev/null || true
    fi
}
trap stop_workers INT TERM

failed=0
for index in "${!pids[@]}"; do
    if wait "${pids[$index]}"; then
        echo "[$(timestamp)] DONE worker log: ${worker_logs[$index]}" | tee -a "${LOG_FILE}"
    else
        echo "[$(timestamp)] FAILED worker log: ${worker_logs[$index]}" | tee -a "${LOG_FILE}"
        failed=1
    fi
done
trap - INT TERM

if (( failed != 0 )); then
    die "one or more download workers failed; rerun the script to resume"
fi

downloaded=$(count_downloaded_slides)
{
    echo "[$(timestamp)] DONE raw ${STUDY} WSI download"
    echo "[info] downloaded slides: ${downloaded}/${expected}"
    du -sh "${RAW_DIR}" || true
} | tee -a "${LOG_FILE}"
