#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PIPELINE="${ROOT_DIR}/feature_extract/tools/scripts/S06-Stream-KIRC-CONCH-v1.5.sh"

export MANIFEST=${MANIFEST:-"${ROOT_DIR}/feature_extract/tools/gdc/gdc_manifest_kirc_slotspe_494.txt"}
export PATHOLOGY_DATA_ROOT=${PATHOLOGY_DATA_ROOT:-"/data0/lfy_data/Pathology"}
export TITAN_MODEL_DIR=${TITAN_MODEL_DIR:-"${PATHOLOGY_DATA_ROOT}/checkpoints/TITAN"}
export FEAT_DIR=${FEAT_DIR:-"${PATHOLOGY_DATA_ROOT}/CONCH_v1.5/kirc"}
export GPU_WAIT_CANDIDATES=${GPU_WAIT_CANDIDATES:-"0,1,2,3,4"}
export GPU_MIN_FREE_MIB=${GPU_MIN_FREE_MIB:-14000}
export GPU_POLL_SECONDS=${GPU_POLL_SECONDS:-60}
export GDC_DIRECT=${GDC_DIRECT:-1}
export GDC_DOWNLOAD_RETRIES=${GDC_DOWNLOAD_RETRIES:-5}
export GDC_RETRY_SECONDS=${GDC_RETRY_SECONDS:-30}

exec bash "${PIPELINE}"
