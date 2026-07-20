#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
echo "[info] CONCH v1.5 patch tokens require learned TITAN text adapters; using the adapter launcher."
exec bash "${ROOT_DIR}/scripts/run_kirc_conch_v15_titan_adapter_v2.sh" "$@"
