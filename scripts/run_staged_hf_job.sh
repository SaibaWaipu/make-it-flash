#!/usr/bin/env bash
# Main staged entrypoint: one HF Job fits all 32 layers and commits each fit.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/run_all_staged_hf_job.sh" "$@"
