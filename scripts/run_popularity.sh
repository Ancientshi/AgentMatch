#!/usr/bin/env bash
# Helper to run the regime shift analysis without import errors.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Update these to point to your dataset root and desired output directory.
: "${DATA_ROOT:=/home/yunxshi/Data/workspace/QueryAgentMatch/QueryAgentMatch-Public/dataset}"
: "${OUTPUT_DIR:=${REPO_ROOT}/analysis/artifacts_content_pop}"

mkdir -p "${OUTPUT_DIR}"

# Ensure the repo is on PYTHONPATH so agent_rec can be imported.
export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}${REPO_ROOT}"

python "${REPO_ROOT}/analysis/popularity_analysis.py" \
  --data_root "${DATA_ROOT}" \
  --output_dir "${OUTPUT_DIR}" \
  "$@"
