#!/usr/bin/env bash
set -euo pipefail


SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"


DATA_ROOT=/home/username/Data/workspace/AgentMatchdataset \
MODEL_PATH=/home/username/Data/workspace/QueryAgentMatch/QueryAgentMatch-Public/dataset/.cache/two_tower_all/models/latest_01915b90.pt \
HOST=0.0.0.0 \
PORT=8000 \
DEVICE=cuda:0 \
TOPK=10 \


DATA_ROOT=${DATA_ROOT:-""}
MODEL_PATH=${MODEL_PATH:-""}
HOST=${HOST:-"0.0.0.0"}
PORT=${PORT:-8000}
DEVICE=${DEVICE:-"cpu"}
TOPK=${TOPK:-10}
MAX_FEATURES=${MAX_FEATURES:-5000}

if [[ -z "${DATA_ROOT}" || -z "${MODEL_PATH}" ]]; then
  echo "[error] Please set DATA_ROOT and MODEL_PATH before running." >&2
  exit 1
fi

python "$SCRIPT_DIR/../infer_twotower_tfidf.py" \
  --data_root "${DATA_ROOT}" \
  --model_path "${MODEL_PATH}" \
  --device "${DEVICE}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --topk "${TOPK}" \
  --max_features "${MAX_FEATURES}" \
