#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../common_env.sh"

EPOCHS=5
BATCH_SIZE=4096

FACTORS="${FACTORS:-128}"
ALPHA_ID="${ALPHA_ID:-1.0}"
ALPHA_FEAT="${ALPHA_FEAT:-1.0}"
MAX_FEATURES="${MAX_FEATURES:-5000}"

python "$SCRIPT_DIR/../../run_lightfm.py" \
  --data_root "$DATA_ROOT" \
  --device "$DEVICE" \
  --epochs "$EPOCHS" \
  --batch_size "$BATCH_SIZE" \
  --factors "$FACTORS" \
  --neg_per_pos "$NEG_PER_POS" \
  --alpha_id "$ALPHA_ID" \
  --alpha_feat "$ALPHA_FEAT" \
  --max_features "$MAX_FEATURES" \
  --knn_N "$KNN_N" \
  --eval_cand_size "$EVAL_CAND_SIZE" \
  --use_llm_id_emb 1 \
  --use_tool_id_emb 1 \
  --use_model_content_vector 1 \
  --use_tool_content_vector 1 \
  --exp_name "lightfm"
