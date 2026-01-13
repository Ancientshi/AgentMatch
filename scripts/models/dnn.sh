#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../common_env.sh"

EPOCHS=5
BATCH_SIZE=4096
LR=1e-3

TEXT_HIDDEN="${TEXT_HIDDEN:-256}"
ID_DIM="${ID_DIM:-64}"
MAX_FEATURES="${MAX_FEATURES:-5000}"


python "$SCRIPT_DIR/../../run_dnn.py" \
  --data_root "$DATA_ROOT" \
  --device "$DEVICE" \
  --epochs "$EPOCHS" \
  --batch_size "$BATCH_SIZE" \
  --lr "$LR" \
  --text_hidden "$TEXT_HIDDEN" \
  --id_dim "$ID_DIM" \
  --max_features "$MAX_FEATURES" \
  --neg_per_pos "$NEG_PER_POS" \
  --eval_cand_size "$EVAL_CAND_SIZE" \
  --use_query_id_emb 0 \
  --use_llm_id_emb 1 \
  --use_tool_id_emb 1 \
  --use_model_content_vector 1 \
  --use_tool_content_vector 1 \
  --exp_name "bpr_dnn"
