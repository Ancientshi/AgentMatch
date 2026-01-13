#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../common_env.sh"

EPOCHS=10
BATCH_SIZE=4096

python "$SCRIPT_DIR/../../run_ngcf.py" \
  --data_root "$DATA_ROOT" \
  --device "$DEVICE" \
  --epochs "$EPOCHS" \
  --batch_size "$BATCH_SIZE" \
  --embed_dim 32 --num_layers 2 --dropout 0.2 \
  --neg_per_pos "$NEG_PER_POS" \
  --knn_N "$KNN_N" --eval_cand_size "$EVAL_CAND_SIZE" --score_mode dot \
  --use_query_id_emb 1 \
  --use_llm_id_emb 1 \
  --use_tool_id_emb 1 \
  --use_model_content_vector 1 \
  --use_tool_content_vector 1 \
  --exp_name "ngcf"
