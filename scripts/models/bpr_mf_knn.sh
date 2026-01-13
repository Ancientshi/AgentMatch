#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../common_env.sh"

EPOCHS=10
BATCH_SIZE=4096

python "$SCRIPT_DIR/../../run_bpr_mf_knn.py" \
  --data_root "$DATA_ROOT" \
  --device "$DEVICE" \
  --epochs "$EPOCHS" \
  --batch_size "$BATCH_SIZE" \
  --factors 128 --neg_per_pos "$NEG_PER_POS" \
  --knn_N "$KNN_N" --eval_cand_size "$EVAL_CAND_SIZE" --score_mode dot \
  --exp_name "bpr_mf_knn"
