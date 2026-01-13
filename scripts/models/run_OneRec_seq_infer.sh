#!/usr/bin/env bash
set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../common_env.sh"

# 默认会自动定位 ckpt:
#   $DATA_ROOT/.cache/OneRec_seq/models/OneRec_seq_<data_sig>.pt
# 如果你要手动指定 ckpt，取消注释并填路径：
# CKPT_PATH="/path/to/OneRec_seq_xxxxxxxx.pt"

python "$SCRIPT_DIR/../../OneRec_seq_infer.py" \
  --data_root "$DATA_ROOT" \
  --device "cuda:0" \
  --topk 10 \
  --valid_ratio 0.2 \
  --split_seed 42 \
  --eval_candidate_size 1000 \
  --eval_per_part 200 \
  --eval_parts "PartI,PartII,PartIII" \
  --seed 1234 \
  --ckpt_prefix "OneRec_seq"
  # --ckpt "$CKPT_PATH"
