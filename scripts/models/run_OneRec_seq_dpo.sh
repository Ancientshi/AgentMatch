#!/usr/bin/env bash
set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../common_env.sh"

python "$SCRIPT_DIR/../../OneRec_seq.py" \
  --data_root "$DATA_ROOT" \
  --device cuda:0 \
  --mode dpo \
  --dpo_part_scope PartI \
  --topk 10 \
  --candidate_size 300 \
  --eval_candidate_size 1000 \
  --train_mask 1 \
  --cand_extra 64 \
  --rand_neg_ratio 0.25 \
  --dpo_steps 500 \
  --dpo_batch 128 \
  --beta 0.05 \
  --dpo_margin 0.01 \
  --dpo_use_gt_reward 1 \
  --freeze_q_enc_dpo 1 \
  --amp 0
