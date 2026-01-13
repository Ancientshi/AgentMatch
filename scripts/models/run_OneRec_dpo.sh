#!/usr/bin/env bash
set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../common_env.sh"

python "$SCRIPT_DIR/../../OneRec_plus.py" \
  --data_root "$DATA_ROOT" \
  --device "${DEVICE:-cuda:0}" \
  \
  --mode dpo \
  --topk 10 \
  \
  --decoder tx \
  --enc_dim 512 --tok_dim 256 \
  --hidden 512 --num_layers 2 --n_heads 8 \
  --dropout 0.1 --max_len 64 \
  \
  --dpo_steps 500 \
  --dpo_batch 64 \
  --beta 0.05 \
  --lr 1e-5 \
  --dpo_lr 5e-5 \
  --freeze_q_enc_dpo 1 \
  \
  --train_mask 1 \
  --candidate_size 500 \
  --cand_extra 64 \
  --rand_neg_ratio 0.10 \
  \
  --bge_feature_dir /home/yunxshi/Data/workspace/QueryAgentMatch/QueryAgentMatch-Public/dataset/.cache/shared/features/twotower_bge_fd6ccb32_01915b90 \
  --init_tok_from_bge 1 \
  --freeze_tok_emb 1 \
  \
  --enc_chunk 512 \
  --eval_candidate_size 200 \
  --skip_eval 0 \
  --amp 1 \
  --dpo_use_gt_reward 1
