#!/usr/bin/env bash
set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../common_env.sh"

BGE_FEATURE_DIR="$DATA_ROOT/.cache/shared/features/twotower_bge_fd6ccb32_01915b90"


python "$SCRIPT_DIR/../../OneRec_plus.py" \
  --data_root "$DATA_ROOT" \
  --device "cuda:0" \
  \
  --mode gen \
  --topk 10 \
  \
  --decoder tx \
  --enc_dim 512 --tok_dim 256 \
  --hidden 512 --num_layers 2 --n_heads 8 \
  --dropout 0.1 --max_len 64 \
  \
  --epochs 5 \
  --batch_size 1024 \
  --lr 1e-3 \
  --weight_decay 0.01 \
  \
  --train_mask 1 \
  --candidate_size 300 \
  --cand_extra 64 \
  --rand_neg_ratio 0.25 \
  \
  --use_nce 1 \
  --nce_lambda 0.1 \
  --nce_temp 0.07 \
  \
  --bge_feature_dir "$BGE_FEATURE_DIR" \
  --init_tok_from_bge 1 \
  --freeze_tok_emb 1 \
  \
  --eval_candidate_size 1000 \
  --eval_per_part 200 \
  --eval_parts "PartI,PartII,PartIII" \
  --skip_eval 0 \
  \
  --enc_chunk 1024 \
  --amp 1
