#!/usr/bin/env bash
set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../common_env.sh"

# OneRec_seq.py uses FEATURE_DIR to find FeatureCache (llm/tool ids, etc.)
export FEATURE_DIR="/home/yunxshi/Data/workspace/QueryAgentMatch/QueryAgentMatch-Public/dataset/.cache/shared/features/twotower_bge_fd6ccb32_01915b90"

python "$SCRIPT_DIR/../../OneRec_seq.py" \
  --data_root "$DATA_ROOT" \
  --device "cuda:0" \
  --mode sft \
  --topk 10 \
  --epochs 5 \
  --batch_size 1024 \
  --candidate_size 200 \
  --train_mask 1 \
  --cand_extra 64 \
  --rand_neg_ratio 0.25 \
  --max_tool_per_agent 8 \
  --amp 1 \
  --eval_candidate_size 1000 \
  --eval_per_part 200 \
  --eval_parts "PartI,PartII,PartIII" \
  --skip_eval 0
