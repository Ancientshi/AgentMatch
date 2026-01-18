#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common_env.sh"

KS="${KS:-10}"
VALID_RATIO="${VALID_RATIO:-0.2}"
MAX_EVAL="${MAX_EVAL:-0}"
RERANK_BATCH="${RERANK_BATCH:-64}"
TIMEOUT="${TIMEOUT:-300}"
MAX_WORKERS="${MAX_WORKERS:-16}"
HTTP_RETRIES="${HTTP_RETRIES:-3}"
HTTP_BACKOFF="${HTTP_BACKOFF:-0.3}"
POOL_MAXSIZE="${POOL_MAXSIZE:-64}"
SEED="${SEED:-1234}"
SPLIT_SEED="${SPLIT_SEED:-42}"

BGE_BASE_MODEL="${BGE_BASE_MODEL:-BAAI/bge-reranker-base}"
BGE_BASE_DIR="${BGE_BASE_DIR:-$BGE_BASE_MODEL}"
BGE_LORA_DIR="${BGE_LORA_DIR:-/home/username/Data/workspace/QueryAgentMatch/benchmark/BGE-Rerank/reranker}"

EASYREC_BASE_URL="${EASYREC_BASE_URL:-http://127.0.0.1:8500/compute_scores}"
EASYREC_LORA_URL="${EASYREC_LORA_URL:-http://127.0.0.1:8501/compute_scores}"

# python "$SCRIPT_DIR/../infer_BGE.py" \
#   --data_root "$DATA_ROOT" \
#   --exp_name "infer_bge_base" \
#   --model_dir "$BGE_BASE_DIR" \
#   --model_name "$BGE_BASE_MODEL" \
#   --peft 0 \
#   --device "$DEVICE" \
#   --eval_cand_size "$EVAL_CAND_SIZE" \
#   --valid_ratio "$VALID_RATIO" \
#   --max_eval "$MAX_EVAL" \
#   --ks "$KS" \
#   --rerank_batch "$RERANK_BATCH" \
#   --seed "$SEED" \
#   --split_seed "$SPLIT_SEED"

# python "$SCRIPT_DIR/../infer_BGE.py" \
#   --data_root "$DATA_ROOT" \
#   --exp_name "infer_bge_lora" \
#   --model_dir "$BGE_LORA_DIR" \
#   --peft 1 \
#   --device "$DEVICE" \
#   --eval_cand_size "$EVAL_CAND_SIZE" \
#   --valid_ratio "$VALID_RATIO" \
#   --max_eval "$MAX_EVAL" \
#   --ks "$KS" \
#   --rerank_batch "$RERANK_BATCH" \
#   --seed "$SEED" \
#   --split_seed "$SPLIT_SEED"

# python "$SCRIPT_DIR/../infer_EasyRec.py" \
#   --data_root "$DATA_ROOT" \
#   --exp_name "infer_easyrec_base" \
#   --service_url "$EASYREC_BASE_URL" \
#   --eval_cand_size "$EVAL_CAND_SIZE" \
#   --valid_ratio "$VALID_RATIO" \
#   --max_eval "$MAX_EVAL" \
#   --ks "$KS" \
#   --rerank_batch "$RERANK_BATCH" \
#   --timeout "$TIMEOUT" \
#   --max_workers "$MAX_WORKERS" \
#   --http_retries "$HTTP_RETRIES" \
#   --http_backoff "$HTTP_BACKOFF" \
#   --pool_maxsize "$POOL_MAXSIZE" \
#   --seed "$SEED" \
#   --split_seed "$SPLIT_SEED" \

python "$SCRIPT_DIR/../infer_EasyRec.py" \
  --data_root "$DATA_ROOT" \
  --exp_name "infer_easyrec_lora" \
  --service_url "$EASYREC_LORA_URL" \
  --eval_cand_size "$EVAL_CAND_SIZE" \
  --valid_ratio "$VALID_RATIO" \
  --max_eval "$MAX_EVAL" \
  --ks "$KS" \
  --rerank_batch "$RERANK_BATCH" \
  --timeout "$TIMEOUT" \
  --max_workers "$MAX_WORKERS" \
  --http_retries "$HTTP_RETRIES" \
  --http_backoff "$HTTP_BACKOFF" \
  --pool_maxsize "$POOL_MAXSIZE" \
  --seed "$SEED" \
  --split_seed "$SPLIT_SEED"
