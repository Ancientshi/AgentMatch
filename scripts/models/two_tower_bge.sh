#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../common_env.sh"

EPOCHS=5
BATCH_SIZE=512

EMBED_URL="${EMBED_URL:-http://127.0.0.1:8500/get_embedding}"
EMBED_BATCH="${EMBED_BATCH:-64}"
HID="${HID:-256}"
TEMPERATURE="${TEMPERATURE:-0.07}"
EVAL_CHUNK="${EVAL_CHUNK:-8192}"
USE_TOOL_ID_EMB="${USE_TOOL_ID_EMB:-${USE_TOOL_EMB:-1}}"
USE_LLM_ID_EMB="${USE_LLM_ID_EMB:-${USE_AGENT_ID_EMB:-1}}"
USE_MODEL_CONTENT_VECTOR="${USE_MODEL_CONTENT_VECTOR:-0}"
USE_TOOL_CONTENT_VECTOR="${USE_TOOL_CONTENT_VECTOR:-0}"
USE_QUERY_ID_EMB="${USE_QUERY_ID_EMB:-0}"
TOPK="${TOPK:-10}"


python "$SCRIPT_DIR/../../run_twotower_bge.py" \
  --data_root "$DATA_ROOT" \
  --device "$DEVICE" \
  --epochs "$EPOCHS" \
  --batch_size "$BATCH_SIZE" \
  --embed_url "$EMBED_URL" \
  --embed_batch "$EMBED_BATCH" \
  --hid "$HID" \
  --temperature "$TEMPERATURE" \
  --topk "$TOPK" \
  --eval_chunk "$EVAL_CHUNK" \
  --use_tool_id_emb "$USE_TOOL_ID_EMB" \
  --use_llm_id_emb "$USE_LLM_ID_EMB" \
  --use_model_content_vector "$USE_MODEL_CONTENT_VECTOR" \
  --use_tool_content_vector "$USE_TOOL_CONTENT_VECTOR" \
  --use_query_id_emb "$USE_QUERY_ID_EMB" \
  --exp_name "two_tower_onlyid"
