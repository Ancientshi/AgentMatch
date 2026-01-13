#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../common_env.sh"

EPOCHS=5
BATCH_SIZE=4096
LR=1e-3
DEVICE="cuda:0"

TUNE_MODE="${TUNE_MODE:-frozen}"
UNFREEZE_LAST_N="${UNFREEZE_LAST_N:-2}"
UNFREEZE_EMB="${UNFREEZE_EMB:-1}"
GRAD_CKPT="${GRAD_CKPT:-0}"
POOLING="${POOLING:-cls}"
MAX_LEN="${MAX_LEN:-128}"
TEXT_HIDDEN="${TEXT_HIDDEN:-256}"
TOPK="${TOPK:-10}"
PRETRAINED_MODEL="${PRETRAINED_MODEL:-distilbert-base-uncased}"
ENCODER_LR="${ENCODER_LR:-5e-5}"
ENCODER_WEIGHT_DECAY="${ENCODER_WEIGHT_DECAY:-0.01}"
LORA_R="${LORA_R:-8}"
LORA_ALPHA="${LORA_ALPHA:-16}"
LORA_DROPOUT="${LORA_DROPOUT:-0.1}"
LORA_TARGETS="${LORA_TARGETS:-q_lin,k_lin,v_lin,out_lin}"


python "$SCRIPT_DIR/../../run_bpr_bert.py" \
  --data_root "$DATA_ROOT" \
  --device "$DEVICE" \
  --epochs "$EPOCHS" \
  --batch_size "$BATCH_SIZE" \
  --lr "$LR" \
  --pretrained_model "$PRETRAINED_MODEL" \
  --max_len "$MAX_LEN" \
  --text_hidden "$TEXT_HIDDEN" \
  --id_dim "$ID_DIM" \
  --neg_per_pos "$NEG_PER_POS" \
  --topk "$TOPK" \
  --pooling "$POOLING" \
  --tune_mode "$TUNE_MODE" \
  --unfreeze_last_n "$UNFREEZE_LAST_N" \
  --unfreeze_emb "$UNFREEZE_EMB" \
  --grad_ckpt "$GRAD_CKPT" \
  --lora_r "$LORA_R" \
  --lora_alpha "$LORA_ALPHA" \
  --lora_dropout "$LORA_DROPOUT" \
  --lora_targets "$LORA_TARGETS" \
  --encoder_lr "$ENCODER_LR" \
  --encoder_weight_decay "$ENCODER_WEIGHT_DECAY" \
  --use_query_id_emb 0 \
  --use_llm_id_emb 1 \
  --use_tool_id_emb 1 \
  --use_model_content_vector 1 \
  --use_tool_content_vector 1 \
  --exp_name "bpr_bert_${TUNE_MODE}"
