#!/usr/bin/env bash
set -euo pipefail

# =================== 基础配置 ===================
SCRIPT="bpr_lightfm.py"   # 对齐改后的手写 LightFM 脚本名
DATA_ROOT="./"

DEVICE="cuda:0"
EPOCHS=5
BATCH_SIZE=4096
FACTORS=128
NEG_PER_POS=1
TOPK=10
EVAL_CAND_SIZE=1000
KNN_N=8

# LightFM 混合项权重（手写实现中的 alpha）
ALPHA_ID=1.0
ALPHA_FEAT=1.0

# ============ 模式 A：自动用 TF-IDF 构建特征 ============
# 若你想用模式 B（预构建特征），把这两个变量改成实际文件路径（非空即启用）
USER_FEATURES_PATH=""   # 例如："/path/to/user_features_csr.pkl"
ITEM_FEATURES_PATH=""   # 例如："/path/to/item_features_csr.pkl"

# 仅在“自动 TF-IDF”模式下有效（新版脚本统一一个 max_features）
MAX_FEATURES=5000
REBUILD_FEATURE_CACHE=0   # 首次切换新管线建议设为 1，重建 TF-IDF 缓存

# 可选：CUDA 设备可控（不需要就注释掉）
# export CUDA_VISIBLE_DEVICES=0

# =================== 日志配置 ===================
mkdir -p logs
TS="$(date +%Y%m%d_%H%M%S)"
LOG="logs/lightfm_${TS}.log"

# =================== 组装命令 ===================
CMD=(python "$SCRIPT"
  --data_root "$DATA_ROOT"
  --epochs "$EPOCHS"
  --batch_size "$BATCH_SIZE"
  --factors "$FACTORS"
  --neg_per_pos "$NEG_PER_POS"
  --topk "$TOPK"
  --device "$DEVICE"
  --eval_cand_size "$EVAL_CAND_SIZE"
  --knn_N "$KNN_N"
  --alpha_id "$ALPHA_ID"
  --alpha_feat "$ALPHA_FEAT"
)

# 如果提供了预构建特征路径，则走“模式 B”
if [[ -n "$USER_FEATURES_PATH" && -n "$ITEM_FEATURES_PATH" ]]; then
  CMD+=( --user_features_path "$USER_FEATURES_PATH"
         --item_features_path "$ITEM_FEATURES_PATH" )
else
  # 否则使用“模式 A：自动 TF-IDF”
  CMD+=( --max_features "$MAX_FEATURES"
         --rebuild_feature_cache "$REBUILD_FEATURE_CACHE" )
fi

# =================== 执行 ===================
echo "[INFO] Running:"
printf ' %q' "${CMD[@]}"; echo
echo "[INFO] Logging to: $LOG"
set -x
"${CMD[@]}" 2>&1 | tee "$LOG"
set +x

echo "[DONE] Finished. Log saved at: $LOG"
