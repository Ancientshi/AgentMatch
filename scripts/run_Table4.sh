#!/usr/bin/env bash
set -euo pipefail


SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"


DATA_ROOT=/home/yunxshi/Data/workspace/QueryAgentMatch/QueryAgentMatch-Public/dataset \
MODEL_PATH=/home/yunxshi/Data/workspace/QueryAgentMatch/QueryAgentMatch-Public/dataset/.cache/run_Table4/models/latest_01915b90.pt \
QUESTIONS_JSONL=/home/yunxshi/Data/workspace/QueryAgentMatch/QueryAgentMatch-Public/dataset/PracticalEval/questions/partIII_sample_200_records.jsonl \
DEVICE=cuda:0 \


N=5
RUNS=5
BASE_SEED=20260001  # 你也可以换成任意整数

# 结果输出目录（带时间戳）
TS="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${SCRIPT_DIR}/../outputs/table4_${TS}"
mkdir -p "${OUT_DIR}"

# 安全：不要在脚本里写 key；在终端里 export OPENAI_API_KEY=...
if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "[error] OPENAI_API_KEY is not set. Please export it in your shell (do NOT hardcode it in scripts)." >&2
  exit 1
fi

# 检查路径
[[ -d "${DATA_ROOT}" ]] || { echo "[error] DATA_ROOT not found: ${DATA_ROOT}" >&2; exit 1; }
[[ -f "${MODEL_PATH}" ]] || { echo "[error] MODEL_PATH not found: ${MODEL_PATH}" >&2; exit 1; }
[[ -f "${QUESTIONS_JSONL}" ]] || { echo "[error] QUESTIONS_JSONL not found: ${QUESTIONS_JSONL}" >&2; exit 1; }

echo "[info] Writing logs to: ${OUT_DIR}"
echo "[info] N=${N}, RUNS=${RUNS}, BASE_SEED=${BASE_SEED}"
echo

# 记录一个总览文件（可选）
SUMMARY_FILE="${OUT_DIR}/summary.txt"
: > "${SUMMARY_FILE}"

for i in $(seq 1 "${RUNS}"); do
  SEED=$((BASE_SEED + i))
  LOG_FILE="${OUT_DIR}/run_${i}_seed_${SEED}.log"

  echo "==================== RUN ${i}/${RUNS} | seed=${SEED} ====================" | tee -a "${SUMMARY_FILE}"
  echo "[log] ${LOG_FILE}" | tee -a "${SUMMARY_FILE}"

  # 跑你的 run_Table4.py，并把输出记录到 log
  python "${SCRIPT_DIR}/../run_Table4.py" \
    --data_root "${DATA_ROOT}" \
    --model_path "${MODEL_PATH}" \
    --questions_jsonl "${QUESTIONS_JSONL}" \
    --device "${DEVICE}" \
    --N "${N}" \
    --seed "${SEED}" \
    --use_gpt_tool_query \
    --use_gpt \
    2>&1 | tee "${LOG_FILE}"

  echo | tee -a "${SUMMARY_FILE}"
done

echo "[done] All runs finished. Summary: ${SUMMARY_FILE}"