#!/usr/bin/env bash
set -euo pipefail

# Convenience wrapper for the generative structured recommender.
# Environment variables allow quick overrides without long CLI strings.
#
# Required:
#   DATA_ROOT=/path/to/dataset_root
#
# Optional:
#   QUERY="用户的查询文本"
#   TOP_K=3
#   WITH_METADATA=1
#   EXPORT_PAIRS=/tmp/generative_pairs.jsonl
#   MAX_EXAMPLES=5000
#   TOOL_SEP_TOKEN="<TOOL_SEP>"
#   END_TOKEN="<SPECIAL_END>"
#   MAX_TOOLS=4
#   TFIDF_MAX_FEATURES=5000
#
# Examples:
#   DATA_ROOT=/data/benchmark QUERY="如何写个爬虫？" TOP_K=3 WITH_METADATA=1 ./scripts/run_generative.sh
#   DATA_ROOT=/data/benchmark EXPORT_PAIRS=/tmp/pairs.jsonl MAX_EXAMPLES=8000 ./scripts/run_generative.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../common_env.sh"

QUERY=${QUERY:-}
TOP_K=${TOP_K:-1}
WITH_METADATA=${WITH_METADATA:-0}
EXPORT_PAIRS=${EXPORT_PAIRS:-}
MAX_EXAMPLES=${MAX_EXAMPLES:-0}
TOOL_SEP_TOKEN=${TOOL_SEP_TOKEN:-"<TOOL_SEP>"}
END_TOKEN=${END_TOKEN:-"<SPECIAL_END>"}
MAX_TOOLS=${MAX_TOOLS:-4}
TFIDF_MAX_FEATURES=${TFIDF_MAX_FEATURES:-5000}

python "$SCRIPT_DIR/../../run_generative.py" \
  --data_root "${DATA_ROOT}" \
  --query "${QUERY}" \
  --top_k "${TOP_K}" \
  --with_metadata "${WITH_METADATA}" \
  --tool_sep_token "${TOOL_SEP_TOKEN}" \
  --end_token "${END_TOKEN}" \
  --max_tools "${MAX_TOOLS}" \
  --tfidf_max_features "${TFIDF_MAX_FEATURES}" \
  ${EXPORT_PAIRS:+--export_pairs "${EXPORT_PAIRS}"} \
  ${MAX_EXAMPLES:+--max_examples "${MAX_EXAMPLES}"}
