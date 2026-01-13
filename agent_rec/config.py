# -*- coding: utf-8 -*-

# 只把排名前K的 agent 当做正样本（按 Part 区分）
POS_TOPK_BY_PART = {
    "PartI": 10,
    "PartII": 1,
    "PartIII": 5,
}

# 默认正样本 K（兼容旧逻辑）
POS_TOPK = POS_TOPK_BY_PART["PartIII"]


def pos_topk_for_part(part: str | None) -> int:
    return POS_TOPK_BY_PART.get(part or "", POS_TOPK)


def pos_topk_for_qid(qid: str, qid_to_part: dict | None) -> int:
    part = (qid_to_part or {}).get(qid)
    return pos_topk_for_part(part)

# 评测固定 top10
EVAL_TOPK = 10

# TF-IDF 默认 max features
TFIDF_MAX_FEATURES = 5000

# 数值稳定
EPS = 1e-8
