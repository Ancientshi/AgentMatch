# -*- coding: utf-8 -*-

POS_TOPK_BY_PART = {
    "PartI": 10,
    "PartII": 1,
    "PartIII": 5,
}

POS_TOPK = POS_TOPK_BY_PART["PartIII"]


def pos_topk_for_part(part: str | None) -> int:
    return POS_TOPK_BY_PART.get(part or "", POS_TOPK)


def pos_topk_for_qid(qid: str, qid_to_part: dict | None) -> int:
    part = (qid_to_part or {}).get(qid)
    return pos_topk_for_part(part)

EVAL_TOPK = 10

TFIDF_MAX_FEATURES = 5000

EPS = 1e-8
