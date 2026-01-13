#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import random
import zlib
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_cache_dir(data_root: str, exp_name: str) -> str:
    d = os.path.join(data_root, f".cache/{exp_name}")
    os.makedirs(d, exist_ok=True)
    return d


def dataset_signature(q_ids: List[str], a_ids: List[str], rankings: Dict[str, List[str]]) -> str:
    payload = {
        "q_ids": sorted(q_ids),
        "a_ids": sorted(a_ids),
        "rankings": {k: rankings[k] for k in sorted(rankings.keys())},
    }
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return f"{(zlib.crc32(blob) & 0xFFFFFFFF):08x}"


@dataclass
class DatasetBundle:
    all_agents: Dict[str, dict]
    all_questions: Dict[str, dict]
    all_rankings: Dict[str, List[str]]
    qid_to_part: Dict[str, str]
    aid_to_part: Dict[str, str]


def collect_data(data_root: str, parts: Optional[List[str]] = None) -> DatasetBundle:
    """Reads PartI/II/III agents/questions/rankings and merges them,
    while tracking qid/aid -> part mapping for per-part evaluation."""
    if parts is None:
        parts = ["PartI", "PartII", "PartIII"]

    all_agents: Dict[str, dict] = {}
    all_questions: Dict[str, dict] = {}
    all_rankings: Dict[str, List[str]] = {}

    qid_to_part: Dict[str, str] = {}
    aid_to_part: Dict[str, str] = {}

    for part in parts:
        agents_path = os.path.join(data_root, part, "agents", "merge.json")
        questions_path = os.path.join(data_root, part, "questions", "merge.json")
        rankings_path = os.path.join(data_root, part, "rankings", "merge.json")

        agents = load_json(agents_path)
        questions = load_json(questions_path)
        rankings = load_json(rankings_path)["rankings"]

        for aid, aobj in agents.items():
            all_agents[aid] = aobj
            aid_to_part[aid] = part

        for qid, qobj in questions.items():
            all_questions[qid] = qobj
            qid_to_part[qid] = part

        for qid, ranked in rankings.items():
            all_rankings[qid] = ranked

    return DatasetBundle(
        all_agents=all_agents,
        all_questions=all_questions,
        all_rankings=all_rankings,
        qid_to_part=qid_to_part,
        aid_to_part=aid_to_part,
    )


def load_tools(data_root: str) -> Dict[str, dict]:
    tools_path = os.path.join(data_root, "Tools", "merge.json")
    if os.path.exists(tools_path):
        return load_json(tools_path)
    return {}

def load_LLMs(data_root: str) -> Dict[str, dict]:
    llms_path = os.path.join(data_root, "LLMs", "merge.json")
    if os.path.exists(llms_path):
        return load_json(llms_path)
    return {}


def qids_with_rankings(q_ids: List[str], rankings: Dict[str, List[str]]) -> List[str]:
    return [qid for qid in q_ids if qid in rankings]


def stratified_train_valid_split(
    qids: List[str],
    qid_to_part: Dict[str, str],
    valid_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[List[str], List[str]]:
    """Stratified split by part."""
    rng = random.Random(seed)

    part_to_qids: Dict[str, List[str]] = {}
    for qid in qids:
        part = qid_to_part.get(qid, "Unknown")
        part_to_qids.setdefault(part, []).append(qid)

    train_qids, valid_qids = [], []
    for part, lst in part_to_qids.items():
        lst = list(lst)
        rng.shuffle(lst)
        n_valid = int(len(lst) * valid_ratio)
        if len(lst) >= 5 and n_valid == 0:
            n_valid = 1
        v = lst[:n_valid]
        t = lst[n_valid:]
        valid_qids.extend(v)
        train_qids.extend(t)

    rng.shuffle(train_qids)
    rng.shuffle(valid_qids)
    return train_qids, valid_qids


def build_training_pairs(
    rankings_train: Dict[str, List[str]],
    all_agent_ids: List[str],
    *,
    qid_to_part: Optional[Dict[str, str]] = None,
    pos_topk_by_part: Optional[Dict[str, int]] = None,
    pos_topk_default: int | None = None,
    neg_per_pos: int = 1,
    rng_seed: int = 42,
) -> List[Tuple[str, str, str]]:
    """Returns [(qid, pos_aid, neg_aid)]"""
    rnd = random.Random(rng_seed)
    pairs: List[Tuple[str, str, str]] = []
    all_agent_set = set(all_agent_ids)
    from agent_rec.config import POS_TOPK, POS_TOPK_BY_PART

    topk_map = pos_topk_by_part or POS_TOPK_BY_PART
    topk_default = POS_TOPK if pos_topk_default is None else pos_topk_default

    for qid, ranked in rankings_train.items():
        if qid_to_part is not None:
            k = topk_map.get(qid_to_part.get(qid), topk_default)
        else:
            k = topk_default
        pos = [aid for aid in ranked[:k] if aid in all_agent_set]
        if not pos:
            continue
        pos_set = set(pos)
        neg_pool = list(all_agent_set - pos_set) or list(all_agent_ids)

        for pos_a in pos:
            for _ in range(neg_per_pos):
                neg_a = rnd.choice(neg_pool)
                pairs.append((qid, pos_a, neg_a))

    return pairs
