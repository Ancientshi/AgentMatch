import json
import os
import random
import zlib
from dataclasses import dataclass
from typing import Callable, Iterable, List, Tuple

import numpy as np
import torch

from agent_rec.config import EVAL_TOPK
from agent_rec.data import collect_data, dataset_signature, ensure_cache_dir, load_tools, qids_with_rankings

DEFAULT_PARTS = ("PartI", "PartII", "PartIII")


def set_global_seed(seed: int = 1234) -> None:
    print(f"[seed] Setting global seed to {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def warn_if_topk_diff(topk: int, expected: int = EVAL_TOPK) -> None:
    if topk != expected:
        print(f"[warn] You set --topk={topk}, but protocol suggests fixed top10. Proceeding.")


def load_data_bundle(
    data_root: str,
    parts: Iterable[str] = DEFAULT_PARTS,
    *,
    with_tools: bool = False,
):
    bundle = collect_data(data_root, parts=list(parts))
    tools = load_tools(data_root) if with_tools else None
    return bundle, tools


def summarize_bundle(bundle, tools=None) -> None:
    if tools is None:
        print(
            f"Loaded {len(bundle.all_agents)} agents, {len(bundle.all_questions)} questions, "
            f"{len(bundle.all_rankings)} ranked entries."
        )
        return
    print(
        f"Loaded {len(bundle.all_agents)} agents, {len(bundle.all_questions)} questions, "
        f"{len(bundle.all_rankings)} ranked entries, {len(tools)} tools."
    )


def build_id_maps(all_questions, all_agents):
    q_ids = sorted(all_questions.keys())
    a_ids = sorted(all_agents.keys())
    qid2idx = {qid: i for i, qid in enumerate(q_ids)}
    aid2idx = {aid: i for i, aid in enumerate(a_ids)}
    return q_ids, a_ids, qid2idx, aid2idx


def qids_with_rankings_and_log(q_ids, all_rankings):
    qids_in_rank = qids_with_rankings(q_ids, all_rankings)
    print(f"Questions with rankings: {len(qids_in_rank)} / {len(q_ids)}")
    return qids_in_rank


def training_cache_paths(cache_dir: str) -> Tuple[str, str, str, str]:
    return (
        os.path.join(cache_dir, "train_qids.json"),
        os.path.join(cache_dir, "valid_qids.json"),
        os.path.join(cache_dir, "pairs_train.npy"),
        os.path.join(cache_dir, "train_cache_meta.json"),
    )


def shared_cache_root(data_root: str) -> str:
    root = os.path.join(data_root, ".cache", "shared")
    os.makedirs(root, exist_ok=True)
    return root


def shared_cache_dir(data_root: str, *parts: str) -> str:
    root = shared_cache_root(data_root)
    d = os.path.join(root, *parts)
    os.makedirs(d, exist_ok=True)
    return d


def cache_key_from_meta(meta: dict) -> str:
    payload = json.dumps(meta, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return f"{(zlib.crc32(payload) & 0xFFFFFFFF):08x}"


def cache_key_from_text(text: str) -> str:
    payload = text.encode("utf-8")
    return f"{(zlib.crc32(payload) & 0xFFFFFFFF):08x}"


@dataclass
class RunBootstrap:
    bundle: object
    tools: object
    q_ids: list
    a_ids: list
    qid2idx: dict
    aid2idx: dict
    qids_in_rank: list
    data_sig: str
    exp_cache_dir: str


def bootstrap_run(
    data_root: str,
    exp_name: str,
    *,
    topk: int,
    seed: int = 1234,
    with_tools: bool = False,
    parts: Iterable[str] = DEFAULT_PARTS,
) -> RunBootstrap:
    """Standardized bootstrap for run_*.py entrypoints.

    Handles seed setting, data loading, summary logging, id mapping,
    ranking filtering, dataset signature calculation, and experiment cache dir creation.
    """

    warn_if_topk_diff(topk)
    set_global_seed(seed)

    bundle, tools = load_data_bundle(data_root, parts=list(parts), with_tools=with_tools)
    summarize_bundle(bundle, tools)

    q_ids, a_ids, qid2idx, aid2idx = build_id_maps(bundle.all_questions, bundle.all_agents)
    qids_in_rank = qids_with_rankings_and_log(q_ids, bundle.all_rankings)
    data_sig = dataset_signature(qids_in_rank, a_ids, {k: bundle.all_rankings[k] for k in qids_in_rank})
    exp_cache_dir = ensure_cache_dir(data_root, exp_name)

    return RunBootstrap(
        bundle=bundle,
        tools=tools,
        q_ids=q_ids,
        a_ids=a_ids,
        qid2idx=qid2idx,
        aid2idx=aid2idx,
        qids_in_rank=qids_in_rank,
        data_sig=data_sig,
        exp_cache_dir=exp_cache_dir,
    )


def load_or_build_training_cache(
    cache_dir: str,
    rebuild: int,
    want_meta: dict,
    build_cache_fn: Callable[[], Tuple[list, list, np.ndarray]],
    *,
    save_message: str = "train/valid/pairs",
):
    train_qids_path, valid_qids_path, pairs_path, meta_path = training_cache_paths(cache_dir)
    use_cache = (
        os.path.exists(train_qids_path)
        and os.path.exists(valid_qids_path)
        and os.path.exists(pairs_path)
        and os.path.exists(meta_path)
        and rebuild == 0
    )
    if use_cache:
        with open(train_qids_path, "r", encoding="utf-8") as f:
            train_qids = json.load(f)
        with open(valid_qids_path, "r", encoding="utf-8") as f:
            valid_qids = json.load(f)
        pairs_idx_np = np.load(pairs_path)
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        if meta != want_meta:
            print("[cache] training cache meta mismatch, rebuilding...")
            use_cache = False
        else:
            print(f"[cache] loaded {save_message} from {cache_dir}")

    if not use_cache:
        train_qids, valid_qids, pairs_idx_np = build_cache_fn()
        with open(train_qids_path, "w", encoding="utf-8") as f:
            json.dump(train_qids, f, ensure_ascii=False)
        with open(valid_qids_path, "w", encoding="utf-8") as f:
            json.dump(valid_qids, f, ensure_ascii=False)
        np.save(pairs_path, pairs_idx_np)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(want_meta, f, ensure_ascii=False, sort_keys=True)
        print(f"[cache] saved {save_message} to {cache_dir}")

    return train_qids, valid_qids, pairs_idx_np


def build_pos_pairs(
    rankings: dict,
    *,
    qid_to_part: dict,
    pos_topk_by_part: dict,
    pos_topk_default: int,
    rng_seed: int = 42,
) -> List[Tuple[str, str]]:
    """Build (qid, pos_aid) pairs using part-aware top-k cutoff."""
    rnd = random.Random(rng_seed)
    pairs: List[Tuple[str, str]] = []
    for qid, ranked in rankings.items():
        k = pos_topk_by_part.get(qid_to_part.get(qid, ""), pos_topk_default)
        pos_list = ranked[:k] if ranked else []
        for pos_a in pos_list:
            pairs.append((qid, pos_a))
    rnd.shuffle(pairs)
    return pairs
