#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KALM embedding (bi-encoder) evaluation for agent recommendation.

This mirrors the BGE/EasyRec evaluation entry points while using a
Sentence-Transformers bi-encoder (e.g., KaLM-Embedding/KaLM-embedding-
multilingual-mini-instruct-v2.5) to score query/agent pairs via dot
product or cosine similarity (when normalized).

Example:
  python infer_KALM.py \
    --data_root /path/to/benchmark \
    --model_dir /path/to/st-biencoder \
    --exp_name infer_kalm \
    --device cuda:0 \
    --encode_batch 64 \
    --eval_cand_size 1000 \
    --pos_topk 10 \
    --ks 10 \
    --max_len 384
"""

from __future__ import annotations

import argparse
import inspect
import os
from collections import defaultdict
from contextlib import nullcontext
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm

from agent_rec.config import EVAL_TOPK
from agent_rec.rerank_eval_utils import (
    accumulate_metrics,
    build_agent_text_cache,
    finalize_metrics,
    metric_template,
    metrics_from_hits,
    prepare_eval_items,
    sample_qids_by_part,
    select_eval_qids,
    topk_hits_from_scores,
)
from agent_rec.run_common import bootstrap_run
from utils import print_metrics_table


def parse_ks(arg: str) -> Tuple[int, ...]:
    return tuple(sorted({int(x) for x in arg.split(",") if x.strip()}))


def encode_texts(
    model: SentenceTransformer,
    texts: Iterable[str],
    *,
    batch_size: int,
    device: str,
    normalize: bool,
    use_amp: bool,
) -> np.ndarray:
    encode_kwargs = dict(
        batch_size=batch_size,
        convert_to_numpy=True,
        device=device,
        show_progress_bar=True,
        normalize_embeddings=normalize,
    )

    supports_use_amp = "use_amp" in inspect.signature(model.encode).parameters
    if supports_use_amp:
        encode_kwargs["use_amp"] = use_amp
        return model.encode(list(texts), **encode_kwargs)

    autocast_ctx = (
        torch.cuda.amp.autocast(dtype=torch.float16) if use_amp and torch.cuda.is_available() else nullcontext()
    )
    with autocast_ctx:
        return model.encode(list(texts), **encode_kwargs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument(
        "--exp_name",
        type=str,
        default="infer_kalm_embedding",
        help="Cache/log dir name; no training cache is saved but kept for naming consistency.",
    )
    ap.add_argument("--model_dir", type=str, required=True, help="Sentence-Transformers bi-encoder directory")
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--encode_batch", type=int, default=64)
    ap.add_argument("--eval_cand_size", type=int, default=1000)
    ap.add_argument(
        "--pos_topk",
        type=int,
        default=0,
        help="Positive cutoff per query. 0 = use per-part defaults (POS_TOPK_BY_PART).",
    )
    ap.add_argument("--ks", type=str, default=str(EVAL_TOPK))
    ap.add_argument("--max_len", type=int, default=384)
    ap.add_argument("--use_amp", type=int, default=1, help="1 to enable autocast on CUDA during encoding")
    ap.add_argument("--normalize", type=int, default=1, help="1 to L2-normalize embeddings before scoring")
    ap.add_argument("--seed", type=int, default=1234, help="Global seed for data prep and negatives.")
    ap.add_argument("--split_seed", type=int, default=42, help="Seed for stratified eval split to match baselines.")
    ap.add_argument("--valid_ratio", type=float, default=0.2, help="Portion of qids (with rankings) used for eval.")
    ap.add_argument("--sample_per_part", type=int, default=200, help="Eval qids sampled per part; 0 disables sampling.")
    ap.add_argument("--max_eval", type=int, default=0, help="Max number of eval queries. 0 = use all.")
    args = ap.parse_args()

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    ks = parse_ks(args.ks)
    if not ks:
        raise ValueError("--ks must provide at least one integer (e.g., 5,10,50)")

    # 1) Data/bootstrap
    boot = bootstrap_run(
        data_root=args.data_root,
        exp_name=args.exp_name,
        topk=EVAL_TOPK,
        seed=args.seed,
        with_tools=True,
    )
    eval_qids = select_eval_qids(
        boot.qids_in_rank,
        seed=args.split_seed,
        valid_ratio=args.valid_ratio,
        qid_to_part=boot.bundle.qid_to_part,
    )
    eval_qids = sample_qids_by_part(
        eval_qids,
        qid_to_part=boot.bundle.qid_to_part,
        per_part=args.sample_per_part,
        seed=args.seed,
    )
    agent_text_cache = build_agent_text_cache(boot.bundle.all_agents, boot.tools or {})

    items = prepare_eval_items(
        eval_qids=eval_qids,
        all_questions=boot.bundle.all_questions,
        all_agents=boot.bundle.all_agents,
        tools=boot.tools or {},
        all_rankings=boot.bundle.all_rankings,
        a_ids=boot.a_ids,
        seed=args.seed,
        cand_size=args.eval_cand_size,
        pos_topk=None if args.pos_topk <= 0 else args.pos_topk,
        qid_to_part=boot.bundle.qid_to_part,
        agent_text_cache=agent_text_cache,
    )
    if args.max_eval and len(items) > args.max_eval:
        items = items[: args.max_eval]
    print(f"Prepared {len(items)} eval items (valid_ratio={args.valid_ratio}, seed={args.seed}).")

    # 2) Model
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type == "cpu" and args.device != "cpu":
        print(f"[warn] CUDA not available, running on CPU instead of {args.device}.")

    model = SentenceTransformer(args.model_dir, device=str(device), trust_remote_code=True)
    if hasattr(model, "max_seq_length"):
        model.max_seq_length = args.max_len
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    use_amp = bool(args.use_amp) and device.type == "cuda"
    normalize = bool(args.normalize)

    # 3) Pre-encode corpus and queries
    agent_vecs = encode_texts(
        model,
        (agent_text_cache.get(aid, "") for aid in boot.a_ids),
        batch_size=args.encode_batch,
        device=str(device),
        normalize=normalize,
        use_amp=use_amp,
    )
    aid_to_vec = {aid: agent_vecs[i] for i, aid in enumerate(boot.a_ids)}

    q_texts = {it.qid: it.qtext for it in items}
    q_vecs_arr = encode_texts(
        model,
        [q_texts[qid] for qid in q_texts.keys()],
        batch_size=args.encode_batch,
        device=str(device),
        normalize=normalize,
        use_amp=use_amp,
    )
    q_vecs = {qid: q_vecs_arr[i] for i, qid in enumerate(q_texts.keys())}

    # 4) Evaluation
    agg = metric_template(ks)
    scored = 0
    skipped = 0
    part_aggs: Dict[str, Dict[int, Dict[str, float]]] = {}
    part_counts = defaultdict(int)
    ref_k = 10 if 10 in ks else max(ks)

    pbar = tqdm(items, desc="Evaluating (KALM Embedding)", dynamic_ncols=True)
    for idx, it in enumerate(pbar, start=1):
        qv = q_vecs.get(it.qid)
        cand_vecs: List[np.ndarray] = []
        cand_ids: List[str] = []
        for aid in it.cand_ids:
            vec = aid_to_vec.get(aid)
            if vec is None:
                continue
            cand_ids.append(aid)
            cand_vecs.append(vec)
        if qv is None or not cand_vecs:
            skipped += 1
            pbar.set_postfix({"done": scored, "skipped": skipped})
            continue

        scores = np.matmul(np.vstack(cand_vecs), qv)
        _, bin_hits = topk_hits_from_scores(scores, cand_ids, it.rel_set, ks)
        per_k = metrics_from_hits(bin_hits, len(it.rel_set), ks)
        accumulate_metrics(agg, per_k, ks)
        part = boot.bundle.qid_to_part.get(it.qid, "Unknown")
        if part not in part_aggs:
            part_aggs[part] = metric_template(ks)
        accumulate_metrics(part_aggs[part], per_k, ks)
        part_counts[part] += 1
        scored += 1

        ref = agg[ref_k]
        denom = max(scored, 1)
        pbar.set_postfix(
            {
                "done": scored,
                "skipped": skipped,
                f"P@{ref_k}": f"{(ref['P'] / denom):.4f}",
                f"nDCG@{ref_k}": f"{(ref['nDCG'] / denom):.4f}",
                f"MRR@{ref_k}": f"{(ref['MRR'] / denom):.4f}",
            }
        )

    metrics = finalize_metrics(agg, scored, ks)
    print_metrics_table("KALM Embedding eval", metrics, ks=ks, filename=args.exp_name)
    seen_parts = {"PartI", "PartII", "PartIII"}
    for part in ["PartI", "PartII", "PartIII"]:
        cnt = part_counts.get(part, 0)
        if cnt <= 0:
            continue
        m_part = finalize_metrics(part_aggs[part], cnt, ks)
        print_metrics_table(f"KALM Embedding eval {part}", m_part, ks=ks, filename=args.exp_name)
    for part in sorted(part_aggs):
        if part in seen_parts:
            continue
        cnt = part_counts.get(part, 0)
        if cnt <= 0:
            continue
        m_part = finalize_metrics(part_aggs[part], cnt, ks)
        print_metrics_table(f"KALM Embedding eval {part}", m_part, ks=ks, filename=args.exp_name)


if __name__ == "__main__":
    main()
