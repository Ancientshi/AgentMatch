#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BGE cross-encoder reranker evaluation for agent recommendation.

This integrates the standalone eval script into the repo's data/loading
utilities so it can share caching, splitting, and logging conventions.

Example:
  python infer_BGE.py \
    --data_root /path/to/benchmark \
    --model_dir /path/to/reranker \
    --model_name BAAI/bge-reranker-base \
    --peft 0 \
    --device cuda:0 \
    --eval_cand_size 1000 \
    --pos_topk 10 \
    --max_len 192 \
    --rerank_batch 2048 \
    --ks 10
"""

from __future__ import annotations

import argparse
import os
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

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


def batched_tokenize_and_score(
    *,
    model: torch.nn.Module,
    tokenizer,
    device: torch.device,
    qtext: str,
    doc_texts: List[str],
    max_len: int,
    rerank_batch: int,
    use_amp: bool,
) -> np.ndarray:
    scores: List[np.ndarray] = []
    for i in range(0, len(doc_texts), rerank_batch):
        batch_docs = doc_texts[i : i + rerank_batch]
        enc = tokenizer(
            [qtext] * len(batch_docs),
            batch_docs,
            truncation=True,
            padding="longest",
            max_length=max_len,
            return_tensors="pt",
        )
        enc = {k: v.to(device, non_blocking=True) for k, v in enc.items()}
        if use_amp and device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                out = model(**enc)
        else:
            out = model(**enc)
        s = out.logits.squeeze(-1).float().detach().cpu().numpy()
        scores.append(s)
    if not scores:
        return np.zeros((0,), dtype=np.float32)
    return np.concatenate(scores, axis=0)


def parse_ks(arg: str) -> Tuple[int, ...]:
    return tuple(sorted({int(x) for x in arg.split(",") if x.strip()}))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument(
        "--exp_name",
        type=str,
        default="infer_bge_reranker",
        help="Cache/log dir name; no training cache is saved but kept for naming consistency.",
    )
    ap.add_argument("--model_dir", type=str, required=True, help="Saved reranker directory (HF or PEFT adapter).")
    ap.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="HF base model name when --peft=0; if omitted, --model_dir is used directly.",
    )
    ap.add_argument("--peft", type=int, default=0, help="1 to load PEFT adapter if available, 0 to load raw HF model.")
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--eval_cand_size", type=int, default=1000)
    ap.add_argument(
        "--pos_topk",
        type=int,
        default=0,
        help="Positive cutoff per query. 0 = use per-part defaults (POS_TOPK_BY_PART).",
    )
    ap.add_argument("--max_len", type=int, default=192)
    ap.add_argument("--rerank_batch", type=int, default=256)
    ap.add_argument("--seed", type=int, default=1234, help="Global seed for data prep and negatives.")
    ap.add_argument("--split_seed", type=int, default=42, help="Seed for stratified eval split to match baselines.")
    ap.add_argument("--ks", type=str, default=str(EVAL_TOPK))
    ap.add_argument("--use_amp", type=int, default=1, help="1 to enable autocast(float16) on CUDA")
    ap.add_argument("--valid_ratio", type=float, default=0.2, help="Portion of qids (with rankings) used for eval.")
    ap.add_argument("--max_eval", type=int, default=0, help="Max number of eval queries. 0 = use all.")
    args = ap.parse_args()

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

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
        per_part=200,
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

    # 2) Model/tokenizer
    use_fast = os.environ.get("HF_NO_FAST_TOKENIZER", "0") != "1"
    tok_src = args.model_name if (args.peft == 0 and args.model_name) else args.model_dir
    tokenizer = AutoTokenizer.from_pretrained(tok_src, use_fast=use_fast)

    if args.peft == 0:
        mdl_src = args.model_name if args.model_name else args.model_dir
        model = AutoModelForSequenceClassification.from_pretrained(mdl_src, num_labels=1)
    else:
        try:
            model = AutoModelForSequenceClassification.from_pretrained(args.model_dir, num_labels=1)
        except Exception:
            from peft import PeftConfig, PeftModel

            peft_cfg = PeftConfig.from_pretrained(args.model_dir)
            base = AutoModelForSequenceClassification.from_pretrained(
                peft_cfg.base_model_name_or_path, num_labels=1
            )
            model = PeftModel.from_pretrained(base, args.model_dir)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type == "cpu" and args.device != "cpu":
        print(f"[warn] CUDA not available, running on CPU instead of {args.device}.")
    model.to(device)
    model.eval()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # 3) Evaluation
    agg = metric_template(ks)
    part_aggs: Dict[str, Dict[int, Dict[str, float]]] = {}
    part_counts = defaultdict(int)
    ref_k = 10 if 10 in ks else max(ks)
    use_amp = bool(args.use_amp)

    pbar = tqdm(items, desc="Evaluating (BGE Reranker)", dynamic_ncols=True)
    for idx, it in enumerate(pbar, start=1):
        scores = batched_tokenize_and_score(
            model=model,
            tokenizer=tokenizer,
            device=device,
            qtext=it.qtext,
            doc_texts=it.doc_texts,
            max_len=args.max_len,
            rerank_batch=args.rerank_batch,
            use_amp=use_amp,
        )
        _, bin_hits = topk_hits_from_scores(scores, it.cand_ids, it.rel_set, ks)
        per_k = metrics_from_hits(bin_hits, len(it.rel_set), ks)
        accumulate_metrics(agg, per_k, ks)
        part = boot.bundle.qid_to_part.get(it.qid, "Unknown")
        if part not in part_aggs:
            part_aggs[part] = metric_template(ks)
        accumulate_metrics(part_aggs[part], per_k, ks)
        part_counts[part] += 1

        ref = agg[ref_k]
        pbar.set_postfix(
            {
                "done": idx,
                f"P@{ref_k}": f"{(ref['P'] / idx):.4f}",
                f"nDCG@{ref_k}": f"{(ref['nDCG'] / idx):.4f}",
                f"MRR@{ref_k}": f"{(ref['MRR'] / idx):.4f}",
            }
        )

    metrics = finalize_metrics(agg, len(items), ks)
    print_metrics_table("BGE-Reranker eval", metrics, ks=ks, filename=args.exp_name)
    seen_parts = {"PartI", "PartII", "PartIII"}
    for part in ["PartI", "PartII", "PartIII"]:
        cnt = part_counts.get(part, 0)
        if cnt <= 0:
            continue
        m_part = finalize_metrics(part_aggs[part], cnt, ks)
        print_metrics_table(f"BGE-Reranker eval {part}", m_part, ks=ks, filename=args.exp_name)
    for part in sorted(part_aggs):
        if part in seen_parts:
            continue
        cnt = part_counts.get(part, 0)
        if cnt <= 0:
            continue
        m_part = finalize_metrics(part_aggs[part], cnt, ks)
        print_metrics_table(f"BGE-Reranker eval {part}", m_part, ks=ks, filename=args.exp_name)


if __name__ == "__main__":
    main()
