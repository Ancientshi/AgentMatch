#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Threaded evaluation via EasyRec HTTP reranker service.

Example:
  python infer_EasyRec.py \
    --data_root /path/to/benchmark \
    --service_url http://127.0.0.1:8500/compute_scores \
    --pos_topk 10 \
    --ks 10 \
    --rerank_batch 32 \
    --timeout 300 \
    --max_workers 16
"""

from __future__ import annotations

import argparse
import os
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple

import numpy as np
import requests
from requests.adapters import HTTPAdapter
from tqdm.auto import tqdm
from urllib3.util.retry import Retry

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


def _build_session(retries: int, backoff: float, pool_maxsize: int) -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        status=retries,
        backoff_factor=backoff,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET", "POST"]),
    )
    adapter = HTTPAdapter(max_retries=retry, pool_maxsize=pool_maxsize)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


_thread_local = threading.local()


def _get_session(retries: int, backoff: float, pool_maxsize: int) -> requests.Session:
    sess: requests.Session = getattr(_thread_local, "session", None)
    if sess is None:
        sess = _build_session(retries, backoff, pool_maxsize)
        _thread_local.session = sess
    return sess


def score_by_service(
    *,
    service_url: str,
    query_text: str,
    doc_texts: List[str],
    timeout: int,
    rerank_batch: int,
    retries: int,
    backoff: float,
    pool_maxsize: int,
) -> np.ndarray:
    if not doc_texts:
        return np.zeros((0,), dtype=np.float32)

    session = _get_session(retries, backoff, pool_maxsize)
    scores_all: List[float] = []
    for i in range(0, len(doc_texts), rerank_batch):
        batch_docs = doc_texts[i : i + rerank_batch]
        payload = {"query": query_text, "documents": batch_docs}
        resp = session.post(service_url, json=payload, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        if "scores" not in data or not isinstance(data["scores"], list):
            raise RuntimeError(f"Bad response from service: {data}")
        scores_all.extend([float(s) for s in data["scores"]])

    if len(scores_all) != len(doc_texts):
        raise RuntimeError(f"Score length mismatch: got {len(scores_all)} for {len(doc_texts)} docs")
    return np.asarray(scores_all, dtype=np.float32)


def parse_ks(arg: str) -> Tuple[int, ...]:
    return tuple(sorted({int(x) for x in arg.split(",") if x.strip()}))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument(
        "--exp_name",
        type=str,
        default="infer_easyrec",
        help="Cache/log dir name; kept for naming consistency (no training cache saved).",
    )
    ap.add_argument("--service_url", type=str, required=True, help="EasyRec service endpoint, e.g., http://127.0.0.1:8500/compute_scores")
    ap.add_argument("--eval_cand_size", type=int, default=1000)
    ap.add_argument(
        "--pos_topk",
        type=int,
        default=0,
        help="Positive cutoff per query. 0 = use per-part defaults (POS_TOPK_BY_PART).",
    )
    ap.add_argument("--ks", type=str, default=str(EVAL_TOPK))
    ap.add_argument("--seed", type=int, default=1234, help="Global seed for data prep and negatives.")
    ap.add_argument("--split_seed", type=int, default=42, help="Seed for stratified eval split to match baselines.")
    ap.add_argument("--valid_ratio", type=float, default=0.2, help="Portion of qids (with rankings) used for eval.")
    ap.add_argument("--rerank_batch", type=int, default=256, help="Max documents per HTTP request.")
    ap.add_argument("--timeout", type=int, default=300, help="HTTP timeout (seconds)")
    ap.add_argument("--max_eval", type=int, default=0, help="Max number of eval queries. 0 = use all.")

    ap.add_argument("--max_workers", type=int, default=16, help="Thread pool size")
    ap.add_argument("--http_retries", type=int, default=3, help="HTTP retry count")
    ap.add_argument("--http_backoff", type=float, default=0.3, help="Exponential backoff factor")
    ap.add_argument("--pool_maxsize", type=int, default=64, help="HTTPAdapter pool size")
    args = ap.parse_args()

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

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

    # 2) Threaded evaluation via HTTP
    agg = metric_template(ks)
    part_aggs: Dict[str, Dict[int, Dict[str, float]]] = {}
    part_counts = defaultdict(int)
    ref_k = 10 if 10 in ks else max(ks)
    lock = threading.Lock()

    def _worker(it, part: str) -> Tuple[str, Dict[int, Dict[str, float]]]:
        scores = score_by_service(
            service_url=args.service_url,
            query_text=it.qtext,
            doc_texts=it.doc_texts,
            timeout=args.timeout,
            rerank_batch=args.rerank_batch,
            retries=args.http_retries,
            backoff=args.http_backoff,
            pool_maxsize=args.pool_maxsize,
        )
        _, bin_hits = topk_hits_from_scores(scores, it.cand_ids, it.rel_set, ks)
        return part, metrics_from_hits(bin_hits, len(it.rel_set), ks)

    with ThreadPoolExecutor(max_workers=args.max_workers) as ex, tqdm(
        total=len(items), desc="Evaluating (EasyRec HTTP)", dynamic_ncols=True
    ) as pbar:
        futures = [ex.submit(_worker, it, boot.bundle.qid_to_part.get(it.qid, "Unknown")) for it in items]
        for done_idx, fut in enumerate(as_completed(futures), start=1):
            part, res = fut.result()
            with lock:
                accumulate_metrics(agg, res, ks)
                if part not in part_aggs:
                    part_aggs[part] = metric_template(ks)
                accumulate_metrics(part_aggs[part], res, ks)
                part_counts[part] += 1
            ref = agg[ref_k]
            pbar.update(1)
            pbar.set_postfix(
                {
                    "done": done_idx,
                    f"P@{ref_k}": f"{(ref['P'] / done_idx):.4f}",
                    f"nDCG@{ref_k}": f"{(ref['nDCG'] / done_idx):.4f}",
                    f"MRR@{ref_k}": f"{(ref['MRR'] / done_idx):.4f}",
                }
            )

    metrics = finalize_metrics(agg, len(items), ks)
    print_metrics_table("EasyRec HTTP eval", metrics, ks=ks, filename=args.exp_name)
    seen_parts = {"PartI", "PartII", "PartIII"}
    for part in ["PartI", "PartII", "PartIII"]:
        cnt = part_counts.get(part, 0)
        if cnt <= 0:
            continue
        m_part = finalize_metrics(part_aggs[part], cnt, ks)
        print_metrics_table(f"EasyRec HTTP eval {part}", m_part, ks=ks, filename=args.exp_name)
    for part in sorted(part_aggs):
        if part in seen_parts:
            continue
        cnt = part_counts.get(part, 0)
        if cnt <= 0:
            continue
        m_part = finalize_metrics(part_aggs[part], cnt, ks)
        print_metrics_table(f"EasyRec HTTP eval {part}", m_part, ks=ks, filename=args.exp_name)


if __name__ == "__main__":
    main()
