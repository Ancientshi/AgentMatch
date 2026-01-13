#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Two-Tower BGE Agent Recommender (InfoNCE) — OOM-safe version.

Key points:
1) Keep embedding matrices on CPU; move ONLY current batch to GPU.
2) Evaluation & inference are chunked over agents (no full-matrix move to GPU).
3) --eval_chunk controls agent-encoding batch size (default 8192).
4) --amp optionally enables autocast(bfloat16) on CUDA.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from contextlib import nullcontext
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import torch
from tqdm.auto import tqdm

from agent_rec.cli_common import add_shared_training_args
from agent_rec.config import EVAL_TOPK, POS_TOPK, POS_TOPK_BY_PART
from agent_rec.data import stratified_train_valid_split
from agent_rec.eval import split_eval_qids_by_part
from agent_rec.features import (
    build_agent_content_view,
    build_twotower_bge_feature_cache,
    feature_cache_exists,
    load_feature_cache,
    save_feature_cache,
)
from agent_rec.models.two_tower import TwoTowerTFIDF
from agent_rec.run_common import (
    bootstrap_run,
    cache_key_from_meta,
    cache_key_from_text,
    build_pos_pairs,
    load_or_build_training_cache,
    shared_cache_dir,
)
from utils import print_metrics_table


def info_nce_loss(qe: torch.Tensor, ae: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    logits = qe @ ae.t()
    labels = torch.arange(qe.size(0), device=qe.device)
    return torch.nn.functional.cross_entropy(logits / temperature, labels)


@torch.no_grad()
def evaluate_sampled_twotower(
    encoder: TwoTowerTFIDF,
    Q_cpu: np.ndarray,
    A_cpu: np.ndarray,
    qid2idx: Dict[str, int],
    a_ids: List[str],
    all_rankings: Dict[str, List[str]],
    eval_qids: List[str],
    device: torch.device,
    ks: Tuple[int, ...] = (5, 10, 50),
    cand_size: int = 100,
    rng_seed: int = 123,
    amp: bool = False,
    qid_to_part: Dict[str, str] | None = None,
    pos_topk_by_part: Dict[str, int] = POS_TOPK_BY_PART,
    pos_topk_default: int = POS_TOPK,
) -> Dict[int, Dict[str, float]]:
    max_k = max(ks)
    aid2idx = {aid: i for i, aid in enumerate(a_ids)}
    agg = {k: {"P": 0.0, "R": 0.0, "F1": 0.0, "Hit": 0.0, "nDCG": 0.0, "MRR": 0.0} for k in ks}
    cnt = 0
    skipped = 0
    ref_k = 10 if 10 in ks else max_k
    all_agent_set = set(a_ids)

    pbar = tqdm(eval_qids, desc="Evaluating (sampled)", leave=True, dynamic_ncols=True)
    for qid in pbar:
        k = pos_topk_by_part.get(qid_to_part.get(qid, ""), pos_topk_default) if qid_to_part else pos_topk_default
        gt_list = [aid for aid in all_rankings.get(qid, [])[:k] if aid in aid2idx]
        if not gt_list:
            skipped += 1
            pbar.set_postfix({"done": cnt, "skipped": skipped})
            continue
        rel_set = set(gt_list)
        neg_pool = list(all_agent_set - rel_set)

        rnd = random.Random((hash(qid) ^ (rng_seed * 16777619)) & 0xFFFFFFFF)
        need_neg = max(0, cand_size - len(gt_list))
        cand_ids = gt_list + (
            rnd.sample(neg_pool, min(need_neg, len(neg_pool))) if need_neg > 0 and neg_pool else []
        )

        qi = qid2idx[qid]
        qv = torch.from_numpy(Q_cpu[qi : qi + 1]).to(device)
        cand_idx = [aid2idx[a] for a in cand_ids]
        av = torch.from_numpy(A_cpu[cand_idx]).to(device)
        q_idx_t = torch.tensor([qi], dtype=torch.long, device=device)

        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if (amp and device.type == "cuda")
            else nullcontext()
        )
        with autocast_ctx:
            qe = encoder.encode_q(qv, q_idx=q_idx_t)
            ae = encoder.encode_a(av, torch.tensor(cand_idx, device=device, dtype=torch.long))
            scores = (qe @ ae.t()).float().squeeze(0).cpu().numpy()

        order = np.argsort(-scores)[:max_k]
        pred_ids = [cand_ids[i] for i in order]

        bin_hits = [1 if aid in rel_set else 0 for aid in pred_ids]
        num_rel = len(rel_set)
        for k in ks:
            top = bin_hits[:k]
            Hk = sum(top)
            P = Hk / float(k)
            R = Hk / float(num_rel)
            F1 = (2 * P * R) / (P + R) if (P + R) else 0.0
            Hit = 1.0 if Hk > 0 else 0.0
            dcg = sum(1.0 / math.log2(i + 2.0) for i, h in enumerate(top) if h)
            ideal = min(k, num_rel)
            idcg = sum(1.0 / math.log2(i + 2.0) for i in range(ideal)) if ideal else 0.0
            nDCG = (dcg / idcg) if idcg > 0 else 0.0
            rr = 0.0
            for i, h in enumerate(top):
                if h:
                    rr = 1.0 / float(i + 1)
                    break
            agg[k]["P"] += P
            agg[k]["R"] += R
            agg[k]["F1"] += F1
            agg[k]["Hit"] += Hit
            agg[k]["nDCG"] += nDCG
            agg[k]["MRR"] += rr

        cnt += 1
        ref = agg[ref_k]
        pbar.set_postfix(
            {
                "done": cnt,
                "skipped": skipped,
                f"P@{ref_k}": f"{(ref['P'] / cnt):.4f}",
                f"nDCG@{ref_k}": f"{(ref['nDCG'] / cnt):.4f}",
                f"MRR@{ref_k}": f"{(ref['MRR'] / cnt):.4f}",
                "Ncand": len(cand_ids),
            }
        )

    if cnt == 0:
        return {k: {m: 0.0 for m in ["P", "R", "F1", "Hit", "nDCG", "MRR"]} for k in ks}

    for k in ks:
        for m in agg[k]:
            agg[k][m] /= cnt
    return agg


def main() -> None:
    parser = argparse.ArgumentParser()
    add_shared_training_args(
        parser,
        exp_name_default="two_tower_bge",
        device_default="cpu",
        epochs_default=3,
        batch_size_default=512,
        lr_default=3e-4,
        include_neg_per_pos=False,
        include_eval_cand=False,
    )
    parser.add_argument("--embed_url", type=str, default="http://127.0.0.1:8502/get_embedding")
    parser.add_argument("--embed_batch", type=int, default=64)
    parser.add_argument("--hid", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--rebuild_feature_cache", type=int, default=0)
    parser.add_argument("--eval_chunk", type=int, default=8192, help="batch size over agents for inference")
    parser.add_argument("--amp", type=int, default=0, help="1 to enable autocast on CUDA (bfloat16)")
    parser.add_argument("--use_tool_id_emb", type=int, default=1)
    parser.add_argument("--use_llm_id_emb", type=int, default=0, help="1 to add agent-ID embedding into agent tower")
    parser.add_argument(
        "--use_model_content_vector", type=int, default=1, help="1 to include V_model(A) in agent content view"
    )
    parser.add_argument(
        "--use_tool_content_vector", type=int, default=1, help="1 to include V_tool_content(A) in agent content view"
    )
    parser.add_argument("--use_tool_emb", type=int, default=None, help="Deprecated alias for --use_tool_id_emb")
    parser.add_argument("--use_agent_id_emb", type=int, default=None, help="Deprecated alias for --use_llm_id_emb")
    parser.add_argument("--use_query_id_emb", type=int, default=0, help="1 to add query-ID embedding into query tower")
    args = parser.parse_args()

    use_tool_id_emb = bool(args.use_tool_id_emb if args.use_tool_emb is None else args.use_tool_emb)
    use_llm_id_emb = bool(args.use_llm_id_emb if args.use_agent_id_emb is None else args.use_agent_id_emb)
    use_model_content_vector = bool(args.use_model_content_vector)
    use_tool_content_vector = bool(args.use_tool_content_vector)

    boot = bootstrap_run(
        data_root=args.data_root,
        exp_name=args.exp_name,
        topk=args.topk,
        seed=1234,
        with_tools=True,
    )

    bundle = boot.bundle
    tools = boot.tools
    all_agents = bundle.all_agents
    all_questions = bundle.all_questions
    all_rankings = bundle.all_rankings
    qid_to_part = bundle.qid_to_part

    q_ids = boot.q_ids
    a_ids = boot.a_ids
    qid2idx = boot.qid2idx
    aid2idx = boot.aid2idx
    qids_in_rank = boot.qids_in_rank
    data_sig = boot.data_sig
    exp_cache_dir = boot.exp_cache_dir
    embed_sig = cache_key_from_text(f"{args.embed_url}|{args.embed_batch}")
    feature_cache_dir = shared_cache_dir(
        args.data_root,
        "features",
        f"twotower_bge_{embed_sig}_{data_sig}",
    )

    if feature_cache_exists(feature_cache_dir) and args.rebuild_feature_cache == 0:
        feature_cache = load_feature_cache(feature_cache_dir)
    else:
        feature_cache = build_twotower_bge_feature_cache(
            all_agents,
            all_questions,
            tools,
            embed_url=args.embed_url,
            embed_batch=args.embed_batch,
        )
        save_feature_cache(feature_cache_dir, feature_cache)
        print(f"[cache] saved features to {feature_cache_dir}")

    Q_cpu = feature_cache.Q.astype(np.float32)
    A_cpu = build_agent_content_view(
        cache=feature_cache,
        use_model_content_vector=use_model_content_vector,
        use_tool_content_vector=use_tool_content_vector,
    )
    tool_ids_np = feature_cache.agent_tool_idx_padded
    tool_mask_np = feature_cache.agent_tool_mask

    want_meta = {
        "data_sig": data_sig,
        "pos_topk_by_part": POS_TOPK_BY_PART,
        "rng_seed_pairs": int(args.rng_seed_pairs),
        "split_seed": int(args.split_seed),
        "valid_ratio": float(args.valid_ratio),
        "pair_type": "q_pos_only_posTopK",
    }
    training_cache_dir = shared_cache_dir(args.data_root, "training", f"{data_sig}_{cache_key_from_meta(want_meta)}")

    def build_cache():
        train_qids, valid_qids = stratified_train_valid_split(
            qids_in_rank, qid_to_part=bundle.qid_to_part, valid_ratio=args.valid_ratio, seed=args.split_seed
        )
        pairs = build_pos_pairs(
            {qid: all_rankings[qid] for qid in train_qids},
            qid_to_part=qid_to_part,
            pos_topk_by_part=POS_TOPK_BY_PART,
            pos_topk_default=POS_TOPK,
            rng_seed=args.rng_seed_pairs,
        )
        pairs_idx = [(qid2idx[q], aid2idx[a]) for (q, a) in pairs]
        pairs_idx_np = np.array(pairs_idx, dtype=np.int64)
        return train_qids, valid_qids, pairs_idx_np

    train_qids, valid_qids, pairs_idx_np = load_or_build_training_cache(
        training_cache_dir,
        args.rebuild_training_cache,
        want_meta,
        build_cache,
    )

    device = torch.device(args.device)
    encoder = TwoTowerTFIDF(
        d_q=int(Q_cpu.shape[1]),
        d_a=int(A_cpu.shape[1]),
        num_tools=int(len(feature_cache.tool_id_vocab)),
        num_llm_ids=int(len(feature_cache.llm_vocab)),
        agent_tool_idx_padded=torch.tensor(tool_ids_np, dtype=torch.long, device=device),
        agent_tool_mask=torch.tensor(tool_mask_np, dtype=torch.float32, device=device),
        agent_llm_idx=torch.tensor(feature_cache.agent_llm_idx, dtype=torch.long, device=device),
        hid=args.hid,
        use_tool_id_emb=use_tool_id_emb,
        use_llm_id_emb=use_llm_id_emb,
        num_agents=len(a_ids),
        num_queries=len(q_ids),
        use_query_id_emb=bool(args.use_query_id_emb),
    ).to(device)

    optimizer = torch.optim.Adam(encoder.parameters(), lr=args.lr)

    num_pairs = pairs_idx_np.shape[0]
    num_batches = math.ceil(num_pairs / args.batch_size)
    print(f"Training pairs: {num_pairs}, batches/epoch: {num_batches}")

    use_amp = args.amp == 1 and device.type == "cuda"
    for epoch in range(1, args.epochs + 1):
        np.random.shuffle(pairs_idx_np)
        total = 0.0
        pbar = tqdm(range(num_batches), desc=f"Epoch {epoch}/{args.epochs}", dynamic_ncols=True)
        encoder.train()
        for b in pbar:
            sl = slice(b * args.batch_size, min((b + 1) * args.batch_size, num_pairs))
            batch = pairs_idx_np[sl]
            if batch.size == 0:
                continue
            q_idx = batch[:, 0]
            a_idx = batch[:, 1]

            q_vec = torch.from_numpy(Q_cpu[q_idx]).to(device, non_blocking=True)
            q_idx_t = torch.from_numpy(q_idx).to(device, non_blocking=True)
            a_pos = torch.from_numpy(A_cpu[a_idx]).to(device, non_blocking=True)
            a_idx_t = torch.from_numpy(a_idx).to(device, non_blocking=True)

            autocast_ctx = (
                torch.autocast(device_type="cuda", dtype=torch.bfloat16) if use_amp else nullcontext()
            )
            with autocast_ctx:
                qe = encoder.encode_q(q_vec, q_idx=q_idx_t)
                ae = encoder.encode_a(a_pos, a_idx_t)
                loss = info_nce_loss(qe, ae, temperature=args.temperature)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total += float(loss.item())
            pbar.set_postfix({"batch_loss": f"{loss.item():.4f}", "avg_loss": f"{(total / (b + 1)):.4f}"})

        print(f"Epoch {epoch}/{args.epochs} - InfoNCE: {(total / max(1, num_batches)):.4f}")

    model_dir = os.path.join(exp_cache_dir, "models")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"two_tower_bge_{data_sig}.pt")
    latest_model = os.path.join(model_dir, f"latest_{data_sig}.pt")
    meta_path = os.path.join(model_dir, f"meta_{data_sig}.json")

    ckpt = {
        "state_dict": encoder.state_dict(),
        "data_sig": data_sig,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "args": vars(args),
        "dims": {
            "d_q": int(Q_cpu.shape[1]),
            "d_a": int(A_cpu.shape[1]),
            "hid": int(args.hid),
            "num_tools": int(len(feature_cache.tool_id_vocab)),
            "num_agents": int(len(a_ids)),
        },
        "flags": {
            "use_tool_id_emb": use_tool_id_emb,
            "use_llm_id_emb": use_llm_id_emb,
            "use_model_content_vector": use_model_content_vector,
            "use_tool_content_vector": use_tool_content_vector,
            "use_query_id_emb": bool(args.use_query_id_emb),
        },
        "mappings": {"q_ids": q_ids, "a_ids": a_ids, "tool_names": feature_cache.tool_names},
    }
    torch.save(ckpt, model_path)
    torch.save(ckpt, latest_model)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {"data_sig": data_sig, "a_ids": a_ids, "tool_names": feature_cache.tool_names},
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"[save] model -> {model_path}")
    print(f"[save] meta  -> {meta_path}")

    topk = int(args.topk)
    valid_metrics = evaluate_sampled_twotower(
        encoder,
        Q_cpu,
        A_cpu,
        qid2idx,
        a_ids,
        all_rankings,
        valid_qids,
        device=device,
        ks=(topk,),
        cand_size=100,
        rng_seed=123,
        amp=use_amp,
        qid_to_part=qid_to_part,
        pos_topk_by_part=POS_TOPK_BY_PART,
        pos_topk_default=POS_TOPK,
    )
    print_metrics_table(
        "Validation Overall (averaged over questions)", valid_metrics, ks=(topk,), filename=args.exp_name
    )

    part_splits = split_eval_qids_by_part(valid_qids, qid_to_part=qid_to_part)
    for part in ["PartI", "PartII", "PartIII"]:
        qids_part = part_splits.get(part, [])
        if not qids_part:
            continue
        m_part = evaluate_sampled_twotower(
            encoder,
            Q_cpu,
            A_cpu,
            qid2idx,
            a_ids,
            all_rankings,
            qids_part,
            device=device,
            ks=(topk,),
            cand_size=100,
            rng_seed=123,
            amp=use_amp,
            qid_to_part=qid_to_part,
            pos_topk_by_part=POS_TOPK_BY_PART,
            pos_topk_default=POS_TOPK,
        )
        print_metrics_table(
            f"Validation {part} (averaged over questions)", m_part, ks=(topk,), filename=args.exp_name
        )

    # @torch.no_grad()
    # def recommend_topk_for_qid(qid: str, topk: int = 10, chunk: int = 8192):
    #     qi = qid2idx[qid]
    #     qv = torch.from_numpy(Q_cpu[qi : qi + 1]).to(device)
    #     q_idx_t = torch.tensor([qi], dtype=torch.long, device=device)
    #     qe = encoder.encode_q(qv, q_idx=q_idx_t)

    #     best_scores: List[float] = []
    #     best_ids: List[int] = []
    #     num_agents = len(a_ids)
    #     for i in range(0, num_agents, chunk):
    #         j = min(i + chunk, num_agents)
    #         a_idx = torch.arange(i, j, dtype=torch.long, device=device)
    #         av = torch.from_numpy(A_cpu[i:j]).to(device)
    #         ae = encoder.encode_a(av, a_idx)
    #         scores = (qe @ ae.t()).squeeze(0)
    #         k = min(topk, j - i)
    #         top_scores, top_local_idx = torch.topk(scores, k)
    #         best_scores.extend(top_scores.cpu().tolist())
    #         best_ids.extend([i + int(t) for t in top_local_idx.cpu().tolist()])

    #     best_scores_t = torch.tensor(best_scores)
    #     best_ids_t = torch.tensor(best_ids)
    #     k = min(topk, best_scores_t.numel())
    #     final_scores, final_idx = torch.topk(best_scores_t, k)
    #     result = [
    #         (a_ids[int(best_ids_t[idx])], float(final_scores[n].item()))
    #         for n, idx in enumerate(final_idx)
    #     ]
    #     return result

    # sample_qids = q_ids[: min(5, len(q_ids))]
    # for qid in sample_qids:
    #     recs = recommend_topk_for_qid(qid, topk=args.topk, chunk=args.eval_chunk)
    #     qtext = all_questions[qid]["input"][:80].replace("\n", " ")
    #     print(f"\nQuestion: {qid}  |  {qtext}")
    #     for r, (aid, s) in enumerate(recs, 1):
    #         print(f"  {r:2d}. {aid:>20s}  score={s:.4f}")


if __name__ == "__main__":
    main()
