#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import math
import os
import random
from datetime import datetime

import numpy as np
import torch
from tqdm.auto import tqdm

from agent_rec.cli_common import add_shared_training_args
from agent_rec.config import EVAL_TOPK, POS_TOPK, POS_TOPK_BY_PART
from agent_rec.data import build_training_pairs, stratified_train_valid_split
from agent_rec.features import build_unified_corpora, UNK_LLM_TOKEN
from agent_rec.knn import build_knn_cache, load_knn_cache
from agent_rec.eval import evaluate_sampled_knn_top10, split_eval_qids_by_part
from agent_rec.models.bpr_mf import BPRMF, bpr_loss
from agent_rec.run_common import bootstrap_run, cache_key_from_meta, load_or_build_training_cache, shared_cache_dir

from utils import print_metrics_table  # 依赖你现有 utils.py


def main():
    parser = argparse.ArgumentParser()
    add_shared_training_args(
        parser,
        exp_name_default="bpr_mf_knn",
        epochs_default=5,
        batch_size_default=4096,
        lr_default=5e-3,
    )
    parser.add_argument("--factors", type=int, default=128)

    parser.add_argument("--knn_N", type=int, default=3)
    parser.add_argument("--score_mode", type=str, default="dot", choices=["dot", "cosine"])

    args = parser.parse_args()
    boot = bootstrap_run(
        data_root=args.data_root,
        exp_name=args.exp_name,
        topk=args.topk,
        with_tools=False,
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

    want_meta = {
        "data_sig": data_sig,
        "pos_topk_by_part": POS_TOPK_BY_PART,
        "neg_per_pos": int(args.neg_per_pos),
        "rng_seed_pairs": int(args.rng_seed_pairs),
        "split_seed": int(args.split_seed),
        "valid_ratio": float(args.valid_ratio),
    }
    training_cache_dir = shared_cache_dir(args.data_root, "training", f"{data_sig}_{cache_key_from_meta(want_meta)}")

    def build_cache():
        train_qids, valid_qids = stratified_train_valid_split(
            qids_in_rank, qid_to_part=qid_to_part, valid_ratio=args.valid_ratio, seed=args.split_seed
        )
        print(f"[split] train={len(train_qids)}  valid={len(valid_qids)}")

        rankings_train = {qid: all_rankings[qid] for qid in train_qids}
        pairs = build_training_pairs(
            rankings_train,
            a_ids,
            qid_to_part=qid_to_part,
            pos_topk_by_part=POS_TOPK_BY_PART,
            pos_topk_default=POS_TOPK,
            neg_per_pos=args.neg_per_pos,
            rng_seed=args.rng_seed_pairs,
        )
        pairs_idx = [(qid2idx[q], aid2idx[p], aid2idx[n]) for (q, p, n) in pairs]
        pairs_idx_np = np.array(pairs_idx, dtype=np.int64)
        return train_qids, valid_qids, pairs_idx_np

    train_qids, valid_qids, pairs_idx_np = load_or_build_training_cache(
        training_cache_dir,
        args.rebuild_training_cache,
        want_meta,
        build_cache,
    )

    _, _, _, _, _, _, _, llm_ids = build_unified_corpora(all_agents, all_questions, tools or {})
    llm_vocab = [UNK_LLM_TOKEN] + [lid for lid in llm_ids if lid]
    llm_vocab = list(dict.fromkeys(llm_vocab))
    llm_vocab_map = {n: i for i, n in enumerate(llm_vocab)}
    agent_llm_idx = np.array([llm_vocab_map.get(lid, 0) for lid in llm_ids], dtype=np.int64)

    device = torch.device(args.device)
    model = BPRMF(
        num_q=len(q_ids),
        num_a=len(a_ids),
        num_llm_ids=len(llm_vocab),
        agent_llm_idx=torch.tensor(agent_llm_idx, dtype=torch.long, device=device),
        factors=args.factors,
        add_bias=True,
        use_llm_id_emb=True,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    pairs = pairs_idx_np.tolist()
    num_pairs = len(pairs)
    num_batches = math.ceil(num_pairs / args.batch_size)
    print(f"Training pairs: {num_pairs}, batches/epoch: {num_batches}")

    for epoch in range(1, args.epochs + 1):
        random.shuffle(pairs)
        total_loss = 0.0
        pbar = tqdm(range(num_batches), desc=f"Epoch {epoch}/{args.epochs}", leave=True, dynamic_ncols=True)
        model.train()
        for b in pbar:
            batch = pairs[b * args.batch_size:(b + 1) * args.batch_size]
            if not batch:
                continue
            q_idx = torch.tensor([t[0] for t in batch], dtype=torch.long, device=device)
            pos_idx = torch.tensor([t[1] for t in batch], dtype=torch.long, device=device)
            neg_idx = torch.tensor([t[2] for t in batch], dtype=torch.long, device=device)

            pos, neg = model(q_idx, pos_idx, neg_idx)
            loss = bpr_loss(pos, neg)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            pbar.set_postfix({"batch_loss": f"{loss.item():.4f}", "avg_loss": f"{(total_loss / (b + 1)):.4f}"})

        print(f"Epoch {epoch}/{args.epochs} - BPR loss: {(total_loss / num_batches if num_batches else 0.0):.4f}")

    model_dir = os.path.join(exp_cache_dir, "models")
    os.makedirs(model_dir, exist_ok=True)
    data_sig = want_meta["data_sig"]

    ckpt_path = os.path.join(model_dir, f"{args.exp_name}_{data_sig}.pt")
    meta_path = os.path.join(model_dir, f"meta_{args.exp_name}_{data_sig}.json")

    ckpt = {
        "state_dict": model.state_dict(),
        "data_sig": data_sig,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "dims": {"num_q": len(q_ids), "num_a": len(a_ids), "factors": args.factors},
        "mappings": {"q_ids": q_ids, "a_ids": a_ids},
        "args": vars(args),
        "model_extra": model.extra_state_dict(),
    }
    torch.save(ckpt, ckpt_path)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({"data_sig": data_sig, "q_ids": q_ids, "a_ids": a_ids}, f, ensure_ascii=False, indent=2)
    print(f"[save] model -> {ckpt_path}")
    print(f"[save] meta  -> {meta_path}")

    knn_path = os.path.join(exp_cache_dir, "knn_copy.pkl")
    build_knn_cache(
        train_qids=train_qids,
        all_questions=all_questions,
        qid2idx=qid2idx,
        model=model,
        cache_path=knn_path,
    )
    print("[cache] saved KNN-copy cache")
    knn_cache = load_knn_cache(knn_path)

    model.eval()
    topk = int(args.topk)

    overall_metrics = evaluate_sampled_knn_top10(
        model=model,
        aid2idx=aid2idx,
        all_rankings=all_rankings,
        all_questions=all_questions,
        eval_qids=valid_qids,
        knn_cache=knn_cache,
        cand_size=args.eval_cand_size,
        knn_N=args.knn_N,
        qid_to_part=qid_to_part,
        pos_topk_by_part=POS_TOPK_BY_PART,
        pos_topk_default=POS_TOPK,
        topk=topk,
        score_mode=args.score_mode,
        desc=f"Valid Overall (KNN q-vector, top{topk})",
    )
    print_metrics_table("Validation Overall (KNN q-vector)", overall_metrics, ks=(topk,), filename=args.exp_name)

    part_splits = split_eval_qids_by_part(valid_qids, qid_to_part=qid_to_part)
    for part in ["PartI", "PartII", "PartIII"]:
        qids_part = part_splits.get(part, [])
        if not qids_part:
            continue
        m_part = evaluate_sampled_knn_top10(
            model=model,
            aid2idx=aid2idx,
            all_rankings=all_rankings,
            all_questions=all_questions,
            eval_qids=qids_part,
            knn_cache=knn_cache,
            cand_size=args.eval_cand_size,
            knn_N=args.knn_N,
            qid_to_part=qid_to_part,
            pos_topk_by_part=POS_TOPK_BY_PART,
            pos_topk_default=POS_TOPK,
            topk=topk,
            score_mode=args.score_mode,
            desc=f"Valid {part} (KNN q-vector, top{topk})",
        )
        print_metrics_table(f"Validation {part} (KNN q-vector)", m_part, ks=(topk,), filename=args.exp_name)


if __name__ == "__main__":
    main()
