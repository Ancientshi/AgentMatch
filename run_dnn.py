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
from agent_rec.config import EVAL_TOPK, POS_TOPK, POS_TOPK_BY_PART, TFIDF_MAX_FEATURES
from agent_rec.data import build_training_pairs, stratified_train_valid_split
from agent_rec.features import (
    build_feature_cache,
    build_unified_corpora,
    feature_cache_exists,
    build_agent_content_view,
    load_feature_cache,
    load_q_vectorizer,
    save_feature_cache,
    save_q_vectorizer,
    UNK_TOOL_TOKEN,
    UNK_LLM_TOKEN,
    build_agent_tool_id_buffers,
)
from agent_rec.eval import evaluate_sampled_direct_top10, split_eval_qids_by_part
from agent_rec.models.dnn import SimpleBPRDNN, bpr_loss
from agent_rec.run_common import bootstrap_run, cache_key_from_meta, load_or_build_training_cache, shared_cache_dir

from utils import print_metrics_table

def main():
    parser = argparse.ArgumentParser()
    add_shared_training_args(
        parser,
        exp_name_default="bpr_dnn",
        device_default="cpu",
        epochs_default=5,
        batch_size_default=1024,
        lr_default=1e-3,
    )
    parser.add_argument("--text_hidden", type=int, default=256)
    parser.add_argument("--id_dim", type=int, default=32)
    parser.add_argument("--max_features", type=int, default=TFIDF_MAX_FEATURES)
    parser.add_argument("--rebuild_feature_cache", type=int, default=0)
    parser.add_argument("--use_query_id_emb", type=int, default=0, help="1 to add optional query-ID embedding")
    parser.add_argument("--use_llm_id_emb", type=int, default=1)
    parser.add_argument("--use_tool_id_emb", type=int, default=1)
    parser.add_argument("--use_model_content_vector", type=int, default=1)
    parser.add_argument("--use_tool_content_vector", type=int, default=1)

    args = parser.parse_args()
    boot = bootstrap_run(
        data_root=args.data_root,
        exp_name=args.exp_name,
        topk=args.topk,
        with_tools=True,
    )

    bundle = boot.bundle
    tools = boot.tools
    all_agents = bundle.all_agents
    all_questions = bundle.all_questions
    all_rankings = bundle.all_rankings
    qid_to_part = bundle.qid_to_part

    tool_names = list(tools.keys())

    q_ids = boot.q_ids
    a_ids = boot.a_ids
    qid2idx = boot.qid2idx
    aid2idx = boot.aid2idx
    qids_in_rank = boot.qids_in_rank
    data_sig = boot.data_sig
    exp_cache_dir = boot.exp_cache_dir
    feature_cache_dir = shared_cache_dir(
        args.data_root,
        "features",
        f"tfidf_{args.max_features}_{data_sig}",
    )

    if feature_cache_exists(feature_cache_dir) and args.rebuild_feature_cache == 0:
        feature_cache = load_feature_cache(feature_cache_dir)
        q_vectorizer_runtime = load_q_vectorizer(feature_cache_dir)
        if q_vectorizer_runtime is None:
            _, q_texts, _, _, _, _, _, _ = build_unified_corpora(all_agents, all_questions, tools)
            from sklearn.feature_extraction.text import TfidfVectorizer

            q_vectorizer_runtime = TfidfVectorizer(
                max_features=args.max_features, lowercase=True
            ).fit(q_texts)
            save_q_vectorizer(feature_cache_dir, q_vectorizer_runtime)
    else:
        feature_cache, q_vectorizer_runtime = build_feature_cache(
            all_agents, all_questions, tools, max_features=args.max_features
        )
        save_feature_cache(feature_cache_dir, feature_cache)
        save_q_vectorizer(feature_cache_dir, q_vectorizer_runtime)
        print(f"[cache] saved features to {feature_cache_dir}")

    Q_np = feature_cache.Q.astype(np.float32)
    A_text_full_np = build_agent_content_view(
        cache=feature_cache,
        use_model_content_vector=bool(args.use_model_content_vector),
        use_tool_content_vector=bool(args.use_tool_content_vector),
    )
    tool_ids_np = feature_cache.agent_tool_idx_padded
    tool_mask_np = feature_cache.agent_tool_mask
    llm_idx_np = feature_cache.agent_llm_idx

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

    device = torch.device(args.device)
    model = SimpleBPRDNN(
        d_q=int(Q_np.shape[1]),
        d_a=int(A_text_full_np.shape[1]),
        num_tools=int(len(feature_cache.tool_id_vocab)),
        num_llm_ids=int(len(feature_cache.llm_vocab)),
        agent_tool_indices_padded=torch.tensor(tool_ids_np, dtype=torch.long, device=device),
        agent_tool_mask=torch.tensor(tool_mask_np, dtype=torch.float32, device=device),
        agent_llm_idx=torch.tensor(feature_cache.agent_llm_idx, dtype=torch.long, device=device),
        text_hidden=args.text_hidden,
        id_dim=args.id_dim,
        num_queries=len(q_ids),
        use_query_id_emb=bool(args.use_query_id_emb),
        use_tool_id_emb=bool(args.use_tool_id_emb),
        use_llm_id_emb=bool(args.use_llm_id_emb),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    pairs = pairs_idx_np.tolist()
    num_pairs = len(pairs)
    num_batches = math.ceil(num_pairs / args.batch_size)
    print(f"Training pairs: {num_pairs}, batches/epoch: {num_batches}")

    Q_t = torch.tensor(Q_np, dtype=torch.float32, device=device)
    A_t = torch.tensor(A_text_full_np, dtype=torch.float32, device=device)

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

            q_vec = Q_t[q_idx]
            pos_vec = A_t[pos_idx]
            neg_vec = A_t[neg_idx]

            pos, neg = model(q_vec, pos_vec, neg_vec, pos_idx, neg_idx, q_idx=q_idx)
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
        "dims": {
            "d_q": int(Q_np.shape[1]),
            "d_a": int(A_text_full_np.shape[1]),
            "num_agents": len(a_ids),
            "num_tools": len(tool_names),
            "text_hidden": args.text_hidden,
            "id_dim": args.id_dim,
        },
        "flags": {
            "use_llm_id_emb": bool(args.use_llm_id_emb),
            "use_tool_id_emb": bool(args.use_tool_id_emb),
            "use_model_content_vector": bool(args.use_model_content_vector),
            "use_tool_content_vector": bool(args.use_tool_content_vector),
            "use_query_id_emb": bool(args.use_query_id_emb),
        },
        "mappings": {"q_ids": q_ids, "a_ids": a_ids, "tool_names": tool_names},
        "args": vars(args),
    }
    torch.save(ckpt, ckpt_path)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({"data_sig": data_sig, "q_ids": q_ids, "a_ids": a_ids}, f, ensure_ascii=False, indent=2)
    print(f"[save] model -> {ckpt_path}")
    print(f"[save] meta  -> {meta_path}")

    model.eval()
    topk = int(args.topk)

    overall_metrics = evaluate_sampled_direct_top10(
        model=model,
        aid2idx=aid2idx,
        qid2idx=qid2idx,
        all_rankings=all_rankings,
        all_questions=all_questions,
        eval_qids=valid_qids,
        q_vectorizer=q_vectorizer_runtime,
        A_text_full=A_text_full_np,
        cand_size=args.eval_cand_size,
        qid_to_part=qid_to_part,
        pos_topk_by_part=POS_TOPK_BY_PART,
        pos_topk_default=POS_TOPK,
        topk=topk,
        desc=f"Valid Overall (direct q-vector, top{topk})",
    )
    print_metrics_table("Validation Overall (direct q-vector)", overall_metrics, ks=(topk,), filename=args.exp_name)

    part_splits = split_eval_qids_by_part(valid_qids, qid_to_part=qid_to_part)
    for part in ["PartI", "PartII", "PartIII"]:
        qids_part = part_splits.get(part, [])
        if not qids_part:
            continue
        m_part = evaluate_sampled_direct_top10(
            model=model,
            aid2idx=aid2idx,
            qid2idx=qid2idx,
            all_rankings=all_rankings,
            all_questions=all_questions,
            eval_qids=qids_part,
            q_vectorizer=q_vectorizer_runtime,
            A_text_full=A_text_full_np,
            cand_size=args.eval_cand_size,
            qid_to_part=qid_to_part,
            pos_topk_by_part=POS_TOPK_BY_PART,
            pos_topk_default=POS_TOPK,
            topk=topk,
            desc=f"Valid {part} (direct q-vector, top{topk})",
        )
        print_metrics_table(f"Validation {part} (direct q-vector)", m_part, ks=(topk,), filename=args.exp_name)


if __name__ == "__main__":
    main()
