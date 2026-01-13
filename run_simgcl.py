#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
from datetime import datetime

import torch

from agent_rec.cli_common import add_shared_training_args
from agent_rec.config import POS_TOPK, POS_TOPK_BY_PART, TFIDF_MAX_FEATURES
from agent_rec.eval import evaluate_sampled_knn_top10, split_eval_qids_by_part
from agent_rec.graph_training import prepare_graph_data, train_graph_bpr
from agent_rec.models.graph import SimGCLRecommender
from agent_rec.knn import build_knn_cache, load_knn_cache
from agent_rec.run_common import bootstrap_run
from utils import print_metrics_table


def env_default(name: str, fallback: int) -> int:
    return int(os.getenv(name, fallback))


def build_model(args, boot, data_pack):
    device = torch.device(args.device)
    fp = data_pack.feature_pack
    agent_content_t = (
        torch.tensor(fp.agent_content, dtype=torch.float32, device=device) if fp.agent_content is not None else None
    )
    model = SimGCLRecommender(
        num_q=len(boot.q_ids),
        num_a=len(boot.a_ids),
        embed_dim=args.embed_dim,
        interactions=data_pack.interactions,
        agent_content=agent_content_t,
        agent_tool_indices_padded=torch.tensor(fp.agent_tool_idx_padded, dtype=torch.long, device=device),
        agent_tool_mask=torch.tensor(fp.agent_tool_mask, dtype=torch.float32, device=device),
        agent_llm_idx=torch.tensor(fp.agent_llm_idx, dtype=torch.long, device=device),
        num_tools=len(fp.tool_id_vocab),
        num_llm_ids=len(fp.llm_vocab),
        use_query_id_emb=bool(args.use_query_id_emb),
        use_tool_id_emb=bool(args.use_tool_id_emb),
        use_llm_id_emb=bool(args.use_llm_id_emb),
        use_agent_content=bool(args.use_model_content_vector or args.use_tool_content_vector),
        num_layers=args.num_layers,
        cl_weight=args.cl_weight,
        perturb_eps=args.perturb_eps,
        temperature=args.temperature,
        device=device,
    ).to(device)
    return model


def main():
    parser = argparse.ArgumentParser()
    add_shared_training_args(
        parser,
        exp_name_default="simgcl",
        epochs_default=5,
        batch_size_default=2048,
        lr_default=1e-3,
    )
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--cl_weight", type=float, default=0.1)
    parser.add_argument("--perturb_eps", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max_features", type=int, default=TFIDF_MAX_FEATURES)
    parser.add_argument("--knn_N", type=int, default=3)
    parser.add_argument("--score_mode", type=str, default="dot", choices=["dot", "cosine"])
    parser.add_argument("--use_query_id_emb", type=int, default=env_default("USE_QUERY_ID_EMB", 1))
    parser.add_argument("--use_tool_id_emb", type=int, default=env_default("USE_TOOL_ID_EMB", 1))
    parser.add_argument("--use_llm_id_emb", type=int, default=env_default("USE_LLM_ID_EMB", 1))
    parser.add_argument("--use_model_content_vector", type=int, default=env_default("USE_MODEL_CONTENT_VECTOR", 1))
    parser.add_argument("--use_tool_content_vector", type=int, default=env_default("USE_TOOL_CONTENT_VECTOR", 1))

    args = parser.parse_args()
    boot = bootstrap_run(
        data_root=args.data_root,
        exp_name=args.exp_name,
        topk=args.topk,
        with_tools=True,
    )

    data_pack = prepare_graph_data(
        args=args,
        boot=boot,
        use_model_content_vector=bool(args.use_model_content_vector),
        use_tool_content_vector=bool(args.use_tool_content_vector),
        max_features=args.max_features,
    )

    device = torch.device(args.device)
    model = build_model(args, boot, data_pack)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    pairs = data_pack.pairs_idx_np.tolist()
    train_graph_bpr(
        model,
        pairs,
        batch_size=args.batch_size,
        epochs=args.epochs,
        device=device,
        optimizer=optimizer,
    )

    exp_cache_dir = boot.exp_cache_dir
    model_dir = os.path.join(exp_cache_dir, "models")
    os.makedirs(model_dir, exist_ok=True)
    data_sig = boot.data_sig

    ckpt_path = os.path.join(model_dir, f"{args.exp_name}_{data_sig}.pt")
    meta_path = os.path.join(model_dir, f"meta_{args.exp_name}_{data_sig}.json")

    ckpt = {
        "state_dict": model.state_dict(),
        "data_sig": data_sig,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "dims": {"num_q": len(boot.q_ids), "num_a": len(boot.a_ids), "embed_dim": args.embed_dim},
        "mappings": {"q_ids": boot.q_ids, "a_ids": boot.a_ids},
        "args": vars(args),
        "model_extra": model.extra_state_dict(),
    }
    torch.save(ckpt, ckpt_path)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({"data_sig": data_sig, "q_ids": boot.q_ids, "a_ids": boot.a_ids}, f, ensure_ascii=False, indent=2)
    print(f"[save] model -> {ckpt_path}")
    print(f"[save] meta  -> {meta_path}")

    knn_path = os.path.join(exp_cache_dir, "knn_copy.pkl")
    build_knn_cache(
        train_qids=data_pack.train_qids,
        all_questions=boot.bundle.all_questions,
        qid2idx=boot.qid2idx,
        model=model,
        cache_path=knn_path,
    )
    print("[cache] saved KNN-copy cache")
    knn_cache = load_knn_cache(knn_path)

    model.eval()
    topk = int(args.topk)

    overall_metrics = evaluate_sampled_knn_top10(
        model=model,
        aid2idx=boot.aid2idx,
        all_rankings=boot.bundle.all_rankings,
        all_questions=boot.bundle.all_questions,
        eval_qids=data_pack.valid_qids,
        knn_cache=knn_cache,
        cand_size=args.eval_cand_size,
        knn_N=args.knn_N,
        qid_to_part=boot.bundle.qid_to_part,
        pos_topk_by_part=POS_TOPK_BY_PART,
        pos_topk_default=POS_TOPK,
        topk=topk,
        score_mode=args.score_mode,
        desc=f"Valid Overall (KNN q-vector, top{topk})",
    )
    print_metrics_table("Validation Overall (KNN q-vector)", overall_metrics, ks=(topk,), filename=args.exp_name)

    part_splits = split_eval_qids_by_part(data_pack.valid_qids, qid_to_part=boot.bundle.qid_to_part)
    for part in ["PartI", "PartII", "PartIII"]:
        qids_part = part_splits.get(part, [])
        if not qids_part:
            continue
        m_part = evaluate_sampled_knn_top10(
            model=model,
            aid2idx=boot.aid2idx,
            all_rankings=boot.bundle.all_rankings,
            all_questions=boot.bundle.all_questions,
            eval_qids=qids_part,
            knn_cache=knn_cache,
            cand_size=args.eval_cand_size,
            knn_N=args.knn_N,
            qid_to_part=boot.bundle.qid_to_part,
            pos_topk_by_part=POS_TOPK_BY_PART,
            pos_topk_default=POS_TOPK,
            topk=topk,
            score_mode=args.score_mode,
            desc=f"Valid {part} (KNN q-vector, top{topk})",
        )
        print_metrics_table(f"Validation {part} (KNN q-vector)", m_part, ks=(topk,), filename=args.exp_name)


if __name__ == "__main__":
    main()
