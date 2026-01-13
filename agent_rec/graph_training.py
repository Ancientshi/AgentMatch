#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from tqdm.auto import tqdm

from agent_rec.config import POS_TOPK, POS_TOPK_BY_PART
from agent_rec.data import build_training_pairs, stratified_train_valid_split
from agent_rec.graph_utils import GraphFeaturePack, build_graph_features
from agent_rec.models.graph import bpr_loss
from agent_rec.run_common import build_pos_pairs, cache_key_from_meta, load_or_build_training_cache, shared_cache_dir


@dataclass
class GraphDataPack:
    train_qids: List[str]
    valid_qids: List[str]
    pairs_idx_np: np.ndarray
    interactions: List[Tuple[int, int]]
    feature_pack: GraphFeaturePack
    rankings_train: Dict[str, List[str]]


def prepare_graph_data(
    *,
    args,
    boot,
    use_model_content_vector: bool,
    use_tool_content_vector: bool,
    max_features: int,
) -> GraphDataPack:
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

    rankings_train = {qid: all_rankings[qid] for qid in train_qids}
    pos_pairs = build_pos_pairs(
        rankings_train,
        qid_to_part=qid_to_part,
        pos_topk_by_part=POS_TOPK_BY_PART,
        pos_topk_default=POS_TOPK,
        rng_seed=args.rng_seed_pairs,
    )
    interactions = [(qid2idx[q], aid2idx[a]) for q, a in pos_pairs]

    feature_pack = build_graph_features(
        all_agents,
        all_questions,
        tools or {},
        q_ids=q_ids,
        a_ids=a_ids,
        max_features=max_features,
        use_model_content_vector=use_model_content_vector,
        use_tool_content_vector=use_tool_content_vector,
    )
    return GraphDataPack(
        train_qids=train_qids,
        valid_qids=valid_qids,
        pairs_idx_np=pairs_idx_np,
        interactions=interactions,
        feature_pack=feature_pack,
        rankings_train=rankings_train,
    )


def train_graph_bpr(
    model: torch.nn.Module,
    pairs: Sequence[Tuple[int, int, int]],
    *,
    batch_size: int,
    epochs: int,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
) -> None:
    num_pairs = len(pairs)
    num_batches = math.ceil(num_pairs / batch_size)
    print(f"Training pairs: {num_pairs}, batches/epoch: {num_batches}")

    for epoch in range(1, epochs + 1):
        random.shuffle(pairs)
        total_loss = 0.0
        pbar = tqdm(range(num_batches), desc=f"Epoch {epoch}/{epochs}", leave=True, dynamic_ncols=True)
        model.train()
        for b in pbar:
            batch = pairs[b * batch_size : (b + 1) * batch_size]
            if not batch:
                continue
            q_idx = torch.tensor([t[0] for t in batch], dtype=torch.long, device=device)
            pos_idx = torch.tensor([t[1] for t in batch], dtype=torch.long, device=device)
            neg_idx = torch.tensor([t[2] for t in batch], dtype=torch.long, device=device)

            out = model(q_idx, pos_idx, neg_idx)
            cl_loss = torch.tensor(0.0, device=device)
            if isinstance(out, tuple) and len(out) == 3:
                pos, neg, cl_loss = out
            else:
                pos, neg = out

            loss = bpr_loss(pos, neg)
            if hasattr(model, "cl_weight"):
                loss = loss + float(getattr(model, "cl_weight", 0.0)) * cl_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            postfix = {
                "batch_loss": f"{loss.item():.4f}",
                "avg_loss": f"{(total_loss / (b + 1)):.4f}",
            }
            if cl_loss.item() != 0.0:
                postfix["cl_loss"] = f"{cl_loss.item():.4f}"
            pbar.set_postfix(postfix)

        print(f"Epoch {epoch}/{epochs} - loss: {(total_loss / num_batches if num_batches else 0.0):.4f}")
