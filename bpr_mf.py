#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple BPR-MF Agent Recommender (ID-only matrix factorization with BPR loss).

- ID-only 矩阵分解 + BPR 损失
- 读取数据布局：
    {data_root}/PartI|PartII|PartIII/{agents,questions,rankings}/merge.json
    {data_root}/Tools/merge.json (不使用也可存在)
- 训练后会构建 TF-IDF KNN 缓存，并在 *评估与推荐* 时：
  用 TF-IDF 在训练问题上找 KNN，对应的 BPR 问题向量按相似度加权平均作为该问题的 q 向量，
  再与所有 agent 向量做点积打分，评估 sampled metrics。

Usage:
  python simple_bpr_mf_agent_rec.py \
    --data_root /path/to/dataset_root \
    --epochs 3 --batch_size 4096 --factors 128 --neg_per_pos 1 --topk 10 --device cuda:0
"""
import os
import json
import math
import argparse
import random
import zlib
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

# ---------------------------------------------------------------------------
filename = os.path.splitext(os.path.basename(__file__))[0]
pos_topk = 5  # 只把排名前K的 agent 当做正样本
# ---------------------------------------------------------------------------


# ------------------------- Helpers: cache & utils -------------------------
def ensure_cache_dir(root: str) -> str:
    d = os.path.join(root, f".cache/{filename}")
    os.makedirs(d, exist_ok=True)
    return d


def dataset_signature(q_ids: List[str], a_ids: List[str], rankings: Dict[str, List[str]]) -> str:
    """Create a lightweight signature (CRC32) for the current dataset identity."""
    payload = {
        "q_ids": q_ids,
        "a_ids": a_ids,
        "rankings": {k: rankings[k] for k in sorted(rankings.keys())},
    }
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return f"{(zlib.crc32(blob) & 0xFFFFFFFF):08x}"


def train_valid_split(qids_in_rankings, valid_ratio=0.2, seed=42):
    rng = random.Random(seed)
    q = list(qids_in_rankings)
    rng.shuffle(q)
    n_valid = int(len(q) * valid_ratio)
    valid_q = q[:n_valid]
    train_q = q[n_valid:]
    return train_q, valid_q


def build_training_pairs(all_rankings: Dict[str, List[str]],
                         all_agent_ids: List[str],
                         neg_per_pos: int = 1,
                         rng_seed: int = 42):
    """
    返回 [(qid_str, pos_agent_id, neg_agent_id)]，qid 为字符串。
    负样本从“非正样本”的 agent 里随机采样。
    """
    rnd = random.Random(rng_seed)
    pairs = []
    all_agent_set = set(all_agent_ids)
    for qid, ranked in all_rankings.items():
        ranked = ranked[:pos_topk]
        ranked_set = set(ranked)
        neg_pool = list(all_agent_set - ranked_set) or list(all_agent_ids)
        for pos_a in ranked:
            for _ in range(neg_per_pos):
                neg_a = rnd.choice(neg_pool)
                pairs.append((qid, pos_a, neg_a))
    return pairs


# ------------------------------- Data loading ------------------------------
def load_json(p: str):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def collect_data(data_root: str):
    """
    Reads PartI/II/III agents/questions/rankings and merges them.
    Returns:
      all_agents: Dict[agent_id, {...}]
      all_questions: Dict[qid, {"input": "..."}]
      all_rankings: Dict[qid, List[agent_id]]
    """
    parts = ["PartI", "PartII", "PartIII"]
    all_agents: Dict[str, dict] = {}
    all_questions: Dict[str, dict] = {}
    all_rankings: Dict[str, List[str]] = {}

    for part in parts:
        agents_path = os.path.join(data_root, part, "agents", "merge.json")
        questions_path = os.path.join(data_root, part, "questions", "merge.json")
        rankings_path = os.path.join(data_root, part, "rankings", "merge.json")

        agents = load_json(agents_path)
        questions = load_json(questions_path)
        rankings = load_json(rankings_path)

        all_agents.update(agents)
        all_questions.update(questions)
        # rankings file format: {"rankings": {qid: [aid,...], ...}}
        all_rankings.update(rankings["rankings"])

    return all_agents, all_questions, all_rankings


# ------------------------------- Evaluation helpers ------------------------
def _ideal_dcg(k, num_rel):
    ideal = min(k, num_rel)
    return sum(1.0 / math.log2(i + 2.0) for i in range(ideal)) if ideal else 0.0


        
    



def knn_qvec_for_question_text(question_text: str, knn_cache_path: str, N: int = 8) -> np.ndarray:
    """
    用 TF-IDF 在训练问题上做 KNN，取相似度加权平均的训练集 Q-embedding 作为该问题的 q 向量。
    返回 shape=(F,) 的 numpy 向量。
    """
    with open(knn_cache_path, "rb") as f:
        cc = pickle.load(f)
    tfidf, X, Q = cc["tfidf"], cc["X"], cc["Q"]   # X: (Nq, V) csr_matrix; Q: (Nq, F)
    x = tfidf.transform([question_text]).astype(np.float32)  # (1, V)
    sims = (x @ X.T).toarray()[0]                            # (Nq,)

    if len(sims) == 0:
        return np.zeros((Q.shape[1],), dtype=np.float32)

    if N >= len(sims):
        idx = np.argsort(-sims)
    else:
        idx = np.argpartition(-sims, N)[:N]
        idx = idx[np.argsort(-sims[idx])]

    w = sims[idx]
    w = w / (w.sum() + 1e-8)
    qv = (w[:, None] * Q[idx]).sum(axis=0)                   # (F,)
    return qv.astype(np.float32)


@torch.no_grad()
def evaluate_sampled_knn(model,
                         qid2idx,
                         aid2idx,
                         all_rankings,
                         all_questions,
                         eval_qids,
                         knn_cache_path: str,
                         ks=(5, 10, 50),
                         cand_size=50,
                         N=8,
                         device="cpu",
                         seed=123):
    """
    Sampled eval（KNN q 向量）：
      - 对每个 eval 问题，用 TF-IDF 在训练问题上取 KNN；
      - 用这些训练问题的 BPR-Q 向量做相似度加权平均，得到该 eval 问题的 q 向量；
      - 候选集 = 正样本 ∪ 随机负样本（数量到 cand_size）；
      - 分数 = (A @ qv) + bias_a；
      - 计算 P/R/F1/Hit/nDCG/MRR。
    """
    max_k = max(ks)
    agg = {k: {"P": 0.0, "R": 0.0, "F1": 0.0, "Hit": 0.0, "nDCG": 0.0, "MRR": 0.0} for k in ks}
    cnt, skipped = 0, 0

    # 预取 agent 矩阵（numpy）与 bias
    A = model.emb_a.weight.detach().cpu().numpy()                     # (Na, F)
    bias_a = model.bias_a.weight.detach().cpu().numpy().squeeze(-1) if hasattr(model, "bias_a") else None

    all_agents = list(aid2idx.keys())
    all_agent_set = set(all_agents)
    ref_k = 10 if 10 in ks else max_k

    pbar = tqdm(eval_qids, desc="Evaluating (KNN q-vector)", leave=True, dynamic_ncols=True)
    for qid in pbar:
        gt = [aid for aid in all_rankings.get(qid, [])[:pos_topk] if aid in aid2idx]
        if not gt:
            skipped += 1
            pbar.set_postfix({"done": cnt, "skipped": skipped})
            continue

        rel_set = set(gt)
        neg_pool = list(all_agent_set - rel_set)
        rnd = random.Random((hash(qid) ^ (seed * 2654435761)) & 0xFFFFFFFF)

        need_neg = max(0, cand_size - len(gt))
        if need_neg > 0 and neg_pool:
            k = min(need_neg, len(neg_pool))
            sampled_negs = rnd.sample(neg_pool, k)
            cand = gt + sampled_negs
        else:
            cand = gt

        # —— KNN 得到该问题的 q 向量（在 BPR-Q 空间） ——
        qtext = all_questions.get(qid, {}).get("input", "")
        qv = knn_qvec_for_question_text(qtext, knn_cache_path, N=N)  # (F,)

        # 对候选打分（从全量 A@qv 中切片）
        s_all = (A @ qv)                                             # (Na,)
        if bias_a is not None:
            s_all = s_all + bias_a

        ai_idx = [aid2idx[a] for a in cand]
        s_sub = s_all[ai_idx]
        order = np.argsort(-s_sub)[:max_k]
        pred = [cand[i] for i in order]

        # metrics
        bin_hits = [1 if a in rel_set else 0 for a in pred]
        num_rel = len(rel_set)

        for k in ks:
            top = bin_hits[:k]
            Hk = int(sum(top))
            P = Hk / float(k)
            R = Hk / float(num_rel)
            F1 = (2 * P * R) / (P + R) if (P + R) else 0.0
            Hit = 1.0 if Hk > 0 else 0.0

            dcg = sum(1.0 / math.log2(i + 2.0) for i, h in enumerate(top) if h)
            idcg = _ideal_dcg(k, num_rel)
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
        pbar.set_postfix({
            "done": cnt,
            "skipped": skipped,
            f"P@{ref_k}": f"{(ref['P']/cnt):.4f}",
            f"nDCG@{ref_k}": f"{(ref['nDCG']/cnt):.4f}",
            f"MRR@{ref_k}": f"{(ref['MRR']/cnt):.4f}",
            "Ncand": len(cand)
        })

    if cnt == 0:
        return {k: {m: 0.0 for m in ["P", "R", "F1", "Hit", "nDCG", "MRR"]} for k in ks}

    for k in ks:
        for m in agg[k]:
            agg[k][m] /= cnt
    return agg


# ------------------------------- BPR-MF Model ------------------------------
class BPRMF(nn.Module):
    def __init__(self, num_q: int, num_a: int, factors: int = 128, add_bias: bool = True):
        super().__init__()
        self.emb_q = nn.Embedding(num_q, factors)
        self.emb_a = nn.Embedding(num_a, factors)
        self.add_bias = add_bias
        if add_bias:
            self.bias_q = nn.Embedding(num_q, 1)
            self.bias_a = nn.Embedding(num_a, 1)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.emb_q.weight)
        nn.init.xavier_uniform_(self.emb_a.weight)
        if self.add_bias:
            nn.init.zeros_(self.bias_q.weight)
            nn.init.zeros_(self.bias_a.weight)

    def score_embeddings(self, q_vec: torch.Tensor, a_vec: torch.Tensor,
                         qi: torch.Tensor = None, ai: torch.Tensor = None) -> torch.Tensor:
        """Dot product + optional biases. q_vec/a_vec: (..., F)"""
        s = (q_vec * a_vec).sum(dim=-1)
        if self.add_bias and qi is not None and ai is not None:
            s = s + self.bias_q(qi).squeeze(-1) + self.bias_a(ai).squeeze(-1)
        return s

    def score_q_a_indices(self, qi: int, ai: torch.LongTensor) -> torch.Tensor:
        """Score a single q index against many agent indices ai: returns (len(ai),)."""
        qv = self.emb_q(torch.tensor([qi], device=ai.device)).repeat(ai.size(0), 1)
        av = self.emb_a(ai)
        qi_t = torch.full((ai.size(0),), qi, dtype=torch.long, device=ai.device)
        return self.score_embeddings(qv, av, qi_t, ai)

    def forward(self, q_idx: torch.LongTensor, pos_idx: torch.LongTensor, neg_idx: torch.LongTensor) -> Tuple[torch.Tensor, torch.Tensor]:
        qv = self.emb_q(q_idx)        # (B, F)
        apv = self.emb_a(pos_idx)     # (B, F)
        anv = self.emb_a(neg_idx)     # (B, F)
        pos = self.score_embeddings(qv, apv, q_idx, pos_idx)
        neg = self.score_embeddings(qv, anv, q_idx, neg_idx)
        return pos, neg


def bpr_loss(pos: torch.Tensor, neg: torch.Tensor) -> torch.Tensor:
    return -torch.log(torch.sigmoid(pos - neg) + 1e-8).mean()


# ----------------------------------- Main ----------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True, help="Root folder containing PartI/PartII/PartIII(/Tools)")
    parser.add_argument("--factors", type=int, default=128, help="Embedding dimension")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--neg_per_pos", type=int, default=1)
    parser.add_argument("--rng_seed_pairs", type=int, default=42)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--valid_ratio", type=float, default=0.2)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--rebuild_training_cache", type=int, default=0, help="Force rebuild train/valid/pairs cache (0/1)")
    parser.add_argument("--knn_N", type=int, default=8, help="KNN 的邻居数量")
    parser.add_argument("--eval_cand_size", type=int, default=1000, help="评估时候的候选集大小（正样本 ∪ 随机负样本）")
    args = parser.parse_args()

    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)

    # Load data
    all_agents, all_questions, all_rankings = collect_data(args.data_root)
    print(f"Loaded {len(all_agents)} agents, {len(all_questions)} questions, {len(all_rankings)} ranked entries.")

    q_ids = list(all_questions.keys())
    a_ids = list(all_agents.keys())
    qid2idx = {qid: i for i, qid in enumerate(q_ids)}
    aid2idx = {aid: i for i, aid in enumerate(a_ids)}

    # Restrict to questions that have rankings
    qids_in_rank = [qid for qid in q_ids if qid in all_rankings]
    print(f"Questions with rankings: {len(qids_in_rank)} / {len(q_ids)}")

    cache_dir = ensure_cache_dir(args.data_root)

    # Build or load training cache (pairs + splits)
    pairs_idx_np = None
    split_paths = (
        os.path.join(cache_dir, "train_qids.json"),
        os.path.join(cache_dir, "valid_qids.json"),
        os.path.join(cache_dir, "pairs_train.npy"),
        os.path.join(cache_dir, "train_cache_meta.json"),
    )

    want_meta = {
        "data_sig": dataset_signature(qids_in_rank, a_ids, {k: all_rankings[k] for k in qids_in_rank}),
        "neg_per_pos": int(args.neg_per_pos),
        "rng_seed_pairs": int(args.rng_seed_pairs),
        "split_seed": int(args.split_seed),
        "valid_ratio": float(args.valid_ratio),
    }

    use_cache = all(os.path.exists(p) for p in split_paths) and (args.rebuild_training_cache == 0)
    if use_cache:
        with open(split_paths[0], "r", encoding="utf-8") as f:
            train_qids = json.load(f)
        with open(split_paths[1], "r", encoding="utf-8") as f:
            valid_qids = json.load(f)
        pairs_idx_np = np.load(split_paths[2])
        with open(split_paths[3], "r", encoding="utf-8") as f:
            meta = json.load(f)
        if meta != want_meta:
            print("[cache] training cache meta mismatch, rebuilding...")
            use_cache = False

    if not use_cache:
        train_qids, valid_qids = train_valid_split(qids_in_rank, valid_ratio=args.valid_ratio, seed=args.split_seed)
        print(f"[split] train={len(train_qids)}  valid={len(valid_qids)}")

        rankings_train = {qid: all_rankings[qid] for qid in train_qids}
        pairs = build_training_pairs(rankings_train, a_ids, neg_per_pos=args.neg_per_pos, rng_seed=args.rng_seed_pairs)
        # map to indices
        pairs_idx = [(qid2idx[q], aid2idx[p], aid2idx[n]) for (q, p, n) in pairs]
        pairs_idx_np = np.array(pairs_idx, dtype=np.int64)

        # save cache
        with open(split_paths[0], "w", encoding="utf-8") as f:
            json.dump(train_qids, f, ensure_ascii=False)
        with open(split_paths[1], "w", encoding="utf-8") as f:
            json.dump(valid_qids, f, ensure_ascii=False)
        np.save(split_paths[2], pairs_idx_np)
        with open(split_paths[3], "w", encoding="utf-8") as f:
            json.dump(want_meta, f, ensure_ascii=False, sort_keys=True)
        print(f"[cache] saved train/valid/pairs to {cache_dir}")

    # Model
    device = torch.device(args.device)
    model = BPRMF(num_q=len(q_ids), num_a=len(a_ids), factors=args.factors, add_bias=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Training
    pairs = pairs_idx_np.tolist()
    num_pairs = len(pairs)
    num_batches = math.ceil(num_pairs / args.batch_size)
    print(f"Training pairs: {num_pairs}, batches/epoch: {num_batches}")

    for epoch in range(1, args.epochs + 1):
        random.shuffle(pairs)
        total_loss = 0.0
        pbar = tqdm(range(num_batches), desc=f"Epoch {epoch}/{args.epochs}", leave=True, dynamic_ncols=True)
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

            total_loss += loss.item()
            pbar.set_postfix({"batch_loss": f"{loss.item():.4f}", "avg_loss": f"{(total_loss / (b + 1)):.4f}"})

        print(f"Epoch {epoch}/{args.epochs} - BPR loss: {(total_loss / num_batches if num_batches else 0.0):.4f}")

    # Save minimal checkpoint
    model_dir = os.path.join(cache_dir, "models")
    os.makedirs(model_dir, exist_ok=True)
    data_sig = want_meta["data_sig"]
    ckpt_path = os.path.join(model_dir, f"{filename}_{data_sig}.pt")
    latest_path = os.path.join(model_dir, f"latest_{filename}_{data_sig}.pt")

    ckpt = {
        "state_dict": model.state_dict(),
        "data_sig": data_sig,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "dims": {
            "num_q": int(len(q_ids)),
            "num_a": int(len(a_ids)),
            "factors": int(args.factors),
        },
        "mappings": {
            "q_ids": q_ids,
            "a_ids": a_ids,
        },
        "args": vars(args),
    }
    torch.save(ckpt, ckpt_path)
    torch.save(ckpt, latest_path)
    meta_path = os.path.join(model_dir, f"meta_{filename}_{data_sig}.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({"data_sig": data_sig, "q_ids": q_ids, "a_ids": a_ids}, f, ensure_ascii=False, indent=2)

    print(f"[save] model -> {ckpt_path}")
    print(f"[save] meta  -> {meta_path}")

    # ---------- Build & save KNN cache ----------
    # 1) 取训练集问题文本
    train_texts = [all_questions[qid]["input"] for qid in train_qids]
    # 2) TF-IDF 拟合并变换
    tfidf = TfidfVectorizer(max_features=5000)
    X = tfidf.fit_transform(train_texts).astype(np.float32)            # (Nq, V)
    # 3) 取训练好的 q-embedding
    Q = model.emb_q.weight.detach().cpu().numpy()[[qid2idx[q] for q in train_qids]]  # (Nq, F)
    # 4) 缓存到磁盘（注意保存 X）
    knn_cache = {"train_qids": train_qids, "Q": Q, "tfidf": tfidf, "X": X}
    with open(os.path.join(cache_dir, "knn_copy.pkl"), "wb") as f:
        pickle.dump(knn_cache, f)
    print("[cache] saved KNN-copy cache")

    # ---------- Validation (KNN-based q-vector) ----------
    knn_path = os.path.join(cache_dir, "knn_copy.pkl")
    valid_metrics = evaluate_sampled_knn(
        model, qid2idx, aid2idx, all_rankings, all_questions, valid_qids,
        knn_cache_path=knn_path,
        ks=(5, 10, 50), cand_size=args.eval_cand_size, N=args.knn_N,
        device=device, seed=123
    )
    print_metrics_table("Validation (KNN q-vector; averaged over questions)", valid_metrics, ks=(5, 10, 50), filename=filename)

    # ---------- Demo: recommend via KNN q-vector ----------
    def _score_agents_from_qvec(qv: np.ndarray, topk: int = 10):
        A = model.emb_a.weight.detach().cpu().numpy()
        s = A @ qv.astype(np.float32)
        if hasattr(model, "bias_a"):
            s = s + model.bias_a.weight.detach().cpu().numpy().squeeze(-1)
        order = np.argsort(-s)[:topk]
        return [(a_ids[i], float(s[i])) for i in order]

    def recommend_topk_for_qid_knn(qid: str, topk: int = 10, N: int = 8):
        qtext = all_questions[qid].get("input", "")
        qv = knn_qvec_for_question_text(qtext, knn_path, N=N)
        return _score_agents_from_qvec(qv, topk)

    sample_qids = qids_in_rank[:min(5, len(qids_in_rank))] or q_ids[:min(5, len(q_ids))]
    for qid in sample_qids:
        recs = recommend_topk_for_qid_knn(qid, topk=args.topk, N=args.knn_N)
        qtext = all_questions[qid].get("input", "")[:80].replace("\n", " ")
        print(f"\nQuestion: {qid} | {qtext}")
        for r, (aid, s) in enumerate(recs, 1):
            print(f"  {r:2d}. {aid:>20s}  score={s:.4f}")


if __name__ == "__main__":
    main()
