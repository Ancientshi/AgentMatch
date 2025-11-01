#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple LightFM (handwritten) Agent Recommender
- Hybrid MF with ID embeddings + sparse user/item feature embeddings (e.g., TF-IDF)
- BPR loss training, same dataset layout as your MF script.
- No dependency on the 'lightfm' Python package.

This version aligns its TEXT PIPELINE 100% with simple_lightfm_agent_rec.py:
  * Question text: questions[qid]["input"]
  * Tool text:     tools[name]["description"] with the tool name concatenated
  * Agent text:    agent["M"]["name"] + concatenated tool texts (names + desc)
  * Vectorizers:   three TF-IDF vectorizers (q_vec, tool_vec, a_vec)
  * Item features: concat(Agent TF-IDF, mean-pooled Tool TF-IDF) then L2 row-normalized
  * User features: Question TF-IDF, L2 row-normalized
  * Cache files:   Q.npy, A_text_full.npy, q_ids.json, a_ids.json, tool_names.json,
                   agent_tool_idx_padded.npy, agent_tool_mask.npy

Usage (TF-IDF auto-build):
  python simple_lightfm_handwritten.py \
    --data_root /path/to/dataset_root \
    --epochs 3 --batch_size 4096 --factors 128 --neg_per_pos 1 \
    --topk 10 --device cuda:0 --max_features 5000

Usage (load prebuilt features):
  # （可选）如果你想手动指定已构建的 CSR，仍可用下列开关，但默认建议走自动缓存对齐参照脚本
  --user_features_path /path/to/user_features_csr.pkl \
  --item_features_path /path/to/item_features_csr.pkl
"""

import os
import json
import math
import argparse
import random
import zlib
import pickle
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

try:
    from scipy import sparse as sp
except Exception as e:
    raise RuntimeError("This script needs SciPy for CSR features. pip install scipy\n" + str(e))

# ---------------------------------------------------------------------------
filename = os.path.splitext(os.path.basename(__file__))[0]
pos_topk = 5  # 只把排名前K的 agent 当做正样本
# ---------------------------------------------------------------------------

# ============== Small pretty table (local fallback) =========================
def print_metrics_table(title: str, metrics: Dict[int, Dict[str, float]], ks=(5, 10, 50), filename: str = None):
    print(f"\n== {title} ==")
    header = "  @K |       P       R      F1     Hit    nDCG     MRR"
    print(header)
    print("-" * len(header))
    for k in ks:
        m = metrics.get(k, {})
        print(f"{k:4d} | {m.get('P',0):.4f} {m.get('R',0):.4f} {m.get('F1',0):.4f} "
              f"{m.get('Hit',0):.4f} {m.get('nDCG',0):.4f} {m.get('MRR',0):.4f}")

# ------------------------- Cache paths & helpers ---------------------------
def ensure_cache_dir(root: str) -> str:
    d = os.path.join(root, f".cache/{filename}")
    os.makedirs(d, exist_ok=True)
    return d

def cache_exists(cache_dir: str) -> bool:
    needed = [
        "q_ids.json", "a_ids.json", "tool_names.json",
        "Q.npy", "A_text_full.npy", "agent_tool_idx_padded.npy", "agent_tool_mask.npy",
    ]
    return all(os.path.exists(os.path.join(cache_dir, f)) for f in needed)

def save_cache(cache_dir: str,
               q_ids, a_ids, tool_names,
               Q, A_text_full,
               agent_tool_idx_padded, agent_tool_mask):
    with open(os.path.join(cache_dir, "q_ids.json"), "w", encoding="utf-8") as f:
        json.dump(q_ids, f, ensure_ascii=False)
    with open(os.path.join(cache_dir, "a_ids.json"), "w", encoding="utf-8") as f:
        json.dump(a_ids, f, ensure_ascii=False)
    with open(os.path.join(cache_dir, "tool_names.json"), "w", encoding="utf-8") as f:
        json.dump(tool_names, f, ensure_ascii=False)

    np.save(os.path.join(cache_dir, "Q.npy"), Q.astype(np.float32))
    np.save(os.path.join(cache_dir, "A_text_full.npy"), A_text_full.astype(np.float32))
    np.save(os.path.join(cache_dir, "agent_tool_idx_padded.npy"), agent_tool_idx_padded.astype(np.int64))
    np.save(os.path.join(cache_dir, "agent_tool_mask.npy"), agent_tool_mask.astype(np.float32))

def load_cache(cache_dir: str):
    with open(os.path.join(cache_dir, "q_ids.json"), "r", encoding="utf-8") as f:
        q_ids = json.load(f)
    with open(os.path.join(cache_dir, "a_ids.json"), "r", encoding="utf-8") as f:
        a_ids = json.load(f)
    with open(os.path.join(cache_dir, "tool_names.json"), "r", encoding="utf-8") as f:
        tool_names = json.load(f)

    Q = np.load(os.path.join(cache_dir, "Q.npy"))
    A_text_full = np.load(os.path.join(cache_dir, "A_text_full.npy"))
    agent_tool_idx_padded = np.load(os.path.join(cache_dir, "agent_tool_idx_padded.npy"))
    agent_tool_mask = np.load(os.path.join(cache_dir, "agent_tool_mask.npy"))

    return (q_ids, a_ids, tool_names, Q, A_text_full, agent_tool_idx_padded, agent_tool_mask)

# ------------------------------- Data loading ------------------------------
def load_json(p: str):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def collect_data(data_root: str):
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
        all_rankings.update(rankings["rankings"])

    # Tools（必须存在，保持与参照脚本一致）
    tools_path = os.path.join(data_root, "Tools", "merge.json")
    tools = load_json(tools_path) if os.path.exists(tools_path) else {}

    return all_agents, all_questions, all_rankings, tools

# ---------------------- Text corpora & vectorizers (aligned) ---------------
def build_text_corpora(all_agents, all_questions, tools):
    """
    与 simple_lightfm_agent_rec.py 完全一致的文本构建：
      - Question: questions[qid]["input"]
      - Tool:     f"{tool_name} {tools[name]['description']}"
      - Agent:    agent["M"]["name"] + 拼接其 tool 的 (name + description)
    """
    # Questions
    q_ids = list(all_questions.keys())
    q_texts = [all_questions[qid].get("input", "") for qid in q_ids]

    # Tools
    tool_names = list(tools.keys())

    def _tool_text(tn: str) -> str:
        t = tools.get(tn, {}) or {}
        desc = t.get("description", "")
        return f"{tn} {desc}".strip()

    tool_texts = [_tool_text(tn) for tn in tool_names]

    # Agents
    a_ids = list(all_agents.keys())
    a_texts = []
    a_tool_lists = []
    for aid in a_ids:
        a = all_agents.get(aid, {}) or {}
        mname = ((a.get("M") or {}).get("name") or "").strip()
        tool_list = ((a.get("T") or {}).get("tools") or [])
        a_tool_lists.append(tool_list)

        concat_tool_desc = " ".join([_tool_text(tn) for tn in tool_list])
        text = f"{mname} {concat_tool_desc}".strip()
        a_texts.append(text)

    return q_ids, q_texts, tool_names, tool_texts, a_ids, a_texts, a_tool_lists

def build_vectorizers(q_texts, tool_texts, a_texts, max_features: int):
    q_vec = TfidfVectorizer(max_features=max_features, lowercase=True)
    tool_vec = TfidfVectorizer(max_features=max_features, lowercase=True)
    a_vec = TfidfVectorizer(max_features=max_features, lowercase=True)

    Q = q_vec.fit_transform(q_texts)        # (num_q, Dq)
    Tm = tool_vec.fit_transform(tool_texts) # (num_tools, Dt)
    Am = a_vec.fit_transform(a_texts)       # (num_agents, Da)
    return q_vec, tool_vec, a_vec, Q, Tm, Am

def agent_tool_text_matrix(agent_tool_lists: List[List[str]], tool_names: List[str], Tm) -> np.ndarray:
    """
    为每个 agent 对其 tools 的 TF-IDF 向量做均值。若无工具则为零向量。
    返回 dense (num_agents, Dt)
    """
    name2idx = {n: i for i, n in enumerate(tool_names)}
    num_agents = len(agent_tool_lists)
    Dt = Tm.shape[1]
    Atool = np.zeros((num_agents, Dt), dtype=np.float32)
    for i, tool_list in enumerate(agent_tool_lists):
        idxs = [name2idx[t] for t in tool_list if t in name2idx]
        if not idxs:
            continue
        vecs = Tm[idxs].toarray()
        Atool[i] = vecs.mean(axis=0).astype(np.float32)
    return Atool

# ------------------------- CSR -> EmbeddingBag inputs ----------------------
def csr_to_bag_lists(csr: "sp.csr_matrix"):
    indptr, indices, data = csr.indptr, csr.indices, csr.data
    rows = csr.shape[0]
    feats_per_row, vals_per_row = [], []
    for r in range(rows):
        s, e = indptr[r], indptr[r+1]
        feats_per_row.append(indices[s:e].tolist())
        vals_per_row.append(data[s:e].astype(np.float32).tolist())
    return feats_per_row, vals_per_row, csr.shape[1]

def build_bag_tensors(batch_rows, feats_per_row, vals_per_row, device):
    idx_cat, w_cat, offsets = [], [], [0]
    total = 0
    for r in batch_rows:
        f = feats_per_row[r]
        v = vals_per_row[r]
        idx_cat.extend(f)
        w_cat.extend(v)
        total += len(f)
        offsets.append(total)
    if total == 0:
        idx_cat = [0]
        w_cat = [0.0]
        offsets = [0, 1]

    idx_t = torch.tensor(idx_cat, dtype=torch.long, device=device)
    # include_last_offset=True 时，offsets 需要长度 = num_bags + 1
    off_t = torch.tensor(offsets, dtype=torch.long, device=device)
    w_t = torch.tensor(w_cat, dtype=torch.float32, device=device)
    return idx_t, off_t, w_t


# ------------------------------- Evaluation helpers ------------------------
def _ideal_dcg(k, num_rel):
    ideal = min(k, num_rel)
    return sum(1.0 / math.log2(i + 2.0) for i in range(ideal)) if ideal else 0.0

def knn_qvec_for_question_text(question_text: str, knn_cache_path: str, N: int = 8) -> np.ndarray:
    """
    使用与训练时一致的 q_vectorizer，从 KNN 缓存里取：
      tfidf: 已拟合的 q_vec
      X:     训练问题文本的 TF-IDF 矩阵
      Q:     训练问题的“混合用户表示”（ID+特征） -> 用于加权平均
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
    qv = (w[:, None] * Q[idx]).sum(axis=0)
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
    max_k = max(ks)
    agg = {k: {"P": 0.0, "R": 0.0, "F1": 0.0, "Hit": 0.0, "nDCG": 0.0, "MRR": 0.0} for k in ks}
    cnt, skipped = 0, 0

    A = model.item_repr_all().detach().cpu().numpy()           # (Na, F)
    bias_a = model.bias_a.weight.detach().cpu().numpy().squeeze(-1) if model.add_bias else None

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

        qtext = all_questions.get(qid, {}).get("input", "")
        qv = knn_qvec_for_question_text(qtext, knn_cache_path, N=N)  # (F,)

        s_all = (A @ qv)
        if bias_a is not None:
            s_all = s_all + bias_a
        ai_idx = [aid2idx[a] for a in cand]
        s_sub = s_all[ai_idx]
        order = np.argsort(-s_sub)[:max_k]
        pred = [cand[i] for i in order]

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
            "done": cnt, "skipped": skipped,
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

# ------------------------------- Model -------------------------------------
class LightFMHandwritten(nn.Module):
    """
    u_vec = alpha_id * emb_q[q] + alpha_feat * EmbBag_u(user_feat_indices, weights)
    i_vec = alpha_id * emb_a[a] + alpha_feat * EmbBag_i(item_feat_indices, weights)
    score = dot(u_vec, i_vec) + biases
    """
    def __init__(self,
                 num_q: int,
                 num_a: int,
                 num_user_feats: int,
                 num_item_feats: int,
                 factors: int = 128,
                 add_bias: bool = True,
                 alpha_id: float = 1.0,
                 alpha_feat: float = 1.0,
                 device: torch.device = torch.device("cpu")):
        super().__init__()
        self.add_bias = add_bias
        self.alpha_id = nn.Parameter(torch.tensor(alpha_id, dtype=torch.float32))
        self.alpha_feat = nn.Parameter(torch.tensor(alpha_feat, dtype=torch.float32))

        self.emb_q = nn.Embedding(num_q, factors)
        self.emb_a = nn.Embedding(num_a, factors)

        # EmbeddingBag: sum weighted features
        self.emb_user_feat = nn.EmbeddingBag(num_user_feats, factors, mode='sum', include_last_offset=True)
        self.emb_item_feat = nn.EmbeddingBag(num_item_feats, factors, mode='sum', include_last_offset=True)

        if add_bias:
            self.bias_q = nn.Embedding(num_q, 1)
            self.bias_a = nn.Embedding(num_a, 1)

        self.reset_parameters()
        self.device = device

        # set_* 接口里赋值（bag 索引缓存）
        self.user_feats_per_row = None
        self.user_vals_per_row = None
        self.item_feats_per_row = None
        self.item_vals_per_row = None

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.emb_q.weight)
        nn.init.xavier_uniform_(self.emb_a.weight)
        nn.init.xavier_uniform_(self.emb_user_feat.weight)
        nn.init.xavier_uniform_(self.emb_item_feat.weight)
        if self.add_bias:
            nn.init.zeros_(self.bias_q.weight)
            nn.init.zeros_(self.bias_a.weight)

    def set_user_feat_lists(self, feats_per_row, vals_per_row):
        self.user_feats_per_row = feats_per_row
        self.user_vals_per_row = vals_per_row

    def set_item_feat_lists(self, feats_per_row, vals_per_row):
        self.item_feats_per_row = feats_per_row
        self.item_vals_per_row = vals_per_row

    def _bag_embed_users(self, q_idx: torch.LongTensor):
        idx_t, off_t, w_t = build_bag_tensors(
            batch_rows=q_idx.tolist(),
            feats_per_row=self.user_feats_per_row,
            vals_per_row=self.user_vals_per_row,
            device=self.device
        )
        return self.emb_user_feat(idx_t, off_t, per_sample_weights=w_t)

    def _bag_embed_items(self, a_idx: torch.LongTensor):
        idx_t, off_t, w_t = build_bag_tensors(
            batch_rows=a_idx.tolist(),
            feats_per_row=self.item_feats_per_row,
            vals_per_row=self.item_vals_per_row,
            device=self.device
        )
        return self.emb_item_feat(idx_t, off_t, per_sample_weights=w_t)

    def user_repr_batch(self, q_idx: torch.LongTensor):
        u_id = self.emb_q(q_idx)
        u_feat = self._bag_embed_users(q_idx)
        return self.alpha_id * u_id + self.alpha_feat * u_feat

    def item_repr_batch(self, a_idx: torch.LongTensor):
        i_id = self.emb_a(a_idx)
        i_feat = self._bag_embed_items(a_idx)
        return self.alpha_id * i_id + self.alpha_feat * i_feat

    @torch.no_grad()
    def item_repr_all(self):
        Na = self.emb_a.num_embeddings
        batch = 4096
        outs = []
        for s in range(0, Na, batch):
            e = min(Na, s + batch)
            idx = torch.arange(s, e, device=self.device, dtype=torch.long)
            outs.append(self.item_repr_batch(idx))
        return torch.cat(outs, dim=0)

    def forward(self, q_idx: torch.LongTensor, pos_idx: torch.LongTensor, neg_idx: torch.LongTensor):
        qv = self.user_repr_batch(q_idx)        # (B, F)
        apv = self.item_repr_batch(pos_idx)     # (B, F)
        anv = self.item_repr_batch(neg_idx)     # (B, F)

        pos = (qv * apv).sum(dim=-1)
        neg = (qv * anv).sum(dim=-1)

        if self.add_bias:
            pos = pos + self.bias_q(q_idx).squeeze(-1) + self.bias_a(pos_idx).squeeze(-1)
            neg = neg + self.bias_q(q_idx).squeeze(-1) + self.bias_a(neg_idx).squeeze(-1)
        return pos, neg

def bpr_loss(pos: torch.Tensor, neg: torch.Tensor) -> torch.Tensor:
    return -torch.log(torch.sigmoid(pos - neg) + 1e-8).mean()

# ---------------------- Dataset signature & pairs build ---------------------
def dataset_signature(q_ids: List[str], a_ids: List[str], rankings: Dict[str, List[str]]) -> str:
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

# ------------------------------ Main (updated) -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--factors", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--neg_per_pos", type=int, default=1)
    parser.add_argument("--rng_seed_pairs", type=int, default=42)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--valid_ratio", type=float, default=0.2)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--device", type=str, default="cpu")
    # hybrid specific
    parser.add_argument("--alpha_id", type=float, default=1.0)
    parser.add_argument("--alpha_feat", type=float, default=1.0)
    # feature building (aligned with reference)
    parser.add_argument("--max_features", type=int, default=5000)
    parser.add_argument("--rebuild_feature_cache", type=int, default=0)
    # legacy optional external features (kept for compatibility)
    parser.add_argument("--user_features_path", type=str, default="")
    parser.add_argument("--item_features_path", type=str, default="")
    # eval
    parser.add_argument("--knn_N", type=int, default=8)
    parser.add_argument("--eval_cand_size", type=int, default=1000)
    args = parser.parse_args()

    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)

    # Load data
    all_agents, all_questions, all_rankings, tools = collect_data(args.data_root)
    print(f"Loaded {len(all_agents)} agents, {len(all_questions)} questions, {len(all_rankings)} rankings, {len(tools)} tools.")

    q_ids_all = list(all_questions.keys())
    a_ids_all = list(all_agents.keys())
    qid2idx = {qid: i for i, qid in enumerate(q_ids_all)}
    aid2idx = {aid: i for i, aid in enumerate(a_ids_all)}

    qids_in_rank = [qid for qid in q_ids_all if qid in all_rankings]
    print(f"Questions with rankings: {len(qids_in_rank)} / {len(q_ids_all)}")

    cache_dir = ensure_cache_dir(args.data_root)

    # ----------------- Build/Load features (aligned pipeline) ----------------
    q_vectorizer_runtime = None  # 保存以供 KNN 复用
    if (not args.user_features_path) and (not args.item_features_path):
        if cache_exists(cache_dir) and args.rebuild_feature_cache == 0:
            (q_ids, a_ids, tool_names, Q_np, A_text_full_np,
             agent_tool_idx_padded_np, agent_tool_mask_np) = load_cache(cache_dir)
            print(f"[cache] loaded features from {cache_dir}")
        else:
            q_ids, q_texts, tool_names, tool_texts, a_ids, a_texts, a_tool_lists = build_text_corpora(
                all_agents, all_questions, tools
            )
            q_vec, tool_vec, a_vec, Q_csr, Tm_csr, Am_csr = build_vectorizers(
                q_texts, tool_texts, a_texts, args.max_features
            )
            Atool = agent_tool_text_matrix(a_tool_lists, tool_names, Tm_csr)              # (Na, Dt)
            Am = Am_csr.toarray().astype(np.float32)                                      # (Na, Da)
            A_text_full_np = np.concatenate([Am, Atool], axis=1).astype(np.float32)       # (Na, Da+Dt)
            Q_np = Q_csr.toarray().astype(np.float32)                                     # (Nq, Dq)

            # 占位的 tool id 缓冲（保持管线一致）
            def build_agent_tool_id_buffers(a_ids: List[str],
                                            agent_tool_lists: List[List[str]],
                                            tool_names: List[str]) -> Tuple[np.ndarray, np.ndarray]:
                t_map = {n: i for i, n in enumerate(tool_names)}
                num_agents = len(a_ids)
                max_t = max([len(lst) for lst in agent_tool_lists]) if num_agents > 0 else 0
                if max_t == 0:
                    max_t = 1
                idx_pad = np.zeros((num_agents, max_t), dtype=np.int64)
                mask = np.zeros((num_agents, max_t), dtype=np.float32)
                for i, lst in enumerate(agent_tool_lists):
                    for j, tn in enumerate(lst[:max_t]):
                        if tn in t_map:
                            idx_pad[i, j] = t_map[tn]
                            mask[i, j] = 1.0
                return idx_pad, mask

            agent_tool_idx_padded_np, agent_tool_mask_np = build_agent_tool_id_buffers(a_ids, a_tool_lists, tool_names)

            save_cache(cache_dir, q_ids, a_ids, tool_names, Q_np, A_text_full_np,
                       agent_tool_idx_padded_np, agent_tool_mask_np)
            print(f"[cache] saved features to {cache_dir}")

            q_vectorizer_runtime = q_vec  # 保存 question 向量器以便 KNN
    else:
        # 兼容路径：从自定义 CSR pkl 读取（不推荐，除非你已有自建特征）
        with open(args.user_features_path, "rb") as f:
            U = pickle.load(f)
        with open(args.item_features_path, "rb") as f:
            V = pickle.load(f)
        assert U.shape[0] == len(q_ids_all) and V.shape[0] == len(a_ids_all), "CSR rows mismatch."
        Q_np = U.toarray().astype(np.float32)
        A_text_full_np = V.toarray().astype(np.float32)
        q_ids = q_ids_all
        a_ids = a_ids_all
        tool_names = list(tools.keys())
        agent_tool_idx_padded_np = np.zeros((len(a_ids), 1), dtype=np.int64)
        agent_tool_mask_np = np.zeros((len(a_ids), 1), dtype=np.float32)

    # 与参照脚本一致：先 L2 行归一化，再转 CSR
    Q_np = normalize(Q_np, norm="l2", axis=1, copy=False)
    A_text_full_np = normalize(A_text_full_np, norm="l2", axis=1, copy=False)
    U_csr = sp.csr_matrix(Q_np, dtype=np.float32)
    V_csr = sp.csr_matrix(A_text_full_np, dtype=np.float32)
    print(f"[features] user_features: {U_csr.shape}, item_features: {V_csr.shape}")

    # Convert CSR -> per-row lists for EmbeddingBag
    u_feats_per_row, u_vals_per_row, num_user_feats = csr_to_bag_lists(U_csr)
    i_feats_per_row, i_vals_per_row, num_item_feats = csr_to_bag_lists(V_csr)

    # ------------------------------ Training data ---------------------------
    data_sig = dataset_signature(qids_in_rank, a_ids, {k: all_rankings[k] for k in qids_in_rank})
    split_paths = (
        os.path.join(cache_dir, "train_qids.json"),
        os.path.join(cache_dir, "valid_qids.json"),
        os.path.join(cache_dir, "pairs_train.npy"),
        os.path.join(cache_dir, "train_cache_meta.json"),
    )
    want_meta = {
        "data_sig": data_sig,
        "neg_per_pos": int(args.neg_per_pos),
        "rng_seed_pairs": int(args.rng_seed_pairs),
        "split_seed": int(args.split_seed),
        "valid_ratio": float(args.valid_ratio),
    }
    use_cache = all(os.path.exists(p) for p in split_paths)
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
        pairs_idx = [(qid2idx[q], aid2idx[p], aid2idx[n]) for (q, p, n) in pairs]
        pairs_idx_np = np.array(pairs_idx, dtype=np.int64)
        with open(split_paths[0], "w", encoding="utf-8") as f:
            json.dump(train_qids, f, ensure_ascii=False)
        with open(split_paths[1], "w", encoding="utf-8") as f:
            json.dump(valid_qids, f, ensure_ascii=False)
        np.save(split_paths[2], pairs_idx_np)
        with open(split_paths[3], "w", encoding="utf-8") as f:
            json.dump(want_meta, f, ensure_ascii=False, sort_keys=True)
        print(f"[cache] saved train/valid/pairs to {cache_dir}")

    # ------------------------------- Model ----------------------------------
    device = torch.device(args.device)
    model = LightFMHandwritten(
        num_q=len(q_ids_all),
        num_a=len(a_ids),
        num_user_feats=U_csr.shape[1],
        num_item_feats=V_csr.shape[1],
        factors=args.factors,
        add_bias=True,
        alpha_id=args.alpha_id,
        alpha_feat=args.alpha_feat,
        device=device
    ).to(device)
    model.set_user_feat_lists(u_feats_per_row, u_vals_per_row)
    model.set_item_feat_lists(i_feats_per_row, i_vals_per_row)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
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

    # Save checkpoint
    model_dir = os.path.join(cache_dir, "models")
    os.makedirs(model_dir, exist_ok=True)
    ck_sig = dataset_signature(qids_in_rank, a_ids, {k: all_rankings[k] for k in qids_in_rank})
    ckpt_path = os.path.join(model_dir, f"{filename}_{ck_sig}.pt")
    latest_path = os.path.join(model_dir, f"latest_{filename}_{ck_sig}.pt")

    ckpt = {
        "state_dict": model.state_dict(),
        "data_sig": ck_sig,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "dims": {
            "num_q": int(len(q_ids_all)),
            "num_a": int(len(a_ids)),
            "factors": int(args.factors),
        },
        "mappings": {
            "q_ids": q_ids_all,
            "a_ids": a_ids,
        },
        "args": vars(args),
    }
    torch.save(ckpt, ckpt_path)
    torch.save(ckpt, latest_path)
    meta_path = os.path.join(model_dir, f"meta_{filename}_{ck_sig}.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({"data_sig": ck_sig, "q_ids": q_ids_all, "a_ids": a_ids}, f, ensure_ascii=False, indent=2)

    print(f"[save] model -> {ckpt_path}")
    print(f"[save] meta  -> {meta_path}")

    # ---------- Build & save KNN cache (uses SAME q_vectorizer) ----------
    # 若本次是从缓存直接载入（没有刚拟合 q_vec），此处尝试从磁盘加载 q_vec；否则保存之
    qvec_path = os.path.join(model_dir, f"{filename}_qvec_{ck_sig}.pkl")
    latest_qvec = os.path.join(model_dir, f"latest_{filename}_qvec_{ck_sig}.pkl")
    if q_vectorizer_runtime is None and os.path.exists(qvec_path):
        with open(qvec_path, "rb") as f:
            q_vectorizer_runtime = pickle.load(f)
    elif q_vectorizer_runtime is not None:
        with open(qvec_path, "wb") as f:
            pickle.dump(q_vectorizer_runtime, f)
        with open(latest_qvec, "wb") as f:
            pickle.dump(q_vectorizer_runtime, f)

    # KNN 缓存里的 X 来自“训练集问题文本”的 TF-IDF；Q 是相应问题的混合表示（ID+特征）
    train_qids_path = os.path.join(cache_dir, "train_qids.json")
    with open(train_qids_path, "r", encoding="utf-8") as f:
        train_qids = json.load(f)

    train_texts = [all_questions[qid]["input"] for qid in train_qids]
    if q_vectorizer_runtime is None:
        # 理论上不该发生：兜底
        q_vectorizer_runtime = TfidfVectorizer(max_features=args.max_features, lowercase=True).fit(train_texts)
    X_knn = q_vectorizer_runtime.transform(train_texts).astype(np.float32)

    with torch.no_grad():
        tq_idx = torch.tensor([qid2idx[q] for q in train_qids], dtype=torch.long, device=device)
        B = 4096
        chunks = []
        for s in range(0, tq_idx.numel(), B):
            e = min(tq_idx.numel(), s + B)
            chunks.append(model.user_repr_batch(tq_idx[s:e]).detach().cpu().numpy())
        Q_train = np.vstack(chunks)

    knn_cache = {"train_qids": train_qids, "Q": Q_train, "tfidf": q_vectorizer_runtime, "X": X_knn}
    with open(os.path.join(cache_dir, "knn_copy.pkl"), "wb") as f:
        pickle.dump(knn_cache, f)
    print("[cache] saved KNN-copy cache (aligned q_vectorizer)")

    # ---------- Validation (KNN-based q-vector) ----------
    knn_path = os.path.join(cache_dir, "knn_copy.pkl")
    valid_metrics = evaluate_sampled_knn(
        model, qid2idx, {aid: i for i, aid in enumerate(a_ids)}, all_rankings, all_questions, valid_qids,
        knn_cache_path=knn_path,
        ks=(5, 10, 50), cand_size=args.eval_cand_size, N=args.knn_N,
        device=device, seed=123
    )
    print_metrics_table("Validation (KNN q-vector; averaged over questions)", valid_metrics, ks=(5, 10, 50), filename=filename)

    # ---------- Demo ----------
    @torch.no_grad()
    def _score_agents_from_qvec(qv: np.ndarray, topk: int = 10):
        A = model.item_repr_all().detach().cpu().numpy()
        s = A @ qv.astype(np.float32)
        if model.add_bias:
            s = s + model.bias_a.weight.detach().cpu().numpy().squeeze(-1)
        order = np.argsort(-s)[:topk]
        return [(a_ids[i], float(s[i])) for i in order]

    def recommend_topk_for_qid_knn(qid: str, topk: int = 10, N: int = 8):
        qtext = all_questions[qid].get("input", "")
        qv = knn_qvec_for_question_text(qtext, knn_path, N=N)
        return _score_agents_from_qvec(qv, topk)

    sample_qids = qids_in_rank[:min(5, len(qids_in_rank))] or q_ids_all[:min(5, len(q_ids_all))]
    for qid in sample_qids:
        recs = recommend_topk_for_qid_knn(qid, topk=args.topk, N=args.knn_N)
        qtext = all_questions[qid].get("input", "")[:80].replace("\n", " ")
        print(f"\nQuestion: {qid} | {qtext}")
        for r, (aid, s) in enumerate(recs, 1):
            print(f"  {r:2d}. {aid:>20s}  score={s:.4f}")

if __name__ == "__main__":
    main()
