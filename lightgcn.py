#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
simple_lightgcn_agent_rec_tfidf.py
LightGCN for Query ↔ Agent (implicit) with BPR loss, **TF-IDF-initialized** embeddings
+ KNN-based cold-start for questions at eval/pred time (graph_scope=train).

Example:
  python simple_lightgcn_agent_rec_tfidf.py \
      --data_root /path/to/dataset_root \
      --epochs 10 --batch_size 4096 --emb_dim 128 --layers 3 \
      --neg_per_pos 1 --reg 1e-4 --topk 10 --device cuda:0 \
      --init_from_tfidf 1 --tfidf_fit_scope train --graph_scope train \
      --knn_N 8 --eval_cand_size 1000
"""
import os, json, math, random, argparse, zlib, pickle, warnings
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

from utils import print_metrics_table

# --------- Globals / small utils ----------
pos_topk = 5   # only regard top-k ranked agents as positives
filename = os.path.splitext(os.path.basename(__file__))[0]

def ensure_cache_dir(root: str) -> str:
    d = os.path.join(root, f".cache/{filename}")
    os.makedirs(d, exist_ok=True)
    return d

def load_json(p: str):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def collect_data(data_root: str):
    parts = ["PartI", "PartII", "PartIII"]
    all_agents: Dict[str, dict] = {}
    all_questions: Dict[str, dict] = {}
    all_rankings: Dict[str, List[str]] = {}
    for part in parts:
        agents = load_json(os.path.join(data_root, part, "agents", "merge.json"))
        questions = load_json(os.path.join(data_root, part, "questions", "merge.json"))
        rankings = load_json(os.path.join(data_root, part, "rankings", "merge.json"))
        all_agents.update(agents)
        all_questions.update(questions)
        all_rankings.update(rankings["rankings"])
    tools = load_json(os.path.join(data_root, "Tools", "merge.json"))
    return all_agents, all_questions, all_rankings, tools

def get_query_text(qdict):   # qdict = all_questions[qid]
    return (qdict.get("input") or "").strip()

def _tool_text(tools: dict, tn: str) -> str:
    t = tools.get(tn, {})
    desc = t.get("description", "")
    return f"{tn} {desc}".strip()

def get_agent_text(adict, tools: dict):
    mname = adict.get("M", {}).get("name", "")
    tool_list = adict.get("T", {}).get("tools", []) or []
    concat_tool_desc = " ".join([_tool_text(tools, tn) for tn in tool_list])
    return f"{mname} {concat_tool_desc}".strip(), tool_list

def build_text_corpora(all_agents, all_questions, tools):
    # Questions
    q_ids = list(all_questions.keys())
    q_texts = [get_query_text(all_questions[qid]) for qid in q_ids]
    # Tools
    tool_names = list(tools.keys())
    tool_texts = [_tool_text(tools, tn) for tn in tool_names]
    # Agents
    a_ids = list(all_agents.keys())
    a_texts, a_tool_lists = [], []
    for aid in a_ids:
        text, tool_list = get_agent_text(all_agents[aid], tools)
        a_texts.append(text)
        a_tool_lists.append(tool_list)
    return q_ids, q_texts, tool_names, tool_texts, a_ids, a_texts, a_tool_lists

def build_vectorizers(q_texts, tool_texts, a_texts, max_features: int):
    q_vec = TfidfVectorizer(max_features=max_features, lowercase=True)
    tool_vec = TfidfVectorizer(max_features=max_features, lowercase=True)
    a_vec = TfidfVectorizer(max_features=max_features, lowercase=True)
    Q = q_vec.fit_transform(q_texts)            # (num_q, Dq)
    Tm = tool_vec.fit_transform(tool_texts)     # (num_tools, Dt)
    Am = a_vec.fit_transform(a_texts)           # (num_agents, Da)
    return q_vec, tool_vec, a_vec, Q, Tm, Am

def agent_tool_text_matrix(agent_tool_lists: List[List[str]], tool_names: List[str], Tm) -> np.ndarray:
    """Mean-aggregate tool TF-IDF per agent; zeros if no tools."""
    name2idx = {n: i for i, n in enumerate(tool_names)}
    num_agents = len(agent_tool_lists)
    Dt = Tm.shape[1]
    Atool = np.zeros((num_agents, Dt), dtype=np.float32)
    for i, tool_list in enumerate(agent_tool_lists):
        idxs = [name2idx[t] for t in tool_list if t in name2idx]
        if not idxs: continue
        vecs = Tm[idxs].toarray()
        Atool[i] = vecs.mean(axis=0).astype(np.float32)
    return Atool

def build_agent_tool_id_buffers(a_ids: List[str],
                                agent_tool_lists: List[List[str]],
                                tool_names: List[str]) -> Tuple[torch.LongTensor, torch.FloatTensor]:
    """Padded tool indices & mask per agent. Shapes: (num_agents, max_tools_per_agent)"""
    t_map = {n: i for i, n in enumerate(tool_names)}
    num_agents = len(a_ids)
    max_t = max([len(lst) for lst in agent_tool_lists]) if num_agents > 0 else 0
    if max_t == 0: max_t = 1
    idx_pad = np.zeros((num_agents, max_t), dtype=np.int64)
    mask = np.zeros((num_agents, max_t), dtype=np.float32)
    for i, lst in enumerate(agent_tool_lists):
        for j, tn in enumerate(lst[:max_t]):
            if tn in t_map:
                idx_pad[i, j] = t_map[tn]; mask[i, j] = 1.0
    return torch.from_numpy(idx_pad), torch.from_numpy(mask)

# ---------- Cache helpers ----------
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
    return (q_ids, a_ids, tool_names, Q, A_text_full,
            agent_tool_idx_padded, agent_tool_mask)

# ---------- Train/valid split & pairs ----------
def train_valid_split(qids_in_rankings, valid_ratio=0.2, seed=42):
    rng = random.Random(seed)
    q = list(qids_in_rankings); rng.shuffle(q)
    n_valid = int(len(q) * valid_ratio)
    return q[n_valid:], q[:n_valid]  # train, valid

def build_training_pairs(all_rankings: Dict[str, List[str]],
                         all_agent_ids: List[str],
                         neg_per_pos: int = 1,
                         rng_seed: int = 42):
    rnd = random.Random(rng_seed)
    pairs = []
    all_agent_set = set(all_agent_ids)
    for qid, ranked_full in all_rankings.items():
        ranked = ranked_full[:pos_topk]
        if not ranked: continue
        negatives_pool = list(all_agent_set - set(ranked)) or all_agent_ids
        for pos_a in ranked:
            for _ in range(neg_per_pos):
                neg_a = rnd.choice(negatives_pool)
                pairs.append((qid, pos_a, neg_a))
    return pairs

# -------------------- Dataset signature & training cache --------------------
def dataset_signature(a_ids: List[str], all_rankings: Dict[str, List[str]]) -> str:
    payload = {"a_ids": a_ids, "rankings": {k: all_rankings[k] for k in sorted(all_rankings.keys())}}
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return f"{(zlib.crc32(blob) & 0xFFFFFFFF):08x}"

def training_cache_paths(cache_dir: str):
    return (
        os.path.join(cache_dir, "train_qids.json"),
        os.path.join(cache_dir, "valid_qids.json"),
        os.path.join(cache_dir, "pairs_train.npy"),
        os.path.join(cache_dir, "train_cache_meta.json"),
    )

def training_cache_exists(cache_dir: str) -> bool:
    p_train, p_valid, p_pairs, p_meta = training_cache_paths(cache_dir)
    return all(os.path.exists(p) for p in (p_train, p_valid, p_pairs, p_meta))

def save_training_cache(cache_dir: str, train_qids, valid_qids, pairs_idx_np, meta):
    p_train, p_valid, p_pairs, p_meta = training_cache_paths(cache_dir)
    with open(p_train, "w", encoding="utf-8") as f: json.dump(train_qids, f, ensure_ascii=False)
    with open(p_valid, "w", encoding="utf-8") as f: json.dump(valid_qids, f, ensure_ascii=False)
    np.save(p_pairs, pairs_idx_np.astype(np.int64))
    with open(p_meta, "w", encoding="utf-8") as f: json.dump(meta, f, ensure_ascii=False, sort_keys=True)

def load_training_cache(cache_dir: str):
    p_train, p_valid, p_pairs, p_meta = training_cache_paths(cache_dir)
    with open(p_train, "r", encoding="utf-8") as f: train_qids = json.load(f)
    with open(p_valid, "r", encoding="utf-8") as f: valid_qids = json.load(f)
    pairs_idx_np = np.load(p_pairs)
    with open(p_meta, "r", encoding="utf-8") as f: meta = json.load(f)
    return train_qids, valid_qids, pairs_idx_np, meta

# ---------- LightGCN ----------
class LightGCN(nn.Module):
    def __init__(self, num_users: int, num_items: int, emb_dim: int, n_layers: int,
                 norm_adj: torch.sparse.FloatTensor, reg: float = 1e-4, device="cpu"):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.emb_dim = emb_dim
        self.n_layers = n_layers
        self.reg = reg
        self.device = device
        self.norm_adj = norm_adj.coalesce().to(device)
        # user/item embeddings
        self.user_emb = nn.Embedding(num_users, emb_dim)
        self.item_emb = nn.Embedding(num_items, emb_dim)
        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.item_emb.weight)

    def propagate(self):
        # Concatenate user/item embeddings -> graph signal
        x0 = torch.cat([self.user_emb.weight, self.item_emb.weight], dim=0)  # (U+I, D)
        xs = [x0]
        x = x0
        for _ in range(self.n_layers):
            x = torch.sparse.mm(self.norm_adj, x)
            xs.append(x)
        out = torch.stack(xs, dim=0).mean(0)
        u_all, i_all = torch.split(out, [self.num_users, self.num_items], dim=0)
        return u_all, i_all

    def get_user_item(self):
        return self.propagate()

    def score(self, u_idx: torch.Tensor, i_idx: torch.Tensor):
        U, I = self.get_user_item()
        u = U[u_idx]
        i = I[i_idx]
        return (u * i).sum(dim=1)

    def bpr_loss(self, u_idx: torch.Tensor, pos_idx: torch.Tensor, neg_idx: torch.Tensor):
        U_all, I_all = self.get_user_item()
        u = U_all[u_idx]            # (B, D)
        pos = I_all[pos_idx]        # (B, D)
        neg = I_all[neg_idx]        # (B, D)
        pos_scores = (u * pos).sum(dim=1)
        neg_scores = (u * neg).sum(dim=1)
        loss_bpr = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8).mean()
        # L2 reg on raw embeddings (as in paper)
        reg = (u.norm(p=2, dim=1).pow(2).mean() +
               pos.norm(p=2, dim=1).pow(2).mean() +
               neg.norm(p=2, dim=1).pow(2).mean())
        return loss_bpr + self.reg * reg

# ---------- Graph building ----------
def build_norm_adj(num_users: int, num_items: int, user_pos: List[Tuple[int, int]]):
    """
    user_pos: list of (u_idx, i_idx) positive edges.
    Returns symmetric normalized adjacency (U+I) x (U+I) sparse tensor.
    """
    U, I = num_users, num_items
    N = U + I
    rows, cols = [], []
    for (u, i) in user_pos:
        ui = U + i
        rows.extend([u, ui])
        cols.extend([ui, u])
    if not rows:
        idx = torch.tensor([[0],[0]], dtype=torch.long)
        val = torch.tensor([0.0], dtype=torch.float32)
        return torch.sparse_coo_tensor(idx, val, (N, N))
    indices = torch.tensor([rows, cols], dtype=torch.long)
    values = torch.ones(len(rows), dtype=torch.float32)
    A = torch.sparse_coo_tensor(indices, values, (N, N))
    deg = torch.sparse.sum(A, dim=1).to_dense()  # (N,)
    deg_inv_sqrt = torch.pow(deg + 1e-8, -0.5)
    di, dj = indices[0], indices[1]
    norm_vals = deg_inv_sqrt[di] * values * deg_inv_sqrt[dj]
    A_norm = torch.sparse_coo_tensor(indices, norm_vals, (N, N))
    return A_norm.coalesce()

# ---------- Evaluation (sampled) ----------
def _ideal_dcg(k, num_rel):
    ideal = min(k, num_rel)
    return sum(1.0 / math.log2(i + 2.0) for i in range(ideal)) if ideal else 0.0

# === KNN cold-start additions ===
def _build_user_degree_mask(num_users: int, train_edges: List[Tuple[int,int]]) -> np.ndarray:
    deg = np.zeros((num_users,), dtype=np.int64)
    for u, _ in train_edges:
        deg[u] += 1
    return (deg > 0)

def _save_knn_cache(cache_dir: str, train_qids: List[str], train_q_texts: List[str],
                    U_train: np.ndarray, max_features: int = 5000, path_name: str = "knn_q_cache.pkl"):
    tfidf = TfidfVectorizer(max_features=max_features)
    X = tfidf.fit_transform(train_q_texts).astype(np.float32)   # (Nq, V)
    payload = {
        "train_qids": train_qids,
        "tfidf": tfidf,
        "X": X,
        "U": U_train.astype(np.float32)                         # (Nq, D)
    }
    p = os.path.join(cache_dir, path_name)
    with open(p, "wb") as f:
        pickle.dump(payload, f)
    print(f"[cache] saved KNN cache -> {p}")
    return p

def _knn_user_vec_for_text(q_text: str, knn_cache_path: str, N: int = 8) -> np.ndarray:
    with open(knn_cache_path, "rb") as f:
        cc = pickle.load(f)
    tfidf, X, U = cc["tfidf"], cc["X"], cc["U"]
    x = tfidf.transform([q_text]).astype(np.float32)            # (1, V)
    sims = (x @ X.T).toarray()[0]                               # (Nq,)
    if sims.size == 0 or np.allclose(sims.max(), 0.0):
        return np.zeros((U.shape[1],), dtype=np.float32)
    if N >= sims.size:
        idx = np.argsort(-sims)
    else:
        idx = np.argpartition(-sims, N)[:N]
        idx = idx[np.argsort(-sims[idx])]
    w = sims[idx]
    w = w / (w.sum() + 1e-8)
    u = (w[:, None] * U[idx]).sum(axis=0)
    return u.astype(np.float32)

@torch.no_grad()
def evaluate_sampled_with_knn(model: LightGCN,
                              q_ids: List[str], a_ids: List[str], qid2idx: Dict[str,int],
                              all_rankings: Dict[str,List[str]],
                              eval_qids: List[str],
                              user_has_edge: np.ndarray,
                              knn_cache_path: str,
                              all_questions: Dict[str,dict],
                              ks=(5,10,50), cand_size=1000, rng_seed=123, knn_N=8):
    max_k = max(ks)
    aid2idx = {aid: i for i, aid in enumerate(a_ids)}
    all_agent_set = set(a_ids)
    U_all, I_all = model.get_user_item()
    agg = {k: {"P":0.0,"R":0.0,"F1":0.0,"Hit":0.0,"nDCG":0.0,"MRR":0.0} for k in ks}
    cnt = 0
    skipped = 0
    ref_k = 10 if 10 in ks else max_k

    pbar = tqdm(eval_qids, desc="Evaluating (sampled + KNN cold-start)", leave=True, dynamic_ncols=True)
    for qid in pbar:
        gt_list = [aid for aid in all_rankings.get(qid, [])[:pos_topk] if aid in aid2idx]
        if not gt_list:
            skipped += 1
            pbar.set_postfix({"done": cnt, "skipped": skipped})
            continue
        rel_set = set(gt_list)
        neg_pool = list(all_agent_set - rel_set)

        rnd = random.Random((hash(qid) ^ (rng_seed * 16777619)) & 0xFFFFFFFF)
        need_neg = max(0, cand_size - len(gt_list))
        if need_neg > 0 and len(neg_pool) > 0:
            k = min(need_neg, len(neg_pool))
            sampled_negs = rnd.sample(neg_pool, k)
            cand_ids = gt_list + sampled_negs
        else:
            cand_ids = gt_list

        qi = qid2idx[qid]
        if user_has_edge[qi]:
            u = U_all[qi:qi+1]                                  # torch (1,D)
        else:
            qtext = all_questions.get(qid, {}).get("input", "")
            u_np = _knn_user_vec_for_text(qtext, knn_cache_path, N=knn_N)   # (D,)
            u = torch.from_numpy(u_np).to(I_all.device, dtype=I_all.dtype).unsqueeze(0)

        cand_idx = torch.tensor([aid2idx[a] for a in cand_ids], dtype=torch.long, device=I_all.device)
        av = I_all[cand_idx]  # (Nc, D)
        scores = (u * av).sum(dim=1).detach().cpu().numpy()
        order = np.argsort(-scores)[:max_k]
        pred_ids = [cand_ids[i] for i in order]

        bin_hits = [1 if aid in rel_set else 0 for aid in pred_ids]
        num_rel = len(rel_set)
        for k in ks:
            top = bin_hits[:k]
            Hk = int(sum(top))
            P = Hk / float(k)
            R = Hk / float(num_rel)
            F1 = (2*P*R)/(P+R) if (P+R) > 0 else 0.0
            Hit = 1.0 if Hk > 0 else 0.0
            dcg = sum(1.0 / math.log2(i + 2.0) for i, h in enumerate(top) if h)
            idcg = _ideal_dcg(k, num_rel)
            nDCG = (dcg / idcg) if idcg > 0 else 0.0
            rr = 0.0
            for i, h in enumerate(top):
                if h:
                    rr = 1.0 / float(i+1); break
            agg[k]["P"]+=P; agg[k]["R"]+=R; agg[k]["F1"]+=F1
            agg[k]["Hit"]+=Hit; agg[k]["nDCG"]+=nDCG; agg[k]["MRR"]+=rr

        cnt += 1
        ref = agg[ref_k]
        pbar.set_postfix({
            "done": cnt, "skipped": skipped,
            f"P@{ref_k}": f"{(ref['P']/cnt):.4f}",
            f"nDCG@{ref_k}": f"{(ref['nDCG']/cnt):.4f}",
            f"MRR@{ref_k}": f"{(ref['MRR']/cnt):.4f}",
            "Ncand": len(cand_ids)
        })

    if cnt == 0:
        return {k: {m:0.0 for m in ["P","R","F1","Hit","nDCG","MRR"]} for k in ks}
    for k in ks:
        for m in agg[k]:
            agg[k][m] /= cnt
    return agg
# === end KNN additions ===

# ---------- Paths ----------
def model_save_paths(cache_dir: str, data_sig: str):
    model_dir = os.path.join(cache_dir, "models")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"{filename}_{data_sig}.pt")
    latest_model = os.path.join(model_dir, f"latest_{data_sig}.pt")
    meta_path = os.path.join(model_dir, f"meta_{data_sig}.json")
    return model_path, latest_model, meta_path

# ---------- Main ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--emb_dim", type=int, default=128)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--neg_per_pos", type=int, default=1)
    parser.add_argument("--reg", type=float, default=1e-4)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--rng_seed_pairs", type=int, default=42)
    parser.add_argument("--valid_ratio", type=float, default=0.2)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--device", type=str, default="cpu")

    # TF-IDF cache & init (aligned with NGCF)
    parser.add_argument("--rebuild_cache", type=int, default=0)
    parser.add_argument("--init_from_tfidf", type=int, default=1, help="1: init user/item embeddings from TF-IDF + SVD")
    parser.add_argument("--max_features", type=int, default=5000, help="TF-IDF max features per vectorizer")
    parser.add_argument("--tfidf_fit_scope", type=str, default="train", choices=["train","all"],
                        help="fit query TF-IDF on train-only or all queries (for SVD init)")

    # Train/valid cache
    parser.add_argument("--rebuild_training_cache", type=int, default=0)

    # Graph scope: train-only (no leakage) vs all (transductive upper bound)
    parser.add_argument("--graph_scope", type=str, default="train", choices=["train","all"])

    # === KNN cold-start args ===
    parser.add_argument("--knn_N", type=int, default=8, help="TF-IDF KNN neighbors for cold queries")
    parser.add_argument("--eval_cand_size", type=int, default=1000, help="Eval candidate size (pos ∪ sampled negs)")

    args = parser.parse_args()

    random.seed(1234); np.random.seed(1234); torch.manual_seed(1234)
    device = torch.device(args.device)

    # Load data
    all_agents, all_questions, all_rankings, tools = collect_data(args.data_root)
    print(f"Loaded {len(all_agents)} agents, {len(all_questions)} questions, {len(tools)} tools.")

    cache_dir = ensure_cache_dir(args.data_root)

    # ---------- Feature cache (aligned with NGCF / TF-IDF + DNN) ----------
    if cache_exists(cache_dir) and args.rebuild_cache == 0:
        (q_ids, a_ids, tool_names, Q, A_text_full,
         agent_tool_idx_padded_np, agent_tool_mask_np) = load_cache(cache_dir)
        print(f"[cache] loaded features from {cache_dir}")
        Q = Q.astype(np.float32)
        A_text_full = A_text_full.astype(np.float32)
        agent_tool_idx_padded = torch.from_numpy(agent_tool_idx_padded_np)
        agent_tool_mask = torch.from_numpy(agent_tool_mask_np)
    else:
        q_ids, q_texts, tool_names, tool_texts, a_ids, a_texts, a_tool_lists = build_text_corpora(
            all_agents, all_questions, tools
        )
        q_vec, tool_vec, a_vec, Q_csr, Tm_csr, Am_csr = build_vectorizers(q_texts, tool_texts, a_texts, args.max_features)
        Atool = agent_tool_text_matrix(a_tool_lists, tool_names, Tm_csr)      # (num_agents, Dt)
        Am = Am_csr.toarray().astype(np.float32)                              # (num_agents, Da)
        A_text_full = np.concatenate([Am, Atool], axis=1).astype(np.float32)  # (num_agents, Da+Dt)
        Q = Q_csr.toarray().astype(np.float32)                                # (num_q, Dq)
        agent_tool_idx_padded, agent_tool_mask = build_agent_tool_id_buffers(a_ids, a_tool_lists, tool_names)
        save_cache(cache_dir, q_ids, a_ids, tool_names, Q, A_text_full,
                   agent_tool_idx_padded.numpy(), agent_tool_mask.numpy())
        print(f"[cache] saved features to {cache_dir}")

    # ID maps
    qid2idx = {qid: i for i, qid in enumerate(q_ids)}
    aid2idx = {aid: i for i, aid in enumerate(a_ids)}

    # Dataset signature & paths
    data_sig = dataset_signature(a_ids, all_rankings)
    model_path, latest_model, meta_path = model_save_paths(cache_dir, data_sig)

    want_meta = {
        "data_sig": data_sig,
        "neg_per_pos": int(args.neg_per_pos),
        "rng_seed_pairs": int(args.rng_seed_pairs),
        "split_seed": int(args.split_seed),
        "valid_ratio": float(args.valid_ratio),
        "graph_scope": args.graph_scope,
        "init_from_tfidf": int(args.init_from_tfidf),
        "tfidf_fit_scope": args.tfidf_fit_scope,
        "max_features": int(args.max_features),
    }

    # Train/valid split & training pairs
    use_training_cache = training_cache_exists(cache_dir) and (args.rebuild_training_cache == 0)
    if use_training_cache:
        train_qids, valid_qids, pairs_idx_np, meta = load_training_cache(cache_dir)
        if meta != want_meta:
            print("[cache] training cache meta mismatch, rebuilding...")
            use_training_cache = False
        else:
            print(f"[cache] loaded train/valid/pairs from {cache_dir} (sig={data_sig})")
    if not use_training_cache:
        qids_in_rank = [qid for qid in q_ids if qid in all_rankings]
        train_qids, valid_qids = train_valid_split(qids_in_rank, valid_ratio=args.valid_ratio, seed=args.split_seed)
        print(f"[split] train={len(train_qids)}  valid={len(valid_qids)}")
        rankings_train = {qid: all_rankings[qid] for qid in train_qids}
        pairs_id = build_training_pairs(rankings_train, a_ids, neg_per_pos=args.neg_per_pos, rng_seed=args.rng_seed_pairs)
        pairs_idx_np = np.array([(qid2idx[q], aid2idx[p], aid2idx[n]) for (q, p, n) in pairs_id], dtype=np.int64)
        save_training_cache(cache_dir, train_qids, valid_qids, pairs_idx_np, want_meta)
        print(f"[cache] saved train/valid/pairs to {cache_dir} (sig={data_sig})")

    # ---------- Build graph edges (STRICT by scope; default: train-only) ----------
    if args.graph_scope == "train":
        edges_source = {qid: all_rankings[qid] for qid in train_qids}
    else:
        edges_source = all_rankings

    pos_edges = []
    for qid, ranked in edges_source.items():
        if not ranked: continue
        u = qid2idx[qid]
        for aid in ranked[:pos_topk]:
            if aid in aid2idx:
                pos_edges.append((u, aid2idx[aid]))

    num_users, num_items = len(q_ids), len(a_ids)
    norm_adj = build_norm_adj(num_users, num_items, pos_edges)

    # mask: which user nodes have at least one train edge
    user_has_edge = _build_user_degree_mask(num_users, pos_edges)  # np.bool_

    # ---------- Model ----------
    model = LightGCN(num_users, num_items, args.emb_dim, args.layers, norm_adj, reg=args.reg, device=device).to(device)

    # ---------- Optional TF-IDF → SVD initialization ----------
    if args.init_from_tfidf == 1:
        # Queries SVD basis
        if args.tfidf_fit_scope == "train":
            q_fit_ids = train_qids
        else:
            q_fit_ids = [qid for qid in q_ids if qid in all_rankings]
        q_vec = TfidfVectorizer(max_features=args.max_features, lowercase=True)
        q_vec.fit([get_query_text(all_questions[qid]) for qid in q_fit_ids])
        Q_basis = q_vec.transform([get_query_text(all_questions[qid]) for qid in q_ids])

        # Agents use cached A_text_full as text feature
        A_basis = A_text_full  # (num_items, Da+Dt)

        def svd_to_dim(X, dim):
            if hasattr(X, "toarray"):
                X = X.toarray()
            comp = min(dim, max(1, min(X.shape[0]-1, X.shape[1]-1)))
            if comp < dim:
                warnings.warn(f"[TFIDF init] reduce emb_dim from {dim} to {comp} for shape {X.shape}")
            svd = TruncatedSVD(n_components=comp, random_state=42)
            Z = svd.fit_transform(X)
            Z = normalize(Z)  # row L2
            if comp < dim:
                Z = np.pad(Z, ((0,0),(0,dim-comp)), mode="constant")
            return Z.astype(np.float32)

        U0 = svd_to_dim(Q_basis, args.emb_dim)   # (num_users, emb_dim)
        V0 = svd_to_dim(A_basis, args.emb_dim)   # (num_items, emb_dim)
        with torch.no_grad():
            model.user_emb.weight.data[:U0.shape[0], :U0.shape[1]] = torch.as_tensor(U0, dtype=model.user_emb.weight.dtype, device=device)
            model.item_emb.weight.data[:V0.shape[0], :V0.shape[1]] = torch.as_tensor(V0, dtype=model.item_emb.weight.dtype, device=device)
        print("[init] user/item embeddings initialized from TF-IDF + SVD")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # ---------- Training ----------
    pairs = pairs_idx_np.tolist()
    num_pairs = len(pairs)
    num_batches = math.ceil(num_pairs / args.batch_size)

    for epoch in range(1, args.epochs+1):
        random.shuffle(pairs)
        total = 0.0
        pbar = tqdm(range(num_batches), desc=f"Epoch {epoch}/{args.epochs}", leave=True, dynamic_ncols=True)
        for b in pbar:
            batch = pairs[b*args.batch_size:(b+1)*args.batch_size]
            if not batch: continue
            u_idx = torch.tensor([t[0] for t in batch], dtype=torch.long, device=device)
            p_idx = torch.tensor([t[1] for t in batch], dtype=torch.long, device=device)
            n_idx = torch.tensor([t[2] for t in batch], dtype=torch.long, device=device)

            loss = model.bpr_loss(u_idx, p_idx, n_idx)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total += float(loss.item())
            pbar.set_postfix({"batch_loss": f"{loss.item():.4f}", "avg_loss": f"{(total/(b+1)):.4f}"})
        print(f"Epoch {epoch}/{args.epochs} - BPR loss: {(total/num_batches if num_batches>0 else 0.0):.4f}")

    # ---------- Save checkpoint + meta ----------
    model_dir = ensure_cache_dir(args.data_root)
    model_path, latest_model, meta_path = model_save_paths(model_dir, data_sig)
    ckpt = {
        "state_dict": model.state_dict(),
        "data_sig": data_sig,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "args": vars(args),
        "dims": {"num_users": num_users, "num_items": num_items, "emb_dim": int(args.emb_dim), "layers": int(args.layers)},
    }
    torch.save(ckpt, model_path); torch.save(ckpt, latest_model)
    serving_meta = {"data_sig": data_sig, "a_ids": a_ids}
    with open(meta_path, "w", encoding="utf-8") as f: json.dump(serving_meta, f, ensure_ascii=False, indent=2)
    print(f"[save] model -> {model_path}")
    print(f"[save] meta  -> {meta_path}")

    # === Build & save KNN cache from TRAIN questions (post-training, using propagated U) ===
    U_all, _I_all = model.get_user_item()
    train_q_texts = [all_questions[qid]["input"] for qid in train_qids]
    train_idx = [qid2idx[qid] for qid in train_qids]
    U_train = U_all[torch.tensor(train_idx, dtype=torch.long, device=U_all.device)].detach().cpu().numpy()  # (Nq,D)
    knn_cache_path = _save_knn_cache(cache_dir, train_qids, train_q_texts, U_train, max_features=args.max_features)

    # ---------- Validation (sampled + KNN cold-start for isolated users) ----------
    valid_metrics = evaluate_sampled_with_knn(
        model, q_ids, a_ids, qid2idx, all_rankings,
        valid_qids,
        user_has_edge=user_has_edge,
        knn_cache_path=knn_cache_path,
        all_questions=all_questions,
        ks=(5,10,50),
        cand_size=args.eval_cand_size,
        rng_seed=123,
        knn_N=args.knn_N
    )
    print_metrics_table("Validation (KNN cold-start; averaged over questions)", valid_metrics, ks=(5,10,50), filename=filename)

    # ---------- Demo ----------
    @torch.no_grad()
    def recommend_topk_for_qid(qid: str, topk: int = 10):
        # If qid has train edges -> use propagated; else use KNN-composed user vector.
        U_all, I_all = model.get_user_item()
        qi = qid2idx[qid]
        if user_has_edge[qi]:
            u = U_all[qi:qi+1]
        else:
            qtext = all_questions[qid].get("input", "")
            u_np = _knn_user_vec_for_text(qtext, knn_cache_path, N=args.knn_N)
            u = torch.from_numpy(u_np).to(I_all.device, dtype=I_all.dtype).unsqueeze(0)
        scores = (u * I_all).sum(dim=1).detach().cpu().numpy()
        order = np.argsort(-scores)[:topk]
        return [(a_ids[i], float(scores[i])) for i in order]

    sample_qids = q_ids[:min(5, len(q_ids))]
    for qid in sample_qids:
        topk = recommend_topk_for_qid(qid, topk=args.topk)
        qtext = all_questions[qid]["input"][:80] if "input" in all_questions[qid] else ""
        print(f"\nQuestion: {qid}  |  {qtext}")
        for rank, (aid, s) in enumerate(topk, 1):
            print(f"  {rank:2d}. {aid:>20s}  score={s:.4f}")

if __name__ == "__main__":
    main()
