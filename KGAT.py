#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
simple_kgat_agent_rec_knn.py  (no-B version, FIXED)
KGAT with semantic neighborhood diffusion for Query ↔ Agent implicit feedback (BPR)
+ TF-IDF KNN cold-start at prediction/eval time.

Fixes:
  - Q–A / Q–T semantic edges are built in a JOINT TF-IDF space to avoid FAISS dim mismatch.
  - FAISS IVF training safety: dtype/contiguity checks + dynamic nlist & nprobe bounds.

Entities: Q (queries), A (agents), T (tools)
Relations:
  - Q—A (interact)           [train-only or all, via --graph_scope]
  - A—T (uses)               [global metadata]
  - Q—A (semantic, top-k)    [global, JOINT TF-IDF]
  - A—A (semantic, top-k)    [OPTIONAL: not used here; keep slot]
  - Q—T (semantic, top-k)    [global, JOINT TF-IDF, optional]

Cold-start:
  - If a query has no train edges, compose its embedding via TF-IDF KNN
    over TRAIN queries' propagated embeddings after training.
"""
import os, json, math, random, argparse, zlib, pickle
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer

from utils import print_metrics_table

filename  = os.path.splitext(os.path.basename(__file__))[0]
pos_topk  = 5  # regard top-k ranked agents as positives

# ---------------- I/O helpers ----------------
def ensure_cache_dir(root: str) -> str:
    d = os.path.join(root, f".cache/{filename}")
    os.makedirs(d, exist_ok=True)
    return d

def load_json(p: str):
    with open(p, "r", encoding="utf-8") as f: return json.load(f)

def collect_data(data_root: str):
    parts = ["PartI", "PartII", "PartIII"]
    all_agents: Dict[str, dict] = {}
    all_questions: Dict[str, dict] = {}
    all_rankings: Dict[str, List[str]] = {}
    for part in parts:
        agents    = load_json(os.path.join(data_root, part, "agents",    "merge.json"))
        questions = load_json(os.path.join(data_root, part, "questions", "merge.json"))
        rankings  = load_json(os.path.join(data_root, part, "rankings",  "merge.json"))
        all_agents.update(agents)
        all_questions.update(questions)
        all_rankings.update(rankings["rankings"])
    tools = load_json(os.path.join(data_root, "Tools", "merge.json"))
    return all_agents, all_questions, all_rankings, tools

def train_valid_split(qids_in_rankings, valid_ratio=0.2, seed=42):
    rng = random.Random(seed)
    q = list(qids_in_rankings); rng.shuffle(q)
    n_valid = int(len(q)*valid_ratio)
    return q[n_valid:], q[:n_valid]

def dataset_signature(a_ids: List[str], all_rankings: Dict[str, List[str]]) -> str:
    payload = {"a_ids": a_ids, "rankings": {k: all_rankings[k] for k in sorted(all_rankings.keys())}}
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return f"{(zlib.crc32(blob) & 0xFFFFFFFF):08x}"

def model_save_paths(cache_dir: str, data_sig: str):
    model_dir = os.path.join(cache_dir, "models"); os.makedirs(model_dir, exist_ok=True)
    model_path  = os.path.join(model_dir, f"{filename}_{data_sig}.pt")
    latest_path = os.path.join(model_dir, f"latest_{data_sig}.pt")
    meta_path   = os.path.join(model_dir, f"meta_{data_sig}.json")
    return model_path, latest_path, meta_path

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

# --------------- Text building (TF-IDF) ---------------
def build_text_corpora(all_agents, all_questions, tools):
    q_ids    = list(all_questions.keys())
    q_texts  = [all_questions[qid].get("input","") for qid in q_ids]

    tool_names = list(tools.keys())
    def tool_text(tn):
        t = tools.get(tn, {})
        return f"{tn} {t.get('description','')}".strip()
    tool_texts = [tool_text(tn) for tn in tool_names]

    a_ids = list(all_agents.keys())
    a_texts = []
    a_tool_lists = []
    for aid in a_ids:
        a = all_agents[aid]
        mname = a.get("M", {}).get("name", "")
        tlist = a.get("T", {}).get("tools", []) or []
        a_tool_lists.append(tlist)
        concat_tool_desc = " ".join([tool_text(tn) for tn in tlist])
        a_texts.append(f"{mname} {concat_tool_desc}".strip())
    return q_ids, q_texts, tool_names, tool_texts, a_ids, a_texts, a_tool_lists

def build_vectorizers(q_texts, tool_texts, a_texts, max_features: int):
    q_vec   = TfidfVectorizer(max_features=max_features, lowercase=True)
    tool_vec= TfidfVectorizer(max_features=max_features, lowercase=True)
    a_vec   = TfidfVectorizer(max_features=max_features, lowercase=True)
    Q  = q_vec.fit_transform(q_texts)         # (nq, Dq)
    Tm = tool_vec.fit_transform(tool_texts)   # (nt, Dt)
    Am = a_vec.fit_transform(a_texts)         # (na, Da)
    return q_vec, tool_vec, a_vec, Q, Tm, Am

def agent_tool_text_matrix(agent_tool_lists: List[List[str]], tool_names: List[str], Tm) -> np.ndarray:
    name2idx = {n:i for i,n in enumerate(tool_names)}
    na = len(agent_tool_lists); Dt = Tm.shape[1]
    Atool = np.zeros((na, Dt), dtype=np.float32)
    for i, lst in enumerate(agent_tool_lists):
        idxs = [name2idx[t] for t in lst if t in name2idx]
        if len(idxs)==0: continue
        Atool[i] = Tm[idxs].toarray().mean(axis=0).astype(np.float32)
    return Atool

def l2_normalize_np(mat: np.ndarray, axis=1, eps=1e-12):
    n = np.linalg.norm(mat, axis=axis, keepdims=True)
    return mat / (n + eps)

# --------------- Training pairs ---------------
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
        neg_pool = list(all_agent_set - set(ranked)) or all_agent_ids
        for pos_a in ranked:
            for _ in range(neg_per_pos):
                neg_a = rnd.choice(neg_pool)
                pairs.append((qid, pos_a, neg_a))
    return pairs

# =============== KGAT Core ===============
class KGATLayer(nn.Module):
    """
    One KGAT layer with relation-aware attention.
    H_in (N,D) -> H_out (N,D)
    Edges stored as COO: src[E], dst[E], rel_id[E]
    """
    def __init__(self, dim: int, rel_dim: int, num_rels: int, negative_slope=0.2):
        super().__init__()
        self.W_node = nn.Linear(dim, dim, bias=False)
        self.W_rel  = nn.Linear(rel_dim, dim, bias=False)
        self.attvec = nn.Linear(3*dim, 1, bias=False)  # [Wh_i || Wh_j || Wr_e]
        self.leaky  = nn.LeakyReLU(negative_slope)

        self.rel_emb = nn.Embedding(num_rels, rel_dim)
        nn.init.xavier_uniform_(self.W_node.weight)
        nn.init.xavier_uniform_(self.W_rel.weight)
        nn.init.xavier_uniform_(self.attvec.weight)
        nn.init.xavier_uniform_(self.rel_emb.weight)

    def forward(self, H: torch.Tensor,
                src: torch.LongTensor, dst: torch.LongTensor, rel: torch.LongTensor,
                num_nodes: int):
        Wh = self.W_node(H)                               # (N,D)
        Wr = self.W_rel(self.rel_emb(rel))                # (E,D)

        h_i = Wh[src]                                     # (E,D)
        h_j = Wh[dst]                                     # (E,D)
        e_ij = self.leaky(self.attvec(torch.cat([h_i, h_j, Wr], dim=-1))).squeeze(-1)  # (E,)

        # softmax over outgoing edges of each src node (stable)
        device = H.device
        e_max = torch.full((num_nodes,), -1e30, device=device)
        e_max = e_max.scatter_reduce(0, src, e_ij, reduce="amax", include_self=True)
        alpha_num = torch.exp(e_ij - e_max[src])          # (E,)
        denom = torch.zeros((num_nodes,), device=device)
        denom = denom.scatter_add(0, src, alpha_num) + 1e-12
        alpha = alpha_num / denom[src]                    # (E,)

        msg = h_j + Wr                                    # (E,D)
        H_out = torch.zeros_like(H)                       # (N,D)
        H_out = H_out.index_add(0, src, alpha.unsqueeze(-1) * msg)
        return H_out

class KGAT(nn.Module):
    def __init__(self, num_nodes:int, dim:int,
                 num_rels:int, rel_dim:int,
                 layers:int,
                 use_residual:bool=True, reg:float=1e-4):
        super().__init__()
        self.num_nodes  = num_nodes
        self.dim        = dim
        self.layers     = layers
        self.use_res    = use_residual
        self.reg        = reg
        self.layers_mod = nn.ModuleList([KGATLayer(dim, rel_dim, num_rels) for _ in range(layers)])

    def forward(self, H0, src, dst, rel):
        H = H0
        outs = [H]
        for layer in self.layers_mod:
            H_next = layer(H, src, dst, rel, self.num_nodes)
            H = 0.5 * H + 0.5 * H_next if self.use_res else H_next
            outs.append(H)
        return torch.stack(outs, dim=0).mean(0)  # (N,D)

    def bpr_loss(self, H_final, q_idx_global, pos_a_global, neg_a_global):
        u = H_final[q_idx_global]      # (B,D)
        ip = H_final[pos_a_global]     # (B,D)
        ineg = H_final[neg_a_global]   # (B,D)
        pos_scores = (u*ip).sum(-1)
        neg_scores = (u*ineg).sum(-1)
        loss_bpr = -torch.log(torch.sigmoid(pos_scores - neg_scores)+1e-8).mean()
        reg = (u.norm(p=2, dim=1).pow(2).mean() +
               ip.norm(p=2, dim=1).pow(2).mean() +
               ineg.norm(p=2, dim=1).pow(2).mean())
        return loss_bpr + self.reg * reg

# =============== FAISS helpers (robust) ===============
import faiss

def build_semantic_edges_topk(X_src: np.ndarray, X_dst: np.ndarray, topk:int,
                              use_ivf:bool=True, nlist:int=4096, nprobe:int=16, gpu_id:int=0):
    """
    Build directed edges from every row in X_src to top-k rows in X_dst using cosine (via inner product on L2-normalized vectors).
    Robust to small databases: dynamically downscale nlist; enforce dtype/contiguity; validate dims.
    """
    # ensure float32 and contiguous
    Xs = l2_normalize_np(np.asarray(X_src, dtype=np.float32))
    Xd = l2_normalize_np(np.asarray(X_dst, dtype=np.float32))

    # dim check
    if Xs.shape[1] != Xd.shape[1]:
        raise ValueError(f"[semantic-topk] Dim mismatch: X_src dim={Xs.shape[1]} vs X_dst dim={Xd.shape[1]}. "
                         "Use JOINT TF-IDF vectorizer for both sides.")

    d  = Xs.shape[1]
    k  = max(1, min(topk, Xd.shape[0]))

    # SAFETY: adjust nlist if IVF, ensure not exceeding DB size
    use_ivf = bool(use_ivf)
    if use_ivf:
        # empirical: at least ~8 vectors per list
        eff_nlist = max(1, min(int(nlist), max(1, Xd.shape[0] // 8)))
    else:
        eff_nlist = None

    # CPU index
    if use_ivf:
        quant = faiss.IndexFlatIP(d)
        cpu_index = faiss.IndexIVFFlat(quant, d, eff_nlist, faiss.METRIC_INNER_PRODUCT)
    else:
        cpu_index = faiss.IndexFlatIP(d)

    # to GPU
    res = faiss.StandardGpuResources()
    co = faiss.GpuClonerOptions()
    co.useFloat16 = True
    index = faiss.index_cpu_to_gpu(res, gpu_id, cpu_index, co)

    # train (IVF)
    if use_ivf:
        Xd_train = np.ascontiguousarray(Xd, dtype=np.float32)
        index.train(Xd_train)
        index.nprobe = min(max(1, int(nprobe)), eff_nlist)

    index.add(np.ascontiguousarray(Xd, dtype=np.float32))

    # batched search
    # heuristic batch size: larger for small dims
    B = max(512, 8192 // max(1, d // 128))
    I_all = []
    for i0 in range(0, Xs.shape[0], B):
        i1 = min(Xs.shape[0], i0 + B)
        _, I = index.search(np.ascontiguousarray(Xs[i0:i1], dtype=np.float32), k)
        I_all.append(I)

    I = np.vstack(I_all)  # (nsrc, k)
    edges = [(i, j) for i in range(I.shape[0]) for j in I[i].tolist()]
    return edges

# === KNN cold-start helpers (query-only) ===
def _build_user_degree_mask(num_users: int, train_edges: List[Tuple[int,int]]) -> np.ndarray:
    deg = np.zeros((num_users,), dtype=np.int64)
    for u, _ in train_edges:
        deg[u] += 1
    return (deg > 0)

def _save_knn_cache(cache_dir: str, train_qids: List[str], train_q_texts: List[str],
                    U_train: np.ndarray, max_features: int = 5000, path_name: str = "knn_q_cache.pkl"):
    tfidf = TfidfVectorizer(max_features=max_features, lowercase=True)
    X = tfidf.fit_transform(train_q_texts).astype(np.float32)   # (Nq, V)
    payload = {"train_qids": train_qids, "tfidf": tfidf, "X": X, "U": U_train.astype(np.float32)}
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

# =============== Evaluation (sampled + KNN cold-start) ===============
@torch.no_grad()
def evaluate_sampled_with_knn(H_final: torch.Tensor,
                              q_ids: List[str], a_ids: List[str],
                              qid2idx: Dict[str,int],
                              all_rankings: Dict[str,List[str]],
                              eval_qids: List[str], ks=(5,10,50),
                              cand_size=1000, rng_seed=123,
                              a_global_offset:int=0,
                              user_has_edge: np.ndarray=None,
                              knn_cache_path: str=None,
                              all_questions: Dict[str,dict]=None,
                              knn_N:int=8):
    max_k = max(ks)
    aid2local = {aid:i for i,aid in enumerate(a_ids)}
    all_agent_set = set(a_ids)

    agg = {k: {"P":0.0,"R":0.0,"F1":0.0,"Hit":0.0,"nDCG":0.0,"MRR":0.0} for k in ks}
    cnt=0; skipped=0
    ref_k = 10 if 10 in ks else max_k

    pbar = tqdm(eval_qids, desc="Evaluating (KGAT + KNN cold-start)", leave=True, dynamic_ncols=True)
    for qid in pbar:
        gt_list = [aid for aid in all_rankings.get(qid, [])[:pos_topk] if aid in aid2local]
        if not gt_list:
            skipped += 1; pbar.set_postfix({"done":cnt,"skipped":skipped}); continue
        rel_set = set(gt_list)
        neg_pool = list(all_agent_set - rel_set)

        rnd = random.Random((hash(qid) ^ (rng_seed * 16777619)) & 0xFFFFFFFF)
        need_neg = max(0, cand_size - len(gt_list))
        if need_neg>0 and len(neg_pool)>0:
            k = min(need_neg, len(neg_pool))
            cand_ids = gt_list + rnd.sample(neg_pool, k)
        else:
            cand_ids = gt_list

        qi_local = qid2idx[qid]
        # choose user vector
        if user_has_edge is not None and user_has_edge[qi_local]:
            u = H_final[qi_local:qi_local+1]               # (1,D)
        else:
            qtext = (all_questions.get(qid, {}) or {}).get("input", "")
            u_np = _knn_user_vec_for_text(qtext, knn_cache_path, N=knn_N) if (knn_cache_path is not None) else np.zeros((H_final.shape[1],), dtype=np.float32)
            u = torch.from_numpy(u_np).to(H_final.device, dtype=H_final.dtype).unsqueeze(0)

        cand_local = torch.tensor([aid2local[a] for a in cand_ids], dtype=torch.long, device=H_final.device)
        cand_global = a_global_offset + cand_local
        av = H_final[cand_global]                       # (Nc,D)
        scores = (u * av).sum(-1).cpu().numpy()
        order = np.argsort(-scores)[:max_k]
        pred_ids = [cand_ids[i] for i in order]

        bin_hits = [1 if aid in rel_set else 0 for aid in pred_ids]
        for k in ks:
            Hk = sum(bin_hits[:k])
            P  = Hk/float(k)
            R  = Hk/float(len(rel_set))
            F1 = (2*P*R)/(P+R) if (P+R)>0 else 0.0
            Hit= 1.0 if Hk>0 else 0.0
            dcg=0.0
            for i,h in enumerate(bin_hits[:k]):
                if h: dcg += 1.0/math.log2(i+2.0)
            ideal = min(len(rel_set), k)
            idcg = sum(1.0/math.log2(i+2.0) for i in range(ideal)) if ideal>0 else 0.0
            nDCG = (dcg/idcg) if idcg>0 else 0.0
            rr=0.0
            for i in range(k):
                if bin_hits[i]:
                    rr = 1.0/float(i+1); break
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

    if cnt==0:
        return {k:{m:0.0 for m in ["P","R","F1","Hit","nDCG","MRR"]} for k in ks}
    for k in ks:
        for m in agg[k]: agg[k][m]/=cnt
    return agg

# =============== Main ===============
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--emb_dim", type=int, default=128)   # final node dim
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--text_hidden", type=int, default=256)
    parser.add_argument("--id_dim", type=int, default=64)
    parser.add_argument("--rel_dim", type=int, default=32)
    parser.add_argument("--max_features", type=int, default=5000)
    parser.add_argument("--semantic_topk", type=int, default=20)
    parser.add_argument("--add_qt_semantic", type=int, default=0)  # Q—T semantic edges
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--neg_per_pos", type=int, default=1)
    parser.add_argument("--reg", type=float, default=1e-4)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--rng_seed_pairs", type=int, default=42)
    parser.add_argument("--valid_ratio", type=float, default=0.2)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--rebuild_cache", type=int, default=0)
    parser.add_argument("--rebuild_training_cache", type=int, default=0)

    # leakage controls & KNN
    parser.add_argument("--graph_scope", type=str, default="train", choices=["train","all"],
                        help="Q–A interaction edges from train-only or all queries")
    parser.add_argument("--tfidf_fit_scope", type=str, default="train", choices=["train","all"],
                        help="fit query TF-IDF on train-only or all queries for semantic edges")
    parser.add_argument("--knn_N", type=int, default=8, help="TF-IDF KNN neighbors for cold queries")
    parser.add_argument("--eval_cand_size", type=int, default=1000, help="Eval candidate size")

    # FAISS
    parser.add_argument("--faiss_gpu", type=int, default=1)
    parser.add_argument("--faiss_ivf", type=int, default=1, help="1=IVF, 0=Flat")
    parser.add_argument("--nlist", type=int, default=4096)
    parser.add_argument("--nprobe", type=int, default=16)
    parser.add_argument("--faiss_gpu_id", type=int, default=0)

    args = parser.parse_args()

    random.seed(1234); np.random.seed(1234); torch.manual_seed(1234)
    device = torch.device(args.device)

    # ---------- Load data ----------
    all_agents, all_questions, all_rankings, tools = collect_data(args.data_root)
    print(f"Loaded {len(all_agents)} agents, {len(all_questions)} questions, {len(tools)} tools.")

    # ---------- Text corpora & TF-IDF (for encoders, not for semantic edges) ----------
    cache_dir = ensure_cache_dir(args.data_root)
    tfidf_cache = os.path.join(cache_dir, "tfidf_cache.pkl")
    if os.path.exists(tfidf_cache) and args.rebuild_cache==0:
        with open(tfidf_cache, "rb") as f:
            (q_ids, q_texts, tool_names, tool_texts, a_ids, a_texts, a_tool_lists,
             Q_np, A_text_full_np, T_np) = pickle.load(f)
        print(f"[cache] loaded TF-IDF from {tfidf_cache}")
    else:
        q_ids, q_texts, tool_names, tool_texts, a_ids, a_texts, a_tool_lists = build_text_corpora(
            all_agents, all_questions, tools
        )
        q_vec, tool_vec, a_vec, Q_csr, Tm_csr, Am_csr = build_vectorizers(q_texts, tool_texts, a_texts, args.max_features)
        Atool = agent_tool_text_matrix(a_tool_lists, tool_names, Tm_csr)  # (na, Dt)
        Am = Am_csr.toarray().astype(np.float32)
        A_text_full_np = np.concatenate([Am, Atool], axis=1).astype(np.float32)  # agent text
        Q_np = Q_csr.toarray().astype(np.float32)
        T_np = Tm_csr.toarray().astype(np.float32)
        with open(tfidf_cache, "wb") as f:
            pickle.dump((q_ids, q_texts, tool_names, tool_texts, a_ids, a_texts, a_tool_lists,
                         Q_np, A_text_full_np, T_np), f)
        print(f"[cache] saved TF-IDF to {tfidf_cache}")

    nq, na, nt = len(q_ids), len(a_ids), len(tool_names)
    # ---- Global node indexing: [Q | A | T] ----
    OFF_Q = 0
    OFF_A = OFF_Q + nq
    OFF_T = OFF_A + na
    NUM_NODES = OFF_T + nt

    qid2idx = {qid:i for i,qid in enumerate(q_ids)}
    aid2idx = {aid:i for i,aid in enumerate(a_ids)}
    tname2idx = {t:i for i,t in enumerate(tool_names)}

    # ---------- ID & Text encoders ----------
    emb_q = nn.Embedding(nq, args.id_dim); nn.init.xavier_uniform_(emb_q.weight)
    emb_a = nn.Embedding(na, args.id_dim); nn.init.xavier_uniform_(emb_a.weight)
    emb_t = nn.Embedding(nt, args.id_dim); nn.init.xavier_uniform_(emb_t.weight)

    d_q = Q_np.shape[1]; d_a = A_text_full_np.shape[1]; d_t = T_np.shape[1]
    proj_q = nn.Linear(d_q, args.text_hidden); nn.init.xavier_uniform_(proj_q.weight); nn.init.zeros_(proj_q.bias)
    proj_a = nn.Linear(d_a, args.text_hidden); nn.init.xavier_uniform_(proj_a.weight); nn.init.zeros_(proj_a.bias)
    proj_t = nn.Linear(d_t, args.text_hidden); nn.init.xavier_uniform_(proj_t.weight); nn.init.zeros_(proj_t.bias)

    fuse_q = nn.Linear(args.id_dim + args.text_hidden, args.emb_dim)
    fuse_a = nn.Linear(args.id_dim + args.text_hidden, args.emb_dim)
    fuse_t = nn.Linear(args.id_dim + args.text_hidden, args.emb_dim)
    for m in [fuse_q, fuse_a, fuse_t]:
        nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)

    Q_feat = torch.from_numpy(Q_np).to(device)
    A_feat = torch.from_numpy(A_text_full_np).to(device)
    T_feat = torch.from_numpy(T_np).to(device)
    emb_q = emb_q.to(device); emb_a = emb_a.to(device); emb_t = emb_t.to(device)
    proj_q = proj_q.to(device); proj_a = proj_a.to(device); proj_t = proj_t.to(device)
    fuse_q = fuse_q.to(device); fuse_a = fuse_a.to(device); fuse_t = fuse_t.to(device)

    def build_H0():
        q_id  = emb_q.weight
        q_txt = F.relu(proj_q(Q_feat))
        a_id  = emb_a.weight
        a_txt = F.relu(proj_a(A_feat))
        t_id  = emb_t.weight
        t_txt = F.relu(proj_t(T_feat))

        Hq = F.relu(fuse_q(torch.cat([q_id, q_txt], dim=-1)))    # (nq, D)
        Ha = F.relu(fuse_a(torch.cat([a_id, a_txt], dim=-1)))    # (na, D)
        Ht = F.relu(fuse_t(torch.cat([t_id, t_txt], dim=-1)))    # (nt, D)
        return torch.cat([Hq, Ha, Ht], dim=0)                    # (N, D)

    # ---------- Relations ----------
    REL_INTERACT = 0   # Q—A
    REL_USES     = 1   # A—T
    REL_QA_SEM   = 2   # semantic Q—A (JOINT space)
    REL_AA_SEM   = 3   # kept for completeness (not built)
    REL_QT_SEM   = 4   # semantic Q—T (JOINT space, optional)
    NUM_RELS     = 5

    src_list, dst_list, rel_list = [], [], []
    train_pos_edges: List[Tuple[int,int]] = []  # for user_has_edge mask (u_local, a_local)

    def add_edge(u,v,r,undirected=True):
        src_list.append(u); dst_list.append(v); rel_list.append(r)
        if undirected:
            src_list.append(v); dst_list.append(u); rel_list.append(r)

    # ---------- Train/valid split ----------
    qids_in_rank = [qid for qid in q_ids if qid in all_rankings]
    train_qids, valid_qids = train_valid_split(qids_in_rank, valid_ratio=args.valid_ratio, seed=args.split_seed)

    # ---------- Q—A interact (scope controlled) ----------
    if args.graph_scope == "train":
        edges_source = {qid: all_rankings[qid] for qid in train_qids}
    else:
        edges_source = all_rankings

    for qid, ranked in edges_source.items():
        if not ranked: continue
        qi_local = qid2idx[qid]
        qi = OFF_Q + qi_local
        for aid in ranked[:pos_topk]:
            if aid not in aid2idx: continue
            aj_local = aid2idx[aid]
            aj = OFF_A + aj_local
            add_edge(qi, aj, REL_INTERACT, undirected=True)
            if args.graph_scope == "train":
                train_pos_edges.append((qi_local, aj_local))

    # ---------- A—T uses (global metadata) ----------
    for aid in a_ids:
        ai = OFF_A + aid2idx[aid]
        tlist = all_agents[aid].get("T",{}).get("tools", []) or []
        for t in tlist:
            if t not in tname2idx: continue
            tj = OFF_T + tname2idx[t]
            add_edge(ai, tj, REL_USES, undirected=True)

    # ---------- Semantic edges (JOINT TF-IDF SPACES) ----------
    # leakage-aware fit scope on queries:
    fit_ids = train_qids if args.tfidf_fit_scope == "train" else qids_in_rank

    # Q–A joint vectorizer: fit on (train/all queries in scope) + ALL agents
    def build_joint_tfidf_q_a(q_texts_all, a_texts_all, fit_qids):
        qa_vec = TfidfVectorizer(max_features=args.max_features, lowercase=True)
        fit_queries = [all_questions[qid].get("input","") for qid in fit_qids]
        qa_vec.fit(fit_queries + a_texts_all)
        Q_sem = qa_vec.transform(q_texts_all).toarray().astype(np.float32)
        A_sem = qa_vec.transform(a_texts_all).toarray().astype(np.float32)
        return Q_sem, A_sem

    Q_sem_QA, A_sem_QA = build_joint_tfidf_q_a(q_texts, a_texts, fit_ids)

    # Build Q–A semantic edges in JOINT space
    if args.faiss_gpu:
        QA_edges = build_semantic_edges_topk(
            Q_sem_QA, A_sem_QA, args.semantic_topk,
            use_ivf=bool(args.faiss_ivf), nlist=args.nlist, nprobe=args.nprobe,
            gpu_id=args.faiss_gpu_id
        )
    else:
        QA_edges = build_semantic_edges_topk(Q_sem_QA, A_sem_QA, args.semantic_topk)

    for (qi_local, aj_local) in QA_edges:
        add_edge(OFF_Q + qi_local, OFF_A + aj_local, REL_QA_SEM, undirected=True)

    # Q–T joint vectorizer (optional): fit on (train/all queries in scope) + ALL tools
    if args.add_qt_semantic:
        def build_joint_tfidf_q_t(q_texts_all, t_texts_all, fit_qids):
            qt_vec = TfidfVectorizer(max_features=args.max_features, lowercase=True)
            fit_queries = [all_questions[qid].get("input","") for qid in fit_qids]
            qt_vec.fit(fit_queries + t_texts_all)
            Q_sem = qt_vec.transform(q_texts_all).toarray().astype(np.float32)
            T_sem = qt_vec.transform(t_texts_all).toarray().astype(np.float32)
            return Q_sem, T_sem

        Q_sem_QT, T_sem_QT = build_joint_tfidf_q_t(q_texts, tool_texts, fit_ids)

        if args.faiss_gpu:
            QT_edges = build_semantic_edges_topk(
                Q_sem_QT, T_sem_QT, max(5, args.semantic_topk//2),
                use_ivf=bool(args.faiss_ivf), nlist=args.nlist, nprobe=args.nprobe,
                gpu_id=args.faiss_gpu_id
            )
        else:
            QT_edges = build_semantic_edges_topk(Q_sem_QT, T_sem_QT, max(5, args.semantic_topk//2))

        for (qi_local, tj_local) in QT_edges:
            add_edge(OFF_Q + qi_local, OFF_T + tj_local, REL_QT_SEM, undirected=True)

    src = torch.tensor(src_list, dtype=torch.long, device=device)
    dst = torch.tensor(dst_list, dtype=torch.long, device=device)
    rel = torch.tensor(rel_list, dtype=torch.long, device=device)
    print(f"[graph] nodes={NUM_NODES}  edges={len(src_list)} (undirected counted)  rel_types={NUM_RELS}")

    # user_has_edge mask (only meaningful if graph_scope=train)
    user_has_edge = _build_user_degree_mask(nq, train_pos_edges) if train_pos_edges else np.zeros((nq,), dtype=bool)

    # ---------- BPR training pairs cache ----------
    data_sig = dataset_signature(a_ids, all_rankings)
    model_path, latest_model, meta_path = model_save_paths(cache_dir, data_sig)
    want_meta = {
        "data_sig": data_sig,
        "neg_per_pos": int(args.neg_per_pos),
        "rng_seed_pairs": int(args.rng_seed_pairs),
        "split_seed": int(args.split_seed),
        "valid_ratio": float(args.valid_ratio),
        "semantic_topk": int(args.semantic_topk),
        "add_qt_semantic": int(args.add_qt_semantic),
        "graph_scope": args.graph_scope,
        "tfidf_fit_scope": args.tfidf_fit_scope,
        "max_features": int(args.max_features),
    }
    use_cache = training_cache_exists(cache_dir) and (args.rebuild_training_cache==0)
    if use_cache:
        train_qids_cached, valid_qids_cached, pairs_idx_np, meta = load_training_cache(cache_dir)
        if meta != want_meta:
            print("[cache] training cache meta mismatch, rebuilding...")
            use_cache = False
        else:
            train_qids, valid_qids = train_qids_cached, valid_qids_cached
            print(f"[cache] loaded train/valid/pairs from {cache_dir} (sig={data_sig})")

    if not use_cache:
        rankings_train = {qid: all_rankings[qid] for qid in train_qids}
        pairs_id = build_training_pairs(rankings_train, a_ids, neg_per_pos=args.neg_per_pos, rng_seed=args.rng_seed_pairs)
        pairs_idx_np = np.array([(qid2idx[q], aid2idx[p], aid2idx[n]) for (q,p,n) in pairs_id], dtype=np.int64)
        save_training_cache(cache_dir, train_qids, valid_qids, pairs_idx_np, want_meta)
        print(f"[cache] saved train/valid/pairs to {cache_dir} (sig={data_sig})")

    # ---------- Model ----------
    kgat = KGAT(num_nodes=NUM_NODES, dim=args.emb_dim,
                num_rels=NUM_RELS, rel_dim=args.rel_dim,
                layers=args.layers, use_residual=True, reg=args.reg).to(device)

    # optimize KGAT + type-specific encoders
    optimizer = torch.optim.Adam(list(kgat.parameters()) +
                                 list(emb_q.parameters()) + list(emb_a.parameters()) +
                                 list(emb_t.parameters()) +
                                 list(proj_q.parameters()) + list(proj_a.parameters()) +
                                 list(proj_t.parameters()) +
                                 list(fuse_q.parameters()) + list(fuse_a.parameters()) + list(fuse_t.parameters()),
                                 lr=args.lr)

    def get_H_final():
        H0 = build_H0()
        Hf = kgat(H0, src, dst, rel)
        return Hf

    # ---------- Training ----------
    pairs = pairs_idx_np.tolist()
    num_batches = math.ceil(len(pairs)/args.batch_size)
    for epoch in range(1, args.epochs+1):
        random.shuffle(pairs)
        total = 0.0
        pbar = tqdm(range(num_batches), desc=f"Epoch {epoch}/{args.epochs}", leave=True, dynamic_ncols=True)
        for b in pbar:
            batch = pairs[b*args.batch_size:(b+1)*args.batch_size]
            if not batch: continue
            q_local  = torch.tensor([t[0] for t in batch], dtype=torch.long, device=device)
            p_local  = torch.tensor([t[1] for t in batch], dtype=torch.long, device=device)
            n_local  = torch.tensor([t[2] for t in batch], dtype=torch.long, device=device)

            q_global = OFF_Q + q_local
            p_global = OFF_A + p_local
            n_global = OFF_A + n_local

            Hf = get_H_final()
            loss = kgat.bpr_loss(Hf, q_global, p_global, n_global)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total += float(loss.item())
            pbar.set_postfix({"batch_loss": f"{loss.item():.4f}", "avg_loss": f"{(total/(b+1)):.4f}"})
        print(f"Epoch {epoch}/{args.epochs} - BPR loss: {(total/max(1,num_batches)):.4f}")

    # ---------- Save ----------
    ckpt = {
        "state_dict": kgat.state_dict(),
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "args": vars(args),
        "dims": {"num_nodes": int(NUM_NODES), "emb_dim": int(args.emb_dim), "layers": int(args.layers)},
        "offsets": {"Q":OFF_Q, "A":OFF_A, "T":OFF_T},  # no B
        "data_sig": dataset_signature(a_ids, all_rankings),
    }
    model_path, latest_model, meta_path = model_save_paths(cache_dir, ckpt["data_sig"])
    torch.save(ckpt, model_path); torch.save(ckpt, latest_model)
    serving_meta = {"data_sig": ckpt["data_sig"], "a_ids": a_ids}
    with open(meta_path, "w", encoding="utf-8") as f: json.dump(serving_meta, f, ensure_ascii=False, indent=2)
    print(f"[save] model -> {model_path}")
    print(f"[save] meta  -> {meta_path}")

    # ---------- Build KNN cache (TRAIN queries, after propagation) ----------
    with torch.no_grad():
        Hf = get_H_final()
    U_all = Hf[OFF_Q:OFF_Q+nq].detach().cpu().numpy()
    train_idx = [qid2idx[qid] for qid in train_qids]
    U_train = U_all[train_idx]
    train_q_texts = [all_questions[qid].get("input","") for qid in train_qids]
    knn_cache_path = _save_knn_cache(cache_dir, train_qids, train_q_texts, U_train,
                                     max_features=args.max_features, path_name="knn_q_cache.pkl")

    # ---------- Validation (KNN cold-start for isolated queries) ----------
    valid_metrics = evaluate_sampled_with_knn(
        Hf, q_ids, a_ids, qid2idx, all_rankings, valid_qids,
        ks=(5,10,50), cand_size=args.eval_cand_size, rng_seed=123,
        a_global_offset=OFF_A,
        user_has_edge=user_has_edge,
        knn_cache_path=knn_cache_path,
        all_questions=all_questions,
        knn_N=args.knn_N
    )
    print_metrics_table("Validation (KGAT + KNN cold-start; averaged over questions)",
                        valid_metrics, ks=(5,10,50), filename=filename)

    # ---------- Demo ----------
    @torch.no_grad()
    def recommend_topk_for_qid(qid: str, topk: int = 10):
        Hf = get_H_final()
        qi_local = qid2idx[qid]
        if user_has_edge[qi_local]:
            u = Hf[OFF_Q + qi_local:OFF_Q + qi_local + 1]
        else:
            qtext = all_questions[qid].get("input","")
            u_np = _knn_user_vec_for_text(qtext, knn_cache_path, N=args.knn_N)
            u = torch.from_numpy(u_np).to(Hf.device, dtype=Hf.dtype).unsqueeze(0)
        Ablock = Hf[OFF_A:OFF_A+na]
        scores = (u * Ablock).sum(-1).detach().cpu().numpy()
        order = np.argsort(-scores)[:topk]
        return [(a_ids[i], float(scores[i])) for i in order]

    for qid in q_ids[:min(5, len(q_ids))]:
        topk = recommend_topk_for_qid(qid, topk=args.topk)
        qtext = all_questions[qid].get("input","")[:80]
        print(f"\nQuestion: {qid} | {qtext}")
        for r,(aid, s) in enumerate(topk, 1):
            print(f"  {r:2d}. {aid:>20s}  score={s:.4f}")

if __name__ == "__main__":
    main()
