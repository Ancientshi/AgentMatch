#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
simple_bpr_hybrid_agent_rec.py

Hybrid recommender for Query→Agent using BPR loss + negative sampling.
- Text features = TF‑IDF (questions + agents + aggregated tool text)  AND  Pretrained Transformer sentence embeddings.
- Non-text features kept: agent ID embedding + mean tool ID embedding.
- Training objective/neg sampling unchanged (BPR); evaluation unchanged (sampled metrics).

Usage:
python /mnt/data/simple_bpr_hybrid_agent_rec.py \
  --data_root /path/to/dataset_root \
  --epochs 3 --batch_size 256 --max_features 5000 \
  --pretrained_model distilbert-base-uncased --max_len 128 \
  --neg_per_pos 1 --topk 10 --device cuda:0

Notes:
- Transformer encoder is used in frozen, offline mode (we cache sentence embeddings).
- You can rebuild caches with --rebuild_cache 1.

Dependencies: torch, numpy, tqdm, scikit-learn, transformers
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
from transformers import AutoTokenizer, AutoModel

filename = os.path.splitext(os.path.basename(__file__))[0]
pos_topk = 5  # treat top-k ranked agents as positives

# -------------------- Paths & caching --------------------

def ensure_cache_dir(root: str) -> str:
    d = os.path.join(root, f".cache/{filename}")
    os.makedirs(d, exist_ok=True)
    return d

def model_save_paths(cache_dir: str, data_sig: str):
    model_dir = os.path.join(cache_dir, "models"); os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"{filename}_{data_sig}.pt")
    latest_model = os.path.join(model_dir, f"latest_{data_sig}.pt")
    meta_path = os.path.join(model_dir, f"meta_{data_sig}.json")
    return model_path, latest_model, meta_path

# For feature caches

def tfidf_cache_exists(cache_dir: str) -> bool:
    needed = [
        "q_ids.json", "a_ids.json", "tool_names.json",
        "Q_tfidf.npy", "A_tfidf.npy", "agent_tool_idx_padded.npy", "agent_tool_mask.npy",
        "q_vec.pkl", "a_vec.pkl", "tool_vec.pkl"
    ]
    return all(os.path.exists(os.path.join(cache_dir, f)) for f in needed)


def trf_cache_exists(cache_dir: str) -> bool:
    needed = [
        "Q_trf.npy", "A_trf.npy", "enc_meta.json"
    ]
    return all(os.path.exists(os.path.join(cache_dir, f)) for f in needed)


def save_tfidf_cache(cache_dir: str,
                     q_ids, a_ids, tool_names,
                     Q_tfidf, A_tfidf,
                     agent_tool_idx_padded, agent_tool_mask,
                     q_vec, a_vec, tool_vec):
    with open(os.path.join(cache_dir, "q_ids.json"), "w", encoding="utf-8") as f: json.dump(q_ids, f, ensure_ascii=False)
    with open(os.path.join(cache_dir, "a_ids.json"), "w", encoding="utf-8") as f: json.dump(a_ids, f, ensure_ascii=False)
    with open(os.path.join(cache_dir, "tool_names.json"), "w", encoding="utf-8") as f: json.dump(tool_names, f, ensure_ascii=False)
    np.save(os.path.join(cache_dir, "Q_tfidf.npy"), Q_tfidf.astype(np.float32))
    np.save(os.path.join(cache_dir, "A_tfidf.npy"), A_tfidf.astype(np.float32))
    np.save(os.path.join(cache_dir, "agent_tool_idx_padded.npy"), agent_tool_idx_padded.astype(np.int64))
    np.save(os.path.join(cache_dir, "agent_tool_mask.npy"), agent_tool_mask.astype(np.float32))
    with open(os.path.join(cache_dir, "q_vec.pkl"), "wb") as f: pickle.dump(q_vec, f)
    with open(os.path.join(cache_dir, "a_vec.pkl"), "wb") as f: pickle.dump(a_vec, f)
    with open(os.path.join(cache_dir, "tool_vec.pkl"), "wb") as f: pickle.dump(tool_vec, f)


def load_tfidf_cache(cache_dir: str):
    with open(os.path.join(cache_dir, "q_ids.json"), "r", encoding="utf-8") as f: q_ids = json.load(f)
    with open(os.path.join(cache_dir, "a_ids.json"), "r", encoding="utf-8") as f: a_ids = json.load(f)
    with open(os.path.join(cache_dir, "tool_names.json"), "r", encoding="utf-8") as f: tool_names = json.load(f)
    Q_tfidf = np.load(os.path.join(cache_dir, "Q_tfidf.npy"))
    A_tfidf = np.load(os.path.join(cache_dir, "A_tfidf.npy"))
    agent_tool_idx_padded = np.load(os.path.join(cache_dir, "agent_tool_idx_padded.npy"))
    agent_tool_mask = np.load(os.path.join(cache_dir, "agent_tool_mask.npy"))
    with open(os.path.join(cache_dir, "q_vec.pkl"), "rb") as f: q_vec = pickle.load(f)
    with open(os.path.join(cache_dir, "a_vec.pkl"), "rb") as f: a_vec = pickle.load(f)
    with open(os.path.join(cache_dir, "tool_vec.pkl"), "rb") as f: tool_vec = pickle.load(f)
    return (q_ids, a_ids, tool_names, Q_tfidf, A_tfidf, agent_tool_idx_padded, agent_tool_mask, q_vec, a_vec, tool_vec)


def save_trf_cache(cache_dir: str, Q_trf, A_trf, enc_meta):
    np.save(os.path.join(cache_dir, "Q_trf.npy"), Q_trf.astype(np.float32))
    np.save(os.path.join(cache_dir, "A_trf.npy"), A_trf.astype(np.float32))
    with open(os.path.join(cache_dir, "enc_meta.json"), "w", encoding="utf-8") as f: json.dump(enc_meta, f, ensure_ascii=False)


def load_trf_cache(cache_dir: str):
    Q_trf = np.load(os.path.join(cache_dir, "Q_trf.npy"))
    A_trf = np.load(os.path.join(cache_dir, "A_trf.npy"))
    with open(os.path.join(cache_dir, "enc_meta.json"), "r", encoding="utf-8") as f: enc_meta = json.load(f)
    return Q_trf, A_trf, enc_meta

# -------------------- Data I/O --------------------

def load_json(p: str):
    with open(p, "r", encoding="utf-8") as f: return json.load(f)


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


def build_text_corpora(all_agents, all_questions, tools):
    q_ids = list(all_questions.keys())
    q_texts = [all_questions[qid]["input"] for qid in q_ids]

    tool_names = list(tools.keys())
    def _tool_text(tn: str) -> str:
        t = tools.get(tn, {}); desc = t.get("description", ""); return f"{tn} {desc}".strip()

    a_ids, a_texts, a_tool_lists = [], [], []
    for aid, a in all_agents.items():
        mname = a.get("M", {}).get("name", "")
        tool_list = a.get("T", {}).get("tools", []) or []
        concat_tool_desc = " ".join([_tool_text(tn) for tn in tool_list])
        a_ids.append(aid); a_tool_lists.append(tool_list)
        a_texts.append(f"{mname} {concat_tool_desc}".strip())
    return q_ids, q_texts, tool_names, a_ids, a_texts, a_tool_lists

# -------------------- TF‑IDF pipeline --------------------

def build_tfidf(q_texts, tool_names, a_texts, a_tool_lists, max_features: int):
    q_vec = TfidfVectorizer(max_features=max_features, lowercase=True)
    tool_vec = TfidfVectorizer(max_features=max_features, lowercase=True)
    a_vec = TfidfVectorizer(max_features=max_features, lowercase=True)

    Q_csr = q_vec.fit_transform(q_texts)            # (num_q, Dq)
    tool_texts = []
    # tool corpus
    for tn in tool_names:
        # tool description may be null; include name as token
        tool_texts.append(tn)
    Tm_csr = tool_vec.fit_transform(tool_texts)     # (num_tools, Dt)
    Am_csr = a_vec.fit_transform(a_texts)           # (num_agents, Da)

    # Aggregate tool tfidf by mean for each agent
    name2idx = {n:i for i,n in enumerate(tool_names)}
    num_agents = len(a_texts)
    Dt = Tm_csr.shape[1]
    Atool = np.zeros((num_agents, Dt), dtype=np.float32)
    for i, tool_list in enumerate(a_tool_lists):
        idxs = [name2idx[t] for t in tool_list if t in name2idx]
        if idxs:
            vecs = Tm_csr[idxs].toarray().astype(np.float32)
            Atool[i] = vecs.mean(axis=0)

    Am = Am_csr.toarray().astype(np.float32)
    A_tfidf = np.concatenate([Am, Atool], axis=1)  # (num_agents, Da+Dt)
    Q_tfidf = Q_csr.toarray().astype(np.float32)
    return q_vec, tool_vec, a_vec, Q_tfidf, A_tfidf

# -------------------- Transformer sentence embeddings --------------------

@torch.no_grad()
def encode_texts(texts: List[str], tokenizer, encoder, device, max_len=128, batch_size=256, use_cls=True):
    embs = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding (Transformer)", dynamic_ncols=True):
        batch = texts[i:i+batch_size]
        toks = tokenizer(batch, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
        toks = {k:v.to(device) for k,v in toks.items()}
        out = encoder(**toks)
        if hasattr(out, "last_hidden_state"):
            if use_cls:
                vec = out.last_hidden_state[:,0,:]
            else:
                attn = toks["attention_mask"].unsqueeze(-1)
                vec = (out.last_hidden_state * attn).sum(1) / (attn.sum(1).clamp(min=1))
        else:
            vec = out.pooler_output
        embs.append(vec.detach().cpu())
    return torch.cat(embs, dim=0).numpy()

# -------------------- Tool/Agent ID buffers --------------------

def build_agent_tool_id_buffers(a_ids: List[str], agent_tool_lists: List[List[str]], tool_names: List[str]):
    t_map = {n:i for i,n in enumerate(tool_names)}
    num_agents = len(a_ids)
    max_t = max([len(lst) for lst in agent_tool_lists]) if num_agents>0 else 0
    if max_t == 0: max_t = 1
    idx_pad = np.zeros((num_agents, max_t), dtype=np.int64)
    mask = np.zeros((num_agents, max_t), dtype=np.float32)
    for i, lst in enumerate(agent_tool_lists):
        for j, tn in enumerate(lst[:max_t]):
            if tn in t_map:
                idx_pad[i, j] = t_map[tn]
                mask[i, j] = 1.0
    return torch.from_numpy(idx_pad), torch.from_numpy(mask)

# -------------------- BPR model (Hybrid) --------------------

class HybridBPR(nn.Module):
    def __init__(self,
                 d_q_tfidf: int, d_a_tfidf: int,
                 d_trf: int,
                 num_agents: int, num_tools: int,
                 agent_tool_indices_padded: torch.LongTensor,
                 agent_tool_mask: torch.FloatTensor,
                 text_hidden: int = 256, id_dim: int = 64):
        super().__init__()
        # project TF-IDF
        self.q_proj_tfidf = nn.Linear(d_q_tfidf, text_hidden)
        self.a_proj_tfidf = nn.Linear(d_a_tfidf, text_hidden)
        # project Transformer embeddings
        self.q_proj_trf = nn.Linear(d_trf, text_hidden)
        self.a_proj_trf = nn.Linear(d_trf, text_hidden)
        # IDs
        self.emb_agent = nn.Embedding(num_agents, id_dim)
        self.emb_tool = nn.Embedding(num_tools, id_dim)
        # buffers
        self.register_buffer("agent_tool_indices_padded", agent_tool_indices_padded)
        self.register_buffer("agent_tool_mask", agent_tool_mask)
        # scorer: concat(q_tf, q_trf, a_tf, a_trf, agent_id, mean_tool_id)
        in_dim = text_hidden*4 + id_dim*2
        self.scorer = nn.Sequential(
            nn.ReLU(),
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        # init
        for m in [self.q_proj_tfidf, self.a_proj_tfidf, self.q_proj_trf, self.a_proj_trf]:
            nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)
        nn.init.xavier_uniform_(self.emb_agent.weight)
        nn.init.xavier_uniform_(self.emb_tool.weight)
        for m in self.scorer:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)

    def forward_score(self,
                      q_vec_tfidf: torch.Tensor, a_vec_tfidf: torch.Tensor,
                      q_vec_trf: torch.Tensor,   a_vec_trf: torch.Tensor,
                      agent_idx: torch.LongTensor) -> torch.Tensor:
        qh_tf = F.relu(self.q_proj_tfidf(q_vec_tfidf))
        ah_tf = F.relu(self.a_proj_tfidf(a_vec_tfidf))
        qh_tr = F.relu(self.q_proj_trf(q_vec_trf))
        ah_tr = F.relu(self.a_proj_trf(a_vec_trf))
        ae = self.emb_agent(agent_idx)
        idxs = self.agent_tool_indices_padded[agent_idx]
        mask = self.agent_tool_mask[agent_idx]
        te = self.emb_tool(idxs)
        te_mean = (te * mask.unsqueeze(-1)).sum(1) / (mask.sum(1, keepdim=True) + 1e-8)
        x = torch.cat([qh_tf, qh_tr, ah_tf, ah_tr, ae, te_mean], dim=1)
        return self.scorer(x).squeeze(1)

# -------------------- BPR loss & helpers --------------------

def bpr_loss(pos: torch.Tensor, neg: torch.Tensor) -> torch.Tensor:
    return -torch.log(torch.sigmoid(pos - neg) + 1e-8).mean()


def train_valid_split(qids_in_rankings, valid_ratio=0.2, seed=42):
    rng = random.Random(seed); q = list(qids_in_rankings); rng.shuffle(q)
    n_valid = int(len(q) * valid_ratio)
    return q[n_valid:], q[:n_valid]


def build_training_pairs(all_rankings: Dict[str, List[str]], all_agent_ids: List[str], neg_per_pos=1, rng_seed=42):
    rnd = random.Random(rng_seed); pairs=[]; all_set=set(all_agent_ids)
    for qid, ranked_full in all_rankings.items():
        ranked = ranked_full[:pos_topk]; rset=set(ranked)
        neg_pool = list(all_set - rset) or all_agent_ids
        for pos_a in ranked:
            for _ in range(neg_per_pos):
                pairs.append((qid, pos_a, rnd.choice(neg_pool)))
    return pairs, list(all_rankings.keys())

@torch.no_grad()
def evaluate_model(model,
                   Q_tf_t, A_tf_t, Q_tr_t, A_tr_t,
                   qid2idx, a_ids, all_rankings, eval_qids,
                   device="cpu", ks=(5,10,30), cand_size=50, rng_seed=123):
    max_k = max(ks); aid2idx = {aid:i for i,aid in enumerate(a_ids)}
    agg = {k:{m:0.0 for m in ["P","R","F1","Hit","nDCG","MRR"]} for k in ks}
    cnt=skipped=0; all_agent_set=set(a_ids); ref_k = 10 if 10 in ks else max_k
    for qid in tqdm(eval_qids, desc="Evaluating (sampled)", dynamic_ncols=True):
        gt_list = [aid for aid in all_rankings.get(qid, [])[:pos_topk] if aid in aid2idx]
        if not gt_list: skipped+=1; continue
        rel_set=set(gt_list); neg_pool = list(all_agent_set - rel_set)
        rnd = random.Random((hash(qid) ^ (rng_seed * 16777619)) & 0xFFFFFFFF)
        need_neg = max(0, cand_size - len(gt_list))
        if need_neg>0 and len(neg_pool)>0:
            sampled_negs = rnd.sample(neg_pool, min(need_neg,len(neg_pool)))
            cand_ids = gt_list + sampled_negs
        else:
            cand_ids = gt_list
        qi = qid2idx[qid]
        Nc = len(cand_ids)
        q_tf = Q_tf_t[qi:qi+1].repeat(Nc,1)
        q_tr = Q_tr_t[qi:qi+1].repeat(Nc,1)
        cand_idx = torch.tensor([aid2idx[a] for a in cand_ids], dtype=torch.long, device=device)
        a_tf = A_tf_t[cand_idx]; a_tr = A_tr_t[cand_idx]
        scores = model.forward_score(q_tf, a_tf, q_tr, a_tr, cand_idx).detach().cpu().numpy()
        order = np.argsort(-scores)[:max_k]
        pred_ids = [cand_ids[i] for i in order]
        bin_hits = [1 if aid in rel_set else 0 for aid in pred_ids]
        import math as _m
        for k in ks:
            Hk=sum(bin_hits[:k]); P=Hk/float(k); R=Hk/float(len(rel_set)); F1=(2*P*R)/(P+R) if (P+R)>0 else 0.0
            Hit=1.0 if Hk>0 else 0.0
            dcg = sum((1.0/_m.log2(i+2.0)) for i,h in enumerate(bin_hits[:k]) if h)
            ideal = min(len(rel_set), k)
            idcg = sum(1.0/_m.log2(i+2.0) for i in range(ideal)) if ideal>0 else 0.0
            nDCG = (dcg/idcg) if idcg>0 else 0.0
            rr=0.0
            for i in range(k):
                if bin_hits[i]: rr=1.0/float(i+1); break
            agg[k]["P"]+=P; agg[k]["R"]+=R; agg[k]["F1"]+=F1; agg[k]["Hit"]+=Hit; agg[k]["nDCG"]+=nDCG; agg[k]["MRR"]+=rr
        cnt+=1
    if cnt==0:
        return {k:{m:0.0 for m in ["P","R","F1","Hit","nDCG","MRR"]} for k in ks}
    for k in ks:
        for m in agg[k]: agg[k][m] /= cnt
    return agg


def print_metrics_table(title, metrics_dict, ks=(5,10,30)):
    print(f"\n== {title} ==")
    header = f"{'@K':>4} | {'P':>7} {'R':>7} {'F1':>7} {'Hit':>7} {'nDCG':>7} {'MRR':>7}"
    print(header); print("-"*len(header))
    for k in ks:
        m = metrics_dict[k]
        print(f"{k:>4} | {m['P']:.4f} {m['R']:.4f} {m['F1']:.4f} {m['Hit']:.4f} {m['nDCG']:.4f} {m['MRR']:.4f}")

# -------------------- Dataset signature & train cache --------------------

def dataset_signature(a_ids: List[str], all_rankings: Dict[str, List[str]]) -> str:
    payload = {"a_ids": a_ids, "rankings": {k: all_rankings[k] for k in sorted(all_rankings.keys())}}
    sig = zlib.crc32(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")) & 0xFFFFFFFF
    return f"{sig:08x}"


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


def save_training_cache(cache_dir: str, train_qids: List[str], valid_qids: List[str], pairs_idx_np: np.ndarray, meta: Dict):
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

# -------------------- Main --------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    # TF‑IDF
    parser.add_argument("--max_features", type=int, default=5000)
    # Transformer
    parser.add_argument("--pretrained_model", type=str, default="distilbert-base-uncased")
    parser.add_argument("--max_len", type=int, default=128)
    # model dims
    parser.add_argument("--text_hidden", type=int, default=256)
    parser.add_argument("--id_dim", type=int, default=64)
    # training
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--neg_per_pos", type=int, default=1)
    parser.add_argument("--rng_seed_pairs", type=int, default=42)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--valid_ratio", type=float, default=0.2)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--rebuild_cache", type=int, default=0)
    args = parser.parse_args()

    random.seed(1234); np.random.seed(1234); torch.manual_seed(1234)
    device = torch.device(args.device)

    # 1) Load
    all_agents, all_questions, all_rankings, tools = collect_data(args.data_root)
    print(f"Loaded {len(all_agents)} agents, {len(all_questions)} questions, {len(tools)} tools.")

    cache_dir = ensure_cache_dir(args.data_root)

    # 2) Build corpora
    q_ids, q_texts, tool_names, a_ids, a_texts, a_tool_lists = build_text_corpora(all_agents, all_questions, tools)
    agent_tool_idx_padded, agent_tool_mask = build_agent_tool_id_buffers(a_ids, a_tool_lists, tool_names)

    # 3) TF‑IDF cache
    if tfidf_cache_exists(cache_dir) and args.rebuild_cache == 0:
        (q_ids_c, a_ids_c, tool_names_c, Q_tfidf, A_tfidf,
         agent_tool_idx_padded_np, agent_tool_mask_np, q_vec, a_vec, tool_vec) = load_tfidf_cache(cache_dir)
        if (q_ids_c==q_ids) and (a_ids_c==a_ids) and (tool_names_c==tool_names):
            print(f"[cache] loaded TF‑IDF features from {cache_dir}")
            agent_tool_idx_padded = torch.from_numpy(agent_tool_idx_padded_np)
            agent_tool_mask = torch.from_numpy(agent_tool_mask_np)
        else:
            print("[cache] TF‑IDF mismatch; rebuilding...")
            q_vec = a_vec = tool_vec = None
            Q_tfidf = A_tfidf = None
    else:
        q_vec = a_vec = tool_vec = None
        Q_tfidf = A_tfidf = None

    if Q_tfidf is None or A_tfidf is None:
        q_vec, tool_vec, a_vec, Q_tfidf, A_tfidf = build_tfidf(q_texts, tool_names, a_texts, a_tool_lists, args.max_features)
        save_tfidf_cache(cache_dir, q_ids, a_ids, tool_names, Q_tfidf, A_tfidf,
                         agent_tool_idx_padded.numpy(), agent_tool_mask.numpy(), q_vec, a_vec, tool_vec)
        print(f"[cache] saved TF‑IDF features to {cache_dir}")

    # 4) Transformer cache
    trf_ok = trf_cache_exists(cache_dir) and args.rebuild_cache == 0
    Q_trf = A_trf = None
    if trf_ok:
        Q_trf_c, A_trf_c, enc_meta = load_trf_cache(cache_dir)
        # do minimal validation by shape; texts change would be caught by length
        if Q_trf_c.shape[0] == len(q_ids) and A_trf_c.shape[0] == len(a_ids):
            Q_trf, A_trf = Q_trf_c, A_trf_c
            print(f"[cache] loaded Transformer embeddings from {cache_dir} ({enc_meta.get('pretrained_model')})")
        else:
            print("[cache] Transformer shapes mismatch; rebuilding...")

    if Q_trf is None or A_trf is None:
        tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)
        encoder = AutoModel.from_pretrained(args.pretrained_model).to(device).eval()
        Q_trf = encode_texts(q_texts, tokenizer, encoder, device, max_len=args.max_len, batch_size=256, use_cls=True)
        A_trf = encode_texts(a_texts, tokenizer, encoder, device, max_len=args.max_len, batch_size=256, use_cls=True)
        save_trf_cache(cache_dir, Q_trf, A_trf, enc_meta={"pretrained_model": args.pretrained_model, "max_len": args.max_len})
        print(f"[cache] saved Transformer embeddings to {cache_dir}")

    # 5) indices & signature
    qid2idx = {qid:i for i,qid in enumerate(q_ids)}
    aid2idx = {aid:i for i,aid in enumerate(a_ids)}
    data_sig = dataset_signature(a_ids, all_rankings)
    model_path, latest_model, meta_path = model_save_paths(cache_dir, data_sig)

    # 6) train cache
    want_meta = {
        "data_sig": data_sig,
        "neg_per_pos": int(args.neg_per_pos),
        "rng_seed_pairs": int(args.rng_seed_pairs),
        "split_seed": int(args.split_seed),
        "valid_ratio": float(args.valid_ratio),
    }
    use_training_cache = training_cache_exists(cache_dir) and (args.rebuild_cache == 0)
    pairs = None

    if use_training_cache:
        cached_train_qids, cached_valid_qids, pairs_idx_np, meta = load_training_cache(cache_dir)
        if meta == want_meta:
            train_qids, valid_qids = cached_train_qids, cached_valid_qids
            pairs = [(int(q), int(p), int(n)) for (q, p, n) in pairs_idx_np.tolist()]
            print(f"[cache] loaded train/valid/pairs from {cache_dir} (sig={data_sig})")
        else:
            print("[cache] training cache meta mismatch, rebuilding...")

    if pairs is None:
        qids_in_rank = [qid for qid in q_ids if qid in all_rankings]
        train_qids, valid_qids = train_valid_split(qids_in_rank, valid_ratio=args.valid_ratio, seed=args.split_seed)
        print(f"[split] train={len(train_qids)}  valid={len(valid_qids)}")
        rankings_train = {qid: all_rankings[qid] for qid in train_qids}
        pairs_id, _ = build_training_pairs(rankings_train, a_ids, neg_per_pos=args.neg_per_pos, rng_seed=args.rng_seed_pairs)
        pairs = [(qid2idx[q], aid2idx[p], aid2idx[n]) for (q,p,n) in pairs_id]
        save_training_cache(cache_dir, train_qids, valid_qids, np.array(pairs, dtype=np.int64), want_meta)
        print(f"[cache] saved train/valid/pairs to {cache_dir} (sig={data_sig})")

    # 7) tensors & model
    Q_tf_t = torch.from_numpy(Q_tfidf).to(device)
    A_tf_t = torch.from_numpy(A_tfidf).to(device)
    Q_tr_t = torch.from_numpy(Q_trf).to(device)
    A_tr_t = torch.from_numpy(A_trf).to(device)

    d_q_tfidf = Q_tf_t.shape[1]
    d_a_tfidf = A_tf_t.shape[1]
    d_trf = Q_tr_t.shape[1]

    model = HybridBPR(
        d_q_tfidf=d_q_tfidf, d_a_tfidf=d_a_tfidf, d_trf=d_trf,
        num_agents=len(a_ids), num_tools=len(tool_names),
        agent_tool_indices_padded=agent_tool_idx_padded.to(device),
        agent_tool_mask=agent_tool_mask.to(device),
        text_hidden=args.text_hidden, id_dim=args.id_dim
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # 8) training loop
    num_pairs = len(pairs); num_batches = math.ceil(num_pairs / args.batch_size)
    for epoch in range(1, args.epochs + 1):
        random.shuffle(pairs); total=0.0
        pbar = tqdm(range(num_batches), desc=f"Epoch {epoch}/{args.epochs}", dynamic_ncols=True)
        for b in pbar:
            batch = pairs[b*args.batch_size:(b+1)*args.batch_size]
            if not batch: continue
            q_idx  = torch.tensor([t[0] for t in batch], dtype=torch.long, device=device)
            pos_idx= torch.tensor([t[1] for t in batch], dtype=torch.long, device=device)
            neg_idx= torch.tensor([t[2] for t in batch], dtype=torch.long, device=device)
            q_tf = Q_tf_t[q_idx]; q_tr = Q_tr_t[q_idx]
            pos_tf = A_tf_t[pos_idx]; pos_tr = A_tr_t[pos_idx]
            neg_tf = A_tf_t[neg_idx]; neg_tr = A_tr_t[neg_idx]
            pos_s = model.forward_score(q_tf, pos_tf, q_tr, pos_tr, pos_idx)
            neg_s = model.forward_score(q_tf, neg_tf, q_tr, neg_tr, neg_idx)
            loss = bpr_loss(pos_s, neg_s)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total += loss.item()
            pbar.set_postfix({"batch_loss": f"{loss.item():.4f}", "avg_loss": f"{(total/(b+1)):.4f}"})
        print(f"Epoch {epoch}/{args.epochs} - BPR loss: {(total/max(1,num_batches)):.4f}")

    # 9) save
    ckpt = {
        "state_dict": model.state_dict(),
        "data_sig": data_sig,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "args": vars(args),
        "dims": {
            "d_q_tfidf": int(d_q_tfidf), "d_a_tfidf": int(d_a_tfidf), "d_trf": int(d_trf),
            "num_agents": int(len(a_ids)), "num_tools": int(len(tool_names)),
            "text_hidden": int(args.text_hidden), "id_dim": int(args.id_dim)
        },
        "pretrained_model": args.pretrained_model,
        "max_features": int(args.max_features),
    }
    model_path, latest_model, meta_path = model_save_paths(cache_dir, data_sig)
    torch.save(ckpt, model_path); torch.save(ckpt, latest_model)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({
            "data_sig": data_sig,
            "a_ids": a_ids,
            "tool_names": tool_names,
            "pretrained_model": args.pretrained_model,
            "max_features": args.max_features,
        }, f, ensure_ascii=False, indent=2)

    print(f"[save] model -> {model_path}")
    print(f"[save] meta  -> {meta_path}")

    # 10) validate & demo
    def recommend_topk_for_qid(qid: str, topk: int = 10) -> List[Tuple[str, float]]:
        qi = qid2idx[qid]
        q_tf = Q_tf_t[qi:qi+1].repeat(len(a_ids),1)
        q_tr = Q_tr_t[qi:qi+1].repeat(len(a_ids),1)
        a_idx = torch.arange(len(a_ids), dtype=torch.long, device=device)
        with torch.no_grad():
            scores = model.forward_score(q_tf, A_tf_t, q_tr, A_tr_t, a_idx).cpu().numpy()
        order = np.argsort(-scores)[:topk]
        return [(a_ids[i], float(scores[i])) for i in order]

    valid_qids = [qid for qid in q_ids if qid in all_rankings]
    # we reused split earlier; for reporting, we evaluate on the formerly built valid set if present
    if training_cache_exists(cache_dir):
        _, valid_qids_loaded, _, _ = load_training_cache(cache_dir)
        if valid_qids_loaded:
            valid_qids = valid_qids_loaded

    metrics = evaluate_model(model, Q_tf_t, A_tf_t, Q_tr_t, A_tr_t, qid2idx, a_ids, all_rankings,
                             valid_qids, device=device, ks=(5,10,30), cand_size=50, rng_seed=123)
    print_metrics_table("Validation (averaged over questions)", metrics, ks=(5,10,30))

    sample_qids = q_ids[:min(5, len(q_ids))]
    for qid in sample_qids:
        topk = recommend_topk_for_qid(qid, topk=args.topk)
        print(f"\nQuestion: {qid}  |  {all_questions[qid]['input'][:80]}")
        for rank, (aid, s) in enumerate(topk, 1):
            print(f"  {rank:2d}. {aid:>20s}  score={s:.4f}")

if __name__ == "__main__":
    main()
