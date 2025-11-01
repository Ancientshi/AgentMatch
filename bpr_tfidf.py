
# Simple, self-contained implementation of a TF-IDF + DNN (BPR loss) agent recommender.
# It reads PartI/PartII/PartIII + Tools folders structured as the user described.
# Usage example (after saving this script):
#   python /mnt/data/simple_bpr_tfidf_agent_rec.py --data_root /path/to/dataset_root --epochs 3 --batch_size 256 --max_features 5000 --neg_per_pos 1 --topk 10
#
# The script will train a tiny model and print top-K recommendations for a few sample questions.
#
# No try/except, minimal dependencies: numpy, torch, scikit-learn.
#
# File is saved to /mnt/data/simple_bpr_tfidf_agent_rec.py

'''
简短说：用了“文本特征 + ID 特征”，且不同对象用得不一样。

* Query（用户问题）

  * **文本特征**：问题文本的 TF-IDF 向量 `Q`（`build_vectorizers` 生成，训练/推理时用 `q_vec`）。
  * **没有**使用 query 的 ID 向量/embedding。

* Agent（待推荐的智能体）

  * **文本特征（两部分拼接）**：

    1. Agent 自身文本的 TF-IDF（模型名等，`Am`）。
    2. 其工具文本 TF-IDF 的**均值聚合**（`Atool`）。
       最终形成 `A_text_full = [Am ; Atool]` 并输入 `a_vec`→`a_proj`。
  * **ID 特征**：Agent ID 的可学习 embedding（`emb_agent` → `ae`）。

* Tool（工具）

  * **文本特征**：用于上面 Agent 的工具文本聚合（只在 `Atool` 中**间接**使用，不单独入模）。
  * **ID 特征**：每个工具有可学习 embedding（`emb_tool`），对同一 Agent 的工具做**掩码加权均值**得到 `te_mean`，作为该 Agent 的工具-ID 表示。

* 最终打分输入（`SimpleBPRDNN.scorer`）
  `concat( q_text_hidden , agent_text_hidden , agent_ID_emb , mean_tool_ID_emb )`
  即：**问题文本(投影) + Agent文本(投影) + Agent ID + 工具ID均值**。
  损失用 BPR，对 (q, pos_agent, neg_agent) 成对优化。
'''

import pickle
from datetime import datetime
import os
import json
import math
import random
import argparse
import zlib
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm.auto import tqdm
from utils import print_metrics_table

filename = os.path.splitext(os.path.basename(__file__))[0]
pos_topk = 5 # only regard top-k ranked agents as positives

def model_save_paths(cache_dir: str, data_sig: str):
    """返回模型与向量器保存路径"""
    model_dir = os.path.join(cache_dir, "models")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"{filename}_{data_sig}.pt")
    qvec_path  = os.path.join(model_dir, f"q_vectorizer_{data_sig}.pkl")
    latest_model = os.path.join(model_dir, f"latest_{data_sig}.pt")  # 便于 server 默认加载
    latest_qvec  = os.path.join(model_dir, f"latest_qvec_{data_sig}.pkl")
    meta_path    = os.path.join(model_dir, f"meta_{data_sig}.json")
    return model_path, qvec_path, latest_model, latest_qvec, meta_path

# ---------- split & metrics helpers ----------
def train_valid_split(qids_in_rankings, valid_ratio=0.2, seed=42):
    rng = random.Random(seed)
    q = list(qids_in_rankings)
    rng.shuffle(q)
    n_valid = int(len(q) * valid_ratio)
    valid_q = q[:n_valid]
    train_q = q[n_valid:]
    return train_q, valid_q

def _dcg_at_k(binary_hits, k):
    # binary_hits 是长度 k 的 0/1 列表
    dcg = 0.0
    for i, h in enumerate(binary_hits[:k]):
        if h:
            dcg += 1.0 / math.log2(i + 2.0)
    return dcg

def _idcg_at_k(num_rels, k):
    ideal = min(num_rels, k)
    idcg = 0.0
    for i in range(ideal):
        idcg += 1.0 / math.log2(i + 2.0)
    return idcg

@torch.no_grad()
def evaluate_model(model, Q_t, A_text_t, qid2idx, a_ids, all_rankings,
                   eval_qids, device="cpu", ks=(5,10,50),
                   cand_size=1000, rng_seed=123):
    """
    评测时仅对一个“候选集”打分：
      candidates = ground-truth agents (来自 ranking list) ∪ 随机负样本
      若 ground-truth 数量 >= cand_size，则 candidates=ground-truth（不再补负样本）
    注意：指标是在抽样候选集上计算的近似值（不是全库上的绝对值）。
    """

    max_k = max(ks)
    # id 映射
    aid2idx = {aid: i for i, aid in enumerate(a_ids)}

    # 累计器
    agg = {k: {"P":0.0,"R":0.0,"F1":0.0,"Hit":0.0,"nDCG":0.0,"MRR":0.0} for k in ks}
    cnt = 0
    skipped = 0

    # 进度条里展示的参考 K（优先 10）
    ref_k = 10 if 10 in ks else max_k

    # 负样本池（全集）
    all_agent_set = set(a_ids)

    pbar = tqdm(eval_qids, desc="Evaluating (sampled)", leave=True, dynamic_ncols=True)
    for qid in pbar:
        # ground-truth 可能是 1 个或列表，我们统一按列表处理
        gt_list = [aid for aid in all_rankings.get(qid, [])[:pos_topk] if aid in aid2idx]
        if len(gt_list) == 0:
            skipped += 1
            pbar.set_postfix({"done": cnt, "skipped": skipped})
            continue

        rel_set = set(gt_list)
        neg_pool = list(all_agent_set - rel_set)

        # 构造候选集：先放全部正样本，再随机补负样本到 cand_size
        # 为了可复现，按 qid 和 rng_seed 混合出一个局部随机源
        rnd = random.Random((hash(qid) ^ (rng_seed * 16777619)) & 0xFFFFFFFF)
        need_neg = max(0, cand_size - len(gt_list))
        if need_neg > 0 and len(neg_pool) > 0:
            # 如果负样本不够，sample 的 k 会被截断到可用数量
            k = min(need_neg, len(neg_pool))
            sampled_negs = rnd.sample(neg_pool, k)
            cand_ids = gt_list + sampled_negs
        else:
            cand_ids = gt_list  # ground-truth 已经 >= cand_size，就不补了

        # 张量准备
        qi = qid2idx[qid]
        qv = Q_t[qi:qi+1].repeat(len(cand_ids), 1)  # (Nc, d_q)
        cand_idx = torch.tensor([aid2idx[a] for a in cand_ids], dtype=torch.long, device=device)
        av = A_text_t[cand_idx]

        # 打分（候选集很小，不再切块）
        scores = model.forward_score(qv, av, cand_idx).detach().cpu().numpy()
        order = np.argsort(-scores)[:max_k]
        pred_ids = [cand_ids[i] for i in order]

        # 二元命中列用于各指标
        bin_hits = [1 if aid in rel_set else 0 for aid in pred_ids]

        for k in ks:
            Hk = sum(bin_hits[:k])
            P = Hk / float(k)
            R = Hk / float(len(rel_set))
            F1 = (2*P*R)/(P+R) if (P+R) > 0 else 0.0
            Hit = 1.0 if Hk > 0 else 0.0

            # nDCG@k（binary relevance）
            dcg = 0.0
            for i, h in enumerate(bin_hits[:k]):
                if h:
                    dcg += 1.0 / math.log2(i + 2.0)
            ideal = min(len(rel_set), k)
            idcg = sum(1.0 / math.log2(i + 2.0) for i in range(ideal)) if ideal > 0 else 0.0
            nDCG = (dcg / idcg) if idcg > 0 else 0.0

            # MRR@k
            rr = 0.0
            for i in range(k):
                if bin_hits[i]:
                    rr = 1.0 / float(i+1)
                    break

            agg[k]["P"]   += P
            agg[k]["R"]   += R
            agg[k]["F1"]  += F1
            agg[k]["Hit"] += Hit
            agg[k]["nDCG"]+= nDCG
            agg[k]["MRR"] += rr

        cnt += 1

        # 进度条显示在线均值（@ref_k）
        ref = agg[ref_k]
        pbar.set_postfix({
            "done": cnt,
            "skipped": skipped,
            f"P@{ref_k}":    f"{(ref['P']/cnt):.4f}",
            f"nDCG@{ref_k}": f"{(ref['nDCG']/cnt):.4f}",
            f"MRR@{ref_k}":  f"{(ref['MRR']/cnt):.4f}",
            "Ncand": len(cand_ids)
        })

    if cnt == 0:
        return {k: {m:0.0 for m in ["P","R","F1","Hit","nDCG","MRR"]} for k in ks}

    for k in ks:
        for m in agg[k]:
            agg[k][m] /= cnt
    return agg





def ensure_cache_dir(root: str) -> str:
    d = os.path.join(root, f".cache/{filename}")
    if not os.path.exists(d):
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
               agent_tool_idx_padded, agent_tool_mask,
               ):
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


def load_json(p: str):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def collect_data(data_root: str):
    """
    Reads:
      - PartI/agents/merge.json, PartI/questions/merge.json, PartI/rankings/merge.json
      - PartII/...
      - PartIII/...
      - Tools/merge.json
    Returns unified dicts and mappings.
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

        # Merge
        all_agents.update(agents)
        all_questions.update(questions)
        all_rankings.update(rankings["rankings"])

    tools_path = os.path.join(data_root, "Tools", "merge.json")
    tools = load_json(tools_path)

    return all_agents, all_questions, all_rankings, tools


def build_text_corpora(all_agents, all_questions, tools):
    """
    Build minimal text strings per entity:
      - Question text: questions[qid]["input"]
      - Tool text: tools[name]["description"]  (fallback to empty string if missing)
      - Agent text: agent["M"]["name"] + tool names + tool descriptions (concatenated)
    """
    # Questions
    q_ids = list(all_questions.keys())
    q_texts = [all_questions[qid]["input"] for qid in q_ids]

    # Tools
    tool_names = list(tools.keys())
    def _tool_text(tn: str) -> str:
        t = tools.get(tn, {})
        desc = t.get("description", "")
        return f"{tn} {desc}".strip()

    tool_texts = [_tool_text(tn) for tn in tool_names]

    # Agents
    a_ids = list(all_agents.keys())
    a_texts = []
    a_tool_lists = []
    for aid in a_ids:
        a = all_agents[aid]
        mname = a.get("M", {}).get("name", "")
        tool_list = a.get("T", {}).get("tools", []) or []
        a_tool_lists.append(tool_list)

        # concat: model name + tool names + tool descriptions
        concat_tool_desc = " ".join([_tool_text(tn) for tn in tool_list])
        text = f"{mname} {concat_tool_desc}".strip()
        a_texts.append(text)

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
    """
    For each agent, aggregate the TF-IDF vectors of its tools by mean. If no tools -> zeros.
    Returns a dense array (num_agents, Dt).
    """
    name2idx = {n: i for i, n in enumerate(tool_names)}
    num_agents = len(agent_tool_lists)
    Dt = Tm.shape[1]

    Atool = np.zeros((num_agents, Dt), dtype=np.float32)
    for i, tool_list in enumerate(agent_tool_lists):
        idxs = [name2idx[t] for t in tool_list if t in name2idx]
        if len(idxs) == 0:
            continue
        vecs = Tm[idxs].toarray()
        Atool[i] = vecs.mean(axis=0).astype(np.float32)
    return Atool


def to_dense_float32_csr(X):
    A = X.astype(np.float32)
    if hasattr(A, "toarray"):
        return A.toarray()
    return np.asarray(A, dtype=np.float32)


class SimpleBPRDNN(nn.Module):
    def __init__(self, d_q: int, d_a: int, num_agents: int, num_tools: int,
                 agent_tool_indices_padded: torch.LongTensor,
                 agent_tool_mask: torch.FloatTensor,
                 text_hidden: int = 256, id_dim: int = 64):
        super().__init__()
        # Project TF-IDF to compact hidden
        self.q_proj = nn.Linear(d_q, text_hidden)
        self.a_proj = nn.Linear(d_a, text_hidden)

        # ID embeddings
        self.emb_agent = nn.Embedding(num_agents, id_dim)
        self.emb_tool = nn.Embedding(num_tools, id_dim)

        # Buffers for per-agent tool lists (for mean embedding)
        self.register_buffer("agent_tool_indices_padded", agent_tool_indices_padded)  # (num_agents, max_t)
        self.register_buffer("agent_tool_mask", agent_tool_mask)                      # (num_agents, max_t)

        # Small MLP to score (q, a)
        in_dim = text_hidden + text_hidden + id_dim + id_dim
        self.scorer = nn.Sequential(
            nn.ReLU(),
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        # init
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.zeros_(self.q_proj.bias)
        nn.init.xavier_uniform_(self.a_proj.weight)
        nn.init.zeros_(self.a_proj.bias)
        nn.init.xavier_uniform_(self.emb_agent.weight)
        nn.init.xavier_uniform_(self.emb_tool.weight)
        for m in self.scorer:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward_score(self, q_vec: torch.Tensor, a_vec: torch.Tensor,
                      agent_idx: torch.LongTensor) -> torch.Tensor:
        """
        q_vec: (B, d_q) dense float
        a_vec: (B, d_a) dense float
        agent_idx: (B,) long
        Returns: (B,) scores
        """
        qh = F.relu(self.q_proj(q_vec))      # (B, Ht)
        ah = F.relu(self.a_proj(a_vec))      # (B, Ht)

        ae = self.emb_agent(agent_idx)       # (B, E)

        # mean tool embedding for this agent
        # Select padded indices/masks for these agents
        idxs = self.agent_tool_indices_padded[agent_idx]   # (B, max_t)
        mask = self.agent_tool_mask[agent_idx]             # (B, max_t)
        te = self.emb_tool(idxs)                           # (B, max_t, E)
        mask3 = mask.unsqueeze(-1)                         # (B, max_t, 1)
        te_mean = (te * mask3).sum(dim=1) / (mask.sum(dim=1, keepdim=True) + 1e-8)  # (B, E)

        x = torch.cat([qh, ah, ae, te_mean], dim=1)        # (B, Ht+Ht+E+E)
        s = self.scorer(x).squeeze(1)                      # (B,)
        return s


def bpr_loss(pos: torch.Tensor, neg: torch.Tensor) -> torch.Tensor:
    return -torch.log(torch.sigmoid(pos - neg) + 1e-8).mean()


def build_agent_tool_id_buffers(a_ids: List[str],
                                agent_tool_lists: List[List[str]],
                                tool_names: List[str]) -> Tuple[torch.LongTensor, torch.FloatTensor]:
    """
    Returns padded tool indices per agent and a mask.
    Shapes: (num_agents, max_tools_per_agent)
    """
    t_map = {n: i for i, n in enumerate(tool_names)}
    num_agents = len(a_ids)
    max_t = max([len(lst) for lst in agent_tool_lists]) if num_agents > 0 else 0
    if max_t == 0:
        max_t = 1  # keep non-zero second dim

    idx_pad = np.zeros((num_agents, max_t), dtype=np.int64)
    mask = np.zeros((num_agents, max_t), dtype=np.float32)

    for i, lst in enumerate(agent_tool_lists):
        for j, tn in enumerate(lst[:max_t]):
            if tn in t_map:
                idx_pad[i, j] = t_map[tn]
                mask[i, j] = 1.0

    return torch.from_numpy(idx_pad), torch.from_numpy(mask)


def build_training_pairs(all_rankings: Dict[str, List[str]],
                         all_agent_ids: List[str],
                         neg_per_pos: int = 1,
                         rng_seed: int = 42
                         ) -> Tuple[List[Tuple[str, str, str]], List[str]]:
    """
    返回 [(qid_str, pos_agent_id, neg_agent_id)], qid 为字符串而不是整数索引
    """
    rnd = random.Random(rng_seed)
    pairs = []
    all_agent_set = set(all_agent_ids)

    qid_list = list(all_rankings.keys())
    for qid in qid_list:
        ranked = all_rankings[qid][:pos_topk]
        ranked_set = set(ranked)
        negatives_pool = list(all_agent_set - ranked_set)
        if not negatives_pool:
            negatives_pool = all_agent_ids  # 极端情况下兜底

        for pos_a in ranked:
            for _ in range(neg_per_pos):
                neg_a = rnd.choice(negatives_pool)
                pairs.append((qid, pos_a, neg_a))

    return pairs, qid_list


# -------------------- Dataset signature & training cache --------------------
def dataset_signature(a_ids: List[str], all_rankings: Dict[str, List[str]]) -> str:
    """Create a lightweight signature (CRC32) for the current dataset identity.
    Uses agent id order and rankings mapping (sorted by key) to detect changes.
    """
    payload = {
        "a_ids": a_ids,
        "rankings": {k: all_rankings[k] for k in sorted(all_rankings.keys())},
    }
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    sig = zlib.crc32(blob) & 0xFFFFFFFF
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

def save_training_cache(cache_dir: str,
                        train_qids: List[str],
                        valid_qids: List[str],
                        pairs_idx_np: np.ndarray,
                        meta: Dict):
    p_train, p_valid, p_pairs, p_meta = training_cache_paths(cache_dir)
    with open(p_train, "w", encoding="utf-8") as f:
        json.dump(train_qids, f, ensure_ascii=False)
    with open(p_valid, "w", encoding="utf-8") as f:
        json.dump(valid_qids, f, ensure_ascii=False)
    np.save(p_pairs, pairs_idx_np.astype(np.int64))
    with open(p_meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, sort_keys=True)

def load_training_cache(cache_dir: str):
    p_train, p_valid, p_pairs, p_meta = training_cache_paths(cache_dir)
    with open(p_train, "r", encoding="utf-8") as f:
        train_qids = json.load(f)
    with open(p_valid, "r", encoding="utf-8") as f:
        valid_qids = json.load(f)
    pairs_idx_np = np.load(p_pairs)
    with open(p_meta, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return train_qids, valid_qids, pairs_idx_np, meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True, help="Root folder containing PartI/PartII/PartIII/Tools")
    parser.add_argument("--max_features", type=int, default=5000, help="TF-IDF max features per vectorizer")
    parser.add_argument("--text_hidden", type=int, default=256, help="Hidden size for text projections")
    parser.add_argument("--id_dim", type=int, default=64, help="Embedding dim for agent/tool IDs")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--neg_per_pos", type=int, default=1)
    parser.add_argument("--rng_seed_pairs", type=int, default=42, help="RNG seed for negative sampling in BPR pairs")
    parser.add_argument("--split_seed", type=int, default=42, help="RNG seed for train/valid split")
    parser.add_argument("--valid_ratio", type=float, default=0.2)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--rebuild_cache", type=int, default=0, help="Force rebuild TF-IDF cache (0/1)")
    parser.add_argument("--rebuild_training_cache", type=int, default=0, help="Force rebuild train/valid/pairs cache (0/1)")

    args = parser.parse_args()

    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)

    # 1) Load data
    all_agents, all_questions, all_rankings, tools = collect_data(args.data_root)
    print(f"Loaded {len(all_agents)} agents, {len(all_questions)} questions, {len(tools)} tools.")

    cache_dir = ensure_cache_dir(args.data_root)

    # ---------- Feature cache (TF-IDF, ids, buffers) ----------
    if cache_exists(cache_dir) and args.rebuild_cache == 0:
        (q_ids, a_ids, tool_names, Q, A_text_full,
        agent_tool_idx_padded_np, agent_tool_mask_np) = load_cache(cache_dir)
        print(f"[cache] loaded features from {cache_dir}")
        qid2idx = {qid: i for i, qid in enumerate(q_ids)}
        aid2idx = {aid: i for i, aid in enumerate(a_ids)}

        Q = Q.astype(np.float32)
        A_text_full = A_text_full.astype(np.float32)
        agent_tool_idx_padded = torch.from_numpy(agent_tool_idx_padded_np)
        agent_tool_mask = torch.from_numpy(agent_tool_mask_np)
        
        q_vectorizer_runtime = None  # 默认没有



    else:
        # -------- 正常全流程计算，并写入缓存 --------
        # 2) 文本语料
        q_ids, q_texts, tool_names, tool_texts, a_ids, a_texts, a_tool_lists = build_text_corpora(
            all_agents, all_questions, tools
        )

        # 3) TF-IDF
        q_vec, tool_vec, a_vec, Q_csr, Tm_csr, Am_csr = build_vectorizers(q_texts, tool_texts, a_texts, args.max_features)

        # 4) 工具文本聚合到 agent
        Atool = agent_tool_text_matrix(a_tool_lists, tool_names, Tm_csr)  # (num_agents, Dt)
        Am = Am_csr.toarray().astype(np.float32)                           # (num_agents, Da)
        A_text_full = np.concatenate([Am, Atool], axis=1)                  # (num_agents, Da+Dt)
        Q = Q_csr.toarray().astype(np.float32)                             # (num_q, Dq)

        # 5) agent 的工具 ID 缓冲
        agent_tool_idx_padded, agent_tool_mask = build_agent_tool_id_buffers(a_ids, a_tool_lists, tool_names)

        # ---- 写缓存 ----
        save_cache(
            cache_dir,
            q_ids, a_ids, tool_names,
            Q, A_text_full,
            agent_tool_idx_padded.numpy(), agent_tool_mask.numpy(),
        )
        print(f"[cache] saved features to {cache_dir}")

        q_vectorizer_runtime = q_vec  # 留一份，训练结束后与模型一起按 data_sig 存盘

        qid2idx = {qid: i for i, qid in enumerate(q_ids)}
        aid2idx = {aid: i for i, aid in enumerate(a_ids)}


    data_sig = dataset_signature(a_ids, all_rankings)
    # 统一在 data_sig 就绪后设置保存/加载路径
    model_path, qvec_path, latest_model, latest_qvec, meta_path = model_save_paths(cache_dir, data_sig)

    # 如果之前是缓存分支（没有 q_vectorizer），此时尝试从磁盘加载
    if q_vectorizer_runtime is None and os.path.exists(qvec_path):
        with open(qvec_path, "rb") as f:
            q_vectorizer_runtime = pickle.load(f)


    want_meta = {
        "data_sig": data_sig,
        "neg_per_pos": int(args.neg_per_pos),
        "rng_seed_pairs": int(args.rng_seed_pairs),
        "split_seed": int(args.split_seed),
        "valid_ratio": float(args.valid_ratio),
    }

    use_training_cache = training_cache_exists(cache_dir) and (args.rebuild_training_cache == 0)
    train_qids = valid_qids = None
    pairs = None

    if use_training_cache:
        cached_train_qids, cached_valid_qids, pairs_idx_np, meta = load_training_cache(cache_dir)
        if meta == want_meta:
            train_qids = cached_train_qids
            valid_qids = cached_valid_qids
            pairs = [(int(q), int(p), int(n)) for (q, p, n) in pairs_idx_np.tolist()]
            print(f"[cache] loaded train/valid/pairs from {cache_dir} (sig={data_sig})")
        else:
            print("[cache] training cache meta mismatch, rebuilding...")

    if train_qids is None or pairs is None:
        # 7) 训练对
        # ------- 8:2 划分（仅对有 ranking 的问题） -------
        qids_in_rank = [qid for qid in q_ids if qid in all_rankings]
        # 固定可复现划分
        train_qids, valid_qids = train_valid_split(qids_in_rank, valid_ratio=args.valid_ratio, seed=args.split_seed)
        print(f"[split] train={len(train_qids)}  valid={len(valid_qids)}")

        # ------- 按 train 构建训练对 -------
        rankings_train = {qid: all_rankings[qid] for qid in train_qids}
        pairs_id, _ = build_training_pairs(rankings_train, a_ids, neg_per_pos=args.neg_per_pos, rng_seed=args.rng_seed_pairs)
        pairs = [(qid2idx[qid], aid2idx[pos], aid2idx[neg]) for (qid, pos, neg) in pairs_id]

        # 写训练缓存（以 idx 表示，便于快速加载）
        pairs_idx_np = np.array(pairs, dtype=np.int64)
        save_training_cache(cache_dir, train_qids, valid_qids, pairs_idx_np, want_meta)
        print(f"[cache] saved train/valid/pairs to {cache_dir} (sig={data_sig})")

    # 8) Model 准备
    d_q = Q.shape[1]
    d_a = A_text_full.shape[1]
    num_agents = len(a_ids)
    num_tools = len(tool_names)

    device = torch.device(args.device)
    model = SimpleBPRDNN(
        d_q=d_q,
        d_a=d_a,
        num_agents=num_agents,
        num_tools=num_tools,
        agent_tool_indices_padded=agent_tool_idx_padded.to(device),
        agent_tool_mask=agent_tool_mask.to(device),
        text_hidden=args.text_hidden,
        id_dim=args.id_dim
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # 9) numpy -> torch
    Q_t = torch.from_numpy(Q).to(device)
    A_text_t = torch.from_numpy(A_text_full).to(device)

    # Training loop with tqdm progress bar
    num_pairs = len(pairs)
    num_batches = math.ceil(num_pairs / args.batch_size)

    for epoch in range(1, args.epochs + 1):
        random.shuffle(pairs)
        total_loss = 0.0

        pbar = tqdm(range(num_batches), desc=f"Epoch {epoch}/{args.epochs}", leave=True, dynamic_ncols=True)
        for b in pbar:
            batch_pairs = pairs[b * args.batch_size : (b + 1) * args.batch_size]
            if not batch_pairs:
                continue

            q_idx  = torch.tensor([t[0] for t in batch_pairs], dtype=torch.long, device=device)
            pos_idx= torch.tensor([t[1] for t in batch_pairs], dtype=torch.long, device=device)
            neg_idx= torch.tensor([t[2] for t in batch_pairs], dtype=torch.long, device=device)

            q_vec   = Q_t[q_idx]        # (B, d_q)
            pos_vec = A_text_t[pos_idx] # (B, d_a)
            neg_vec = A_text_t[neg_idx] # (B, d_a)

            pos_score = model.forward_score(q_vec, pos_vec, pos_idx)  # (B,)
            neg_score = model.forward_score(q_vec, neg_vec, neg_idx)  # (B,)

            loss = bpr_loss(pos_score, neg_score)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            running_avg = total_loss / (b + 1)

            # 在进度条右侧实时展示 batch loss 和平均损失
            pbar.set_postfix({
                "batch_loss": f"{loss.item():.4f}",
                "avg_loss": f"{running_avg:.4f}"
            })

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        print(f"Epoch {epoch}/{args.epochs} - BPR loss: {avg_loss:.4f}")

    # ===== 默认持久化：模型 + 向量器 + 元信息 =====
    ckpt = {
        "state_dict": model.state_dict(),
        "data_sig": data_sig,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "args": vars(args),
        "dims": {
            "d_q": int(Q.shape[1]),
            "d_a": int(A_text_full.shape[1]),
            "num_agents": int(len(a_ids)),
            "num_tools": int(len(tool_names)),
            "text_hidden": int(args.text_hidden),
            "id_dim": int(args.id_dim),
        },
    }
    torch.save(ckpt, model_path)
    torch.save(ckpt, latest_model)  # 方便 server 用 latest_*.pt

    # 若本轮有可用的 question 向量器，则也一并保存
    if q_vectorizer_runtime is not None:
        with open(qvec_path, "wb") as f:
            pickle.dump(q_vectorizer_runtime, f)
        with open(latest_qvec, "wb") as f:
            pickle.dump(q_vectorizer_runtime, f)

    # 保存少量服务端需要的元信息（映射等）
    serving_meta = {
        "data_sig": data_sig,
        "a_ids": a_ids,
        "tool_names": tool_names,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(serving_meta, f, ensure_ascii=False, indent=2)

    print(f"[save] model -> {model_path}")
    print(f"[save] q_vectorizer -> {qvec_path}")
    print(f"[save] meta -> {meta_path}")


    # ===== Inference helper: recommend topK agents for a question id =====
    def recommend_topk_for_qid(qid: str, topk: int = 10) -> List[Tuple[str, float]]:
        qi = qid2idx[qid]
        qv = Q_t[qi : qi + 1].repeat(len(a_ids), 1)           # (num_a, d_q)
        av = A_text_t                                         # (num_a, d_a)
        a_idx = torch.arange(len(a_ids), dtype=torch.long, device=device)
        with torch.no_grad():
            scores = model.forward_score(qv, av, a_idx).cpu().numpy()
        order = np.argsort(-scores)[:topk]
        return [(a_ids[i], float(scores[i])) for i in order]

    # ------- Validation metrics -------
    valid_metrics = evaluate_model(
        model, Q_t, A_text_t, qid2idx, a_ids, all_rankings,
        valid_qids, device=device, ks=(5,10,50), cand_size=1000, rng_seed=123
    )
    print_metrics_table("Validation (averaged over questions)", valid_metrics, ks=(5,10,50), filename=filename)

    # Demo: show a few questions and top-K agents
    sample_qids = q_ids[:min(5, len(q_ids))]
    for qid in sample_qids:
        topk = recommend_topk_for_qid(qid, topk=args.topk)
        print(f"\\nQuestion: {qid}  |  {all_questions[qid]['input'][:80]}")
        for rank, (aid, s) in enumerate(topk, 1):
            print(f"  {rank:2d}. {aid:>20s}  score={s:.4f}")

if __name__ == "__main__":
    main()
