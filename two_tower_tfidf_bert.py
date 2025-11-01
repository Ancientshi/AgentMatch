#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
关键设计点（贴合你的诉求）
用可学习的 logit_scale取代了固定的 temperature。
InfoNCE 的温度本质上就是把 logits 除以 
𝑇
T 来控制分布的“锐度”；用一个可学习缩放（比如 logit_scale）等价于让模型自己学一个有效温度，更稳也更省调参。

四路向量：q_tfidf / q_bert / a_tfidf / a_bert；各自经过 MLP → gate 融合（每侧一个 gate，初值 0.5，可学习）。

DNN 输出：不是单纯余弦/点积；打分使用 logit = scale*(q @ a^T + q @ W @ a^T)，其中 W 可学习（双线性层），scale 可学习。既保留双塔高效性，又比单纯点积更有表达力。

in-batch 负样本：与你 Two-Tower 的 InfoNCE 思路一致，显式正对构造（TopK 当正），负样本来自同批其它正对的 a。

OOM 安全：训练时只搬当前 batch 的 (Q_tf,Q_bt,A_tf,A_bt)；评测/推理对 agent 侧分块编码与打分。

工具信息双通道：文本层面已经并入 TF-IDF/BERT；另外还可以打开 ID-embedding 聚合（与之前脚本风格一致）。

你可能会想调的超参/开关

--hid（默认 256）：四路 MLP 的隐藏维与融合后的塔维度。

--amp 1：CUDA 上用 bfloat16 autocast，进一步省显存提吞吐。

--eval_chunk：推理/评测时 agent 分块大小，建议按显存调。

--max_features、--max_len：分别控制 TF-IDF 词表与 BERT 截断。

--use_tool_emb 0/1：是否叠加工具 ID 嵌入的均值作为先验。

Hybrid Two-Tower (TF-IDF + BERT) Agent Recommender with In-Batch InfoNCE
-----------------------------------------------------------------------

What this is
------------
A memory-safe two-tower recommender that fuses **four** textual views:
  1) Question TF‑IDF (q_tfidf)
  2) Question BERT sentence embedding (q_bert)
  3) Agent TF‑IDF (agent text + aggregated tool text) (a_tfidf)
  4) Agent BERT sentence embedding (a_bert)

The two towers project (q_tfidf, q_bert) and (a_tfidf, a_bert) independently
through small MLPs, then **fuse** them with learned gates. We produce final
question/agent representations and compute an **in-batch InfoNCE** loss using a
bilinear + dot-product scorer (parameter-efficient and O(B^2)).

Why this design
---------------
- Keeps your **OOM-safe** pattern: features cached on CPU; only batches
  move to device.
- **In-batch negatives** as in your Two-Tower script, but the logits
  are computed via a trainable scorer instead of a plain dot-product.
- Encodes agent tools two ways:
  (i) their **text** joins agent TF‑IDF & BERT text, and
  (ii) optional **tool-ID embeddings** aggregated per agent (toggleable).

Usage
-----
python simple_twotower_hybrid_tfidf_bert_agent_rec.py \
  --data_root /path/to/dataset \
  --epochs 3 --batch_size 512 --max_features 5000 \
  --pretrained_model distilbert-base-uncased --max_len 128 \
  --eval_chunk 8192 --device cuda --amp 1

Notes
-----
- Caches TF‑IDF matrices, BERT embeddings, tool-id buffers.
- Validation uses sampled evaluation identical to your prior scripts.
- Metrics: P/R/F1/Hit/nDCG/MRR at {5,10,30}.
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
from contextlib import nullcontext

from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer, AutoModel

filename = os.path.splitext(os.path.basename(__file__))[0]
pos_topk = 5

# -------------------- I/O helpers --------------------
def ensure_cache_dir(root: str) -> str:
    d = os.path.join(root, f".cache/{filename}")
    os.makedirs(d, exist_ok=True)
    return d

def load_json(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

# -------------------- data collection --------------------
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
        all_rankings.update(rankings["rankings"])  # {qid: [aid,...]}

    tools = load_json(os.path.join(data_root, "Tools", "merge.json"))
    return all_agents, all_questions, all_rankings, tools

# -------------------- text building --------------------
def build_text_corpora(all_agents, all_questions, tools):
    q_ids = list(all_questions.keys())
    q_texts = [all_questions[qid]["input"] for qid in q_ids]

    tool_names = list(tools.keys())

    def tool_text(tn: str) -> str:
        t = tools.get(tn, {})
        return f"{tn} {t.get('description', '')}".strip()

    a_ids: List[str] = []
    a_texts: List[str] = []
    a_tool_lists: List[List[str]] = []

    for aid, a in all_agents.items():
        mname = a.get("M", {}).get("name", "")
        tlist = a.get("T", {}).get("tools", []) or []
        a_ids.append(aid)
        a_tool_lists.append(tlist)
        concat_tool_desc = " ".join(tool_text(tn) for tn in tlist)
        a_texts.append(f"{mname} {concat_tool_desc}".strip())

    return q_ids, q_texts, tool_names, a_ids, a_texts, a_tool_lists

# -------------------- TF-IDF --------------------
def build_tfidf(q_texts, a_texts, tool_names, a_tool_lists, max_features: int):
    q_vec = TfidfVectorizer(max_features=max_features, lowercase=True)
    a_vec = TfidfVectorizer(max_features=max_features, lowercase=True)
    tool_vec = TfidfVectorizer(max_features=max_features, lowercase=True)

    Q = q_vec.fit_transform(q_texts).toarray().astype(np.float32)
    A_text = a_vec.fit_transform(a_texts).toarray().astype(np.float32)

    # Build tool text matrix and average per agent
    tool_texts = [tn for tn in tool_names]  # vocabulary seed by names
    Tm = tool_vec.fit_transform(tool_texts).toarray().astype(np.float32)
    name2idx = {n: i for i, n in enumerate(tool_names)}
    Dt = Tm.shape[1]
    Atool = np.zeros((len(a_tool_lists), Dt), dtype=np.float32)
    for i, tlist in enumerate(a_tool_lists):
        idxs = [name2idx[t] for t in tlist if t in name2idx]
        if len(idxs) > 0:
            Atool[i] = Tm[idxs].mean(axis=0)

    A = np.concatenate([A_text, Atool], axis=1).astype(np.float32)
    return Q, A

# -------------------- Tool ID buffers (optional ID embeddings) --------------------
def build_agent_tool_id_buffers(a_ids: List[str], agent_tool_lists: List[List[str]], tool_names: List[str]):
    t_map = {n: i for i, n in enumerate(tool_names)}
    num_agents = len(a_ids)
    max_t = max((len(lst) for lst in agent_tool_lists), default=0)
    if max_t == 0:
        max_t = 1
    idx_pad = np.zeros((num_agents, max_t), dtype=np.int64)
    mask = np.zeros((num_agents, max_t), dtype=np.float32)
    for i, lst in enumerate(agent_tool_lists):
        for j, tn in enumerate(lst[:max_t]):
            if tn in t_map:
                idx_pad[i, j] = t_map[tn]
                mask[i, j] = 1.0
    return torch.from_numpy(idx_pad), torch.from_numpy(mask)

# -------------------- BERT sentence encoding (frozen, offline) --------------------
@torch.no_grad()
def encode_texts_bert(texts: List[str], tokenizer, encoder, device, max_len=128, batch_size=256, use_cls=True):
    outs = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding BERT", dynamic_ncols=True):
        batch = texts[i:i + batch_size]
        toks = tokenizer(batch, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
        toks = {k: v.to(device) for k, v in toks.items()}
        out = encoder(**toks)
        if hasattr(out, "last_hidden_state"):
            if use_cls:
                vec = out.last_hidden_state[:, 0, :]
            else:
                attn = toks["attention_mask"].unsqueeze(-1)
                vec = (out.last_hidden_state * attn).sum(1) / (attn.sum(1).clamp(min=1))
        else:
            vec = out.pooler_output
        outs.append(vec.detach().cpu())
    return torch.cat(outs, dim=0).numpy().astype(np.float32)

# -------------------- cache --------------------
def cache_exists(cache_dir: str) -> bool:
    needed = [
        "q_ids.json", "a_ids.json", "tool_names.json",
        "Q_tfidf.npy", "A_tfidf.npy",
        "Q_bert.npy", "A_bert.npy",
        "agent_tool_idx_padded.npy", "agent_tool_mask.npy",
        "enc_meta.json"
    ]
    return all(os.path.exists(os.path.join(cache_dir, f)) for f in needed)


def save_cache(cache_dir: str,
               q_ids, a_ids, tool_names,
               Q_tfidf, A_tfidf, Q_bert, A_bert,
               idx_pad, mask,
               enc_meta):
    with open(os.path.join(cache_dir, "q_ids.json"), "w", encoding="utf-8") as f:
        json.dump(q_ids, f, ensure_ascii=False)
    with open(os.path.join(cache_dir, "a_ids.json"), "w", encoding="utf-8") as f:
        json.dump(a_ids, f, ensure_ascii=False)
    with open(os.path.join(cache_dir, "tool_names.json"), "w", encoding="utf-8") as f:
        json.dump(tool_names, f, ensure_ascii=False)
    with open(os.path.join(cache_dir, "enc_meta.json"), "w", encoding="utf-8") as f:
        json.dump(enc_meta, f, ensure_ascii=False)

    np.save(os.path.join(cache_dir, "Q_tfidf.npy"), Q_tfidf.astype(np.float32))
    np.save(os.path.join(cache_dir, "A_tfidf.npy"), A_tfidf.astype(np.float32))
    np.save(os.path.join(cache_dir, "Q_bert.npy"), Q_bert.astype(np.float32))
    np.save(os.path.join(cache_dir, "A_bert.npy"), A_bert.astype(np.float32))
    np.save(os.path.join(cache_dir, "agent_tool_idx_padded.npy"), idx_pad.astype(np.int64))
    np.save(os.path.join(cache_dir, "agent_tool_mask.npy"), mask.astype(np.float32))


def load_cache(cache_dir: str):
    with open(os.path.join(cache_dir, "q_ids.json"), "r", encoding="utf-8") as f:
        q_ids = json.load(f)
    with open(os.path.join(cache_dir, "a_ids.json"), "r", encoding="utf-8") as f:
        a_ids = json.load(f)
    with open(os.path.join(cache_dir, "tool_names.json"), "r", encoding="utf-8") as f:
        tool_names = json.load(f)
    with open(os.path.join(cache_dir, "enc_meta.json"), "r", encoding="utf-8") as f:
        enc_meta = json.load(f)

    Q_tfidf = np.load(os.path.join(cache_dir, "Q_tfidf.npy"))
    A_tfidf = np.load(os.path.join(cache_dir, "A_tfidf.npy"))
    Q_bert = np.load(os.path.join(cache_dir, "Q_bert.npy"))
    A_bert = np.load(os.path.join(cache_dir, "A_bert.npy"))
    idx_pad = np.load(os.path.join(cache_dir, "agent_tool_idx_padded.npy"))
    mask = np.load(os.path.join(cache_dir, "agent_tool_mask.npy"))

    return q_ids, a_ids, tool_names, Q_tfidf, A_tfidf, Q_bert, A_bert, idx_pad, mask, enc_meta

# -------------------- splits & metrics --------------------
def train_valid_split(qids_in_rankings, valid_ratio=0.2, seed=42):
    rng = random.Random(seed)
    q = list(qids_in_rankings)
    rng.shuffle(q)
    n_valid = int(len(q) * valid_ratio)
    return q[n_valid:], q[:n_valid]


def _ideal_dcg(k, num_rel):
    ideal = min(k, num_rel)
    return sum(1.0 / math.log2(i + 2.0) for i in range(ideal)) if ideal else 0.0


@torch.no_grad()
def evaluate_model(encoder, Q_tf_cpu, Q_bt_cpu, A_tf_cpu, A_bt_cpu,
                   qid2idx, a_ids, all_rankings, eval_qids,
                   device="cpu", ks=(5, 10, 50), cand_size=1000, rng_seed=123, amp=False, chunk=8192):
    max_k = max(ks)
    aid2idx = {aid: i for i, aid in enumerate(a_ids)}
    agg = {k: {m: 0.0 for m in ["P", "R", "F1", "Hit", "nDCG", "MRR"]} for k in ks}
    cnt = 0
    skipped = 0
    ref_k = 10 if 10 in ks else max_k
    all_agent_set = set(a_ids)

    pbar = tqdm(eval_qids, desc="Evaluating (sampled)", dynamic_ncols=True)
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
        cand_ids = gt_list + (rnd.sample(neg_pool, min(need_neg, len(neg_pool))) if need_neg > 0 and neg_pool else [])

        qi = qid2idx[qid]
        q_tf = torch.from_numpy(Q_tf_cpu[qi:qi + 1]).to(device)
        q_bt = torch.from_numpy(Q_bt_cpu[qi:qi + 1]).to(device)

        # Encode question once
        qe = encoder.encode_q(q_tf, q_bt)

        # Chunked agent scoring
        best_scores = []
        best_ids = []
        cand_idx = [aid2idx[a] for a in cand_ids]
        for i in range(0, len(cand_idx), chunk):
            sub_idx_list = cand_idx[i:i + chunk]
            a_tf = torch.from_numpy(A_tf_cpu[sub_idx_list]).to(device)
            a_bt = torch.from_numpy(A_bt_cpu[sub_idx_list]).to(device)
            a_ids_t = torch.tensor(sub_idx_list, dtype=torch.long, device=device)
            ae = encoder.encode_a(a_tf, a_bt, a_ids_t)
            scores = encoder.pairwise_scores(qe, ae)  # (1, Bsub)
            k = min(len(sub_idx_list), max_k)
            top_scores, top_local = torch.topk(scores.squeeze(0), k)
            best_scores.extend(top_scores.detach().cpu().tolist())
            best_ids.extend([sub_idx_list[int(x)] for x in top_local.detach().cpu().tolist()])

        # Final top-k across chunks
        if len(best_scores) == 0:
            continue
        best_scores = torch.tensor(best_scores)
        best_ids = torch.tensor(best_ids)
        k = min(max_k, best_scores.numel())
        final_scores, final_idx = torch.topk(best_scores, k)
        pred_ids = [a_ids[int(best_ids[int(idx)])] for idx in final_idx]

        bin_hits = [1 if aid in rel_set else 0 for aid in pred_ids]
        num_rel = len(rel_set)
        for k in ks:
            top = bin_hits[:k]
            Hk = sum(top)
            P = Hk / float(k)
            R = Hk / float(num_rel)
            F1 = (2 * P * R) / (P + R) if (P + R) else 0.0
            Hit = 1.0 if Hk > 0 else 0.0
            dcg = sum(1.0 / math.log2(i + 2.0) for i, h in enumerate(top) if h)
            nDCG = (dcg / _ideal_dcg(k, num_rel)) if num_rel > 0 else 0.0
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
            f"P@{ref_k}": f"{(ref['P'] / max(1, cnt)):.4f}",
            f"nDCG@{ref_k}": f"{(ref['nDCG'] / max(1, cnt)):.4f}",
            f"MRR@{ref_k}": f"{(ref['MRR'] / max(1, cnt)):.4f}",
        })

    if cnt == 0:
        return {k: {m: 0.0 for m in ["P", "R", "F1", "Hit", "nDCG", "MRR"]} for k in ks}
    for k in ks:
        for m in agg[k]:
            agg[k][m] /= cnt
    return agg


def print_metrics_table(title, metrics_dict, ks=(5, 10, 50)):
    print(f"\n== {title} ==")
    header = f"{'@K':>4} | {'P':>7} {'R':>7} {'F1':>7} {'Hit':>7} {'nDCG':>7} {'MRR':>7}"
    print(header)
    print("-" * len(header))
    for k in ks:
        m = metrics_dict[k]
        print(f"{k:>4} | {m['P']:.4f} {m['R']:.4f} {m['F1']:.4f} {m['Hit']:.4f} {m['nDCG']:.4f} {m['MRR']:.4f}")

# -------------------- training pairs (in-batch negatives) --------------------
def build_training_pairs(all_rankings: Dict[str, List[str]], rng_seed=42):
    """
    Create positive pairs only; in-batch negatives are implicit in InfoNCE.
    """
    rnd = random.Random(rng_seed)
    pairs = []
    for qid, ranked in all_rankings.items():
        pos_list = ranked[:pos_topk] if ranked else []
        for pos_a in pos_list:
            pairs.append((qid, pos_a))
    rnd.shuffle(pairs)
    return pairs

# -------------------- model --------------------
class HybridTwoTower(nn.Module):
    def __init__(self, d_q_tf, d_q_bt, d_a_tf, d_a_bt,
                 num_tools: int,
                 agent_tool_idx_pad: torch.LongTensor,
                 agent_tool_mask: torch.FloatTensor,
                 hid: int = 256,
                 use_tool_emb: bool = True,
                 gate_init: float = 0.5):
        super().__init__()

        # TF‑IDF branches
        self.q_tf_mlp = nn.Sequential(nn.Linear(d_q_tf, hid), nn.ReLU(), nn.Linear(hid, hid))
        self.a_tf_mlp = nn.Sequential(nn.Linear(d_a_tf, hid), nn.ReLU(), nn.Linear(hid, hid))

        # BERT branches
        self.q_bt_mlp = nn.Sequential(nn.Linear(d_q_bt, hid), nn.ReLU(), nn.Linear(hid, hid))
        self.a_bt_mlp = nn.Sequential(nn.Linear(d_a_bt, hid), nn.ReLU(), nn.Linear(hid, hid))

        # Gated fusion per side
        self.q_gate = nn.Parameter(torch.tensor([gate_init], dtype=torch.float32))  # weight on BERT
        self.a_gate = nn.Parameter(torch.tensor([gate_init], dtype=torch.float32))

        # Optional tool-ID embeddings (averaged per agent)
        self.use_tool_emb = use_tool_emb and (num_tools > 0)
        if self.use_tool_emb:
            self.emb_tool = nn.Embedding(num_tools, hid)
            nn.init.xavier_uniform_(self.emb_tool.weight)
        else:
            self.emb_tool = None

        # Register tool buffers for aggregation
        self.register_buffer("tool_idx", agent_tool_idx_pad.long())
        self.register_buffer("tool_mask", agent_tool_mask.float())

        # Bilinear + dot scorer (trainable). Produces (B, B) logits.
        self.bilinear = nn.Parameter(torch.empty(hid, hid))
        nn.init.xavier_uniform_(self.bilinear)
        self.logit_scale = nn.Parameter(torch.tensor(1.0))

        # Init linear layers
        for mlp in [self.q_tf_mlp, self.a_tf_mlp, self.q_bt_mlp, self.a_bt_mlp]:
            for m in mlp:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.zeros_(m.bias)

    def tool_agg(self, agent_idx):
        if not self.use_tool_emb:
            return 0.0
        te = self.emb_tool(self.tool_idx[agent_idx])  # (B, T, H)
        m = self.tool_mask[agent_idx].unsqueeze(-1)   # (B, T, 1)
        return (te * m).sum(1) / (m.sum(1) + 1e-8)    # (B, H)

    def encode_q(self, q_tf, q_bt):
        qt = self.q_tf_mlp(q_tf)
        qb = self.q_bt_mlp(q_bt)
        g = torch.sigmoid(self.q_gate)
        qh = F.normalize((1 - g) * qt + g * qb, dim=-1)
        return qh  # (B, H)

    def encode_a(self, a_tf, a_bt, agent_idx):
        at = self.a_tf_mlp(a_tf)
        ab = self.a_bt_mlp(a_bt)
        g = torch.sigmoid(self.a_gate)
        ah = (1 - g) * at + g * ab
        if self.use_tool_emb:
            ah = ah + 0.5 * self.tool_agg(agent_idx)
        ah = F.normalize(ah, dim=-1)
        return ah  # (B, H)

    def pairwise_scores(self, qh, ah):
        """Compute logits for all pairs in a batch: (Bq, H) x (Ba, H) -> (Bq, Ba).
        A light-weight DNN-style scorer: dot + bilinear, scaled.
        """
        # dot-product channel
        dot = qh @ ah.t()
        # bilinear channel
        bil = (qh @ self.bilinear) @ ah.t()
        return self.logit_scale * (dot + bil)

# -------------------- loss --------------------
def info_nce_from_logits(logits: torch.Tensor):
    """Cross-entropy with labels = diagonal. logits: (B, B)."""
    labels = torch.arange(logits.size(0), device=logits.device)
    return F.cross_entropy(logits, labels)

# -------------------- model save paths --------------------
def model_save_paths(cache_dir: str, data_sig: str):
    model_dir = os.path.join(cache_dir, "models")
    os.makedirs(model_dir, exist_ok=True)
    return (
        os.path.join(model_dir, f"{filename}_{data_sig}.pt"),
        os.path.join(model_dir, f"latest_{data_sig}.pt"),
        os.path.join(model_dir, f"meta_{data_sig}.json"),
    )

# -------------------- dataset signature --------------------
def dataset_signature(a_ids, all_rankings):
    payload = {"a_ids": a_ids, "rankings": {k: all_rankings[k] for k in sorted(all_rankings.keys())}}
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return f"{(zlib.crc32(blob) & 0xFFFFFFFF):08x}"

# -------------------- training cache --------------------
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


def save_training_cache(cache_dir, train_qids, valid_qids, pairs_idx_np, meta):
    p_train, p_valid, p_pairs, p_meta = training_cache_paths(cache_dir)
    with open(p_train, "w", encoding="utf-8") as f:
        json.dump(train_qids, f, ensure_ascii=False)
    with open(p_valid, "w", encoding="utf-8") as f:
        json.dump(valid_qids, f, ensure_ascii=False)
    np.save(p_pairs, pairs_idx_np.astype(np.int64))
    with open(p_meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, sort_keys=True)


def load_training_cache(cache_dir):
    p_train, p_valid, p_pairs, p_meta = training_cache_paths(cache_dir)
    with open(p_train, "r", encoding="utf-8") as f:
        train_qids = json.load(f)
    with open(p_valid, "r", encoding="utf-8") as f:
        valid_qids = json.load(f)
    pairs_idx_np = np.load(p_pairs)
    with open(p_meta, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return train_qids, valid_qids, pairs_idx_np, meta

# -------------------- main --------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--max_features", type=int, default=5000)
    parser.add_argument("--pretrained_model", type=str, default="distilbert-base-uncased")
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--hid", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--rng_seed_pairs", type=int, default=42)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--valid_ratio", type=float, default=0.2)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--eval_chunk", type=int, default=8192)
    parser.add_argument("--amp", type=int, default=0)
    parser.add_argument("--rebuild_cache", type=int, default=0, help="1 to recompute TFIDF/BERT cache")
    parser.add_argument("--use_tool_emb", type=int, default=1)
    args = parser.parse_args()

    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)

    device = torch.device(args.device)

    # 1) Load data
    all_agents, all_questions, all_rankings, tools = collect_data(args.data_root)
    print(f"Loaded {len(all_agents)} agents, {len(all_questions)} questions, {len(tools)} tools.")

    cache_dir = ensure_cache_dir(args.data_root)

    # 2) Build corpora and tool buffers
    q_ids, q_texts, tool_names, a_ids, a_texts, a_tool_lists = build_text_corpora(all_agents, all_questions, tools)
    idx_pad, mask = build_agent_tool_id_buffers(a_ids, a_tool_lists, tool_names)

    # 3) Cache features (TF‑IDF + BERT)
    enc_meta = None
    if cache_exists(cache_dir) and args.rebuild_cache == 0:
        (q_ids_c, a_ids_c, tool_names_c,
         Q_tf, A_tf, Q_bt, A_bt,
         idx_pad_np, mask_np, enc_meta) = load_cache(cache_dir)
        if q_ids_c == q_ids and a_ids_c == a_ids and tool_names_c == tool_names \
           and enc_meta.get("pretrained_model") == args.pretrained_model \
           and enc_meta.get("max_len") == args.max_len \
           and enc_meta.get("max_features") == args.max_features:
            print(f"[cache] loaded features from {cache_dir}")
            idx_pad = torch.from_numpy(idx_pad_np)
            mask = torch.from_numpy(mask_np)
        else:
            print("[cache] mismatch; rebuilding features...")
            Q_tf = A_tf = Q_bt = A_bt = None
    else:
        Q_tf = A_tf = Q_bt = A_bt = None

    if any(x is None for x in [Q_tf, A_tf, Q_bt, A_bt]):
        # TF‑IDF
        Q_tf, A_tf = build_tfidf(q_texts, a_texts, tool_names, a_tool_lists, args.max_features)
        # BERT
        tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)
        encoder = AutoModel.from_pretrained(args.pretrained_model).to(device).eval()
        Q_bt = encode_texts_bert(q_texts, tokenizer, encoder, device, max_len=args.max_len, batch_size=256, use_cls=True)
        A_bt = encode_texts_bert(a_texts, tokenizer, encoder, device, max_len=args.max_len, batch_size=256, use_cls=True)

        save_cache(
            cache_dir,
            q_ids, a_ids, tool_names,
            Q_tf, A_tf, Q_bt, A_bt,
            idx_pad.numpy(), mask.numpy(),
            enc_meta={"pretrained_model": args.pretrained_model, "max_len": args.max_len, "max_features": args.max_features}
        )
        print(f"[cache] saved features to {cache_dir}")

    # 4) Mappings & signature
    qid2idx = {qid: i for i, qid in enumerate(q_ids)}
    aid2idx = {aid: i for i, aid in enumerate(a_ids)}

    data_sig = dataset_signature(a_ids, all_rankings)
    model_path, latest_model, meta_path = model_save_paths(cache_dir, data_sig)

    # 5) Train/valid pairs (positive only)
    want_meta = {
        "data_sig": data_sig,
        "rng_seed_pairs": int(args.rng_seed_pairs),
        "split_seed": int(args.split_seed),
        "valid_ratio": float(args.valid_ratio),
        "pair_type": "q_pos_only_posTopK"
    }

    use_cache = training_cache_exists(cache_dir)
    if use_cache:
        train_qids, valid_qids, pairs_idx_np, meta = load_training_cache(cache_dir)
        if meta != want_meta:
            use_cache = False
    if not use_cache:
        qids_in_rank = [qid for qid in q_ids if qid in all_rankings]
        train_qids, valid_qids = train_valid_split(qids_in_rank, valid_ratio=args.valid_ratio, seed=args.split_seed)
        pairs = build_training_pairs({qid: all_rankings[qid] for qid in train_qids}, rng_seed=args.rng_seed_pairs)
        pairs_idx_np = np.array([(qid2idx[q], aid2idx[a]) for (q, a) in pairs], dtype=np.int64)
        save_training_cache(cache_dir, train_qids, valid_qids, pairs_idx_np, want_meta)
        print(f"[cache] saved train/valid/pairs to {cache_dir} (sig={data_sig})")

    # 6) Model
    d_q_tf = int(Q_tf.shape[1])
    d_a_tf = int(A_tf.shape[1])
    d_q_bt = int(Q_bt.shape[1])
    d_a_bt = int(A_bt.shape[1])

    encoder = HybridTwoTower(
        d_q_tf=d_q_tf, d_q_bt=d_q_bt,
        d_a_tf=d_a_tf, d_a_bt=d_a_bt,
        num_tools=len(tool_names),
        agent_tool_idx_pad=idx_pad.to(device),
        agent_tool_mask=mask.to(device),
        hid=int(args.hid),
        use_tool_emb=(args.use_tool_emb == 1),
    ).to(device)

    optimizer = torch.optim.Adam(encoder.parameters(), lr=args.lr)

    # Keep features on CPU; move batch slices to device
    Q_tf_cpu = Q_tf
    Q_bt_cpu = Q_bt
    A_tf_cpu = A_tf
    A_bt_cpu = A_bt

    # 7) Train with in-batch InfoNCE over pairwise logits
    num_pairs = pairs_idx_np.shape[0]
    num_batches = (num_pairs + args.batch_size - 1) // args.batch_size
    print(f"Training pairs: {num_pairs}, batches/epoch: {num_batches}")

    use_amp = (args.amp == 1 and device.type == 'cuda')

    for epoch in range(1, args.epochs + 1):
        # shuffle in-place
        np.random.shuffle(pairs_idx_np)
        total = 0.0
        pbar = tqdm(range(num_batches), desc=f"Epoch {epoch}/{args.epochs}", dynamic_ncols=True)
        for b in pbar:
            sl = slice(b * args.batch_size, min((b + 1) * args.batch_size, num_pairs))
            batch = pairs_idx_np[sl]
            if batch.size == 0:
                continue

            q_idx = torch.from_numpy(batch[:, 0]).long()
            a_idx = torch.from_numpy(batch[:, 1]).long()

            q_tf = torch.from_numpy(Q_tf_cpu[q_idx]).to(device, non_blocking=True)
            q_bt = torch.from_numpy(Q_bt_cpu[q_idx]).to(device, non_blocking=True)
            a_tf = torch.from_numpy(A_tf_cpu[a_idx]).to(device, non_blocking=True)
            a_bt = torch.from_numpy(A_bt_cpu[a_idx]).to(device, non_blocking=True)
            a_ids_t = a_idx.to(device)

            if use_amp:
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    qe = encoder.encode_q(q_tf, q_bt)
                    ae = encoder.encode_a(a_tf, a_bt, a_ids_t)
                    logits = encoder.pairwise_scores(qe, ae)  # (B, B)
                    loss = info_nce_from_logits(logits)
            else:
                qe = encoder.encode_q(q_tf, q_bt)
                ae = encoder.encode_a(a_tf, a_bt, a_ids_t)
                logits = encoder.pairwise_scores(qe, ae)  # (B, B)
                loss = info_nce_from_logits(logits)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total += float(loss.item())
            pbar.set_postfix({"batch_loss": f"{loss.item():.4f}", "avg_loss": f"{(total / (b + 1)):.4f}"})

        print(f"Epoch {epoch}/{args.epochs} - InfoNCE: {(total / max(1, num_batches)):.4f}")

    # 8) Save
    ckpt = {
        "state_dict": encoder.state_dict(),
        "data_sig": data_sig,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "args": vars(args),
        "dims": {
            "d_q_tf": int(d_q_tf),
            "d_q_bt": int(d_q_bt),
            "d_a_tf": int(d_a_tf),
            "d_a_bt": int(d_a_bt),
            "hid": int(args.hid),
            "num_tools": int(len(tool_names)),
        },
        "mappings": {
            "q_ids": q_ids,
            "a_ids": a_ids,
            "tool_names": tool_names,
        },
    }
    torch.save(ckpt, model_path)
    torch.save(ckpt, latest_model)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({"data_sig": data_sig, "a_ids": a_ids, "tool_names": tool_names}, f, ensure_ascii=False, indent=2)
    print(f"[save] model -> {model_path}")
    print(f"[save] meta  -> {meta_path}")

    # 9) Validation
    valid_metrics = evaluate_model(
        encoder,
        Q_tf_cpu, Q_bt_cpu, A_tf_cpu, A_bt_cpu,
        qid2idx, a_ids, all_rankings,
        valid_qids, device=device, ks=(5, 10, 50), cand_size=1000, rng_seed=123, amp=use_amp, chunk=args.eval_chunk
    )
    print_metrics_table("Validation (averaged over questions)", valid_metrics, ks=(5, 10, 50))

    # 10) Inference (chunked)
    @torch.no_grad()
    def recommend_topk_for_qid(qid: str, topk: int = 10, chunk: int = 8192):
        qi = qid2idx[qid]
        q_tf = torch.from_numpy(Q_tf_cpu[qi:qi + 1]).to(device)
        q_bt = torch.from_numpy(Q_bt_cpu[qi:qi + 1]).to(device)
        qe = encoder.encode_q(q_tf, q_bt)  # (1, H)

        best_scores = []
        best_ids_local = []
        N = len(a_ids)
        for i in range(0, N, chunk):
            j = min(i + chunk, N)
            a_tf = torch.from_numpy(A_tf_cpu[i:j]).to(device)
            a_bt = torch.from_numpy(A_bt_cpu[i:j]).to(device)
            a_idx = torch.arange(i, j, dtype=torch.long, device=device)
            ae = encoder.encode_a(a_tf, a_bt, a_idx)
            scores = encoder.pairwise_scores(qe, ae).squeeze(0)
            k = min(topk, j - i)
            top_scores, top_local = torch.topk(scores, k)
            best_scores.extend(top_scores.detach().cpu().tolist())
            best_ids_local.extend([i + int(t) for t in top_local.detach().cpu().tolist()])

        if len(best_scores) == 0:
            return []
        best_scores = torch.tensor(best_scores)
        best_ids_local = torch.tensor(best_ids_local)
        k = min(topk, best_scores.numel())
        final_scores, final_idx = torch.topk(best_scores, k)
        return [(a_ids[int(best_ids_local[int(idx)])], float(final_scores[n].item())) for n, idx in enumerate(final_idx)]

    sample_qids = q_ids[:min(5, len(q_ids))]
    for qid in sample_qids:
        recs = recommend_topk_for_qid(qid, topk=args.topk, chunk=args.eval_chunk)
        qtext = all_questions[qid]["input"][:80].replace("\n", " ")
        print(f"\nQuestion: {qid}  |  {qtext}")
        for r, (aid, s) in enumerate(recs, 1):
            print(f"  {r:2d}. {aid:>20s}  score={s:.4f}")


if __name__ == "__main__":
    main()
