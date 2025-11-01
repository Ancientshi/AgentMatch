#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Two-Tower TF-IDF Agent Recommender (InfoNCE) — OOM-safe version.

Key changes vs the previous script:
1) Keep TF-IDF tensors on CPU; move ONLY the current batch to GPU.
2) Evaluation & inference are chunked over agents (no full-matrix move to GPU).
3) Added --eval_chunk to control agent-encoding batch size (default 8192).
4) Added --amp to optionally enable autocast(float16/bfloat16) on GPU.

Usage example:
  python simple_twotower_tfidf_agent_rec_oomfix.py --data_root /path/to/dataset \
    --epochs 3 --batch_size 512 --max_features 5000 --eval_chunk 8192 --device cuda --amp 1
    
    
Query（问题）塔

用到的特征：只用问题文本的 TF-IDF 向量（Q，由 TfidfVectorizer 拟合在 q_texts 上得到）。

不用的特征：不使用 query ID、不使用任何 tool 信息。

处理方式：q_vec → q_proj(MLP) → L2 normalize 得到查询嵌入。

Agent（智能体）塔

用到的文本特征（两部分拼接）：

Agent 文本 TF-IDF（Am，在 a_texts 上拟合，内容是 model_name + 所有工具的文本串联）；

Agent 的工具文本均值 TF-IDF（Atool，把该 agent 关联的工具在 tool_texts 上的 TF-IDF 向量取均值）。
最终把两部分 concatenate 成 A = concat[Am, Atool] 送入 agent 塔。

用到的 ID 特征（可选）：Tool-ID embedding（nn.Embedding(num_tools, H)），对该 agent 关联的工具 ID 做掩码均值聚合，并以 0.5 的权重加到 agent 文本投影上：
ah = a_proj(a_vec) + 0.5 * mean(emb_tool[tool_ids_of_agent])（代码路径：encode_a() → tool_agg()）。

不用的特征：没有单独的 Agent-ID embedding。

处理方式：a_vec → a_proj(MLP) (+ 0.5*tool_id_agg) → L2 normalize 得到 agent 嵌入。

Tool（工具）特征来自两条路径

Tool 文本 TF-IDF：在 tool_texts（"tool_name + description"）上拟合得到矩阵 Tm，随后在构造 agent 向量时对该 agent 依赖的工具对应行做均值，形成 Atool 并与 Am 拼接进 agent 塔输入。

Tool-ID embedding：训练时作为结构化 ID 特征参与 agent 塔（可开关 --use_tool_emb 1/0），按 agent 的工具列表做掩码均值聚合后加到 agent 表征中。

其它要点

两塔输出均做 L2 归一化，相似度用内积；

训练损失是 InfoNCE，负样本来自 in-batch negatives；

全流程不使用 query/agent 的独立 ID 向量，工具 ID 是唯一的 ID 信号并只注入到 agent 塔。
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
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm.auto import tqdm
from utils import print_metrics_table

filename = os.path.splitext(os.path.basename(__file__))[0]
pos_topk = 5

# ---------------- paths ----------------
def model_save_paths(cache_dir: str, data_sig: str):
    model_dir = os.path.join(cache_dir, "models")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"{filename}_{data_sig}.pt")
    qvec_path  = os.path.join(model_dir, f"q_vectorizer_{data_sig}.pkl")
    latest_model = os.path.join(model_dir, f"latest_{data_sig}.pt")
    latest_qvec  = os.path.join(model_dir, f"latest_qvec_{data_sig}.pkl")
    meta_path    = os.path.join(model_dir, f"meta_{data_sig}.json")
    return model_path, qvec_path, latest_model, latest_qvec, meta_path

# ---------------- split & metrics ----------------
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
def evaluate_model(encoder, Q_cpu, A_cpu, qid2idx, a_ids, all_rankings,
                   eval_qids, device="cpu", ks=(5,10,50),
                   cand_size=1000, rng_seed=123, amp=False):
    max_k = max(ks)
    aid2idx = {aid: i for i, aid in enumerate(a_ids)}
    agg = {k: {"P":0.0,"R":0.0,"F1":0.0,"Hit":0.0,"nDCG":0.0,"MRR":0.0} for k in ks}
    cnt = 0; skipped = 0
    ref_k = 10 if 10 in ks else max_k
    all_agent_set = set(a_ids)

    scaler = torch.autocast(device_type='cuda', dtype=torch.bfloat16) if (amp and device.type=='cuda') else nullcontext()

    pbar = tqdm(eval_qids, desc="Evaluating (sampled)", leave=True, dynamic_ncols=True)
    for qid in pbar:
        gt_list = [aid for aid in all_rankings.get(qid, [])[:pos_topk] if aid in aid2idx]
        if not gt_list:
            skipped += 1; pbar.set_postfix({"done": cnt, "skipped": skipped}); continue
        rel_set = set(gt_list)
        neg_pool = list(all_agent_set - rel_set)

        rnd = random.Random((hash(qid) ^ (rng_seed * 16777619)) & 0xFFFFFFFF)
        need_neg = max(0, cand_size - len(gt_list))
        cand_ids = gt_list + (rnd.sample(neg_pool, min(need_neg, len(neg_pool))) if need_neg>0 and neg_pool else [])

        qi = qid2idx[qid]
        qv = torch.from_numpy(Q_cpu[qi:qi+1]).to(device)      # (1, d_q)
        cand_idx = [aid2idx[a] for a in cand_ids]
        av = torch.from_numpy(A_cpu[cand_idx]).to(device)     # (Nc, d_a)

        with (torch.autocast(device_type='cuda', dtype=torch.bfloat16) if (amp and device.type=='cuda') else nullcontext()):
            qe = encoder.encode_q(qv)                         # (1, H)
            ae = encoder.encode_a(av, torch.tensor(cand_idx, device=device, dtype=torch.long))  # (Nc, H)
            scores = (qe @ ae.t()).float().squeeze(0).cpu().numpy()

        order = np.argsort(-scores)[:max_k]
        pred_ids = [cand_ids[i] for i in order]

        bin_hits = [1 if aid in rel_set else 0 for aid in pred_ids]
        num_rel = len(rel_set)
        for k in ks:
            top = bin_hits[:k]
            Hk = sum(top); P = Hk/float(k); R = Hk/float(num_rel); F1 = (2*P*R)/(P+R) if (P+R) else 0.0
            Hit = 1.0 if Hk>0 else 0.0
            dcg = sum(1.0 / math.log2(i + 2.0) for i,h in enumerate(top) if h)
            nDCG = (dcg/_ideal_dcg(k, num_rel)) if num_rel>0 else 0.0
            rr = 0.0
            for i,h in enumerate(top):
                if h: rr = 1.0/float(i+1); break
            agg[k]["P"]+=P; agg[k]["R"]+=R; agg[k]["F1"]+=F1; agg[k]["Hit"]+=Hit; agg[k]["nDCG"]+=nDCG; agg[k]["MRR"]+=rr

        cnt += 1
        ref = agg[ref_k]
        pbar.set_postfix({"done":cnt,"skipped":skipped,f"P@{ref_k}":f"{(ref['P']/cnt):.4f}",f"nDCG@{ref_k}":f"{(ref['nDCG']/cnt):.4f}",f"MRR@{ref_k}":f"{(ref['MRR']/cnt):.4f}","Ncand":len(cand_ids)})

    if cnt==0:
        return {k:{m:0.0 for m in ["P","R","F1","Hit","nDCG","MRR"]} for k in ks}
    for k in ks:
        for m in agg[k]: agg[k][m]/=cnt
    return agg

# -------------- cache/io --------------
def ensure_cache_dir(root: str) -> str:
    d = os.path.join(root, f".cache/{filename}")
    os.makedirs(d, exist_ok=True); return d

def cache_exists(cache_dir: str) -> bool:
    needed = ["q_ids.json","a_ids.json","tool_names.json","Q.npy","A_text_full.npy","agent_tool_idx_padded.npy","agent_tool_mask.npy"]
    return all(os.path.exists(os.path.join(cache_dir, f)) for f in needed)

def save_cache(cache_dir, q_ids,a_ids,tool_names, Q,A_text_full, agent_tool_idx_padded,agent_tool_mask):
    with open(os.path.join(cache_dir,"q_ids.json"),"w",encoding="utf-8") as f: json.dump(q_ids,f,ensure_ascii=False)
    with open(os.path.join(cache_dir,"a_ids.json"),"w",encoding="utf-8") as f: json.dump(a_ids,f,ensure_ascii=False)
    with open(os.path.join(cache_dir,"tool_names.json"),"w",encoding="utf-8") as f: json.dump(tool_names,f,ensure_ascii=False)
    np.save(os.path.join(cache_dir,"Q.npy"), Q.astype(np.float32))
    np.save(os.path.join(cache_dir,"A_text_full.npy"), A_text_full.astype(np.float32))
    np.save(os.path.join(cache_dir,"agent_tool_idx_padded.npy"), agent_tool_idx_padded.astype(np.int64))
    np.save(os.path.join(cache_dir,"agent_tool_mask.npy"), agent_tool_mask.astype(np.float32))

def load_cache(cache_dir):
    with open(os.path.join(cache_dir,"q_ids.json"),"r",encoding="utf-8") as f: q_ids=json.load(f)
    with open(os.path.join(cache_dir,"a_ids.json"),"r",encoding="utf-8") as f: a_ids=json.load(f)
    with open(os.path.join(cache_dir,"tool_names.json"),"r",encoding="utf-8") as f: tool_names=json.load(f)
    Q = np.load(os.path.join(cache_dir,"Q.npy"))
    A = np.load(os.path.join(cache_dir,"A_text_full.npy"))
    idx_pad = np.load(os.path.join(cache_dir,"agent_tool_idx_padded.npy"))
    mask = np.load(os.path.join(cache_dir,"agent_tool_mask.npy"))
    return q_ids,a_ids,tool_names,Q,A,idx_pad,mask

# -------------- data utils --------------
def load_json(p): 
    with open(p,"r",encoding="utf-8") as f: return json.load(f)

def collect_data(data_root: str):
    parts = ["PartI", "PartII", "PartIII"]
    all_agents: Dict[str, dict] = {}
    all_questions: Dict[str, dict] = {}
    all_rankings: Dict[str, List[str]] = {}

    for part in parts:
        agents_path = os.path.join(data_root, part, "agents", "merge.json")
        questions_path = os.path.join(data_root, part, "questions", "merge.json")
        rankings_path = os.path.join(data_root, part, "rankings", "merge.json")
        agents = load_json(agents_path); questions = load_json(questions_path); rankings = load_json(rankings_path)
        all_agents.update(agents); all_questions.update(questions); all_rankings.update(rankings["rankings"])

    tools_path = os.path.join(data_root, "Tools", "merge.json")
    tools = load_json(tools_path)
    return all_agents, all_questions, all_rankings, tools

def build_text_corpora(all_agents, all_questions, tools):
    q_ids = list(all_questions.keys())
    q_texts = [all_questions[qid]["input"] for qid in q_ids]

    tool_names = list(tools.keys())
    def _tool_text(tn: str) -> str:
        t = tools.get(tn, {}); return f"{tn} {t.get('description','')}".strip()
    tool_texts = [_tool_text(tn) for tn in tool_names]

    a_ids = list(all_agents.keys())
    a_texts = []; a_tool_lists = []
    for aid in a_ids:
        a = all_agents[aid]
        mname = a.get("M",{}).get("name","")
        tool_list = a.get("T",{}).get("tools",[]) or []
        a_tool_lists.append(tool_list)
        concat_tool_desc = " ".join([_tool_text(tn) for tn in tool_list])
        a_texts.append(f"{mname} {concat_tool_desc}".strip())
    return q_ids, q_texts, tool_names, tool_texts, a_ids, a_texts, a_tool_lists

def build_vectorizers(q_texts, tool_texts, a_texts, max_features: int):
    q_vec = TfidfVectorizer(max_features=max_features, lowercase=True)
    tool_vec = TfidfVectorizer(max_features=max_features, lowercase=True)
    a_vec = TfidfVectorizer(max_features=max_features, lowercase=True)
    Q = q_vec.fit_transform(q_texts)
    Tm = tool_vec.fit_transform(tool_texts)
    Am = a_vec.fit_transform(a_texts)
    return q_vec, tool_vec, a_vec, Q, Tm, Am

def agent_tool_text_matrix(agent_tool_lists, tool_names, Tm):
    name2idx = {n:i for i,n in enumerate(tool_names)}
    Dt = Tm.shape[1]; num_agents = len(agent_tool_lists)
    Atool = np.zeros((num_agents, Dt), dtype=np.float32)
    for i, tool_list in enumerate(agent_tool_lists):
        idxs = [name2idx[t] for t in tool_list if t in name2idx]
        if len(idxs)==0: continue
        vecs = Tm[idxs].toarray()
        Atool[i] = vecs.mean(axis=0).astype(np.float32)
    return Atool

# -------------- Two-Tower --------------
class TwoTower(nn.Module):
    def __init__(self, d_q, d_a, num_tools, agent_tool_idx_pad, agent_tool_mask,
                 hid=256, tool_emb=True,
                 num_agents=0, use_agent_id_emb=False, agent_id_weight=0.5):
        super().__init__()
        self.q_proj = nn.Sequential(nn.Linear(d_q, hid), nn.ReLU(), nn.Linear(hid, hid))
        self.a_proj = nn.Sequential(nn.Linear(d_a, hid), nn.ReLU(), nn.Linear(hid, hid))

        # ---- tool-ID embedding (已有) ----
        self.use_tool_emb = tool_emb and (num_tools > 0)
        if self.use_tool_emb:
            self.emb_tool = nn.Embedding(num_tools, hid)
            nn.init.xavier_uniform_(self.emb_tool.weight)
        else:
            self.emb_tool = None

        # ---- agent-ID embedding（新增，可选）----
        self.use_agent_id_emb = bool(use_agent_id_emb) and (num_agents > 0)
        self.agent_id_weight = float(agent_id_weight)
        if self.use_agent_id_emb:
            self.emb_agent = nn.Embedding(num_agents, hid)
            nn.init.xavier_uniform_(self.emb_agent.weight)
        else:
            self.emb_agent = None

        self.register_buffer("tool_idx", agent_tool_idx_pad.long())
        self.register_buffer("tool_mask", agent_tool_mask.float())

        # init
        for m in list(self.q_proj)+list(self.a_proj):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)

    def tool_agg(self, agent_idx):
        if not self.use_tool_emb: return 0.0
        te = self.emb_tool(self.tool_idx[agent_idx])  # (B,T,H)
        m  = self.tool_mask[agent_idx].unsqueeze(-1)  # (B,T,1)
        return (te*m).sum(1) / (m.sum(1)+1e-8)        # (B,H)

    def encode_q(self, q_vec):
        qh = self.q_proj(q_vec)
        return F.normalize(qh, dim=-1)

    def encode_a(self, a_vec, agent_idx):
        ah = self.a_proj(a_vec)
        if self.use_tool_emb:
            ah = ah + 0.5 * self.tool_agg(agent_idx)
        if self.use_agent_id_emb:
            ah = ah + self.agent_id_weight * self.emb_agent(agent_idx)  # << 新增一行
        return F.normalize(ah, dim=-1)


def info_nce_loss(qe, ae, temperature=0.07):
    logits = qe @ ae.t()
    labels = torch.arange(qe.size(0), device=qe.device)
    return F.cross_entropy(logits/temperature, labels)

# -------------- pairs --------------
def build_training_pairs(all_rankings: Dict[str, List[str]], rng_seed=42):
    '''
    这段函数只在“训练样本构造”里收集正样本对，并不做任何显式“负采样”。

    pos_list = ranked[:pos_topk]：把该 qid 的**前 top-k（如 5）**当正样本。

    pairs.append((qid, pos_a))：为每个正样本生成一条 (q, a⁺) 训练对。

    rnd.shuffle(pairs)：仅打乱顺序，不改变正/负的角色。

    然后在训练环节里，你用 InfoNCE + in-batch negatives：

    一个 batch 形如 [(q1,a1⁺), (q2,a2⁺), …]；

    对 q1：a1⁺ 是正，其它 a2⁺, a3⁺, … 都被当成负（这就是“in-batch 负样本”）；

    所以这段 build_training_pairs 没有采负样本，负样本由损失函数在同批次内自动提供。
    '''
    rnd = random.Random(rng_seed)
    pairs = []
    for qid, ranked in all_rankings.items():
        pos_list = ranked[:pos_topk] if ranked else []
        for pos_a in pos_list:
            pairs.append((qid, pos_a))
    rnd.shuffle(pairs); return pairs

def dataset_signature(a_ids, all_rankings):
    payload = {"a_ids": a_ids, "rankings": {k: all_rankings[k] for k in sorted(all_rankings.keys())}}
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return f"{(zlib.crc32(blob) & 0xFFFFFFFF):08x}"

def training_cache_paths(cache_dir: str):
    return (os.path.join(cache_dir,"train_qids.json"),
            os.path.join(cache_dir,"valid_qids.json"),
            os.path.join(cache_dir,"pairs_train.npy"),
            os.path.join(cache_dir,"train_cache_meta.json"))

def training_cache_exists(cache_dir: str) -> bool:
    p_train,p_valid,p_pairs,p_meta = training_cache_paths(cache_dir)
    return all(os.path.exists(p) for p in (p_train,p_valid,p_pairs,p_meta))

def save_training_cache(cache_dir, train_qids, valid_qids, pairs_idx_np, meta):
    p_train,p_valid,p_pairs,p_meta = training_cache_paths(cache_dir)
    with open(p_train,"w",encoding="utf-8") as f: json.dump(train_qids,f,ensure_ascii=False)
    with open(p_valid,"w",encoding="utf-8") as f: json.dump(valid_qids,f,ensure_ascii=False)
    np.save(p_pairs, pairs_idx_np.astype(np.int64))
    with open(p_meta,"w",encoding="utf-8") as f: json.dump(meta,f,ensure_ascii=False,sort_keys=True)

def load_training_cache(cache_dir):
    p_train,p_valid,p_pairs,p_meta = training_cache_paths(cache_dir)
    with open(p_train,"r",encoding="utf-8") as f: train_qids=json.load(f)
    with open(p_valid,"r",encoding="utf-8") as f: valid_qids=json.load(f)
    pairs_idx_np = np.load(p_pairs)
    with open(p_meta,"r",encoding="utf-8") as f: meta=json.load(f)
    return train_qids, valid_qids, pairs_idx_np, meta

# -------------- main --------------
from contextlib import nullcontext

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--max_features", type=int, default=5000)
    parser.add_argument("--hid", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--rng_seed_pairs", type=int, default=42)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--valid_ratio", type=float, default=0.2)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--rebuild_cache", type=int, default=0)
    parser.add_argument("--rebuild_training_cache", type=int, default=0)
    parser.add_argument("--eval_chunk", type=int, default=8192, help="batch size over agents for inference")
    parser.add_argument("--amp", type=int, default=0, help="1 to enable autocast on CUDA (bfloat16)")
    parser.add_argument("--use_tool_emb", type=int, default=1)
    parser.add_argument("--use_agent_id_emb", type=int, default=0, help="1 to add agent-ID embedding into agent tower")
    parser.add_argument("--agent_id_weight", type=float, default=0.5, help="scaling for agent-ID embedding")
    args = parser.parse_args()

    random.seed(1234); np.random.seed(1234); torch.manual_seed(1234)
    device = torch.device(args.device)

    all_agents, all_questions, all_rankings, tools = collect_data(args.data_root)
    print(f"Loaded {len(all_agents)} agents, {len(all_questions)} questions, {len(tools)} tools.")

    cache_dir = ensure_cache_dir(args.data_root)

    # Feature cache (always keep on CPU)
    if cache_exists(cache_dir) and args.rebuild_cache==0:
        q_ids, a_ids, tool_names, Q, A, idx_pad_np, mask_np = load_cache(cache_dir)
        print(f"[cache] loaded features from {cache_dir}")
        qid2idx = {qid:i for i,qid in enumerate(q_ids)}
        aid2idx = {aid:i for i,aid in enumerate(a_ids)}
        idx_pad = torch.from_numpy(idx_pad_np)  # kept on CPU and moved when needed
        mask = torch.from_numpy(mask_np)
    else:
        q_ids, q_texts, tool_names, tool_texts, a_ids, a_texts, a_tool_lists = build_text_corpora(all_agents, all_questions, tools)
        q_vec, tool_vec, a_vec, Q_csr, Tm_csr, Am_csr = build_vectorizers(q_texts, tool_texts, a_texts, args.max_features)
        Atool = agent_tool_text_matrix(a_tool_lists, tool_names, Tm_csr)
        Am = Am_csr.toarray().astype(np.float32)
        A = np.concatenate([Am, Atool], axis=1).astype(np.float32)
        Q = Q_csr.toarray().astype(np.float32)
        idx_pad, mask = build_agent_tool_id_buffers(a_ids, a_tool_lists, tool_names)
        save_cache(cache_dir, q_ids, a_ids, tool_names, Q, A, idx_pad.numpy(), mask.numpy())
        qid2idx = {qid:i for i,qid in enumerate(q_ids)}
        aid2idx = {aid:i for i,aid in enumerate(a_ids)}

    data_sig = dataset_signature(a_ids, all_rankings)
    model_path, qvec_path, latest_model, latest_qvec, meta_path = model_save_paths(cache_dir, data_sig)

    # Train/valid pairs
    want_meta = {"data_sig":data_sig,"rng_seed_pairs":int(args.rng_seed_pairs),"split_seed":int(args.split_seed),
                 "valid_ratio":float(args.valid_ratio),"pair_type":"q_pos_only_posTopK"}
    use_cache = training_cache_exists(cache_dir) and (args.rebuild_training_cache==0)
    if use_cache:
        train_qids, valid_qids, pairs_idx_np, meta = load_training_cache(cache_dir)
        if meta!=want_meta: use_cache=False
    if not use_cache:
        qids_in_rank = [qid for qid in q_ids if qid in all_rankings]
        train_qids, valid_qids = train_valid_split(qids_in_rank, valid_ratio=args.valid_ratio, seed=args.split_seed)
        pairs = build_training_pairs({qid:all_rankings[qid] for qid in train_qids}, rng_seed=args.rng_seed_pairs)
        pairs_idx_np = np.array([(qid2idx[q], aid2idx[a]) for (q,a) in pairs], dtype=np.int64)
        save_training_cache(cache_dir, train_qids, valid_qids, pairs_idx_np, want_meta)
        print(f"[cache] saved train/valid/pairs to {cache_dir} (sig={data_sig})")

    # Model
    d_q = Q.shape[1]; d_a = A.shape[1]; num_tools = len(tool_names); num_agents = len(a_ids)
    encoder = TwoTower(
        d_q, d_a, num_tools,
        idx_pad.to(device), mask.to(device),
        hid=args.hid, tool_emb=args.use_tool_emb,
        num_agents=num_agents,
        use_agent_id_emb=bool(args.use_agent_id_emb),
        agent_id_weight=args.agent_id_weight
    ).to(device)

    optimizer = torch.optim.Adam(encoder.parameters(), lr=args.lr)

    # Keep feature matrices on CPU
    Q_cpu = Q; A_cpu = A

    # Train with in-batch negatives; move only the batch to device
    num_pairs = pairs_idx_np.shape[0]
    num_batches = (num_pairs + args.batch_size - 1)//args.batch_size
    print(f"Training pairs: {num_pairs}, batches/epoch: {num_batches}")

    use_amp = (args.amp==1 and device.type=='cuda')
    for epoch in range(1, args.epochs+1):
        # shuffle pairs in-place
        np.random.shuffle(pairs_idx_np)
        total = 0.0
        pbar = tqdm(range(num_batches), desc=f"Epoch {epoch}/{args.epochs}", dynamic_ncols=True)
        for b in pbar:
            sl = slice(b*args.batch_size, min((b+1)*args.batch_size, num_pairs))
            batch = pairs_idx_np[sl]
            if batch.size==0: continue
            q_idx = torch.from_numpy(batch[:,0]).long()
            a_idx = torch.from_numpy(batch[:,1]).long()

            q_vec = torch.from_numpy(Q_cpu[q_idx]).to(device, non_blocking=True)
            a_pos = torch.from_numpy(A_cpu[a_idx]).to(device, non_blocking=True)
            a_idx = a_idx.to(device)

            if use_amp:
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    qe = encoder.encode_q(q_vec)
                    ae = encoder.encode_a(a_pos, a_idx)
                    loss = info_nce_loss(qe, ae, temperature=args.temperature)
            else:
                qe = encoder.encode_q(q_vec)
                ae = encoder.encode_a(a_pos, a_idx)
                loss = info_nce_loss(qe, ae, temperature=args.temperature)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total += float(loss.item())
            pbar.set_postfix({"batch_loss":f"{loss.item():.4f}","avg_loss":f"{(total/(b+1)):.4f}"})

        print(f"Epoch {epoch}/{args.epochs} - InfoNCE: {(total/max(1,num_batches)):.4f}")

    # Save
    model_dir = os.path.join(cache_dir, "models"); os.makedirs(model_dir, exist_ok=True)
    ckpt = {"state_dict":encoder.state_dict(), "data_sig":data_sig, "saved_at":datetime.now().isoformat(timespec="seconds"),
            "args":vars(args), "dims": {"d_q": int(d_q), "d_a": int(d_a), "hid": int(args.hid),
           "num_tools": int(num_tools), "num_agents": int(num_agents)}, "flags": {"use_tool_emb": bool(args.use_tool_emb),
            "use_agent_id_emb": bool(args.use_agent_id_emb),
            "agent_id_weight": float(args.agent_id_weight)},
            "mappings":{"q_ids":q_ids,"a_ids":a_ids,"tool_names":tool_names}}
    torch.save(ckpt, model_path); torch.save(ckpt, latest_model)
    with open(meta_path,"w",encoding="utf-8") as f:
        json.dump({"data_sig":data_sig,"a_ids":a_ids,"tool_names":tool_names}, f, ensure_ascii=False, indent=2)
    print(f"[save] model -> {model_path}")
    print(f"[save] meta  -> {meta_path}")

    # Validation (sampled) still small-batch moves
    valid_metrics = evaluate_model(encoder, Q_cpu, A_cpu, qid2idx, a_ids, all_rankings,
                                   valid_qids, device=device, ks=(5,10,50), cand_size=1000, rng_seed=123, amp=use_amp)
    print_metrics_table("Validation (averaged over questions)", valid_metrics, ks=(5,10,50), filename=filename)

    # Inference: chunk over all agents
    @torch.no_grad()
    def recommend_topk_for_qid(qid: str, topk: int = 10, chunk: int = 8192):
        qi = qid2idx[qid]
        qv = torch.from_numpy(Q_cpu[qi:qi+1]).to(device)
        qe = encoder.encode_q(qv)  # (1,H)

        best_scores = []
        best_ids = []
        N = len(a_ids)
        for i in range(0, N, chunk):
            j = min(i+chunk, N)
            a_idx = torch.arange(i, j, dtype=torch.long, device=device)
            av = torch.from_numpy(A_cpu[i:j]).to(device)
            ae = encoder.encode_a(av, a_idx)       # (B,H)
            scores = (qe @ ae.t()).squeeze(0)      # (B,)
            # take partial topk
            k = min(topk, j-i)
            top_scores, top_local_idx = torch.topk(scores, k)
            best_scores.extend(top_scores.cpu().tolist())
            best_ids.extend([i+int(t) for t in top_local_idx.cpu().tolist()])

        # final topk among chunk winners
        best_scores = torch.tensor(best_scores)
        best_ids = torch.tensor(best_ids)
        k = min(topk, best_scores.numel())
        final_scores, final_idx = torch.topk(best_scores, k)
        result = [(a_ids[int(best_ids[idx])], float(final_scores[n].item())) for n, idx in enumerate(final_idx)]
        return result

    sample_qids = q_ids[:min(5, len(q_ids))]
    for qid in sample_qids:
        recs = recommend_topk_for_qid(qid, topk=args.topk, chunk=args.eval_chunk)
        qtext = all_questions[qid]["input"][:80].replace("\n"," ")
        print(f"\nQuestion: {qid}  |  {qtext}")
        for r,(aid,s) in enumerate(recs,1):
            print(f"  {r:2d}. {aid:>20s}  score={s:.4f}")


# ---- helper: build_agent_tool_id_buffers ----
def build_agent_tool_id_buffers(a_ids: List[str], agent_tool_lists: List[List[str]], tool_names: List[str]):
    t_map = {n:i for i,n in enumerate(tool_names)}
    num_agents = len(a_ids)
    max_t = max([len(lst) for lst in agent_tool_lists]) if num_agents>0 else 0
    if max_t==0: max_t=1
    idx_pad = np.zeros((num_agents, max_t), dtype=np.int64)
    mask = np.zeros((num_agents, max_t), dtype=np.float32)
    for i,lst in enumerate(agent_tool_lists):
        for j,tn in enumerate(lst[:max_t]):
            if tn in t_map: idx_pad[i,j]=t_map[tn]; mask[i,j]=1.0
    return torch.from_numpy(idx_pad), torch.from_numpy(mask)

if __name__ == "__main__":
    main()
