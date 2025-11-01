#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OneRec++: Session-Aware Generative Recommender with Lite-DPO (drop-in upgrade)
------------------------------------------------------------------------------
This file upgrades your previous onerec_lite_agent_rec.py with:
  1) Session Encoder (Transformer) — encodes a short history of questions/agents
     when available; falls back to single-query encoding if sessions are absent.
  2) Unified vocab scoring — decoder head is tied to agent token embeddings so
     we can train without per-batch TF-IDF candidate masks (still supported).
  3) Reward Model (RM) + Lite-DPO finetune — optional stage after CE training.
     RM is a small MLP scoring (query/session, generated list). Lite-DPO uses
     RM preferences between two sampled lists to do preference optimization.

Dataset assumptions (same as before):
  {data_root}/PartI|PartII|PartIII/{agents,questions,rankings}/merge.json + Tools/merge.json
Optional (new):
  {data_root}/sessions.json  # [{"qid": qid, "history": [prev_qid1,...], "positive_agents": [aid,...]}]
If sessions.json is absent, we synthesize a tiny pseudo-session per qid using
TF-IDF nearest neighbours among *train* questions (text-only).

Quick Start (same metrics & eval flow retained):
  # CE train (generator)
  python onerec_plus.py --data_root /path/to/root --epochs 3 --mode gen \
         --topk 10 --device cuda:0 --use_sessions 1 --session_len 4

  # Lite-DPO finetune (after CE ckpt saved). This will:
  #   * load the CE model, freeze token embeddings (opt),
  #   * sample two lists per query with temperature, get RM preference,
  #   * run DPO loss to align the generator.
  python onerec_plus.py --data_root /path/to/root --mode dpo \
         --topk 10 --device cuda:0 --dpo_steps 2000 --dpo_batch 64

Notes:
  • This is a **research baseline** mimicking OneRec ideas — not an industrial
    replica. It keeps your caches/metrics and adds minimal new files.
  • MoE and large-scale sharding are NOT included; hooks are left for future.
"""

import os, json, math, argparse, random, zlib, pickle
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from utils import print_metrics_table

filename = os.path.splitext(os.path.basename(__file__))[0]
POS_TOPK = 5

# ---------------------- I/O helpers ----------------------

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
        agents_path = os.path.join(data_root, part, "agents", "merge.json")
        questions_path = os.path.join(data_root, part, "questions", "merge.json")
        rankings_path = os.path.join(data_root, part, "rankings", "merge.json")
        if os.path.exists(agents_path):
            all_agents.update(load_json(agents_path))
        if os.path.exists(questions_path):
            all_questions.update(load_json(questions_path))
        if os.path.exists(rankings_path):
            r = load_json(rankings_path)
            all_rankings.update(r.get("rankings", {}))
    tools_path = os.path.join(data_root, "Tools", "merge.json")
    tools = load_json(tools_path) if os.path.exists(tools_path) else {}
    return all_agents, all_questions, all_rankings, tools

# ---------------------- Text & TF-IDF ----------------------

def build_text_corpora(all_agents, all_questions, tools):
    q_ids = list(all_questions.keys())
    q_texts = [all_questions[qid].get("input", "") for qid in q_ids]

    tool_names = list(tools.keys())
    def _tool_text(tn: str) -> str:
        t = tools.get(tn, {})
        desc = t.get("description", "")
        return f"{tn} {desc}".strip()

    tool_texts = [_tool_text(tn) for tn in tool_names]

    a_ids = list(all_agents.keys())
    a_texts = []
    a_tool_lists = []
    for aid in a_ids:
        a = all_agents[aid]
        mname = a.get("M", {}).get("name", "")
        tool_list = a.get("T", {}).get("tools", []) or []
        a_tool_lists.append(tool_list)
        concat_tool_desc = " ".join([_tool_text(tn) for tn in tool_list])
        text = f"{mname} {concat_tool_desc}".strip()
        a_texts.append(text)

    return q_ids, q_texts, tool_names, tool_texts, a_ids, a_texts, a_tool_lists


def build_vectorizers(q_texts, tool_texts, a_texts, max_features: int):
    q_vec = TfidfVectorizer(max_features=max_features, lowercase=True)
    tool_vec = TfidfVectorizer(max_features=max_features, lowercase=True)
    a_vec = TfidfVectorizer(max_features=max_features, lowercase=True)

    Q = q_vec.fit_transform(q_texts)
    Tm = tool_vec.fit_transform(tool_texts)
    Am = a_vec.fit_transform(a_texts)
    return q_vec, tool_vec, a_vec, Q, Tm, Am


def agent_tool_text_matrix(agent_tool_lists: List[List[str]], tool_names: List[str], Tm) -> np.ndarray:
    name2idx = {n: i for i, n in enumerate(tool_names)}
    num_agents = len(agent_tool_lists)
    Dt = Tm.shape[1] if hasattr(Tm, 'shape') else 0
    Atool = np.zeros((num_agents, Dt), dtype=np.float32)
    if Dt == 0:
        return Atool
    for i, tool_list in enumerate(agent_tool_lists):
        idxs = [name2idx[t] for t in tool_list if t in name2idx]
        if not idxs:
            continue
        vecs = Tm[idxs].toarray()
        Atool[i] = vecs.mean(axis=0).astype(np.float32)
    return Atool

# ---------------------- Sessions ----------------------

def load_sessions_or_build(
    data_root: str,
    q_ids: List[str],
    Q_csr,                       # 稀疏 TF-IDF (scipy.sparse CSR)
    use_sessions: bool,
    session_len: int,
    train_qids: List[str],
    nn_extra: int = 8,           # 额外多取一些邻居，用于去重/去自身
    nn_batch: int = 10000,       # kneighbors 的查询批大小，避免一次性爆内存
    n_jobs: int = -1             # 并行线程
) -> Dict[str, List[str]]:
    """
    使用 sklearn NearestNeighbors(metric='cosine', algorithm='brute')
    在 train 子集上拟合，再对全体 q 进行最近邻查询，返回 qid -> 历史 qid 列表。
    - 稀疏 CSR + 余弦度量，速度和内存均较稳定
    - 分批 kneighbors 避免一次性大矩阵乘
    - 自动剔除自身与重复，并只保留 session_len 个

    若 use_sessions 且存在 sessions.json，则仍优先使用文件。
    """
    import numpy as np
    from sklearn.preprocessing import normalize
    from sklearn.neighbors import NearestNeighbors
    import os, json

    sess_path = os.path.join(data_root, "sessions.json")
    if use_sessions and os.path.exists(sess_path):
        try:
            raw = load_json(sess_path)
            out = {}
            for it in raw:
                qid = it.get("qid"); hist = it.get("history", [])
                if qid in q_ids:
                    out[qid] = hist[-session_len:]
            return out
        except Exception:
            pass  # 回退到伪 session

    # 没有 train 就返回空历史
    if not train_qids or session_len <= 0:
        return {qid: [] for qid in q_ids}

    # 确保 CSR + L2 行归一化（cosine 下与点积等价）
    from scipy.sparse import issparse
    if not issparse(Q_csr):
        raise ValueError("Q_csr must be a scipy.sparse matrix (CSR).")
    Qn = normalize(Q_csr.tocsr().astype(np.float32), norm="l2", axis=1, copy=True)

    # 建立映射
    qid2idx = {qid: i for i, qid in enumerate(q_ids)}
    train_idx = np.array([qid2idx[q] for q in train_qids if q in qid2idx], dtype=np.int64)

    if train_idx.size == 0:
        return {qid: [] for qid in q_ids}

    # 在 train 子集上拟合 NN
    # brute+cosine 对稀疏 CSR 很可靠；n_jobs=-1 并行
    k_query = min(session_len + nn_extra, train_idx.size)
    nbrs = NearestNeighbors(
        n_neighbors=k_query,
        metric="cosine",
        algorithm="brute",
        n_jobs=n_jobs
    ).fit(Qn[train_idx])

    # 分批查询全体 q 的邻居（相对 train 子集的局部索引）
    N = len(q_ids)
    # 预分配（可选）或逐批填充
    out = {}

    for s in range(0, N, nn_batch):
        e = min(s + nn_batch, N)
        # 距离越小越近；cosine 距离 d，余弦相似度 sim = 1 - d
        dist, idx_local = nbrs.kneighbors(Qn[s:e], return_distance=True)  # (B, k_query)
        # 映射为全局行号
        idx_global = train_idx[idx_local]  # (B, k_query)

        # 为每个查询样本构建历史
        for i, qrow in enumerate(range(s, e)):
            qid = q_ids[qrow]
            # 将自身（若在 train 内）剔除
            # 注意：idx_global[i] 是候选行号（全局）；qrow 为当前查询行号
            cand = idx_global[i].tolist()

            # 去掉自身 + 去重（按最小距离优先）
            # 根据 dist 排序再过滤，更稳（但我们已经是按最近邻返回的）
            # 这里仍按顺序扫描，遇到自身或重复就跳过
            hist = []
            seen = set()
            for j, gidx in enumerate(cand):
                if gidx == qrow:
                    continue
                qid_hist = q_ids[gidx]
                if qid_hist in seen:
                    continue
                seen.add(qid_hist)
                hist.append(qid_hist)
                if len(hist) >= session_len:
                    break

            out[qid] = hist

    # 若某些 qid 没被覆盖（极端情况下），补空列表
    for qid in q_ids:
        if qid not in out:
            out[qid] = []

    return out



# ---------------------- Metrics ----------------------

def _dcg_at_k(binary_hits, k):
    dcg = 0.0
    for i, h in enumerate(binary_hits[:k]):
        if h:
            dcg += 1.0 / math.log2(i + 2.0)
    return dcg

@torch.no_grad()
def evaluate_sampled(pred_ids_topk: List[str], rel_set: set, ks=(5, 10, 50)):
    bin_hits = [1 if aid in rel_set else 0 for aid in pred_ids_topk]
    out = {}
    for k in ks:
        Hk = sum(bin_hits[:k])
        P = Hk / float(k)
        R = Hk / float(len(rel_set)) if len(rel_set) > 0 else 0.0
        F1 = (2*P*R)/(P+R) if (P+R) > 0 else 0.0
        Hit = 1.0 if Hk > 0 else 0.0
        dcg = _dcg_at_k(bin_hits, k)
        ideal = min(len(rel_set), k)
        idcg = sum(1.0 / math.log2(i + 2.0) for i in range(ideal)) if ideal > 0 else 0.0
        nDCG = (dcg / idcg) if idcg > 0 else 0.0
        rr = 0.0
        for i in range(k):
            if bin_hits[i]:
                rr = 1.0 / float(i+1)
                break
        out[k] = {"P":P, "R":R, "F1":F1, "Hit":Hit, "nDCG":nDCG, "MRR":rr}
    return out

# ---------------------- Models ----------------------

class AgentVocab:
    def __init__(self, a_ids: List[str]):
        self.PAD = 0; self.BOS = 1; self.offset = 2
        self.a_ids = a_ids
        self.vocab_size = len(a_ids) + self.offset
        self.aid2tok = {aid: i + self.offset for i, aid in enumerate(a_ids)}
        self.tok2aid = {i + self.offset: aid for i, aid in enumerate(a_ids)}
    def aid_to_token(self, aid: str) -> int:
        return self.aid2tok[aid]
    def token_to_aid(self, tok: int) -> Optional[str]:
        if tok < self.offset: return None
        return self.tok2aid.get(tok, None)

class SessionEncoder(nn.Module):
    """Encodes a short history of question texts (TF-IDF projected) and/or
    positive agent token embeddings into a fixed vector.
    - inputs: q_hist_vec: (B, Lq, Dq), a_hist_tok: (B, La) [optional]
    - outputs: (B, H)
    """
    def __init__(self, d_q: int, vocab_size: int, tok_dim: int = 256, hidden: int = 512, n_heads: int = 8, n_layers: int = 2):
        super().__init__()
        self.q_proj = nn.Linear(d_q, hidden)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden, nhead=n_heads, batch_first=True)
        self.enc = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.tok_emb = nn.Embedding(vocab_size, tok_dim)
        self.mix = nn.Linear(hidden + tok_dim, hidden)
        self.norm = nn.LayerNorm(hidden)
        for m in [self.q_proj, self.mix]:
            nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, q_hist_vec: torch.Tensor, a_hist_tok: Optional[torch.Tensor] = None):
        if q_hist_vec.device != self.q_proj.weight.device:
            q_hist_vec = q_hist_vec.to(self.q_proj.weight.device, non_blocking=True)
        hq = self.q_proj(q_hist_vec)  # (B,Lq,H)
        h = self.enc(hq)              # (B,Lq,H)
        h_last = h[:, -1]             # (B,H)
        if a_hist_tok is not None:
            ae = self.tok_emb(a_hist_tok)
            ae = ae.mean(dim=1)  # (B, E)
            h_cat = torch.cat([h_last, ae], dim=1)
            h_last = self.norm(torch.relu(self.mix(h_cat)))
        return h_last  # (B,H)

class OneRecPlus(nn.Module):
    """Generative list-wise decoder with tied vocab (agent tokens).
       Decoder: GRU (can be swapped to Transformer decoder later).
    """
    def __init__(self, enc_dim: int, vocab_size: int, hidden: int = 512, num_layers: int = 2, tok_dim: int = 256, tie_weights: bool = True):
        super().__init__()
        self.hidden = hidden
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.tok_emb = nn.Embedding(vocab_size, tok_dim)
        self.enc_to_h = nn.Linear(enc_dim, hidden)
        self.dec = nn.GRU(input_size=tok_dim, hidden_size=hidden, num_layers=num_layers, batch_first=True)
        self.head = nn.Linear(hidden, vocab_size, bias=False)
        if tie_weights and tok_dim == hidden:
            self.head.weight = self.tok_emb.weight  # weight tying
        else:
            # add a bridge if dims mismatch
            self.bridge = nn.Linear(hidden, tok_dim)
        for m in [self.tok_emb, self.enc_to_h, self.head]:
            if hasattr(m, 'weight'):
                nn.init.xavier_uniform_(m.weight)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.zeros_(m.bias)
                
        nn.init.xavier_uniform_(self.tok_emb.weight)
        # 置零 PAD(0) 与 BOS(1) 的 embedding
        with torch.no_grad():
            self.tok_emb.weight.data[0].zero_()
            self.tok_emb.weight.data[1].zero_()

    def forward(self, enc_vec: torch.Tensor, tgt_inp_tok: torch.Tensor, cand_mask: Optional[torch.Tensor] = None):
        B, L = tgt_inp_tok.shape
        h0 = self.enc_to_h(enc_vec).unsqueeze(0).repeat(self.num_layers, 1, 1)
        x = self.tok_emb(tgt_inp_tok)
        out, _ = self.dec(x, h0)
        if hasattr(self, 'bridge'):
            logits = F.linear(self.bridge(out), self.tok_emb.weight)
        else:
            logits = self.head(out)
        if cand_mask is not None:
            logits = logits.masked_fill(~cand_mask.unsqueeze(1), float('-inf'))
        return logits

    @torch.no_grad()
    def generate(self, enc_vec: torch.Tensor, topk: int, cand_mask: Optional[torch.Tensor] = None, temperature: float = 0.0):
        # preserve mode
        was_training = self.training
        self.eval()
        try:
            B = enc_vec.size(0)
            h = self.enc_to_h(enc_vec).unsqueeze(0).repeat(self.num_layers, 1, 1)
            prev = torch.full((B, 1), 1, dtype=torch.long, device=enc_vec.device)  # BOS
            used = torch.zeros((B, self.vocab_size), dtype=torch.bool, device=enc_vec.device)
            used[:, 0:2] = True
            generated = []
            for _ in range(topk):
                x = self.tok_emb(prev)
                out, h = self.dec(x, h)
                if hasattr(self, 'bridge'):
                    logits = F.linear(self.bridge(out[:, -1]), self.tok_emb.weight)
                else:
                    logits = self.head(out[:, -1])
                logits = logits.masked_fill(used, float('-inf'))
                if cand_mask is not None:
                    logits = logits.masked_fill(~cand_mask, float('-inf'))
                if temperature and temperature > 0:
                    probs = F.softmax(logits / temperature, dim=-1)
                    next_tok = torch.multinomial(probs, num_samples=1).squeeze(1)
                else:
                    next_tok = torch.argmax(logits, dim=-1)
                generated.append(next_tok)
                prev = next_tok.unsqueeze(1)
                used.scatter_(1, next_tok.unsqueeze(1), True)
            return torch.stack(generated, dim=1)
        finally:
            # restore original mode
            self.train(was_training)

# ---------------------- Reward Model & Lite-DPO ----------------------

class RewardModel(nn.Module):
    """Scores (enc_vec, list_of_tokens) -> scalar reward.
       We pool token embeddings and concat with enc_vec, MLP to scalar.
    """
    def __init__(self, enc_dim: int, tok_emb: nn.Embedding, hidden: int = 512):
        super().__init__()
        self.tok_emb = tok_emb
        self.ff = nn.Sequential(
            nn.Linear(enc_dim + tok_emb.embedding_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden//2), nn.ReLU(),
            nn.Linear(hidden//2, 1)
        )
        for m in self.ff:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, enc_vec: torch.Tensor, tok_seq: torch.Tensor):
        # tok_seq: (B, L)
        te = self.tok_emb(tok_seq)  # (B,L,E)

        # mask: 仅统计真实 agent token，忽略 PAD(0) & BOS(1)
        mask = (tok_seq >= 2).float().unsqueeze(-1)  # (B,L,1)
        te_sum = (te * mask).sum(dim=1)             # (B,E)
        denom = mask.sum(dim=1).clamp_min(1.0)      # (B,1)
        te_mean = te_sum / denom                    # (B,E)

        x = torch.cat([enc_vec, te_mean], dim=1)
        r = self.ff(x).squeeze(1)
        return r

class DPOTrainer:
    def __init__(self, model: OneRecPlus, rm: RewardModel, beta: float = 0.1):
        self.model = model
        self.rm = rm
        self.beta = beta

    def dpo_loss(self, enc_vec: torch.Tensor, pref_seq: torch.Tensor, nonpref_seq: torch.Tensor):
        # compute logprobs under current model
        def seq_logprob(seq):
            B, L = seq.shape
            bos = torch.full((B,1), 1, dtype=torch.long, device=seq.device)
            inp = torch.cat([bos, seq[:, :-1]], dim=1)
            logits = self.model(enc_vec, inp, cand_mask=None)   # (B,L,V)
            lp = F.log_softmax(logits, dim=-1)                  # (B,L,V)

            # 仅统计有效 token（tok>=2）
            mask = (seq >= 2).float()                           # (B,L)
            gather = lp.gather(-1, seq.unsqueeze(-1)).squeeze(-1)  # (B,L)
            # 避免样本全 PAD：分母>=1
            token_counts = mask.sum(dim=1).clamp_min(1.0)       # (B,)
            lp_sum = (gather * mask).sum(dim=1)                 # (B,)
            return lp_sum / token_counts  
        lp_pref = seq_logprob(pref_seq)
        lp_nonp = seq_logprob(nonpref_seq)
        with torch.no_grad():
            r_pref = self.rm(enc_vec, pref_seq)
            r_nonp = self.rm(enc_vec, nonpref_seq)
            adv = r_pref - r_nonp
        # standard DPO objective: maximize log σ(β * (Δr) + Δlogπ)
        logits = self.beta * adv + (lp_pref - lp_nonp)
        loss = -F.logsigmoid(logits).mean()
        return loss



def train_valid_split(qids_in_rank, valid_ratio=0.2, seed=42):
    rng = random.Random(seed)
    q = list(qids_in_rank); rng.shuffle(q)
    n_valid = int(len(q) * valid_ratio)
    return q[n_valid:], q[:n_valid]


def dataset_signature(a_ids: List[str], all_rankings: Dict[str, List[str]]) -> str:
    payload = {"a_ids": a_ids, "rankings": {k: all_rankings[k] for k in sorted(all_rankings.keys())}}
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    sig = zlib.crc32(blob) & 0xFFFFFFFF
    return f"{sig:08x}"

@torch.no_grad()
def evaluate_model_gen(
    gen_model: OneRecPlus,
    enc_vecs: torch.Tensor,
    qid2idx: Dict[str,int],
    a_ids: List[str],
    all_rankings: Dict[str, List[str]],
    eval_qids: List[str],
    device: torch.device,
    ks=(5, 10, 50),
    cand_size: int = 1000,
    rng_seed: int = 123,
    ref_k: int = 10,      # choose which K to display on the bar
    bar_update_every: int = 1       # set >1 to reduce bar update overhead
):
    # pick a display K (default: largest K, which also matches generate(topk=max(ks)))
    if ref_k is None:
        ref_k = max(ks)

    agg = {k: {"P":0.0,"R":0.0,"F1":0.0,"Hit":0.0,"nDCG":0.0,"MRR":0.0} for k in ks}
    cnt = 0
    skipped = 0

    all_agent_set = set(a_ids)
    rnd = random.Random(rng_seed)
    vocab = AgentVocab(a_ids)

    pbar = tqdm(eval_qids, desc="Evaluating (gen, sampled)", total=len(eval_qids))
    for i, qid in enumerate(pbar, start=1):
        gt_list = [aid for aid in all_rankings.get(qid, [])[:POS_TOPK] if aid in all_agent_set]
        if not gt_list:
            skipped += 1
            # still reflect progress on the bar
            if (i % bar_update_every) == 0:
                # with no valid samples yet, avoid div-by-zero
                done = cnt
                if cnt > 0:
                    ref = agg[ref_k]
                    pbar.set_postfix({
                        "done": done,
                        "skipped": skipped,
                        f"P@{ref_k}": f"{(ref['P']/cnt):.4f}",
                        f"nDCG@{ref_k}": f"{(ref['nDCG']/cnt):.4f}",
                        f"MRR@{ref_k}": f"{(ref['MRR']/cnt):.4f}",
                        "Ncand": 0
                    })
                else:
                    pbar.set_postfix({"done": done, "skipped": skipped, f"P@{ref_k}": "0.0000",
                                      f"nDCG@{ref_k}": "0.0000", f"MRR@{ref_k}": "0.0000",
                                      "Ncand": 0})
            continue

        rel_set = set(gt_list)
        neg_pool = list(all_agent_set - rel_set)
        need_neg = max(0, cand_size - len(gt_list))
        if need_neg > 0 and len(neg_pool) > 0:
            k = min(need_neg, len(neg_pool))
            sampled_negs = rnd.sample(neg_pool, k)
            cand_ids = gt_list + sampled_negs
        else:
            cand_ids = gt_list

        # candidate mask
        mask = torch.zeros((1, vocab.vocab_size), dtype=torch.bool, device=device)
        for aid in cand_ids:
            mask[0, vocab.aid_to_token(aid)] = True

        qi = qid2idx[qid]
        enc = enc_vecs[qi:qi+1].to(device, non_blocking=True)
        pred_tok = gen_model.generate(enc, topk=max(ks), cand_mask=mask)

        pred_ids = []
        for t in pred_tok[0].tolist():
            aid = vocab.token_to_aid(t)
            if aid is not None:
                pred_ids.append(aid)

        md = evaluate_sampled(pred_ids, rel_set, ks)
        for k in ks:
            for m in md[k]:
                agg[k][m] += md[k][m]
        cnt += 1

        # update bar postfix
        if (i % bar_update_every) == 0:
            ref = agg[ref_k]
            pbar.set_postfix({
                "done": cnt,
                "skipped": skipped,
                f"P@{ref_k}": f"{(ref['P']/cnt):.4f}",
                f"nDCG@{ref_k}": f"{(ref['nDCG']/cnt):.4f}",
                f"MRR@{ref_k}": f"{(ref['MRR']/cnt):.4f}",
                "Ncand": len(cand_ids)
            })

    # final averages
    if cnt == 0:
        return {k:{m:0.0 for m in ["P","R","F1","Hit","nDCG","MRR"]} for k in ks}
    for k in ks:
        for m in agg[k]:
            agg[k][m] /= cnt
    return agg


# ---------------------- Main ----------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--max_features", type=int, default=5000)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--valid_ratio", type=float, default=0.2)
    ap.add_argument("--split_seed", type=int, default=42)
    ap.add_argument("--use_sessions", type=int, default=1)
    ap.add_argument("--session_len", type=int, default=4)
    ap.add_argument("--train_mask", type=int, default=0)
    ap.add_argument("--candidate_size", type=int, default=200)
    ap.add_argument("--mode", choices=["gen","dpo"], default="gen")
    ap.add_argument("--dpo_steps", type=int, default=1000)
    ap.add_argument("--dpo_batch", type=int, default=64)
    ap.add_argument("--beta", type=float, default=0.05, help="DPO beta")
    ap.add_argument("--nn_extra", type=int, default=8, help="会话邻居额外冗余个数，用于去重/去自身后仍能拿满 L")
    ap.add_argument("--nn_batch", type=int, default=10000, help="kneighbors 查询批大小，避免一次性内存暴涨")
    ap.add_argument("--nn_jobs", type=int, default=4, help="NearestNeighbors 的并行线程数")
    ap.add_argument("--amp", type=int, default=1, help="use torch.cuda.amp autocast")
    ap.add_argument("--enc_chunk", type=int, default=512, help="micro-batch size for session encoding")

    args = ap.parse_args()

    random.seed(1234); np.random.seed(1234); torch.manual_seed(1234)
    device = torch.device(args.device)

    # load data & tfidf features
    all_agents, all_questions, all_rankings, tools = collect_data(args.data_root)
    q_ids, q_texts, tool_names, tool_texts, a_ids, a_texts, a_tool_lists = build_text_corpora(all_agents, all_questions, tools)
    q_vec, tool_vec, a_vec, Q_csr, Tm_csr, Am_csr = build_vectorizers(q_texts, tool_texts, a_texts, args.max_features)
    Atool = agent_tool_text_matrix(a_tool_lists, tool_names, Tm_csr)
    Am = Am_csr.toarray().astype(np.float32)
    Q = Q_csr.toarray().astype(np.float32)
    A_text_full = np.concatenate([Am, Atool], axis=1).astype(np.float32)
    # ★ 关键：把 agent 文本用 q_vec 词表做 transform，得到与 Q 同维度的表征
    A_in_Q_csr = q_vec.transform(a_texts)                     # (num_agents, D_q)
    A_in_Q = A_in_Q_csr.toarray().astype(np.float32)          # 若内存吃紧，见下文“省内存版本”
    A_in_Q_t = torch.from_numpy(A_in_Q).to(device)            # 供训练时候选筛选用


    qid2idx = {qid:i for i,qid in enumerate(q_ids)}
    aid2idx = {aid:i for i,aid in enumerate(a_ids)}

    # split
    qids_in_rank = [qid for qid in q_ids if qid in all_rankings]
    train_qids, valid_qids = train_valid_split(qids_in_rank, valid_ratio=args.valid_ratio, seed=args.split_seed)

    # sessions
    sessions = load_sessions_or_build(
        args.data_root, q_ids, Q_csr,
        bool(args.use_sessions), args.session_len, train_qids,
        nn_extra=args.nn_extra, nn_batch=args.nn_batch, n_jobs=args.nn_jobs
    )


    # tensors
    Q_t_cpu = torch.from_numpy(Q).pin_memory()   # ★ 留在CPU，便于无阻塞拷贝
    A_text_t = torch.from_numpy(A_text_full).to(device)

    data_sig = dataset_signature(a_ids, all_rankings)
    cache_dir = ensure_cache_dir(args.data_root)
    model_dir = os.path.join(cache_dir, "models"); os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"{filename}_{data_sig}.pt")
    meta_path = os.path.join(model_dir, f"meta_{data_sig}.json")

    vocab = AgentVocab(a_ids)

    def encode_sessions_in_chunks(qid_batch: List[str], chunk: int = 512) -> torch.Tensor:
        outs = []
        sess_enc.eval()
        with torch.no_grad():
            for i in range(0, len(qid_batch), chunk):
                qids_s = qid_batch[i:i+chunk]
                h_cpu = build_session_tensor(qids_s)                 # CPU
                h = h_cpu.to(device, non_blocking=True)
                enc = sess_enc(h, None).cpu()                        # <- back to CPU
                outs.append(enc)
                del h, h_cpu, enc
                torch.cuda.empty_cache()
        return torch.cat(outs, dim=0)                                # CPU



    # build session encoder inputs (q_hist -> stacked TF-IDF vectors)
    def build_session_tensor(qid_batch: List[str]) -> torch.Tensor:
        """
        CPU 端构建会话张量 (B, L, Dq)；返回的是 *CPU tensor*。
        只在调用处把小块搬到 GPU。
        """
        L = args.session_len if args.use_sessions else 1
        B = len(qid_batch)
        Dq = Q_t_cpu.shape[1]

        out = torch.zeros((B, L, Dq), dtype=Q_t_cpu.dtype)  # ★ 不指定 device
        for i, qid in enumerate(qid_batch):
            hist = sessions.get(qid, [])[-L:]
            if not hist:
                qi = qid2idx[qid]
                out[i, -1] = Q_t_cpu[qi]
            else:
                start = L - len(hist)
                for j, hq in enumerate(hist):
                    if hq in qid2idx:
                        out[i, start+j] = Q_t_cpu[qid2idx[hq]]
                out[i, -1] = Q_t_cpu[qid2idx[qid]]
        return out

    # simple agent-history tokens (optional, empty for now)
    def build_agent_hist_tokens(qid_batch: List[str]) -> Optional[torch.Tensor]:
        return None

    # instantiate models
    sess_enc = SessionEncoder(d_q=Q_t_cpu.shape[1], vocab_size=vocab.vocab_size, tok_dim=256, hidden=512, n_heads=8, n_layers=2).to(device)
    gen = OneRecPlus(enc_dim=512, vocab_size=vocab.vocab_size, hidden=512, num_layers=2, tok_dim=256, tie_weights=True).to(device)

    if args.mode == "gen":
        params = list(sess_enc.parameters()) + list(gen.parameters())
        opt = torch.optim.Adam(params, lr=args.lr)

        # build training sequences from rankings (teacher-forcing targets)
        def build_targets(qids: List[str]) -> Tuple[List[str], List[List[int]]]:
            in_q = []; tgt_tok = []
            for qid in qids:
                ranked = [aid for aid in all_rankings.get(qid, [])[:args.topk] if aid in vocab.aid2tok]
                if not ranked:
                    continue
                toks = [vocab.aid_to_token(a) for a in ranked]
                if len(toks) < args.topk:
                    toks += [vocab.PAD]*(args.topk - len(toks))
                in_q.append(qid); tgt_tok.append(toks)
            return in_q, tgt_tok

        train_q, train_targets = build_targets(train_qids)
        valid_q, valid_targets = build_targets(valid_qids)
        print(f"[gen] train sequences={len(train_q)}  valid sequences={len(valid_q)}  topk={args.topk}")

        nb = math.ceil(len(train_q) / args.batch_size)
        for epoch in range(1, args.epochs+1):
            order = list(range(len(train_q))); random.shuffle(order)
            total = 0.0
            pbar = tqdm(range(nb), desc=f"Epoch {epoch}/{args.epochs} [GEN]")
            for b in pbar:
                sl = order[b*args.batch_size:(b+1)*args.batch_size]
                if not sl: continue
                qid_batch = [train_q[i] for i in sl]
                tgt = torch.tensor([train_targets[i] for i in sl], dtype=torch.long, device=device)
                B = tgt.size(0)
                bos = torch.full((B,1), 1, dtype=torch.long, device=device)
                inp = torch.cat([bos, tgt[:, :-1]], dim=1)

                q_hist = build_session_tensor(qid_batch).to(device, non_blocking=True)
                a_hist = build_agent_hist_tokens(qid_batch)

                cand_mask = None
                if args.train_mask:
                    # restrict to candidate_size by cosine in TF-IDF space
                    # 在与 Q 同一词表/空间下做相似度
                    Av = F.normalize(A_in_Q_t, dim=1)                                                # (N_agents, D_q)
                    qv = F.normalize(Q_t_cpu[[qid2idx[q] for q in qid_batch]].to(device), dim=1)                    # (B, D_q)
                    sims = qv @ Av.T                                                                  # (B, N_agents)

                    topN = torch.topk(sims, k=min(args.candidate_size, sims.size(1)), dim=1).indices
                    cand_mask = torch.zeros((B, vocab.vocab_size), dtype=torch.bool, device=device)
                    for i in range(B):
                        cand_mask[i, 0:2] = False
                        cand_mask[i, (topN[i] + vocab.offset).to(device)] = True


                use_amp = bool(args.amp)
                # 训练循环里：
                with torch.cuda.amp.autocast(enabled=use_amp):
                    enc_vec = sess_enc(q_hist, a_hist)          # <<< autocast here, too
                    logits = gen(enc_vec, inp, cand_mask)
                loss = F.cross_entropy(logits.reshape(-1, vocab.vocab_size), tgt.reshape(-1), ignore_index=vocab.PAD)
                opt.zero_grad(); loss.backward(); opt.step()
                total += loss.item()
                pbar.set_postfix({"ce": f"{loss.item():.4f}", "avg": f"{total/(b+1):.4f}"})
            print(f"Epoch {epoch}: avg CE={total/max(1,nb):.4f}")

        # save
        ckpt = {
            "mode": "gen",
            "sess_enc": sess_enc.state_dict(),
            "gen": gen.state_dict(),
            "data_sig": data_sig,
            "saved_at": datetime.now().isoformat(timespec="seconds"),
            "dims": {"d_q": int(Q_t_cpu.shape[1]), "vocab_size": vocab.vocab_size}
        }
        torch.save(ckpt, model_path)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({"a_ids": a_ids, "q_ids": q_ids, "data_sig": data_sig}, f, ensure_ascii=False, indent=2)
        print(f"[save] model -> {model_path}\n[save] meta  -> {meta_path}")

        # build full encodings for eval
        enc_all = []
        bs = 2048
        sess_enc.eval()
        with torch.no_grad():  # no grads in eval
            for i in range(0, len(q_ids), bs):
                qid_batch = q_ids[i:i+bs]
                enc_vec = encode_sessions_in_chunks(qid_batch, chunk=getattr(args, "enc_chunk", 512))
                enc_all.append(enc_vec.cpu())   # <- keep on CPU
                del enc_vec
        enc_all = torch.cat(enc_all, dim=0)    # CPU tensor

        valid_metrics = evaluate_model_gen(gen_model=gen, enc_vecs=enc_all, qid2idx=qid2idx, a_ids=a_ids, all_rankings=all_rankings, eval_qids=valid_qids, device=device, ks=(5, 10, 50), cand_size=1000)
        print_metrics_table("Validation (GEN+Session, sampled)", valid_metrics, filename=filename)
        return

    # ---------------------- Lite-DPO finetune ----------------------
    if args.mode == "dpo":
        assert os.path.exists(model_path), f"CE checkpoint not found: {model_path}"
        ckpt = torch.load(model_path, map_location=device)
        sess_enc.load_state_dict(ckpt["sess_enc"])
        gen.load_state_dict(ckpt["gen"])
        rm = RewardModel(enc_dim=512, tok_emb=gen.tok_emb, hidden=512).to(device)
        dpo = DPOTrainer(model=gen, rm=rm, beta=args.beta)

        # simple RM pretrain: distinguish GT list vs random list
        print("[RM] Pretraining reward model...")
        opt_rm = torch.optim.Adam(rm.parameters(), lr=args.lr)
        rm_steps = max(500, args.dpo_steps//4)
        rm_batch = args.dpo_batch
        train_pool = [qid for qid in train_qids if qid in qid2idx]
        for step in tqdm(range(rm_steps), desc="RM pretrain"):
            batch_q = [q for q in random.sample(train_pool, min(rm_batch, len(train_pool))) 
                    if len(all_rankings.get(q, [])[:args.topk]) > 0]
            if not batch_q:
                continue

            q_hist = build_session_tensor(batch_q).to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=bool(args.amp)):
                enc = sess_enc(q_hist, build_agent_hist_tokens(batch_q))
                seq_a = gen.generate(enc, topk=args.topk, cand_mask=None, temperature=0.7)
                seq_b = gen.generate(enc, topk=args.topk, cand_mask=None, temperature=1.3)
            # build GT and random lists
            gt_tok = []
            rnd_tok = []
            for q in batch_q:
                ranked = [aid for aid in all_rankings.get(q, [])[:args.topk] if aid in vocab.aid2tok]
                if not ranked:
                    # 没有 GT 时，用随机正样本替代；长度 <= topk
                    k = min(args.topk, len(a_ids))
                    ranked = random.sample(a_ids, k=k)
                gt = [vocab.aid_to_token(a) for a in ranked]
                # 让随机序列与 gt 等长（再统一 pad 到 topk）
                k = len(gt)
                if k == 0:
                    k = 1  # 防极端情况，至少 1 个占位再 pad
                    gt = [vocab.PAD]
                rnd = [vocab.aid_to_token(a) for a in random.sample(a_ids, k=len(gt))]
                # 右侧 PAD 到固定长度 topk
                L = args.topk
                if len(gt) < L: gt = gt + [vocab.PAD] * (L - len(gt))
                if len(rnd) < L: rnd = rnd + [vocab.PAD] * (L - len(rnd))

                gt_tok.append(gt)
                rnd_tok.append(rnd)
            gt_tok = torch.tensor(gt_tok, dtype=torch.long, device=device)
            rnd_tok = torch.tensor(rnd_tok, dtype=torch.long, device=device)
            r_pos = rm(enc, gt_tok)
            r_rand = rm(enc, rnd_tok)
            r_a = rm(enc, seq_a)
            r_b = rm(enc, seq_b)

            # 三元 hinge（任选其一）：希望 r_pos > r_a, r_pos > r_b, r_pos > r_rand
            loss_rm = (
                F.softplus(-(r_pos - r_rand)) +
                F.softplus(-(r_pos - r_a)) +
                F.softplus(-(r_pos - r_b))
            ).mean()
            opt_rm.zero_grad(); loss_rm.backward(); opt_rm.step()
        print("[RM] done.")

        # DPO finetune the generator (sess_enc frozen for stability; unfreeze if needed)
        for p in sess_enc.parameters():
            p.requires_grad = False
        opt_dpo = torch.optim.Adam(gen.parameters(), lr=args.lr)

        dpo_steps = args.dpo_steps
        dpo_batch = args.dpo_batch
        for step in tqdm(range(dpo_steps), desc="Lite-DPO"):
            batch_q = [q for q in random.sample(train_pool, min(rm_batch, len(train_pool))) 
                    if len(all_rankings.get(q, [])[:args.topk]) > 0]
            if not batch_q:
                continue
            q_hist = build_session_tensor(batch_q).to(device, non_blocking=True)  # <<< fix
            with torch.cuda.amp.autocast(enabled=bool(args.amp)):
                enc = sess_enc(q_hist, build_agent_hist_tokens(batch_q))
                # sample two lists by non-zero temperature
                seq_a = gen.generate(enc, topk=args.topk, cand_mask=None, temperature=0.7)
                seq_b = gen.generate(enc, topk=args.topk, cand_mask=None, temperature=1.3)
                
                # --- 新增：按 RM 排序 + 过滤弱偏好 ---
                with torch.no_grad():
                    r_a = rm(enc, seq_a)
                    r_b = rm(enc, seq_b)
                    prefer_a = (r_a >= r_b)
                    # 过滤掉 |Δr| 很小的样本（可选，提升信噪比）
                    margin = 0.05  # 可调：0.02~0.2 视 RM 尺度而定
                    keep = (r_a - r_b).abs() >= margin
                    if keep.sum() == 0:
                        continue  # 本步跳过

                # 组装 (pref, nonpref)
                pref_seq  = torch.where(prefer_a.unsqueeze(1), seq_a, seq_b)[keep]
                nonpref_seq = torch.where(prefer_a.unsqueeze(1), seq_b, seq_a)[keep]
                enc_keep = enc[keep]
                
            gen.train()
            loss = dpo.dpo_loss(enc_keep, pref_seq, nonpref_seq)
            opt_dpo.zero_grad(); loss.backward(); opt_dpo.step()
            if (step+1) % 100 == 0:
                print(f"[DPO] step {step+1}/{dpo_steps} loss={loss.item():.4f}")

        # save updated model
        ckpt_out = {
            "mode": "gen+dpo",
            "sess_enc": sess_enc.state_dict(),
            "gen": gen.state_dict(),
            "data_sig": data_sig,
            "saved_at": datetime.now().isoformat(timespec="seconds")
        }
        out_path = model_path.replace(".pt", "_dpo.pt")
        torch.save(ckpt_out, out_path)
        print(f"[save] dpo model -> {out_path}")

        # quick eval (encode all q)
        enc_all = []
        bs = 2048
        for i in range(0, len(q_ids), bs):
            qid_batch = q_ids[i:i+bs]
            enc = encode_sessions_in_chunks(qid_batch, chunk=getattr(args, "enc_chunk", 512))
            enc_all.append(enc)

        enc_all = torch.cat(enc_all, dim=0)
        valid_metrics = evaluate_model_gen(gen_model=gen, enc_vecs=enc_all, qid2idx=qid2idx, a_ids=a_ids, all_rankings=all_rankings, eval_qids=valid_qids, device=device, ks=(5, 10, 50), cand_size=1000)
        print_metrics_table("Validation (GEN+DPO, sampled)", valid_metrics, filename=filename)
        return

if __name__ == "__main__":
    main()
