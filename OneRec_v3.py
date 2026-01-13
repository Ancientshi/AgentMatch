#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OneRec++ (Query->Agent) with structured sequence output:
  <LLM_TOKEN> <SEP> <TOOL_TOKEN_1> ... <TOOL_TOKEN_T> <END>

- Candidate-only scoring: logits (B,C) computed by sequence logprob per candidate agent.
- SFT: multi-positive softmax over candidates using these logits.
- InfoNCE: optional (query emb vs agent-seq pooled emb).
- DPO: list sampling without replacement over candidates; logprob uses candidate-wise logits.

Assumptions:
- You still evaluate on agent IDs (ground-truth is a list of agent IDs).
- For scoring/eval, we still pick candidate agent IDs; the model scores candidates
  by how likely their (llm_id, tool_ids) sequence is.

This avoids needing a global agent-token vocabulary.
"""

from __future__ import annotations
import os, json, math, argparse, random, zlib, copy
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from sklearn.neighbors import NearestNeighbors

from agent_rec.features import load_feature_cache, build_agent_content_view
from agent_rec.config import EVAL_TOPK, TFIDF_MAX_FEATURES, pos_topk_for_qid
from agent_rec.data import stratified_train_valid_split
from agent_rec.run_common import set_global_seed, warn_if_topk_diff
from utils import print_metrics_table


filename = os.path.splitext(os.path.basename(__file__))[0]


# ---------------------- basic utils ----------------------

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
    qid_to_part: Dict[str, str] = {}

    for part in parts:
        agents_path = os.path.join(data_root, part, "agents", "merge.json")
        questions_path = os.path.join(data_root, part, "questions", "merge.json")
        rankings_path = os.path.join(data_root, part, "rankings", "merge.json")

        if os.path.exists(agents_path):
            all_agents.update(load_json(agents_path))

        if os.path.exists(questions_path):
            qd = load_json(questions_path)
            all_questions.update(qd)
            for qid in qd.keys():
                qid_to_part.setdefault(qid, part)

        if os.path.exists(rankings_path):
            r = load_json(rankings_path)
            rr = r.get("rankings", {})
            all_rankings.update(rr)
            for qid in rr.keys():
                qid_to_part.setdefault(qid, part)

    tools_path = os.path.join(data_root, "Tools", "merge.json")
    tools = load_json(tools_path) if os.path.exists(tools_path) else {}
    return all_agents, all_questions, all_rankings, tools, qid_to_part

def build_text_corpora(all_agents, all_questions, tools):
    q_ids = sorted(all_questions.keys())
    q_texts = [all_questions[qid].get("input", "") for qid in q_ids]

    def _tool_text(tn: str) -> str:
        t = tools.get(tn, {})
        desc = t.get("description", "")
        return f"{tn} {desc}".strip()

    a_ids = sorted(all_agents.keys())
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

    return q_ids, q_texts, a_ids, a_texts, a_tool_lists

def stratified_split_by_part(qids, qid_to_part, valid_ratio, seed):
    return stratified_train_valid_split(qids, qid_to_part=qid_to_part, valid_ratio=valid_ratio, seed=seed)

def sample_qids_by_part(qids, qid_to_part, per_part, seed, parts=None):
    if per_part <= 0:
        return list(qids)
    rng = random.Random(seed)
    buckets = defaultdict(list)
    for q in qids:
        buckets[qid_to_part.get(q, "Unknown")].append(q)
    if parts is None:
        parts = sorted(buckets.keys())
    out = []
    for p in parts:
        lst = buckets.get(p, [])
        rng.shuffle(lst)
        out.extend(lst[: min(per_part, len(lst))])
    rng.shuffle(out)
    return out

def dataset_signature(a_ids: List[str], all_rankings: Dict[str, List[str]]) -> str:
    payload = {"a_ids": a_ids, "rankings": {k: all_rankings[k] for k in sorted(all_rankings.keys())}}
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    sig = zlib.crc32(blob) & 0xFFFFFFFF
    return f"{sig:08x}"


# ---------------------- metrics (same as your sampled binary relevance) ----------------------

def _dcg_at_k(binary_hits, k):
    dcg = 0.0
    for i, h in enumerate(binary_hits[:k]):
        if h:
            dcg += 1.0 / math.log2(i + 2.0)
    return dcg

@torch.no_grad()
def evaluate_sampled(pred_ids_topk: List[str], rel_set: set, ks=(10,)):
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


# ---------------------- query encoder (Transformer) ----------------------

class QueryTransformerEncoder(nn.Module):
    """
    Encode TF-IDF dense vector -> token sequence -> TransformerEncoder -> pooled.
    This is a "known" transformer backbone (PyTorch nn.TransformerEncoder).
    """
    def __init__(self, d_in: int, d_model: int = 512, n_heads: int = 8, n_layers: int = 2, dropout: float = 0.1, n_tokens: int = 16):
        super().__init__()
        self.d_in = d_in
        self.d_model = d_model
        self.n_tokens = n_tokens

        # project TFIDF vector to a small token sequence
        self.proj = nn.Linear(d_in, n_tokens * d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=4*d_model,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True
        )
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.ln = nn.LayerNorm(d_model)

        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, q_vec: torch.Tensor) -> torch.Tensor:
        """
        q_vec: (B, d_in)
        return: (B, d_model)
        """
        B = q_vec.size(0)
        x = self.proj(q_vec).view(B, self.n_tokens, self.d_model)  # (B, T, H)
        x = self.enc(x)                                            # (B, T, H)
        x = self.ln(x)
        # mean pool
        return x.mean(dim=1)


# ---------------------- token vocab for structured output ----------------------

class StructuredTokenVocab:
    """
    Token space:
      0: PAD
      1: BOS
      2: SEP
      3: END
      4: UNK_LLM
      5: UNK_TOOL
      6..(6+num_llm-1): LLM tokens
      (offset_tool)..: TOOL tokens
    """
    PAD = 0
    BOS = 1
    SEP = 2
    END = 3
    UNK_LLM = 4
    UNK_TOOL = 5

    def __init__(self, llm_vocab: List[str], tool_vocab: List[str]):
        self.llm_vocab = llm_vocab
        self.tool_vocab = tool_vocab
        self.offset_llm = 6
        self.offset_tool = self.offset_llm + len(llm_vocab)
        self.vocab_size = self.offset_tool + len(tool_vocab)

    def llm_token(self, llm_idx: int) -> int:
        if llm_idx < 0 or llm_idx >= len(self.llm_vocab):
            return self.UNK_LLM
        return self.offset_llm + llm_idx

    def tool_token(self, tool_idx: int) -> int:
        if tool_idx < 0 or tool_idx >= len(self.tool_vocab):
            return self.UNK_TOOL
        return self.offset_tool + tool_idx


def build_agent_seq_tokens(
    agent_llm_idx: np.ndarray,                 # (Na,)
    agent_tool_idx_padded: np.ndarray,         # (Na,T) tool indices in tool_vocab
    agent_tool_mask: np.ndarray,               # (Na,T) float mask
    vocab: StructuredTokenVocab,
    max_tools_per_agent: int,
) -> np.ndarray:
    """
    Build per-agent token sequence:
      [LLM] [SEP] [TOOL_1] ... [TOOL_T] [END]
    Output: (Na, L) int64, where L = 1 + 1 + T + 1
    """
    Na = agent_tool_idx_padded.shape[0]
    T = max_tools_per_agent
    L = 1 + 1 + T + 1
    out = np.full((Na, L), vocab.PAD, dtype=np.int64)

    for i in range(Na):
        out[i, 0] = vocab.llm_token(int(agent_llm_idx[i]))
        out[i, 1] = vocab.SEP
        # tools
        for j in range(T):
            if j >= agent_tool_idx_padded.shape[1]:
                break
            if agent_tool_mask[i, j] <= 0:
                continue
            out[i, 2 + j] = vocab.tool_token(int(agent_tool_idx_padded[i, j]))
        out[i, 2 + T] = vocab.END
    return out


# ---------------------- structured generator (candidate-only) ----------------------

class StructuredSeqGenerator(nn.Module):
    """
    Score candidates by log P(seq_agent | query).
    We do a tiny conditional LM:
      - Query vector -> prefix embedding (1 token)
      - Then teacher-forcing over agent sequence tokens with TransformerEncoder over (prefix + shifted seq)

    Candidate-only: score B,C by running forward on (B*C, L) sequences.
    """
    def __init__(self, enc_dim: int, vocab_size: int, d_model: int = 256, n_heads: int = 8, n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.enc_dim = enc_dim
        self.vocab_size = vocab_size
        self.d_model = d_model

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.q_prefix = nn.Linear(enc_dim, d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=4*d_model,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True
        )
        self.tr = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.ln = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # tie weights (optional but good)
        self.lm_head.weight = self.tok_emb.weight

        nn.init.xavier_uniform_(self.tok_emb.weight)
        nn.init.xavier_uniform_(self.q_prefix.weight)
        nn.init.zeros_(self.q_prefix.bias)

        with torch.no_grad():
            self.tok_emb.weight.data[StructuredTokenVocab.PAD].zero_()

    def _causal_mask(self, L: int, device) -> torch.Tensor:
        # subsequent mask: True means masked
        m = torch.triu(torch.ones((L, L), device=device, dtype=torch.bool), diagonal=1)
        return m

    def seq_logprob(self, enc_vec: torch.Tensor, seq_tok: torch.Tensor, pad_id: int = 0) -> torch.Tensor:
        """
        enc_vec: (B, H)
        seq_tok: (B, L)  target sequence tokens (includes END, may include PAD in tool slots)
        Return: (B,) average logprob over non-pad positions (excluding the first token prediction).
        """
        B, L = seq_tok.shape
        device = seq_tok.device

        # inputs: [prefix] + shift_right(seq_tok) where first input token is BOS
        bos = torch.full((B, 1), StructuredTokenVocab.BOS, dtype=torch.long, device=device)
        inp = torch.cat([bos, seq_tok[:, :-1]], dim=1)  # (B, L)

        x = self.tok_emb(inp)                           # (B, L, d)
        prefix = self.q_prefix(enc_vec).unsqueeze(1)    # (B, 1, d)
        x = torch.cat([prefix, x], dim=1)               # (B, 1+L, d)

        # causal over (1+L)
        attn_mask = self._causal_mask(1 + L, device=device)
        h = self.tr(x, mask=attn_mask)                  # (B, 1+L, d)
        h = self.ln(h)

        # predict tokens for positions 1..L based on h at those positions
        logits = self.lm_head(h[:, 1:, :])              # (B, L, V)

        # target is seq_tok
        logp = F.log_softmax(logits, dim=-1)            # (B, L, V)
        tgt = seq_tok.unsqueeze(-1)                     # (B, L, 1)
        lp = logp.gather(-1, tgt).squeeze(-1)           # (B, L)

        mask = (seq_tok != pad_id).float()
        denom = mask.sum(dim=1).clamp_min(1.0)
        return (lp * mask).sum(dim=1) / denom

    def score_candidates(self, enc_vec: torch.Tensor, cand_seq_tok: torch.Tensor) -> torch.Tensor:
        """
        enc_vec: (B,H)
        cand_seq_tok: (B,C,L) token sequences for candidate agents
        Return logits: (B,C) = avg logprob per candidate
        """
        B, C, L = cand_seq_tok.shape
        flat_seq = cand_seq_tok.reshape(B*C, L)
        flat_enc = enc_vec.unsqueeze(1).expand(B, C, enc_vec.size(-1)).reshape(B*C, enc_vec.size(-1))
        lp = self.seq_logprob(flat_enc, flat_seq, pad_id=StructuredTokenVocab.PAD)  # (B*C,)
        return lp.view(B, C)

    @torch.no_grad()
    def sample_topk_candidates(self, enc_vec: torch.Tensor, cand_seq_tok: torch.Tensor, topk: int) -> torch.Tensor:
        """
        Just rank candidates by score and return indices (B, K).
        """
        logits = self.score_candidates(enc_vec, cand_seq_tok)  # (B,C)
        K = min(topk, logits.size(1))
        return torch.topk(logits, k=K, dim=1).indices


# ---------------------- multi-positive softmax over candidate scores ----------------------

def multi_pos_softmax_loss_from_scores(
    scores: torch.Tensor,           # (B,C) candidate scores
    cand_is_pos: torch.Tensor,      # (B,C) bool
) -> torch.Tensor:
    """
    L = -log ( sum_{pos} exp(s) / sum_{all} exp(s) )
    """
    log_denom = torch.logsumexp(scores, dim=1)                    # (B,)
    scores_pos = scores.masked_fill(~cand_is_pos, float("-inf"))
    log_num = torch.logsumexp(scores_pos, dim=1)                  # (B,)

    has_pos = cand_is_pos.any(dim=1)
    if has_pos.sum().item() == 0:
        return scores.new_zeros(())
    loss = -(log_num - log_denom)
    return loss[has_pos].mean()


# ---------------------- GT reward (same vectorized, but on candidate list output) ----------------------

@torch.no_grad()
def reward_from_gt_vectorized_ids(
    pred_ids: torch.Tensor,     # (B,K) candidate agent indices (0..Na-1), PAD=-1
    gt_ids: torch.Tensor,       # (B,K) same, PAD=-1
    pad_id: int = -1,
    w_overlap: float = 0.5,
    w_ndcg: float = 0.5,
) -> torch.Tensor:
    device = pred_ids.device
    B, K = pred_ids.shape

    gt_mask = (gt_ids != pad_id)
    pred_mask = (pred_ids != pad_id)

    match = (pred_ids.unsqueeze(2) == gt_ids.unsqueeze(1)) & gt_mask.unsqueeze(1)
    hits = match.any(dim=2).float() * pred_mask.float()

    overlap = hits.sum(dim=1) / (pred_mask.float().sum(dim=1).clamp_min(1.0))

    discounts = 1.0 / torch.log2(torch.arange(K, device=device).float() + 2.0)
    dcg = (hits * discounts.unsqueeze(0)).sum(dim=1)

    gt_counts = gt_mask.float().sum(dim=1).clamp_min(0.0)
    ideal_k = torch.minimum(gt_counts, torch.tensor(float(K), device=device))
    cum_disc = torch.cumsum(discounts, dim=0)
    ideal_k_int = ideal_k.to(torch.long)
    idcg = torch.zeros((B,), device=device, dtype=torch.float32)
    pos = ideal_k_int - 1
    valid = ideal_k_int > 0
    idcg[valid] = cum_disc[pos[valid]]
    ndcg = torch.where(idcg > 0, dcg / idcg, torch.zeros_like(dcg))

    return w_overlap * overlap + w_ndcg * ndcg


# ---------------------- DPO trainer: list logprob under "sampling without replacement" on candidate scores ----------------------

class DPOTrainerCandidateList:
    """
    We model the list selection as sampling without replacement from softmax(scores/temp).
    Logprob of an ordered list is sum_t log softmax over remaining candidates.
    """
    def __init__(self, beta: float = 0.05):
        self.beta = beta

    def list_logprob(self, scores: torch.Tensor, seq_idx: torch.Tensor, pad_id: int = -1) -> torch.Tensor:
        """
        scores: (B,C)
        seq_idx: (B,K) indices into [0..C-1], pad=-1
        return (B,) avg logprob per non-pad item
        """
        B, C = scores.shape
        K = seq_idx.size(1)

        used = torch.zeros((B, C), dtype=torch.bool, device=scores.device)
        lp_sum = torch.zeros((B,), dtype=torch.float32, device=scores.device)
        cnt = torch.zeros((B,), dtype=torch.float32, device=scores.device)

        for t in range(K):
            idx = seq_idx[:, t]
            active = (idx != pad_id)
            if active.sum().item() == 0:
                continue

            s_t = scores.masked_fill(used, float("-inf"))
            log_denom = torch.logsumexp(s_t, dim=1)

            idx_safe = idx.clamp(0, C-1)
            log_num = s_t.gather(1, idx_safe.unsqueeze(1)).squeeze(1)
            logp = log_num - log_denom

            ok = active & torch.isfinite(logp)
            lp_sum = lp_sum + torch.where(ok, logp, torch.zeros_like(logp))
            cnt = cnt + ok.float()

            # mark used
            used = used | (torch.arange(C, device=scores.device).unsqueeze(0) == idx_safe.unsqueeze(1))

        cnt = cnt.clamp_min(1.0)
        return lp_sum / cnt

    def dpo_loss(self, scores_pi: torch.Tensor, scores_ref: torch.Tensor, pref: torch.Tensor, nonpref: torch.Tensor) -> torch.Tensor:
        lp_pref = self.list_logprob(scores_pi, pref)
        lp_nonp = self.list_logprob(scores_pi, nonpref)
        with torch.no_grad():
            lp_pref_ref = self.list_logprob(scores_ref, pref)
            lp_nonp_ref = self.list_logprob(scores_ref, nonpref)
        logratio = (lp_pref - lp_pref_ref) - (lp_nonp - lp_nonp_ref)
        return -F.logsigmoid(self.beta * logratio).mean()


# ---------------------- evaluation ----------------------

@torch.no_grad()
def evaluate_model(
    gen_model: StructuredSeqGenerator,
    enc_vecs_cpu: torch.Tensor,         # (Nq,H) on CPU
    qid2idx: Dict[str,int],
    a_ids: List[str],
    all_rankings: Dict[str, List[str]],
    eval_qids: List[str],
    device: torch.device,
    ks=(10,),
    cand_size: int = 200,
    rng_seed: int = 123,
    qid_to_part: Optional[Dict[str, str]] = None,
    agent_seq_tok: Optional[torch.Tensor] = None,   # (Na,L) on CPU
):
    assert agent_seq_tok is not None
    Kref = max(ks)
    agg = {k: {"P":0.0,"R":0.0,"F1":0.0,"Hit":0.0,"nDCG":0.0,"MRR":0.0} for k in ks}
    cnt = 0
    skipped = 0

    all_agent_set = set(a_ids)
    rnd = random.Random(rng_seed)

    pbar = tqdm(eval_qids, desc="Evaluating (structured-seq, sampled)", total=len(eval_qids))
    for i, qid in enumerate(pbar, start=1):
        k_pos = pos_topk_for_qid(qid, qid_to_part)
        gt_list = [aid for aid in all_rankings.get(qid, [])[:k_pos] if aid in all_agent_set]
        if not gt_list:
            skipped += 1
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

        # candidate agent indices
        aid2idxA = {aid: j for j, aid in enumerate(a_ids)}
        cand_aidx = []
        seen = set()
        for aid in cand_ids:
            j = aid2idxA.get(aid, None)
            if j is None or j in seen:
                continue
            seen.add(j)
            cand_aidx.append(j)
        if len(cand_aidx) < Kref:
            # pad with random agents to allow topK
            while len(cand_aidx) < Kref and len(cand_aidx) < len(a_ids):
                j = rnd.randrange(len(a_ids))
                if j not in seen:
                    seen.add(j)
                    cand_aidx.append(j)

        cand_aidx = cand_aidx[:cand_size]
        while len(cand_aidx) < cand_size:
            cand_aidx.append(-1)

        cand_aidx_t = torch.tensor([cand_aidx], dtype=torch.long, device=device)  # (1,C)

        # build cand sequences (1,C,L)
        # agent_seq_tok is CPU (Na,L)
        L = agent_seq_tok.size(1)
        cand_seq = torch.full((1, cand_size, L), StructuredTokenVocab.PAD, dtype=torch.long, device=device)
        for c in range(cand_size):
            j = cand_aidx[c]
            if j >= 0:
                cand_seq[0, c] = agent_seq_tok[j].to(device)

        qi = qid2idx[qid]
        enc = enc_vecs_cpu[qi:qi+1].to(device, non_blocking=True)

        scores = gen_model.score_candidates(enc, cand_seq)  # (1,C)
        topk_idx = torch.topk(scores, k=Kref, dim=1).indices[0].tolist()

        pred_ids = []
        for ci in topk_idx:
            j = cand_aidx[ci]
            if j >= 0:
                pred_ids.append(a_ids[j])

        md = evaluate_sampled(pred_ids, rel_set, ks)
        for k in ks:
            for m in md[k]:
                agg[k][m] += md[k][m]
        cnt += 1

        if cnt > 0 and (i % 50 == 0):
            ref = agg[Kref]
            pbar.set_postfix({
                "done": cnt, "skipped": skipped,
                f"P@{Kref}": f"{(ref['P']/cnt):.4f}",
                f"nDCG@{Kref}": f"{(ref['nDCG']/cnt):.4f}",
                f"MRR@{Kref}": f"{(ref['MRR']/cnt):.4f}",
            })

    if cnt == 0:
        return {k:{m:0.0 for m in ["P","R","F1","Hit","nDCG","MRR"]} for k in ks}
    for k in ks:
        for m in agg[k]:
            agg[k][m] /= cnt
    return agg


# ---------------------- main ----------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--max_features", type=int, default=TFIDF_MAX_FEATURES)

    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--skip_eval", type=int, default=0)

    ap.add_argument("--topk", type=int, default=EVAL_TOPK)
    ap.add_argument("--valid_ratio", type=float, default=0.2)
    ap.add_argument("--split_seed", type=int, default=42)

    # candidate building
    ap.add_argument("--train_mask", type=int, default=1)
    ap.add_argument("--candidate_size", type=int, default=200)
    ap.add_argument("--cand_extra", type=int, default=32)
    ap.add_argument("--rand_neg_ratio", type=float, default=0.25)
    ap.add_argument("--eval_candidate_size", type=int, default=200)

    # backbone/model
    ap.add_argument("--enc_dim", type=int, default=512)
    ap.add_argument("--enc_heads", type=int, default=8)
    ap.add_argument("--enc_layers", type=int, default=2)
    ap.add_argument("--enc_tokens", type=int, default=16)

    ap.add_argument("--tok_dim", type=int, default=256)
    ap.add_argument("--gen_heads", type=int, default=8)
    ap.add_argument("--gen_layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.1)

    # optimizer
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--amp", type=int, default=1)
    ap.add_argument("--enc_chunk", type=int, default=1024)
    ap.add_argument("--nn_jobs", type=int, default=4)

    # structured seq
    ap.add_argument("--max_tool_per_agent", type=int, default=8)

    # eval sampling
    ap.add_argument("--eval_per_part", type=int, default=200)
    ap.add_argument("--eval_parts", type=str, default="PartI,PartII,PartIII")

    # mode
    ap.add_argument("--mode", choices=["sft", "dpo"], default="sft")

    # DPO params
    ap.add_argument("--dpo_steps", type=int, default=1000)
    ap.add_argument("--dpo_batch", type=int, default=64)
    ap.add_argument("--beta", type=float, default=0.05)
    ap.add_argument("--dpo_lr", type=float, default=None)
    ap.add_argument("--freeze_q_enc_dpo", type=int, default=1)

    ap.add_argument("--gt_reward_w_overlap", type=float, default=0.5)
    ap.add_argument("--gt_reward_w_ndcg", type=float, default=0.5)
    ap.add_argument("--dpo_margin", type=float, default=0.01)

    ap.add_argument("--dpo_temp_a", type=float, default=0.7)
    ap.add_argument("--dpo_temp_b", type=float, default=1.3)

    ap.add_argument("--dpo_part_scope", type=str, default="ALL", choices=["ALL", "PartI", "PartII", "PartIII"])

    args = ap.parse_args()
    eval_parts = [x.strip() for x in args.eval_parts.split(",") if x.strip()]

    warn_if_topk_diff(args.topk, expected=EVAL_TOPK)
    set_global_seed(args.seed)

    device = torch.device(args.device)
    use_cuda = (device.type == "cuda")
    use_amp = bool(args.amp) and use_cuda
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    if args.candidate_size < args.topk:
        raise ValueError("candidate_size must be >= topk")
    if not (0.0 <= args.rand_neg_ratio <= 1.0):
        raise ValueError("--rand_neg_ratio must be in [0,1].")

    wsum = args.gt_reward_w_overlap + args.gt_reward_w_ndcg
    w_overlap = args.gt_reward_w_overlap / wsum
    w_ndcg = args.gt_reward_w_ndcg / wsum

    # ---------------- load data ----------------
    all_agents, all_questions, all_rankings, tools, qid_to_part = collect_data(args.data_root)
    q_ids, q_texts, a_ids, a_texts, _ = build_text_corpora(all_agents, all_questions, tools)
    qid2idx = {qid:i for i,qid in enumerate(q_ids)}
    aid2idxA = {aid:i for i,aid in enumerate(a_ids)}

    # ---------------- TF-IDF for retrieval ----------------
    q_vec = TfidfVectorizer(max_features=args.max_features, lowercase=True)
    Q_csr = q_vec.fit_transform(q_texts)
    A_in_Q_csr = q_vec.transform(a_texts)

    Qn = normalize(Q_csr.tocsr().astype(np.float32), norm="l2", axis=1, copy=True)
    Aq = normalize(A_in_Q_csr.tocsr().astype(np.float32), norm="l2", axis=1, copy=True)

    print("[retriever] fitting NearestNeighbors on agents (CPU)...")
    nbrsA = NearestNeighbors(
        n_neighbors=min(args.candidate_size + args.cand_extra, Aq.shape[0]),
        metric="cosine", algorithm="brute", n_jobs=args.nn_jobs
    ).fit(Aq)
    print("[retriever] done.")

    def retrieve_agent_topN_indices(qid_batch: List[str], k: int) -> np.ndarray:
        idx = np.array([qid2idx[q] for q in qid_batch], dtype=np.int64)
        k = min(k, Aq.shape[0])
        _, ind = nbrsA.kneighbors(Qn[idx], n_neighbors=k, return_distance=True)
        return ind  # (B,k)

    # ---------------- split ----------------
    qids_in_rank = [qid for qid in q_ids if qid in all_rankings]
    train_qids, valid_qids = stratified_split_by_part(qids_in_rank, qid_to_part, args.valid_ratio, args.split_seed)

    # ---------------- build query tensor ----------------
    def build_query_tensor(qid_batch: List[str]) -> torch.Tensor:
        idx = np.array([qid2idx[q] for q in qid_batch], dtype=np.int64)
        q_dense = Q_csr[idx].toarray().astype(np.float32)
        return torch.from_numpy(q_dense)

    # ---------------- load feature cache for llm/tool ids ----------------
    # you already have a bge cache pipeline; here we just reuse the stored id buffers
    cache_dir = os.path.join(args.data_root, ".cache", "shared", "features")
    # if你不是这个路径，改成你的feature_dir；也可以加参数
    # 这里为了“可跑”，你可以手动把 cache_dir 指到你生成的 bge_feature_dir
    # 我们做一个更宽松的策略：优先从 --feature_dir 环境变量读
    feature_dir = os.environ.get("FEATURE_DIR", None)
    if feature_dir is None:
        feature_dir = cache_dir
    cache = load_feature_cache(feature_dir)

    # align cache order to a_ids order
    cache_aid2i = {aid:i for i,aid in enumerate(cache.a_ids)}
    Na = len(a_ids)
    llm_idx = np.zeros((Na,), dtype=np.int64)
    tool_pad = np.zeros((Na, args.max_tool_per_agent), dtype=np.int64)
    tool_msk = np.zeros((Na, args.max_tool_per_agent), dtype=np.float32)

    missing = 0
    for i, aid in enumerate(a_ids):
        j = cache_aid2i.get(aid, None)
        if j is None:
            missing += 1
            continue
        llm_idx[i] = int(cache.agent_llm_idx[j])
        tp = np.asarray(cache.agent_tool_idx_padded[j], dtype=np.int64).reshape(-1)[:args.max_tool_per_agent]
        tm = np.asarray(cache.agent_tool_mask[j], dtype=np.float32).reshape(-1)[:args.max_tool_per_agent]
        tool_pad[i, :len(tp)] = tp
        tool_msk[i, :len(tm)] = tm
    if missing:
        print(f"[warn] {missing} agents missing in feature cache alignment.")

    tok_vocab = StructuredTokenVocab(cache.llm_vocab, cache.tool_id_vocab)
    agent_seq_np = build_agent_seq_tokens(llm_idx, tool_pad, tool_msk, tok_vocab, args.max_tool_per_agent)
    agent_seq_tok_cpu = torch.from_numpy(agent_seq_np).long()  # (Na,L) CPU

    # ---------------- models ----------------
    q_enc = QueryTransformerEncoder(
        d_in=Q_csr.shape[1],
        d_model=args.enc_dim,
        n_heads=args.enc_heads,
        n_layers=args.enc_layers,
        dropout=args.dropout,
        n_tokens=args.enc_tokens
    ).to(device)

    gen = StructuredSeqGenerator(
        enc_dim=args.enc_dim,
        vocab_size=tok_vocab.vocab_size,
        d_model=args.tok_dim,
        n_heads=args.gen_heads,
        n_layers=args.gen_layers,
        dropout=args.dropout
    ).to(device)

    # ---------------- targets (GT agent IDs -> GT indices) ----------------
    def build_targets(qids: List[str]) -> Tuple[List[str], List[List[int]]]:
        in_q, tgt = [], []
        for qid in qids:
            k_pos = min(args.topk, pos_topk_for_qid(qid, qid_to_part))
            ranked = [aid for aid in all_rankings.get(qid, [])[:k_pos] if aid in aid2idxA]
            if not ranked:
                continue
            idxs = [aid2idxA[a] for a in ranked]
            if len(idxs) < args.topk:
                idxs += [-1] * (args.topk - len(idxs))
            in_q.append(qid)
            tgt.append(idxs)
        return in_q, tgt

    train_q, train_tgt = build_targets(train_qids)
    valid_q, valid_tgt = build_targets(valid_qids)
    print(f"[train] sequences={len(train_q)} valid={len(valid_q)} topk={args.topk}")

    # ---------------- candidate builder (GT + hard + random) ----------------
    def build_candidate_aidx_batch(qid_batch: List[str], gt_aidx: torch.Tensor, use_retrieval: bool) -> torch.Tensor:
        """
        gt_aidx: (B,K) agent indices, PAD=-1
        return: (B,C) agent indices, PAD=-1
        """
        B = gt_aidx.size(0)
        C = args.candidate_size
        cand_list = []

        top_idx = None
        if use_retrieval:
            k_ret = min(C + args.cand_extra, len(a_ids))
            top_idx = retrieve_agent_topN_indices(qid_batch, k=k_ret)

        for i in range(B):
            gt = [int(x) for x in gt_aidx[i].tolist() if x >= 0]
            cand = []
            for j in gt:
                if j not in cand:
                    cand.append(j)
                    if len(cand) >= C:
                        break

            if len(cand) < C:
                rem = C - len(cand)
                rand_slots = int(round(rem * args.rand_neg_ratio)) if use_retrieval else rem
                hard_slots = rem - rand_slots

                if use_retrieval and hard_slots > 0:
                    for aidx in top_idx[i].tolist():
                        j = int(aidx)
                        if j in cand:
                            continue
                        cand.append(j)
                        hard_slots -= 1
                        if hard_slots <= 0 or len(cand) >= C:
                            break

                need = C - len(cand)
                if need > 0:
                    forbid = set(gt)
                    tries = 0
                    while need > 0 and tries < need * 50:
                        j = random.randrange(len(a_ids))
                        tries += 1
                        if j in forbid or j in cand:
                            continue
                        cand.append(j)
                        need -= 1

            cand = cand[:C]
            if len(cand) < C:
                cand += [-1] * (C - len(cand))
            cand_list.append(cand)

        return torch.tensor(cand_list, dtype=torch.long, device=device)

    def build_candidate_seq_batch(cand_aidx: torch.Tensor) -> torch.Tensor:
        """
        cand_aidx: (B,C) agent indices, PAD=-1
        return: (B,C,L) token seq
        """
        B, C = cand_aidx.shape
        L = agent_seq_tok_cpu.size(1)
        out = torch.full((B, C, L), StructuredTokenVocab.PAD, dtype=torch.long, device=device)
        for b in range(B):
            for c in range(C):
                j = int(cand_aidx[b, c].item())
                if j >= 0:
                    out[b, c] = agent_seq_tok_cpu[j].to(device)
        return out

    # ---------------- checkpoints ----------------
    data_sig = dataset_signature(a_ids, all_rankings)
    cache_root = ensure_cache_dir(args.data_root)
    model_dir = os.path.join(cache_root, "models"); os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"{filename}_{data_sig}.pt")

    def part_suffix() -> str:
        return "" if args.dpo_part_scope == "ALL" else f"_{args.dpo_part_scope}"

    def build_optimizer(params, lr):
        return torch.optim.AdamW(params, lr=lr, weight_decay=args.weight_decay)

    @torch.no_grad()
    def encode_queries_in_chunks(qid_list: List[str], chunk: int) -> torch.Tensor:
        outs = []
        q_enc.eval()
        for i in range(0, len(qid_list), chunk):
            qb = qid_list[i:i+chunk]
            qx = build_query_tensor(qb).to(device, non_blocking=True)
            enc = q_enc(qx).cpu()
            outs.append(enc)
        return torch.cat(outs, dim=0)

    # ===================== SFT =====================
    if args.mode == "sft":
        params = [p for p in list(q_enc.parameters()) + list(gen.parameters()) if p.requires_grad]
        opt = build_optimizer(params, lr=args.lr)

        nb = math.ceil(len(train_q) / args.batch_size)
        for epoch in range(1, args.epochs + 1):
            order = list(range(len(train_q)))
            random.shuffle(order)
            total_loss = 0.0

            pbar = tqdm(range(nb), desc=f"Epoch {epoch}/{args.epochs} [SFT structured-seq]")
            for b in pbar:
                sl = order[b*args.batch_size:(b+1)*args.batch_size]
                if not sl:
                    continue

                qid_batch = [train_q[i] for i in sl]
                gt = torch.tensor([train_tgt[i] for i in sl], dtype=torch.long, device=device)  # (B,K)

                q_x = build_query_tensor(qid_batch).to(device, non_blocking=True)
                cand_aidx = build_candidate_aidx_batch(qid_batch, gt, use_retrieval=bool(args.train_mask))  # (B,C)
                cand_seq = build_candidate_seq_batch(cand_aidx)  # (B,C,L)

                # pos mask over candidates
                # candidate is positive if its agent idx is in gt set (ignoring -1)
                gt_set = []
                for irow in range(gt.size(0)):
                    gt_set.append(set([int(x) for x in gt[irow].tolist() if x >= 0]))
                cand_is_pos = torch.zeros((gt.size(0), cand_aidx.size(1)), dtype=torch.bool, device=device)
                for irow in range(gt.size(0)):
                    for c in range(cand_aidx.size(1)):
                        j = int(cand_aidx[irow, c].item())
                        if j >= 0 and j in gt_set[irow]:
                            cand_is_pos[irow, c] = True

                opt.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=use_amp):
                    enc = q_enc(q_x)                              # (B,H)
                    scores = gen.score_candidates(enc, cand_seq)  # (B,C)
                    loss = multi_pos_softmax_loss_from_scores(scores, cand_is_pos)

                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                scaler.step(opt)
                scaler.update()

                total_loss += float(loss.detach().cpu())
                pbar.set_postfix({"loss": f"{loss.item():.4f}", "avg": f"{total_loss/(b+1):.4f}"})

            print(f"Epoch {epoch}: avg loss={total_loss/max(1,nb):.4f}")

        ckpt = {
            "mode": "sft_structured_seq",
            "q_enc": q_enc.state_dict(),
            "gen": gen.state_dict(),
            "data_sig": data_sig,
            "saved_at": datetime.now().isoformat(timespec="seconds"),
            "tok_vocab": {
                "llm_vocab": cache.llm_vocab,
                "tool_vocab": cache.tool_id_vocab,
                "vocab_size": tok_vocab.vocab_size,
                "max_tool_per_agent": args.max_tool_per_agent,
            }
        }
        torch.save(ckpt, model_path)
        print(f"[save] model -> {model_path}")

        if not bool(args.skip_eval):
            print("[eval] encoding all queries...")
            enc_all = encode_queries_in_chunks(q_ids, chunk=args.enc_chunk)

            eval_qids = sample_qids_by_part(valid_qids, qid_to_part, args.eval_per_part, args.seed, eval_parts)
            print(f"[eval] valid={len(valid_qids)} -> sampled={len(eval_qids)} per_part={args.eval_per_part}")

            m = evaluate_model(
                gen_model=gen,
                enc_vecs_cpu=enc_all,
                qid2idx=qid2idx,
                a_ids=a_ids,
                all_rankings=all_rankings,
                eval_qids=eval_qids,
                device=device,
                ks=(args.topk,),
                cand_size=args.eval_candidate_size,
                rng_seed=args.seed,
                qid_to_part=qid_to_part,
                agent_seq_tok=agent_seq_tok_cpu,
            )
            print_metrics_table("Validation (SFT structured-seq)", m, ks=(args.topk,), filename=filename)

    # ===================== DPO =====================
    if args.mode == "dpo":
        assert os.path.exists(model_path), f"SFT checkpoint not found: {model_path}"
        ckpt = torch.load(model_path, map_location=device)
        q_enc.load_state_dict(ckpt["q_enc"])
        gen.load_state_dict(ckpt["gen"])

        # ref snapshot
        gen_ref = copy.deepcopy(gen).to(device)
        gen_ref.eval()
        for p in gen_ref.parameters():
            p.requires_grad = False

        dpo_tr = DPOTrainerCandidateList(beta=args.beta)
        scaler_dpo = torch.cuda.amp.GradScaler(enabled=use_amp)

        # scope
        if args.dpo_part_scope == "ALL":
            dpo_q = train_q
            dpo_tgt = train_tgt
            dpo_eval_parts = eval_parts
            print(f"[DPO] scope=ALL train_n={len(dpo_q)}")
        else:
            scope = args.dpo_part_scope
            dpo_q = [qid for qid in train_q if qid_to_part.get(qid, "Unknown") == scope]
            dpo_tgt = [train_tgt[i] for i, qid in enumerate(train_q) if qid_to_part.get(qid, "Unknown") == scope]
            dpo_eval_parts = [scope]
            print(f"[DPO] scope={scope} train_n={len(dpo_q)}")

        assert len(dpo_q) > 0

        dpo_rng = random.Random(args.seed + 20250101)

        if args.dpo_part_scope == "ALL":
            buckets = defaultdict(list)
            for idx, qid in enumerate(dpo_q):
                buckets[qid_to_part.get(qid, "Unknown")].append(idx)
            prefer = ["PartI","PartII","PartIII"]
            parts = [p for p in prefer if p in buckets] + [p for p in sorted(buckets.keys()) if p not in prefer]

            def sample_indices(bs: int) -> List[int]:
                P = max(1, len(parts))
                base = bs // P
                rem = bs - base * P
                extra = [0]*P
                for j in dpo_rng.sample(range(P), rem) if rem>0 else []:
                    extra[j]+=1
                out=[]
                for j,p in enumerate(parts):
                    pool=buckets[p]
                    take=base+extra[j]
                    if len(pool)>=take:
                        out.extend(dpo_rng.sample(pool,take))
                    else:
                        out.extend(dpo_rng.sample(pool,len(pool)))
                        while take-len(pool)>0:
                            out.append(dpo_rng.choice(pool))
                            take-=1
                dpo_rng.shuffle(out)
                return out[:bs]
        else:
            def sample_indices(bs: int) -> List[int]:
                n=len(dpo_q)
                if n>=bs:
                    return dpo_rng.sample(range(n), bs)
                return [dpo_rng.randrange(n) for _ in range(bs)]

        if args.freeze_q_enc_dpo:
            for p in q_enc.parameters():
                p.requires_grad = False
            q_enc.eval()
            print("[DPO] frozen q_enc")
        else:
            q_enc.train()

        lr_dpo = args.dpo_lr if args.dpo_lr is not None else args.lr * 0.1
        opt_dpo = build_optimizer([p for p in gen.parameters() if p.requires_grad], lr=lr_dpo)

        update_steps = 0
        skipped_keep = 0

        for step in tqdm(range(args.dpo_steps), desc=f"DPO structured-seq{part_suffix()}"):
            sl = sample_indices(args.dpo_batch)
            batch_q = [dpo_q[i] for i in sl]
            gt = torch.tensor([dpo_tgt[i] for i in sl], dtype=torch.long, device=device)  # (B,K)

            cand_aidx = build_candidate_aidx_batch(batch_q, gt, use_retrieval=bool(args.train_mask))  # (B,C)
            cand_seq = build_candidate_seq_batch(cand_aidx)  # (B,C,L)
            q_x = build_query_tensor(batch_q).to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                enc = q_enc(q_x)
                scores = gen.score_candidates(enc, cand_seq)           # (B,C)
                scores_ref = gen_ref.score_candidates(enc, cand_seq)   # (B,C)

                # sample two lists from scores with two temps
                def sample_list(scores_, temp, K):
                    B,C = scores_.shape
                    K = min(K, C)
                    probs = F.softmax(scores_ / temp, dim=1)
                    idx = torch.multinomial(probs, num_samples=K, replacement=False)  # (B,K)
                    return idx

                seq_a = sample_list(scores, args.dpo_temp_a, args.topk)
                seq_b = sample_list(scores, args.dpo_temp_b, args.topk)

            # convert list indices -> agent indices
            def gather_agent_idx(seq_idx):
                # seq_idx (B,K) over candidate positions -> agent index in [0..Na-1], pad=-1
                B,K = seq_idx.shape
                out = torch.full((B,K), -1, dtype=torch.long, device=seq_idx.device)
                for b in range(B):
                    for t in range(K):
                        c = int(seq_idx[b,t].item())
                        j = int(cand_aidx[b,c].item())
                        out[b,t] = j if j>=0 else -1
                return out

            with torch.no_grad():
                pred_a = gather_agent_idx(seq_a)
                pred_b = gather_agent_idx(seq_b)
                # gt already agent idx list with -1 pad
                r_a = reward_from_gt_vectorized_ids(pred_a, gt, pad_id=-1, w_overlap=w_overlap, w_ndcg=w_ndcg)
                r_b = reward_from_gt_vectorized_ids(pred_b, gt, pad_id=-1, w_overlap=w_overlap, w_ndcg=w_ndcg)

                prefer_a = (r_a >= r_b)
                keep = (r_a - r_b).abs() >= float(args.dpo_margin)
                if keep.sum().item() == 0:
                    skipped_keep += 1
                    continue

            pref = torch.where(prefer_a.unsqueeze(1), seq_a, seq_b)[keep]
            nonp = torch.where(prefer_a.unsqueeze(1), seq_b, seq_a)[keep]

            scores_keep = scores[keep]
            scores_ref_keep = scores_ref[keep]

            gen.train()
            opt_dpo.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                loss = dpo_tr.dpo_loss(scores_keep, scores_ref_keep, pref, nonp)

            scaler_dpo.scale(loss).backward()
            scaler_dpo.unscale_(opt_dpo)
            torch.nn.utils.clip_grad_norm_(gen.parameters(), 1.0)
            scaler_dpo.step(opt_dpo)
            scaler_dpo.update()

            update_steps += 1
            if update_steps % 100 == 0:
                tqdm.write(f"[DPO] update={update_steps} loss={loss.item():.4f} skipped_keep={skipped_keep}")

        out_path = model_path.replace(".pt", f"_dpo{part_suffix()}.pt")
        torch.save({
            "mode": f"sft+dpo_structured_seq{part_suffix()}",
            "dpo_part_scope": args.dpo_part_scope,
            "q_enc": q_enc.state_dict(),
            "gen": gen.state_dict(),
            "data_sig": data_sig,
            "saved_at": datetime.now().isoformat(timespec="seconds")
        }, out_path)
        print(f"[save] dpo model -> {out_path}")

        if not bool(args.skip_eval):
            print("[eval] encoding all queries...")
            enc_all = encode_queries_in_chunks(q_ids, chunk=args.enc_chunk)

            if args.dpo_part_scope == "ALL":
                eval_qids = sample_qids_by_part(valid_qids, qid_to_part, args.eval_per_part, args.seed, eval_parts)
            else:
                scope = args.dpo_part_scope
                valid_scope = [q for q in valid_qids if qid_to_part.get(q, "Unknown") == scope]
                rnd = random.Random(args.seed)
                rnd.shuffle(valid_scope)
                eval_qids = valid_scope[: min(args.eval_per_part, len(valid_scope))]

            m = evaluate_model(
                gen_model=gen,
                enc_vecs_cpu=enc_all,
                qid2idx=qid2idx,
                a_ids=a_ids,
                all_rankings=all_rankings,
                eval_qids=eval_qids,
                device=device,
                ks=(args.topk,),
                cand_size=args.eval_candidate_size,
                rng_seed=args.seed,
                qid_to_part=qid_to_part,
                agent_seq_tok=agent_seq_tok_cpu,
            )
            print_metrics_table(f"Validation (SFT+DPO structured-seq{part_suffix()})", m, ks=(args.topk,), filename=filename)


if __name__ == "__main__":
    main()
