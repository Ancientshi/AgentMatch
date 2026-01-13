#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OneRec++ (Query->Agent, one-step generative ranking)

New task definition:
  ✅ No sessions (true query->agent)
  ✅ Candidate-only scoring: logits (B,C) over candidate agents
  ✅ Training = multi-positive one-step softmax (orderless positives)
      - PartI: top-10 are positives
      - PartII: top-1 is positive
      - PartIII: top-5 are positives
  ✅ Inference = one-step, return prob top-10 (no autoregressive decoding)
  ✅ CPU TF-IDF retrieval for candidate set + mixed negatives (hard + random)
  ✅ Auxiliary multi-positive InfoNCE (in-batch negatives) for stronger retrieval space
  ✅ Optional: init agent token embeddings from BGE content + (llm_id, tool_id) IDs
  ✅ DPO: balanced sampling across parts + GT-based reward (overlap + nDCG@K) by default
     - DPO uses the same one-step distribution and "sampling without replacement" lists

NEW (your request):
  ✅ DPO can run in 3 "part-specific" modes (controlled by --dpo_part_scope):
      1) finetune on PartI and eval only on PartI
      2) finetune on PartII and eval only on PartII
      3) finetune on PartIII and eval only on PartIII
  ✅ Saved ckpt filename will include suffix _PartI/_PartII/_PartIII accordingly.

Dataset:
  {data_root}/PartI|PartII|PartIII/{agents,questions,rankings}/merge.json
  {data_root}/Tools/merge.json

Optional BGE feature init:
  --bge_feature_dir path/to/cache
"""

from agent_rec.features import load_feature_cache, build_agent_content_view
from agent_rec.config import (
    EVAL_TOPK,
    TFIDF_MAX_FEATURES,
    pos_topk_for_qid,
)
from agent_rec.data import stratified_train_valid_split
from agent_rec.run_common import set_global_seed, warn_if_topk_diff

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

from utils import print_metrics_table


filename = os.path.splitext(os.path.basename(__file__))[0]


# ---------------------- helpers ----------------------

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

def build_q_vectorizer(q_texts, max_features: int):
    q_vec = TfidfVectorizer(max_features=max_features, lowercase=True)
    Q_csr = q_vec.fit_transform(q_texts)
    return q_vec, Q_csr

def stratified_split_by_part(
    qids: List[str],
    qid_to_part: Dict[str, str],
    valid_ratio: float,
    seed: int
) -> Tuple[List[str], List[str]]:
    return stratified_train_valid_split(qids, qid_to_part=qid_to_part, valid_ratio=valid_ratio, seed=seed)

def sample_qids_by_part(
    qids: List[str],
    qid_to_part: Dict[str, str],
    per_part: int,
    seed: int,
    parts: Optional[List[str]] = None
) -> List[str]:
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


# ---------------------- metrics ----------------------

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


# ---------------------- vocab ----------------------

class AgentVocab:
    def __init__(self, a_ids: List[str]):
        self.PAD = 0
        self.BOS = 1
        self.offset = 2
        self.a_ids = a_ids
        self.vocab_size = len(a_ids) + self.offset
        self.aid2tok = {aid: i + self.offset for i, aid in enumerate(a_ids)}
        self.tok2aid = {i + self.offset: aid for i, aid in enumerate(a_ids)}

    def aid_to_token(self, aid: str) -> int:
        return self.aid2tok[aid]

    def token_to_aid(self, tok: int) -> Optional[str]:
        if tok < self.offset:
            return None
        return self.tok2aid.get(tok, None)


# ---------------------- QueryEncoder ----------------------

class QueryEncoder(nn.Module):
    def __init__(self, d_q: int, hidden: int = 512, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_q, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
        )
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, q_vec: torch.Tensor) -> torch.Tensor:
        return self.net(q_vec)


# ---------------------- One-step Generator ----------------------

class OneStepGenerator(nn.Module):
    """
    One-step generative scoring over candidate agents.

    - Token embedding table: tok_emb (vocab_size, tok_dim) with PAD/BOS fixed to zeros.
    - Score(query, agent) = dot( q_proj(enc_vec), tok_emb(agent_token) )
    - Candidate-only logits: (B,C)

    Also includes a_proj (tok_dim -> enc_dim) for InfoNCE.
    """
    def __init__(self, enc_dim: int, vocab_size: int, tok_dim: int = 256, hidden: int = 512, dropout: float = 0.1):
        super().__init__()
        self.enc_dim = enc_dim
        self.vocab_size = vocab_size
        self.tok_dim = tok_dim

        self.tok_emb = nn.Embedding(vocab_size, tok_dim)
        self.q_proj = nn.Sequential(
            nn.Linear(enc_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, tok_dim, bias=False),
        )

        # for InfoNCE: tok_dim -> enc_dim
        self.a_proj = nn.Linear(tok_dim, enc_dim, bias=False)

        nn.init.xavier_uniform_(self.tok_emb.weight)
        for m in self.q_proj:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.xavier_uniform_(self.a_proj.weight)

        with torch.no_grad():
            self.tok_emb.weight.data[0].zero_()  # PAD
            self.tok_emb.weight.data[1].zero_()  # BOS

    def score(self, enc_vec: torch.Tensor, cand_tok: torch.Tensor) -> torch.Tensor:
        """
        enc_vec: (B, enc_dim)
        cand_tok: (B, C) full vocab tokens
        returns logits: (B, C)
        """
        q = self.q_proj(enc_vec)             # (B, tok_dim)
        Wc = self.tok_emb(cand_tok)          # (B, C, tok_dim)
        logits = torch.einsum("be,bce->bc", q, Wc)
        return logits

    @torch.no_grad()
    def generate(self, enc_vec: torch.Tensor, topk: int, cand_tok: torch.Tensor, temperature: float = 0.0):
        """
        One-step inference:
          - temperature=0: take topk by logits
          - temperature>0: sample topk without replacement from softmax(logits/temperature)
        Returns: (B, topk) tokens (PAD padded if topk > C)
        """
        was_training = self.training
        self.eval()
        try:
            B, C = cand_tok.shape
            K = min(topk, C)

            logits = self.score(enc_vec, cand_tok)                   # (B,C)
            invalid = (cand_tok < 2)

            if temperature and temperature > 0:
                logits2 = logits.masked_fill(invalid, -1e9)
                probs = F.softmax(logits2 / temperature, dim=-1)
                row_sum = probs.sum(dim=1, keepdim=True)
                probs = torch.where(row_sum > 0, probs / row_sum, torch.full_like(probs, 1.0 / C))
                idx = torch.multinomial(probs, num_samples=K, replacement=False)  # (B,K)
            else:
                logits2 = logits.masked_fill(invalid, float("-inf"))
                idx = torch.topk(logits2, k=K, dim=-1).indices                    # (B,K)

            out = cand_tok.gather(1, idx)                                         # (B,K)

            if K < topk:
                pad = torch.full((B, topk - K), 0, dtype=torch.long, device=out.device)
                out = torch.cat([out, pad], dim=1)

            return out
        finally:
            self.train(was_training)


# ---------------------- One-step multi-positive loss ----------------------

def multi_pos_softmax_loss(
    logits: torch.Tensor,        # (B, C) one-step logits over candidates
    cand_tok: torch.Tensor,      # (B, C)
    gt_tok: torch.Tensor,        # (B, K) padded positives (orderless)
    pad_tok: int = 0,
    offset: int = 2,
) -> torch.Tensor:
    """
    Multi-positive softmax (orderless):
      L = -log ( sum_{pos in P} exp(s_pos) / sum_{c in C} exp(s_c) )
    """
    gt_mask = (gt_tok != pad_tok) & (gt_tok >= offset)             # (B,K)
    match = (cand_tok.unsqueeze(1) == gt_tok.unsqueeze(2)) & gt_mask.unsqueeze(2)  # (B,K,C)
    pos_any = match.any(dim=1)                                      # (B,C)

    invalid = (cand_tok < offset)
    logits = logits.masked_fill(invalid, float("-inf"))

    log_denom = torch.logsumexp(logits, dim=1)                      # (B,)

    logits_pos = logits.masked_fill(~pos_any, float("-inf"))
    log_num = torch.logsumexp(logits_pos, dim=1)                    # (B,)

    has_pos = pos_any.any(dim=1)
    if has_pos.sum().item() == 0:
        return logits.new_zeros(())
    loss = -(log_num - log_denom)
    return loss[has_pos].mean()


# ---------------------- InfoNCE (multi-positive, in-batch negatives) ----------------------

def info_nce_multi_pos(
    q_emb: torch.Tensor,            # (B, H)
    pos_tok: torch.Tensor,          # (B, K) full-vocab tokens, PAD=0
    tok_emb: nn.Embedding,          # gen.tok_emb
    a_proj: nn.Linear,              # tok_dim -> H
    temperature: float = 0.07,
    pad_tok: int = 0,
):
    """
    In-batch negatives: pool = all positives from all samples in batch.
    Multi-positive for each query: numerator is logsumexp over its positives.
    """
    B, K = pos_tok.shape
    pos_mask = (pos_tok != pad_tok) & (pos_tok >= 2)     # (B,K)

    flat_tok = pos_tok.reshape(-1)
    flat_mask = pos_mask.reshape(-1)

    if flat_mask.sum().item() == 0:
        return q_emb.new_zeros(())

    pool_tok = flat_tok[flat_mask]                        # (M,)
    a_pool = a_proj(tok_emb(pool_tok))                    # (M,H)
    a_pool = F.normalize(a_pool, dim=-1)

    q = F.normalize(q_emb, dim=-1)                        # (B,H)
    logits = (q @ a_pool.t()) / temperature               # (B,M)

    match = (pos_tok.unsqueeze(-1) == pool_tok.view(1, 1, -1)) & pos_mask.unsqueeze(-1)
    pos_any = match.any(dim=1)                            # (B,M)

    has_pos = pos_any.any(dim=1)                          # (B,)
    if has_pos.sum().item() == 0:
        return q_emb.new_zeros(())

    log_denom = torch.logsumexp(logits, dim=1)            # (B,)
    logits_pos = logits.masked_fill(~pos_any, float("-inf"))
    log_num = torch.logsumexp(logits_pos, dim=1)          # (B,)

    loss = -(log_num - log_denom)
    loss = loss[has_pos].mean()
    return loss


# ---------------------- Optional: Agent token init (BGE + IDs) ----------------------

class AgentTokenComposer(nn.Module):
    """
    Build agent token embeddings from content vector + (llm_id emb + mean(tool_id emb)).
    """
    def __init__(self, d_content: int, num_tools: int, num_llm: int, tok_dim: int, id_dim: int = 256):
        super().__init__()
        self.emb_tool = nn.Embedding(num_tools, id_dim)
        self.emb_llm  = nn.Embedding(num_llm, id_dim)
        self.proj_content = nn.Linear(d_content, tok_dim, bias=False)
        self.proj_ids     = nn.Linear(id_dim * 2, tok_dim, bias=False)
        nn.init.xavier_uniform_(self.emb_tool.weight)
        nn.init.xavier_uniform_(self.emb_llm.weight)
        nn.init.xavier_uniform_(self.proj_content.weight)
        nn.init.xavier_uniform_(self.proj_ids.weight)

    def forward(self, A_content: torch.Tensor, tool_idx_pad: torch.LongTensor, tool_mask: torch.FloatTensor, llm_idx: torch.LongTensor):
        te = self.emb_tool(tool_idx_pad)                 # (Na,T,id_dim)
        m  = tool_mask.unsqueeze(-1)                     # (Na,T,1)
        tool_mean = (te * m).sum(1) / (m.sum(1) + 1e-8)  # (Na,id_dim)
        llm_e = self.emb_llm(llm_idx)                    # (Na,id_dim)
        ids = torch.cat([llm_e, tool_mean], dim=-1)      # (Na,2*id_dim)
        e = self.proj_content(A_content) + self.proj_ids(ids)
        e = F.normalize(e, dim=-1)
        return e

def init_tok_emb_from_bge_plus_ids(
    gen: OneStepGenerator,
    vocab: AgentVocab,
    a_ids: List[str],
    feature_dir: str,
    device: torch.device,
    freeze_tok: bool = False,
    max_tool_per_agent: int = 8,
    init_chunk: int = 50000,
    use_fp16: bool = True,
):
    cache = load_feature_cache(feature_dir)

    A_content_np = build_agent_content_view(
        cache=cache,
        use_model_content_vector=True,
        use_tool_content_vector=True,
    ).astype(np.float32)  # (Na_cache, d_content)

    cache_a_ids = cache.a_ids
    idx_map = {aid: i for i, aid in enumerate(cache_a_ids)}

    tool_pad = cache.agent_tool_idx_padded
    tool_msk = cache.agent_tool_mask
    llm_idx  = cache.agent_llm_idx

    Na = len(a_ids)
    d_content = A_content_np.shape[1]

    A_aligned = np.zeros((Na, d_content), dtype=np.float32)
    llm_idx_aligned = np.zeros((Na,), dtype=np.int64)

    T = int(max_tool_per_agent)
    tool_pad_aligned = np.zeros((Na, T), dtype=np.int64)
    tool_msk_aligned = np.zeros((Na, T), dtype=np.float32)

    missing = 0
    for i, aid in enumerate(a_ids):
        j = idx_map.get(aid, None)
        if j is None:
            missing += 1
            continue
        A_aligned[i] = A_content_np[j]
        llm_idx_aligned[i] = int(llm_idx[j])

        tp = np.asarray(tool_pad[j], dtype=np.int64).reshape(-1)[:T]
        tm = np.asarray(tool_msk[j], dtype=np.float32).reshape(-1)[:T]
        tool_pad_aligned[i, :len(tp)] = tp
        tool_msk_aligned[i, :len(tm)] = tm

    if missing:
        print(f"[warn] {missing} agents missing in cache alignment.")

    num_tools = len(cache.tool_id_vocab)
    num_llm   = len(cache.llm_vocab)
    tok_dim   = gen.tok_emb.embedding_dim

    composer = AgentTokenComposer(
        d_content=d_content, num_tools=num_tools, num_llm=num_llm, tok_dim=tok_dim, id_dim=256
    ).to(device)

    if freeze_tok:
        for p in gen.tok_emb.parameters():
            p.requires_grad = False

    was_training = gen.training
    gen.eval()
    composer.eval()
    dtype_ctx = torch.cuda.amp.autocast(enabled=(use_fp16 and device.type == "cuda"))

    try:
        with torch.no_grad():
            for s in tqdm(range(0, Na, init_chunk), desc="[tok-init] chunked", total=(Na + init_chunk - 1)//init_chunk):
                e = min(Na, s + init_chunk)

                A_t = torch.from_numpy(A_aligned[s:e]).to(device, non_blocking=True)
                tool_pad_t = torch.from_numpy(tool_pad_aligned[s:e]).to(device, non_blocking=True)
                tool_msk_t = torch.from_numpy(tool_msk_aligned[s:e]).to(device, non_blocking=True)
                llm_idx_t  = torch.from_numpy(llm_idx_aligned[s:e]).to(device, non_blocking=True)

                with dtype_ctx:
                    E_agents = composer(A_t, tool_pad_t, tool_msk_t, llm_idx_t)  # (chunk, tok_dim)

                gen.tok_emb.weight.data[vocab.offset + s : vocab.offset + e] = E_agents.to(gen.tok_emb.weight.dtype)
                gen.tok_emb.weight.data[vocab.PAD].zero_()
                gen.tok_emb.weight.data[vocab.BOS].zero_()

                del A_t, tool_pad_t, tool_msk_t, llm_idx_t, E_agents
                if device.type == "cuda":
                    torch.cuda.empty_cache()
    finally:
        gen.train(was_training)
        composer.train()

    print(f"[tok-init] done. tool_pad=T={T}, chunk={init_chunk}, fp16={use_fp16}")
    return composer


# ---------------------- GT reward (vectorized) ----------------------

@torch.no_grad()
def reward_from_gt_vectorized(
    pred_tok: torch.Tensor,     # (B, K)
    gt_tok: torch.Tensor,       # (B, K) padded, may include PAD
    pad_tok: int,
    offset: int,
    w_overlap: float = 0.5,
    w_ndcg: float = 0.5,
) -> torch.Tensor:
    """
    reward = w_overlap * overlap + w_ndcg * nDCG@K
    - overlap: hits / K (precision-like since pred length fixed K)
    - nDCG@K: based on whether each predicted token appears in GT set
    """
    device = pred_tok.device
    B, K = pred_tok.shape
    assert gt_tok.shape[0] == B and gt_tok.shape[1] == K

    gt_mask = (gt_tok != pad_tok) & (gt_tok >= offset)          # (B,K)
    pred_mask = (pred_tok != pad_tok) & (pred_tok >= offset)    # (B,K)

    match = (pred_tok.unsqueeze(2) == gt_tok.unsqueeze(1)) & gt_mask.unsqueeze(1)
    hits = match.any(dim=2).float() * pred_mask.float()         # (B,K)

    overlap = hits.sum(dim=1) / (pred_mask.float().sum(dim=1).clamp_min(1.0))  # (B,)

    discounts = 1.0 / torch.log2(torch.arange(K, device=device).float() + 2.0)  # (K,)
    dcg = (hits * discounts.unsqueeze(0)).sum(dim=1)                            # (B,)

    gt_counts = gt_mask.float().sum(dim=1).clamp_min(0.0)                       # (B,)
    ideal_k = torch.minimum(gt_counts, torch.tensor(float(K), device=device))   # (B,)
    cum_disc = torch.cumsum(discounts, dim=0)                                   # (K,)
    ideal_k_int = ideal_k.to(torch.long)
    idcg = torch.zeros((B,), device=device, dtype=torch.float32)
    pos = ideal_k_int - 1
    valid = ideal_k_int > 0
    idcg[valid] = cum_disc[pos[valid]]
    ndcg = torch.where(idcg > 0, dcg / idcg, torch.zeros_like(dcg))             # (B,)

    return w_overlap * overlap + w_ndcg * ndcg


# ---------------------- DPO (one-step, sampling w/o replacement) ----------------------

class DPOTrainerOneStep:
    """
    DPO logprob for a sampled ordered list under "sampling without replacement"
    from a fixed logits vector over candidates.

    This matches generate(... temperature>0) which does multinomial w/o replacement
    from softmax(logits/temperature) after masking already-chosen items.
    """
    def __init__(self, model: OneStepGenerator, ref_model: OneStepGenerator, beta: float = 0.1, pad_tok: int = 0, offset: int = 2):
        self.model = model
        self.ref_model = ref_model
        self.beta = beta
        self.pad_tok = pad_tok
        self.offset = offset

    def _seq_logprob_under_model(self, enc_vec: torch.Tensor, seq: torch.Tensor, cand_tok: torch.Tensor, model: OneStepGenerator) -> torch.Tensor:
        """
        enc_vec: (B,H)
        seq: (B,K) tokens (may include PAD at tail)
        cand_tok: (B,C) tokens
        returns: (B,) average logprob per non-pad token
        """
        B, K = seq.shape
        _, C = cand_tok.shape

        logits = model.score(enc_vec, cand_tok)  # (B,C)
        invalid = (cand_tok < self.offset)       # PAD/BOS etc.
        used = invalid.clone()

        seq_mask = (seq != self.pad_tok) & (seq >= self.offset)  # (B,K)

        lp_sum = torch.zeros((B,), device=seq.device, dtype=torch.float32)
        cnt = torch.zeros((B,), device=seq.device, dtype=torch.float32)

        for t in range(K):
            active = seq_mask[:, t]
            if active.sum().item() == 0:
                continue

            tok_t = seq[:, t]  # (B,)

            eq = (cand_tok == tok_t.unsqueeze(1))  # (B,C)
            has = eq.any(dim=1)
            idx = eq.float().argmax(dim=1)         # (B,)

            logits_t = logits.masked_fill(used, float("-inf"))  # (B,C)
            log_denom = torch.logsumexp(logits_t, dim=1)         # (B,)

            idx_safe = idx.clamp(0, C-1)
            log_num = logits_t.gather(1, idx_safe.unsqueeze(1)).squeeze(1)  # (B,)
            logp = log_num - log_denom

            ok = active & has & torch.isfinite(logp)
            lp_sum = lp_sum + torch.where(ok, logp, torch.zeros_like(logp))
            cnt = cnt + ok.float()

            used = used | eq

        cnt = cnt.clamp_min(1.0)
        return lp_sum / cnt

    def seq_logprob(self, enc_vec: torch.Tensor, seq: torch.Tensor, cand_tok: torch.Tensor, model=None) -> torch.Tensor:
        model = self.model if model is None else model
        return self._seq_logprob_under_model(enc_vec, seq, cand_tok, model)

    def dpo_loss(self, enc_vec: torch.Tensor, pref_seq: torch.Tensor, nonpref_seq: torch.Tensor, cand_tok: torch.Tensor) -> torch.Tensor:
        lp_pref = self.seq_logprob(enc_vec, pref_seq, cand_tok, model=self.model)
        lp_nonp = self.seq_logprob(enc_vec, nonpref_seq, cand_tok, model=self.model)
        with torch.no_grad():
            lp_pref_ref = self.seq_logprob(enc_vec, pref_seq, cand_tok, model=self.ref_model)
            lp_nonp_ref = self.seq_logprob(enc_vec, nonpref_seq, cand_tok, model=self.ref_model)

        logratio = (lp_pref - lp_pref_ref) - (lp_nonp - lp_nonp_ref)
        return -F.logsigmoid(self.beta * logratio).mean()


# ---------------------- Evaluation (candidate-only, one-step topK) ----------------------

@torch.no_grad()
def evaluate_model_gen(
    gen_model: OneStepGenerator,
    enc_vecs_cpu: torch.Tensor,         # (Nq,H) on CPU
    qid2idx: Dict[str,int],
    a_ids: List[str],
    all_rankings: Dict[str, List[str]],
    eval_qids: List[str],
    device: torch.device,
    ks=(10,),
    cand_size: int = 1000,
    rng_seed: int = 123,
    ref_k: Optional[int] = None,
    bar_update_every: int = 1,
    qid_to_part: Optional[Dict[str, str]] = None
):
    if ref_k is None:
        ref_k = max(ks)

    agg = {k: {"P":0.0,"R":0.0,"F1":0.0,"Hit":0.0,"nDCG":0.0,"MRR":0.0} for k in ks}
    cnt = 0
    skipped = 0

    all_agent_set = set(a_ids)
    rnd = random.Random(rng_seed)
    vocab = AgentVocab(a_ids)

    pbar = tqdm(eval_qids, desc="Evaluating (one-step, sampled)", total=len(eval_qids))
    for i, qid in enumerate(pbar, start=1):
        k_pos = pos_topk_for_qid(qid, qid_to_part)
        gt_list = [aid for aid in all_rankings.get(qid, [])[:k_pos] if aid in all_agent_set]
        if not gt_list:
            skipped += 1
            if (i % bar_update_every) == 0:
                done = cnt
                if cnt > 0:
                    ref = agg[ref_k]
                    pbar.set_postfix({
                        "done": done, "skipped": skipped,
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

        cand_tok_list = [vocab.aid_to_token(a) for a in cand_ids]
        seen = set()
        cand_tok_u = []
        for t in cand_tok_list:
            if t in seen:
                continue
            seen.add(t)
            cand_tok_u.append(t)

        while len(cand_tok_u) < max(ks):
            cand_tok_u.append(vocab.PAD)

        cand_tok_u = cand_tok_u[:cand_size] if len(cand_tok_u) > cand_size else cand_tok_u
        while len(cand_tok_u) < cand_size:
            cand_tok_u.append(vocab.PAD)

        cand_tok = torch.tensor([cand_tok_u], dtype=torch.long, device=device)

        qi = qid2idx[qid]
        enc = enc_vecs_cpu[qi:qi+1].to(device, non_blocking=True)

        pred_tok = gen_model.generate(enc, topk=max(ks), cand_tok=cand_tok, temperature=0.0)

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

    # shared
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--skip_eval", type=int, default=0)

    # task / eval
    ap.add_argument("--topk", type=int, default=EVAL_TOPK, help="Output top-K for inference and eval (keep 10).")
    ap.add_argument("--valid_ratio", type=float, default=0.2)
    ap.add_argument("--split_seed", type=int, default=42)

    # candidate building
    ap.add_argument("--train_mask", type=int, default=1,
                    help="1: use CPU TF-IDF retrieval as hard negatives; 0: random-only")
    ap.add_argument("--candidate_size", type=int, default=200, help="C for training (>=topk and >=max positives)")
    ap.add_argument("--cand_extra", type=int, default=32, help="extra neighbors retrieved before trimming")
    ap.add_argument("--rand_neg_ratio", type=float, default=0.25,
                    help="When train_mask=1: fraction of candidate slots (excluding GT) filled by random negatives [0..1]")
    ap.add_argument("--eval_candidate_size", type=int, default=200)

    # model
    ap.add_argument("--enc_dim", type=int, default=512)
    ap.add_argument("--tok_dim", type=int, default=256)
    ap.add_argument("--gen_hidden", type=int, default=512)
    ap.add_argument("--dropout", type=float, default=0.1)

    # InfoNCE
    ap.add_argument("--use_nce", type=int, default=1)
    ap.add_argument("--nce_lambda", type=float, default=0.1)
    ap.add_argument("--nce_temp", type=float, default=0.07)

    # optimizer / perf
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--amp", type=int, default=1)
    ap.add_argument("--enc_chunk", type=int, default=1024)
    ap.add_argument("--nn_jobs", type=int, default=4)

    # BGE init
    ap.add_argument("--bge_feature_dir", type=str, default=None)
    ap.add_argument("--init_tok_from_bge", type=int, default=1)
    ap.add_argument("--freeze_tok_emb", type=int, default=0)
    ap.add_argument("--max_tool_per_agent", type=int, default=8)
    ap.add_argument("--tok_init_chunk", type=int, default=50000)
    ap.add_argument("--tok_init_fp16", type=int, default=1)

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

    # default=1: stable "one-labor" GT-reward DPO
    ap.add_argument("--dpo_use_gt_reward", type=int, default=1,
                    help="1: use GT-based reward (overlap + nDCG) instead of learned RM during DPO (default=1)")

    # GT reward weights / margin
    ap.add_argument("--gt_reward_w_overlap", type=float, default=0.5)
    ap.add_argument("--gt_reward_w_ndcg", type=float, default=0.5)
    ap.add_argument("--dpo_margin", type=float, default=0.01,
                    help="Skip pairs with |r_a-r_b| < margin to avoid noisy updates")

    # DPO sampling temperatures
    ap.add_argument("--dpo_temp_a", type=float, default=0.7)
    ap.add_argument("--dpo_temp_b", type=float, default=1.3)

    # NEW: DPO part-specific scope (and matching eval scope)
    ap.add_argument(
        "--dpo_part_scope",
        type=str,
        default="ALL",
        choices=["ALL", "PartI", "PartII", "PartIII"],
        help="DPO training/eval scope. "
             "ALL: balanced sampler across parts (original). "
             "PartI/PartII/PartIII: finetune on that part only and eval only on that part."
    )

    args = ap.parse_args()
    eval_parts = [x.strip() for x in args.eval_parts.split(",") if x.strip()]

    warn_if_topk_diff(args.topk, expected=EVAL_TOPK)
    set_global_seed(args.seed)

    device = torch.device(args.device)
    use_cuda = (device.type == "cuda")
    use_amp = bool(args.amp) and use_cuda

    if args.candidate_size < args.topk:
        raise ValueError(f"candidate_size ({args.candidate_size}) must be >= topk ({args.topk}).")
    if not (0.0 <= args.rand_neg_ratio <= 1.0):
        raise ValueError("--rand_neg_ratio must be in [0,1].")
    if args.gt_reward_w_overlap + args.gt_reward_w_ndcg <= 0:
        raise ValueError("GT reward weights must sum to > 0.")

    wsum = args.gt_reward_w_overlap + args.gt_reward_w_ndcg
    w_overlap = args.gt_reward_w_overlap / wsum
    w_ndcg = args.gt_reward_w_ndcg / wsum

    # ---------------- load data ----------------
    all_agents, all_questions, all_rankings, tools, qid_to_part = collect_data(args.data_root)
    q_ids, q_texts, a_ids, a_texts, _ = build_text_corpora(all_agents, all_questions, tools)
    qid2idx = {qid:i for i,qid in enumerate(q_ids)}
    vocab = AgentVocab(a_ids)

    # ---------------- TF-IDF ----------------
    q_vec, Q_csr = build_q_vectorizer(q_texts, args.max_features)
    A_in_Q_csr = q_vec.transform(a_texts)  # (Na, Dq) sparse

    from scipy.sparse import issparse
    if not issparse(Q_csr) or not issparse(A_in_Q_csr):
        raise ValueError("Q_csr / A_in_Q_csr must be sparse matrices.")

    Qn = normalize(Q_csr.tocsr().astype(np.float32), norm="l2", axis=1, copy=True)
    Aq = normalize(A_in_Q_csr.tocsr().astype(np.float32), norm="l2", axis=1, copy=True)

    # agent retriever (CPU)
    print("[retriever] fitting NearestNeighbors on agents (CPU)...")
    nbrsA = NearestNeighbors(
        n_neighbors=min(args.candidate_size + args.cand_extra, Aq.shape[0]),
        metric="cosine",
        algorithm="brute",
        n_jobs=args.nn_jobs
    ).fit(Aq)
    print("[retriever] done.")

    def retrieve_agent_topN_indices(qid_batch: List[str], k: int) -> np.ndarray:
        idx = np.array([qid2idx[q] for q in qid_batch], dtype=np.int64)
        k = min(k, Aq.shape[0])
        _, ind = nbrsA.kneighbors(Qn[idx], n_neighbors=k, return_distance=True)
        return ind  # (B,k) agent indices

    # ---------------- split ----------------
    qids_in_rank = [qid for qid in q_ids if qid in all_rankings]
    train_qids, valid_qids = stratified_split_by_part(
        qids_in_rank,
        qid_to_part=qid_to_part,
        valid_ratio=args.valid_ratio,
        seed=args.split_seed
    )

    # ---------------- batch query tensor ----------------
    def build_query_tensor(qid_batch: List[str]) -> torch.Tensor:
        idx = np.array([qid2idx[q] for q in qid_batch], dtype=np.int64)
        q_dense = Q_csr[idx].toarray().astype(np.float32)   # (B, Dq) only batch densify
        return torch.from_numpy(q_dense)                    # CPU

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

    # ---------------- models ----------------
    q_enc = QueryEncoder(d_q=Q_csr.shape[1], hidden=args.enc_dim, dropout=args.dropout).to(device)
    gen = OneStepGenerator(
        enc_dim=args.enc_dim,
        vocab_size=vocab.vocab_size,
        tok_dim=args.tok_dim,
        hidden=args.gen_hidden,
        dropout=args.dropout
    ).to(device)

    if args.init_tok_from_bge and args.bge_feature_dir:
        _ = init_tok_emb_from_bge_plus_ids(
            gen, vocab, a_ids, args.bge_feature_dir, device,
            freeze_tok=bool(args.freeze_tok_emb),
            max_tool_per_agent=args.max_tool_per_agent,
            init_chunk=args.tok_init_chunk,
            use_fp16=bool(args.tok_init_fp16),
        )

    if args.freeze_tok_emb:
        for p in gen.tok_emb.parameters():
            p.requires_grad = False
        print("[tok-init] frozen gen.tok_emb")

    # ---------------- training targets ----------------
    def build_targets(qids: List[str]) -> Tuple[List[str], List[List[int]]]:
        """
        Always returns length=args.topk token list (PAD padded).
        Positives count varies by part via pos_topk_for_qid(qid, qid_to_part).
        """
        in_q = []
        tgt_tok = []
        for qid in qids:
            k_pos = min(args.topk, pos_topk_for_qid(qid, qid_to_part))
            ranked = [aid for aid in all_rankings.get(qid, [])[:k_pos] if aid in vocab.aid2tok]
            if not ranked:
                continue
            toks = [vocab.aid_to_token(a) for a in ranked]
            if len(toks) < args.topk:
                toks += [vocab.PAD] * (args.topk - len(toks))
            in_q.append(qid)
            tgt_tok.append(toks)
        return in_q, tgt_tok

    train_q, train_targets = build_targets(train_qids)
    valid_q, valid_targets = build_targets(valid_qids)
    print(f"[train] sequences={len(train_q)}  valid={len(valid_q)}  topk={args.topk}")

    # ---------------- candidate builder (GT + hard + random) ----------------
    def build_cand_tok_batch(
        qid_batch: List[str],
        tgt_tok: torch.Tensor,              # (B,topk) full-vocab targets (PAD padded)
        use_retrieval: bool
    ) -> torch.Tensor:
        """
        Return cand_tok: (B,C) full-vocab token ids, PAD padded.
        Always inserts ALL GT tokens first (unique).
        Then fills remaining with hard negatives (TF-IDF) and random negatives.
        """
        B = tgt_tok.size(0)
        C = args.candidate_size
        cand_list = []

        top_idx = None
        if use_retrieval:
            k_ret = min(C + args.cand_extra, len(a_ids))
            top_idx = retrieve_agent_topN_indices(qid_batch, k=k_ret)  # (B,k_ret)

        for i in range(B):
            gt = [t for t in tgt_tok[i].tolist() if t >= vocab.offset]
            cand = []
            for t in gt:
                if t not in cand:
                    cand.append(t)
                    if len(cand) >= C:
                        break

            if len(cand) < C:
                rem = C - len(cand)
                rand_slots = int(round(rem * args.rand_neg_ratio)) if use_retrieval else rem
                hard_slots = rem - rand_slots

                # hard negatives
                if use_retrieval and hard_slots > 0:
                    for aidx in top_idx[i].tolist():
                        t = vocab.offset + int(aidx)
                        if t in cand:
                            continue
                        cand.append(t)
                        hard_slots -= 1
                        if hard_slots <= 0 or len(cand) >= C:
                            break

                # random negatives
                need = C - len(cand)
                if need > 0:
                    forbid_agent = set([t - vocab.offset for t in gt if t >= vocab.offset])
                    tries = 0
                    while need > 0 and tries < need * 50:
                        aidx = random.randrange(len(a_ids))
                        tries += 1
                        if aidx in forbid_agent:
                            continue
                        t = vocab.offset + aidx
                        if t in cand:
                            continue
                        cand.append(t)
                        need -= 1

            if len(cand) < C:
                cand += [vocab.PAD] * (C - len(cand))
            else:
                cand = cand[:C]

            cand_list.append(cand)

        return torch.tensor(cand_list, dtype=torch.long, device=device)

    # ---------------- checkpoints ----------------
    data_sig = dataset_signature(a_ids, all_rankings)
    cache_dir = ensure_cache_dir(args.data_root)
    model_dir = os.path.join(cache_dir, "models"); os.makedirs(model_dir, exist_ok=True)

    # base SFT path (same as before)
    model_path = os.path.join(model_dir, f"{filename}_{data_sig}.pt")
    meta_path = os.path.join(model_dir, f"meta_{data_sig}.json")

    # suffix helper for DPO outputs
    def part_suffix() -> str:
        return "" if args.dpo_part_scope == "ALL" else f"_{args.dpo_part_scope}"

    # ---------------- optim / scaler ----------------
    def build_optimizer(params, lr: float):
        return torch.optim.AdamW(params, lr=lr, weight_decay=args.weight_decay)

    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # ===================== SFT =====================
    if args.mode == "sft":
        params = [p for p in list(q_enc.parameters()) + list(gen.parameters()) if p.requires_grad]
        opt = build_optimizer(params, lr=args.lr)

        nb = math.ceil(len(train_q) / args.batch_size)
        for epoch in range(1, args.epochs + 1):
            order = list(range(len(train_q)))
            random.shuffle(order)

            total_ce = 0.0
            total_nce = 0.0

            pbar = tqdm(range(nb), desc=f"Epoch {epoch}/{args.epochs} [SFT one-step]")
            for b in pbar:
                sl = order[b*args.batch_size:(b+1)*args.batch_size]
                if not sl:
                    continue

                qid_batch = [train_q[i] for i in sl]
                tgt = torch.tensor([train_targets[i] for i in sl], dtype=torch.long, device=device)  # (B,topk)

                q_x = build_query_tensor(qid_batch).to(device, non_blocking=True)  # (B,Dq)
                cand_tok = build_cand_tok_batch(qid_batch, tgt_tok=tgt, use_retrieval=bool(args.train_mask))  # (B,C)

                opt.zero_grad(set_to_none=True)

                with torch.cuda.amp.autocast(enabled=use_amp):
                    enc_vec = q_enc(q_x)                       # (B,H)
                    logits = gen.score(enc_vec, cand_tok)      # (B,C)

                    loss_ce = multi_pos_softmax_loss(
                        logits=logits,
                        cand_tok=cand_tok,
                        gt_tok=tgt,
                        pad_tok=vocab.PAD,
                        offset=vocab.offset,
                    )

                    if args.use_nce:
                        loss_nce = info_nce_multi_pos(
                            q_emb=enc_vec,
                            pos_tok=tgt,
                            tok_emb=gen.tok_emb,
                            a_proj=gen.a_proj,
                            temperature=args.nce_temp,
                            pad_tok=vocab.PAD,
                        )
                        loss = loss_ce + args.nce_lambda * loss_nce
                    else:
                        loss_nce = enc_vec.new_zeros(())
                        loss = loss_ce

                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                scaler.step(opt)
                scaler.update()

                total_ce += float(loss_ce.detach().cpu())
                total_nce += float(loss_nce.detach().cpu())

                pbar.set_postfix({
                    "ce": f"{loss_ce.item():.4f}",
                    "nce": f"{loss_nce.item():.4f}",
                    "avg_ce": f"{total_ce/(b+1):.4f}",
                    "avg_nce": f"{total_nce/(b+1):.4f}",
                    "C": cand_tok.size(1),
                })

            print(f"Epoch {epoch}: avg CE={total_ce/max(1,nb):.4f}  avg NCE={total_nce/max(1,nb):.4f}")

        # save
        ckpt = {
            "mode": "sft",
            "q_enc": q_enc.state_dict(),
            "gen": gen.state_dict(),
            "data_sig": data_sig,
            "saved_at": datetime.now().isoformat(timespec="seconds"),
            "dims": {
                "d_q": int(Q_csr.shape[1]),
                "enc_dim": int(args.enc_dim),
                "tok_dim": int(args.tok_dim),
                "vocab_size": int(vocab.vocab_size),
            }
        }
        torch.save(ckpt, model_path)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({"a_ids": a_ids, "q_ids": q_ids, "data_sig": data_sig}, f, ensure_ascii=False, indent=2)
        print(f"[save] model -> {model_path}\n[save] meta  -> {meta_path}")

        # eval
        if not bool(args.skip_eval):
            print("[eval] encoding all queries...")
            enc_all = encode_queries_in_chunks(q_ids, chunk=args.enc_chunk)  # CPU (Nq,H)

            eval_qids = sample_qids_by_part(
                valid_qids,
                qid_to_part=qid_to_part,
                per_part=args.eval_per_part,
                seed=args.seed,
                parts=eval_parts
            )
            print(f"[eval] valid_qids={len(valid_qids)} -> sampled_eval={len(eval_qids)} (per_part={args.eval_per_part})")

            valid_metrics = evaluate_model_gen(
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
            )
            print_metrics_table("Validation (SFT one-step, per-part sampled)", valid_metrics, ks=(args.topk,), filename=filename)

            for part in eval_parts:
                part_qids = [q for q in eval_qids if qid_to_part.get(q, "Unknown") == part]
                if not part_qids:
                    continue
                m_part = evaluate_model_gen(
                    gen_model=gen,
                    enc_vecs_cpu=enc_all,
                    qid2idx=qid2idx,
                    a_ids=a_ids,
                    all_rankings=all_rankings,
                    eval_qids=part_qids,
                    device=device,
                    ks=(args.topk,),
                    cand_size=args.eval_candidate_size,
                    rng_seed=args.seed + 17,
                    qid_to_part=qid_to_part,
                )
                print_metrics_table(f"Validation (SFT one-step) {part} (n={len(part_qids)})", m_part, ks=(args.topk,), filename=filename)

    # ===================== DPO =====================
    if args.mode == "dpo":
        # load SFT checkpoint
        assert os.path.exists(model_path), f"SFT checkpoint not found: {model_path}"
        ckpt = torch.load(model_path, map_location=device)

        q_enc.load_state_dict(ckpt["q_enc"])
        gen.load_state_dict(ckpt["gen"])

        # reference policy snapshot
        gen_ref = copy.deepcopy(gen).to(device)
        gen_ref.eval()
        for p in gen_ref.parameters():
            p.requires_grad = False

        # Only GT-reward DPO
        if not bool(args.dpo_use_gt_reward):
            raise ValueError("This script is configured for GT-reward DPO (set --dpo_use_gt_reward 1).")

        dpo = DPOTrainerOneStep(model=gen, ref_model=gen_ref, beta=args.beta, pad_tok=vocab.PAD, offset=vocab.offset)
        scaler_dpo = torch.cuda.amp.GradScaler(enabled=use_amp)

        # -------- build DPO train/valid by scope --------
        if args.dpo_part_scope == "ALL":
            # Use SAME filtered training set as SFT (only queries with valid GT)
            dpo_q = train_q
            dpo_tgt = train_targets
            # evaluation scope defaults to args.eval_parts (unchanged)
            dpo_eval_parts = eval_parts
            print(f"[DPO] scope=ALL (balanced across parts). train_n={len(dpo_q)}")
        else:
            scope = args.dpo_part_scope
            # filter on already-built (train_q, valid_q) which have valid GT
            dpo_q = [qid for qid in train_q if qid_to_part.get(qid, "Unknown") == scope]
            dpo_tgt = [train_targets[i] for i, qid in enumerate(train_q) if qid_to_part.get(qid, "Unknown") == scope]
            dpo_eval_parts = [scope]
            print(f"[DPO] scope={scope} (part-specific). train_n={len(dpo_q)}")

        assert len(dpo_q) == len(dpo_tgt) and len(dpo_q) > 0, \
            f"[DPO] no training samples found under scope={args.dpo_part_scope}. Check qid_to_part mapping."

        # -------- sampler --------
        dpo_rng = random.Random(args.seed + 20250101)

        if args.dpo_part_scope == "ALL":
            # strict balanced sampler across parts (original)
            dpo_part_buckets: Dict[str, List[int]] = defaultdict(list)
            for idx, qid in enumerate(dpo_q):
                part = qid_to_part.get(qid, "Unknown")
                dpo_part_buckets[part].append(idx)

            prefer_order = ["PartI", "PartII", "PartIII"]
            parts_all = list(dpo_part_buckets.keys())
            dpo_parts = [p for p in prefer_order if p in dpo_part_buckets] + [p for p in sorted(parts_all) if p not in prefer_order]

            def sample_indices(batch_size: int) -> List[int]:
                if batch_size <= 0:
                    return []
                P = max(1, len(dpo_parts))
                base = batch_size // P
                rem = batch_size - base * P

                extra = [0] * P
                for j in dpo_rng.sample(range(P), rem) if rem > 0 else []:
                    extra[j] += 1

                indices: List[int] = []
                for j, part in enumerate(dpo_parts):
                    pool = dpo_part_buckets.get(part, [])
                    take = base + extra[j]
                    if take <= 0 or len(pool) == 0:
                        continue
                    if len(pool) >= take:
                        indices.extend(dpo_rng.sample(pool, take))
                    else:
                        indices.extend(dpo_rng.sample(pool, len(pool)))
                        while len(pool) > 0 and (take - len(pool)) > 0:
                            indices.append(dpo_rng.choice(pool))
                            take -= 1

                if len(indices) < batch_size:
                    all_pool = [i for lst in dpo_part_buckets.values() for i in lst]
                    need = batch_size - len(indices)
                    if len(all_pool) > 0:
                        for _ in range(need):
                            indices.append(dpo_rng.choice(all_pool))

                if len(indices) > batch_size:
                    indices = dpo_rng.sample(indices, batch_size)

                dpo_rng.shuffle(indices)
                return indices
        else:
            # part-specific: simple uniform sampling from this part only
            def sample_indices(batch_size: int) -> List[int]:
                n = len(dpo_q)
                if n == 0 or batch_size <= 0:
                    return []
                if n >= batch_size:
                    return dpo_rng.sample(range(n), batch_size)
                # with replacement
                return [dpo_rng.randrange(n) for _ in range(batch_size)]

        # optionally freeze query encoder during DPO
        if args.freeze_q_enc_dpo:
            for p in q_enc.parameters():
                p.requires_grad = False
            q_enc.eval()
            print("[DPO] frozen q_enc for stability.")
        else:
            q_enc.train()

        lr_dpo = args.dpo_lr if args.dpo_lr is not None else args.lr * 0.1
        opt_dpo = build_optimizer([p for p in gen.parameters() if p.requires_grad], lr=lr_dpo)

        update_steps = 0
        skipped_keep = 0

        for step in tqdm(range(args.dpo_steps), desc=f"DPO (one-step{part_suffix()})"):
            sl = sample_indices(args.dpo_batch)
            batch_q = [dpo_q[i] for i in sl]
            gt_tok = torch.tensor([dpo_tgt[i] for i in sl], dtype=torch.long, device=device)  # (B,K)

            cand_tok = build_cand_tok_batch(batch_q, tgt_tok=gt_tok, use_retrieval=bool(args.train_mask))
            q_x = build_query_tensor(batch_q).to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                enc = q_enc(q_x)
                seq_a = gen.generate(enc, topk=args.topk, cand_tok=cand_tok, temperature=float(args.dpo_temp_a))
                seq_b = gen.generate(enc, topk=args.topk, cand_tok=cand_tok, temperature=float(args.dpo_temp_b))

            with torch.no_grad():
                r_a = reward_from_gt_vectorized(seq_a, gt_tok, pad_tok=vocab.PAD, offset=vocab.offset,
                                                w_overlap=w_overlap, w_ndcg=w_ndcg)
                r_b = reward_from_gt_vectorized(seq_b, gt_tok, pad_tok=vocab.PAD, offset=vocab.offset,
                                                w_overlap=w_overlap, w_ndcg=w_ndcg)

                prefer_a = (r_a >= r_b)
                keep = (r_a - r_b).abs() >= float(args.dpo_margin)

                if keep.sum().item() == 0:
                    skipped_keep += 1
                    if (step + 1) % 100 == 0:
                        tqdm.write(f"[DPO] step {step+1}: skip (keep=0) skipped_keep={skipped_keep} update_steps={update_steps}")
                    continue

            pref_seq = torch.where(prefer_a.unsqueeze(1), seq_a, seq_b)
            nonpref_seq = torch.where(prefer_a.unsqueeze(1), seq_b, seq_a)

            pref_seq = pref_seq[keep]
            nonpref_seq = nonpref_seq[keep]
            enc_keep = enc[keep]
            cand_keep = cand_tok[keep]

            gen.train()
            opt_dpo.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                loss = dpo.dpo_loss(enc_keep, pref_seq, nonpref_seq, cand_keep)

            scaler_dpo.scale(loss).backward()
            scaler_dpo.unscale_(opt_dpo)
            torch.nn.utils.clip_grad_norm_(gen.parameters(), 1.0)
            scaler_dpo.step(opt_dpo)
            scaler_dpo.update()

            update_steps += 1
            if update_steps % 100 == 0:
                tqdm.write(f"[DPO] update {update_steps} (step {step+1}/{args.dpo_steps}) loss={loss.item():.4f} skipped_keep={skipped_keep}")

        ckpt_out = {
            "mode": f"sft+dpo{part_suffix()}",
            "dpo_part_scope": args.dpo_part_scope,
            "q_enc": q_enc.state_dict(),
            "gen": gen.state_dict(),
            "data_sig": data_sig,
            "saved_at": datetime.now().isoformat(timespec="seconds")
        }
        out_path = model_path.replace(".pt", f"_dpo{part_suffix()}.pt")
        torch.save(ckpt_out, out_path)
        print(f"[save] dpo model -> {out_path}")

        # eval (scope-aware)
        if not bool(args.skip_eval):
            print("[eval] encoding all queries...")
            enc_all = encode_queries_in_chunks(q_ids, chunk=args.enc_chunk)

            # select eval_qids:
            # - If part-specific, evaluate ONLY that part on validation split
            # - If ALL, keep original behavior (sample per-part from valid_qids, using args.eval_parts)
            if args.dpo_part_scope == "ALL":
                eval_qids = sample_qids_by_part(
                    valid_qids,
                    qid_to_part=qid_to_part,
                    per_part=args.eval_per_part,
                    seed=args.seed,
                    parts=eval_parts
                )
                print(f"[eval] scope=ALL valid_qids={len(valid_qids)} -> sampled_eval={len(eval_qids)} (per_part={args.eval_per_part}) parts={eval_parts}")
            else:
                scope = args.dpo_part_scope
                valid_scope = [q for q in valid_qids if qid_to_part.get(q, "Unknown") == scope]
                # keep your sampling knob: take up to eval_per_part (for this single part)
                if args.eval_per_part > 0:
                    rnd = random.Random(args.seed)
                    rnd.shuffle(valid_scope)
                    eval_qids = valid_scope[: min(args.eval_per_part, len(valid_scope))]
                else:
                    eval_qids = valid_scope
                print(f"[eval] scope={scope} valid_scope={len(valid_scope)} -> eval_n={len(eval_qids)} (eval_per_part={args.eval_per_part})")

            valid_metrics = evaluate_model_gen(
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
            )
            print_metrics_table(f"Validation (SFT+DPO one-step{part_suffix()}, sampled)", valid_metrics, ks=(args.topk,), filename=filename)

            # If ALL, still print per-part breakdown as before; if part-specific, this is redundant so skip.
            if args.dpo_part_scope == "ALL":
                for part in eval_parts:
                    part_qids = [q for q in eval_qids if qid_to_part.get(q, "Unknown") == part]
                    if not part_qids:
                        continue
                    m_part = evaluate_model_gen(
                        gen_model=gen,
                        enc_vecs_cpu=enc_all,
                        qid2idx=qid2idx,
                        a_ids=a_ids,
                        all_rankings=all_rankings,
                        eval_qids=part_qids,
                        device=device,
                        ks=(args.topk,),
                        cand_size=args.eval_candidate_size,
                        rng_seed=args.seed + 17,
                        qid_to_part=qid_to_part,
                    )
                    print_metrics_table(f"Validation (SFT+DPO one-step) {part} (n={len(part_qids)})", m_part, ks=(args.topk,), filename=filename)


if __name__ == "__main__":
    main()
