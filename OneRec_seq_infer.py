#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Infer/Eval script for OneRec++ one-step model (Query->Agent).

What it does:
  - Rebuilds TF-IDF features exactly like training (fit on all questions).
  - Rebuilds vocab (sorted agent ids) to match ckpt token table.
  - Uses the same split logic (stratified by part).
  - Samples 200 queries per part from train + valid and prints metric tables.
  - Adds OOD diagnostics:
      * How much of predicted TopK are "train-seen GT agents"
      * Unseen-valid-GT coverage and Hit@K on queries whose GT is unseen.
"""

import os, json, math, argparse, random, zlib
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
from scipy.sparse import issparse

from agent_rec.config import TFIDF_MAX_FEATURES, EVAL_TOPK, pos_topk_for_qid
from agent_rec.data import stratified_train_valid_split
from agent_rec.run_common import set_global_seed, warn_if_topk_diff

try:
    from utils import print_metrics_table
except Exception:
    # fallback if your utils import path changes
    def print_metrics_table(title, metrics, ks=(10,), filename=""):
        print(f"\n==== {title} ====")
        for k in ks:
            m = metrics[k]
            print(f"K={k} | P={m['P']:.4f} R={m['R']:.4f} F1={m['F1']:.4f} "
                  f"Hit={m['Hit']:.4f} nDCG={m['nDCG']:.4f} MRR={m['MRR']:.4f}")

# ---------------------- json / data ----------------------

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
    for aid in a_ids:
        a = all_agents[aid]
        mname = a.get("M", {}).get("name", "")
        tool_list = a.get("T", {}).get("tools", []) or []
        concat_tool_desc = " ".join([_tool_text(tn) for tn in tool_list])
        text = f"{mname} {concat_tool_desc}".strip()
        a_texts.append(text)

    return q_ids, q_texts, a_ids, a_texts

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

# ---------------------- vocab / model defs ----------------------

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

class OneStepGenerator(nn.Module):
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
        self.a_proj = nn.Linear(tok_dim, enc_dim, bias=False)

        nn.init.xavier_uniform_(self.tok_emb.weight)
        for m in self.q_proj:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.xavier_uniform_(self.a_proj.weight)

        with torch.no_grad():
            self.tok_emb.weight.data[0].zero_()
            self.tok_emb.weight.data[1].zero_()

    def score(self, enc_vec: torch.Tensor, cand_tok: torch.Tensor) -> torch.Tensor:
        q = self.q_proj(enc_vec)             # (B, tok_dim)
        Wc = self.tok_emb(cand_tok)          # (B, C, tok_dim)
        logits = torch.einsum("be,bce->bc", q, Wc)
        return logits

    @torch.no_grad()
    def generate(self, enc_vec: torch.Tensor, topk: int, cand_tok: torch.Tensor, temperature: float = 0.0):
        was_training = self.training
        self.eval()
        try:
            B, C = cand_tok.shape
            K = min(topk, C)

            logits = self.score(enc_vec, cand_tok)
            invalid = (cand_tok < 2)

            logits2 = logits.masked_fill(invalid, float("-inf"))
            idx = torch.topk(logits2, k=K, dim=-1).indices
            out = cand_tok.gather(1, idx)

            if K < topk:
                pad = torch.full((B, topk - K), 0, dtype=torch.long, device=out.device)
                out = torch.cat([out, pad], dim=1)
            return out
        finally:
            self.train(was_training)

# ---------------------- sampling helpers ----------------------

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

def build_train_seen_agent_set(
    qids: List[str],
    all_rankings: Dict[str, List[str]],
    qid_to_part: Dict[str, str],
    topk_eval: int
) -> set:
    s = set()
    for qid in qids:
        k_pos = min(topk_eval, pos_topk_for_qid(qid, qid_to_part))
        gt = all_rankings.get(qid, [])[:k_pos]
        s.update(gt)
    return s

# ---------------------- eval with diagnostics ----------------------

@torch.no_grad()
def eval_with_diag(
    gen: OneStepGenerator,
    q_enc: QueryEncoder,
    Q_csr,
    q_ids: List[str],
    qid2idx: Dict[str,int],
    a_ids: List[str],
    all_rankings: Dict[str, List[str]],
    eval_qids: List[str],
    qid_to_part: Dict[str, str],
    train_seen_agents: set,
    device: torch.device,
    topk: int,
    cand_size: int,
    rng_seed: int,
):
    vocab = AgentVocab(a_ids)
    all_agent_set = set(a_ids)
    rnd = random.Random(rng_seed)

    ks = (topk,)
    agg = {k: {"P":0.0,"R":0.0,"F1":0.0,"Hit":0.0,"nDCG":0.0,"MRR":0.0} for k in ks}
    cnt = 0
    skipped = 0

    # diagnostics
    frac_pred_in_train_seen_sum = 0.0
    all_pred_in_train_seen_cnt = 0
    unique_pred = set()

    # unseen-valid-GT diagnostics (query-level)
    unseen_gt_query_cnt = 0
    unseen_gt_hit_cnt = 0
    unseen_gt_pred_cover_sum = 0.0  # fraction of preds in unseen_gt_set, averaged over those queries

    # precompute query encodings on the fly (batch size 1, cheap enough)
    q_enc.eval()
    gen.eval()

    pbar = tqdm(eval_qids, desc="Eval", total=len(eval_qids))
    for qid in pbar:
        part = qid_to_part.get(qid, "Unknown")
        k_pos = min(topk, pos_topk_for_qid(qid, qid_to_part))
        gt_list = [aid for aid in all_rankings.get(qid, [])[:k_pos] if aid in all_agent_set]
        if not gt_list:
            skipped += 1
            continue

        rel_set = set(gt_list)

        # build sampled candidate pool (ensure GT included)
        neg_pool = list(all_agent_set - rel_set)
        need_neg = max(0, cand_size - len(gt_list))
        if need_neg > 0 and len(neg_pool) > 0:
            k = min(need_neg, len(neg_pool))
            sampled_negs = rnd.sample(neg_pool, k)
            cand_ids = gt_list + sampled_negs
        else:
            cand_ids = gt_list

        # unique + pad to cand_size
        cand_tok_list = [vocab.aid_to_token(a) for a in cand_ids]
        seen = set()
        cand_tok_u = []
        for t in cand_tok_list:
            if t in seen:
                continue
            seen.add(t)
            cand_tok_u.append(t)

        while len(cand_tok_u) < cand_size:
            cand_tok_u.append(vocab.PAD)
        cand_tok_u = cand_tok_u[:cand_size]
        cand_tok = torch.tensor([cand_tok_u], dtype=torch.long, device=device)

        # encode q (densify only this row)
        qi = qid2idx[qid]
        q_dense = Q_csr[qi:qi+1].toarray().astype(np.float32)
        q_x = torch.from_numpy(q_dense).to(device, non_blocking=True)

        enc = q_enc(q_x)
        pred_tok = gen.generate(enc, topk=topk, cand_tok=cand_tok, temperature=0.0)

        pred_ids = []
        for t in pred_tok[0].tolist():
            aid = vocab.token_to_aid(t)
            if aid is not None:
                pred_ids.append(aid)

        for a in pred_ids:
            unique_pred.add(a)

        # metrics
        md = evaluate_sampled(pred_ids, rel_set, ks)
        for k in ks:
            for m in md[k]:
                agg[k][m] += md[k][m]
        cnt += 1

        # diag: train-seen coverage
        if len(pred_ids) > 0:
            in_train_seen = sum(1 for a in pred_ids if a in train_seen_agents)
            frac = in_train_seen / float(len(pred_ids))
            frac_pred_in_train_seen_sum += frac
            if in_train_seen == len(pred_ids):
                all_pred_in_train_seen_cnt += 1

        # diag: unseen-valid-GT
        unseen_gt_set = rel_set - train_seen_agents
        if len(unseen_gt_set) > 0:
            unseen_gt_query_cnt += 1
            hit_unseen = 1 if any(a in unseen_gt_set for a in pred_ids) else 0
            unseen_gt_hit_cnt += hit_unseen
            unseen_gt_pred_cover_sum += sum(1 for a in pred_ids if a in unseen_gt_set) / float(len(pred_ids) if pred_ids else 1.0)

        if cnt > 0 and (cnt % 50 == 0):
            pbar.set_postfix({
                "done": cnt,
                "skip": skipped,
                "P@K": f"{agg[topk]['P']/cnt:.4f}",
                "nDCG@K": f"{agg[topk]['nDCG']/cnt:.4f}",
                "trainSeenFrac": f"{(frac_pred_in_train_seen_sum/cnt):.3f}",
            })

    if cnt == 0:
        metrics = {topk:{m:0.0 for m in ["P","R","F1","Hit","nDCG","MRR"]}}
        diag = {}
        return metrics, diag

    for m in agg[topk]:
        agg[topk][m] /= cnt

    diag = {
        "n_eval": cnt,
        "n_skipped": skipped,
        "unique_pred_agents": len(unique_pred),
        "avg_frac_pred_in_train_seen": frac_pred_in_train_seen_sum / cnt,
        "pct_queries_all_pred_in_train_seen": all_pred_in_train_seen_cnt / cnt,
        "unseen_gt_query_cnt": unseen_gt_query_cnt,
        "unseen_gt_hit_rate": (unseen_gt_hit_cnt / unseen_gt_query_cnt) if unseen_gt_query_cnt > 0 else 0.0,
        "unseen_gt_avg_frac_pred": (unseen_gt_pred_cover_sum / unseen_gt_query_cnt) if unseen_gt_query_cnt > 0 else 0.0,
    }
    return agg, diag

def pretty_print_diag(title: str, diag: dict):
    print(f"\n---- {title} (OOD diagnostics) ----")
    for k, v in diag.items():
        if isinstance(v, float):
            print(f"{k:>32s}: {v:.6f}")
        else:
            print(f"{k:>32s}: {v}")

# ---------------------- main ----------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--device", type=str, default="cpu")

    # TF-IDF / shape must match ckpt
    ap.add_argument("--max_features", type=int, default=TFIDF_MAX_FEATURES)

    # split
    ap.add_argument("--valid_ratio", type=float, default=0.2)
    ap.add_argument("--split_seed", type=int, default=42)

    # eval
    ap.add_argument("--topk", type=int, default=EVAL_TOPK)
    ap.add_argument("--eval_candidate_size", type=int, default=1000)
    ap.add_argument("--eval_per_part", type=int, default=200)
    ap.add_argument("--eval_parts", type=str, default="PartI,PartII,PartIII")
    ap.add_argument("--seed", type=int, default=1234)

    # ckpt
    ap.add_argument("--ckpt", type=str, default=None,
                    help="Path to ckpt .pt. If empty, auto-locate via data_sig under .cache/OneRec_seq/models/")
    ap.add_argument("--ckpt_prefix", type=str, default="OneRec_seq",
                    help="Cache prefix used during training (default: OneRec_seq).")

    args = ap.parse_args()
    eval_parts = [x.strip() for x in args.eval_parts.split(",") if x.strip()]

    warn_if_topk_diff(args.topk, expected=EVAL_TOPK)
    set_global_seed(args.seed)

    device = torch.device(args.device)

    # ---------- load data ----------
    all_agents, all_questions, all_rankings, tools, qid_to_part = collect_data(args.data_root)
    q_ids, q_texts, a_ids, a_texts = build_text_corpora(all_agents, all_questions, tools)
    qid2idx = {qid:i for i,qid in enumerate(q_ids)}
    vocab = AgentVocab(a_ids)

    # ---------- locate ckpt ----------
    data_sig = dataset_signature(a_ids, all_rankings)

    if args.ckpt is None:
        ckpt = os.path.join(
            args.data_root,
            f".cache/{args.ckpt_prefix}",
            "models",
            f"{args.ckpt_prefix}_{data_sig}.pt",
        )
        args.ckpt = ckpt

    assert os.path.exists(args.ckpt), f"ckpt not found: {args.ckpt}"
    print(f"[ckpt] load: {args.ckpt}")

    # ---------- rebuild TF-IDF ----------
    q_vec = TfidfVectorizer(max_features=args.max_features, lowercase=True)
    Q_csr = q_vec.fit_transform(q_texts)
    A_in_Q_csr = q_vec.transform(a_texts)
    if not issparse(Q_csr) or not issparse(A_in_Q_csr):
        raise ValueError("Q_csr / A_in_Q_csr must be sparse.")
    print(f"[tfidf] Q={Q_csr.shape} A_in_Q={A_in_Q_csr.shape}")

    # ---------- split ----------
    qids_in_rank = [qid for qid in q_ids if qid in all_rankings]
    train_qids, valid_qids = stratified_train_valid_split(
        qids_in_rank,
        qid_to_part=qid_to_part,
        valid_ratio=args.valid_ratio,
        seed=args.split_seed
    )
    print(f"[split] train={len(train_qids)} valid={len(valid_qids)} (valid_ratio={args.valid_ratio}, seed={args.split_seed})")

    # ---------- build train-seen GT agent set ----------
    train_seen_agents = build_train_seen_agent_set(
        train_qids, all_rankings, qid_to_part, topk_eval=args.topk
    )
    valid_seen_agents = build_train_seen_agent_set(
        valid_qids, all_rankings, qid_to_part, topk_eval=args.topk
    )
    unseen_valid_agents = valid_seen_agents - train_seen_agents
    print(f"[agents] train_seen_gt={len(train_seen_agents)} valid_gt={len(valid_seen_agents)} unseen_valid_gt={len(unseen_valid_agents)}")

    # ---------- build models and load weights ----------
    ck = torch.load(args.ckpt, map_location=device)

    # ckpt may be "SFT or SFT+DPO; just read state dict keys
    q_sd = ck.get("q_enc", None)
    g_sd = ck.get("gen", None)
    assert q_sd is not None and g_sd is not None, f"bad ckpt keys: {list(ck.keys())}"

    # infer dims from ckpt tensors (DO NOT GUESS)
    d_q = Q_csr.shape[1]

    # QueryEncoder hidden (=enc_dim) from first Linear: net.0.weight is (enc_dim, d_q)
    enc_dim = int(q_sd["net.0.weight"].shape[0])

    # Generator dims from q_proj.0.weight: (gen_hidden, enc_dim)
    gen_hidden = int(g_sd["q_proj.0.weight"].shape[0])
    tok_dim = int(g_sd["tok_emb.weight"].shape[1])

    q_enc = QueryEncoder(d_q=d_q, hidden=enc_dim, dropout=0.0).to(device)
    gen = OneStepGenerator(
        enc_dim=enc_dim,
        vocab_size=vocab.vocab_size,
        tok_dim=tok_dim,
        hidden=gen_hidden,
        dropout=0.0
    ).to(device)


    q_enc.load_state_dict(q_sd, strict=True)
    gen.load_state_dict(g_sd, strict=True)
    q_enc.eval(); gen.eval()

    print(f"[model] enc_dim={enc_dim} tok_dim={tok_dim} vocab={vocab.vocab_size} ck_mode={ck.get('mode','?')}")

    # ---------- sample qids ----------
    parts = eval_parts
    train_eval_qids = sample_qids_by_part(train_qids, qid_to_part, per_part=args.eval_per_part, seed=args.seed, parts=parts)
    valid_eval_qids = sample_qids_by_part(valid_qids, qid_to_part, per_part=args.eval_per_part, seed=args.seed + 7, parts=parts)
    print(f"[eval] train_sampled={len(train_eval_qids)} valid_sampled={len(valid_eval_qids)} per_part={args.eval_per_part} parts={parts}")

    # ---------- run eval: train ----------
    m_train, d_train = eval_with_diag(
        gen=gen, q_enc=q_enc, Q_csr=Q_csr, q_ids=q_ids, qid2idx=qid2idx,
        a_ids=a_ids, all_rankings=all_rankings, eval_qids=train_eval_qids,
        qid_to_part=qid_to_part, train_seen_agents=train_seen_agents,
        device=device, topk=args.topk, cand_size=args.eval_candidate_size,
        rng_seed=args.seed + 11
    )
    print_metrics_table("TRAIN (sampled per-part)", m_train, ks=(args.topk,), filename="OneRec_infer")
    pretty_print_diag("TRAIN", d_train)

    # per-part train
    for p in parts:
        q_part = [q for q in train_eval_qids if qid_to_part.get(q, "Unknown") == p]
        if not q_part:
            continue
        mp, dp = eval_with_diag(
            gen=gen, q_enc=q_enc, Q_csr=Q_csr, q_ids=q_ids, qid2idx=qid2idx,
            a_ids=a_ids, all_rankings=all_rankings, eval_qids=q_part,
            qid_to_part=qid_to_part, train_seen_agents=train_seen_agents,
            device=device, topk=args.topk, cand_size=args.eval_candidate_size,
            rng_seed=args.seed + 100 + hash(p) % 1000
        )
        print_metrics_table(f"TRAIN {p} (n={len(q_part)})", mp, ks=(args.topk,), filename="OneRec_infer")
        pretty_print_diag(f"TRAIN {p}", dp)

    # ---------- run eval: valid ----------
    m_valid, d_valid = eval_with_diag(
        gen=gen, q_enc=q_enc, Q_csr=Q_csr, q_ids=q_ids, qid2idx=qid2idx,
        a_ids=a_ids, all_rankings=all_rankings, eval_qids=valid_eval_qids,
        qid_to_part=qid_to_part, train_seen_agents=train_seen_agents,
        device=device, topk=args.topk, cand_size=args.eval_candidate_size,
        rng_seed=args.seed + 19
    )
    print_metrics_table("VALID (sampled per-part)", m_valid, ks=(args.topk,), filename="OneRec_infer")
    pretty_print_diag("VALID", d_valid)

    # per-part valid
    for p in parts:
        q_part = [q for q in valid_eval_qids if qid_to_part.get(q, "Unknown") == p]
        if not q_part:
            continue
        mp, dp = eval_with_diag(
            gen=gen, q_enc=q_enc, Q_csr=Q_csr, q_ids=q_ids, qid2idx=qid2idx,
            a_ids=a_ids, all_rankings=all_rankings, eval_qids=q_part,
            qid_to_part=qid_to_part, train_seen_agents=train_seen_agents,
            device=device, topk=args.topk, cand_size=args.eval_candidate_size,
            rng_seed=args.seed + 200 + hash(p) % 1000
        )
        print_metrics_table(f"VALID {p} (n={len(q_part)})", mp, ks=(args.topk,), filename="OneRec_infer")
        pretty_print_diag(f"VALID {p}", dp)

    # extra: global unseen-valid-GT summary
    print("\n==== Global unseen-valid-GT summary (set-level) ====")
    print(f"unseen_valid_gt_agents={len(unseen_valid_agents)} (valid_gt - train_gt)")
    # 这里不做更重的统计了（你上面 query-level unseen_hit 已经够判断“只背训练 agent 还是能命中 valid GT”）

if __name__ == "__main__":
    main()
