#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
finetune_bge_reranker_agentrec.py
---------------------------------
微调 BGE-reranker 用于“一句话 Query → Agent/Tool”的内容匹配（非序列）。
数据协议：{data_root}/PartI|PartII|PartIII/{agents,questions,rankings}/merge.json + Tools/merge.json

训练范式：Pairwise (q, pos, neg) → MarginRankingLoss
负样本：random + TF-IDF 硬负例（默认混合）

新增：
  --triples_cache 支持缓存/加载 (q, pos, neg) 三元组
    * .npy / .npy.gz : 索引版 (q_idx, pos_idx, neg_idx)  —— 推荐
    * .jsonl.gz      : 文本版 {"q":..., "p":..., "n":...}

输出：
  - {save_dir}/reranker/  (HF/PEFT 格式)
  - 控制台打印：二阶段评测（Vector Recall → Rerank）指标

显存优化：
  - 可选 LoRA（默认开启）+ 梯度检查点
  - Pairwise 正负样本合并为一次前向，减少显存占用
  - 评测阶段 rerank 分批评分，避免一次性 tokenizer/前向过多候选
"""
import os, json, math, random, argparse, pickle, gzip
from dataclasses import dataclass
from typing import Dict, List, Tuple, Iterable

import numpy as np
from tqdm.auto import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)

# -------------------- 可选：PEFT / LoRA --------------------
try:
    from peft import LoraConfig, get_peft_model, PeftModel, PeftConfig
    _PEFT_AVAILABLE = True
except Exception:
    _PEFT_AVAILABLE = False

# -------------------- 全局设置 --------------------
pos_topk = 5  # 每个qid的前k当正样本


# -------------------- I/O 与准备 --------------------
def load_json(p: str):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def collect_data(data_root: str):
    parts = ["PartI", "PartII", "PartIII"]
    all_agents, all_questions, all_rankings = {}, {}, {}
    for part in parts:
        agents    = load_json(os.path.join(data_root, part, "agents",    "merge.json"))
        questions = load_json(os.path.join(data_root, part, "questions", "merge.json"))
        rankings  = load_json(os.path.join(data_root, part, "rankings",  "merge.json"))
        all_agents.update(agents)
        all_questions.update(questions)
        all_rankings.update(rankings["rankings"])
    tools = load_json(os.path.join(data_root, "Tools", "merge.json"))
    return all_agents, all_questions, all_rankings, tools

def build_text_corpora(all_agents, all_questions, tools):
    q_ids   = list(all_questions.keys())
    q_texts = [(all_questions[qid].get("input","") or "") for qid in q_ids]

    def tool_text(tn):
        t = tools.get(tn, {}) or {}
        return f"{tn} {t.get('description','')}".strip()

    a_ids = list(all_agents.keys())
    a_texts = []
    for aid in a_ids:
        a = all_agents[aid]
        mname = a.get("M", {}).get("name", "") or ""
        tlst  = a.get("T", {}).get("tools", []) or []
        concat_tools = " | ".join([tool_text(t) for t in tlst])
        a_text = (mname + " || " + concat_tools).strip(" |")
        a_texts.append(a_text if a_text else mname)
    return q_ids, q_texts, a_ids, a_texts

def train_valid_split(qids_in_rankings, valid_ratio=0.2, seed=42):
    rng = random.Random(seed)
    q = list(qids_in_rankings)
    rng.shuffle(q)
    n_valid = int(len(q)*valid_ratio)
    return q[n_valid:], q[:n_valid]


# -------------------- Pairwise 数据集 --------------------
class PairwiseRecDataset(Dataset):
    """
    每条样本 = (q_text, pos_text, neg_text)
    """
    def __init__(self, triples, tokenizer, max_len=256):
        self.triples = triples
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self): return len(self.triples)

    def __getitem__(self, i):
        q, p, n = self.triples[i]
        enc_p = self.tok(q, p, truncation=True, padding="max_length", max_length=self.max_len)
        enc_n = self.tok(q, n, truncation=True, padding="max_length", max_length=self.max_len)
        return {"pos": enc_p, "neg": enc_n}

class PairwiseRecDatasetIdx(Dataset):
    """
    每条样本 = (q_idx, pos_a_idx, neg_a_idx)，取用时再映射为文本并 tokenizer。
    用于 .npy/.npy.gz 索引缓存。
    """
    def __init__(self, triples_np: np.ndarray, q_texts: List[str], a_texts: List[str], tokenizer, max_len=256):
        assert triples_np.ndim == 2 and triples_np.shape[1] == 3
        self.arr = triples_np.astype(np.int32)
        self.q_texts = q_texts
        self.a_texts = a_texts
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self): return self.arr.shape[0]

    def __getitem__(self, i):
        q_idx, p_idx, n_idx = map(int, self.arr[i])
        q = self.q_texts[q_idx]
        p = self.a_texts[p_idx]
        n = self.a_texts[n_idx]
        enc_p = self.tok(q, p, truncation=True, padding="max_length", max_length=self.max_len)
        enc_n = self.tok(q, n, truncation=True, padding="max_length", max_length=self.max_len)
        return {"pos": enc_p, "neg": enc_n}

@dataclass
class PairwiseCollator:
    tokenizer: AutoTokenizer
    def __call__(self, batch):
        assert batch and "pos" in batch[0] and "neg" in batch[0], f"Bad batch keys: {list(batch[0].keys()) if batch else None}"
        pos = {k: torch.tensor([b["pos"][k] for b in batch]) for k in batch[0]["pos"]}
        neg = {k: torch.tensor([b["neg"][k] for b in batch]) for k in batch[0]["neg"]}
        return {"pos": pos, "neg": neg}


# -------------------- Trainer（单次前向 + 设备修复） --------------------
class PairwiseTrainer(Trainer):
    """
    直接优化 MarginRankingLoss: maximize s(q,pos) - s(q,neg)
    —— 合并一次前向，省显存、更快；兼容 DataParallel/DDP 的设备获取
    """
    def _dev(self, model):
        try:
            return self.args.device
        except Exception:
            try:
                return next(model.parameters()).device
            except StopIteration:
                return torch.device("cpu")

    @staticmethod
    def _cat_inputs(a, b, device):
        out = {}
        for k in a:
            av, bv = a[k].to(device), b[k].to(device)
            out[k] = torch.cat([av, bv], dim=0)
        return out

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs: bool = False,
        num_items_in_batch: int | None = None,
        **kwargs
    ):
        device = self._dev(model)
        pos = inputs["pos"]
        neg = inputs["neg"]
        B = pos["input_ids"].size(0)

        merged = self._cat_inputs(pos, neg, device)   # (2B, L)
        logits = model(**merged).logits.squeeze(-1)    # (2B,)
        out_pos, out_neg = logits[:B], logits[B:]

        target = torch.ones_like(out_pos, device=out_pos.device)
        loss = F.margin_ranking_loss(out_pos, out_neg, target, margin=0.0)
        if return_outputs:
            return loss, {"pos": out_pos, "neg": out_neg}
        return loss


# -------------------- 三元组构造（随机负 + TF-IDF硬负例） --------------------
def build_pairs_for_finetune(
    train_qids: List[str],
    all_rankings: Dict[str, List[str]],
    q_text_map: Dict[str,str],
    a_text_map: Dict[str,str],
    a_ids: List[str],
    hard_neg_per_pos: int = 2,
    rand_neg_per_pos: int = 1,
    tfidf_max_features: int = 20000,
    seed: int = 42,
    hard_pool_size: int = 200,
    chunk_size: int = 512,
    cache_path: str = None
):
    rng = random.Random(seed)
    triples: List[Tuple[str,str,str]] = []

    if cache_path and os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            hard_pool_map = pickle.load(f)  # {qid: [aid1, aid2, ...]}
    else:
        hard_pool_map = {}

    pending_qids = [qid for qid in train_qids if qid not in hard_pool_map]

    if pending_qids:
        q_corpus = [q_text_map[qid] for qid in pending_qids]
        a_corpus = [a_text_map[aid]  for aid  in a_ids]

        q_vec = TfidfVectorizer(max_features=tfidf_max_features, lowercase=True)
        a_vec = TfidfVectorizer(max_features=tfidf_max_features, lowercase=True)
        Q_all = q_vec.fit_transform(q_corpus).astype(np.float32)  # (Nq, Vq)
        A_all = a_vec.fit_transform(a_corpus).astype(np.float32)  # (Na, Va)

        Na = len(a_ids)
        for i0 in tqdm(range(0, Q_all.shape[0], chunk_size), desc="TF-IDF hard-neg (batched)"):
            i1 = min(Q_all.shape[0], i0 + chunk_size)
            Q_chunk = Q_all[i0:i1]                               # (B, Vq)
            S = (Q_chunk @ A_all.T).toarray().astype(np.float32) # (B, Na)

            for r in range(S.shape[0]):
                sims = S[r]
                k = min(hard_pool_size, Na)
                top_idx = np.argpartition(-sims, k-1)[:k]
                top_idx = top_idx[np.argsort(-sims[top_idx])]
                qid = pending_qids[i0 + r]
                hard_pool_map[qid] = [a_ids[j] for j in top_idx]

        if cache_path:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, "wb") as f:
                pickle.dump(hard_pool_map, f)

    all_agent_set = set(a_ids)
    for qid in tqdm(train_qids, desc="Build pairwise triples (assemble)"):
        pos_list = [aid for aid in (all_rankings.get(qid, []) or [])[:pos_topk] if aid in all_agent_set]
        if not pos_list:
            continue
        pos_set = set(pos_list)
        neg_pool_rand = list(all_agent_set - pos_set)
        hard_list = [aid for aid in hard_pool_map.get(qid, []) if aid not in pos_set]

        qtext = q_text_map[qid]
        for pos_a in pos_list:
            used = 0
            for hn in hard_list:
                triples.append((qtext, a_text_map[pos_a], a_text_map[hn]))
                used += 1
                if used >= hard_neg_per_pos:
                    break
            for _ in range(rand_neg_per_pos):
                if neg_pool_rand:
                    rn = rng.choice(neg_pool_rand)
                    triples.append((qtext, a_text_map[pos_a], a_text_map[rn]))
                else:
                    triples.append((qtext, a_text_map[pos_a], a_text_map[pos_a]))
    return triples


# -------------------- 缓存 I/O --------------------
def save_triples_npy(triples_idx: np.ndarray, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if path.endswith(".gz"):
        with gzip.open(path, "wb") as f:
            np.save(f, triples_idx.astype(np.int32), allow_pickle=False)
    else:
        np.save(path, triples_idx.astype(np.int32), allow_pickle=False)

def load_triples_npy(path: str) -> np.ndarray:
    if path.endswith(".gz"):
        with gzip.open(path, "rb") as f:
            arr = np.load(f, allow_pickle=False)
    else:
        arr = np.load(path, allow_pickle=False)
    return arr.astype(np.int32)

def save_triples_jsonl_gz(triples_text: List[Tuple[str,str,str]], path: str):
    assert path.endswith(".jsonl.gz")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for q, p, n in triples_text:
            f.write(json.dumps({"q": q, "p": p, "n": n}, ensure_ascii=False) + "\n")

def load_triples_jsonl_gz(path: str) -> List[Tuple[str,str,str]]:
    triples = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            triples.append((obj["q"], obj["p"], obj["n"]))
    return triples


# -------------------- 评测：Vector Recall → Rerank（分批） --------------------
@torch.no_grad()
def batched_tokenize_and_score(
    model, tokenizer, qtext: str, doc_texts: List[str], device, max_len=256, batch=256
) -> np.ndarray:
    """将 q 与 doc_texts 成对编码并分批做前向，避免一次塞太多导致 OOM。"""
    scores = []
    for i0 in range(0, len(doc_texts), batch):
        i1 = min(len(doc_texts), i0 + batch)
        enc = tokenizer([qtext]*(i1-i0), doc_texts[i0:i1],
                        truncation=True, padding=True, max_length=max_len, return_tensors="pt")
        enc = {k: v.to(device) for k,v in enc.items()}
        out = model(**enc).logits.squeeze(-1).detach().float().cpu().numpy()
        scores.append(out)
    if not scores:
        return np.zeros((0,), dtype=np.float32)
    return np.concatenate(scores, axis=0).astype(np.float32)

import math, random, numpy as np
from typing import List, Dict
from tqdm import tqdm
import torch

@torch.no_grad()
def evaluate(
    model, tokenizer,
    q_ids: List[str], a_ids: List[str],
    all_questions: Dict[str, dict], all_agents: Dict[str, dict], tools: Dict[str, dict],
    all_rankings: Dict[str, List[str]],
    eval_qids: List[str],
    *,
    cand_size: int = 1000,                 # 每条 query 的候选池大小（正例注入 + 随机负例补齐）
    pos_topk: int = 10,                    # 取多少个正例作为“相关集”参与评测（上限）
    ks=(5, 10, 50),
    device: str = "cuda:0",
    seed: int = 123,
    max_len: int = 256,
    rerank_batch: int = 256,
    use_amp: bool = True                   # 混合精度以提升吞吐/省显存
):
    """
    仅用 BGE-Rerank 对候选进行全量精排的评测函数。
    - 不再做向量召回；候选 = 正例注入 + 随机负例采样。
    - 指标：P/R/F1/Hit/nDCG/MRR@K
    """
    device = torch.device(device)

    # ---------- 文本拼装 ----------
    def tool_text(tn):
        t = tools.get(tn, {}) or {}
        return f"{tn} {t.get('description', '')}".strip()

    def agent_text(aid):
        a = all_agents.get(aid, {}) or {}
        mname = a.get("M", {}).get("name", "") or ""
        tlst = a.get("T", {}).get("tools", []) or []
        txt = (mname + " || " + " | ".join([tool_text(t) for t in tlst])).strip(" |")
        return txt or mname

    a_texts = {aid: agent_text(aid) for aid in a_ids}
    a_set = set(a_ids)

    # ---------- batched 评分（交叉编码器） ----------
    def batched_tokenize_and_score(qtext: str, docs: List[str]) -> np.ndarray:
        scores = []
        # 统一设置 padding/truncation，避免形状抖动
        for i in range(0, len(docs), rerank_batch):
            batch_docs = docs[i:i + rerank_batch]
            enc = tokenizer(
                [qtext] * len(batch_docs),
                batch_docs,
                truncation=True,
                padding="longest",
                max_length=max_len,
                return_tensors="pt"
            )
            enc = {k: v.to(device, non_blocking=True) for k, v in enc.items()}
            if use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    out = model(**enc)
            else:
                out = model(**enc)

            # 取模型的相关性分数（常见是 logits 或者最后层CLS回归头）
            if hasattr(out, "logits"):
                s = out.logits
            elif isinstance(out, (list, tuple)) and len(out) > 0:
                s = out[0]
            else:
                raise RuntimeError("Unsupported model output; please adapt score extraction.")

            s = s.squeeze(-1).float().detach().cpu().numpy()
            scores.append(s)
        return np.concatenate(scores, axis=0) if scores else np.zeros((0,), dtype=np.float32)

    # ---------- 指标累计 ----------
    agg = {k: {"P": 0.0, "R": 0.0, "F1": 0.0, "Hit": 0.0, "nDCG": 0.0, "MRR": 0.0} for k in ks}
    max_k = max(ks)
    ref_k = 10 if 10 in ks else max_k

    rnd = random.Random(seed)
    cnt, skipped = 0, 0

    # 性能设置（可选）
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    model.eval()

    pbar = tqdm(eval_qids, desc="Evaluating (Full BGE-Rerank)", dynamic_ncols=True)
    for qid in pbar:
        # 构建该问题的“相关集”（ground truth）
        gt_all = [aid for aid in (all_rankings.get(qid, []) or []) if aid in a_set]
        gt = gt_all[:pos_topk]
        if not gt:
            skipped += 1
            pbar.set_postfix({"done": cnt, "skipped": skipped})
            continue
        rel_set = set(gt)

        # 构建候选池：强制包含正例 + 随机负例
        neg_pool = list(a_set - rel_set)
        need_neg = max(0, cand_size - len(gt))
        cand_ids = gt + (rnd.sample(neg_pool, min(need_neg, len(neg_pool))) if need_neg > 0 else [])
        # 若 cand_size 小于正例数，也不会丢正例（此时 cand_ids 就是全正例的一个切片）

        # 交叉编码器全量精排
        qtext = (all_questions.get(qid, {}) or {}).get("input", "") or ""
        doc_texts = [a_texts[aid] for aid in cand_ids]
        ce_scores = batched_tokenize_and_score(qtext, doc_texts)

        # 取 topK 作为最终预测
        order = np.argsort(-ce_scores)[:max_k]
        pred_ids = [cand_ids[i] for i in order]

        # 计算@K指标
        bin_hits = [1 if aid in rel_set else 0 for aid in pred_ids]
        for k in ks:
            topk_hits = bin_hits[:k]
            Hk = sum(topk_hits)
            P = Hk / float(k)
            R = Hk / float(len(rel_set))
            F1 = (2 * P * R) / (P + R) if (P + R) > 0 else 0.0
            Hit = 1.0 if Hk > 0 else 0.0

            # nDCG@k
            dcg = 0.0
            for i, h in enumerate(topk_hits):
                if h:
                    dcg += 1.0 / math.log2(i + 2.0)
            ideal = min(len(rel_set), k)
            idcg = sum(1.0 / math.log2(i + 2.0) for i in range(ideal)) if ideal > 0 else 0.0
            nDCG = (dcg / idcg) if idcg > 0 else 0.0

            # MRR@k
            rr = 0.0
            for i, h in enumerate(topk_hits):
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
            f"P@{ref_k}": f"{(ref['P']/cnt):.4f}",
            f"nDCG@{ref_k}": f"{(ref['nDCG']/cnt):.4f}",
            f"MRR@{ref_k}": f"{(ref['MRR']/cnt):.4f}",
            "Ncand": len(cand_ids),
        })

    # 汇总
    if cnt == 0:
        return {k: {"P": 0.0, "R": 0.0, "F1": 0.0, "Hit": 0.0, "nDCG": 0.0, "MRR": 0.0} for k in ks}
    for k in ks:
        for m in agg[k]:
            agg[k][m] /= cnt
    return agg



# -------------------- 主流程 --------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--model_name", type=str, default="BAAI/bge-reranker-base")
    ap.add_argument("--save_dir", type=str, required=True)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=8)       # 默认保守
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max_len", type=int, default=192)        # 默认保守
    ap.add_argument("--hard_neg_per_pos", type=int, default=2)
    ap.add_argument("--rand_neg_per_pos", type=int, default=1)
    ap.add_argument("--tfidf_max_features", type=int, default=20000)
    ap.add_argument("--valid_ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--eval_cand_size", type=int, default=1000)
    ap.add_argument("--recall_topk", type=int, default=200)
    ap.add_argument("--rerank_batch", type=int, default=256, help="评测阶段 rerank 分批大小")
    ap.add_argument("--triples_cache", type=str, default=None,
                    help="若提供：先尝试从该路径加载三元组；如不存在则构建并保存。支持 .npy/.npy.gz(索引) 或 .jsonl.gz(文本)")
    # LoRA/显存相关
    ap.add_argument("--use_lora", type=int, default=1)
    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--lora_alpha", type=float, default=16.0)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--grad_ckpt", type=int, default=1, help="开启梯度检查点，省显存")
    ap.add_argument("--accum_steps", type=int, default=2, help="梯度累计步数")
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    # —— 放在 random.seed 之后、构建数据/模型之前：
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")  # 避免 Rust tokenizer 线程冲突
    os.environ.setdefault("OMP_NUM_THREADS", "1")              # 控制 MKL/OpenMP 线程数
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    try:
        torch.set_num_threads(1)
    except Exception:
        pass


    # 1) 读数据
    all_agents, all_questions, all_rankings, tools = collect_data(args.data_root)
    print(f"[data] agents={len(all_agents)}  questions={len(all_questions)}  tools={len(tools)}")

    # 2) 文本语料
    q_ids, q_texts, a_ids, a_texts = build_text_corpora(all_agents, all_questions, tools)
    q_text_map = {qid:qt for qid,qt in zip(q_ids, q_texts)}
    a_text_map = {aid:at for aid,at in zip(a_ids, a_texts)}

    # 3) Train/Valid split
    qids_in_rank = [qid for qid in q_ids if qid in all_rankings]
    train_qids, valid_qids = train_valid_split(qids_in_rank, args.valid_ratio, args.seed)
    print(f"[split] train={len(train_qids)}  valid={len(valid_qids)}")

    # 3.5) 文本->索引映射（给索引缓存使用）
    qtext2idx = {txt: i for i, txt in enumerate(q_texts)}
    atext2idx = {txt: i for i, txt in enumerate(a_texts)}

    # 4) 三元组：优先加载缓存，否则构建并保存
    triples_idx_np = None
    triples_txt = None

    if args.triples_cache and os.path.exists(args.triples_cache):
        if args.triples_cache.endswith(".npy") or args.triples_cache.endswith(".npy.gz") or args.triples_cache.endswith(".gz"):
            print(f"[cache] load triples (index) from {args.triples_cache}")
            triples_idx_np = load_triples_npy(args.triples_cache)
        elif args.triples_cache.endswith(".jsonl.gz"):
            print(f"[cache] load triples (text) from {args.triples_cache}")
            triples_txt = load_triples_jsonl_gz(args.triples_cache)
        else:
            print(f"[cache] unknown extension: {args.triples_cache}, skip loading.")

    if triples_idx_np is None and triples_txt is None:
        cache_path = os.path.join(args.save_dir, "hardneg_cache.pkl")
        triples_txt = build_pairs_for_finetune(
            train_qids, all_rankings, q_text_map, a_text_map, a_ids,
            hard_neg_per_pos=args.hard_neg_per_pos,
            rand_neg_per_pos=args.rand_neg_per_pos,
            tfidf_max_features=args.tfidf_max_features,
            seed=args.seed,
            hard_pool_size=200,
            chunk_size=512,
            cache_path=cache_path
        )
        print(f"[train] triples={len(triples_txt)}")

        # 保存缓存
        if args.triples_cache:
            if args.triples_cache.endswith(".npy") or args.triples_cache.endswith(".npy.gz") or args.triples_cache.endswith(".gz"):
                print(f"[cache] convert text→index & save -> {args.triples_cache}")
                triples_idx = []
                miss_q = miss_p = miss_n = 0
                for q_text, p_text, n_text in triples_txt:
                    qi = qtext2idx.get(q_text, -1)
                    pi = atext2idx.get(p_text, -1)
                    ni = atext2idx.get(n_text, -1)
                    if qi < 0: miss_q += 1; continue
                    if pi < 0: miss_p += 1; continue
                    if ni < 0: miss_n += 1; continue
                    triples_idx.append((qi, pi, ni))
                if miss_q or miss_p or miss_n:
                    print(f"[cache][warn] missing maps -> q:{miss_q} p:{miss_p} n:{miss_n} (已跳过)")
                triples_idx_np = np.asarray(triples_idx, dtype=np.int32)
                save_triples_npy(triples_idx_np, args.triples_cache)
            else:
                if not args.triples_cache.endswith(".jsonl.gz"):
                    args.triples_cache += ".jsonl.gz"
                print(f"[cache] save text triples -> {args.triples_cache}")
                save_triples_jsonl_gz(triples_txt, args.triples_cache)

    # 5) 模型与Tokenizer（LoRA + 梯度检查点）
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    use_fast = True
    if os.environ.get("HF_NO_FAST_TOKENIZER", "0") == "1":
        use_fast = False
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=use_fast)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=1)

    # TF32 可提速&稳住显存碎片
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
    except Exception:
        pass

    if args.grad_ckpt and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    if args.use_lora:
        if not _PEFT_AVAILABLE:
            raise RuntimeError("PEFT 未安装，请先: pip install -U peft")
        # 目标模块命名在不同版本略有差异，先尝试常见名；失败则兜底 all-linear
        lora_cfg = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="SEQ_CLS",
            target_modules=["query","key","value","dense","intermediate.dense","output.dense"]
        )
        try:
            model = get_peft_model(model, lora_cfg)
        except Exception:
            lora_cfg.target_modules = "all-linear"
            model = get_peft_model(model, lora_cfg)

    model.to(device)

    # 6) Dataset & Trainer
    if triples_idx_np is not None:
        print(f"[train] use cached index triples: {triples_idx_np.shape[0]}")
        train_ds = PairwiseRecDatasetIdx(triples_idx_np, q_texts, a_texts, tokenizer, max_len=args.max_len)
    else:
        if triples_txt is None and args.triples_cache and args.triples_cache.endswith(".jsonl.gz"):
            print(f"[cache] lazy-load text triples from {args.triples_cache}")
            triples_txt = load_triples_jsonl_gz(args.triples_cache)
        if triples_txt is None:
            raise RuntimeError("No training triples available. Check --triples_cache or building pipeline.")
        print(f"[train] use text triples: {len(triples_txt)}")
        train_ds = PairwiseRecDataset(triples_txt, tokenizer, max_len=args.max_len)

    collator = PairwiseCollator(tokenizer)

    out_dir = os.path.join(args.save_dir, "reranker")
    os.makedirs(out_dir, exist_ok=True)
    targs = TrainingArguments(
        output_dir=out_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.accum_steps,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        fp16=("cuda" in args.device),
        logging_steps=50,
        save_strategy="no",
        report_to=[],
        remove_unused_columns=False,   # 保留自定义 'pos'/'neg'
        label_names=[],                # 避免 Trainer 去找 labels
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        optim="adamw_torch_fused" if torch.cuda.is_available() else "adamw_torch",
    )

    trainer = PairwiseTrainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        data_collator=collator,
    )

    print("[train] start")
    trainer.train()
    print("[train] done")

    # 7) 保存权重（HF/PEFT 格式）
    tokenizer.save_pretrained(out_dir)
    # 如果需要合并 LoRA 到基模再保存，把下面两行取消注释：
    # if isinstance(model, PeftModel):
    #     model = model.merge_and_unload()
    model.save_pretrained(out_dir)
    print(f"[save] reranker -> {out_dir}")

    # 8) 二阶段验证（向量召回→BGE重排）
    print("[eval] Vector TF-IDF Recall -> Rerank with fine-tuned model")
    metrics = evaluate(
        model, tokenizer,
        q_ids, a_ids,
        all_questions, all_agents, tools,
        all_rankings, valid_qids,
        cand_size=args.eval_cand_size, recall_topk=args.recall_topk,
        ks=(5,10,50), device=device, seed=123,
        max_len=args.max_len, rerank_batch=args.rerank_batch
    )
    for k in (5,10,50):
        m = metrics[k]
        print(f"@{k}: P={m['P']:.4f} R={m['R']:.4f} F1={m['F1']:.4f} Hit={m['Hit']:.4f} nDCG={m['nDCG']:.4f} MRR={m['MRR']:.4f}")


if __name__ == "__main__":
    # 推荐在 shell 设置（可减少显存碎片）：
    # export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    main()
