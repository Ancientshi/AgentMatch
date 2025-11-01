import os
from FlagEmbedding import BGEM3FlagModel
import torch
import numpy as np
import random
import json


BGEM3_model = None

# 设置缓存目录
os.environ["TRANSFORMERS_CACHE"] = './model'

def load_BGEM3_model():
    global BGEM3_model
    if BGEM3_model is None:
        BGEM3_model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True, device="cuda:0")
    else:
        pass
    
def get_embeddings(doc_list):
    embeddings = BGEM3_model.encode(doc_list, 
                            batch_size=64, 
                            max_length=2560, # If you don't need such a long length, you can set a smaller value to speed up the encoding process.
                            )['dense_vecs']
    return embeddings

import logging
from datetime import datetime


def print_metrics_table(title, metrics_dict, ks=(5, 10, 50), filename=''):
    #要把这个也记录在log/{filename}_{timestamp}.log里面
    log_dir = "log"
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"{filename}_{timestamp}.log")
    with open(log_path, "a", encoding="utf-8") as log_file:
        print(f"\n== {title} ==")
        header = f"{'@K':>4} | {'P':>7} {'R':>7} {'F1':>7} {'Hit':>7} {'nDCG':>7} {'MRR':>7}"
        print(header)
        print("-" * len(header))
        log_file.write(f"\n== {title} ==\n")
        log_file.write(header + "\n")
        log_file.write("-" * len(header) + "\n")
        for k in ks:
            m = metrics_dict[k]
            print(f"{k:>4} | {m['P']:.4f} {m['R']:.4f} {m['F1']:.4f} {m['Hit']:.4f} {m['nDCG']:.4f} {m['MRR']:.4f}")
            log_file.write(f"{k:>4} | {m['P']:.4f} {m['R']:.4f} {m['F1']:.4f} {m['Hit']:.4f} {m['nDCG']:.4f} {m['MRR']:.4f}\n")
            
            
            
from math import log2
from typing import List, Dict, Tuple, Set, Any, Iterable

# ---------- helpers ----------
def _alpha_for_gt(gt: Tuple[Set[str], Set[str]], alpha: float) -> float | None:
    """
    新规则：
      - M_gt 为空且 T_gt 非空   → 返回 0.0  （只用工具召回）0*1+1*T_S 假如T_S为0.6 则相似度为0.6；如果返回0.5, 则相似度为0.5*1+0.5*0.6=0.8
      - T_gt 为空且 M_gt 非空   → 返回 1.0  （只用模型匹配）
      - M_gt 与 T_gt 都为空     → 返回 None（调用处直接视为相似度=1）
      - 二者都非空              → 返回给定 alpha
    """
    M_gt, T_gt = gt
    if len(M_gt) == 0 and len(T_gt) == 0:
        return None
    if len(M_gt) == 0 and len(T_gt) > 0:
        return 0.5
    if len(T_gt) == 0 and len(M_gt) > 0:
        return 1.0
    return alpha


def _collect_strs_raw(x: Any) -> Iterable[str]:
    """只抽取原始字符串，不做任何清洗/归一化。"""
    out = []
    if isinstance(x, str):
        out.append(x)
    elif isinstance(x, dict):
        for k in ("name", "id", "tool", "model", "title"):
            v = x.get(k)
            if isinstance(v, str) and v:
                out.append(v)
    elif isinstance(x, (list, tuple, set)):
        for v in x:
            out.extend(_collect_strs_raw(v))
    return out

def _as_agent_tuple(x: Any) -> Tuple[Set[str], Set[str]]:
    """
    硬匹配：模型来自 M.name（或其他字符串字段），工具来自 T.tools（字符串列表）。
    同时兼容 models/tools 顶层字段。均为原样字符串集合，不做任何规整。
    """
    if isinstance(x, tuple) and len(x) == 2:
        m, t = x
        M = {s for s in m if isinstance(s, str)}
        T = {s for s in t if isinstance(s, str)}
        return M, T

    if not isinstance(x, dict):
        raise ValueError("Agent must be dict or (models, tools) tuple.")

    models, tools = set(), set()

    # 标准位：M.name
    M = x.get("M") or {}
    if isinstance(M, dict):
        for s in _collect_strs_raw(M):
            models.add(s)

    # 标准位：T.tools
    T = x.get("T") or {}
    if isinstance(T, dict):
        for s in _collect_strs_raw(T.get("tools", [])):
            tools.add(s)

    # 兼容旧字段
    for s in _collect_strs_raw(x.get("models", [])):
        models.add(s)
    for s in _collect_strs_raw(x.get("tools", [])):
        tools.add(s)

    return models, tools



def gt_position_weights(L: int, scheme: str = "log") -> List[float]:
    """Weight earlier GT positions higher; sum to 1."""
    if L == 0:
        return []
    if scheme == "log":
        raw = [1.0 / log2(i + 2) for i in range(L)]  # i:0-based -> rank i+1
    elif scheme == "linear":
        raw = [float(L - i) for i in range(L)]
    else:
        raw = [1.0] * L
    s = sum(raw)
    return [x / s for x in raw]

def tool_recall(T_rec: Set[str], T_gt: Set[str]) -> float:
    """
    工具召回率： 用“推荐工具集”对“目标工具集”的召回率。
    - 若 GT 工具集为空，返回 1.0（不惩罚）。
    """
    if len(T_gt) == 0:
        return 1.0
    return len(T_rec & T_gt) / len(T_gt)

def model_match(M_rec: Set[str], M_gt: Set[str]) -> float:
    """
    模型完全匹配：任意一个模型字符串相同则判 1，否则 0。
    - 若 GT 模型集为空，返回 1.0（GT 未指定，不惩罚）。
    """
    if len(M_gt) == 0:
        return 1.0
    return 1.0 if (M_rec & M_gt) else 0.0

def agent_similarity(A: Tuple[Set[str], Set[str]],
                     B: Tuple[Set[str], Set[str]],
                     alpha: float = 0.4) -> float:
    """
    混合相似度：
      sim = alpha * ModelMatch + (1 - alpha) * ToolRecall
    注：ModelMatch=1 若存在任一相同模型（M_gt 为空时该函数原本返回 1，但在外层我们会用 alpha=0 绕过）；
        ToolRecall=|T_rec ∩ T_gt|/|T_gt|（T_gt 为空时该函数原本返回 1，但在外层我们会用 alpha=1 绕过）。
    """
    M_rec, T_rec = A
    M_gt,  T_gt  = B

    # 仍沿用你原有的两个子打分
    mm = model_match(M_rec, M_gt)
    tr = tool_recall(T_rec, T_gt)

    return alpha * mm + (1 - alpha) * tr


def soft_rel(rec_agent: Tuple[Set[str], Set[str]],
             gt_agents: List[Tuple[Set[str], Set[str]]],
             beta: List[float],
             alpha: float) -> float:
    """
    软相关度：对每个 GT 计算相似度并乘以该 GT 的位置权重 beta[i]，取最大值。
    采用新规则的 α 选择逻辑（见 _alpha_for_gt）。
    """
    best = 0.0
    for i, g in enumerate(gt_agents):
        alpha_i = _alpha_for_gt(g, alpha)
        if alpha_i is None:
            # GT 的 M 与 T 都为空 → 直接记为满分相似度 1.0
            s = 1.0
        else:
            s = agent_similarity(rec_agent, g, alpha=alpha_i)
        val = beta[i] * s
        if val > best:
            best = val
    return best




def config_ndcg_at_k(R, G, K, alpha, beta):
    """
    R: List[agent_obj]  (此次评测的候选池/推荐序列)
    G: List[gt_agent_obj]
    K: int
    alpha: float
    beta: List[float]  # 由 gt_position_weights 得到
    """
    if K <= 0 or not R or not G:
        return 0.0

    # 与 DCG 完全同构的相关度定义
    rels_all = [soft_rel(r, G, beta, alpha) for r in R]  # 对候选池逐个算 soft_rel

    k = min(K, len(rels_all))
    top_rels = rels_all[:k]                              # 这里假设 R 已按你的打分排序
    dcg = sum(top_rels[j] / log2(j + 2) for j in range(k))

    # IDCG 用同一候选池 & 同一 soft_rel，但理想排序（降序）
    ideal_rels = sorted(rels_all, reverse=True)[:k]
    idcg = sum(ideal_rels[j] / log2(j + 2) for j in range(k))

    return (dcg / idcg) if idcg > 0 else 0.0


# ---------- configuration tailored evaluation ----------

def evaluate_agents(
    gt_agents: List[Any],
    rec_agents: List[Any],
    ks: List[int] = [5, 10, 50],
    alpha: float = 0.4,
    theta: float = 0.67,
) -> Dict[int, Dict[str, float]]:
    # 归一化
    G = [_as_agent_tuple(x) for x in gt_agents]
    R = [_as_agent_tuple(x) for x in rec_agents]
    L = len(G)
    beta = [1] * L  # 你当前用等权

    results: Dict[int, Dict[str, float]] = {}
    for K in ks:
        k = min(K, len(R))
        if k == 0 or L == 0:
            results[K] = {"Precision": 0, "Recall": 0, "F1": 0, "Config-nDCG": 0, "MRR": 0}
            continue

        # 先为 top-K 每个推荐，找到“最佳匹配 GT（索引）”及其 soft_rel 分数
        best_scores = []
        best_idx = []
        for j in range(k):
            r = R[j]
            s_best, i_best = 0.0, None
            for i, g in enumerate(G):
                alpha_i = _alpha_for_gt(g, alpha)
                s = 1.0 if alpha_i is None else agent_similarity(r, g, alpha=alpha_i)
                s *= beta[i]  # 等权就是原值
                if s > s_best:
                    s_best, i_best = s, i
            best_scores.append(s_best)
            best_idx.append(i_best)

        # 覆盖式计数：一个 GT 只算一次
        covered = set()
        unique_hits = []
        for rank, (s, gi) in enumerate(zip(best_scores, best_idx), start=1):
            if gi is not None and s >= theta and gi not in covered:
                covered.add(gi)
                unique_hits.append(1)
            else:
                unique_hits.append(0)

        tp = len(covered)                 # 覆盖到的 GT 数
        precision = tp / max(1, k)        # 覆盖式 Precision
        recall    = tp / max(1, L)        # 一定 ≤ 1
        f1 = (2*precision*recall/(precision+recall)) if (precision+recall)>0 else 0.0

        # MRR：第一次“新增覆盖”出现的排名
        mrr = 0.0
        seen = set()
        for rank, (s, gi) in enumerate(zip(best_scores, best_idx), start=1):
            if gi is not None and s >= theta and gi not in seen:
                mrr = 1.0 / rank
                break

        # Config-nDCG@K（你的定义保留）
        cndcg = config_ndcg_at_k(R, G, K, alpha, beta)

        results[K] = {"Precision": precision, "Recall": recall, "F1": f1, "Config-nDCG": cndcg, "MRR": mrr}

    return results

# ---------------- data I/O ----------------
def load_json(p: str):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)
    
def collect_data(parts = ["PartI", "PartII", "PartIII"], data_root = '.'):
    all_agents: Dict[str, dict] = {}
    all_questions: Dict[str, dict] = {}
    all_rankings: Dict[str, List[str]] = {}
    for part in parts:
        agents_path    = os.path.join(data_root, part, "agents",    "merge.json")
        questions_path = os.path.join(data_root, part, "questions", "merge.json")
        rankings_path  = os.path.join(data_root, part, "rankings",  "merge.json")
        agents    = load_json(agents_path)
        questions = load_json(questions_path)
        rankings  = load_json(rankings_path)
        all_agents.update(agents)
        all_questions.update(questions)
        all_rankings.update(rankings["rankings"])
    return all_agents, all_questions, all_rankings

def ensure_cache_dir_of(data_root: str, train_filename: str) -> str:
    d = os.path.join(data_root, f".cache/{train_filename}")
    if not os.path.isdir(d):
        raise FileNotFoundError(f"[cache] not found: {d}\n"
                                f"Hint: run training script first to create this namespace.")
    return d

# ---------------- printer ----------------
def print_table(avg: Dict[int, Dict[str, float]], ks: List[int]):
    headers = ["@K", "Precision", "Recall", "F1", "Config-nDCG", "MRR"]
    print("\nEvaluation (utils.evaluate_agents) — averaged over questions")
    print("".join([f"{h:>14s}" for h in headers]))
    print("-" * (14 * len(headers)))
    for k in ks:
        m = avg[k]
        print(f"{('@'+str(k)):>14s}"
              f"{m['Precision']:14.4f}"
              f"{m['Recall']:14.4f}"
              f"{m['F1']:14.4f}"
              f"{m['Config-nDCG']:14.4f}"
              f"{m['MRR']:14.4f}")