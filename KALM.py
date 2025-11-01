#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
finetune_kalm_bi_encoder_agentrec.py
------------------------------------
用 Sentence-Transformers v3 训练 KaLM-Embedding (bi-encoder) 做
“一句话 Query → Agent/Tool” 内容匹配/检索。

数据协议：
{data_root}/PartI|PartII|PartIII/{agents,questions,rankings}/merge.json  +  {data_root}/Tools/merge.json
- agents[aid]: {"M": {"name": ...}, "T": {"tools": ["ToolA", ...]}}
- questions[qid]: {"input": "..."}  # query 文本
- rankings["rankings"][qid] = [aid1, aid2, ...]  # 从好到差排序（正例在前）

训练：
- Loss: MultipleNegativesRankingLoss（in-batch negatives）
- 训练样本： (query_text, pos_agent_text)；正样本来自排名前 pos_topk
- 可选：混入随机硬负 pairs（会增强难度，但 MNR 本身已用到 in-batch 负样本）
- 评测：InformationRetrievalEvaluator（q → 正相关 aid 集）

输出：
- {save_dir}/st-biencoder/ （Sentence-Transformers 格式，可直接 encode()）
- 控制台打印：IR 指标（MAP/MRR/nDCG/Recall 等）

依赖版本（建议）：
- sentence-transformers >= 3.0
- transformers >= 4.42
- torch >= 2.2
"""

import os, json, argparse, random
from typing import Dict, List, Tuple
from dataclasses import dataclass

import numpy as np
from tqdm.auto import tqdm

# ==== v2.x 兼容训练与评测 ====
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses, models, evaluation

from peft import LoraConfig, TaskType
from sentence_transformers import SentenceTransformer, InputExample, losses, models, evaluation


# -------------------- 数据加载与组装 --------------------

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
        # rankings 文件里通常包了一层 {"rankings": {...}}
        rmap = rankings["rankings"] if "rankings" in rankings else rankings
        all_rankings.update(rmap)
    tools = load_json(os.path.join(data_root, "Tools", "merge.json"))
    return all_agents, all_questions, all_rankings, tools

def tool_text(tname: str, tools: Dict[str, dict]) -> str:
    t = tools.get(tname, {}) or {}
    desc = t.get("description", "") or ""
    return f"{tname} {desc}".strip()

def agent_text(aid: str, all_agents: Dict[str, dict], tools: Dict[str, dict]) -> str:
    a = all_agents.get(aid, {}) or {}
    mname = (a.get("M", {}) or {}).get("name", "") or ""
    tlst  = (a.get("T", {}) or {}).get("tools", []) or []
    concat_tools = " | ".join([tool_text(t, tools) for t in tlst]) if tlst else ""
    text = (mname + (" || " if concat_tools else "") + concat_tools).strip(" |")
    return text or mname or "[EMPTY_AGENT]"

def build_text_corpora(all_agents, all_questions, tools):
    q_ids = list(all_questions.keys())
    q_texts = [(all_questions[qid].get("input","") or "") for qid in q_ids]
    a_ids = list(all_agents.keys())
    a_texts = [agent_text(aid, all_agents, tools) for aid in a_ids]
    return q_ids, q_texts, a_ids, a_texts

def train_valid_split(qids_in_rank, valid_ratio=0.2, seed=42):
    rng = random.Random(seed)
    q = list(qids_in_rank)
    rng.shuffle(q)
    n_valid = max(1, int(len(q) * valid_ratio))
    return q[n_valid:], q[:n_valid]

# -------------------- 样本构建（MNR） --------------------

@dataclass
class BuildSamplesConfig:
    pos_topk: int = 5      # 每个 q 取前 k 个正例
    max_pairs_per_q: int = 5
    seed: int = 42
    add_rand_hard_neg_pairs: bool = False
    rand_neg_ratio: float = 0.5  # 负样本配比（若开启）

def build_mnr_pairs(
    qids: List[str],
    all_rankings: Dict[str, List[str]],
    all_questions: Dict[str, dict],
    all_agents: Dict[str, dict],
    tools: Dict[str, dict],
    cfg: BuildSamplesConfig
) -> List[InputExample]:
    rng = random.Random(cfg.seed)
    # 预构造 agent 文本映射，避免重复拼装
    a_text_map = {}
    for aid in all_agents.keys():
        a_text_map[aid] = agent_text(aid, all_agents, tools)

    examples: List[InputExample] = []
    a_ids_all = list(all_agents.keys())
    a_set_all = set(a_ids_all)

    for qid in tqdm(qids, desc="Build MNR pairs"):
        qtext = (all_questions.get(qid, {}) or {}).get("input", "") or ""
        if not qtext:
            continue
        ranked = [aid for aid in (all_rankings.get(qid, []) or []) if aid in a_set_all]
        if not ranked:
            continue
        pos = ranked[:cfg.pos_topk]
        # 主要样本： (q, pos_agent)
        cnt = 0
        for aid in pos:
            examples.append(InputExample(texts=[qtext, a_text_map[aid]]))
            cnt += 1
            if cnt >= cfg.max_pairs_per_q:
                break

        # 可选：混入“伪负样本”的 (q, neg_agent) 对，增加难度（MNR 本身会用 batch 内其它样本作负）
        if cfg.add_rand_hard_neg_pairs:
            neg_pool = list(a_set_all - set(pos))
            if neg_pool:
                n_neg = int(cnt * cfg.rand_neg_ratio)
                for _ in range(n_neg):
                    na = rng.choice(neg_pool)
                    examples.append(InputExample(texts=[qtext, a_text_map[na]]))

    return examples

# -------------------- IR Evaluator --------------------

def build_ir_evaluator(
    eval_qids: List[str],
    all_rankings: Dict[str, List[str]],
    all_questions: Dict[str, dict],
    all_agents: Dict[str, dict],
    tools: Dict[str, dict],
    name: str = "agentrec"
):
    """
    构造 Sentence-Transformers 的 InformationRetrievalEvaluator：
      - queries: {qid: q_text}
      - corpus:  {aid: agent_text}
      - relevant_docs: {qid: {aid: 1, ...}}  # 正相关集合
    """
    queries = {}
    for qid in eval_qids:
        qtext = (all_questions.get(qid, {}) or {}).get("input", "") or ""
        if qtext:
            queries[qid] = qtext

    corpus = {}
    for aid in all_agents.keys():
        corpus[aid] = agent_text(aid, all_agents, tools)

    rel = {}
    a_set_all = set(all_agents.keys())
    for qid in eval_qids:
        ranked = [aid for aid in (all_rankings.get(qid, []) or []) if aid in a_set_all]
        if ranked:
            # 只把前 pos_topk 视作“相关集”（和训练一致）
            top_rel = ranked[:5]
            rel[qid] = {aid: 1 for aid in top_rel}

    return evaluation.InformationRetrievalEvaluator(
        queries=queries,
        corpus=corpus,
        relevant_docs=rel,
        name=name,
        show_progress_bar=True
    )

# -------------------- 主流程 --------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--save_dir", type=str, required=True)
    ap.add_argument("--model_name", type=str, default="KaLM-Embedding/KaLM-embedding-multilingual-mini-instruct-v2.5")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--warmup_ratio", type=float, default=0.05)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--max_seq_len", type=int, default=384)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--valid_ratio", type=float, default=0.2)
    ap.add_argument("--pos_topk", type=int, default=5)
    ap.add_argument("--pairs_per_q", type=int, default=5)
    ap.add_argument("--add_rand_neg", type=int, default=0)
    ap.add_argument("--neg_ratio", type=float, default=0.5)
    ap.add_argument("--eval_steps", type=int, default=1000)
    ap.add_argument("--use_amp", type=int, default=1)
    ap.add_argument("--grad_accum_steps", type=int, default=1)
    ap.add_argument("--save_ckpt", type=int, default=0, help="是否在训练中间保存 checkpoint")
    ap.add_argument("--tune_mode", type=str, default="lora",
                choices=["lora", "full", "frozen"],
                help="LoRA（参数高效微调）/ full（全参）/ frozen（仅推理，不训练）")
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--lora_target_modules", type=str, default="",
                    help="逗号分隔的子模块名，留空用PEFT对BERT/DeBERTa等的默认选择")

    args = ap.parse_args()

    random.seed(args.seed)
    np = __import__("numpy")
    np.random.seed(args.seed)

    os.makedirs(args.save_dir, exist_ok=True)
    out_dir = os.path.join(args.save_dir, "st-biencoder")
    os.makedirs(out_dir, exist_ok=True)

    # 1) 读数据
    all_agents, all_questions, all_rankings, tools = collect_data(args.data_root)
    q_ids, q_texts, a_ids, a_texts = build_text_corpora(all_agents, all_questions, tools)
    qids_in_rank = [qid for qid in q_ids if qid in all_rankings]

    train_qids, valid_qids = train_valid_split(qids_in_rank, args.valid_ratio, args.seed)
    print(f"[data] agents={len(all_agents)}  questions={len(all_questions)}")
    print(f"[split] train={len(train_qids)}  valid={len(valid_qids)}")

    # 2) 构造训练样本（MNR）
    cfg = BuildSamplesConfig(
        pos_topk=args.pos_topk,
        max_pairs_per_q=args.pairs_per_q,
        seed=args.seed,
        add_rand_hard_neg_pairs=bool(args.add_rand_neg),
        rand_neg_ratio=args.neg_ratio
    )
    train_samples = build_mnr_pairs(
        train_qids, all_rankings, all_questions, all_agents, tools, cfg
    )
    print(f"[train] InputExample count = {len(train_samples)}")

    # 3) 构造评测器
    evaluator = build_ir_evaluator(
        valid_qids, all_rankings, all_questions, all_agents, tools, name="agentrec"
    )


    # 1) 构造模型（保留 KaLM）
    model = SentenceTransformer(args.model_name, device=None, trust_remote_code=True)

    # --- Adapter/训练模式开关 ---
    model = SentenceTransformer(args.model_name, device=None)

    if args.tune_mode == "lora":
        # 若命令行没手动指定，就用 Qwen 的默认集合
        if args.lora_target_modules.strip():
            target_modules = [m.strip() for m in args.lora_target_modules.split(",") if m.strip()]
        else:
            target_modules = ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]

        print(f"[LoRA] using target_modules = {target_modules}")

        peft_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,   # Qwen = decoder-only
            inference_mode=False,
            r=args.lora_r,                  # 例如 16
            lora_alpha=args.lora_alpha,     # 例如 32
            lora_dropout=args.lora_dropout, # 例如 0.05
            target_modules=target_modules
        )
        model.add_adapter(peft_cfg)

    elif args.tune_mode == "frozen":
        for p in model.parameters():
            p.requires_grad = False

    elif args.tune_mode == "full":
        # 全参微调：不用改；保持默认即可
        pass


    # 覆盖最大长度和 padding
    if hasattr(model, "max_seq_length"):
        model.max_seq_length = args.max_seq_len
    if hasattr(model, "tokenizer"):
        tok = model.tokenizer
        if getattr(tok, "pad_token", None) is None:
            tok.pad_token = getattr(tok, "eos_token", None) or "[PAD]"
        tok.padding_side = "right"

    # 2) 训练样本（沿用上面 build_mnr_pairs 的结果）
    train_samples = build_mnr_pairs(
        train_qids, all_rankings, all_questions, all_agents, tools,
        BuildSamplesConfig(
            pos_topk=args.pos_topk,
            max_pairs_per_q=args.pairs_per_q,
            seed=args.seed,
            add_rand_hard_neg_pairs=bool(args.add_rand_neg),
            rand_neg_ratio=args.neg_ratio
        )
    )

    train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=args.batch_size, drop_last=True)
    train_loss = losses.MultipleNegativesRankingLoss(model)

    # 3) IR evaluator（v2 也支持）
    evaluator = build_ir_evaluator(
        valid_qids, all_rankings, all_questions, all_agents, tools, name="agentrec"
    )

    # 4) 训练（v2 经典 API）
    warmup_steps = int(len(train_dataloader) * args.epochs * args.warmup_ratio)
    out_dir = os.path.join(args.save_dir, "st-biencoder")
    os.makedirs(out_dir, exist_ok=True)

    print("[train] start (v2 .fit())")
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=args.epochs,
        warmup_steps=warmup_steps,
        optimizer_params={"lr": args.lr},
        weight_decay=args.weight_decay,
        scheduler="constantlr",  # 或 "warmuplinear"/"warmupcosine"
        evaluation_steps=args.eval_steps,
        output_path=out_dir,
        use_amp=bool(args.use_amp),
        evaluator=evaluator,
        show_progress_bar=True
    )
    print("[train] done")

    print(f"[save] Sentence-Transformers model -> {out_dir}")
    print("[eval] final evaluation")
    evaluator(model, output_path=os.path.join(out_dir, "eval"))


if __name__ == "__main__":
    # 建议环境变量（减少显存碎片）
    # export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    main()
