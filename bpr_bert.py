#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
simple_bpr_transformer_agent_rec.py  (with optional finetune + LoRA)

BPR + 负采样不变；默认冻结 Transformer（离线编码 + 缓存）。
新增：
- --tune_mode {frozen,full,lora}
- LoRA 轻量微调（对 Linear 注入低秩适配器）：--lora_r --lora_alpha --lora_dropout --lora_targets

Usage (frozen, 原路径不变):
python /mnt/data/simple_bpr_transformer_agent_rec.py \
  --data_root /path/to/dataset_root \
  --epochs 3 --batch_size 256 --pretrained_model distilbert-base-uncased \
  --max_len 128 --text_hidden 256 --id_dim 64 --neg_per_pos 1 --topk 10 \
  --device cuda:0

Enable full finetune:
  --tune_mode full

Enable LoRA:
  --tune_mode lora --lora_r 8 --lora_alpha 16 --lora_dropout 0.1 \
  --lora_targets query,key,value,dense
  
下面按「Query / Agent / Tool」把这份 simple_bpr_transformer_agent_rec.py 实际用到的特征说清楚（就是进入打分 forward_score 的那几路信号）：

Query（问题）

文本语义向量：将 questions[qid]["input"] 用 HuggingFace Transformer（默认 DistilBERT）编码为句向量（默认取 [CLS]，可切换 mean-pool），再过一层线性投影 q_proj + ReLU 得到 qh。

✅ 没有使用 query 的 ID 向量、历史行为、时间等额外特征。

Agent（智能体）

文本语义向量：Agent 文本由 M.name 拼接其所挂工具（tools）的“工具名 + 描述”组成：f"{M.name} {tool_1_name desc_1} ..."；同样用同一 Transformer 编码，再经 a_proj + ReLU 得到 ah。

Agent ID 嵌入：emb_agent[agent_idx]，与文本向量并联做为显式 ID 特征。

⚠️ 代码里没有用到 agent 自身的“描述字段”（如果有的话），而是用 M.name + 工具文本来代表 agent 文本侧信息。

Tool（工具）

工具 ID 嵌入聚合：为每个 agent 取其工具列表的 工具 ID 嵌入 emb_tool[tool_id]，按掩码做 均值池化 得到 te_mean，作为“工具存在性/组合”的离散特征。

工具文本的作用路径：工具的“名字+描述”已被并入 agent 文本，在上面的 Agent 文本语义向量里起作用；在打分阶段并没有单独再编码工具文本。
"""


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
simple_bpr_transformer_agent_rec.py  (with optional finetune + LoRA + partial-unfreeze)

Additions in this version:
- --tune_mode {frozen,full,lora} (existing) still supported
- NEW: --unfreeze_last_n (only for tune_mode=full): unfreezes the last N transformer blocks; 0 => unfreeze all
- NEW: --unfreeze_emb (0/1): also unfreeze word embeddings & LayerNorms when using partial unfreeze
- NEW: --encoder_lr: separate LR for encoder params; model head still uses --lr
- NEW: --encoder_weight_decay: weight decay for encoder param group
- NEW: --grad_ckpt (0/1): enable gradient checkpointing if the encoder supports it
- NEW: --pooling {cls,mean}: control sentence pooling strategy for query/agent texts
- LoRA path unchanged (targets via --lora_targets)

Usage (keep frozen, cache sentence vectors):
python /mnt/data/simple_bpr_transformer_agent_rec_finetune.py \
  --data_root /path/to/dataset_root \
  --epochs 3 --batch_size 256 --pretrained_model distilbert-base-uncased \
  --max_len 128 --text_hidden 256 --id_dim 64 --neg_per_pos 1 --topk 10 \
  --pooling cls --device cuda:0

Enable full finetune (all layers):
  --tune_mode full --unfreeze_last_n 0 --encoder_lr 5e-5

Enable partial finetune (last 2 blocks):
  --tune_mode full --unfreeze_last_n 2 --encoder_lr 5e-5

Enable LoRA:
  --tune_mode lora --lora_r 8 --lora_alpha 16 --lora_dropout 0.1 \
  --lora_targets query,key,value,dense --encoder_lr 5e-5
"""

import os, json, math, random, argparse, zlib, pickle
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

from transformers import AutoTokenizer, AutoModel

filename = os.path.splitext(os.path.basename(__file__))[0]
pos_topk = 5  # 仍只把每个问题前K的 agent 当正样本

# -------------------- 小工具 --------------------
def print_metrics_table(title, metrics, ks=(5,10,50), filename="script"):
    print(f"\n[{filename}] {title}")
    header = "k\tP\tR\tF1\tHit\tnDCG\tMRR"
    print(header)
    for k in ks:
        m = metrics[k]
        print(f"{k}\t{m['P']:.4f}\t{m['R']:.4f}\t{m['F1']:.4f}\t{m['Hit']:.4f}\t{m['nDCG']:.4f}\t{m['MRR']:.4f}")

# -------------------- 常用 I/O --------------------
def model_save_paths(cache_dir: str, data_sig: str):
    model_dir = os.path.join(cache_dir, "models"); os.makedirs(model_dir, exist_ok=True)
    return (os.path.join(model_dir, f"{filename}_{data_sig}.pt"),
            os.path.join(model_dir, f"q_encoder_{data_sig}.pkl"),  # 保留命名
            os.path.join(model_dir, f"latest_{data_sig}.pt"),
            os.path.join(model_dir, f"meta_{data_sig}.json"))

def ensure_cache_dir(root: str) -> str:
    d = os.path.join(root, f".cache/{filename}_transformer"); os.makedirs(d, exist_ok=True); return d

def cache_exists(cache_dir: str) -> bool:
    needed = ["q_ids.json","a_ids.json","tool_names.json","Q_emb.npy","A_emb.npy",
              "agent_tool_idx_padded.npy","agent_tool_mask.npy","enc_meta.json"]
    return all(os.path.exists(os.path.join(cache_dir, f)) for f in needed)

def save_cache(cache_dir: str, q_ids,a_ids,tool_names, Q_emb,A_emb, agent_tool_idx_padded,agent_tool_mask, enc_meta):
    with open(os.path.join(cache_dir,"q_ids.json"),"w",encoding="utf-8") as f: json.dump(q_ids,f,ensure_ascii=False)
    with open(os.path.join(cache_dir,"a_ids.json"),"w",encoding="utf-8") as f: json.dump(a_ids,f,ensure_ascii=False)
    with open(os.path.join(cache_dir,"tool_names.json"),"w",encoding="utf-8") as f: json.dump(tool_names,f,ensure_ascii=False)
    np.save(os.path.join(cache_dir,"Q_emb.npy"), Q_emb.astype(np.float32))
    np.save(os.path.join(cache_dir,"A_emb.npy"), A_emb.astype(np.float32))
    np.save(os.path.join(cache_dir,"agent_tool_idx_padded.npy"), agent_tool_idx_padded.astype(np.int64))
    np.save(os.path.join(cache_dir,"agent_tool_mask.npy"), agent_tool_mask.astype(np.float32))
    with open(os.path.join(cache_dir,"enc_meta.json"),"w",encoding="utf-8") as f: json.dump(enc_meta,f,ensure_ascii=False)

def load_cache(cache_dir: str):
    with open(os.path.join(cache_dir,"q_ids.json"),"r",encoding="utf-8") as f: q_ids=json.load(f)
    with open(os.path.join(cache_dir,"a_ids.json"),"r",encoding="utf-8") as f: a_ids=json.load(f)
    with open(os.path.join(cache_dir,"tool_names.json"),"r",encoding="utf-8") as f: tool_names=json.load(f)
    with open(os.path.join(cache_dir,"enc_meta.json"),"r",encoding="utf-8") as f: enc_meta=json.load(f)
    Q_emb = np.load(os.path.join(cache_dir,"Q_emb.npy"))
    A_emb = np.load(os.path.join(cache_dir,"A_emb.npy"))
    agent_tool_idx_padded = np.load(os.path.join(cache_dir,"agent_tool_idx_padded.npy"))
    agent_tool_mask = np.load(os.path.join(cache_dir,"agent_tool_mask.npy"))
    return (q_ids,a_ids,tool_names,Q_emb,A_emb,agent_tool_idx_padded,agent_tool_mask,enc_meta)

def load_json(p: str):
    with open(p,"r",encoding="utf-8") as f: return json.load(f)

# -------------------- 数据拼装 --------------------
def collect_data(data_root: str):
    parts = ["PartI","PartII","PartIII"]
    all_agents, all_questions, all_rankings = {}, {}, {}
    for part in parts:
        agents = load_json(os.path.join(data_root, part,"agents","merge.json"))
        questions = load_json(os.path.join(data_root, part,"questions","merge.json"))
        rankings = load_json(os.path.join(data_root, part,"rankings","merge.json"))
        all_agents.update(agents); all_questions.update(questions); all_rankings.update(rankings["rankings"])
    tools = load_json(os.path.join(data_root,"Tools","merge.json"))
    return all_agents, all_questions, all_rankings, tools

def build_text_corpora(all_agents, all_questions, tools):
    q_ids = list(all_questions.keys())
    q_texts = [all_questions[qid]["input"] for qid in q_ids]

    tool_names = list(tools.keys())
    def _tool_text(tn: str) -> str:
        t = tools.get(tn, {}); desc = t.get("description",""); return f"{tn} {desc}".strip()

    a_ids, a_texts, a_tool_lists = [], [], []
    for aid, a in all_agents.items():
        mname = a.get("M",{}).get("name","")
        tool_list = a.get("T",{}).get("tools",[]) or []
        a_ids.append(aid); a_tool_lists.append(tool_list)
        concat_tool_desc = " ".join([_tool_text(tn) for tn in tool_list])
        a_texts.append(f"{mname} {concat_tool_desc}".strip())
    return q_ids,q_texts,tool_names,a_ids,a_texts,a_tool_lists

# -------------------- Transformer 编码 --------------------
@torch.no_grad()
def encode_texts(texts: List[str], tokenizer, encoder, device, max_len=128, batch_size=256, pooling="cls"):
    embs = []
    use_cls = (pooling == "cls")
    for i in tqdm(range(0,len(texts),batch_size), desc="Encoding with Transformer", dynamic_ncols=True):
        batch = texts[i:i+batch_size]
        toks = tokenizer(batch, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
        toks = {k:v.to(device) for k,v in toks.items()}
        out = encoder(**toks)
        if hasattr(out, "last_hidden_state"):
            if use_cls:
                vec = out.last_hidden_state[:,0,:]             # [CLS]
            else:
                attn = toks["attention_mask"].unsqueeze(-1)     # mean-pooling
                sum_h = (out.last_hidden_state * attn).sum(1)
                vec = sum_h / (attn.sum(1).clamp(min=1))
        else:
            vec = out.pooler_output
        embs.append(vec.detach().cpu())
    return torch.cat(embs, dim=0).numpy()

def encode_batch(tokenizer, encoder, texts: List[str], device, max_len=128, pooling="cls"):
    toks = tokenizer(texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
    toks = {k:v.to(device) for k,v in toks.items()}
    out = encoder(**toks)
    if hasattr(out, "last_hidden_state"):
        if pooling == "cls":
            vec = out.last_hidden_state[:,0,:]
        else:
            attn = toks["attention_mask"].unsqueeze(-1)
            sum_h = (out.last_hidden_state * attn).sum(1)
            vec = sum_h / (attn.sum(1).clamp(min=1))
    else:
        vec = out.pooler_output
    return vec  # [B, d_text]

def build_agent_tool_id_buffers(a_ids: List[str], agent_tool_lists: List[List[str]], tool_names: List[str]):
    t_map = {n:i for i,n in enumerate(tool_names)}
    num_agents = len(a_ids)
    max_t = max([len(lst) for lst in agent_tool_lists]) if num_agents>0 else 0
    max_t = max_t if max_t>0 else 1
    idx_pad = np.zeros((num_agents,max_t),dtype=np.int64)
    mask = np.zeros((num_agents,max_t),dtype=np.float32)
    for i,lst in enumerate(agent_tool_lists):
        for j,tn in enumerate(lst[:max_t]):
            if tn in t_map:
                idx_pad[i,j]=t_map[tn]; mask[i,j]=1.0
    return torch.from_numpy(idx_pad), torch.from_numpy(mask)

# -------------------- LoRA 模块（轻量实现） --------------------
class LoRALinear(nn.Module):
    def __init__(self, base_linear: nn.Linear, r=8, alpha=16, dropout=0.0):
        super().__init__()
        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / max(1, r)
        # 冻结原始权重
        self.weight = base_linear.weight
        self.bias = base_linear.bias
        for p in (self.weight, self.bias):
            if p is not None: p.requires_grad = False
        # LoRA 低秩分解
        self.A = nn.Parameter(torch.zeros((r, self.in_features)))
        self.B = nn.Parameter(torch.zeros((self.out_features, r)))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.zeros_(self.B)
        self.dropout = nn.Dropout(dropout) if dropout and dropout>0 else nn.Identity()

    def forward(self, x):
        base = F.linear(x, self.weight, self.bias)
        lora = (self.B @ (self.A @ x.transpose(-2,-1))).transpose(-2,-1)  # [*, out]
        return base + self.dropout(lora) * self.scaling

def apply_lora_to_encoder(encoder: nn.Module, target_keywords: List[str], r=8, alpha=16, dropout=0.0):
    """将包含关键词的 nn.Linear 替换为 LoRALinear (BERT/DistilBERT/Roberta 适用)"""
    repl = 0
    for name, module in list(encoder.named_modules()):
        for child_name, child in list(module.named_children()):
            if isinstance(child, nn.Linear) and any(k in child_name.lower() for k in target_keywords):
                wrapped = LoRALinear(child, r=r, alpha=alpha, dropout=dropout)
                setattr(module, child_name, wrapped)
                repl += 1
    print(f"[LoRA] injected into {repl} Linear layers (targets={target_keywords}, r={r}, alpha={alpha}, dropout={dropout})")

# -------------------- Finetune 细粒度控制 --------------------

def _get_transformer_layers(model: nn.Module):
    """Return (path, layers_list) for common HF encoders."""
    # BERT / RoBERTa
    if hasattr(model, "encoder") and hasattr(model.encoder, "layer"):
        return "encoder.layer", model.encoder.layer
    # RoBERTa often nests under .roberta
    if hasattr(model, "roberta") and hasattr(model.roberta, "encoder") and hasattr(model.roberta.encoder, "layer"):
        return "roberta.encoder.layer", model.roberta.encoder.layer
    # DistilBERT
    if hasattr(model, "transformer") and hasattr(model.transformer, "layer"):
        return "transformer.layer", model.transformer.layer
    return None, None


def set_finetune_scope(encoder: nn.Module, unfreeze_last_n: int = 0, unfreeze_emb: bool = False):
    """
    Freeze all encoder params, then unfreeze the last N blocks (and optionally embeddings/LayerNorms).
    If unfreeze_last_n == 0: unfreeze all blocks (true full finetune).
    """
    for p in encoder.parameters():
        p.requires_grad = False

    path, layers = _get_transformer_layers(encoder)
    if layers is None:
        print("[warn] could not locate transformer layers; keeping encoder frozen")
        return

    if unfreeze_last_n <= 0 or unfreeze_last_n >= len(layers):
        # unfreeze all
        for p in encoder.parameters():
            p.requires_grad = True
    else:
        # unfreeze last N blocks
        for block in layers[-unfreeze_last_n:]:
            for p in block.parameters():
                p.requires_grad = True

    if unfreeze_emb:
        # common embedding places
        if hasattr(encoder, "embeddings"):
            for p in encoder.embeddings.parameters():
                p.requires_grad = True
        if hasattr(encoder, "roberta") and hasattr(encoder.roberta, "embeddings"):
            for p in encoder.roberta.embeddings.parameters():
                p.requires_grad = True
        if hasattr(encoder, "distilbert") and hasattr(encoder.distilbert, "embeddings"):
            for p in encoder.distilbert.embeddings.parameters():
                p.requires_grad = True
        # also unfreeze top-level LayerNorm modules if present
        for m in encoder.modules():
            if isinstance(m, nn.LayerNorm):
                for p in m.parameters():
                    if not p.requires_grad:
                        p.requires_grad = True

# -------------------- BPR 模型 --------------------
class TransformerBPR(nn.Module):
    def __init__(self, d_text: int, num_agents: int, num_tools: int,
                 agent_tool_indices_padded: torch.LongTensor,
                 agent_tool_mask: torch.FloatTensor,
                 text_hidden: int = 256, id_dim: int = 64):
        super().__init__()
        self.q_proj = nn.Linear(d_text, text_hidden)
        self.a_proj = nn.Linear(d_text, text_hidden)
        self.emb_agent = nn.Embedding(num_agents, id_dim)
        self.emb_tool = nn.Embedding(num_tools, id_dim)
        self.register_buffer("agent_tool_indices_padded", agent_tool_indices_padded)
        self.register_buffer("agent_tool_mask", agent_tool_mask)
        in_dim = text_hidden + text_hidden + id_dim + id_dim
        self.scorer = nn.Sequential(nn.ReLU(), nn.Linear(in_dim,128), nn.ReLU(), nn.Linear(128,1))
        # init
        for m in [self.q_proj,self.a_proj,self.emb_agent,self.emb_tool]:
            if isinstance(m, nn.Embedding):
                nn.init.xavier_uniform_(m.weight)
            else:
                nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)
        for m in self.scorer:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)

    def forward_score(self, q_vec, a_vec, agent_idx):
        qh = F.relu(self.q_proj(q_vec))
        ah = F.relu(self.a_proj(a_vec))
        ae = self.emb_agent(agent_idx)
        idxs = self.agent_tool_indices_padded[agent_idx]
        mask = self.agent_tool_mask[agent_idx]
        te = self.emb_tool(idxs)
        te_mean = (te * mask.unsqueeze(-1)).sum(1) / (mask.sum(1,keepdim=True)+1e-8)
        x = torch.cat([qh,ah,ae,te_mean], dim=1)
        return self.scorer(x).squeeze(1)

def bpr_loss(pos, neg):
    return -torch.log(torch.sigmoid(pos - neg) + 1e-8).mean()

# -------------------- 训练数据与评测 --------------------
def train_valid_split(qids_in_rankings, valid_ratio=0.2, seed=42):
    rng = random.Random(seed); q=list(qids_in_rankings); rng.shuffle(q)
    n_valid = int(len(q)*valid_ratio); return q[n_valid:], q[:n_valid]

def build_training_pairs(all_rankings: Dict[str,List[str]], all_agent_ids: List[str], neg_per_pos=1, rng_seed=42):
    rnd = random.Random(rng_seed); pairs=[]; all_set=set(all_agent_ids)
    for qid, ranked_full in all_rankings.items():
        ranked = ranked_full[:pos_topk]; rset=set(ranked)
        neg_pool = list(all_set - rset) or all_agent_ids
        for pos_a in ranked:
            for _ in range(neg_per_pos):
                pairs.append((qid, pos_a, rnd.choice(neg_pool)))
    return pairs, list(all_rankings.keys())

@torch.no_grad()
def evaluate_model(model, Q_t, A_t, qid2idx, a_ids, all_rankings, eval_qids, device="cpu",
                   ks=(5,10,50), cand_size=1000, rng_seed=123):
    import numpy as np, math
    max_k = max(ks); aid2idx={aid:i for i,aid in enumerate(a_ids)}
    agg = {k: {"P":0.0,"R":0.0,"F1":0.0,"Hit":0.0,"nDCG":0.0,"MRR":0.0} for k in ks}
    cnt=skipped=0; all_agent_set=set(a_ids); ref_k = 10 if 10 in ks else max_k
    pbar = tqdm(eval_qids, desc="Evaluating (sampled)", dynamic_ncols=True)
    for qid in pbar:
        gt_list = [aid for aid in all_rankings.get(qid, [])[:pos_topk] if aid in aid2idx]
        if not gt_list: skipped+=1; pbar.set_postfix({"done":cnt,"skipped":skipped}); continue
        rel_set=set(gt_list); neg_pool = list(all_agent_set - rel_set)
        rnd = random.Random((hash(qid) ^ (rng_seed*16777619)) & 0xFFFFFFFF)
        need_neg = max(0, cand_size - len(gt_list))
        if need_neg>0 and len(neg_pool)>0:
            sampled_negs = rnd.sample(neg_pool, min(need_neg,len(neg_pool)))
            cand_ids = gt_list + sampled_negs
        else:
            cand_ids = gt_list
        qi = qid2idx[qid]
        qv = Q_t[qi:qi+1].repeat(len(cand_ids),1)
        cand_idx = torch.tensor([aid2idx[a] for a in cand_ids], dtype=torch.long, device=device)
        av = A_t[cand_idx]
        scores = model.forward_score(qv,av,cand_idx).detach().cpu().numpy()
        order = np.argsort(-scores)[:max_k]; pred_ids=[cand_ids[i] for i in order]
        bin_hits = [1 if aid in rel_set else 0 for aid in pred_ids]
        for k in ks:
            Hk=sum(bin_hits[:k]); P=Hk/float(k); R=Hk/float(len(rel_set))
            F1=(2*P*R)/(P+R) if (P+R)>0 else 0.0; Hit=1.0 if Hk>0 else 0.0
            dcg = sum((1.0/math.log2(i+2.0)) for i,h in enumerate(bin_hits[:k]) if h)
            ideal = min(len(rel_set), k)
            idcg = sum(1.0/math.log2(i+2.0) for i in range(ideal)) if ideal>0 else 0.0
            nDCG = (dcg/idcg) if idcg>0 else 0.0
            rr=0.0
            for i in range(k):
                if bin_hits[i]: rr=1.0/float(i+1); break
            agg[k]["P"]+=P; agg[k]["R"]+=R; agg[k]["F1"]+=F1; agg[k]["Hit"]+=Hit; agg[k]["nDCG"]+=nDCG; agg[k]["MRR"]+=rr
        cnt+=1
        ref=agg[ref_k]; pbar.set_postfix({"done":cnt,"skipped":skipped,f"P@{ref_k}":f"{(ref['P']/cnt):.4f}",
                                          f"nDCG@{ref_k}":f"{(ref['nDCG']/cnt):.4f}", f"MRR@{ref_k}":f"{(ref['MRR']/cnt):.4f}",
                                          "Ncand":len(cand_ids)})
    if cnt==0:
        return {k:{m:0.0 for m in ["P","R","F1","Hit","nDCG","MRR"]} for k in ks}
    for k in ks:
        for m in agg[k]: agg[k][m]/=cnt
    return agg

def dataset_signature(a_ids: List[str], all_rankings: Dict[str,List[str]]) -> str:
    payload = {"a_ids": a_ids, "rankings": {k: all_rankings[k] for k in sorted(all_rankings.keys())}}
    sig = zlib.crc32(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")) & 0xFFFFFFFF
    return f"{sig:08x}"

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

# -------------------- 主流程 --------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--pretrained_model", type=str, default="distilbert-base-uncased")
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--text_hidden", type=int, default=256)
    parser.add_argument("--id_dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3, help="LR for BPR head / embeddings")
    parser.add_argument("--encoder_lr", type=float, default=5e-5, help="LR for Transformer encoder when finetuning")
    parser.add_argument("--encoder_weight_decay", type=float, default=0.0)
    parser.add_argument("--neg_per_pos", type=int, default=1)
    parser.add_argument("--rng_seed_pairs", type=int, default=42)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--valid_ratio", type=float, default=0.2)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--rebuild_cache", type=int, default=0)
    parser.add_argument("--pooling", type=str, choices=["cls","mean"], default="cls")
    # 微调模式 & LoRA
    parser.add_argument("--tune_mode", type=str, choices=["frozen","full","lora"], default="frozen",
                        help="frozen: 离线缓存句向量; full/lora: 端到端在线编码并反传")
    parser.add_argument("--unfreeze_last_n", type=int, default=0, help="full 模式下仅解冻最后 N 个 Transformer block，0 = 全部解冻")
    parser.add_argument("--unfreeze_emb", type=int, default=0, help="full 模式下是否同时解冻 embeddings 和 LayerNorm (0/1)")
    parser.add_argument("--grad_ckpt", type=int, default=0, help="是否对 encoder 启用 gradient checkpointing (0/1)")
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--lora_targets", type=str, default="query,key,value,dense",
                        help="逗号分隔子串，匹配 Linear 名称进行 LoRA 注入")
    args = parser.parse_args()

    random.seed(1234); np.random.seed(1234); torch.manual_seed(1234)
    device = torch.device(args.device)

    # 1) 读原始数据
    all_agents, all_questions, all_rankings, tools = collect_data(args.data_root)
    print(f"Loaded {len(all_agents)} agents, {len(all_questions)} questions, {len(tools)} tools.")

    cache_dir = ensure_cache_dir(args.data_root)

    # 2) 文本与工具列表
    q_ids, q_texts, tool_names, a_ids, a_texts, a_tool_lists = build_text_corpora(all_agents, all_questions, tools)
    agent_tool_idx_padded, agent_tool_mask = build_agent_tool_id_buffers(a_ids, a_tool_lists, tool_names)

    # 3) 准备编码器（根据 tune_mode 决定是否用缓存）
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)
    encoder = AutoModel.from_pretrained(args.pretrained_model).to(device)

    if args.grad_ckpt:
        try:
            encoder.gradient_checkpointing_enable()
            print("[encoder] gradient checkpointing enabled")
        except Exception as e:
            print(f"[encoder] gradient checkpointing not supported: {e}")

    if args.tune_mode == "lora":
        targets = [s.strip().lower() for s in args.lora_targets.split(",") if s.strip()]
        apply_lora_to_encoder(encoder, targets, r=args.lora_r, alpha=args.lora_alpha, dropout=args.lora_dropout)
    
    if args.tune_mode == "full":
        set_finetune_scope(encoder, unfreeze_last_n=args.unfreeze_last_n, unfreeze_emb=bool(args.unfreeze_emb))
    elif args.tune_mode == "frozen":
        for p in encoder.parameters():
            p.requires_grad = False
    else:  # lora
        # LoRALinear 内部已冻结 base weights，仅 A/B 可训练
        for p in encoder.parameters():
            # respect wrappers' requires_grad flags
            pass

    use_cache = (args.tune_mode == "frozen")
    Q_emb = A_emb = None

    if use_cache:
        # 尝试加载缓存
        if cache_exists(cache_dir) and args.rebuild_cache == 0:
            (q_ids_c,a_ids_c,tool_names_c,Q_emb,A_emb,
             agent_tool_idx_padded_np, agent_tool_mask_np, enc_meta) = load_cache(cache_dir)
            if (q_ids_c==q_ids) and (a_ids_c==a_ids) and (tool_names_c==tool_names) and \
               (enc_meta.get("pretrained_model")==args.pretrained_model) and (enc_meta.get("max_len")==args.max_len) and \
               (enc_meta.get("pooling", "cls") == args.pooling):
                print(f"[cache] loaded transformer embeddings from {cache_dir}")
                agent_tool_idx_padded = torch.from_numpy(agent_tool_idx_padded_np)
                agent_tool_mask = torch.from_numpy(agent_tool_mask_np)
            else:
                print("[cache] mismatch; rebuilding embeddings...")
                Q_emb = A_emb = None
        if Q_emb is None or A_emb is None:
            encoder.eval()
            with torch.no_grad():
                Q_emb = encode_texts(q_texts, tokenizer, encoder, device, max_len=args.max_len, batch_size=256, pooling=args.pooling)
                A_emb = encode_texts(a_texts, tokenizer, encoder, device, max_len=args.max_len, batch_size=256, pooling=args.pooling)
            save_cache(
                cache_dir, q_ids,a_ids,tool_names, Q_emb,A_emb,
                agent_tool_idx_padded.numpy(), agent_tool_mask.numpy(),
                enc_meta={"pretrained_model": args.pretrained_model, "max_len": args.max_len, "pooling": args.pooling}
            )
            print(f"[cache] saved transformer embeddings to {cache_dir}")
    else:
        encoder.train()  # full 或 lora

    # 4) 索引映射
    qid2idx = {qid:i for i,qid in enumerate(q_ids)}
    aid2idx = {aid:i for i,aid in enumerate(a_ids)}

    # 5) 数据签名与训练缓存
    data_sig = dataset_signature(a_ids, all_rankings)
    model_path, encoder_stub_path, latest_model, meta_path = model_save_paths(cache_dir, data_sig)

    want_meta = {
        "data_sig": data_sig,
        "neg_per_pos": int(args.neg_per_pos),
        "rng_seed_pairs": int(args.rng_seed_pairs),
        "split_seed": int(args.split_seed),
        "valid_ratio": float(args.valid_ratio),
    }
    use_training_cache_flag = training_cache_exists(cache_dir) and (args.rebuild_cache == 0)
    pairs = None

    if use_training_cache_flag:
        cached_train_qids, cached_valid_qids, pairs_idx_np, meta = load_training_cache(cache_dir)
        if meta == want_meta:
            train_qids, valid_qids = cached_train_qids, cached_valid_qids
            pairs = [(int(q), int(p), int(n)) for (q,p,n) in pairs_idx_np.tolist()]
            print(f"[cache] loaded train/valid/pairs from {cache_dir} (sig={data_sig})")
        else:
            print("[cache] training cache meta mismatch, rebuilding...")

    if pairs is None:
        qids_in_rank = [qid for qid in q_ids if qid in all_rankings]
        train_qids, valid_qids = train_valid_split(qids_in_rank, valid_ratio=args.valid_ratio, seed=args.split_seed)
        print(f"[split] train={len(train_qids)}  valid={len(valid_qids)}")
        rankings_train = {qid: all_rankings[qid] for qid in train_qids}
        pairs_id, _ = build_training_pairs(rankings_train, a_ids, neg_per_pos=args.neg_per_pos, rng_seed=args.rng_seed_pairs)
        pairs = [(qid2idx[q], aid2idx[p], aid2idx[n]) for (q,p,n) in pairs_id]
        save_training_cache(cache_dir, train_qids, valid_qids, np.array(pairs, dtype=np.int64), want_meta)
        print(f"[cache] saved train/valid/pairs to {cache_dir} (sig={data_sig})")

    # 6) 模型与张量
    if use_cache:
        d_text = int(Q_emb.shape[1])
    else:
        encoder.eval()
        with torch.no_grad():
            tmp = encode_batch(tokenizer, encoder, ["hello"], device, max_len=args.max_len, pooling=args.pooling)
            d_text = int(tmp.shape[1])
        encoder.train() if args.tune_mode != "frozen" else encoder.eval()

    num_agents, num_tools = len(a_ids), len(tool_names)
    model = TransformerBPR(
        d_text=d_text, num_agents=num_agents, num_tools=num_tools,
        agent_tool_indices_padded=agent_tool_idx_padded.to(device),
        agent_tool_mask=agent_tool_mask.to(device),
        text_hidden=args.text_hidden, id_dim=args.id_dim
    ).to(device)

    # 7) 优化器 & 参数组
    head_params = list(model.parameters())
    if args.tune_mode == "frozen":
        encoder_params = []
        encoder.eval()
        for p in encoder.parameters(): p.requires_grad = False
    else:
        encoder_params = [p for p in encoder.parameters() if p.requires_grad]

    param_groups = [
        {"params": head_params, "lr": args.lr},
    ]
    if encoder_params:
        param_groups.append({
            "params": encoder_params,
            "lr": args.encoder_lr,
            "weight_decay": args.encoder_weight_decay,
        })

    optimizer = torch.optim.Adam(param_groups)

    if use_cache:
        Q_t = torch.from_numpy(Q_emb).to(device)
        A_t = torch.from_numpy(A_emb).to(device)
    else:
        Q_t = A_t = None  # 在线编码路径

    # 8) 训练
    num_pairs = len(pairs)
    num_batches = math.ceil(num_pairs / args.batch_size)
    for epoch in range(1, args.epochs+1):
        random.shuffle(pairs); total_loss=0.0
        pbar = tqdm(range(num_batches), desc=f"Epoch {epoch}/{args.epochs}", dynamic_ncols=True)
        for b in pbar:
            batch = pairs[b*args.batch_size:(b+1)*args.batch_size]
            if not batch: continue
            q_idx = torch.tensor([t[0] for t in batch], dtype=torch.long, device=device)
            pos_idx = torch.tensor([t[1] for t in batch], dtype=torch.long, device=device)
            neg_idx = torch.tensor([t[2] for t in batch], dtype=torch.long, device=device)

            if use_cache:
                q_vec = Q_t[q_idx]
                pos_vec = A_t[pos_idx]
                neg_vec = A_t[neg_idx]
            else:
                # 在线编码（full / lora）
                uniq_q, inv_q = torch.unique(q_idx, sorted=True, return_inverse=True)
                uniq_pos, inv_pos = torch.unique(pos_idx, sorted=True, return_inverse=True)
                uniq_neg, inv_neg = torch.unique(neg_idx, sorted=True, return_inverse=True)

                q_text_batch = [q_texts[i] for i in uniq_q.tolist()]
                pos_text_batch = [a_texts[i] for i in uniq_pos.tolist()]
                neg_text_batch = [a_texts[i] for i in uniq_neg.tolist()]

                q_vec_uniq = encode_batch(tokenizer, encoder, q_text_batch, device, max_len=args.max_len, pooling=args.pooling)
                pos_vec_uniq = encode_batch(tokenizer, encoder, pos_text_batch, device, max_len=args.max_len, pooling=args.pooling)
                neg_vec_uniq = encode_batch(tokenizer, encoder, neg_text_batch, device, max_len=args.max_len, pooling=args.pooling)

                q_vec = q_vec_uniq[inv_q]
                pos_vec = pos_vec_uniq[inv_pos]
                neg_vec = neg_vec_uniq[inv_neg]

            pos_score = model.forward_score(q_vec, pos_vec, pos_idx)
            neg_score = model.forward_score(q_vec, neg_vec, neg_idx)
            loss = bpr_loss(pos_score, neg_score)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({"batch_loss": f"{loss.item():.4f}",
                              "avg_loss": f"{(total_loss/(b+1)):.4f}"})
        print(f"Epoch {epoch}/{args.epochs} - BPR loss: {(total_loss/max(1,num_batches)):.4f}")

    # 9) （可选）训练后全量重编码（full / lora）
    if not use_cache:
        encoder.eval()
        with torch.no_grad():
            Q_emb_eval = encode_texts(q_texts, tokenizer, encoder, device, max_len=args.max_len, batch_size=256, pooling=args.pooling)
            A_emb_eval = encode_texts(a_texts, tokenizer, encoder, device, max_len=args.max_len, batch_size=256, pooling=args.pooling)
        Q_t = torch.from_numpy(Q_emb_eval).to(device)
        A_t = torch.from_numpy(A_emb_eval).to(device)

    # 10) 持久化
    ckpt = {
        "state_dict": model.state_dict(),
        "data_sig": data_sig,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "args": vars(args),
        "dims": {"d_text": int(Q_t.shape[1]), "num_agents": int(num_agents),
                 "num_tools": int(num_tools), "text_hidden": int(args.text_hidden),
                 "id_dim": int(args.id_dim)},
        "pretrained_model": args.pretrained_model,
        "tune_mode": args.tune_mode,
    }
    torch.save(ckpt, model_path); torch.save(ckpt, latest_model)

    serving_meta = {"data_sig": data_sig, "a_ids": a_ids, "tool_names": tool_names,
                    "pretrained_model": args.pretrained_model, "tune_mode": args.tune_mode}
    with open(meta_path,"w",encoding="utf-8") as f: json.dump(serving_meta,f,ensure_ascii=False,indent=2)

    print(f"[save] model -> {model_path}")
    print(f"[save] meta  -> {meta_path}")

    # 11) 推理 + 验证（统一用最新的 Q_t/A_t）
    def recommend_topk_for_qid(qid: str, topk: int=10):
        qi = qid2idx[qid]
        qv = Q_t[qi:qi+1].repeat(len(a_ids),1)
        a_idx = torch.arange(len(a_ids), dtype=torch.long, device=device)
        with torch.no_grad():
            scores = model.forward_score(qv, A_t, a_idx).cpu().numpy()
        order = np.argsort(-scores)[:topk]
        return [(a_ids[i], float(scores[i])) for i in order]

    valid_metrics = evaluate_model(
        model, Q_t, A_t, qid2idx, a_ids, all_rankings,
        valid_qids, device=device, ks=(5,10,50), cand_size=1000, rng_seed=123
    )
    print_metrics_table("Validation (averaged over questions)", valid_metrics, ks=(5,10,50), filename=filename)

    sample_qids = q_ids[:min(5, len(q_ids))]
    for qid in sample_qids:
        topk = recommend_topk_for_qid(qid, topk=args.topk)
        print(f"\nQuestion: {qid}  |  {all_questions[qid]['input'][:80]}")
        for rank,(aid,s) in enumerate(topk,1):
            print(f"  {rank:2d}. {aid:>20s}  score={s:.4f}")

if __name__ == "__main__":
    main()
