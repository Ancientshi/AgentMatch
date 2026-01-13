#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import math
import os
import random
from datetime import datetime
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer

from agent_rec.cli_common import add_shared_training_args
from agent_rec.config import EVAL_TOPK, POS_TOPK, POS_TOPK_BY_PART
from agent_rec.data import build_training_pairs, stratified_train_valid_split
from agent_rec.eval import evaluate_sampled_embedding_topk, split_eval_qids_by_part
from agent_rec.features import (
    build_agent_content_view,
    build_agent_tool_id_buffers,
    build_unified_corpora,
    UNK_TOOL_TOKEN,
    UNK_LLM_TOKEN,
)
from agent_rec.models.dnn import SimpleBPRDNN, bpr_loss
from agent_rec.run_common import (
    cache_key_from_meta,
    cache_key_from_text,
    bootstrap_run,
    load_or_build_training_cache,
    shared_cache_dir,
)

from utils import print_metrics_table


def ensure_transformer_cache_dir(cache_dir: str) -> str:
    d = os.path.join(cache_dir, "transformer_cache")
    os.makedirs(d, exist_ok=True)
    return d


def transformer_cache_exists(cache_dir: str) -> bool:
    needed = [
        "q_ids.json",
        "a_ids.json",
        "tool_names.json",
        "tool_id_vocab.json",
        "llm_ids.json",
        "llm_vocab.json",
        "Q_emb.npy",
        "A_model_content.npy",
        "A_tool_content.npy",
        "A_emb.npy",
        "agent_tool_idx_padded.npy",
        "agent_tool_mask.npy",
        "agent_llm_idx.npy",
        "enc_meta.json",
    ]
    return all(os.path.exists(os.path.join(cache_dir, name)) for name in needed)


def save_transformer_cache(
    cache_dir: str,
    q_ids,
    a_ids,
    tool_names,
    tool_id_vocab,
    llm_ids,
    llm_vocab,
    Q_emb,
    A_model_emb,
    A_tool_emb,
    A_emb,
    agent_tool_idx_padded,
    agent_tool_mask,
    agent_llm_idx,
    enc_meta,
):
    with open(os.path.join(cache_dir, "q_ids.json"), "w", encoding="utf-8") as f:
        json.dump(q_ids, f, ensure_ascii=False)
    with open(os.path.join(cache_dir, "a_ids.json"), "w", encoding="utf-8") as f:
        json.dump(a_ids, f, ensure_ascii=False)
    with open(os.path.join(cache_dir, "tool_names.json"), "w", encoding="utf-8") as f:
        json.dump(tool_names, f, ensure_ascii=False)
    with open(os.path.join(cache_dir, "tool_id_vocab.json"), "w", encoding="utf-8") as f:
        json.dump(tool_id_vocab, f, ensure_ascii=False)
    with open(os.path.join(cache_dir, "llm_ids.json"), "w", encoding="utf-8") as f:
        json.dump(llm_ids, f, ensure_ascii=False)
    with open(os.path.join(cache_dir, "llm_vocab.json"), "w", encoding="utf-8") as f:
        json.dump(llm_vocab, f, ensure_ascii=False)
    np.save(os.path.join(cache_dir, "Q_emb.npy"), Q_emb.astype(np.float32))
    np.save(os.path.join(cache_dir, "A_model_content.npy"), A_model_emb.astype(np.float32))
    np.save(os.path.join(cache_dir, "A_tool_content.npy"), A_tool_emb.astype(np.float32))
    np.save(os.path.join(cache_dir, "A_emb.npy"), A_emb.astype(np.float32))
    np.save(os.path.join(cache_dir, "agent_tool_idx_padded.npy"), agent_tool_idx_padded.astype(np.int64))
    np.save(os.path.join(cache_dir, "agent_tool_mask.npy"), agent_tool_mask.astype(np.float32))
    np.save(os.path.join(cache_dir, "agent_llm_idx.npy"), agent_llm_idx.astype(np.int64))
    with open(os.path.join(cache_dir, "enc_meta.json"), "w", encoding="utf-8") as f:
        json.dump(enc_meta, f, ensure_ascii=False)


def load_transformer_cache(cache_dir: str):
    with open(os.path.join(cache_dir, "q_ids.json"), "r", encoding="utf-8") as f:
        q_ids = json.load(f)
    with open(os.path.join(cache_dir, "a_ids.json"), "r", encoding="utf-8") as f:
        a_ids = json.load(f)
    with open(os.path.join(cache_dir, "tool_names.json"), "r", encoding="utf-8") as f:
        tool_names = json.load(f)
    with open(os.path.join(cache_dir, "tool_id_vocab.json"), "r", encoding="utf-8") as f:
        tool_id_vocab = json.load(f)
    with open(os.path.join(cache_dir, "llm_ids.json"), "r", encoding="utf-8") as f:
        llm_ids = json.load(f)
    with open(os.path.join(cache_dir, "llm_vocab.json"), "r", encoding="utf-8") as f:
        llm_vocab = json.load(f)
    with open(os.path.join(cache_dir, "enc_meta.json"), "r", encoding="utf-8") as f:
        enc_meta = json.load(f)
    Q_emb = np.load(os.path.join(cache_dir, "Q_emb.npy"))
    A_model = np.load(os.path.join(cache_dir, "A_model_content.npy"))
    A_tool = np.load(os.path.join(cache_dir, "A_tool_content.npy"))
    A_emb = np.load(os.path.join(cache_dir, "A_emb.npy"))
    agent_tool_idx_padded = np.load(os.path.join(cache_dir, "agent_tool_idx_padded.npy"))
    agent_tool_mask = np.load(os.path.join(cache_dir, "agent_tool_mask.npy"))
    agent_llm_idx = np.load(os.path.join(cache_dir, "agent_llm_idx.npy"))
    return (
        q_ids,
        a_ids,
        tool_names,
        tool_id_vocab,
        llm_ids,
        llm_vocab,
        Q_emb,
        A_model,
        A_tool,
        A_emb,
        agent_tool_idx_padded,
        agent_tool_mask,
        agent_llm_idx,
        enc_meta,
    )


@torch.no_grad()
def encode_texts(
    texts: List[str],
    tokenizer,
    encoder,
    device,
    max_len: int = 128,
    batch_size: int = 256,
    pooling: str = "cls",
):
    if not texts:
        dim = getattr(getattr(encoder, "config", None), "hidden_size", 0) or 0
        return np.zeros((0, dim), dtype=np.float32)
    embs = []
    use_cls = pooling == "cls"
    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding with Transformer", dynamic_ncols=True):
        batch = texts[i : i + batch_size]
        toks = tokenizer(batch, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
        toks = {k: v.to(device) for k, v in toks.items()}
        out = encoder(**toks)
        if hasattr(out, "last_hidden_state"):
            if use_cls:
                vec = out.last_hidden_state[:, 0, :]
            else:
                attn = toks["attention_mask"].unsqueeze(-1)
                sum_h = (out.last_hidden_state * attn).sum(1)
                vec = sum_h / attn.sum(1).clamp(min=1)
        else:
            vec = out.pooler_output
        embs.append(vec.detach().cpu())
    return torch.cat(embs, dim=0).numpy()


def encode_batch(tokenizer, encoder, texts: List[str], device, max_len: int = 128, pooling: str = "cls"):
    if not texts:
        dim = getattr(getattr(encoder, "config", None), "hidden_size", 0) or 0
        return torch.zeros((0, dim), device=device)
    toks = tokenizer(texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
    toks = {k: v.to(device) for k, v in toks.items()}
    out = encoder(**toks)
    if hasattr(out, "last_hidden_state"):
        if pooling == "cls":
            vec = out.last_hidden_state[:, 0, :]
        else:
            attn = toks["attention_mask"].unsqueeze(-1)
            sum_h = (out.last_hidden_state * attn).sum(1)
            vec = sum_h / attn.sum(1).clamp(min=1)
    else:
        vec = out.pooler_output
    return vec

class LoRALinear(nn.Module):
    def __init__(self, base_linear: nn.Linear, r: int = 8, alpha: int = 16, dropout: float = 0.0):
        super().__init__()
        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / max(1, r)

        # keep references to frozen base weights
        self.weight = base_linear.weight
        self.bias = base_linear.bias
        for p in (self.weight, self.bias):
            if p is not None:
                p.requires_grad = False

        # ✅ create LoRA params on SAME device/dtype as base weight
        dev = base_linear.weight.device
        dt = base_linear.weight.dtype
        self.A = nn.Parameter(torch.empty((r, self.in_features), device=dev, dtype=dt))
        self.B = nn.Parameter(torch.empty((self.out_features, r), device=dev, dtype=dt))

        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.zeros_(self.B)

        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # base: (..., out_features)
        base = F.linear(x, self.weight, self.bias)

        # ✅ lora path using F.linear (works for 2D/3D inputs, no transpose, no device mismatch)
        # x: (..., in_features) -> (..., r) -> (..., out_features)
        h = F.linear(self.dropout(x), self.A)     # (..., r)
        lora = F.linear(h, self.B)                # (..., out_features)

        return base + lora * self.scaling



def apply_lora_to_encoder(encoder: nn.Module, target_keywords: List[str], r: int, alpha: int, dropout: float):
    repl = 0
    for _, module in list(encoder.named_modules()):
        for child_name, child in list(module.named_children()):
            if isinstance(child, nn.Linear) and any(k in child_name.lower() for k in target_keywords):
                wrapped = LoRALinear(child, r=r, alpha=alpha, dropout=dropout)
                setattr(module, child_name, wrapped)
                repl += 1
    print(
        f"[LoRA] injected into {repl} Linear layers "
        f"(targets={target_keywords}, r={r}, alpha={alpha}, dropout={dropout})"
    )


def _get_transformer_layers(model: nn.Module):
    if hasattr(model, "encoder") and hasattr(model.encoder, "layer"):
        return model.encoder.layer
    if hasattr(model, "roberta") and hasattr(model.roberta, "encoder") and hasattr(model.roberta.encoder, "layer"):
        return model.roberta.encoder.layer
    if hasattr(model, "transformer") and hasattr(model.transformer, "layer"):
        return model.transformer.layer
    return None


def set_finetune_scope(encoder: nn.Module, unfreeze_last_n: int, unfreeze_emb: bool) -> None:
    for p in encoder.parameters():
        p.requires_grad = False

    layers = _get_transformer_layers(encoder)
    if layers is None:
        print("[warn] could not locate transformer layers; keeping encoder frozen")
        return

    if unfreeze_last_n <= 0 or unfreeze_last_n >= len(layers):
        for p in encoder.parameters():
            p.requires_grad = True
    else:
        for block in layers[-unfreeze_last_n:]:
            for p in block.parameters():
                p.requires_grad = True

    if unfreeze_emb:
        if hasattr(encoder, "embeddings"):
            for p in encoder.embeddings.parameters():
                p.requires_grad = True
        if hasattr(encoder, "roberta") and hasattr(encoder.roberta, "embeddings"):
            for p in encoder.roberta.embeddings.parameters():
                p.requires_grad = True
        if hasattr(encoder, "distilbert") and hasattr(encoder.distilbert, "embeddings"):
            for p in encoder.distilbert.embeddings.parameters():
                p.requires_grad = True
        for m in encoder.modules():
            if isinstance(m, nn.LayerNorm):
                for p in m.parameters():
                    if not p.requires_grad:
                        p.requires_grad = True


def main():
    import argparse

    parser = argparse.ArgumentParser()
    add_shared_training_args(
        parser,
        exp_name_default="bpr_bert",
        device_default="cuda:0",
        epochs_default=3,
        batch_size_default=256,
        lr_default=1e-3,
        lr_help="LR for BPR head / embeddings",
    )
    parser.add_argument("--pretrained_model", type=str, default="distilbert-base-uncased")
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--text_hidden", type=int, default=256)
    parser.add_argument("--id_dim", type=int, default=64)
    parser.add_argument("--encoder_lr", type=float, default=5e-5)
    parser.add_argument("--encoder_weight_decay", type=float, default=0.0)
    parser.add_argument("--rebuild_embedding_cache", type=int, default=0)
    parser.add_argument("--pooling", type=str, choices=["cls", "mean"], default="cls")
    parser.add_argument(
        "--tune_mode",
        type=str,
        choices=["frozen", "full", "lora"],
        default="frozen",
        help="frozen: offline cache; full/lora: online encoding and finetune",
    )
    parser.add_argument("--unfreeze_last_n", type=int, default=0)
    parser.add_argument("--unfreeze_emb", type=int, default=0)
    parser.add_argument("--grad_ckpt", type=int, default=0)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--lora_targets", type=str, default="q_lin,k_lin,v_lin,out_lin")
    parser.add_argument("--use_query_id_emb", type=int, default=0, help="1 to enable query-ID embedding")
    parser.add_argument("--use_llm_id_emb", type=int, default=1)
    parser.add_argument("--use_tool_id_emb", type=int, default=1)
    parser.add_argument(
        "--use_model_content_vector", type=int, default=1, help="1 to include V_model(A) in content view"
    )
    parser.add_argument(
        "--use_tool_content_vector", type=int, default=1, help="1 to include V_tool_content(A) in content view"
    )
    args = parser.parse_args()

    active_content_parts = int(bool(args.use_model_content_vector)) + int(bool(args.use_tool_content_vector))
    if active_content_parts == 0:
        raise ValueError("Enable at least one of use_model_content_vector/use_tool_content_vector.")

    boot = bootstrap_run(
        data_root=args.data_root,
        exp_name=args.exp_name,
        topk=args.topk,
        with_tools=True,
    )

    bundle = boot.bundle
    tools = boot.tools
    all_agents = bundle.all_agents
    all_questions = bundle.all_questions
    all_rankings = bundle.all_rankings
    qid_to_part = bundle.qid_to_part

    (
        q_ids,
        q_texts,
        tool_names,
        tool_texts,
        a_ids,
        a_model_names,
        a_tool_lists,
        llm_ids,
    ) = build_unified_corpora(all_agents, all_questions, tools)
    if q_ids != boot.q_ids or a_ids != boot.a_ids:
        raise ValueError("ID ordering mismatch between data bootstrap and transformer corpora.")

    tool_id_vocab = [UNK_TOOL_TOKEN] + tool_names
    tool_vocab_map = {n: i for i, n in enumerate(tool_id_vocab)}
    agent_tool_idx_padded, agent_tool_mask = build_agent_tool_id_buffers(a_tool_lists, tool_vocab_map)
    agent_tool_idx_padded = torch.from_numpy(agent_tool_idx_padded).long()
    agent_tool_mask = torch.from_numpy(agent_tool_mask).float()
    llm_vocab = [UNK_LLM_TOKEN] + [lid for lid in llm_ids if lid]
    llm_vocab = list(dict.fromkeys(llm_vocab))
    llm_vocab_map = {n: i for i, n in enumerate(llm_vocab)}
    agent_llm_idx = torch.tensor([llm_vocab_map.get(lid, 0) for lid in llm_ids], dtype=torch.long)

    qid2idx = boot.qid2idx
    aid2idx = boot.aid2idx
    qids_in_rank = boot.qids_in_rank

    data_sig = boot.data_sig
    exp_cache_dir = boot.exp_cache_dir
    transformer_cache_key = f"{data_sig}_{cache_key_from_text(args.pretrained_model)}"
    transformer_cache_dir = ensure_transformer_cache_dir(
        shared_cache_dir(args.data_root, "transformer", transformer_cache_key)
    )

    want_meta = {
        "data_sig": data_sig,
        "pos_topk_by_part": POS_TOPK_BY_PART,
        "neg_per_pos": int(args.neg_per_pos),
        "rng_seed_pairs": int(args.rng_seed_pairs),
        "split_seed": int(args.split_seed),
        "valid_ratio": float(args.valid_ratio),
    }
    training_cache_dir = shared_cache_dir(args.data_root, "training", f"{data_sig}_{cache_key_from_meta(want_meta)}")

    def build_cache():
        train_qids, valid_qids = stratified_train_valid_split(
            qids_in_rank, qid_to_part=qid_to_part, valid_ratio=args.valid_ratio, seed=args.split_seed
        )
        print(f"[split] train={len(train_qids)}  valid={len(valid_qids)}")

        rankings_train = {qid: all_rankings[qid] for qid in train_qids}
        pairs = build_training_pairs(
            rankings_train,
            a_ids,
            qid_to_part=qid_to_part,
            pos_topk_by_part=POS_TOPK_BY_PART,
            pos_topk_default=POS_TOPK,
            neg_per_pos=args.neg_per_pos,
            rng_seed=args.rng_seed_pairs,
        )
        pairs_idx = [(qid2idx[q], aid2idx[p], aid2idx[n]) for (q, p, n) in pairs]
        pairs_idx_np = np.array(pairs_idx, dtype=np.int64)
        return train_qids, valid_qids, pairs_idx_np

    train_qids, valid_qids, pairs_idx_np = load_or_build_training_cache(
        training_cache_dir,
        args.rebuild_training_cache,
        want_meta,
        build_cache,
    )

    device = torch.device(args.device)

    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)
    encoder = AutoModel.from_pretrained(args.pretrained_model).to(device)

    if args.grad_ckpt:
        try:
            encoder.gradient_checkpointing_enable()
            print("[encoder] gradient checkpointing enabled")
        except Exception as exc:
            print(f"[encoder] gradient checkpointing not supported: {exc}")

    if args.tune_mode == "lora":
        # 1) freeze everything first (so we ONLY train LoRA params)
        for p in encoder.parameters():
            p.requires_grad = False

        # 2) inject LoRA
        targets = [s.strip().lower() for s in args.lora_targets.split(",") if s.strip()]
        apply_lora_to_encoder(
            encoder,
            targets,
            r=args.lora_r,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
        )

        # 3) sanity: show trainable params
        trainable = [(n, p.numel()) for n, p in encoder.named_parameters() if p.requires_grad]
        print(f"[LoRA] trainable encoder params: {len(trainable)} tensors, {sum(x for _, x in trainable):,} params")
        if len(trainable) <= 20:
            print("[LoRA] trainable names:", [n for n, _ in trainable])

    elif args.tune_mode == "full":
        set_finetune_scope(encoder, unfreeze_last_n=args.unfreeze_last_n, unfreeze_emb=bool(args.unfreeze_emb))

    elif args.tune_mode == "frozen":
        for p in encoder.parameters():
            p.requires_grad = False

    use_embedding_cache = args.tune_mode == "frozen"
    Q_emb = A_emb = A_model_emb = A_tool_emb = None

    if use_embedding_cache:
        if transformer_cache_exists(transformer_cache_dir) and args.rebuild_embedding_cache == 0:
            (
                q_ids_c,
                a_ids_c,
                tool_names_c,
                tool_id_vocab_c,
                llm_ids_c,
                llm_vocab_c,
                Q_emb,
                A_model_emb,
                A_tool_emb,
                A_emb,
                agent_tool_idx_padded_np,
                agent_tool_mask_np,
                agent_llm_idx_np,
                enc_meta,
            ) = load_transformer_cache(transformer_cache_dir)
            if (
                q_ids_c == q_ids
                and a_ids_c == a_ids
                and tool_names_c == tool_names
                and tool_id_vocab_c == tool_id_vocab
                and llm_ids_c == llm_ids
                and llm_vocab_c == llm_vocab
                and enc_meta.get("pretrained_model") == args.pretrained_model
                and enc_meta.get("max_len") == args.max_len
                and enc_meta.get("pooling", "cls") == args.pooling
            ):
                print(f"[cache] loaded transformer embeddings from {transformer_cache_dir}")
                agent_tool_idx_padded = torch.from_numpy(agent_tool_idx_padded_np).long()
                agent_tool_mask = torch.from_numpy(agent_tool_mask_np).float()
                agent_llm_idx = torch.from_numpy(agent_llm_idx_np).long()
            else:
                print("[cache] transformer cache mismatch; rebuilding embeddings...")
                Q_emb = A_emb = A_model_emb = A_tool_emb = None

        if Q_emb is None or A_model_emb is None or A_tool_emb is None:
            encoder.eval()
            with torch.no_grad():
                Q_emb = encode_texts(
                    q_texts,
                    tokenizer,
                    encoder,
                    device,
                    max_len=args.max_len,
                    batch_size=256,
                    pooling=args.pooling,
                )
                A_model_emb = encode_texts(
                    a_model_names,
                    tokenizer,
                    encoder,
                    device,
                    max_len=args.max_len,
                    batch_size=256,
                    pooling=args.pooling,
                )
                tool_emb = encode_texts(
                    tool_texts,
                    tokenizer,
                    encoder,
                    device,
                    max_len=args.max_len,
                    batch_size=256,
                    pooling=args.pooling,
                )
                tool_emb = tool_emb / (np.linalg.norm(tool_emb, axis=1, keepdims=True) + 1e-8)
                A_tool_emb = []
                for tools_for_agent in a_tool_lists:
                    if tools_for_agent:
                        idxs = [tool_names.index(t) for t in tools_for_agent if t in tool_names]
                        if idxs:
                            A_tool_emb.append(tool_emb[idxs].mean(axis=0))
                            continue
                    A_tool_emb.append(np.zeros((tool_emb.shape[1],), dtype=np.float32))
                A_tool_emb = np.stack(A_tool_emb, axis=0)
                A_model_emb = A_model_emb / (np.linalg.norm(A_model_emb, axis=1, keepdims=True) + 1e-8)
                if A_tool_emb.size > 0:
                    A_tool_emb = A_tool_emb / (np.linalg.norm(A_tool_emb, axis=1, keepdims=True) + 1e-8)
                A_emb = build_agent_content_view(
                    A_model_content=A_model_emb,
                    A_tool_content=A_tool_emb,
                    use_model_content_vector=bool(args.use_model_content_vector),
                    use_tool_content_vector=bool(args.use_tool_content_vector),
                )
            save_transformer_cache(
                transformer_cache_dir,
                q_ids,
                a_ids,
                tool_names,
                tool_id_vocab,
                llm_ids,
                llm_vocab,
                Q_emb,
                A_model_emb,
                A_tool_emb,
                A_emb,
                agent_tool_idx_padded.cpu().numpy()
                if torch.is_tensor(agent_tool_idx_padded)
                else agent_tool_idx_padded,
                agent_tool_mask.cpu().numpy() if torch.is_tensor(agent_tool_mask) else agent_tool_mask,
                agent_llm_idx.cpu().numpy() if torch.is_tensor(agent_llm_idx) else agent_llm_idx,
                enc_meta={
                    "pretrained_model": args.pretrained_model,
                    "max_len": args.max_len,
                    "pooling": args.pooling,
                    "use_model_content_vector": bool(args.use_model_content_vector),
                    "use_tool_content_vector": bool(args.use_tool_content_vector),
                },
            )
            print(f"[cache] saved transformer embeddings to {transformer_cache_dir}")
        else:
            A_emb = build_agent_content_view(
                A_model_content=A_model_emb,
                A_tool_content=A_tool_emb,
                use_model_content_vector=bool(args.use_model_content_vector),
                use_tool_content_vector=bool(args.use_tool_content_vector),
            )
    else:
        encoder.train()

    if use_embedding_cache:
        d_text = int(Q_emb.shape[1])
    else:
        encoder.eval()
        with torch.no_grad():
            tmp = encode_batch(tokenizer, encoder, ["hello"], device, max_len=args.max_len, pooling=args.pooling)
            d_text = int(tmp.shape[1])
        encoder.train()
    agent_content_dim = int(A_emb.shape[1]) if use_embedding_cache else int(d_text * active_content_parts)

    model = SimpleBPRDNN(
        d_q=d_text,
        d_a=agent_content_dim,
        num_tools=len(tool_id_vocab),
        num_llm_ids=len(llm_vocab),
        agent_tool_indices_padded=agent_tool_idx_padded.to(device),
        agent_tool_mask=agent_tool_mask.to(device),
        agent_llm_idx=agent_llm_idx.to(device),
        text_hidden=args.text_hidden,
        id_dim=args.id_dim,
        num_queries=len(q_ids),
        use_query_id_emb=bool(args.use_query_id_emb),
        use_tool_id_emb=bool(args.use_tool_id_emb),
        use_llm_id_emb=bool(args.use_llm_id_emb),
    ).to(device)

    head_params = list(model.parameters())
    encoder_params = [p for p in encoder.parameters() if p.requires_grad]

    param_groups = [{"params": head_params, "lr": args.lr}]
    if encoder_params:
        param_groups.append(
            {
                "params": encoder_params,
                "lr": args.encoder_lr,
                "weight_decay": args.encoder_weight_decay,
            }
        )
    optimizer = torch.optim.Adam(param_groups)

    pairs = pairs_idx_np.tolist()
    num_pairs = len(pairs)
    num_batches = math.ceil(num_pairs / args.batch_size)
    print(f"Training pairs: {num_pairs}, batches/epoch: {num_batches}")

    if use_embedding_cache:
        Q_t = torch.tensor(Q_emb, dtype=torch.float32, device=device)
        A_t = torch.tensor(A_emb, dtype=torch.float32, device=device)
    else:
        Q_t = A_t = None

    for epoch in range(1, args.epochs + 1):
        random.shuffle(pairs)
        total_loss = 0.0
        pbar = tqdm(range(num_batches), desc=f"Epoch {epoch}/{args.epochs}", leave=True, dynamic_ncols=True)
        model.train()
        for b in pbar:
            batch = pairs[b * args.batch_size : (b + 1) * args.batch_size]
            if not batch:
                continue
            q_idx = torch.tensor([t[0] for t in batch], dtype=torch.long, device=device)
            pos_idx = torch.tensor([t[1] for t in batch], dtype=torch.long, device=device)
            neg_idx = torch.tensor([t[2] for t in batch], dtype=torch.long, device=device)

            if use_embedding_cache:
                q_vec = Q_t[q_idx]
                pos_vec = A_t[pos_idx]
                neg_vec = A_t[neg_idx]
            else:
                uniq_q, inv_q = torch.unique(q_idx, sorted=True, return_inverse=True)
                uniq_agents = torch.unique(torch.cat([pos_idx, neg_idx]), sorted=True)

                q_text_batch = [q_texts[i] for i in uniq_q.tolist()]
                q_vec_uniq = encode_batch(
                    tokenizer, encoder, q_text_batch, device, max_len=args.max_len, pooling=args.pooling
                )
                q_vec = q_vec_uniq[inv_q]

                agent_texts = [a_model_names[i] for i in uniq_agents.tolist()]
                model_emb = encode_batch(
                    tokenizer, encoder, agent_texts, device, max_len=args.max_len, pooling=args.pooling
                )
                model_emb = F.normalize(model_emb, dim=-1)

                content_parts = []
                if args.use_model_content_vector:
                    content_parts.append(model_emb)

                tool_feats_t: torch.Tensor
                if args.use_tool_content_vector:
                    needed_tools = []
                    for idx_agent in uniq_agents.tolist():
                        needed_tools.extend([t for t in a_tool_lists[idx_agent]])
                    needed_tools = sorted(set(needed_tools))
                    tool_emb_map = {}
                    if needed_tools:
                        names_in_vocab = [t for t in needed_tools if t in tool_names]
                        tool_text_batch = [tool_texts[tool_names.index(t)] for t in names_in_vocab]
                        if tool_text_batch:
                            tool_emb_batch = encode_batch(
                                tokenizer, encoder, tool_text_batch, device, max_len=args.max_len, pooling=args.pooling
                            )
                            tool_emb_batch = F.normalize(tool_emb_batch, dim=-1)
                            for name, emb in zip(names_in_vocab, tool_emb_batch):
                                tool_emb_map[name] = emb
                    tool_dim = model_emb.shape[1]
                    zero_tool = torch.zeros((tool_dim,), device=device)
                    tool_feats = []
                    for idx_agent in uniq_agents.tolist():
                        names = [t for t in a_tool_lists[idx_agent] if t in tool_emb_map]
                        if names:
                            stacked = torch.stack([tool_emb_map[n] for n in names], dim=0)
                            tool_feats.append(stacked.mean(dim=0))
                        else:
                            tool_feats.append(zero_tool)
                    tool_feats_t = torch.stack(tool_feats, dim=0)
                    tool_feats_t = F.normalize(tool_feats_t, dim=-1) if tool_feats_t.numel() > 0 else tool_feats_t
                    content_parts.append(tool_feats_t)
                else:
                    tool_feats_t = torch.zeros((model_emb.size(0), 0), device=device)

                if not content_parts:
                    raise ValueError("Enable at least one content component for agent representation.")

                content_all = torch.cat(content_parts, dim=-1) if len(content_parts) > 1 else content_parts[0]
                agent_order = uniq_agents.tolist()
                pos_vec = content_all[[agent_order.index(i.item()) for i in pos_idx]]
                neg_vec = content_all[[agent_order.index(i.item()) for i in neg_idx]]

            pos, neg = model(q_vec, pos_vec, neg_vec, pos_idx, neg_idx, q_idx=q_idx)
            loss = bpr_loss(pos, neg)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            pbar.set_postfix({"batch_loss": f"{loss.item():.4f}", "avg_loss": f"{(total_loss / (b + 1)):.4f}"})

        print(f"Epoch {epoch}/{args.epochs} - BPR loss: {(total_loss / num_batches if num_batches else 0.0):.4f}")

    if not use_embedding_cache:
        encoder.eval()
        with torch.no_grad():
            Q_emb_eval = encode_texts(
                q_texts,
                tokenizer,
                encoder,
                device,
                max_len=args.max_len,
                batch_size=256,
                pooling=args.pooling,
            )
            A_model_eval = encode_texts(
                a_model_names,
                tokenizer,
                encoder,
                device,
                max_len=args.max_len,
                batch_size=256,
                pooling=args.pooling,
            )
            tool_emb_eval = encode_texts(
                tool_texts,
                tokenizer,
                encoder,
                device,
                max_len=args.max_len,
                batch_size=256,
                pooling=args.pooling,
            )
        Q_t = torch.from_numpy(Q_emb_eval).to(device)
        A_model_eval = A_model_eval / (np.linalg.norm(A_model_eval, axis=1, keepdims=True) + 1e-8)
        tool_emb_eval = tool_emb_eval / (np.linalg.norm(tool_emb_eval, axis=1, keepdims=True) + 1e-8)
        A_tool_eval = []
        for tools_for_agent in a_tool_lists:
            if tools_for_agent:
                idxs = [tool_names.index(t) for t in tools_for_agent if t in tool_names]
                if idxs:
                    A_tool_eval.append(tool_emb_eval[idxs].mean(axis=0))
                    continue
            A_tool_eval.append(np.zeros((tool_emb_eval.shape[1],), dtype=np.float32))
        A_tool_eval = np.stack(A_tool_eval, axis=0)
        A_tool_eval = A_tool_eval / (np.linalg.norm(A_tool_eval, axis=1, keepdims=True) + 1e-8)
        A_emb_eval = build_agent_content_view(
            A_model_content=A_model_eval,
            A_tool_content=A_tool_eval,
            use_model_content_vector=bool(args.use_model_content_vector),
            use_tool_content_vector=bool(args.use_tool_content_vector),
        )
        A_t = torch.from_numpy(A_emb_eval).to(device)

    model_dir = os.path.join(exp_cache_dir, "models")
    os.makedirs(model_dir, exist_ok=True)

    ckpt_path = os.path.join(model_dir, f"{args.exp_name}_{data_sig}.pt")
    meta_path = os.path.join(model_dir, f"meta_{args.exp_name}_{data_sig}.json")

    ckpt = {
        "state_dict": model.state_dict(),
        "data_sig": data_sig,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "dims": {
            "d_text": int(Q_t.shape[1]),
            "num_agents": len(a_ids),
            "num_tools": len(tool_id_vocab),
            "num_llm_ids": len(llm_vocab),
            "text_hidden": args.text_hidden,
            "id_dim": args.id_dim,
        },
        "flags": {
            "use_llm_id_emb": bool(args.use_llm_id_emb),
            "use_tool_id_emb": bool(args.use_tool_id_emb),
            "use_model_content_vector": bool(args.use_model_content_vector),
            "use_tool_content_vector": bool(args.use_tool_content_vector),
            "use_query_id_emb": bool(args.use_query_id_emb),
        },
        "mappings": {"q_ids": q_ids, "a_ids": a_ids, "tool_names": tool_names},
        "args": vars(args),
        "pretrained_model": args.pretrained_model,
        "tune_mode": args.tune_mode,
    }
    torch.save(ckpt, ckpt_path)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({"data_sig": data_sig, "q_ids": q_ids, "a_ids": a_ids}, f, ensure_ascii=False, indent=2)
    print(f"[save] model -> {ckpt_path}")
    print(f"[save] meta  -> {meta_path}")

    model.eval()

    overall_metrics = evaluate_sampled_embedding_topk(
        model=model,
        qid2idx=qid2idx,
        aid2idx=aid2idx,
        all_rankings=all_rankings,
        eval_qids=valid_qids,
        Q_t=Q_t,
        A_t=A_t,
        cand_size=args.eval_cand_size,
        qid_to_part=qid_to_part,
        pos_topk_by_part=POS_TOPK_BY_PART,
        pos_topk_default=POS_TOPK,
        topk=int(args.topk),
        desc=f"Valid Overall (transformer, top{int(args.topk)})",
    )
    print_metrics_table("Validation Overall (transformer)", overall_metrics, ks=(int(args.topk),), filename=args.exp_name)

    part_splits = split_eval_qids_by_part(valid_qids, qid_to_part=qid_to_part)
    for part in ["PartI", "PartII", "PartIII"]:
        qids_part = part_splits.get(part, [])
        if not qids_part:
            continue
        m_part = evaluate_sampled_embedding_topk(
            model=model,
            qid2idx=qid2idx,
            aid2idx=aid2idx,
            all_rankings=all_rankings,
            eval_qids=qids_part,
            Q_t=Q_t,
            A_t=A_t,
            cand_size=args.eval_cand_size,
            qid_to_part=qid_to_part,
            pos_topk_by_part=POS_TOPK_BY_PART,
            pos_topk_default=POS_TOPK,
            topk=int(args.topk),
            desc=f"Valid {part} (transformer, top{int(args.topk)})",
        )
        print_metrics_table(
            f"Validation {part} (transformer)", m_part, ks=(int(args.topk),), filename=args.exp_name
        )


if __name__ == "__main__":
    main()
