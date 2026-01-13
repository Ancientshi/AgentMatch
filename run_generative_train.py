#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Seq2seq training helper for the generative structured recommender.

This trains a text-to-text model (e.g., T5) to emit structured tokens in the
format produced by ``run_generative.py``::

    <LLM_TOKEN> <TOOL_SEP> <TOOL_A> ... <SPECIAL_END>

Notes:
- The script builds supervised pairs straight from the benchmark data (no extra
  preprocessing required).
- Training happens entirely here; ``run_generative.py`` remains inference-only.
"""

from __future__ import annotations

import argparse
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Sequence

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

from agent_rec.models.generative import GenerationConfig, build_training_pairs_from_data_root
from agent_rec.run_common import set_global_seed


@dataclass
class TrainExample:
    query: str
    target: str
    qid: str
    agent_id: str


class PairDataset(Dataset):
    def __init__(
        self,
        pairs: Sequence[TrainExample],
        tokenizer,
        *,
        max_source_length: int,
        max_target_length: int,
    ) -> None:
        self.pairs = list(pairs)
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ex = self.pairs[idx]
        enc = self.tokenizer(
            ex.query,
            truncation=True,
            max_length=self.max_source_length,
        )
        with self.tokenizer.as_target_tokenizer():
            target_enc = self.tokenizer(
                ex.target,
                truncation=True,
                max_length=self.max_target_length,
            )
        return {
            "input_ids": torch.tensor(enc["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(enc["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(target_enc["input_ids"], dtype=torch.long),
        }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Train a seq2seq model to generate structured agent tokens. "
            "Pairs are constructed automatically from the benchmark data."
        )
    )
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--output_dir", type=str, required=True, help="Where to save checkpoints and tokenizer.")
    ap.add_argument("--model_name", type=str, default="t5-small", help="HF seq2seq base model to fine-tune.")
    ap.add_argument("--max_examples", type=int, default=0, help="Limit number of supervised pairs (0 = all).")
    ap.add_argument("--valid_ratio", type=float, default=0.1, help="Portion of pairs used for validation.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup_steps", type=int, default=0)
    ap.add_argument("--max_source_length", type=int, default=256)
    ap.add_argument("--max_target_length", type=int, default=64)
    ap.add_argument(
        "--tool_sep_token",
        type=str,
        default="<TOOL_SEP>",
        help="Special separator used between LLM token and tool tokens.",
    )
    ap.add_argument("--end_token", type=str, default="<SPECIAL_END>", help="Special end token.")
    ap.add_argument("--max_tools", type=int, default=4)
    ap.add_argument("--tfidf_max_features", type=int, default=5000)
    return ap.parse_args()


def split_train_valid(pairs: Sequence[Dict[str, str]], valid_ratio: float, seed: int) -> tuple[list, list]:
    rnd = random.Random(seed)
    rows = list(pairs)
    rnd.shuffle(rows)
    n_valid = max(1, int(len(rows) * valid_ratio)) if rows else 0
    return rows[n_valid:], rows[:n_valid]


def to_examples(rows: Sequence[Dict[str, str]]) -> List[TrainExample]:
    return [TrainExample(query=r["query"], target=r["target"], qid=r["qid"], agent_id=r["agent_id"]) for r in rows]


def add_special_tokens(tokenizer, cfg: GenerationConfig) -> int:
    special_tokens = [cfg.tool_sep_token, cfg.end_token]
    add_spec = {"additional_special_tokens": [t for t in special_tokens if t not in tokenizer.get_vocab()]}
    if add_spec["additional_special_tokens"]:
        return tokenizer.add_special_tokens(add_spec)
    return 0


def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    gen_cfg = GenerationConfig(
        tool_sep_token=args.tool_sep_token,
        end_token=args.end_token,
        max_tools=args.max_tools,
        tfidf_max_features=args.tfidf_max_features,
    )
    max_examples = None if args.max_examples <= 0 else args.max_examples
    pairs = build_training_pairs_from_data_root(
        args.data_root,
        config=gen_cfg,
        max_examples=max_examples,
    )
    if not pairs:
        raise SystemExit("No supervised pairs found; check data_root.")

    train_rows, valid_rows = split_train_valid(pairs, args.valid_ratio, args.seed)
    print(f"[data] train={len(train_rows)}  valid={len(valid_rows)} (max_examples={max_examples or 'all'})")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    added = add_special_tokens(tokenizer, gen_cfg)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)
    if added > 0:
        model.resize_token_embeddings(len(tokenizer))

    train_ds = PairDataset(
        to_examples(train_rows),
        tokenizer,
        max_source_length=args.max_source_length,
        max_target_length=args.max_target_length,
    )
    valid_ds = PairDataset(
        to_examples(valid_rows),
        tokenizer,
        max_source_length=args.max_source_length,
        max_target_length=args.max_target_length,
    ) if valid_rows else None

    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding="longest")
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        logging_steps=50,
        save_steps=500,
        evaluation_strategy="steps" if valid_ds is not None else "no",
        eval_steps=500 if valid_ds is not None else None,
        save_total_limit=2,
        predict_with_generate=True,
        report_to=[],
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        tokenizer=tokenizer,
        data_collator=collator,
    )
    trainer.train()
    print(f"[save] saving model + tokenizer to {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
