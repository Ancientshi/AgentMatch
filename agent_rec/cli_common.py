#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from typing import Optional

from agent_rec.config import EVAL_TOPK


def add_shared_training_args(
    parser: argparse.ArgumentParser,
    *,
    exp_name_default: str,
    device_default: str = "cuda:0",
    epochs_default: int = 5,
    batch_size_default: int = 1024,
    lr_default: Optional[float] = None,
    lr_help: Optional[str] = None,
    include_neg_per_pos: bool = True,
    include_eval_cand: bool = True,
    eval_cand_default: int = 100,
    topk_default: int = EVAL_TOPK,
) -> argparse.ArgumentParser:
    """
    Attach the common training/data arguments used across runner scripts.

    Parameters allow each script to override defaults while keeping the argument
    surface consistent. Returning the parser makes the helper chainable.
    """

    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--exp_name", type=str, default=exp_name_default, help="Cache folder name under .cache/")
    parser.add_argument("--epochs", type=int, default=epochs_default)
    parser.add_argument("--batch_size", type=int, default=batch_size_default)
    if lr_default is not None:
        parser.add_argument("--lr", type=float, default=lr_default, help=lr_help)
    if include_neg_per_pos:
        parser.add_argument("--neg_per_pos", type=int, default=1)
    parser.add_argument("--rng_seed_pairs", type=int, default=42)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--valid_ratio", type=float, default=0.2)
    parser.add_argument("--device", type=str, default=device_default)
    parser.add_argument("--rebuild_training_cache", type=int, default=0)
    parser.add_argument("--train_parts", type=str, nargs="+", default=["PartI", "PartII", "PartIII"], help="data parts to use for training")
    parser.add_argument("--eval_parts", type=str, nargs="+", default=["PartI", "PartII", "PartIII"], help="data parts to use for evaluation")

    if include_eval_cand:
        parser.add_argument("--eval_cand_size", type=int, default=eval_cand_default)
    if topk_default is not None:
        parser.add_argument("--topk", type=int, default=topk_default, help="Fixed to 10 by default")
    return parser
