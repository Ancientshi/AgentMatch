#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Command-line helper for the generative structured recommender.

Two main modes:
1) Generate sequences for a query:
   python run_generative.py --data_root /path --query "如何写个爬虫？" --top_k 3

2) Export supervised pairs for seq2seq finetuning:
   python run_generative.py --data_root /path --export_pairs /tmp/pairs.jsonl --max_examples 5000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List

from agent_rec.models.generative import (
    GenerationConfig,
    GenerativeStructuredRecommender,
    build_training_pairs_from_data_root,
)
from agent_rec.run_common import bootstrap_run


def _write_jsonl(rows: Iterable[dict], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Inference-only helper for the lightweight generative recommender. "
            "Use --query to run retrieval-format generation, or --export_pairs to dump "
            "supervised targets for finetuning an external seq2seq model."
        )
    )
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument(
        "--query",
        type=str,
        default="",
        help="Single query to generate tokens for (inference mode). Leave empty to skip inference.",
    )
    ap.add_argument("--top_k", type=int, default=10, help="Number of agents to return")
    ap.add_argument("--with_metadata", type=int, default=1, help="1 to include scores/ids in output")

    ap.add_argument("--tool_sep_token", type=str, default="<TOOL_SEP>")
    ap.add_argument("--end_token", type=str, default="<SPECIAL_END>")
    ap.add_argument("--max_tools", type=int, default=8)
    ap.add_argument("--tfidf_max_features", type=int, default=5000)

    ap.add_argument(
        "--export_pairs",
        type=str,
        default="",
        help="Optional path to write supervised pairs (JSONL) for downstream seq2seq training. "
        "If set, the script does not train any model—it only exports data.",
    )
    ap.add_argument("--max_examples", type=int, default=0, help="Limit number of supervised pairs (0 = all)")
    return ap.parse_args()


def build_generator(args: argparse.Namespace) -> GenerativeStructuredRecommender:
    cfg = GenerationConfig(
        tool_sep_token=args.tool_sep_token,
        end_token=args.end_token,
        max_tools=args.max_tools,
        tfidf_max_features=args.tfidf_max_features,
    )
    boot = bootstrap_run(
        data_root=args.data_root,
        exp_name="generative",
        topk=args.top_k,
        with_tools=True,
    )
    return GenerativeStructuredRecommender.from_bundle(
        boot.bundle, tools=boot.tools or {}, config=cfg, agent_order=boot.a_ids
    )


def maybe_export_pairs(args: argparse.Namespace) -> None:
    if not args.export_pairs:
        return
    max_examples = None if args.max_examples <= 0 else args.max_examples
    pairs = build_training_pairs_from_data_root(
        args.data_root,
        config=GenerationConfig(
            tool_sep_token=args.tool_sep_token,
            end_token=args.end_token,
            max_tools=args.max_tools,
            tfidf_max_features=args.tfidf_max_features,
        ),
        max_examples=max_examples,
    )
    out_path = Path(args.export_pairs)
    n = _write_jsonl(pairs, out_path)
    print(f"[export] wrote {n} pairs to {out_path}")


def main() -> None:
    args = parse_args()

    if not args.query and not args.export_pairs:
        raise SystemExit("Nothing to do: provide --query for inference and/or --export_pairs for data export.")

    if args.export_pairs:
        print("[mode] exporting supervised pairs (no training performed).")
        maybe_export_pairs(args)

    if args.query:
        print("[mode] running inference for a single query.")
        gen = build_generator(args)
        results: List[str | dict] = gen.generate(
            args.query, top_k=args.top_k, with_metadata=bool(args.with_metadata)
        )
        print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
