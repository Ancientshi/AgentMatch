#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute regime-shift evidence between dense (Part I) and one-off (Part II/III)
interaction topologies. Generates one table with summary metrics and one figure
with frequency curves + degree histograms.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

from agent_rec.config import pos_topk_for_part
from agent_rec.data import DatasetBundle, collect_data


@dataclass
class PartStats:
    label: str
    parts: List[str]
    question_count: int
    positive_interactions: int
    agent_degree: Counter[str]
    unique_agents: set[str]

    @property
    def unique_agent_ratio(self) -> float:
        if self.positive_interactions == 0:
            return 0.0
        return len(self.unique_agents) / self.positive_interactions

    @property
    def entropy(self) -> float:
        total = self.positive_interactions
        if total == 0 or not self.agent_degree:
            return 0.0
        return -sum((cnt / total) * math.log(cnt / total) for cnt in self.agent_degree.values())

    @property
    def effective_size(self) -> float:
        return math.exp(self.entropy)

    def head_coverage(self, percentile: float) -> float:
        if self.positive_interactions == 0 or not self.agent_degree:
            return 0.0
        if percentile <= 0:
            return 0.0
        top_n = max(1, math.ceil(len(self.agent_degree) * percentile / 100.0))
        covered = sum(cnt for _, cnt in self.agent_degree.most_common(top_n))
        return covered / self.positive_interactions

    def degree_values(self) -> List[int]:
        return list(self.agent_degree.values())


def collect_part_stats(bundle: DatasetBundle, parts: Iterable[str]) -> PartStats:
    part_set = set(parts)
    degree = Counter()
    unique_agents: set[str] = set()
    question_count = 0
    positive_interactions = 0

    for qid, part in bundle.qid_to_part.items():
        if part not in part_set:
            continue
        ranking = bundle.all_rankings.get(qid, [])
        if not ranking:
            continue

        question_count += 1
        top_k = pos_topk_for_part(part)
        positives = [aid for aid in ranking[:top_k] if aid in bundle.all_agents]
        if not positives:
            continue

        degree.update(positives)
        unique_agents.update(positives)
        positive_interactions += len(positives)

    label = "/".join(sorted(part_set))
    return PartStats(
        label=label,
        parts=sorted(part_set),
        question_count=question_count,
        positive_interactions=positive_interactions,
        agent_degree=degree,
        unique_agents=unique_agents,
    )


def export_metrics_table(stats_list: Sequence[PartStats], output_csv: Path) -> None:
    fieldnames = [
        "label",
        "parts",
        "questions",
        "positive_interactions",
        "unique_agents",
        "unique_agent_ratio",
        "entropy",
        "effective_size",
        "top1pct_coverage",
        "top5pct_coverage",
    ]
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for stats in stats_list:
            writer.writerow(
                {
                    "label": stats.label,
                    "parts": ",".join(stats.parts),
                    "questions": stats.question_count,
                    "positive_interactions": stats.positive_interactions,
                    "unique_agents": len(stats.unique_agents),
                    "unique_agent_ratio": f"{stats.unique_agent_ratio:.4f}",
                    "entropy": f"{stats.entropy:.4f}",
                    "effective_size": f"{stats.effective_size:.2f}",
                    "top1pct_coverage": f"{stats.head_coverage(1):.4f}",
                    "top5pct_coverage": f"{stats.head_coverage(5):.4f}",
                }
            )


def _apply_nips_style() -> None:
    mpl.rcParams.update(
        {
            "figure.dpi": 150,
            "font.size": 12,
            "axes.titlesize": 16,
            "axes.labelsize": 13,
            "legend.fontsize": 12,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.alpha": 0.3,
        }
    )
    plt.style.use("seaborn-v0_8-whitegrid")


def _plot_frequency(
    ax,
    stats: PartStats,
    color: str,
    title: str,
    log_x: bool = False,
    log_y: bool = True,
    y_max: float | None = None,
    max_yticks: int | None = None,
) -> None:
    freqs = [cnt for _, cnt in stats.agent_degree.most_common()]
    xs = range(1, len(freqs) + 1)
    ax.plot(xs, freqs, color=color, linewidth=2, label=title.split(":")[0])
    ax.set_title(title)
    ax.set_xlabel("Agent rank by positives")
    ax.set_ylabel("#Positive assignments")
    if log_y:
        ax.set_yscale("log")
    if log_x:
        ax.set_xscale("log")
    if y_max is not None:
        ax.set_ylim(0, y_max)
    if max_yticks is not None:
        ax.yaxis.set_major_locator(MaxNLocator(nbins=max_yticks))
    ax.grid(True, linestyle="--", linewidth=0.8, alpha=0.6)
    ax.legend(frameon=False)


from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

def _plot_frequency_merged_segmented_by_part(
    ax,
    per_part: Sequence[PartStats],
    color_map: dict[str, str],
    title: str,
    log_x: bool = True,
    log_y: bool = True,
    rule: str = "argmax",   # currently only argmax
) -> None:
    """
    One merged frequency curve, but colored segment-by-segment by which part
    contributes most to each agent's total degree (argmax over parts).
    """

    # build per-agent counts per part
    parts = [s.parts[0] if len(s.parts) == 1 else s.label for s in per_part]  # PartI/PartII/PartIII
    per_agent = {}  # aid -> dict(part->cnt)
    for s in per_part:
        p = s.parts[0] if len(s.parts) == 1 else s.label
        for aid, cnt in s.agent_degree.items():
            if aid not in per_agent:
                per_agent[aid] = {pp: 0 for pp in parts}
            per_agent[aid][p] += cnt

    if not per_agent:
        ax.axis("off")
        return

    # total + sort by total desc
    items = []
    for aid, d in per_agent.items():
        total = sum(d.values())
        if total > 0:
            items.append((aid, total, d))
    if not items:
        ax.axis("off")
        return

    items.sort(key=lambda x: x[1], reverse=True)

    ys = np.array([it[1] for it in items], dtype=float)
    xs = np.arange(1, len(items) + 1, dtype=float)

    # assign a part label for each point
    labels = []
    for _, _, d in items:
        # argmax contribution
        best_part = max(d.items(), key=lambda kv: kv[1])[0]
        labels.append(best_part)

    # create colored segments between consecutive points
    points = np.column_stack([xs, ys]).reshape(-1, 1, 2)
    segs = np.concatenate([points[:-1], points[1:]], axis=1)

    seg_colors = [color_map.get(labels[i], "#333333") for i in range(len(labels) - 1)]
    lc = LineCollection(segs, colors=seg_colors, linewidths=2.5)

    ax.add_collection(lc)

    # (optional) draw a thin black outline curve for readability
    ax.plot(xs, ys, color="#111111", linewidth=0.8, alpha=0.25, zorder=1)

    ax.set_title(title)
    ax.set_xlabel("Agent rank by positives")
    ax.set_ylabel("#Positive assignments")

    if log_x:
        ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")

    ax.set_xlim(1, len(xs))
    ax.set_ylim(1, max(ys) * 1.2)

    ax.grid(True, linestyle="--", linewidth=0.8, alpha=0.6)

    # custom legend
    handles = [
        Line2D([0], [0], color=color_map.get(p, "#333333"), lw=3, label=p)
        for p in ["PartI", "PartII", "PartIII"]
        if p in color_map
    ]
    ax.legend(handles=handles, frameon=False)




def _plot_degree_hist(ax, stats: PartStats, color: str) -> None:
    bins = 40
    degrees = [v for v in stats.degree_values() if v > 0]
    if not degrees:
        return

    ax.hist(
        degrees,
        bins=bins,
        alpha=0.6,
        label=stats.label,
        log=True,
        color=color,
        edgecolor="white",
        linewidth=0.5,
    )
    ax.set_title(f"{stats.label}: Agent-degree distribution (log)")
    ax.set_xlabel("#Positive assignments per agent")
    ax.set_ylabel("Agent count (log)")
    ax.grid(True, linestyle="--", linewidth=0.8, alpha=0.6)
    ax.legend(frameon=True)


def _plot_popularity_overlay(ax, per_part: Sequence[PartStats], color_map: dict[str, str]) -> None:
    """
    Merged popularity (agent-degree) distribution across parts.
    Uses log-scaled x-axis + log-spaced bins for readability under heavy tails.
    """
    degrees_by_label: dict[str, list[int]] = {}
    all_degrees: list[int] = []
    for s in per_part:
        ds = [v for v in s.degree_values() if v > 0]
        if not ds:
            continue
        degrees_by_label[s.label] = ds
        all_degrees.extend(ds)

    if not all_degrees:
        return

    bins = 40
    min_v = max(1, int(min(all_degrees)))
    max_v = int(max(all_degrees))

    if min_v == max_v:
        bin_edges = np.array([min_v * 0.9, max_v * 1.1])
    else:
        bin_edges = np.logspace(np.log10(min_v), np.log10(max_v), bins + 1)

    for label, ds in degrees_by_label.items():
        ax.hist(
            ds,
            bins=bin_edges,
            alpha=0.25,
            log=True,  # y-axis log
            label=label,
            color=color_map.get(label, "#333333"),
            edgecolor="white",
            linewidth=0.5,
            histtype="stepfilled",
        )

    ax.set_xscale("log")
    ax.set_xlim(min_v, max_v)
    ax.set_title("Popularity distribution (agent degree, overlaid)")
    ax.set_xlabel("#Positive assignments per agent (log scale)")
    ax.set_ylabel("Agent count (log)")
    ax.grid(True, linestyle="--", linewidth=0.8, alpha=0.6)
    ax.legend(frameon=True)

def _plot_popularity_ccdf(ax, per_part: Sequence[PartStats], color_map: dict[str, str]) -> None:
    """
    CCDF of agent degree: P(Degree >= x) over agents (degree > 0).
    log-log CCDF; for degenerate cases (e.g., all degrees == 1), draw a point.
    """
    any_plotted = False
    for s in per_part:
        ds = np.array([v for v in s.degree_values() if v > 0], dtype=float)
        if ds.size == 0:
            continue

        ds.sort()
        n = float(ds.size)

        uniq = np.unique(ds)
        idxs = np.searchsorted(ds, uniq, side="left")
        ccdf = (n - idxs) / n  # fraction with degree >= x

        color = color_map.get(s.label, "#333333")

        if uniq.size == 1:
            # NEW: degenerate distribution -> show as a visible point
            ax.plot(
                [uniq[0]],
                [ccdf[0]],
                marker="o",
                markersize=7,
                linestyle="None",
                color=color,
                label=s.label,
                zorder=5,
            )
        else:
            ax.step(
                uniq,
                ccdf,
                where="post",
                linewidth=2,
                color=color,
                label=s.label,
                zorder=3,
            )

        any_plotted = True

    if not any_plotted:
        return

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("Popularity CCDF (agent degree, log-log)")
    ax.set_xlabel("#Positive assignments per agent (log)")
    ax.set_ylabel("P(Degree ≥ x) (log)")
    ax.grid(True, linestyle="--", linewidth=0.8, alpha=0.6)
    ax.legend(frameon=True)


def _plot_head_coverage_curve(ax, per_part: Sequence[PartStats], color_map: dict[str, str]) -> None:
    """
    NEW: Head coverage / Pareto curve.
    x: fraction of agents (top-ranked by degree, descending)
    y: fraction of positives covered by those agents.
    More intuitive for long-tail than hist/CCDF.
    """
    any_plotted = False

    for s in per_part:
        degrees = np.array([v for v in s.degree_values() if v > 0], dtype=float)
        if degrees.size == 0:
            continue

        degrees = np.sort(degrees)[::-1]          # descending
        total = float(degrees.sum())
        if total <= 0:
            continue

        cum = np.cumsum(degrees) / total          # y: positives covered
        x = np.arange(1, degrees.size + 1) / degrees.size  # x: fraction of agents

        ax.plot(x, cum, linewidth=2, color=color_map.get(s.label, "#333333"), label=s.label)
        any_plotted = True

    if not any_plotted:
        return

    # uniform baseline
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.5, color="#666666", label="Uniform (y=x)")

    ax.set_title("Head coverage (Pareto curve)")
    ax.set_xlabel("Top fraction of agents (sorted by popularity)")
    ax.set_ylabel("Fraction of positives covered")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, linestyle="--", linewidth=0.8, alpha=0.6)
    ax.legend(frameon=True)

def _plot_head_coverage_curve_merged(ax, per_part: Sequence[PartStats]) -> None:
    """
    Merged Head coverage / Pareto curve over ALL parts.
    We first aggregate per-agent degrees across parts, then compute:
      x = top fraction of agents
      y = fraction of positives covered by those agents
    """
    total_degree = Counter()
    for s in per_part:
        total_degree.update(s.agent_degree)   # sums counts per agent across parts

    degrees = np.array([v for v in total_degree.values() if v > 0], dtype=float)
    if degrees.size == 0:
        ax.axis("off")
        return

    degrees = np.sort(degrees)[::-1]          # descending
    total = float(degrees.sum())
    cum = np.cumsum(degrees) / total          # y
    x = np.arange(1, degrees.size + 1) / degrees.size  # x

    ax.plot(x, cum, linewidth=2.5, color="#111111", label="All parts")
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.5, color="#666666", label="Uniform (y=x)")

    ax.set_title("Head coverage (Pareto curve, merged)")
    ax.set_xlabel("Top fraction of agents (sorted by popularity)")
    ax.set_ylabel("Fraction of positives covered")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, linestyle="--", linewidth=0.8, alpha=0.6)
    ax.legend(frameon=True)


def plot_regime_shift(per_part: Sequence[PartStats], output_path: Path) -> None:
    _apply_nips_style()

    # Part III color changed to GREEN everywhere
    color_map = {
        "PartI": "#1f77b4",   # blue
        "PartII": "#d62728",  # red
        "PartIII": "#2ca02c", # green
    }

    # 3x3: first 2 rows keep frequency + per-part degree;
    # third row: overlay + CCDF.
    fig, axes = plt.subplots(
        3, 3, figsize=(21, 13.0), gridspec_kw={"wspace": 0.25, "hspace": 0.45}
    )

    part_lookup = {"/".join(s.parts): s for s in per_part}
    part_i = part_lookup.get("PartI")
    part_ii = part_lookup.get("PartII")
    part_iii = part_lookup.get("PartIII")

    # Row 0: frequency curves
    if part_i:
        _plot_frequency(
            axes[0, 0],
            part_i,
            color_map["PartI"],
            "Part I: Agent frequency curve",
            log_x=False,
            log_y=True,
        )
    else:
        axes[0, 0].axis("off")

    if part_ii:
        _plot_frequency(
            axes[0, 1],
            part_ii,
            color_map["PartII"],
            "Part II: Agent frequency curve",
            log_x=True,
            log_y=True,
        )
    else:
        axes[0, 1].axis("off")

    if part_iii:
        freqs = [cnt for _, cnt in part_iii.agent_degree.most_common()]
        y_max = (max(freqs) + 1) if freqs else 1
        _plot_frequency(
            axes[0, 2],
            part_iii,
            color_map["PartIII"],
            "Part III: Agent frequency curve",
            log_x=True,
            log_y=False,
            y_max=float(y_max),
            max_yticks=4,
        )
    else:
        axes[0, 2].axis("off")

    # Row 1: degree histograms (separate)
    if part_i:
        _plot_degree_hist(axes[1, 0], part_i, color_map["PartI"])
    else:
        axes[1, 0].axis("off")

    if part_ii:
        _plot_degree_hist(axes[1, 1], part_ii, color_map["PartII"])
    else:
        axes[1, 1].axis("off")

    if part_iii:
        _plot_degree_hist(axes[1, 2], part_iii, color_map["PartIII"])
    else:
        axes[1, 2].axis("off")

    # Row 2: merged popularity + CCDF
    _plot_head_coverage_curve_merged(axes[2, 0], per_part)   # NEW
    #_plot_popularity_overlay(axes[2, 1], per_part, color_map)
    merged = _merge_stats_for_frequency(per_part)
    _plot_frequency_merged_segmented_by_part(
        axes[2, 1],
        per_part,
        color_map=color_map,
        title="All parts: Agent frequency curve (segmented by dominant part)",
        log_x=False,
        log_y=True,
    )
    _plot_popularity_ccdf(axes[2, 2], per_part, color_map)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

def _merge_stats_for_frequency(per_part: Sequence[PartStats]) -> PartStats:
    total_degree = Counter()
    unique_agents: set[str] = set()
    question_count = 0
    positive_interactions = 0

    for s in per_part:
        total_degree.update(s.agent_degree)          # sum degrees across parts
        unique_agents.update(s.unique_agents)
        question_count += s.question_count
        positive_interactions += s.positive_interactions

    return PartStats(
        label="AllParts",
        parts=["AllParts"],
        question_count=question_count,
        positive_interactions=positive_interactions,
        agent_degree=total_degree,
        unique_agents=unique_agents,
    )



def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Regime shift evidence: dense vs one-off interactions.")
    ap.add_argument("--data_root", required=True, help="Path to dataset root containing PartI/PartII/PartIII.")
    ap.add_argument(
        "--output_dir",
        type=Path,
        default=Path("analysis") / "artifacts",
        help="Directory to save the figure and metrics table.",
    )
    ap.add_argument(
        "--parts",
        default="PartI,PartII,PartIII",
        help="Comma separated list of parts to include (default: all parts).",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    parts = [p.strip() for p in args.parts.split(",") if p.strip()]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    bundle = collect_data(args.data_root, parts=parts)

    part_stats: list[PartStats] = []
    for part in ["PartI", "PartII", "PartIII"]:
        if part in parts:
            part_stats.append(collect_part_stats(bundle, [part]))

    if not part_stats:
        raise ValueError(f"No part statistics collected. Check parts filter: {parts}")

    metrics_path = args.output_dir / "regime_shift_metrics.csv"
    export_metrics_table(part_stats, metrics_path)

    figure_path = args.output_dir / "regime_shift_frequency.png"
    plot_regime_shift(per_part=part_stats, output_path=figure_path)

    print(f"[OK] Metrics table saved to: {metrics_path}")
    print(f"[OK] Figure saved to: {figure_path}")


if __name__ == "__main__":
    main()
