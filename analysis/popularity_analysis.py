#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Regime-shift evidence based on *content-derived popularity*.

We replace "agent degree (#positive assignments by agent id)" with a continuous
popularity score computed from agent content:

    popularity(agent) = LLM_frequency(llm(agent)) + mean( Tool_frequency(tool in agent) )

Tool frequencies are winsorized (clipped) at a percentile (default 99.5) to reduce
the impact of extreme tools.

This script produces the SAME 3x3 (9 plots) layout as the ID-degree version, but
all plots are computed on popularity scores (continuous) instead of raw degrees.

Outputs:
  - analysis/artifacts/regime_shift_popularity.png
  - analysis/artifacts/regime_shift_popularity_metrics.csv
  - cache file with popularity stats json
"""

from __future__ import annotations
import matplotlib.patheffects as pe

import argparse
import csv
import math
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple, Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

from agent_rec.data import DatasetBundle, collect_data, ensure_cache_dir
from agent_rec.config import pos_topk_for_part

from collections import Counter

# -------------------------
# Popularity computation
# -------------------------

UNKNOWN_LLM = "<UNK_LLM>"

def _mode_of_values(values: Iterable[float]) -> float:
    """
    Robust mode over numeric values.
    - We round to avoid float tiny-diff issues.
    - If multiple modes (ties), return median of tied values for stability.
    - If all unique, this will still return one value (tie set = all), so median works.
    """
    vals = [float(v) for v in values]
    if not vals:
        return 0.0

    # round to stabilize float keys (counts are usually integer-like)
    rounded = [round(v, 6) for v in vals]
    c = Counter(rounded)
    max_cnt = max(c.values())
    modes = sorted([v for v, k in c.items() if k == max_cnt])
    if len(modes) == 1:
        return float(modes[0])
    return float(np.median(modes))

def compute_mode_fill_values(
    llm_frequency: Mapping[str, float],
    tool_frequency: Mapping[str, float],
) -> Tuple[float, float]:
    """
    Compute (mode_llm_freq, mode_tool_freq) from frequency tables.
    Used when LLM/tools are missing.
    """
    mode_llm = _mode_of_values(llm_frequency.values()) if llm_frequency else 0.0
    mode_tool = _mode_of_values(tool_frequency.values()) if tool_frequency else 0.0
    print(f"[Info] Mode LLM frequency fill value: {mode_llm:.4f}")
    print(f"[Info] Mode Tool frequency fill value: {mode_tool:.4f}")
    return float(mode_llm), float(mode_tool)



def _agent_llm_and_tools(agent: Dict | None) -> Tuple[str, list[str]]:
    agent = agent or {}
    model = agent.get("M") or {}
    tool_info = agent.get("T") or {}
    llm_name = (model.get("name") or "").strip()
    llm_token = llm_name
    tools = list(tool_info.get("tools") or [])
    return llm_token, tools


def _winsorize_counter(counter: Mapping[str, float], percentile: float) -> Tuple[Dict[str, float], float]:
    """Clip values to the given percentile threshold."""
    if not counter:
        return {}, 0.0
    values = np.array(list(counter.values()), dtype=float)
    threshold = float(np.percentile(values, percentile))
    clipped = {k: float(min(v, threshold)) for k, v in counter.items()}
    return clipped, threshold


def _minmax_normalize_counter(counter: Mapping[str, float]) -> Dict[str, float]:
    """
    Min-max normalize values to [0,1].
    If all values are equal (or only one item), return 0.0 for all to avoid NaNs.
    """
    if not counter:
        return {}
    vals = np.array(list(counter.values()), dtype=float)
    vmin = float(vals.min())
    vmax = float(vals.max())
    if vmax <= vmin:  # all equal
        return {k: 0.0 for k in counter.keys()}
    return {k: float((float(v) - vmin) / (vmax - vmin)) for k, v in counter.items()}


def compute_llm_and_tool_frequencies_from_agent_degree(
    agent_degree: Mapping[str, float],
    all_agents: Dict[str, dict],
    *,
    tool_percentile: float = 99.5,
) -> Tuple[Dict[str, float], Dict[str, float], float]:
    """
    Aggregate assignment counts by LLM token and tool token from agent_degree.
    Note: agent_degree counts come from positives (same as your previous scripts).
    """
    llm_freq: Dict[str, float] = {}
    tool_freq: Dict[str, float] = {}

    for aid, cnt in agent_degree.items():
        llm_id, tools = _agent_llm_and_tools(all_agents.get(aid))
        llm_freq[llm_id] = llm_freq.get(llm_id, 0.0) + float(cnt)
        for t in tools:
            tool_freq[t] = tool_freq.get(t, 0.0) + float(cnt)

    clipped_tool_freq, threshold = _winsorize_counter(tool_freq, tool_percentile)
    return llm_freq, clipped_tool_freq, threshold

def compute_agent_popularity(
    agent_ids: Iterable[str],
    all_agents: Dict[str, dict],
    llm_frequency: Mapping[str, float],
    tool_frequency: Mapping[str, float],
) -> Dict[str, float]:
    """
    Minimal logic (no filling):
      - only M present -> pop = M
      - only T present -> pop = T
      - both present   -> pop = mean(M, T)
      - neither        -> 0
    Missing in frequency table => treat as absent.
    """
    popularity: Dict[str, float] = {}

    for aid in agent_ids:
        llm_id, tools = _agent_llm_and_tools(all_agents.get(aid))

        # ----- M part -----
        m_val = None
        if llm_id and llm_id != UNKNOWN_LLM:
            v = llm_frequency.get(llm_id, None)
            if v is not None and float(v) > 0:
                m_val = float(v)

        # ----- T part -----
        t_val = None
        if tools:
            tool_vals = [float(tool_frequency[t]) for t in tools if t in tool_frequency and float(tool_frequency[t]) > 0]
            if tool_vals:
                t_val = float(np.mean(tool_vals))

        # ----- combine -----
        if m_val is not None and t_val is not None:
            popularity[aid] = 0.5 * (m_val + t_val)
        elif m_val is not None:
            popularity[aid] = m_val
        elif t_val is not None:
            popularity[aid] = t_val
        else:
            popularity[aid] = 0.0

    return popularity



def cache_popularity(
    cache_dir: Path,
    *,
    parts: Sequence[str],
    llm_frequency: Mapping[str, float],
    tool_frequency: Mapping[str, float],
    tool_clip_threshold: float,
    tool_percentile: float,
    agent_popularity: Mapping[str, float],
) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "parts": list(parts),
        "tool_percentile": tool_percentile,
        "tool_clip_threshold": tool_clip_threshold,
        "llm_frequency": dict(llm_frequency),
        "tool_frequency": dict(tool_frequency),
        "agent_popularity": dict(agent_popularity),
    }
    cache_path = cache_dir / "popularity_stats.json"
    with cache_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return cache_path


# -------------------------
# Stats objects (now on popularity)
# -------------------------

@dataclass
class PartPopStats:
    label: str
    parts: List[str]
    question_count: int
    positive_interactions: int
    # original degrees still kept so we can build llm/tool frequencies consistently
    agent_degree: Dict[str, float]
    unique_agents: set[str]

    # popularity score per agent id
    agent_popularity: Dict[str, float]

    def popularity_values(self) -> List[float]:
        return [float(v) for v in self.agent_popularity.values() if v is not None]

    @property
    def unique_agent_ratio(self) -> float:
        if self.positive_interactions == 0:
            return 0.0
        return len(self.unique_agents) / self.positive_interactions

    @property
    def entropy_popularity(self) -> float:
        """
        Entropy over normalized popularity mass (treat popularity as weight).
        This replaces degree-entropy (which was on interaction counts).
        """
        vals = np.array(self.popularity_values(), dtype=float)
        vals = vals[vals > 0]
        if vals.size == 0:
            return 0.0
        p = vals / vals.sum()
        return float(-(p * np.log(p)).sum())

    @property
    def effective_size_popularity(self) -> float:
        return float(math.exp(self.entropy_popularity))

    def head_coverage_by_popularity(self, percentile: float) -> float:
        """
        Coverage measured on popularity mass:
          top x% agents (ranked by popularity) cover y% of total popularity.
        """
        vals = np.array(sorted(self.popularity_values(), reverse=True), dtype=float)
        vals = vals[vals > 0]
        if vals.size == 0:
            return 0.0
        top_n = max(1, int(math.ceil(vals.size * percentile / 100.0)))
        return float(vals[:top_n].sum() / vals.sum())


def collect_part_pop_stats(
    bundle: DatasetBundle,
    parts: Iterable[str],
    *,
    llm_freq: Mapping[str, float],
    tool_freq: Mapping[str, float],
    mode_llm, mode_tool
) -> PartPopStats:
    part_set = set(parts)
    agent_degree: Dict[str, float] = {}
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

        for aid in positives:
            agent_degree[aid] = agent_degree.get(aid, 0.0) + 1.0
        unique_agents.update(positives)
        positive_interactions += len(positives)

    label = "/".join(sorted(part_set))
    agent_popularity = compute_agent_popularity(
        agent_degree.keys(),
        bundle.all_agents,
        llm_freq,
        tool_freq,
    )
    
    return PartPopStats(
        label=label,
        parts=sorted(part_set),
        question_count=question_count,
        positive_interactions=positive_interactions,
        agent_degree=agent_degree,
        unique_agents=unique_agents,
        agent_popularity=agent_popularity,
    )


def _merge_pop_stats(per_part: Sequence[PartPopStats]) -> PartPopStats:
    total_degree: Dict[str, float] = {}
    unique_agents: set[str] = set()
    question_count = 0
    positive_interactions = 0

    for s in per_part:
        for aid, v in s.agent_degree.items():
            total_degree[aid] = total_degree.get(aid, 0.0) + float(v)
        unique_agents.update(s.unique_agents)
        question_count += s.question_count
        positive_interactions += s.positive_interactions

    # merged popularity will be set later by caller (depends on global llm/tool freq)
    return PartPopStats(
        label="AllParts",
        parts=["AllParts"],
        question_count=question_count,
        positive_interactions=positive_interactions,
        agent_degree=total_degree,
        unique_agents=unique_agents,
        agent_popularity={},  # fill later
    )


# -------------------------
# Export metrics
# -------------------------

def export_popularity_metrics_table(stats_list: Sequence[PartPopStats], output_csv: Path) -> None:
    fieldnames = [
        "label",
        "parts",
        "questions",
        "positive_interactions",
        "unique_agents",
        "unique_agent_ratio",
        "entropy_popularity",
        "effective_size_popularity",
        "top1pct_popularity_coverage",
        "top5pct_popularity_coverage",
    ]
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for s in stats_list:
            writer.writerow(
                {
                    "label": s.label,
                    "parts": ",".join(s.parts),
                    "questions": s.question_count,
                    "positive_interactions": s.positive_interactions,
                    "unique_agents": len(s.unique_agents),
                    "unique_agent_ratio": f"{s.unique_agent_ratio:.4f}",
                    "entropy_popularity": f"{s.entropy_popularity:.4f}",
                    "effective_size_popularity": f"{s.effective_size_popularity:.2f}",
                    "top1pct_popularity_coverage": f"{s.head_coverage_by_popularity(1):.4f}",
                    "top5pct_popularity_coverage": f"{s.head_coverage_by_popularity(5):.4f}",
                }
            )


# -------------------------
# Plot styling
# -------------------------
def _apply_nips_style() -> None:
    """NeurIPS-like, print-friendly defaults."""
    mpl.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,

            "font.size": 12,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 11,

            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 1.0,
            "lines.linewidth": 2.2,

            # subtle grid (major only)
            "axes.grid": True,
            "grid.linestyle": ":",
            "grid.linewidth": 0.8,
            "grid.alpha": 0.25,

            "axes.facecolor": "white",
            "figure.facecolor": "white",
        }
    )
    # 不要 seaborn style（更像论文）



# -------------------------
# Plot helpers (now on popularity values)
# -------------------------

def _plot_popularity_curve(
    ax,
    stats: PartPopStats,
    color: str,
    title: str,
    log_x: bool = False,
    log_y: bool = True,
    y_max: float | None = None,
    max_yticks: int | None = None,
) -> None:
    vals = sorted(stats.popularity_values(), reverse=True)
    if not vals:
        ax.axis("off")
        return

    xs = np.arange(1, len(vals) + 1, dtype=float)
    ax.plot(xs, vals, color=color, linewidth=2, label=title.split(":")[0])
    ax.set_title(title)
    ax.set_xlabel("Agent rank by popularity")
    ax.set_ylabel("Popularity score")
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

import matplotlib as mpl

def apply_pub_style(fontsize=12):
    """NeurIPS-like, print-friendly."""
    mpl.rcParams.update({
        # Fonts
        "font.size": fontsize,
        "axes.titlesize": fontsize + 1,
        "axes.labelsize": fontsize,
        "xtick.labelsize": fontsize - 1,
        "ytick.labelsize": fontsize - 1,
        "legend.fontsize": fontsize - 1,

        # Lines
        "lines.linewidth": 2.0,
        "axes.linewidth": 1.0,

        # Layout
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,

        # Clean look
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
    })
    
import numpy as np
from matplotlib.ticker import LogLocator, NullFormatter

def _plot_freq_hist_and_ccdf(
    ax_hist,
    ax_ccdf,
    values,
    title_prefix: str,
    *,
    subtitle: str | None = None,
    bins: int = 36,
    xlog: bool = False,
    hist_ylog: bool = False,
    ccdf_ylog: bool = False,
    density: bool = False,      # True: 画密度；False: 画count
    annotate: bool = False,
):
    # ---- sanitize ----
    vals = np.asarray([float(v) for v in values], dtype=float)
    vals = vals[np.isfinite(vals) & (vals > 0)]
    if vals.size == 0:
        ax_hist.axis("off")
        ax_ccdf.axis("off")
        return

    vals.sort()
    n = vals.size
    vmin = float(vals[0])
    vmax = float(vals[-1])

    # ---- bins: log-spaced bins on x if xlog else linear ----
    if vmin == vmax:
        edges = np.array([vmin * 0.9, vmax * 1.1], dtype=float)
    else:
        if xlog:
            edges = np.logspace(np.log10(vmin), np.log10(vmax), bins + 1)
        else:
            edges = np.linspace(vmin, vmax, bins + 1)

    # =========================
    # Histogram
    # =========================
    ax_hist.hist(
        vals,
        bins=edges,
        density=density,
        alpha=0.35,
        edgecolor="black",
        linewidth=0.6,
    )

    # Scales
    if xlog:
        ax_hist.set_xscale("log")
    if hist_ylog:
        ax_hist.set_yscale("log")

    # Labels
    ax_hist.set_xlabel("Frequency")
    ax_hist.set_ylabel("Density" if density else "Count")

    # Title (short) + subtitle (small)
    ax_hist.set_title(f"{title_prefix} · Frequency distribution", loc="left", pad=6)
    if subtitle:
        ax_hist.text(
            0.0, 1.02, subtitle,
            transform=ax_hist.transAxes,
            ha="left", va="bottom",
            fontsize=max(9, mpl.rcParams["font.size"] - 2),
            alpha=0.9,
        )

    # Grid: light, only major
    ax_hist.grid(True, which="major", linestyle="--", linewidth=0.8, alpha=0.35)
    ax_hist.tick_params(axis="both", which="major", length=5, width=1.0)
    ax_hist.tick_params(axis="both", which="minor", length=3, width=0.8)

    # Better log ticks (avoid clutter)
    if xlog:
        ax_hist.xaxis.set_major_locator(LogLocator(base=10.0, numticks=6))
        ax_hist.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1))
        ax_hist.xaxis.set_minor_formatter(NullFormatter())

    # Info box
    if annotate:
        p50 = float(np.median(vals))
        p90 = float(np.quantile(vals, 0.90))
        p99 = float(np.quantile(vals, 0.99))
        txt = (
            f"n={n:,}\n"
            f"min={vmin:.2g}\n"
            f"median={p50:.2g}\n"
            f"p90={p90:.2g}\n"
            f"p99={p99:.2g}"
        )
        ax_hist.text(
            0.98, 0.98, txt,
            transform=ax_hist.transAxes,
            ha="right", va="top",
            fontsize=max(9, mpl.rcParams["font.size"] - 2),
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="black", linewidth=0.6, alpha=0.85),
        )

    # =========================
    # CCDF
    # =========================
    # empirical CCDF on unique support
    uniq = np.unique(vals)
    idxs = np.searchsorted(vals, uniq, side="left")
    ccdf = (n - idxs) / float(n)

    if uniq.size == 1:
        ax_ccdf.plot([uniq[0]], [ccdf[0]], marker="o", markersize=7, linestyle="None", zorder=5)
    else:
        ax_ccdf.step(uniq, ccdf, where="post")

    if xlog:
        ax_ccdf.set_xscale("log")
    if ccdf_ylog:
        ax_ccdf.set_yscale("log")

    ax_ccdf.set_xlabel("Frequency")
    ax_ccdf.set_ylabel("CCDF  P(Frequency ≥ x)")
    ax_ccdf.set_title(f"{title_prefix} · CCDF", loc="left", pad=6)

    ax_ccdf.grid(True, which="major", linestyle="--", linewidth=0.8, alpha=0.35)
    ax_ccdf.tick_params(axis="both", which="major", length=5, width=1.0)
    ax_ccdf.tick_params(axis="both", which="minor", length=3, width=0.8)

    if xlog:
        ax_ccdf.xaxis.set_major_locator(LogLocator(base=10.0, numticks=6))
        ax_ccdf.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1))
        ax_ccdf.xaxis.set_minor_formatter(NullFormatter())
    if ccdf_ylog:
        ax_ccdf.yaxis.set_major_locator(LogLocator(base=10.0, numticks=6))
        ax_ccdf.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1))
        ax_ccdf.yaxis.set_minor_formatter(NullFormatter())

    # make both panels share xlim neatly
    left = vmin * (0.85 if xlog else 1.0)
    right = vmax * (1.15 if xlog else 1.0)
    ax_hist.set_xlim(left, right)
    ax_ccdf.set_xlim(left, right)

def plot_llm_tool_frequency_distributions(
    llm_freq: Mapping[str, float],
    tool_freq: Mapping[str, float],
    output_path: Path,
    *,
    llm_percentile: float = 99.5,
    llm_min_freq: float = 2.0,
) -> None:
    _apply_nips_style()

    # ---- LLM: drop tiny + winsorize for plot ----
    llm_freq_f = _filter_min_frequency(llm_freq, llm_min_freq)
    llm_freq_plot, llm_thr = _winsorize_counter(llm_freq_f, llm_percentile)
    llm_mode = _mode_of_values(llm_freq_f.values()) if llm_freq_f else 0.0

    # ---- Tool: already winsorized upstream usually, but we just plot as-is ----
    tool_mode = _mode_of_values(tool_freq.values()) if tool_freq else 0.0


    fig, axes = plt.subplots(
        2, 2, figsize=(8, 7),
        gridspec_kw={"wspace": 0.25, "hspace": 0.35}
    )

    _plot_freq_hist_and_ccdf(
        axes[0, 0], axes[0, 1],
        list(llm_freq_plot.values()),
        f"LLM (drop <{llm_min_freq:g}, winsor p={llm_percentile:g}, thr={llm_thr:.2g})",
    )
    _plot_freq_hist_and_ccdf(
        axes[1, 0], axes[1, 1],
        list(tool_freq.values()),
        "Tool (winsorized)",
    )


    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)





def _plot_popularity_hist(ax, stats: PartPopStats, color: str) -> None:
    bins = 40
    vals = np.array(stats.popularity_values(), dtype=float)
    vals = vals[vals > 0]
    if vals.size == 0:
        ax.axis("off")
        return

    # log-spaced bins for heavy tail readability
    min_v = float(max(1e-9, vals.min()))
    max_v = float(vals.max())
    if min_v == max_v:
        edges = np.array([min_v * 0.9, max_v * 1.1])
    else:
        edges = np.logspace(np.log10(min_v), np.log10(max_v), bins + 1)

    ax.hist(
        vals,
        bins=edges,
        alpha=0.6,
        label=stats.label,
        log=True,  # y log
        color=color,
        edgecolor="white",
        linewidth=0.5,
    )
    ax.set_xscale("log")
    ax.set_title(f"{stats.label}: Popularity distribution (log bins, y log)")
    ax.set_xlabel("Popularity score (log)")
    ax.set_ylabel("Agent count (log)")
    ax.grid(True, linestyle="--", linewidth=0.8, alpha=0.6)
    ax.legend(frameon=True)


def _plot_popularity_ccdf(ax, per_part: Sequence[PartPopStats], color_map: dict[str, str]) -> None:
    """
    CCDF of popularity: P(Popularity >= x) over agents.
    log-log, with marker styles to make PartI salient.
    """
    markers = {"PartI": "^", "PartII": "o", "PartIII": "s"}
    lw_map = {"PartI": 3.2, "PartII": 2.2, "PartIII": 2.2}
    any_plotted = False

    for s in per_part:
        label = s.parts[0] if len(s.parts) == 1 else s.label
        vals = np.array([v for v in s.popularity_values() if v > 0], dtype=float)
        if vals.size == 0:
            continue
        vals.sort()
        n = float(vals.size)

        uniq = np.unique(vals)
        idxs = np.searchsorted(vals, uniq, side="left")
        ccdf = (n - idxs) / n  # in (0,1]

        color = color_map.get(label, "#333333")
        mk = markers.get(label, None)
        lw = lw_map.get(label, 2.2)

        # Use plot with drawstyle steps-post so markers are allowed
        line = ax.plot(
            uniq, ccdf,
            color=color,
            linewidth=lw,
            drawstyle="steps-post",
            marker=mk if label == "PartI" else None,   # markers only for PartI (less clutter)
            markevery=max(1, len(uniq) // 12) if label == "PartI" else None,
            markersize=7 if label == "PartI" else 0,
            markerfacecolor=color if label == "PartI" else None,
            markeredgecolor="black" if label == "PartI" else None,
            markeredgewidth=0.6 if label == "PartI" else None,
            label=label,
            zorder=4 if label == "PartI" else 3,
        )[0]

        if label == "PartI":
            # white halo so the curve stands out on top of others
            line.set_path_effects([pe.Stroke(linewidth=lw + 2.2, foreground="white"), pe.Normal()])

        any_plotted = True

    if not any_plotted:
        ax.axis("off")
        return

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("Popularity CCDF (log-log)")
    ax.set_xlabel("Popularity score (log)")
    ax.set_ylabel("P(Popularity ≥ x) (log)")

    ax.grid(True, which="major", linestyle=":", linewidth=0.8, alpha=0.25)
    ax.legend(frameon=False, loc="upper right")



def _plot_head_coverage_curve_merged(ax, per_part: Sequence[PartPopStats]) -> None:
    """
    Pareto curve on merged popularity:
      x = top fraction of agents (by popularity)
      y = fraction of total popularity mass covered
    Adds key quantile markers for readability.
    """
    total_pop = {}
    for s in per_part:
        for aid, v in s.agent_popularity.items():
            total_pop[aid] = total_pop.get(aid, 0.0) + float(v)

    vals = np.array([v for v in total_pop.values() if v > 0], dtype=float)
    if vals.size == 0:
        ax.axis("off")
        return

    vals = np.sort(vals)[::-1]
    total = float(vals.sum())
    cum = np.cumsum(vals) / total
    x = np.arange(1, vals.size + 1) / vals.size

    ax.plot(x, cum, linewidth=2.6, color="#111111", label="All parts", zorder=3)
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.4, color="#666666", alpha=0.9, label="Uniform (y=x)", zorder=1)

    # mark key points
    for frac in [0.01, 0.05, 0.10]:
        idx = max(1, int(np.ceil(frac * vals.size))) - 1
        ax.scatter([x[idx]], [cum[idx]], s=36, marker="o", edgecolor="black", facecolor="white", linewidth=0.8, zorder=4)
        ax.text(
            x[idx] + 0.03, cum[idx] + 0.03,
            f"top {int(frac*100)}% → {cum[idx]*100:.1f}%",
            fontsize=10,
            ha="left", va="top",
        )

    ax.set_title("Head coverage (Pareto curve, All parts)")
    ax.set_xlabel("Top fraction of agents (sorted by popularity)")
    ax.set_ylabel("Fraction of total popularity covered")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.grid(True, which="major", linestyle=":", linewidth=0.8, alpha=0.25)
    ax.legend(frameon=False, loc="lower right")



def _plot_popularity_overlay(ax, per_part: Sequence[PartPopStats], color_map: dict[str, str]) -> None:
    """
    Overlaid popularity distributions (log x + log y).
    """
    any_plotted = False
    all_vals = []
    by_label: Dict[str, np.ndarray] = {}

    for s in per_part:
        vals = np.array([v for v in s.popularity_values() if v > 0], dtype=float)
        if vals.size == 0:
            continue
        by_label[s.label] = vals
        all_vals.append(vals)
        any_plotted = True

    if not any_plotted:
        ax.axis("off")
        return

    all_vals = np.concatenate(all_vals)
    bins = 40
    min_v = float(max(1e-9, all_vals.min()))
    max_v = float(all_vals.max())
    if min_v == max_v:
        edges = np.array([min_v * 0.9, max_v * 1.1])
    else:
        edges = np.logspace(np.log10(min_v), np.log10(max_v), bins + 1)

    for label, vals in by_label.items():
        ax.hist(
            vals,
            bins=edges,
            alpha=0.25,
            log=True,
            label=label,
            color=color_map.get(label, "#333333"),
            edgecolor="white",
            linewidth=0.5,
            histtype="stepfilled",
        )

    ax.set_xscale("log")
    ax.set_title("Popularity distribution (overlaid)")
    ax.set_xlabel("Popularity score (log)")
    ax.set_ylabel("Agent count (log)")
    ax.grid(True, linestyle="--", linewidth=0.8, alpha=0.6)
    ax.legend(frameon=True)

def _plot_popularity_curve_per_part_style(
    ax,
    per_part: Sequence[PartPopStats],
    color_map: dict[str, str],
    title: str,
    *,
    log_x: bool = False,
    log_y: bool = False,
    max_points: int = 16,     # 稀疏程度：12~25都行
    head_keep: int = 4,       # 头部保留更密一点
    draw_merged_black: bool = True,
) -> None:
    """
    Draw per-part contribution curves on the same merged rank axis + optional
    black solid merged-total curve.

    x: agent rank by TOTAL merged popularity
    y: per-part contribution / total popularity at that rank

    Style: dashed lines + hollow markers for parts; solid black for merged.
    """
    # --- determine parts (labels) ---
    part_names = [s.parts[0] if len(s.parts) == 1 else s.label for s in per_part]

    # --- build per-agent contributions by part ---
    per_agent: Dict[str, Dict[str, float]] = {}
    for s in per_part:
        p = s.parts[0] if len(s.parts) == 1 else s.label
        for aid, v in s.agent_popularity.items():
            if aid not in per_agent:
                per_agent[aid] = {pp: 0.0 for pp in part_names}
            per_agent[aid][p] += float(v)

    # --- make merged ranking by TOTAL popularity ---
    items = []
    for aid, d in per_agent.items():
        total = float(sum(d.values()))
        if total > 0:
            items.append((aid, total, d))
    if not items:
        ax.axis("off")
        return

    items.sort(key=lambda x: x[1], reverse=True)

    total_y_full = np.array([it[1] for it in items], dtype=float)
    xs_full = np.arange(1, len(items) + 1, dtype=float)

    # --- sparse sampling indices: keep head + evenly spaced tail ---
    n = len(items)
    if n <= max_points:
        keep = np.arange(n, dtype=int)
    else:
        head = min(head_keep, n)
        rest = max(1, max_points - head)
        tail_idx = np.unique(np.round(np.linspace(head, n - 1, rest)).astype(int))
        keep = np.unique(np.concatenate([np.arange(head, dtype=int), tail_idx]))
        keep.sort()

    xs = xs_full[keep]
    total_y = total_y_full[keep]

    # --- style map like your example ---
    marker_map = {"PartI": "^", "PartII": "o", "PartIII": "s"}
    ls_part = "--"

    # 1) merged black solid curve (behind)
    if draw_merged_black:
        ax.plot(
            xs,
            total_y,
            linestyle="-",
            linewidth=2.0,
            color="black",
            label="All parts",
            zorder=2,
        )

    # 2) per-part dashed + hollow markers (on top)
    for p in ["PartI", "PartII", "PartIII"]:
        if p not in part_names:
            continue

        ys_full = np.array([it[2].get(p, 0.0) for it in items], dtype=float)
        ys = ys_full[keep]

        ax.plot(
            xs,
            ys,
            linestyle=ls_part,
            linewidth=1.6,
            color=color_map.get(p, "#333333"),
            marker=marker_map.get(p, "o"),
            markersize=10,
            markerfacecolor="none",  # hollow
            markeredgecolor=color_map.get(p, "#333333"),
            markeredgewidth=1.6,
            label=p,
            zorder=3,
        )

    # --- axes formatting ---
    ax.set_title(title)
    ax.set_xlabel("Agent rank by popularity")
    ax.set_ylabel("Popularity score")

    if log_x:
        ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")

    # y-limits (use merged total for neat view)
    y_pos = total_y[np.isfinite(total_y) & (total_y > 0)]
    if y_pos.size > 0:
        ax.set_ylim(0, float(np.max(y_pos)) * 1.10)

    ax.grid(True, which="major", linestyle=":", linewidth=0.8, alpha=0.25)

    ax.legend(
        frameon=True,
        fancybox=True,
        framealpha=0.9,
        loc="upper right",
        borderpad=0.6,
        handlelength=2.6,
        handletextpad=0.8,
        labelspacing=0.45,
    )





def _plot_popularity_merged_segmented_by_part(
    ax,
    per_part: Sequence[PartPopStats],
    color_map: dict[str, str],
    title: str,
    log_x: bool = False,
    log_y: bool = False,
    *,
    highlight_part: str = "PartI",
    # visibility for PartII/III
    lw_part23: float = 3.0,
    alpha_part23: float = 0.90,
    part3_dashed: bool = True,
    # PartI highlight (salient but not occluding)
    lw_part1_base: float = 1.8,
    alpha_part1_base: float = 0.55,
    lw_part1_highlight: float = 3.4,
    # markers (make sparse)
    marker_size: float = 50.0,
    max_markers: int = 16,
    min_marker_gap: int = 1000,   # rank gap between markers
    # downsample curve to avoid over-dense rendering
    max_plot_points: int = 30000,
    head_keep: int = 400,
) -> None:
    """
    Merged popularity curve segmented by dominant part.
    Improvements:
      1) Downsample the curve (keep head dense, tail sparse) to avoid over-plot density.
      2) PartII/III: thick + semi-transparent + white halo to stay visible under overlap.
      3) PartI: highlighted segments + very sparse hollow triangle markers on boundary points.
    """

    # ------------------------------
    # Build per-agent contributions
    # ------------------------------
    parts = [s.parts[0] if len(s.parts) == 1 else s.label for s in per_part]
    per_agent_pop: Dict[str, Dict[str, float]] = {}

    for s in per_part:
        p = s.parts[0] if len(s.parts) == 1 else s.label
        for aid, popv in s.agent_popularity.items():
            if aid not in per_agent_pop:
                per_agent_pop[aid] = {pp: 0.0 for pp in parts}
            per_agent_pop[aid][p] += float(popv)

    items = []
    for aid, d in per_agent_pop.items():
        total = float(sum(d.values()))
        if total > 0:
            items.append((aid, total, d))

    if not items:
        ax.axis("off")
        return

    # sort by total popularity
    items.sort(key=lambda x: x[1], reverse=True)
    ys_full = np.array([it[1] for it in items], dtype=float)
    xs_full = np.arange(1, len(items) + 1, dtype=float)

    # dominant label per point
    dom_full = []
    for _, _, d in items:
        dom_full.append(max(d.items(), key=lambda kv: kv[1])[0])
    dom_full = np.array(dom_full, dtype=object)

    # ------------------------------
    # Downsample indices (head dense + tail log-spaced)
    # ------------------------------
    n = len(items)
    if n <= max_plot_points:
        keep_idx = np.arange(n, dtype=int)
    else:
        head = min(head_keep, n)
        tail = n - head
        # allocate remaining points to tail
        tail_keep = max(50, max_plot_points - head)

        # log-spaced positions in [0, tail-1]
        # (more points earlier in tail, fewer later)
        if tail <= 1:
            tail_idx = np.array([], dtype=int)
        else:
            pos = np.logspace(0, np.log10(tail), tail_keep, base=10) - 1.0
            pos = np.unique(np.clip(np.round(pos).astype(int), 0, tail - 1))
            tail_idx = head + pos

        keep_idx = np.unique(np.concatenate([np.arange(head, dtype=int), tail_idx]))
        keep_idx.sort()

    xs = xs_full[keep_idx]
    ys = ys_full[keep_idx]
    dom = dom_full[keep_idx]

    # ------------------------------
    # Build segments on downsampled points
    # ------------------------------
    pts = np.column_stack([xs, ys]).reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    seg_dom = dom[:-1]

    ls_part3 = (0, (6, 3)) if part3_dashed else "solid"
    style = {
        "PartII": dict(lw=lw_part23, alpha=alpha_part23, ls="solid", z=2),
        "PartIII": dict(lw=lw_part23, alpha=alpha_part23, ls=ls_part3, z=2),
        "PartI": dict(lw=lw_part1_base, alpha=alpha_part1_base, ls="solid", z=3),
    }

    # Draw PartII/III first, then PartI
    for p in ["PartII", "PartIII", "PartI"]:
        mask = (seg_dom == p)
        if not np.any(mask):
            continue

        lc = LineCollection(
            segs[mask],
            colors=[color_map.get(p, "#333333")] * int(np.sum(mask)),
            linewidths=float(style.get(p, {}).get("lw", 2.0)),
            alpha=float(style.get(p, {}).get("alpha", 0.8)),
            linestyles=style.get(p, {}).get("ls", "solid"),
            zorder=int(style.get(p, {}).get("z", 2)),
        )

        if p in ("PartII", "PartIII"):
            lw = float(style[p]["lw"])
            lc.set_path_effects([pe.Stroke(linewidth=lw + 1.8, foreground="white"), pe.Normal()])

        ax.add_collection(lc)

    # ------------------------------
    # Highlight PartI segments (halo)
    # ------------------------------
    hi_mask = (seg_dom == highlight_part)
    if np.any(hi_mask):
        hi_color = color_map.get(highlight_part, "#000000")
        lc_hi = LineCollection(
            segs[hi_mask],
            colors=[hi_color] * int(np.sum(hi_mask)),
            linewidths=lw_part1_highlight,
            alpha=1.0,
            linestyles="solid",
            zorder=5,
        )
        lc_hi.set_path_effects([pe.Stroke(linewidth=lw_part1_highlight + 2.4, foreground="white"), pe.Normal()])
        ax.add_collection(lc_hi)

        # ------------------------------
        # Sparse hollow triangle markers (boundary + gap constraint)
        # ------------------------------
        hi_points = np.where(dom == highlight_part)[0]
        if hi_points.size > 0:
            # boundaries of contiguous runs in *downsampled* index space
            starts = hi_points[np.r_[True, np.diff(hi_points) > 1]]
            ends = hi_points[np.r_[np.diff(hi_points) > 1, True]]
            boundary = np.unique(np.r_[starts, ends])

            # enforce minimum rank gap
            # (xs is rank, not index; use xs spacing)
            chosen = []
            last_x = -1e18
            for bi in boundary:
                if xs[bi] - last_x >= float(min_marker_gap):
                    chosen.append(bi)
                    last_x = xs[bi]
            boundary = np.array(chosen, dtype=int)

            # cap number of markers
            if boundary.size > max_markers:
                pick = np.unique(np.round(np.linspace(0, boundary.size - 1, max_markers)).astype(int))
                boundary = boundary[pick]

            ax.scatter(
                xs[boundary],
                ys[boundary],
                marker="^",
                s=marker_size,
                facecolors="none",
                edgecolors=hi_color,
                linewidths=1.2,
                alpha=0.98,
                zorder=6,
            )

    # ------------------------------
    # Axes formatting
    # ------------------------------
    ax.set_title(title)
    ax.set_xlabel("Agent rank by popularity")
    ax.set_ylabel("Popularity score")

    if log_x:
        ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")

    ax.set_xlim(float(xs[0]), float(xs[-1]))
    y_pos = ys[np.isfinite(ys) & (ys > 0)]
    y0 = float(np.min(y_pos)) if y_pos.size else 1e-9
    ax.set_ylim(max(1e-9, y0), float(np.max(ys)) * 1.15)

    ax.grid(True, which="major", linestyle=":", linewidth=0.8, alpha=0.25)

    # ------------------------------
    # Legend (match styles)
    # ------------------------------
    handles = []
    if "PartI" in color_map:
        handles.append(
            Line2D(
                [0], [0],
                color=color_map["PartI"],
                lw=lw_part1_highlight,
                marker="^",
                markersize=8,
                markerfacecolor="none",
                markeredgecolor=color_map["PartI"],
                markeredgewidth=1.2,
                label="PartI",
            )
        )
    if "PartII" in color_map:
        handles.append(
            Line2D(
                [0], [0],
                color=color_map["PartII"],
                lw=lw_part23,
                alpha=alpha_part23,
                label="PartII",
                path_effects=[pe.Stroke(linewidth=lw_part23 + 1.8, foreground="white"), pe.Normal()],
            )
        )
    if "PartIII" in color_map:
        handles.append(
            Line2D(
                [0], [0],
                color=color_map["PartIII"],
                lw=lw_part23,
                alpha=alpha_part23,
                linestyle=ls_part3,
                label="PartIII",
                path_effects=[pe.Stroke(linewidth=lw_part23 + 1.8, foreground="white"), pe.Normal()],
            )
        )

    ax.legend(handles=handles, frameon=False, loc="upper right")




# -------------------------
# Main 3x3 plotting
# -------------------------

def plot_regime_shift_popularity(per_part: Sequence[PartPopStats], output_path: Path) -> None:
    _apply_nips_style()

    # keep your colors: Part III is GREEN everywhere
    color_map = {
        "PartI": "#1f77b4",   # blue
        "PartII": "#d62728",  # red
        "PartIII": "#2ca02c", # green
    }

    fig, axes = plt.subplots(
        3, 3, figsize=(16, 13.0), gridspec_kw={"wspace": 0.25, "hspace": 0.45}
    )

    part_lookup = {"/".join(s.parts): s for s in per_part}
    part_i = part_lookup.get("PartI")
    part_ii = part_lookup.get("PartII")
    part_iii = part_lookup.get("PartIII")

    # Row 0: popularity curves (per part)
    if part_i:
        _plot_popularity_curve(
            axes[0, 0], part_i, color_map["PartI"],
            "Part I: Agent popularity curve", log_x=False, log_y=False
        )
    else:
        axes[0, 0].axis("off")

    if part_ii:
        _plot_popularity_curve(
            axes[0, 1], part_ii, color_map["PartII"],
            "Part II: Agent popularity curve", log_x=False, log_y=False
        )
    else:
        axes[0, 1].axis("off")

    if part_iii:
        vals = sorted([v for v in part_iii.popularity_values() if v > 0], reverse=True)
        y_max = (max(vals) * 1.05) if vals else 1.0
        _plot_popularity_curve(
            axes[0, 2], part_iii, color_map["PartIII"],
            "Part III: Agent popularity curve", log_x=False, log_y=False, y_max=float(y_max), max_yticks=4
        )
    else:
        axes[0, 2].axis("off")

    # Row 1: popularity histograms (per part)
    if part_i:
        _plot_popularity_hist(axes[1, 0], part_i, color_map["PartI"])
    else:
        axes[1, 0].axis("off")

    if part_ii:
        _plot_popularity_hist(axes[1, 1], part_ii, color_map["PartII"])
    else:
        axes[1, 1].axis("off")

    if part_iii:
        _plot_popularity_hist(axes[1, 2], part_iii, color_map["PartIII"])
    else:
        axes[1, 2].axis("off")

    # Row 2: merged Pareto + merged segmented curve + CCDF
    _plot_head_coverage_curve_merged(axes[2, 0], per_part)
    _plot_popularity_curve_per_part_style(
        axes[2, 1],
        per_part,
        color_map=color_map,
        title="Agent Popularity Curve",
        log_x=False,
        log_y=False,
        max_points=16,
        head_keep=4,
        draw_merged_black=True,
    )

    _plot_popularity_ccdf(axes[2, 2], per_part, color_map)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)



def _pctl_max_normalize_counter(
    counter: Mapping[str, float],
    *,
    percentile: float = 99.5,
    eps: float = 1e-12,
) -> Tuple[Dict[str, float], float]:
    """
    0-1 标准化（用 p{percentile} 作为 max）：
      x_norm = min(x / pctl_value, 1.0)

    返回：(normalized_dict, pctl_value)
    - pctl_value 作为 max；若 pctl_value 很小则避免除零
    """
    if not counter:
        return {}, 0.0
    vals = np.array(list(counter.values()), dtype=float)
    pmax = float(np.percentile(vals, percentile))
    denom = max(pmax, eps)
    norm = {k: float(min(float(v) / denom, 1.0)) for k, v in counter.items()}
    return norm, pmax


# -------------------------
# Args / main
# -------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Regime shift evidence using content-derived popularity.")
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
    ap.add_argument(
        "--tool_percentile",
        type=float,
        default=99.5,
        help="Upper percentile for winsorizing tool frequencies (default: 99.5).",
    )
    ap.add_argument(
        "--llm_percentile",
        type=float,
        default=99.5,
        help="Upper percentile for winsorizing LLM frequencies for visualization (default: 99.5).",
    )

    return ap.parse_args()

def _filter_min_frequency(freq: Mapping[str, float], min_freq: float = 2.0) -> Dict[str, float]:
    """Keep only entries with frequency >= min_freq."""
    return {k: float(v) for k, v in freq.items() if float(v) >= float(min_freq)}



def main() -> None:
    args = parse_args()
    parts = [p.strip() for p in args.parts.split(",") if p.strip()]
    if not parts:
        raise ValueError("No parts specified for analysis.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    bundle = collect_data(args.data_root, parts=parts)

    # Step 1) collect per-part *degree* first (needed to build llm/tool frequencies)
    # We'll reuse your pos_topk_for_part positivity extraction logic.
    tmp_degree_all: Dict[str, float] = {}
    per_part_degree: Dict[str, Dict[str, float]] = {}

    for part in ["PartI", "PartII", "PartIII"]:
        if part not in parts:
            continue
        degree: Dict[str, float] = {}
        for qid, p in bundle.qid_to_part.items():
            if p != part:
                continue
            ranking = bundle.all_rankings.get(qid, [])
            if not ranking:
                continue
            top_k = pos_topk_for_part(part)
            positives = [aid for aid in ranking[:top_k] if aid in bundle.all_agents]
            for aid in positives:
                degree[aid] = degree.get(aid, 0.0) + 1.0
                tmp_degree_all[aid] = tmp_degree_all.get(aid, 0.0) + 1.0
        per_part_degree[part] = degree

    # Step 2) build GLOBAL llm/tool frequencies from merged degrees (so per-part popularity is comparable)
    llm_freq, tool_freq, clip_threshold = compute_llm_and_tool_frequencies_from_agent_degree(
        tmp_degree_all, bundle.all_agents, tool_percentile=args.tool_percentile
    )
    llm_freq_for_mode_raw = _filter_min_frequency(llm_freq, 10.0)
    
    # ---- NEW: normalize to [0,1] using p99.5 as max (cap > max to 1.0) ----
    llm_freq_norm, llm_pmax = _pctl_max_normalize_counter(llm_freq, percentile=99.5)
    tool_freq_norm, tool_pmax = _pctl_max_normalize_counter(tool_freq, percentile=99.5)

    llm_freq = llm_freq_norm
    tool_freq = tool_freq_norm

    print(f"[Info] LLM norm max uses p99.5={llm_pmax:.4g}")
    print(f"[Info] Tool norm max uses p99.5={tool_pmax:.4g}")

    # mode fill values computed on normalized tables
    mode_llm, mode_tool = compute_mode_fill_values(llm_freq, tool_freq)
    
    # Step 3) compute per-part popularity stats using the SAME llm/tool frequency tables
    per_part_stats: list[PartPopStats] = []
    for part in ["PartI", "PartII", "PartIII"]:
        if part in parts:
            per_part_stats.append(
                collect_part_pop_stats(bundle, [part], llm_freq=llm_freq, tool_freq=tool_freq,mode_llm=mode_llm, mode_tool=mode_tool)
            )

    if not per_part_stats:
        raise ValueError(f"No part statistics collected. Check parts filter: {parts}")

    # Optional: cache merged popularity
    merged = _merge_pop_stats(per_part_stats)
    merged.agent_popularity = compute_agent_popularity(merged.agent_degree.keys(), bundle.all_agents, llm_freq, tool_freq)

    cache_root = Path(ensure_cache_dir(args.data_root, f"popularity_regime_shift_{'_'.join(parts)}"))
    cache_path = cache_popularity(
        cache_root,
        parts=parts,
        llm_frequency=llm_freq,
        tool_frequency=tool_freq,
        tool_clip_threshold=clip_threshold,
        tool_percentile=args.tool_percentile,
        agent_popularity=merged.agent_popularity,
    )

    # Step 4) export metrics + figure
    metrics_path = args.output_dir / "regime_shift_popularity_metrics.csv"
    export_popularity_metrics_table(per_part_stats, metrics_path)

    figure_path = args.output_dir / "regime_shift_popularity.png"
    plot_regime_shift_popularity(per_part=per_part_stats, output_path=figure_path)

    print(f"[OK] Popularity cache saved to: {cache_path}")
    print(f"[OK] Metrics table saved to: {metrics_path}")
    print(f"[OK] Figure saved to: {figure_path}")
    
    freq_fig_path = args.output_dir / "llm_tool_frequency_distribution.png"
    plot_llm_tool_frequency_distributions(
        llm_freq, tool_freq, freq_fig_path,
        llm_percentile=args.llm_percentile,
        llm_min_freq=0.0004
    )

    print(f"[OK] Frequency distribution figure saved to: {freq_fig_path}")



if __name__ == "__main__":
    main()
