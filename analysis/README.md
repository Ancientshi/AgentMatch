## Regime shift analysis

This folder contains a lightweight script to quantify the dense → one-off shift between benchmark parts.

### Usage

```bash
python analysis/regime_shift_analysis.py \
  --data_root /path/to/dataset_root \
  --output_dir analysis/artifacts
```

Requirements:

- The dataset root should contain `PartI`, `PartII`, and `PartIII` subfolders with `agents/merge.json`, `questions/merge.json`, and `rankings/merge.json`.
- `matplotlib` must be available in the environment (preinstalled in the scaffold containers).

Outputs (written to `--output_dir`):

- `regime_shift_metrics.csv`: one table of per-part topology metrics (unique agent ratio, head coverage for top 1%/5%, entropy and effective directory size).
- `regime_shift_frequency.png`: NIPS-friendly visualization with three separate frequency curves (Part I, Part II, Part III) and a log-scale agent-degree histogram that caps tails at the 99.5th percentile to keep Parts II/III readable.

The defaults cover Parts I/II/III, but you can limit analysis to a subset with `--parts PartI,PartII`.

## Popularity analysis (LLM/Tool frequency + agent popularity)

`analysis/popularity_analysis.py` computes LLM frequencies (using `M.id` or falling back to `M.name`), winsorized tool frequencies (99.5th percentile), caches them under `${data_root}/.cache/popularity_analysis_*`, derives per-agent popularity (LLM frequency + mean tool frequency), and plots the popularity curve.

### Usage

```bash
python analysis/popularity_analysis.py \
  --data_root /path/to/dataset_root \
  --output_dir analysis/artifacts \
  --tool_percentile 99.5
```

Outputs:

- `${data_root}/.cache/popularity_analysis_*/popularity_stats.json`: cached LLM frequency, winsorized tool frequency, agent popularity, and the clipping threshold.
- `agent_popularity_curve.png`: log-log popularity curve saved to `--output_dir`.
