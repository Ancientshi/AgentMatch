# AgentMatch

AgentMatch is the first benchmark designed to evaluate the recommendation of suitable agents for a given query.This repository provides datasets, models, and training scripts used in our paper.
Future updates will include cleaner abstractions, standardized benchmarks, and extended evaluation modules.

---

## 📚 Dataset Structure

The datasets described in the paper are organized into three main parts:
`PartI/`, `PartII/`, and `PartIII/`.
Each part contains three subfolders:

```
agents/      # All constructed agent definitions
questions/   # All dialogue and query prompts
rankings/    # Interaction rankings between questions and agents
```

The tools used by agents are stored separately in the `Tools/` directory.

---

## 🧠 Recommendation Models

Each recommendation model is implemented as a standalone Python script, following a consistent interface for training and evaluation.
For each model, a corresponding shell script is provided:

```
xxx.py      # Model definition and training logic
run_xxx.sh        # Shell script for training and validation
```

These scripts support reproducible experimentation under standardized configurations.

---

## 🚧 Roadmap

This repository is under active development.

---

## 📄 Citation

If you find this repository useful, please cite our accompanying paper once released.

---
