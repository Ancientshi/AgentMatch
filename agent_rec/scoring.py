#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
from typing import Optional, Literal
import numpy as np

from .config import EPS


ScoreMode = Literal["dot", "cosine"]


def score_candidates(
    A: np.ndarray,              # (Na,F)
    qv: np.ndarray,             # (F,)
    ai_idx: np.ndarray,         # (Ncand,)
    bias_a: Optional[np.ndarray] = None,  # (Na,)
    mode: ScoreMode = "dot",
) -> np.ndarray:
    """
    Score a subset of agents.
    """
    if mode == "dot":
        s = A[ai_idx] @ qv.astype(np.float32)
    elif mode == "cosine":
        A2 = A[ai_idx] / (np.linalg.norm(A[ai_idx], axis=1, keepdims=True) + EPS)
        q2 = qv / (np.linalg.norm(qv) + EPS)
        s = A2 @ q2.astype(np.float32)
    else:
        raise ValueError(f"Unknown score mode: {mode}")

    if bias_a is not None:
        s = s + bias_a[ai_idx]
    return s.astype(np.float32)
