# -*- coding: utf-8 -*-

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Sequence
import numpy as np
import torch


class RecommenderBase(ABC, torch.nn.Module):
    @abstractmethod
    def export_agent_embeddings(self) -> np.ndarray:
        """Return agent embedding matrix A: (Na, F) as numpy float32."""
        raise NotImplementedError

    @abstractmethod
    def export_query_embeddings(self, q_indices: Sequence[int]) -> np.ndarray:
        """Return query embeddings for given q_indices: (Nq, F) as numpy float32."""
        raise NotImplementedError

    @abstractmethod
    def export_agent_bias(self) -> Optional[np.ndarray]:
        """Return agent bias: (Na,) float32 or None."""
        raise NotImplementedError

    @abstractmethod
    def extra_state_dict(self) -> Dict[str, Any]:
        """Any extra metadata to checkpoint."""
        raise NotImplementedError
