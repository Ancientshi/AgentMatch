#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
from typing import Tuple, Optional, Sequence, Dict, Any
import numpy as np
import torch
import torch.nn as nn

from .base import RecommenderBase


def bpr_loss(pos: torch.Tensor, neg: torch.Tensor) -> torch.Tensor:
    return -torch.log(torch.sigmoid(pos - neg) + 1e-8).mean()


class BPRMF(RecommenderBase):
    def __init__(
        self,
        num_q: int,
        num_a: int,
        num_llm_ids: int,
        agent_llm_idx: torch.LongTensor,
        factors: int = 128,
        add_bias: bool = True,
        use_llm_id_emb: bool = True,
    ):
        super().__init__()
        self.use_llm_id_emb = bool(use_llm_id_emb) and num_llm_ids > 0
        self.emb_q = nn.Embedding(num_q, factors)
        self.emb_llm = nn.Embedding(num_llm_ids, factors) if self.use_llm_id_emb else None
        self.add_bias = add_bias
        if add_bias:
            self.bias_q = nn.Embedding(num_q, 1)
            self.bias_a = nn.Embedding(num_a, 1)
        self.register_buffer("agent_llm_idx", agent_llm_idx)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.emb_q.weight)
        if self.emb_llm is not None:
            nn.init.xavier_uniform_(self.emb_llm.weight)
        if self.add_bias:
            nn.init.zeros_(self.bias_q.weight)
            nn.init.zeros_(self.bias_a.weight)

    def score_embeddings(
        self,
        q_vec: torch.Tensor,
        a_vec: torch.Tensor,
        qi: Optional[torch.Tensor] = None,
        ai: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        s = (q_vec * a_vec).sum(dim=-1)
        if self.add_bias and qi is not None and ai is not None:
            s = s + self.bias_q(qi).squeeze(-1) + self.bias_a(ai).squeeze(-1)
        return s

    def forward(
        self,
        q_idx: torch.LongTensor,
        pos_idx: torch.LongTensor,
        neg_idx: torch.LongTensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # ensure long + same device
        q_idx = q_idx.long()
        pos_idx = pos_idx.long()
        neg_idx = neg_idx.long()

        qv = self.emb_q(q_idx)

        if not self.use_llm_id_emb:
            zero = torch.zeros((q_idx.size(0), self.emb_q.embedding_dim), device=q_idx.device, dtype=qv.dtype)
            apv = zero
            anv = zero
        else:
            # agent -> llm id index
            pos_llm_idx = self.agent_llm_idx[pos_idx].long()
            neg_llm_idx = self.agent_llm_idx[neg_idx].long()

            # FIX: Embedding is called, not subscripted
            apv = self.emb_llm(pos_llm_idx)
            anv = self.emb_llm(neg_llm_idx)

        pos = self.score_embeddings(qv, apv, q_idx, pos_idx)
        neg = self.score_embeddings(qv, anv, q_idx, neg_idx)
        return pos, neg

    # ---- RecommenderBase ----
    def export_agent_embeddings(self) -> np.ndarray:
        if not self.use_llm_id_emb:
            return np.zeros((self.agent_llm_idx.numel(), self.emb_q.embedding_dim), dtype=np.float32)
        embs = self.emb_llm.weight.detach().cpu().numpy().astype(np.float32)
        return embs[self.agent_llm_idx.cpu().numpy()]

    def export_query_embeddings(self, q_indices: Sequence[int]) -> np.ndarray:
        w = self.emb_q.weight.detach().cpu().numpy().astype(np.float32)
        return w[np.array(list(q_indices), dtype=np.int64)]

    def export_agent_bias(self) -> Optional[np.ndarray]:
        if hasattr(self, "bias_a"):
            return self.bias_a.weight.detach().cpu().numpy().squeeze(-1).astype(np.float32)
        return None

    def extra_state_dict(self) -> Dict[str, Any]:
        return {"add_bias": bool(self.add_bias)}
