#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def bpr_loss(pos: torch.Tensor, neg: torch.Tensor) -> torch.Tensor:
    """Bayesian Personalized Ranking loss."""
    return -torch.log(torch.sigmoid(pos - neg) + 1e-8).mean()


class SimpleBPRDNN(nn.Module):
    """
    Pairwise ranking DNN (BPR) for Query->Agent.

    Key tweaks for TF-IDF robustness:
      - L2 normalize projected representations (qh, ah)
      - Add explicit interaction features: qh*ah and |qh-ah|
      - Fix in_dim accordingly (4*text_hidden instead of 2*text_hidden)
    """

    def __init__(
        self,
        d_q: int,
        d_a: int,
        num_tools: int,
        num_llm_ids: int,
        agent_tool_indices_padded: torch.LongTensor,
        agent_tool_mask: torch.FloatTensor,
        agent_llm_idx: torch.LongTensor,
        text_hidden: int = 256,
        id_dim: int = 64,
        num_queries: int = 0,
        use_query_id_emb: bool = False,
        use_tool_id_emb: bool = True,
        use_llm_id_emb: bool = True,
        use_layernorm: bool = False,   
        dropout: float = 0.0,          
    ):
        super().__init__()

        self.text_hidden = int(text_hidden)
        self.id_dim = int(id_dim)
        self.use_layernorm = bool(use_layernorm)
        self.dropout_p = float(dropout)

        # Project raw features -> hidden
        self.q_proj = nn.Linear(d_q, self.text_hidden)
        self.a_proj = nn.Linear(d_a, self.text_hidden)

        self.q_ln = nn.LayerNorm(self.text_hidden) if self.use_layernorm else None
        self.a_ln = nn.LayerNorm(self.text_hidden) if self.use_layernorm else None
        self.drop = nn.Dropout(self.dropout_p) if self.dropout_p > 0 else None

        # Optional ID embeddings
        self.use_llm_id_emb = bool(use_llm_id_emb) and num_llm_ids > 0
        self.use_tool_id_emb = bool(use_tool_id_emb) and num_tools > 0
        self.use_query_id_emb = bool(use_query_id_emb) and num_queries > 0

        self.emb_llm = nn.Embedding(num_llm_ids, self.id_dim) if self.use_llm_id_emb else None
        self.emb_tool = nn.Embedding(num_tools, self.id_dim) if self.use_tool_id_emb else None
        self.emb_query = nn.Embedding(num_queries, self.id_dim) if self.use_query_id_emb else None

        # Agent-side metadata buffers
        self.register_buffer("agent_tool_indices_padded", agent_tool_indices_padded)  # (A, max_tools)
        self.register_buffer("agent_tool_mask", agent_tool_mask)                      # (A, max_tools) float {0,1}
        self.register_buffer("agent_llm_idx", agent_llm_idx)                          # (A,)

        # ---- IMPORTANT: interaction features add 2*text_hidden more
        # parts = [qh, ah, qh*ah, |qh-ah|] => 4*text_hidden
        in_dim = 4 * self.text_hidden
        if self.use_llm_id_emb:
            in_dim += self.id_dim
        if self.use_tool_id_emb:
            in_dim += self.id_dim
        if self.use_query_id_emb:
            in_dim += self.id_dim

        self.scorer = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

        # Init
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.zeros_(self.q_proj.bias)
        nn.init.xavier_uniform_(self.a_proj.weight)
        nn.init.zeros_(self.a_proj.bias)

        if self.emb_llm is not None:
            nn.init.xavier_uniform_(self.emb_llm.weight)
        if self.emb_tool is not None:
            nn.init.xavier_uniform_(self.emb_tool.weight)
        if self.emb_query is not None:
            nn.init.xavier_uniform_(self.emb_query.weight)

        for m in self.scorer:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def _agent_tool_mean_emb(self, agent_idx: torch.LongTensor) -> torch.Tensor:
        """
        Return mean tool-id embedding for each agent in agent_idx.
        """
        idxs = self.agent_tool_indices_padded[agent_idx]  # (B, max_tools)
        mask = self.agent_tool_mask[agent_idx]            # (B, max_tools)
        te = self.emb_tool(idxs)                          # (B, max_tools, id_dim)
        mask3 = mask.unsqueeze(-1)                        # (B, max_tools, 1)
        te_sum = (te * mask3).sum(dim=1)                  # (B, id_dim)
        denom = mask.sum(dim=1, keepdim=True) + 1e-8      # (B, 1)
        return te_sum / denom

    def forward_score(
        self,
        q_vec: torch.Tensor,
        a_vec: torch.Tensor,
        agent_idx: torch.LongTensor,
        q_idx: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        """
        q_vec: (B, d_q)
        a_vec: (B, d_a)
        agent_idx: (B,)
        q_idx: (B,) optional, required if use_query_id_emb=True
        return: (B,) scores
        """
        # Project + nonlinearity
        qh = F.relu(self.q_proj(q_vec))
        ah = F.relu(self.a_proj(a_vec))

        # Optional LN + Dropout
        if self.q_ln is not None:
            qh = self.q_ln(qh)
        if self.a_ln is not None:
            ah = self.a_ln(ah)
        if self.drop is not None:
            qh = self.drop(qh)
            ah = self.drop(ah)

        # Normalize for scale stability (important for TF-IDF)
        qh = F.normalize(qh, dim=1)
        ah = F.normalize(ah, dim=1)

        # Explicit interactions (key)
        parts = [qh, ah, qh * ah, torch.abs(qh - ah)]

        # Agent-side ID features
        if self.use_llm_id_emb:
            parts.append(self.emb_llm(self.agent_llm_idx[agent_idx]))

        if self.use_tool_id_emb:
            parts.append(self._agent_tool_mean_emb(agent_idx))

        # Query ID (optional)
        if self.use_query_id_emb:
            if q_idx is None:
                raise ValueError("q_idx is required when use_query_id_emb=True")
            parts.append(self.emb_query(q_idx))

        x = torch.cat(parts, dim=1)          # (B, in_dim)
        s = self.scorer(x).squeeze(1)        # (B,)
        return s

    def forward(
        self,
        q_vec: torch.Tensor,
        pos_vec: torch.Tensor,
        neg_vec: torch.Tensor,
        pos_idx: torch.LongTensor,
        neg_idx: torch.LongTensor,
        q_idx: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return (pos_scores, neg_scores), each shape (B,).
        """
        pos = self.forward_score(q_vec, pos_vec, pos_idx, q_idx=q_idx)
        neg = self.forward_score(q_vec, neg_vec, neg_idx, q_idx=q_idx)
        return pos, neg
