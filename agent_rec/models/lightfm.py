#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
from typing import Dict, Any, Optional, Sequence, Tuple, List

import numpy as np
import torch
import torch.nn as nn

try:
    from scipy import sparse as sp
except Exception:  # pragma: no cover - import error handled by runtime
    sp = None

from .base import RecommenderBase


def bpr_loss(pos: torch.Tensor, neg: torch.Tensor) -> torch.Tensor:
    return -torch.log(torch.sigmoid(pos - neg) + 1e-8).mean()


def csr_to_bag_lists(csr: "sp.csr_matrix") -> Tuple[List[List[int]], List[List[float]], int]:
    indptr, indices, data = csr.indptr, csr.indices, csr.data
    rows = csr.shape[0]
    feats_per_row: List[List[int]] = []
    vals_per_row: List[List[float]] = []
    for r in range(rows):
        s, e = indptr[r], indptr[r + 1]
        feats_per_row.append(indices[s:e].tolist())
        vals_per_row.append(data[s:e].astype(np.float32).tolist())
    return feats_per_row, vals_per_row, csr.shape[1]


def build_bag_tensors(
    batch_rows: Sequence[int],
    feats_per_row: List[List[int]],
    vals_per_row: List[List[float]],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    idx_cat: List[int] = []
    w_cat: List[float] = []
    offsets = [0]
    total = 0
    for r in batch_rows:
        feats = feats_per_row[r]
        vals = vals_per_row[r]
        idx_cat.extend(feats)
        w_cat.extend(vals)
        total += len(feats)
        offsets.append(total)

    if total == 0:
        idx_cat = [0]
        w_cat = [0.0]
        offsets = [0, 1]

    idx_t = torch.tensor(idx_cat, dtype=torch.long, device=device)
    off_t = torch.tensor(offsets, dtype=torch.long, device=device)
    w_t = torch.tensor(w_cat, dtype=torch.float32, device=device)
    return idx_t, off_t, w_t


class LightFM(RecommenderBase):
    """Hybrid MF with ID embeddings + sparse user/item feature embeddings."""

    def __init__(
        self,
        num_q: int,
        num_a: int,
        num_llm_ids: int,
        num_user_feats: int,
        num_item_feats: int,
        num_tool_ids: int = 0,
        factors: int = 128,
        add_bias: bool = True,
        alpha_id: float = 1.0,
        alpha_feat: float = 1.0,
        alpha_tool: float = 1.0,
        device: torch.device = torch.device("cpu"),
        agent_llm_idx: Optional[torch.LongTensor] = None,
        use_llm_id_emb: bool = True,
    ):
        super().__init__()
        self.num_a = int(num_a)
        self.add_bias = add_bias
        self.alpha_id = nn.Parameter(torch.tensor(alpha_id, dtype=torch.float32))
        self.alpha_feat = nn.Parameter(torch.tensor(alpha_feat, dtype=torch.float32))
        self.alpha_tool = nn.Parameter(torch.tensor(alpha_tool, dtype=torch.float32))
        self.use_llm_id_emb = bool(use_llm_id_emb) and num_llm_ids > 0
        self.use_tool_id_emb = num_tool_ids > 0

        self.emb_q = nn.Embedding(num_q, factors)
        self.emb_llm = nn.Embedding(num_llm_ids, factors) if self.use_llm_id_emb else None
        self.emb_user_feat = nn.EmbeddingBag(num_user_feats, factors, mode="sum", include_last_offset=True)
        self.emb_item_feat = nn.EmbeddingBag(num_item_feats, factors, mode="sum", include_last_offset=True)
        if self.use_tool_id_emb:
            self.emb_tool = nn.Embedding(num_tool_ids, factors)

        if add_bias:
            self.bias_q = nn.Embedding(num_q, 1)
            self.bias_a = nn.Embedding(num_a, 1)

        self.reset_parameters()
        self.device = device

        self.user_feats_per_row: Optional[List[List[int]]] = None
        self.user_vals_per_row: Optional[List[List[float]]] = None
        self.item_feats_per_row: Optional[List[List[int]]] = None
        self.item_vals_per_row: Optional[List[List[float]]] = None
        if agent_llm_idx is not None:
            self.register_buffer("agent_llm_idx", agent_llm_idx)
        else:
            self.agent_llm_idx = None
        self.item_tool_ids: Optional[torch.LongTensor] = None
        self.item_tool_mask: Optional[torch.FloatTensor] = None

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.emb_q.weight)
        if self.emb_llm is not None:
            nn.init.xavier_uniform_(self.emb_llm.weight)
        nn.init.xavier_uniform_(self.emb_user_feat.weight)
        nn.init.xavier_uniform_(self.emb_item_feat.weight)
        if self.use_tool_id_emb:
            nn.init.xavier_uniform_(self.emb_tool.weight)
        if self.add_bias:
            nn.init.zeros_(self.bias_q.weight)
            nn.init.zeros_(self.bias_a.weight)

    def set_user_feat_lists(self, feats_per_row: List[List[int]], vals_per_row: List[List[float]]) -> None:
        self.user_feats_per_row = feats_per_row
        self.user_vals_per_row = vals_per_row

    def set_item_feat_lists(self, feats_per_row: List[List[int]], vals_per_row: List[List[float]]) -> None:
        self.item_feats_per_row = feats_per_row
        self.item_vals_per_row = vals_per_row

    def set_item_tool_id_buffers(self, tool_ids: torch.LongTensor, tool_mask: torch.FloatTensor) -> None:
        self.item_tool_ids = tool_ids
        self.item_tool_mask = tool_mask

    def _mean_embed_tools(self, a_idx: torch.LongTensor) -> torch.Tensor:
        if not self.use_tool_id_emb:
            raise RuntimeError("Tool ID embedding not enabled.")
        if self.item_tool_ids is None or self.item_tool_mask is None:
            raise RuntimeError("Item tool buffers not set.")
        tool_ids = self.item_tool_ids[a_idx]
        tool_mask = self.item_tool_mask[a_idx]
        tool_emb = self.emb_tool(tool_ids)
        tool_mask = tool_mask.unsqueeze(-1)
        weighted = tool_emb * tool_mask
        denom = tool_mask.sum(dim=1).clamp_min(1.0)
        return weighted.sum(dim=1) / denom

    def _bag_embed_users(self, q_idx: torch.LongTensor) -> torch.Tensor:
        if self.user_feats_per_row is None or self.user_vals_per_row is None:
            raise RuntimeError("User feature lists not set.")
        idx_t, off_t, w_t = build_bag_tensors(
            batch_rows=q_idx.tolist(),
            feats_per_row=self.user_feats_per_row,
            vals_per_row=self.user_vals_per_row,
            device=self.device,
        )
        return self.emb_user_feat(idx_t, off_t, per_sample_weights=w_t)

    def _bag_embed_items(self, a_idx: torch.LongTensor) -> torch.Tensor:
        if self.item_feats_per_row is None or self.item_vals_per_row is None:
            raise RuntimeError("Item feature lists not set.")
        idx_t, off_t, w_t = build_bag_tensors(
            batch_rows=a_idx.tolist(),
            feats_per_row=self.item_feats_per_row,
            vals_per_row=self.item_vals_per_row,
            device=self.device,
        )
        return self.emb_item_feat(idx_t, off_t, per_sample_weights=w_t)

    def user_repr_batch(self, q_idx: torch.LongTensor) -> torch.Tensor:
        u_id = self.emb_q(q_idx)
        u_feat = self._bag_embed_users(q_idx)
        return self.alpha_id * u_id + self.alpha_feat * u_feat

    def item_repr_batch(self, a_idx: torch.LongTensor) -> torch.Tensor:
        if self.use_llm_id_emb:
            if self.agent_llm_idx is None:
                raise RuntimeError("agent_llm_idx buffer not set.")
            i_id = self.emb_llm(self.agent_llm_idx[a_idx])
        else:
            i_id = torch.zeros((a_idx.size(0), self.emb_user_feat.embedding_dim), device=self.device)
        i_feat = self._bag_embed_items(a_idx)
        out = self.alpha_id * i_id + self.alpha_feat * i_feat
        if self.use_tool_id_emb:
            out = out + self.alpha_tool * self._mean_embed_tools(a_idx)
        return out

    def forward(
        self,
        q_idx: torch.LongTensor,
        pos_idx: torch.LongTensor,
        neg_idx: torch.LongTensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        qv = self.user_repr_batch(q_idx)
        apv = self.item_repr_batch(pos_idx)
        anv = self.item_repr_batch(neg_idx)

        pos = (qv * apv).sum(dim=-1)
        neg = (qv * anv).sum(dim=-1)

        if self.add_bias:
            pos = pos + self.bias_q(q_idx).squeeze(-1) + self.bias_a(pos_idx).squeeze(-1)
            neg = neg + self.bias_q(q_idx).squeeze(-1) + self.bias_a(neg_idx).squeeze(-1)
        return pos, neg

    def export_agent_embeddings(self, batch_size: Optional[int] = 4096) -> np.ndarray:
        with torch.no_grad():
            num_items = self.num_a
            if batch_size is None or batch_size >= num_items:
                a_idx = torch.arange(num_items, device=self.device, dtype=torch.long)
                return self.item_repr_batch(a_idx).detach().cpu().numpy().astype(np.float32)

            chunks: List[np.ndarray] = []
            for start in range(0, num_items, batch_size):
                end = min(start + batch_size, num_items)
                a_idx = torch.arange(start, end, device=self.device, dtype=torch.long)
                chunk = self.item_repr_batch(a_idx).detach().cpu().numpy().astype(np.float32)
                chunks.append(chunk)
            return np.concatenate(chunks, axis=0)

    def export_query_embeddings(self, q_indices: Sequence[int]) -> np.ndarray:
        with torch.no_grad():
            q_idx = torch.tensor(list(q_indices), dtype=torch.long, device=self.device)
            return self.user_repr_batch(q_idx).detach().cpu().numpy().astype(np.float32)

    def export_agent_bias(self) -> Optional[np.ndarray]:
        if hasattr(self, "bias_a"):
            return self.bias_a.weight.detach().cpu().numpy().squeeze(-1).astype(np.float32)
        return None

    def extra_state_dict(self) -> Dict[str, Any]:
        return {
            "add_bias": bool(self.add_bias),
            "use_tool_id_emb": bool(self.use_tool_id_emb),
            "use_llm_id_emb": bool(self.use_llm_id_emb),
        }
