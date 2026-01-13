#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import RecommenderBase


def bpr_loss(pos: torch.Tensor, neg: torch.Tensor) -> torch.Tensor:
    return -torch.log(torch.sigmoid(pos - neg) + 1e-8).mean()


def _normalized_adj(
    num_q: int,
    num_a: int,
    interactions: Sequence[Tuple[int, int]],
    device: torch.device,
) -> torch.Tensor:
    """
    Build the symmetric normalized adjacency matrix for a bipartite graph.
    Nodes are ordered as [queries, agents].
    """
    n_nodes = num_q + num_a
    if n_nodes == 0:
        raise ValueError("Cannot build adjacency with zero nodes.")

    if not interactions:
        indices = torch.empty((2, 0), dtype=torch.long, device=device)
        values = torch.empty((0,), dtype=torch.float32, device=device)
        return torch.sparse_coo_tensor(indices, values, (n_nodes, n_nodes), device=device)

    rows = []
    cols = []
    for qi, ai in interactions:
        rows.extend([qi, num_q + ai])
        cols.extend([num_q + ai, qi])
    indices = torch.tensor([rows, cols], dtype=torch.long, device=device)
    data = torch.ones(len(rows), dtype=torch.float32, device=device)
    adj = torch.sparse_coo_tensor(indices, data, (n_nodes, n_nodes), device=device)
    deg = torch.sparse.sum(adj, dim=1).to_dense()
    deg_inv_sqrt = torch.pow(deg + 1e-8, -0.5)
    norm_values = data * deg_inv_sqrt[indices[0]] * deg_inv_sqrt[indices[1]]
    norm_adj = torch.sparse_coo_tensor(indices, norm_values, (n_nodes, n_nodes), device=device)
    return norm_adj.coalesce()


class GraphRecommenderBase(RecommenderBase):
    def __init__(
        self,
        num_q: int,
        num_a: int,
        embed_dim: int,
        interactions: Sequence[Tuple[int, int]],
        *,
        agent_content: Optional[torch.Tensor] = None,
        agent_tool_indices_padded: Optional[torch.LongTensor] = None,
        agent_tool_mask: Optional[torch.FloatTensor] = None,
        agent_llm_idx: Optional[torch.LongTensor] = None,
        num_tools: int = 0,
        num_llm_ids: int = 0,
        use_query_id_emb: bool = True,
        use_tool_id_emb: bool = True,
        use_llm_id_emb: bool = True,
        use_agent_content: bool = True,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.num_q = int(num_q)
        self.num_a = int(num_a)
        self.embed_dim = int(embed_dim)
        self.use_query_id_emb = bool(use_query_id_emb)
        self.use_tool_id_emb = bool(use_tool_id_emb) and num_tools > 0
        self.use_llm_id_emb = bool(use_llm_id_emb) and num_llm_ids > 0
        self.use_agent_content = bool(use_agent_content) and agent_content is not None

        self.emb_q = nn.Embedding(num_q, embed_dim)
        self.emb_a = nn.Embedding(num_a, embed_dim)

        if self.use_agent_content:
            self.content_proj = nn.Linear(agent_content.size(1), embed_dim)
            self.register_buffer("agent_content", agent_content.float())
        else:
            self.content_proj = None
            self.register_buffer("agent_content", torch.zeros((num_a, 0), dtype=torch.float32))

        if self.use_tool_id_emb:
            self.emb_tool = nn.Embedding(num_tools, embed_dim)
            self.register_buffer(
                "agent_tool_indices",
                (agent_tool_indices_padded if agent_tool_indices_padded is not None else torch.zeros((num_a, 1), dtype=torch.long)),
            )
            self.register_buffer(
                "agent_tool_mask",
                (agent_tool_mask if agent_tool_mask is not None else torch.zeros((num_a, 1), dtype=torch.float32)),
            )
        else:
            self.emb_tool = None
            self.register_buffer("agent_tool_indices", torch.zeros((num_a, 1), dtype=torch.long))
            self.register_buffer("agent_tool_mask", torch.zeros((num_a, 1), dtype=torch.float32))

        if self.use_llm_id_emb:
            self.emb_llm = nn.Embedding(num_llm_ids, embed_dim)
            self.register_buffer(
                "agent_llm_idx",
                agent_llm_idx if agent_llm_idx is not None else torch.zeros((num_a,), dtype=torch.long),
            )
        else:
            self.emb_llm = None
            self.register_buffer("agent_llm_idx", torch.zeros((num_a,), dtype=torch.long))

        self.register_buffer("norm_adj", _normalized_adj(num_q, num_a, interactions, device=device or torch.device("cpu")))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.emb_q.weight)
        nn.init.xavier_uniform_(self.emb_a.weight)
        if self.emb_tool is not None:
            nn.init.xavier_uniform_(self.emb_tool.weight)
        if self.emb_llm is not None:
            nn.init.xavier_uniform_(self.emb_llm.weight)
        if self.content_proj is not None:
            nn.init.xavier_uniform_(self.content_proj.weight)
            nn.init.zeros_(self.content_proj.bias)

    def _initial_embeddings(self) -> Tuple[torch.Tensor, torch.Tensor]:
        q0 = self.emb_q.weight
        a0 = self.emb_a.weight
        if not self.use_query_id_emb:
            q0 = torch.zeros_like(q0)
        if self.use_agent_content and self.content_proj is not None:
            a0 = a0 + self.content_proj(self.agent_content)
        if self.use_tool_id_emb and self.emb_tool is not None:
            mask = self.agent_tool_mask
            counts = mask.sum(dim=1).to(torch.long)  # (Na,)
            valid_idx = mask.bool()
            flat_idx = self.agent_tool_indices[valid_idx]  # (N_valid,)

            if flat_idx.numel() > 0:
                offsets = torch.cat(
                    [
                        torch.zeros(1, device=counts.device, dtype=torch.long),
                        counts.cumsum(0)[:-1],
                    ]
                )
                tool_mean = F.embedding_bag(
                    flat_idx,
                    self.emb_tool.weight,
                    offsets,
                    mode="mean",
                )
            else:
                tool_mean = torch.zeros_like(a0)

            a0 = a0 + tool_mean
            
        if self.use_llm_id_emb and self.emb_llm is not None:
            a0 = a0 + self.emb_llm(self.agent_llm_idx)
        return q0, a0

    def _final_embeddings(self) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def forward(
        self,
        q_idx: torch.LongTensor,
        pos_idx: torch.LongTensor,
        neg_idx: torch.LongTensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        q_emb, a_emb = self._final_embeddings()
        qv = q_emb[q_idx.long()]
        pos_v = a_emb[pos_idx.long()]
        neg_v = a_emb[neg_idx.long()]
        pos = (qv * pos_v).sum(dim=-1)
        neg = (qv * neg_v).sum(dim=-1)
        return pos, neg

    # --- RecommenderBase ---
    def export_agent_embeddings(self) -> np.ndarray:
        with torch.no_grad():
            _, a_emb = self._final_embeddings()
            return a_emb.detach().cpu().numpy().astype(np.float32)

    def export_query_embeddings(self, q_indices: Sequence[int]) -> np.ndarray:
        with torch.no_grad():
            q_emb, _ = self._final_embeddings()
            q_indices = list(q_indices)
            return q_emb[q_indices].detach().cpu().numpy().astype(np.float32)

    def export_agent_bias(self) -> Optional[np.ndarray]:
        return None

    def extra_state_dict(self) -> Dict[str, object]:
        return {
            "embed_dim": self.embed_dim,
            "use_query_id_emb": self.use_query_id_emb,
            "use_tool_id_emb": self.use_tool_id_emb,
            "use_llm_id_emb": self.use_llm_id_emb,
            "use_agent_content": self.use_agent_content,
        }


class LightGCNRecommender(GraphRecommenderBase):
    def __init__(
        self,
        *args,
        num_layers: int = 2,
        **kwargs,
    ) -> None:
        self.num_layers = int(num_layers)
        super().__init__(*args, **kwargs)

    def _propagate(self, all_emb: torch.Tensor) -> torch.Tensor:
        embs = [all_emb]
        h = all_emb
        for _ in range(self.num_layers):
            h = torch.sparse.mm(self.norm_adj, h)
            embs.append(h)
        return torch.stack(embs, dim=0).mean(dim=0)

    def _final_embeddings(self) -> Tuple[torch.Tensor, torch.Tensor]:
        q0, a0 = self._initial_embeddings()
        h0 = torch.cat([q0, a0], dim=0)
        h = self._propagate(h0)
        return h[: self.num_q], h[self.num_q :]

    def extra_state_dict(self) -> Dict[str, object]:
        base = super().extra_state_dict()
        base.update({"num_layers": self.num_layers})
        return base


class NGCFRecommender(GraphRecommenderBase):
    def __init__(
        self,
        *args,
        num_layers: int = 2,
        dropout: float = 0.1,
        act: str = "leakyrelu",
        **kwargs,
    ) -> None:
        self.num_layers = int(num_layers)
        self.dropout = float(dropout)
        self.act = act
        super().__init__(*args, **kwargs)
        self.W1 = nn.ModuleList([nn.Linear(self.embed_dim, self.embed_dim) for _ in range(self.num_layers)])
        self.W2 = nn.ModuleList([nn.Linear(self.embed_dim, self.embed_dim) for _ in range(self.num_layers)])
        self.activation = nn.LeakyReLU() if act == "leakyrelu" else nn.ReLU()
        for w1, w2 in zip(self.W1, self.W2):
            nn.init.xavier_uniform_(w1.weight)
            nn.init.zeros_(w1.bias)
            nn.init.xavier_uniform_(w2.weight)
            nn.init.zeros_(w2.bias)

    def _final_embeddings(self) -> Tuple[torch.Tensor, torch.Tensor]:
        q0, a0 = self._initial_embeddings()
        h = torch.cat([q0, a0], dim=0)
        all_layers = [h]
        for i in range(self.num_layers):
            neigh = torch.sparse.mm(self.norm_adj, h)
            bi = torch.sparse.mm(self.norm_adj, h * all_layers[0])
            h = self.activation(self.W1[i](neigh) + self.W2[i](bi))
            h = F.dropout(h, p=self.dropout, training=self.training)
            all_layers.append(h)
        out = torch.stack(all_layers, dim=0).mean(dim=0)
        return out[: self.num_q], out[self.num_q :]

    def extra_state_dict(self) -> Dict[str, object]:
        base = super().extra_state_dict()
        base.update({"num_layers": self.num_layers, "dropout": self.dropout, "act": self.act})
        return base


class KGATRecommender(GraphRecommenderBase):
    def __init__(
        self,
        *args,
        num_layers: int = 2,
        att_dropout: float = 0.1,
        **kwargs,
    ) -> None:
        self.num_layers = int(num_layers)
        self.att_dropout = float(att_dropout)
        super().__init__(*args, **kwargs)
        self.att_weight = nn.Parameter(torch.randn(self.embed_dim))
        nn.init.xavier_uniform_(self.att_weight.view(1, -1))

        # Precompute edge indices for attention
        self.register_buffer("edge_index", self.norm_adj.indices())

    def _agg_attention(self, h: torch.Tensor) -> torch.Tensor:
        src = self.edge_index[0]
        dst = self.edge_index[1]
        h_src = h[src]
        h_dst = h[dst]
        score = (h_src * h_dst).mul(self.att_weight).sum(dim=-1)
        att = torch.sigmoid(score)
        att = F.dropout(att, p=self.att_dropout, training=self.training)

        msg = h_dst * att.unsqueeze(-1)
        out = torch.zeros_like(h)
        out.index_add_(0, src, msg)

        denom = torch.zeros((h.size(0),), device=h.device)
        denom.index_add_(0, src, att + 1e-8)
        out = out / (denom.unsqueeze(-1) + 1e-8)
        return out

    def _final_embeddings(self) -> Tuple[torch.Tensor, torch.Tensor]:
        q0, a0 = self._initial_embeddings()
        h = torch.cat([q0, a0], dim=0)
        embs = [h]
        for _ in range(self.num_layers):
            h = self._agg_attention(h)
            embs.append(h)
        out = torch.stack(embs, dim=0).mean(dim=0)
        return out[: self.num_q], out[self.num_q :]

    def extra_state_dict(self) -> Dict[str, object]:
        base = super().extra_state_dict()
        base.update({"num_layers": self.num_layers, "att_dropout": self.att_dropout})
        return base


class SimGCLRecommender(LightGCNRecommender):
    def __init__(
        self,
        *args,
        cl_weight: float = 0.1,
        perturb_eps: float = 0.1,
        temperature: float = 0.2,
        **kwargs,
    ) -> None:
        self.cl_weight = float(cl_weight)
        self.perturb_eps = float(perturb_eps)
        self.temperature = float(temperature)
        super().__init__(*args, **kwargs)

    def _propagate(self, all_emb: torch.Tensor) -> torch.Tensor:
        return super()._propagate(all_emb)

    def _cl_view(self, base: torch.Tensor) -> torch.Tensor:
        noise = F.normalize(torch.randn_like(base), dim=-1)
        return base + self.perturb_eps * noise

    def contrastive_loss(self, z1: torch.Tensor, z2: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        # Only compute logits for the participating indices to keep memory usage bounded.
        idx = idx.unique()
        z1 = F.normalize(z1[idx], dim=-1)
        z2 = F.normalize(z2[idx], dim=-1)
        logits = z1 @ z2.T / self.temperature
        labels = torch.arange(idx.numel(), device=logits.device)
        return F.cross_entropy(logits, labels)

    def forward(
        self,
        q_idx: torch.LongTensor,
        pos_idx: torch.LongTensor,
        neg_idx: torch.LongTensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q0, a0 = self._initial_embeddings()
        base = torch.cat([q0, a0], dim=0)
        h_clean = self._propagate(base)
        q_clean, a_clean = h_clean[: self.num_q], h_clean[self.num_q :]

        pos = (q_clean[q_idx.long()] * a_clean[pos_idx.long()]).sum(dim=-1)
        neg = (q_clean[q_idx.long()] * a_clean[neg_idx.long()]).sum(dim=-1)

        if self.cl_weight <= 0:
            return pos, neg, torch.tensor(0.0, device=pos.device)

        view1 = self._propagate(self._cl_view(base))
        view2 = self._propagate(self._cl_view(base))
        q_v1, a_v1 = view1[: self.num_q], view1[self.num_q :]
        q_v2, a_v2 = view2[: self.num_q], view2[self.num_q :]

        cl_q = self.contrastive_loss(q_v1, q_v2, q_idx.to(pos.device))
        agent_idx = torch.cat([pos_idx, neg_idx]).to(pos.device)
        cl_a = self.contrastive_loss(a_v1, a_v2, agent_idx)
        cl_loss = 0.5 * (cl_q + cl_a)
        return pos, neg, cl_loss

    def export_agent_embeddings(self) -> np.ndarray:
        # Override to avoid contrastive noise during export.
        with torch.no_grad():
            return super().export_agent_embeddings()

    def export_query_embeddings(self, q_indices: Sequence[int]) -> np.ndarray:
        with torch.no_grad():
            return super().export_query_embeddings(q_indices)

    def extra_state_dict(self) -> Dict[str, object]:
        base = super().extra_state_dict()
        base.update(
            {
                "num_layers": self.num_layers,
                "cl_weight": self.cl_weight,
                "perturb_eps": self.perturb_eps,
                "temperature": self.temperature,
            }
        )
        return base
