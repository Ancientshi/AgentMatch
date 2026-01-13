#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import RecommenderBase


class TwoTowerTFIDF(RecommenderBase):
    def __init__(
        self,
        d_q: int,
        d_a: int,
        num_tools: int,
        num_llm_ids: int,
        agent_tool_idx_padded: torch.LongTensor,
        agent_tool_mask: torch.FloatTensor,
        agent_llm_idx: torch.LongTensor,
        hid: int = 256,
        use_tool_id_emb: bool = True,
        use_llm_id_emb: bool = False,
        num_agents: int = 0,
        num_queries: int = 0,
        use_query_id_emb: bool = False,
        export_batch_size: int = 4096,
        # -------- NEW --------
        num_parts: int = 3,                 # Part I/II/III
        use_part_emb: bool = True,          # if you can pass part_idx explicitly
        use_moe_q: bool = True,             # gated experts on query tower
        gate_hid: int = 128,
        use_part_aux_loss: bool = True,     # optional: add part classification loss
    ) -> None:
        super().__init__()
        self.q_proj = nn.Sequential(nn.Linear(d_q, hid), nn.ReLU(), nn.Linear(hid, hid))
        self.a_proj = nn.Sequential(nn.Linear(d_a, hid), nn.ReLU(), nn.Linear(hid, hid))

        self.use_tool_id_emb = bool(use_tool_id_emb) and num_tools > 0
        if self.use_tool_id_emb:
            self.emb_tool = nn.Embedding(num_tools, hid)
            nn.init.xavier_uniform_(self.emb_tool.weight)
        else:
            self.emb_tool = None

        self.use_llm_id_emb = bool(use_llm_id_emb) and num_llm_ids > 0
        if self.use_llm_id_emb:
            self.emb_llm = nn.Embedding(num_llm_ids, hid)
            nn.init.xavier_uniform_(self.emb_llm.weight)
        else:
            self.emb_llm = None

        self.use_query_id_emb = bool(use_query_id_emb) and num_queries > 0
        if self.use_query_id_emb:
            if q_idx is None:
                # 推理时没有 query id -> 用 0 向量占位，避免维度不一致/崩溃
                parts.append(torch.zeros_like(base))
            else:
                parts.append(self.emb_query(q_idx))

        else:
            self.emb_query = None

        # -------- NEW: part embedding --------
        self.num_parts = int(num_parts)
        self.use_part_emb = bool(use_part_emb) and self.num_parts > 0
        if self.use_part_emb:
            self.emb_part = nn.Embedding(self.num_parts, hid)
            nn.init.xavier_uniform_(self.emb_part.weight)
        else:
            self.emb_part = None

        self.use_moe_q = bool(use_moe_q) and self.num_parts > 1

        self.register_buffer("tool_idx", agent_tool_idx_padded.long())
        self.register_buffer("tool_mask", agent_tool_mask.float())
        self.register_buffer("llm_idx", agent_llm_idx.long())

        # query input dim to head(s)
        combine_q_dim = hid + (hid if self.use_query_id_emb else 0) + (hid if self.use_part_emb else 0)
        # agent input dim to head
        combine_a_dim = hid + (hid if self.use_tool_id_emb else 0) + (hid if self.use_llm_id_emb else 0)

        # -------- NEW: MoE query heads + gate --------
        if self.use_moe_q:
            self.q_heads = nn.ModuleList([nn.Linear(combine_q_dim, hid) for _ in range(self.num_parts)])
            self.q_gate = nn.Sequential(
                nn.Linear(hid, gate_hid),
                nn.ReLU(),
                nn.Linear(gate_hid, self.num_parts),
            )
        else:
            self.q_head = nn.Linear(combine_q_dim, hid)

        self.a_head = nn.Linear(combine_a_dim, hid)

        # optional auxiliary part classifier (helps gate learn fast)
        self.use_part_aux_loss = bool(use_part_aux_loss) and self.num_parts > 1
        self.part_cls = nn.Linear(hid, self.num_parts) if self.use_part_aux_loss else None

        # init
        modules = list(self.q_proj) + list(self.a_proj) + [self.a_head]
        if self.use_moe_q:
            modules += list(self.q_heads) + list(self.q_gate)
        else:
            modules += [self.q_head]
        if self.part_cls is not None:
            modules += [self.part_cls]

        for m in modules:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

        self._agent_features: Optional[np.ndarray] = None
        self._query_features: Optional[np.ndarray] = None
        self.export_batch_size = int(export_batch_size)

    def tool_agg(self, agent_idx: torch.LongTensor) -> torch.Tensor:
        if not self.use_tool_id_emb:
            return torch.zeros((agent_idx.size(0), self.q_proj[-1].out_features), device=agent_idx.device)
        te = self.emb_tool(self.tool_idx[agent_idx])  # (B,T,H)
        mask = self.tool_mask[agent_idx].unsqueeze(-1)  # (B,T,1)
        return (te * mask).sum(1) / (mask.sum(1) + 1e-8)  # (B,H)

    # -------- NEW: optional auxiliary head --------
    def part_logits(self, q_vec: torch.Tensor) -> torch.Tensor:
        if self.part_cls is None:
            raise RuntimeError("part_cls is disabled (use_part_aux_loss=False).")
        base = self.q_proj(q_vec)  # (B,H)
        return self.part_cls(base)

    def encode_q(
        self,
        q_vec: torch.Tensor,
        q_idx: Optional[torch.LongTensor] = None,
        part_idx: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        base = self.q_proj(q_vec)  # (B,H)

        parts = [base]
        if self.use_query_id_emb:
            if q_idx is None:
                raise ValueError("q_idx is required when use_query_id_emb=True")
            parts.append(self.emb_query(q_idx))
        if self.use_part_emb:
            if part_idx is None:
                # allow missing at inference: use gate prediction as a pseudo part id
                if self.use_moe_q:
                    pred = torch.argmax(self.q_gate(base), dim=-1)
                    parts.append(self.emb_part(pred))
                else:
                    # if no MoE, you can just use a zero vector or a default part
                    parts.append(torch.zeros_like(base))
            else:
                parts.append(self.emb_part(part_idx))

        qh_in = torch.cat(parts, dim=-1) if len(parts) > 1 else parts[0]

        if self.use_moe_q:
            gate = F.softmax(self.q_gate(base), dim=-1)  # (B,P)
            outs = torch.stack([h(qh_in) for h in self.q_heads], dim=1)  # (B,P,H)
            qh = (gate.unsqueeze(-1) * outs).sum(dim=1)  # (B,H)
        else:
            qh = self.q_head(qh_in)

        return F.normalize(qh, dim=-1)

    def encode_a(self, a_vec: torch.Tensor, agent_idx: torch.LongTensor) -> torch.Tensor:
        parts = [self.a_proj(a_vec)]
        if self.use_tool_id_emb:
            parts.append(self.tool_agg(agent_idx))
        if self.use_llm_id_emb:
            parts.append(self.emb_llm(self.llm_idx[agent_idx]))
        ah = torch.cat(parts, dim=-1) if len(parts) > 1 else parts[0]
        ah = self.a_head(ah)
        return F.normalize(ah, dim=-1)

    def forward_score(
        self,
        q_vec: torch.Tensor,
        a_vec: torch.Tensor,
        agent_idx: torch.LongTensor,
        q_idx: Optional[torch.LongTensor] = None,
        part_idx: Optional[torch.LongTensor] = None,  # NEW
    ) -> torch.Tensor:
        qe = self.encode_q(q_vec, q_idx=q_idx, part_idx=part_idx)
        ae = self.encode_a(a_vec, agent_idx)
        return (qe * ae).sum(dim=-1)

    def set_agent_features(self, A_text_full: np.ndarray) -> None:
        self._agent_features = A_text_full.astype(np.float32, copy=False)

    def set_query_features(self, Q: np.ndarray) -> None:
        self._query_features = Q.astype(np.float32, copy=False)

    def export_agent_embeddings(self) -> np.ndarray:
        if self._agent_features is None:
            raise RuntimeError("Agent features not set. Call set_agent_features() before export.")
        A_cpu = self._agent_features
        device = next(self.parameters()).device
        num_agents = A_cpu.shape[0]
        out = []
        with torch.no_grad():
            for start in range(0, num_agents, self.export_batch_size):
                end = min(start + self.export_batch_size, num_agents)
                idx = torch.arange(start, end, device=device)
                av = torch.from_numpy(A_cpu[start:end]).to(device)
                ae = self.encode_a(av, idx).cpu().numpy()
                out.append(ae)
        return np.vstack(out).astype(np.float32)

    def export_query_embeddings(self, q_indices: Sequence[int]) -> np.ndarray:
        if self._query_features is None:
            raise RuntimeError("Query features not set. Call set_query_features() before export.")
        device = next(self.parameters()).device
        q_indices = list(q_indices)
        Q_cpu = self._query_features
        out = []
        with torch.no_grad():
            for start in range(0, len(q_indices), self.export_batch_size):
                batch_idx = q_indices[start : start + self.export_batch_size]
                qv = torch.from_numpy(Q_cpu[batch_idx]).to(device)
                q_idx = torch.tensor(batch_idx, dtype=torch.long, device=device) if self.use_query_id_emb else None
                # 注意：这里 part_idx 不传，让 gate 自己判别
                qe = self.encode_q(qv, q_idx=q_idx, part_idx=None).cpu().numpy()
                out.append(qe)
        return np.vstack(out).astype(np.float32)

    def export_agent_bias(self) -> Optional[np.ndarray]:
        return None

    def extra_state_dict(self) -> Dict[str, Any]:
        return {
            "use_tool_id_emb": self.use_tool_id_emb,
            "use_llm_id_emb": self.use_llm_id_emb,
            "use_query_id_emb": self.use_query_id_emb,
            "use_part_emb": self.use_part_emb,
            "use_moe_q": self.use_moe_q,
            "use_part_aux_loss": self.use_part_aux_loss,
            "export_batch_size": self.export_batch_size,
            "num_parts": self.num_parts,
        }
