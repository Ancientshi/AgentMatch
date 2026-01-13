#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from agent_rec.features import (
    UNK_LLM_TOKEN,
    UNK_TOOL_TOKEN,
    agent_tool_text_matrix,
    build_agent_content_view,
    build_agent_tool_id_buffers,
    build_unified_corpora,
)


@dataclass
class GraphFeaturePack:
    agent_content: Optional[np.ndarray]
    agent_tool_idx_padded: np.ndarray
    agent_tool_mask: np.ndarray
    agent_llm_idx: np.ndarray
    tool_id_vocab: List[str]
    llm_vocab: List[str]


def build_graph_features(
    all_agents: Dict[str, dict],
    all_questions: Dict[str, dict],
    tools: Dict[str, dict],
    *,
    q_ids: Sequence[str],
    a_ids: Sequence[str],
    max_features: int,
    use_model_content_vector: bool,
    use_tool_content_vector: bool,
) -> GraphFeaturePack:
    (
        q_ids_ordered,
        _,
        tool_names,
        tool_texts,
        a_ids_ordered,
        model_names,
        agent_tool_lists,
        llm_ids,
    ) = build_unified_corpora(
        all_agents,
        all_questions,
        tools,
        q_id_order=list(q_ids),
        a_id_order=list(a_ids),
        tool_name_order=list(tools.keys()),
    )

    if list(q_ids_ordered) != list(q_ids) or list(a_ids_ordered) != list(a_ids):
        raise ValueError("ID ordering mismatch between bootstrap and feature builder.")

    content_parts: List[np.ndarray] = []
    if use_model_content_vector:
        model_vec = TfidfVectorizer(max_features=max_features, lowercase=True)
        A_model_content = model_vec.fit_transform(model_names).toarray().astype(np.float32)
        content_parts.append(A_model_content)
    else:
        A_model_content = np.zeros((len(a_ids), 0), dtype=np.float32)

    if use_tool_content_vector:
        tool_vec = TfidfVectorizer(max_features=max_features, lowercase=True)
        Tm_csr = tool_vec.fit_transform(tool_texts)
        A_tool_content = agent_tool_text_matrix(agent_tool_lists, tool_names, Tm_csr)
        content_parts.append(A_tool_content)
    else:
        A_tool_content = np.zeros((len(a_ids), 0), dtype=np.float32)

    if content_parts:
        agent_content = build_agent_content_view(
            A_model_content=A_model_content,
            A_tool_content=A_tool_content,
            use_model_content_vector=use_model_content_vector,
            use_tool_content_vector=use_tool_content_vector,
        )
    else:
        agent_content = None

    tool_id_vocab = [UNK_TOOL_TOKEN] + list(tool_names)
    tool_vocab_map = {n: i for i, n in enumerate(tool_id_vocab)}
    agent_tool_idx_padded, agent_tool_mask = build_agent_tool_id_buffers(agent_tool_lists, tool_vocab_map)

    llm_vocab = [UNK_LLM_TOKEN] + [lid for lid in llm_ids if lid]
    llm_vocab = list(dict.fromkeys(llm_vocab))
    llm_vocab_map = {n: i for i, n in enumerate(llm_vocab)}
    agent_llm_idx = np.array([llm_vocab_map.get(lid, 0) for lid in llm_ids], dtype=np.int64)

    return GraphFeaturePack(
        agent_content=agent_content,
        agent_tool_idx_padded=agent_tool_idx_padded,
        agent_tool_mask=agent_tool_mask,
        agent_llm_idx=agent_llm_idx,
        tool_id_vocab=tool_id_vocab,
        llm_vocab=llm_vocab,
    )
