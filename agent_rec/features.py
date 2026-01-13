#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass
import os
import tempfile
import uuid
from typing import Dict, List, Tuple, Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from numpy.lib.format import open_memmap

try:
    from scipy import sparse as sp
except Exception as e:  # pragma: no cover - import error is user environment
    raise RuntimeError("This module needs SciPy for CSR features. pip install scipy\n" + str(e))

from .config import TFIDF_MAX_FEATURES

UNK_LLM_TOKEN = "<UNK_LLM>"
UNK_TOOL_TOKEN = "<UNK_TOOL>"



@dataclass
class TFIDFVectorizers:
    q_vec: TfidfVectorizer
    model_vec: TfidfVectorizer
    tool_vec: TfidfVectorizer

@dataclass
class FeatureCache:
    q_ids: List[str]
    a_ids: List[str]
    tool_names: List[str]
    tool_id_vocab: List[str]
    llm_ids: List[str]
    llm_vocab: List[str]
    Q: np.ndarray
    A_model_content: np.ndarray
    A_tool_content: np.ndarray
    A_text_full: np.ndarray
    agent_llm_idx: np.ndarray
    agent_tool_idx_padded: np.ndarray
    agent_tool_mask: np.ndarray


def build_agent_content_view(
    *,
    cache: FeatureCache | None = None,
    A_model_content: np.ndarray | None = None,
    A_tool_content: np.ndarray | None = None,
    use_model_content_vector: bool = True,
    use_tool_content_vector: bool = True,
) -> np.ndarray:
    """
    Build the agent content representation
    ϕ_content(A)=concat(V_model(A), V_tool_content(A)).

    Set use_model_content_vector/use_tool_content_vector to control which parts
    are included. At least one of them must be True.
    """
    if cache is not None:
        if A_model_content is None:
            A_model_content = cache.A_model_content
        if A_tool_content is None:
            A_tool_content = cache.A_tool_content

    parts: List[np.ndarray] = []
    if use_model_content_vector:
        if A_model_content is None:
            raise ValueError("A_model_content is required when use_model_content_vector=True")
        parts.append(np.array(A_model_content, dtype=np.float32, copy=False))
    if use_tool_content_vector:
        if A_tool_content is None:
            raise ValueError("A_tool_content is required when use_tool_content_vector=True")
        parts.append(np.array(A_tool_content, dtype=np.float32, copy=False))

    if not parts:
        #构建与A_model_content和A_tool_content横轴数轴相同形状的0矩阵
        parts.append(np.zeros((A_model_content.shape[0], A_model_content.shape[1]), dtype=np.float32))
        parts.append(np.zeros((A_tool_content.shape[0], A_tool_content.shape[1]), dtype=np.float32))
         

    num_agents = parts[0].shape[0]
    for p in parts[1:]:
        if p.shape[0] != num_agents:
            raise ValueError(f"Content part row mismatch: {p.shape[0]} vs expected {num_agents}")

    out = np.concatenate(parts, axis=1).astype(np.float32)
    if out.shape[1] == 0:
        raise ValueError("Agent content view has zero width; check content flags and source vectors.")
    return out


def _tool_text(name: str, tools: Dict[str, dict]) -> str:
    t = tools.get(name, {}) or {}
    desc = t.get("description", "")
    return f"{name} {desc}".strip()


def _extract_agent_fields(all_agents: Dict[str, dict]) -> Tuple[List[str], List[str], List[List[str]], List[str]]:
    a_ids = list(all_agents.keys())
    model_names: List[str] = []
    tool_lists: List[List[str]] = []
    llm_ids: List[str] = []
    for aid in a_ids:
        a = all_agents.get(aid, {}) or {}
        m = (a.get("M") or {}) if isinstance(a, dict) else {}
        model_names.append((m.get("name") or "").strip())
        tool_lists.append(((a.get("T") or {}).get("tools") or []))
        llm_ids.append((m.get("id") or m.get("name") or "").strip())
    return a_ids, model_names, tool_lists, llm_ids


def _build_llm_vocab(llm_ids: List[str]) -> List[str]:
    vocab = [UNK_LLM_TOKEN]
    for lid in llm_ids:
        if lid and lid not in vocab:
            vocab.append(lid)
    return vocab


def _build_tool_vocab(tool_names: List[str]) -> List[str]:
    return [UNK_TOOL_TOKEN] + list(tool_names)


def _map_llm_ids(llm_ids: List[str], vocab_map: Dict[str, int]) -> np.ndarray:
    unk = vocab_map.get(UNK_LLM_TOKEN, 0)
    return np.array([vocab_map.get(lid, unk) for lid in llm_ids], dtype=np.int64)


def build_agent_tool_id_buffers(
    agent_tool_lists: List[List[str]],
    tool_vocab_map: Dict[str, int],
    *,
    max_tools_per_agent: Optional[int] = 8,
) -> Tuple[np.ndarray, np.ndarray]:
    unk_idx = tool_vocab_map.get(UNK_TOOL_TOKEN, 0)
    num_agents = len(agent_tool_lists)

    if max_tools_per_agent is not None and max_tools_per_agent <= 0:
        raise ValueError("max_tools_per_agent must be positive or None for unlimited")

    max_len_from_data = max([len(lst) for lst in agent_tool_lists]) if num_agents > 0 else 0
    if max_tools_per_agent is None:
        max_t = max_len_from_data
    else:
        max_t = min(max_len_from_data, max_tools_per_agent)
        
    if max_t == 0:
        max_t = 1
    idx_pad = np.full((num_agents, max_t), unk_idx, dtype=np.int64)
    mask = np.zeros((num_agents, max_t), dtype=np.float32)
    for i, lst in enumerate(agent_tool_lists):
        if not lst:
            # Leave mask as zeros so padding/UNK slots are ignored in mean pooling.
            idx_pad[i, 0] = unk_idx
            continue
        capped_list = lst if max_tools_per_agent is None else lst[:max_tools_per_agent]
        for j, name in enumerate(capped_list[:max_t]):
            idx_pad[i, j] = tool_vocab_map.get(name, unk_idx)
            mask[i, j] = 1.0
    return idx_pad, mask


def agent_tool_text_matrix(
    agent_tool_lists: List[List[str]],
    tool_names: List[str],
    tool_matrix: "sp.csr_matrix",
) -> np.ndarray:
    name2idx = {n: i for i, n in enumerate(tool_names)}
    num_agents = len(agent_tool_lists)
    dim = tool_matrix.shape[1]
    out = np.zeros((num_agents, dim), dtype=np.float32)
    for i, tool_list in enumerate(agent_tool_lists):
        idxs = [name2idx[t] for t in tool_list if t in name2idx]
        if not idxs:
            continue
        vecs = tool_matrix[idxs].toarray()
        out[i] = vecs.mean(axis=0).astype(np.float32)
    return out


def _reorder_ids(ids: List[str], desired_order: Optional[List[str]]) -> List[str]:
    """
    Return a deterministic ordering for IDs.

    If `desired_order` is provided, ensure it matches the ID set and follow that order.
    Otherwise, return the IDs sorted alphabetically.
    """
    if desired_order is None:
        return sorted(ids)
    ids_set = set(ids)
    desired_set = set(desired_order)
    if ids_set != desired_set:
        missing = ids_set - desired_set
        extra = desired_set - ids_set
        raise ValueError(f"Desired ID order mismatch. missing={missing} extra={extra}")
    return [i for i in desired_order if i in ids_set]


def _base_corpora(
    all_agents: Dict[str, dict],
    all_questions: Dict[str, dict],
    tools: Dict[str, dict],
    *,
    q_id_order: Optional[List[str]] = None,
    a_id_order: Optional[List[str]] = None,
    tool_name_order: Optional[List[str]] = None,
) -> Tuple[List[str], List[str], List[str], List[str], List[str], List[str], List[List[str]], List[str]]:
    q_ids = _reorder_ids(list(all_questions.keys()), q_id_order)
    q_texts = [all_questions[qid].get("input", "") for qid in q_ids]

    tool_names = _reorder_ids(list(tools.keys()), tool_name_order)
    tool_texts = [_tool_text(name, tools) for name in tool_names]

    orig_a_ids, model_names_raw, tool_lists_raw, llm_ids_raw = _extract_agent_fields(all_agents)
    a_ids = _reorder_ids(orig_a_ids, a_id_order)
    model_map = dict(zip(orig_a_ids, model_names_raw))
    tool_map = dict(zip(orig_a_ids, tool_lists_raw))
    llm_map = dict(zip(orig_a_ids, llm_ids_raw))
    model_names = [model_map[aid] for aid in a_ids]
    tool_lists = [tool_map[aid] for aid in a_ids]
    llm_ids = [llm_map[aid] for aid in a_ids]
    return q_ids, q_texts, tool_names, tool_texts, a_ids, model_names, tool_lists, llm_ids


def build_unified_corpora(
    all_agents: Dict[str, dict],
    all_questions: Dict[str, dict],
    tools: Dict[str, dict],
    *,
    q_id_order: Optional[List[str]] = None,
    a_id_order: Optional[List[str]] = None,
    tool_name_order: Optional[List[str]] = None,
) -> Tuple[List[str], List[str], List[str], List[str], List[str], List[str], List[List[str]], List[str]]:
    """
    Unified corpora builder for all models.

    Returns (q_ids, q_texts, tool_names, tool_texts, a_ids, model_names, a_tool_lists, llm_ids)
    so every model can construct agent content (model name + tool content) and ID (llm_id + tool ids) views consistently.
    The optional *_order arguments enforce deterministic ordering (matching bootstrap-run id maps).
    """
    return _base_corpora(
        all_agents,
        all_questions,
        tools,
        q_id_order=q_id_order,
        a_id_order=a_id_order,
        tool_name_order=tool_name_order,
    )


def build_feature_cache(
    all_agents: Dict[str, dict],
    all_questions: Dict[str, dict],
    tools: Dict[str, dict],
    max_features: int = TFIDF_MAX_FEATURES,
) -> Tuple[FeatureCache, TFIDFVectorizers]:
    (
        q_ids,
        q_texts,
        tool_names,
        tool_texts,
        a_ids,
        model_names,
        a_tool_lists,
        llm_ids,
    ) = _base_corpora(all_agents, all_questions, tools)

    q_vec = TfidfVectorizer(max_features=max_features, lowercase=True)
    tool_vec = TfidfVectorizer(max_features=max_features, lowercase=True)
    model_vec = TfidfVectorizer(max_features=max_features, lowercase=True)

    Q = q_vec.fit_transform(q_texts).toarray().astype(np.float32)
    Tm_csr = tool_vec.fit_transform(tool_texts)
    A_model_csr = model_vec.fit_transform(model_names)

    A_model_content = A_model_csr.toarray().astype(np.float32)
    A_tool_content = agent_tool_text_matrix(a_tool_lists, tool_names, Tm_csr)
    A_text_full = build_agent_content_view(
        A_model_content=A_model_content, A_tool_content=A_tool_content, use_model_content_vector=True
    )

    tool_id_vocab = _build_tool_vocab(tool_names)
    tool_vocab_map = {n: i for i, n in enumerate(tool_id_vocab)}
    agent_tool_idx_padded, agent_tool_mask = build_agent_tool_id_buffers(a_tool_lists, tool_vocab_map)

    llm_vocab = _build_llm_vocab(llm_ids)
    llm_vocab_map = {n: i for i, n in enumerate(llm_vocab)}
    agent_llm_idx = _map_llm_ids(llm_ids, llm_vocab_map)

    return (
        FeatureCache(
            q_ids=q_ids,
            a_ids=a_ids,
            tool_names=tool_names,
            tool_id_vocab=tool_id_vocab,
            llm_ids=llm_ids,
            llm_vocab=llm_vocab,
            Q=Q,
            A_model_content=A_model_content,
            A_tool_content=A_tool_content,
            A_text_full=A_text_full,
            agent_llm_idx=agent_llm_idx,
            agent_tool_idx_padded=agent_tool_idx_padded,
            agent_tool_mask=agent_tool_mask,
        ),
        TFIDFVectorizers(q_vec=q_vec, model_vec=model_vec, tool_vec=tool_vec),
    )


# Backward-compatible alias with the unified agent content view
def build_twotower_feature_cache(
    all_agents: Dict[str, dict],
    all_questions: Dict[str, dict],
    tools: Dict[str, dict],
    max_features: int = TFIDF_MAX_FEATURES,
) -> Tuple[FeatureCache, TFIDFVectorizers]:
    return build_feature_cache(all_agents, all_questions, tools, max_features=max_features)


def save_feature_cache(cache_dir: str, cache: FeatureCache) -> None:
    def dump_json(name: str, payload: object) -> None:
        with open(f"{cache_dir}/{name}", "w", encoding="utf-8") as f:
            import json

            json.dump(payload, f, ensure_ascii=False)

    dump_json("q_ids.json", cache.q_ids)
    dump_json("a_ids.json", cache.a_ids)
    dump_json("tool_names.json", cache.tool_names)
    dump_json("tool_id_vocab.json", cache.tool_id_vocab)
    dump_json("llm_ids.json", cache.llm_ids)
    dump_json("llm_vocab.json", cache.llm_vocab)

    np.save(f"{cache_dir}/Q.npy", cache.Q.astype(np.float32))
    np.save(f"{cache_dir}/A_model_content.npy", cache.A_model_content.astype(np.float32))
    np.save(f"{cache_dir}/A_tool_content.npy", cache.A_tool_content.astype(np.float32))
    np.save(f"{cache_dir}/A_text_full.npy", cache.A_text_full.astype(np.float32))
    np.save(f"{cache_dir}/agent_llm_idx.npy", cache.agent_llm_idx.astype(np.int64))
    np.save(f"{cache_dir}/agent_tool_idx_padded.npy", cache.agent_tool_idx_padded.astype(np.int64))
    np.save(f"{cache_dir}/agent_tool_mask.npy", cache.agent_tool_mask.astype(np.float32))


def save_vectorizers(cache_dir: str, vecs: TFIDFVectorizers) -> None:
    import pickle
    with open(f"{cache_dir}/q_vectorizer.pkl", "wb") as f:
        pickle.dump(vecs.q_vec, f)
    with open(f"{cache_dir}/model_vectorizer.pkl", "wb") as f:
        pickle.dump(vecs.model_vec, f)
    with open(f"{cache_dir}/tool_vectorizer.pkl", "wb") as f:
        pickle.dump(vecs.tool_vec, f)


def load_vectorizers(cache_dir: str) -> Optional[TFIDFVectorizers]:
    import os, pickle
    q_path = f"{cache_dir}/q_vectorizer.pkl"
    m_path = f"{cache_dir}/model_vectorizer.pkl"
    t_path = f"{cache_dir}/tool_vectorizer.pkl"
    if not (os.path.exists(q_path) and os.path.exists(m_path) and os.path.exists(t_path)):
        return None
    with open(q_path, "rb") as f:
        q_vec = pickle.load(f)
    with open(m_path, "rb") as f:
        model_vec = pickle.load(f)
    with open(t_path, "rb") as f:
        tool_vec = pickle.load(f)
    return TFIDFVectorizers(q_vec=q_vec, model_vec=model_vec, tool_vec=tool_vec)


def save_q_vectorizer(cache_dir: str, q_vec: TfidfVectorizer) -> None:
    import pickle

    with open(f"{cache_dir}/q_vectorizer.pkl", "wb") as f:
        pickle.dump(q_vec, f)


def load_q_vectorizer(cache_dir: str) -> Optional[TfidfVectorizer]:
    import os
    import pickle

    path = f"{cache_dir}/q_vectorizer.pkl"
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def load_feature_cache(cache_dir: str) -> FeatureCache:
    import json

    with open(f"{cache_dir}/q_ids.json", "r", encoding="utf-8") as f:
        q_ids = json.load(f)
    with open(f"{cache_dir}/a_ids.json", "r", encoding="utf-8") as f:
        a_ids = json.load(f)
    with open(f"{cache_dir}/tool_names.json", "r", encoding="utf-8") as f:
        tool_names = json.load(f)
    with open(f"{cache_dir}/tool_id_vocab.json", "r", encoding="utf-8") as f:
        tool_id_vocab = json.load(f)
    with open(f"{cache_dir}/llm_ids.json", "r", encoding="utf-8") as f:
        llm_ids = json.load(f)
    with open(f"{cache_dir}/llm_vocab.json", "r", encoding="utf-8") as f:
        llm_vocab = json.load(f)

    Q = np.load(f"{cache_dir}/Q.npy")
    A_model_content = np.load(f"{cache_dir}/A_model_content.npy")
    A_tool_content = np.load(f"{cache_dir}/A_tool_content.npy")
    A_text_full = np.load(f"{cache_dir}/A_text_full.npy")
    agent_llm_idx = np.load(f"{cache_dir}/agent_llm_idx.npy")
    agent_tool_idx_padded = np.load(f"{cache_dir}/agent_tool_idx_padded.npy")
    agent_tool_mask = np.load(f"{cache_dir}/agent_tool_mask.npy")

    return FeatureCache(
        q_ids=q_ids,
        a_ids=a_ids,
        tool_names=tool_names,
        tool_id_vocab=tool_id_vocab,
        llm_ids=llm_ids,
        llm_vocab=llm_vocab,
        Q=Q,
        A_model_content=A_model_content,
        A_tool_content=A_tool_content,
        A_text_full=A_text_full,
        agent_llm_idx=agent_llm_idx,
        agent_tool_idx_padded=agent_tool_idx_padded,
        agent_tool_mask=agent_tool_mask,
    )


def feature_cache_exists(cache_dir: str) -> bool:
    import os

    needed = [
        "q_ids.json",
        "a_ids.json",
        "tool_names.json",
        "tool_id_vocab.json",
        "llm_ids.json",
        "llm_vocab.json",
        "Q.npy",
        "A_model_content.npy",
        "A_tool_content.npy",
        "A_text_full.npy",
        "agent_llm_idx.npy",
        "agent_tool_idx_padded.npy",
        "agent_tool_mask.npy",
        "model_vectorizer.pkl",
        "tool_vectorizer.pkl",
    ]
    return all(os.path.exists(f"{cache_dir}/{name}") for name in needed)


def to_2d_float32(x: np.ndarray | list | tuple) -> np.ndarray:
    if isinstance(x, tuple) and len(x) == 3:
        arr, inv_order, _ = x
        x = arr if inv_order is None else arr[inv_order, :]
    x = np.array(x)
    if x.dtype == np.object_ or x.ndim != 2:
        try:
            x = np.vstack([np.array(row, dtype=np.float32) for row in x])
        except Exception as e:
            raise ValueError(
                f"Embedding batch is ragged or non-2D: {type(x)}, shape={getattr(x, 'shape', None)}"
            ) from e
    return x.astype(np.float32, copy=False)


def l2_normalize(mat: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    mat = to_2d_float32(mat)
    n = np.linalg.norm(mat, axis=1, keepdims=True)
    n = np.maximum(n, eps)
    return mat / n


def _post_embed(embed_url: str, docs: List[str]) -> np.ndarray:
    if embed_url and embed_url.startswith("http"):
        import requests

        response = requests.post(embed_url, json={"documents": docs}, timeout=120)
        response.raise_for_status()
        payload = response.json()
        if "embeddings" not in payload:
            raise ValueError("Embedding service response missing `embeddings` field.")
        return np.array(payload["embeddings"], dtype=np.float32)

    from utils import load_BGEM3_model, get_embeddings

    load_BGEM3_model()
    embs = get_embeddings(docs)
    return np.array(embs, dtype=np.float32)


def batch_embed(
    texts: List[str],
    embed_url: str,
    batch_size: int = 64,
    desc: str = "Embedding",
    *,
    use_memmap: bool = True,
    memmap_path: str | None = None,
    return_mode: str = "array",
    sort_by_length: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray, str]:
    num_texts = len(texts)
    if num_texts == 0:
        return np.zeros((0, 0), dtype=np.float32)

    order = np.arange(num_texts)
    if sort_by_length:
        order = np.argsort([len(t) for t in texts])
    inv_order = np.empty_like(order)
    inv_order[order] = np.arange(num_texts)
    texts_sorted = [texts[i] for i in order]

    out = None
    mm = None
    dim = None

    for start in range(0, num_texts, batch_size):
        end = min(start + batch_size, num_texts)
        chunk = texts_sorted[start:end]
        embs = _post_embed(embed_url, chunk)
        if dim is None:
            dim = int(embs.shape[1])
            if use_memmap:
                if memmap_path is None:
                    memmap_path = os.path.join(tempfile.gettempdir(), f"emb_{uuid.uuid4().hex}.npy")
                mm = open_memmap(memmap_path, mode="w+", dtype=np.float32, shape=(num_texts, dim))
            else:
                out = np.empty((num_texts, dim), dtype=np.float32)

        if embs.shape[1] != dim:
            raise ValueError(f"Embedding dim changed: got {embs.shape[1]} vs {dim}")

        target = mm if use_memmap else out
        target[start:end, :] = embs

    if use_memmap:
        mm.flush()
        view = np.load(memmap_path, mmap_mode="r")
        if sort_by_length:
            view = view[inv_order, :]
        if return_mode == "mmap":
            return view, inv_order, memmap_path
        return np.array(view, copy=True)

    return out if not sort_by_length else out[inv_order, :]


def agent_tool_text_matrix_bge(
    agent_tool_lists: List[List[str]],
    tool_names: List[str],
    tool_embs: np.ndarray,
) -> np.ndarray:
    name2idx = {n: i for i, n in enumerate(tool_names)}
    dim = tool_embs.shape[1]
    num_agents = len(agent_tool_lists)
    out = np.zeros((num_agents, dim), dtype=np.float32)
    for i, tool_list in enumerate(agent_tool_lists):
        idxs = [name2idx[t] for t in tool_list if t in name2idx]
        if not idxs:
            continue
        vecs = tool_embs[idxs]
        out[i] = vecs.mean(axis=0).astype(np.float32)
    return out


def build_twotower_bge_feature_cache(
    all_agents: Dict[str, dict],
    all_questions: Dict[str, dict],
    tools: Dict[str, dict],
    *,
    embed_url: str,
    embed_batch: int = 64,
    use_memmap: bool = True,
    sort_by_length: bool = False,
) -> FeatureCache:
    (
        q_ids,
        q_texts,
        tool_names,
        tool_texts,
        a_ids,
        model_names,
        a_tool_lists,
        llm_ids,
    ) = _base_corpora(all_agents, all_questions, tools)

    Q = batch_embed(q_texts, embed_url, embed_batch, desc="Embedding questions", use_memmap=use_memmap)
    Q = l2_normalize(Q)

    ToolE = batch_embed(tool_texts, embed_url, embed_batch, desc="Embedding tools", use_memmap=use_memmap)
    ToolE = l2_normalize(ToolE)

    A_model_emb = batch_embed(
        model_names, embed_url, embed_batch, desc="Embedding agents (model name)", use_memmap=use_memmap
    )
    A_model_emb = l2_normalize(A_model_emb)

    A_tool_emb = agent_tool_text_matrix_bge(a_tool_lists, tool_names, ToolE)
    if A_tool_emb.size > 0:
        A_tool_emb = l2_normalize(A_tool_emb)

    A_text_full = build_agent_content_view(
        A_model_content=A_model_emb, A_tool_content=A_tool_emb, use_model_content_vector=True
    )

    tool_id_vocab = _build_tool_vocab(tool_names)
    tool_vocab_map = {n: i for i, n in enumerate(tool_id_vocab)}
    agent_tool_idx_padded, agent_tool_mask = build_agent_tool_id_buffers(a_tool_lists, tool_vocab_map)

    llm_vocab = _build_llm_vocab(llm_ids)
    llm_vocab_map = {n: i for i, n in enumerate(llm_vocab)}
    agent_llm_idx = _map_llm_ids(llm_ids, llm_vocab_map)

    return FeatureCache(
        q_ids=q_ids,
        a_ids=a_ids,
        tool_names=tool_names,
        tool_id_vocab=tool_id_vocab,
        llm_ids=llm_ids,
        llm_vocab=llm_vocab,
        Q=Q.astype(np.float32),
        A_model_content=A_model_emb.astype(np.float32),
        A_tool_content=A_tool_emb.astype(np.float32),
        A_text_full=A_text_full.astype(np.float32),
        agent_llm_idx=agent_llm_idx,
        agent_tool_idx_padded=agent_tool_idx_padded,
        agent_tool_mask=agent_tool_mask,
    )


def normalize_features(user_np: np.ndarray, item_np: np.ndarray) -> Tuple["sp.csr_matrix", "sp.csr_matrix"]:
    user_np = normalize(user_np, norm="l2", axis=1, copy=False)
    item_np = normalize(item_np, norm="l2", axis=1, copy=False)
    return sp.csr_matrix(user_np, dtype=np.float32), sp.csr_matrix(item_np, dtype=np.float32)
