#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
from typing import Dict, List, Any
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .config import EPS, TFIDF_MAX_FEATURES
from .models.base import RecommenderBase


def build_knn_cache(
    train_qids: List[str],
    all_questions: Dict[str, dict],
    qid2idx: Dict[str, int],
    model: RecommenderBase,
    cache_path: str,
    tfidf_max_features: int = TFIDF_MAX_FEATURES,
) -> None:
    train_texts = [all_questions[qid].get("input", "") for qid in train_qids]
    tfidf = TfidfVectorizer(max_features=tfidf_max_features)
    X = tfidf.fit_transform(train_texts).astype(np.float32)

    q_indices = [qid2idx[qid] for qid in train_qids]
    Q = model.export_query_embeddings(q_indices)

    knn_cache = {"train_qids": train_qids, "tfidf": tfidf, "X": X, "Q": Q}
    with open(cache_path, "wb") as f:
        pickle.dump(knn_cache, f)


def build_knn_cache_with_vectorizer(
    train_qids: List[str],
    all_questions: Dict[str, dict],
    qid2idx: Dict[str, int],
    model: RecommenderBase,
    cache_path: str,
    tfidf: TfidfVectorizer,
) -> None:
    train_texts = [all_questions[qid].get("input", "") for qid in train_qids]
    X = tfidf.transform(train_texts).astype(np.float32)

    q_indices = [qid2idx[qid] for qid in train_qids]
    Q = model.export_query_embeddings(q_indices)

    knn_cache = {"train_qids": train_qids, "tfidf": tfidf, "X": X, "Q": Q}
    with open(cache_path, "wb") as f:
        pickle.dump(knn_cache, f)


def load_knn_cache(cache_path: str) -> Dict[str, Any]:
    with open(cache_path, "rb") as f:
        return pickle.load(f)


def knn_qvec_for_question_text(question_text: str, knn_cache: dict, N: int = 8) -> np.ndarray:
    tfidf, X, Q = knn_cache["tfidf"], knn_cache["X"], knn_cache["Q"]
    x = tfidf.transform([question_text]).astype(np.float32)
    sims = (x @ X.T).toarray()[0]

    if sims.size == 0:
        return Q.mean(axis=0).astype(np.float32)

    if N >= sims.size:
        idx = np.argsort(-sims)
    else:
        idx = np.argpartition(-sims, N - 1)[:N]
        idx = idx[np.argsort(-sims[idx])]

    w = sims[idx]
    s = float(w.sum())
    if s <= EPS:
        return np.zeros((Q.shape[1],), dtype=np.float32)

    w = w / (s + EPS)
    qv = (w[:, None] * Q[idx]).sum(axis=0)
    return qv.astype(np.float32)