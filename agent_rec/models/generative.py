# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from agent_rec.data import DatasetBundle, collect_data, load_tools
from agent_rec.features import UNK_LLM_TOKEN, UNK_TOOL_TOKEN
from agent_rec.rerank_eval_utils import build_agent_text_cache


@dataclass
class GenerationConfig:
    """Config for formatting and retrieval used by the generator."""

    tool_sep_token: str = "<TOOL_SEP>"
    end_token: str = "<SPECIAL_END>"
    max_tools: int = 4
    tfidf_max_features: int = 5000


@dataclass
class AgentGenerationInfo:
    """Lightweight view over agent metadata needed for generation."""

    aid: str
    llm_token: str
    tool_tokens: List[str]
    text: str


def _normalize_token(token: str) -> str:
    token = token.strip() if token else ""
    if not token:
        return ""
    return token.replace(" ", "_")


class GenerativeStructuredRecommender:
    """
    Lightweight (inference-only) generative-style recommender that produces
    structured token outputs like::

        <LLM_TOKEN> <TOOL_SEP> <TOOL_A> <TOOL_B> <SPECIAL_END>

    Retrieval uses TF-IDF over agent descriptions to pick the most relevant
    agent(s), then formats the agent's LLM/tool metadata into the requested
    token template.
    """

    def __init__(
        self,
        *,
        vectorizer: TfidfVectorizer,
        agent_matrix,
        agents: Sequence[AgentGenerationInfo],
        config: GenerationConfig,
    ) -> None:
        self.vectorizer = vectorizer
        self.agent_matrix = agent_matrix
        self.agents = list(agents)
        self.config = config

    @classmethod
    def from_bundle(
        cls,
        bundle: DatasetBundle,
        *,
        tools: Mapping[str, dict],
        config: GenerationConfig | None = None,
        agent_order: Iterable[str] | None = None,
    ) -> "GenerativeStructuredRecommender":
        """Build a generator from an in-memory dataset bundle.

        ``bundle`` should come from the shared data loader utilities (e.g.
        :func:`agent_rec.run_common.bootstrap_run`). This keeps the generator
        aligned with the rest of the ``run_*.py`` entrypoints so they share
        logging, seeding, and data splits.
        """

        cfg = config or GenerationConfig()
        agent_text_cache = build_agent_text_cache(bundle.all_agents, tools)

        agent_infos: List[AgentGenerationInfo] = []
        agent_ids = list(agent_order) if agent_order is not None else sorted(bundle.all_agents.keys())
        for aid in agent_ids:
            agent = bundle.all_agents.get(aid, {})
            m = (agent.get("M") or {}) if isinstance(agent, dict) else {}
            t = (agent.get("T") or {}) if isinstance(agent, dict) else {}

            llm_token = _normalize_token(m.get("id") or m.get("name") or "")
            tool_tokens = [
                tok
                for tok in (_normalize_token(x) for x in (t.get("tools") or []))
                if tok
            ]

            agent_infos.append(
                AgentGenerationInfo(
                    aid=aid,
                    llm_token=llm_token or UNK_LLM_TOKEN,
                    tool_tokens=tool_tokens or [UNK_TOOL_TOKEN],
                    text=agent_text_cache.get(aid, ""),
                )
            )

        vec = TfidfVectorizer(max_features=cfg.tfidf_max_features, lowercase=True)
        agent_matrix = vec.fit_transform([a.text for a in agent_infos])
        agent_matrix = normalize(agent_matrix, norm="l2", axis=1)

        return cls(
            vectorizer=vec,
            agent_matrix=agent_matrix,
            agents=agent_infos,
            config=cfg,
        )

    @classmethod
    def from_data_root(
        cls, data_root: str, *, config: GenerationConfig | None = None, parts: Iterable[str] | None = None
    ) -> "GenerativeStructuredRecommender":
        """
        Build a generator from the benchmark data directory.

        This initializes TF-IDF weights over agent descriptions and tracks the
        LLM/tool metadata needed to build the structured outputs.
        """

        bundle = collect_data(data_root, parts=list(parts) if parts is not None else None)
        tools = load_tools(data_root)
        return cls.from_bundle(bundle, tools=tools, config=config)

    def _score_query(self, query: str) -> Tuple[np.ndarray, np.ndarray]:
        q_vec = self.vectorizer.transform([query])
        q_vec = normalize(q_vec, norm="l2", axis=1)
        scores = (q_vec @ self.agent_matrix.T).toarray()[0]
        order = np.argsort(-scores)
        return scores, order

    def format_tokens(self, info: AgentGenerationInfo) -> str:
        tools = info.tool_tokens[: max(1, self.config.max_tools)]
        if not tools:
            tools = [UNK_TOOL_TOKEN]
        return " ".join(
            [info.llm_token, self.config.tool_sep_token, *tools, self.config.end_token]
        )

    def generate(
        self, query: str, *, top_k: int = 1, with_metadata: bool = False
    ) -> List[Dict[str, object] | str]:
        """Generate structured sequences for a single query."""

        scores, order = self._score_query(query)
        n = min(max(1, top_k), len(order))

        outputs: List[Dict[str, object] | str] = []
        for idx in order[:n]:
            info = self.agents[int(idx)]
            seq = self.format_tokens(info)
            if not with_metadata:
                outputs.append(seq)
                continue
            outputs.append(
                {
                    "sequence": seq,
                    "agent_id": info.aid,
                    "llm_token": info.llm_token,
                    "tool_tokens": info.tool_tokens,
                    "score": float(scores[int(idx)]),
                }
            )
        return outputs

    def build_supervised_pairs(
        self,
        *,
        rankings: Dict[str, List[str]],
        questions: Dict[str, dict],
        max_examples: int | None = None,
    ) -> List[Dict[str, str]]:
        """
        Convert the benchmark's query->ranking annotations into supervised
        training pairs for finetuning a seq2seq or instruction model.
        """

        aid_to_info = {a.aid: a for a in self.agents}
        pairs: List[Dict[str, str]] = []

        for qid, ranked in rankings.items():
            if not ranked:
                continue
            top_agent = ranked[0]
            info = aid_to_info.get(top_agent)
            if info is None:
                continue

            qtext = (questions.get(qid, {}) or {}).get("input", "")
            if not qtext:
                continue

            pairs.append(
                {
                    "qid": qid,
                    "query": qtext,
                    "target": self.format_tokens(info),
                    "agent_id": top_agent,
                }
            )
            if max_examples is not None and len(pairs) >= max_examples:
                break

        return pairs


def build_training_pairs_from_data_root(
    data_root: str,
    *,
    config: GenerationConfig | None = None,
    max_examples: int | None = None,
    parts: Iterable[str] | None = None,
) -> List[Dict[str, str]]:
    """
    Helper wrapper to build supervised pairs directly from the data root.

    This is handy when exporting the pairs to JSONL for finetuning a text
    generator: the ``target`` string already encodes the
    ``LLM_TOKEN <TOOL_SEP> TOOL_TOKEN ... <SPECIAL_END>`` template.
    """

    bundle = collect_data(data_root, parts=list(parts) if parts is not None else None)
    tools = load_tools(data_root)
    gen = GenerativeStructuredRecommender.from_bundle(bundle, tools=tools, config=config)
    return gen.build_supervised_pairs(
        rankings=bundle.all_rankings,
        questions=bundle.all_questions,
        max_examples=max_examples,
    )
