#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Table-4 Counterfactual Capability Sensitivity (offline eval)

Pipeline per query Q:
1) TwoTower retriever:
   - LLM shortlist: topK_llm from global LLMs/merge.json (via best agent score)
   - Tool pool: topK_tool_pool from global Tools/merge.json (via best agent score, using tool_query)
2) Build "Most idealized" agent A_full:
   - Default: GPT picks backbone + tools (can pick from large tool pool, not limited to top10)
   - Fallback (no OPENAI_API_KEY): heuristic = top1 LLM + top3 tools
3) Counterfactual interventions (single controlled change):
   - Remove key tool
   - Remove secondary tool
   - Add irrelevant tool
   - Add redundant tool
   - Swap backbone (rank 2–5 from LLM shortlist)
4) Call external scoring API once per query with 1+5 docs:
   POST { "query": str, "documents": [str, ...] } -> returns {"scores":[...]} or similar
5) Aggregate:
   Δs = s(Q, A_full) - s(Q, A_cf)
   Δr = rank(A_cf) - rank(A_full) within the 6-item set (A_full + 5 counterfactuals)
   Consistency = I[s_full > s_cf]
Output LaTeX rows to fill Table 4.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple, Optional

import numpy as np
import torch
from tqdm.auto import tqdm

from agent_rec.config import TFIDF_MAX_FEATURES
from agent_rec.features import (
    UNK_LLM_TOKEN,
    UNK_TOOL_TOKEN,
    build_agent_content_view,
    feature_cache_exists,
    load_feature_cache,
    load_vectorizers,
)
from agent_rec.models.two_tower import TwoTowerTFIDF
from agent_rec.run_common import bootstrap_run, shared_cache_dir
from agent_rec.data import load_tools as load_tools_json, load_LLMs as load_llms_json


# ----------------------------
# External Scoring API
# ----------------------------
def _http_post_json(url: str, payload: dict, timeout_s: float = 30.0) -> object:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url=url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            body = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        err_body = ""
        try:
            err_body = e.read().decode("utf-8", errors="replace")
        except Exception:
            pass
        raise RuntimeError(f"Scoring API HTTPError {e.code}: {err_body or str(e)}")
    except Exception as e:
        raise RuntimeError(f"Scoring API request failed: {e}")

    try:
        return json.loads(body)
    except Exception:
        return body


def _extract_scores(api_resp: object, n_docs: int) -> List[float]:
    scores: List[float] = []

    if isinstance(api_resp, dict):
        for k in ("scores", "similarities", "logits"):
            v = api_resp.get(k)
            if isinstance(v, list):
                scores = [float(x) for x in v]
                break
        if not scores and isinstance(api_resp.get("data"), dict):
            data = api_resp["data"]
            for k in ("scores", "similarities", "logits"):
                v = data.get(k)
                if isinstance(v, list):
                    scores = [float(x) for x in v]
                    break
        if not scores and isinstance(api_resp.get("results"), list):
            tmp = []
            for it in api_resp["results"]:
                if isinstance(it, dict) and "score" in it:
                    tmp.append(float(it["score"]))
            if tmp:
                scores = tmp

    elif isinstance(api_resp, list):
        if api_resp and all(isinstance(x, (int, float)) for x in api_resp):
            scores = [float(x) for x in api_resp]
        elif api_resp and all(isinstance(x, dict) for x in api_resp):
            tmp = []
            for it in api_resp:
                if "score" in it:
                    tmp.append(float(it["score"]))
            if tmp:
                scores = tmp

    if not scores:
        raise RuntimeError(f"Scoring API response has no parsable scores. resp={str(api_resp)[:500]}")

    if len(scores) < n_docs:
        scores = scores + ([-1e9] * (n_docs - len(scores)))
    elif len(scores) > n_docs:
        scores = scores[:n_docs]
    return scores


# ----------------------------
# OpenAI (optional)
# ----------------------------
IDEAL_AGENT_PROMPT = """You are designing an IDEAL agent configuration for a user query.
Goal: choose the backbone LLM and a toolset that best matches the query intent, even if some tools are NOT in the top-10 retrieved list.

Return ONLY valid JSON (no markdown), with keys:
- A_full: {{ "M": {{"name": str}}, "T": {{"tools": [str, ...]}} }}
- key_tool: str   (most query-critical tool; must be in A_full.T.tools if possible)
- secondary_tool: str (second most important; in tools if possible; can be empty)
- irrelevant_tool: str (a tool to add that is likely irrelevant / distracting; can be outside pool)
- redundant_tool: str (a tool to add that is likely redundant / clutter; can be outside pool)

User query:
{query}

LLM shortlist (ranked, with descriptions):
{llm_list_json}

Tool pool (ranked, with descriptions):
{tool_pool_json}

Constraints:
- Prefer selecting tools from Tool pool; if none fits, you may propose a tool name not in pool. Because some functionalities may be missing from the pool. When proposing new tools, use prefix "CustomTool_" to indicate they are not from the pool but need to implementation.
- Keep tools list concise (typically 3-5 tools).
"""


# ----------------------------
# GPT Tool-Query Rewriting
# ----------------------------
TOOL_QUERY_PROMPT = """You are a search query rewriter for TOOL retrieval in an agent recommender.

Given a user's natural-language request, rewrite it into a compact "tool search query" that helps match tools.
Rules:
- Output ONLY JSON (no markdown fences).
- Keys: tool_query, rationale
- tool_query should be <= 32 tokens, English preferred, include concrete actions/APIs (e.g., "weather forecast", "currency exchange rate", "send email", "calendar event create").
- Do NOT include model names. Focus on tools / APIs / operations.
- If user request contains multiple intents, keep the top 2-3 tool intents, separated by "; ".
- Keep important constraints (location, format, source) if present.

User query:
{query}
"""

def _try_import_openai():
    try:
        import openai  # type: ignore
        return openai
    except Exception:
        return None


def _gpt_chat(prompt: str, model_name: str, temperature: float = 0.0) -> str:
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")
    openai = _try_import_openai()
    if openai is None:
        raise RuntimeError("openai python package not found. Please `pip install openai`.")

    client = openai.OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        n=1,
        stream=False,
        service_tier="default",
    )
    return resp.choices[0].message.content


def rewrite_tool_query_gpt(query: str, model_name: str) -> str:
    prompt = TOOL_QUERY_PROMPT.format(query=query)
    raw = _gpt_chat(prompt, model_name=model_name, temperature=0.0)
    try:
        obj = json.loads(raw)
        tq = str(obj.get("tool_query", "")).strip()
        return tq if tq else query
    except Exception:
        return query


@dataclass
class AgentCfg:
    llm_name: str
    tools: List[str]


# ----------------------------
# TwoTower inference backbone
# ----------------------------
def _device_from_arg(device_str: str) -> torch.device:
    device = torch.device(device_str)
    if device.type.startswith("cuda") and not torch.cuda.is_available():
        print(f"[warn] CUDA 不可用，回退到 CPU (请求: {device_str}).")
        return torch.device("cpu")
    return device


def _load_checkpoint(model_path: str, device: torch.device) -> dict:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"找不到模型文件: {model_path}")
    ckpt = torch.load(model_path, map_location=device)
    if not isinstance(ckpt, dict) or "state_dict" not in ckpt:
        raise RuntimeError(f"模型文件不合法: {model_path}")
    return ckpt


def _resolve_feature_cache_dir(data_root: str, max_features: int, data_sig: str) -> str:
    return shared_cache_dir(data_root, "features", f"twotower_tfidf_{max_features}_{data_sig}")


def _build_encoder(*, ckpt: dict, feature_cache, device: torch.device) -> TwoTowerTFIDF:
    flags = ckpt.get("flags", {}) if isinstance(ckpt, dict) else {}
    dims = ckpt.get("dims", {}) if isinstance(ckpt, dict) else {}

    encoder = TwoTowerTFIDF(
        d_q=int(dims.get("d_q", feature_cache.Q.shape[1])),
        d_a=int(
            dims.get(
                "d_a",
                feature_cache.A_text_full.shape[1]
                if hasattr(feature_cache, "A_text_full")
                else feature_cache.A_model_content.shape[1],
            )
        ),
        hid=int(dims.get("hid", 256)),
        num_tools=int(dims.get("num_tools", len(feature_cache.tool_id_vocab))),
        num_llm_ids=int(len(feature_cache.llm_vocab)),
        agent_tool_idx_padded=torch.tensor(feature_cache.agent_tool_idx_padded, dtype=torch.long, device=device),
        agent_tool_mask=torch.tensor(feature_cache.agent_tool_mask, dtype=torch.float32, device=device),
        agent_llm_idx=torch.tensor(feature_cache.agent_llm_idx, dtype=torch.long, device=device),
        use_tool_id_emb=bool(flags.get("use_tool_id_emb", True)),
        use_llm_id_emb=bool(flags.get("use_llm_id_emb", False)),
        num_agents=len(feature_cache.a_ids),
        num_queries=len(feature_cache.q_ids),
        use_query_id_emb=bool(flags.get("use_query_id_emb", False)),
    ).to(device)
    encoder.load_state_dict(ckpt["state_dict"], strict=False)
    encoder.eval()
    return encoder


class TwoTowerRetriever:
    """
    Minimal retriever:
      - score agents by dot(q_emb, agent_emb)
      - lift to LLM/tool candidates by max score over agents containing that component
    """
    def __init__(self, *, data_root: str, model_path: str, device: torch.device, max_features: int) -> None:
        self.data_root = data_root
        self.device = device
        self.max_features = max_features

        boot = bootstrap_run(
            data_root=data_root,
            exp_name="eval_table4_cf",
            topk=10,
            seed=1234,
            with_tools=True,
        )
        self.bundle = boot.bundle

        ckpt = _load_checkpoint(model_path, device)
        self.ckpt = ckpt
        self.data_sig = ckpt.get("data_sig", boot.data_sig)

        cache_dir = _resolve_feature_cache_dir(data_root, max_features, self.data_sig)
        if not feature_cache_exists(cache_dir):
            raise RuntimeError(
                f"未找到特征缓存: {cache_dir}\n"
                "请确认使用相同 data_root/max_features 训练过 TwoTower TF-IDF 并生成了 feature cache。"
            )
        self.feature_cache = load_feature_cache(cache_dir)

        vecs = load_vectorizers(cache_dir)
        if vecs is None or not hasattr(vecs, "q_vec"):
            raise RuntimeError(f"未找到 TF-IDF q_vectorizer: {cache_dir}")
        self.q_vectorizer = vecs.q_vec

        flags = ckpt.get("flags", {}) if isinstance(ckpt, dict) else {}
        self.use_query_id_emb = bool(flags.get("use_query_id_emb", False))
        use_model_content_vector = bool(flags.get("use_model_content_vector", True))
        use_tool_content_vector = bool(flags.get("use_tool_content_vector", True))

        self.agent_content = build_agent_content_view(
            cache=self.feature_cache,
            use_model_content_vector=use_model_content_vector,
            use_tool_content_vector=use_tool_content_vector,
        )

        self.encoder = _build_encoder(ckpt=ckpt, feature_cache=self.feature_cache, device=device)
        self.encoder.set_agent_features(self.agent_content)
        self.agent_embeddings = self.encoder.export_agent_embeddings()  # (N_agents, d)

        self.agent_ids = list(self.feature_cache.a_ids)

        # parse agent components
        self.agent_tools: List[List[str]] = []
        self.agent_llm_names: List[str] = []
        for aid in self.agent_ids:
            agent = self.bundle.all_agents.get(aid, {}) or {}
            m = (agent.get("M") or {}) if isinstance(agent, dict) else {}
            t = (agent.get("T") or {}) if isinstance(agent, dict) else {}
            self.agent_llm_names.append((m.get("name") or m.get("id") or "").strip())
            self.agent_tools.append(list((t.get("tools") or [])))

        # global vocab
        llm_json = load_llms_json(data_root) or {}
        tool_json = load_tools_json(data_root) or {}

        self.llm_candidates: List[str] = [k for k in llm_json.keys() if k and k != UNK_LLM_TOKEN]
        self.tool_candidates: List[str] = [k for k in tool_json.keys() if k and k != UNK_TOOL_TOKEN]

        self.llm_desc_map: Dict[str, str] = {k: str((v or {}).get("description", "")).strip() for k, v in llm_json.items() if k}
        self.tool_desc_map: Dict[str, str] = {k: str((v or {}).get("description", "")).strip() for k, v in tool_json.items() if k}

        # reverse indices: component -> agent indices
        self._llm_to_agent_indices: Dict[str, List[int]] = {}
        self._tool_to_agent_indices: Dict[str, List[int]] = {}

        for i in range(len(self.agent_ids)):
            ln = (self.agent_llm_names[i] or "").strip().lower()
            if ln:
                self._llm_to_agent_indices.setdefault(ln, []).append(i)
            for tool in self.agent_tools[i]:
                if tool:
                    self._tool_to_agent_indices.setdefault(tool, []).append(i)

    def _encode_query(self, text: str) -> np.ndarray:
        vec = self.q_vectorizer.transform([text]).toarray().astype(np.float32)
        q = torch.from_numpy(vec).to(self.device)
        q_idx = torch.zeros(1, dtype=torch.long, device=self.device) if self.use_query_id_emb else None
        with torch.no_grad():
            qe = self.encoder.encode_q(q, q_idx=q_idx).cpu().numpy()  # (1, d)
        return qe

    def score_all_agents(self, text: str) -> np.ndarray:
        qe = self._encode_query(text)
        return np.dot(qe, self.agent_embeddings.T).reshape(-1)  # (N_agents,)

    def rank_llms(self, agent_scores: np.ndarray, topk: int) -> List[Dict[str, object]]:
        out = []
        for llm in self.llm_candidates:
            idxs = self._llm_to_agent_indices.get(llm.strip().lower(), [])
            if not idxs:
                continue
            best_i = max(idxs, key=lambda i: float(agent_scores[i]))
            out.append({"name": llm, "score": float(agent_scores[best_i]), "description": self.llm_desc_map.get(llm, "")})
        out.sort(key=lambda x: -float(x["score"]))
        return out[:topk]

    def rank_tools(self, agent_scores: np.ndarray, topk: int) -> List[Dict[str, object]]:
        out = []
        for tool in self.tool_candidates:
            idxs = self._tool_to_agent_indices.get(tool, [])
            if not idxs:
                continue
            best_i = max(idxs, key=lambda i: float(agent_scores[i]))
            out.append({"name": tool, "score": float(agent_scores[best_i]), "description": self.tool_desc_map.get(tool, "")})
        out.sort(key=lambda x: -float(x["score"]))
        return out[:topk]

    def pick_low_rank_tool(self, agent_scores: np.ndarray, avoid: set, tail_k: int = 200) -> Optional[str]:
        """
        Choose a low-ranked tool as "irrelevant" fallback.
        """
        ranked = self.rank_tools(agent_scores, topk=min(tail_k, max(50, tail_k)))
        if not ranked:
            return None
        # take from the tail of this list
        tail = list(reversed(ranked))
        for it in tail:
            name = it["name"]
            if name not in avoid:
                return name
        return None

    def tool_desc(self, name: str) -> str:
        return (self.tool_desc_map.get(name, "") or "").strip()

    def llm_desc(self, name: str) -> str:
        return (self.llm_desc_map.get(name, "") or "").strip()


# ----------------------------
# Doc building for scoring API
# ----------------------------
def agent_to_document(agent: AgentCfg, llm_desc: str, tool_desc_map: Dict[str, str]) -> str:
    tools = [t for t in agent.tools if t]
    parts = [f"Backbone LLM: {agent.llm_name}"]
    if llm_desc:
        parts.append(f"LLM description: {llm_desc}")
    if tools:
        parts.append("Tools:")
        for t in tools:
            td = (tool_desc_map.get(t, "") or "").strip()
            parts.append(f"- {t}: {td}" if td else f"- {t}")
    else:
        parts.append("Tools: (none)")
    return "\n".join(parts).strip()


# ----------------------------
# Ideal agent generation
# ----------------------------
def parse_ideal_agent(raw: str) -> Tuple[Optional[AgentCfg], dict]:
    try:
        obj = json.loads(raw)
    except Exception:
        return None, {"error": "json_parse_failed", "raw": raw[:500]}

    af = obj.get("A_full") or {}
    m = (af.get("M") or {}) if isinstance(af, dict) else {}
    t = (af.get("T") or {}) if isinstance(af, dict) else {}
    llm = (m.get("name") or "").strip() if isinstance(m, dict) else ""
    tools = (t.get("tools") or []) if isinstance(t, dict) else []
    if not isinstance(tools, list):
        tools = []
    tools = [str(x).strip() for x in tools if str(x).strip()]

    if not llm:
        return None, {"error": "missing_llm", "obj": obj}

    meta = {
        "key_tool": str(obj.get("key_tool", "") or "").strip(),
        "secondary_tool": str(obj.get("secondary_tool", "") or "").strip(),
        "irrelevant_tool": str(obj.get("irrelevant_tool", "") or "").strip(),
        "redundant_tool": str(obj.get("redundant_tool", "") or "").strip(),
        "obj": obj,
    }
    return AgentCfg(llm_name=llm, tools=tools), meta


def build_ideal_agent(
    query: str,
    llm_list: List[Dict[str, object]],
    tool_pool: List[Dict[str, object]],
    *,
    use_gpt: bool,
    gpt_model: str,
) -> Tuple[AgentCfg, dict, Optional[str]]:
    """
    Returns (A_full, meta, gpt_error)
    meta includes key_tool/secondary_tool/irrelevant_tool/redundant_tool.
    """
    if use_gpt:
        prompt = IDEAL_AGENT_PROMPT.format(
            query=query,
            llm_list_json=json.dumps(llm_list, ensure_ascii=False, indent=2),
            tool_pool_json=json.dumps(tool_pool, ensure_ascii=False, indent=2),
        )
        try:
            raw = _gpt_chat(prompt, model_name=gpt_model, temperature=0.0)
            agent, meta = parse_ideal_agent(raw)
            if agent is not None:
                return agent, meta, None
            return heuristic_ideal_agent(llm_list, tool_pool), meta, "GPT returned invalid JSON; fallback to heuristic."
        except Exception as e:
            return heuristic_ideal_agent(llm_list, tool_pool), {}, f"GPT failed; fallback to heuristic. err={e}"

    return heuristic_ideal_agent(llm_list, tool_pool), {}, None


def heuristic_ideal_agent(llm_list: List[Dict[str, object]], tool_pool: List[Dict[str, object]]) -> AgentCfg:
    llm = (llm_list[0]["name"] if llm_list else "UNKNOWN_LLM")
    tools = []
    for it in tool_pool[:3]:
        tools.append(str(it["name"]))
    return AgentCfg(llm_name=str(llm), tools=tools)


# ----------------------------
# Counterfactual construction
# ----------------------------
def unique_tools(tools: Iterable[str]) -> List[str]:
    seen = set()
    out = []
    for t in tools:
        t = (t or "").strip()
        if t and t not in seen:
            seen.add(t)
            out.append(t)
    return out


def remove_tool(agent: AgentCfg, tool_name: str) -> AgentCfg:
    tools = [t for t in agent.tools if t != tool_name]
    return AgentCfg(llm_name=agent.llm_name, tools=tools)


def add_tool(agent: AgentCfg, tool_name: str) -> AgentCfg:
    tools = unique_tools(agent.tools + [tool_name])
    return AgentCfg(llm_name=agent.llm_name, tools=tools)


def swap_llm(agent: AgentCfg, new_llm: str) -> AgentCfg:
    return AgentCfg(llm_name=new_llm, tools=list(agent.tools))


# ----------------------------
# Metrics aggregation
# ----------------------------
@dataclass
class Agg:
    ds: List[float]
    dr: List[float]
    ok: List[int]  # consistency 0/1

    def add(self, ds: float, dr: float, ok: int) -> None:
        self.ds.append(float(ds))
        self.dr.append(float(dr))
        self.ok.append(int(ok))

    def summary(self) -> Tuple[float, float, float]:
        m_ds = float(np.mean(self.ds)) if self.ds else 0.0
        m_dr = float(np.mean(self.dr)) if self.dr else 0.0
        m_ok = float(np.mean(self.ok)) * 100.0 if self.ok else 0.0
        return m_ds, m_dr, m_ok


# ----------------------------
# Load jsonl
# ----------------------------
def load_questions(jsonl_path: str) -> List[str]:
    qs = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            q = (obj.get("question") or "").strip()
            if q:
                qs.append(q)
    return qs


# ----------------------------
# Main
# ----------------------------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser("Table-4 counterfactual capability sensitivity")
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--model_path", type=str, required=True)
    ap.add_argument("--questions_jsonl", type=str, required=True)
    ap.add_argument("--scoring_url", type=str, default="http://127.0.0.1:8501/compute_scores")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--max_features", type=int, default=TFIDF_MAX_FEATURES)

    ap.add_argument("--N", type=int, default=100, help="number of queries sampled")
    ap.add_argument("--seed", type=int, default=1234)

    ap.add_argument("--topk_llm", type=int, default=10)
    ap.add_argument("--topk_tool_pool", type=int, default=20)

    ap.add_argument("--use_gpt", action="store_true", help="use GPT to build ideal A_full (needs OPENAI_API_KEY)")
    ap.add_argument("--gpt_model", type=str, default="gpt-5")

    ap.add_argument("--use_gpt_tool_query", action="store_true", help="use GPT tool-query rewriting (optional)")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    device = _device_from_arg(args.device)
    retriever = TwoTowerRetriever(
        data_root=args.data_root,
        model_path=args.model_path,
        device=device,
        max_features=args.max_features,
    )

    questions = load_questions(args.questions_jsonl)
    if not questions:
        raise RuntimeError(f"No valid 'question' found in {args.questions_jsonl}")

    if len(questions) >= args.N:
        sampled = random.sample(questions, args.N)
    else:
        sampled = questions

    aggs = {
        "remove_key": Agg([], [], []),
        "remove_secondary": Agg([], [], []),
        "add_irrelevant": Agg([], [], []),
        "add_redundant": Agg([], [], []),
        "swap_backbone": Agg([], [], []),
    }

    gpt_fail_cnt = 0

    for q in tqdm(sampled, desc="Table4 counterfactual", total=len(sampled)):
        # (1) LLM shortlist uses original query
        agent_scores = retriever.score_all_agents(q)
        llm_list = retriever.rank_llms(agent_scores, topk=args.topk_llm)

        # (2) Tool pool uses tool_query (optionally rewritten)
        tool_query = q
        if args.use_gpt_tool_query and args.use_gpt:
            try:
                tool_query = rewrite_tool_query_gpt(q, model_name=args.gpt_model)
            except Exception:
                tool_query = q

        tool_agent_scores = retriever.score_all_agents(tool_query)
        tool_pool = retriever.rank_tools(tool_agent_scores, topk=args.topk_tool_pool)

        # (3) Build A_full
        A_full, meta, gpt_err = build_ideal_agent(
            q, llm_list, tool_pool, use_gpt=args.use_gpt, gpt_model=args.gpt_model
        )
        print("Ideal agent for query:\n",A_full)
        
        if gpt_err:
            gpt_fail_cnt += 1

        A_full.tools = unique_tools(A_full.tools)

        # determine key/secondary
        key_tool = (meta.get("key_tool") or "").strip()
        secondary_tool = (meta.get("secondary_tool") or "").strip()

        if not key_tool and A_full.tools:
            key_tool = A_full.tools[0]
        if (not secondary_tool or secondary_tool == key_tool) and len(A_full.tools) >= 2:
            secondary_tool = A_full.tools[1]

        avoid = set(A_full.tools)

        # irrelevant / redundant tool picks
        irr_tool = (meta.get("irrelevant_tool") or "").strip()
        red_tool = (meta.get("redundant_tool") or "").strip()

        if not irr_tool:
            irr_tool = retriever.pick_low_rank_tool(tool_agent_scores, avoid=avoid) or "IrrelevantTool"
        if not red_tool:
            # fallback: pick a high-ranked tool not already included; else duplicate key_tool
            red_tool = ""
            for it in tool_pool[:20]:
                cand = str(it["name"])
                if cand and cand not in avoid:
                    red_tool = cand
                    break
            if not red_tool:
                red_tool = key_tool or "RedundantTool"

        # swap backbone: choose rank 2-5
        swap_llm_name = ""
        for it in llm_list[1:5]:
            cand = str(it["name"]).strip()
            if cand and cand.lower() != A_full.llm_name.strip().lower():
                swap_llm_name = cand
                break

        # (4) Build counterfactual variants
        cf_remove_key = remove_tool(A_full, key_tool) if key_tool else A_full
        cf_remove_secondary = remove_tool(A_full, secondary_tool) if secondary_tool else A_full
        cf_add_irrelevant = add_tool(A_full, irr_tool) if irr_tool else A_full
        cf_add_redundant = add_tool(A_full, red_tool) if red_tool else A_full
        cf_swap = swap_llm(A_full, swap_llm_name) if swap_llm_name else A_full

        # (5) Score via external API (one call per query)
        variants = [
            ("full", A_full),
            ("remove_key", cf_remove_key),
            ("remove_secondary", cf_remove_secondary),
            ("add_irrelevant", cf_add_irrelevant),
            ("add_redundant", cf_add_redundant),
            ("swap_backbone", cf_swap),
        ]

        docs = []
        for _, a in variants:
            docs.append(agent_to_document(a, retriever.llm_desc(a.llm_name), retriever.tool_desc_map))

        api_resp = _http_post_json(args.scoring_url, {"query": q, "documents": docs}, timeout_s=35.0)
        scores = _extract_scores(api_resp, n_docs=len(docs))
        scores_np = np.array(scores, dtype=np.float32)

        # ranks within the 6-item set (higher score = better rank)
        order = np.argsort(-scores_np)
        ranks = np.empty_like(order)
        ranks[order] = np.arange(1, len(scores_np) + 1)  # rank starts at 1

        s_full = float(scores_np[0])
        r_full = int(ranks[0])

        for i, (name, _) in enumerate(variants[1:], start=1):
            s_cf = float(scores_np[i])
            r_cf = int(ranks[i])
            ds = s_full - s_cf
            dr = float(r_cf - r_full)
            ok = 1 if (s_full > s_cf) else 0
            aggs[name].add(ds, dr, ok)

    # ----------------------------
    # Print Table 4 (LaTeX rows)
    # ----------------------------
    def fmt_row(label: str, key: str) -> str:
        m_ds, m_dr, m_ok = aggs[key].summary()
        return f"{label} & {m_ds:.4f} & {m_dr:.2f} & {m_ok:.1f}\\% \\\\"

    print("\n================ Table 4 rows (LaTeX) ================\n")
    print(fmt_row("Remove key tool", "remove_key"))
    print(fmt_row("Remove secondary tool", "remove_secondary"))
    print(fmt_row("Add irrelevant tool", "add_irrelevant"))
    print(fmt_row("Add redundant tool", "add_redundant"))
    print(fmt_row("Swap backbone (rank 2--5)", "swap_backbone"))

    print("\n------------------------------------------------------")
    print(f"N_queries={len(sampled)} | use_gpt={args.use_gpt} | gpt_fail_fallback={gpt_fail_cnt}")
    print("======================================================\n")


if __name__ == "__main__":
    main()
