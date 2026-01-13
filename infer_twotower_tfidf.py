#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Two-Tower (TF-IDF) inference and Flask UI (4-column pipeline).

Col 1: rerank over ALL LLM candidates (global llm_vocab / LLMs/merge.json keys)
Col 2: rerank over ALL Tool candidates (global tool_vocab / Tools/merge.json keys)
Col 3: compose Suggestion from Col1+Col2 -> GPT generate top-10 agent configs -> parse -> show
Col 4: expand Col3 configs -> build combo texts -> call external scoring API (/compute_scores) ->
       sort by returned scores -> Top10

External scoring API contract (your service):
POST { "query": str, "documents": [str, ...] } -> returns JSON containing scores.

You can run your scoring service separately:
  app.run(host="0.0.0.0", port=8501)  # exposes /compute_scores

This script runs the UI service (default port 8000), and calls the scoring service via HTTP.
"""

from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.request
from functools import lru_cache
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from flask import Flask, jsonify, render_template_string, request

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
{{query}}
"""

def _try_import_openai():
    try:
        import openai  # type: ignore
        return openai
    except Exception:
        return None

def gpt_qa_not_stream(prompt: str, model_name: str, temperature: float = 0.0) -> str:
    """
    Uses OpenAI python SDK if available and OPENAI_API_KEY is set.
    """
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. Column-3 GPT generation disabled.")
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

def rewrite_tool_query(query: str, model_name: str = "gpt-5-nano") -> Tuple[str, str, str]:
    """
    Returns (tool_query, prompt, raw_response).
    """
    prompt = TOOL_QUERY_PROMPT.replace("{{query}}", query)
    raw = gpt_qa_not_stream(prompt, model_name=model_name, temperature=0.0)

    try:
        obj = json.loads(raw)
        tq = str(obj.get("tool_query", "")).strip()
        if not tq:
            tq = query.strip()
        return tq, prompt, raw
    except Exception:
        tq = raw.strip()
        if not tq or len(tq) > 300:
            tq = query.strip()
        return tq, prompt, raw


# ----------------------------
# GPT agent generation prompt
# ----------------------------
GENERATE_AGENTS_PROMPT = '''### According to [Suggestion], construct agents suitable for answering the user's [question].

### Question
{{question}}

### Suggestion
{{suggestion}}

### Agent Structure Requirements
- **M (Model)**: Specify only backbone LLM
- **T (Tools)**: List of tool names as an array

### Deliverables
1. Generate top 10 different agent configurations with varying Backbone LLM/Toolkit combinations
2. Rank agents by suitability for solving the question
3. Order from most suitable (top) to least suitable (bottom)
4. In [Suggestion], you are provided with suitable components , including Backbone LLMs  or  Tools.
   Please design agents that combine both Backbone LLMs and Toolkits. Do not fabricate, can only use probided components.

### Expected Output Format
Return json formal, but do not enclose them in ``` symbols. For example:

[{
  "M": {"name": "Backbone LLM Name"},
  "T": {"tools": ["tool1", "tool2", "tool3"]},
  "C": {}
},
{
  "M": {"name": "Backbone LLM Name"},
  "T": {"tools": ["tool1", "tool2", "tool3"]}
},
...
]
'''

def generate_agents_for_question(question: str, suggestion: str) -> Tuple[List[Dict[str, object]], str, str]:
    """
    Returns (agents, prompt, raw_response).
    """
    model = "gpt-5-nano"
    prompt = GENERATE_AGENTS_PROMPT.replace("{{question}}", question).replace("{{suggestion}}", suggestion)
    raw = gpt_qa_not_stream(prompt, model_name=model)

    try:
        agents = json.loads(raw)
        if not isinstance(agents, list):
            return [], prompt, raw

        cleaned: List[Dict[str, object]] = []
        for a in agents:
            if not isinstance(a, dict):
                continue
            m = a.get("M") or {}
            t = a.get("T") or {}
            if not isinstance(m, dict) or not isinstance(t, dict):
                continue
            m_name = (m.get("name") or "").strip()
            tools = t.get("tools") or []
            if not isinstance(tools, list):
                tools = []
            tools = [str(x).strip() for x in tools if str(x).strip()]
            if m_name:
                cleaned.append({"M": {"name": m_name}, "T": {"tools": tools}, "C": a.get("C") or {}})
        return cleaned[:10], prompt, raw
    except Exception:
        return [], prompt, raw


# ----------------------------
# External Scoring API (Col4)
# ----------------------------
def _http_post_json(url: str, payload: dict, timeout_s: float = 20.0) -> object:
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
        # allow raw text fallback
        return body

def _extract_scores(api_resp: object, n_docs: int) -> List[float]:
    """
    Try to parse scores from various possible response shapes.
    Expected/handled examples:
      {"scores":[...]}
      {"similarities":[...]}
      {"data":{"scores":[...]}}
      [{"score":0.1}, {"score":0.2}, ...]
      [...]
    """
    scores: List[float] = []

    if isinstance(api_resp, dict):
        # direct keys
        for k in ("scores", "similarities", "logits"):
            v = api_resp.get(k)
            if isinstance(v, list):
                scores = [float(x) for x in v]
                break

        # nested
        if not scores and isinstance(api_resp.get("data"), dict):
            data = api_resp["data"]
            for k in ("scores", "similarities", "logits"):
                v = data.get(k)
                if isinstance(v, list):
                    scores = [float(x) for x in v]
                    break

        # list of dicts under "results"
        if not scores and isinstance(api_resp.get("results"), list):
            arr = api_resp["results"]
            tmp = []
            for it in arr:
                if isinstance(it, dict) and ("score" in it):
                    tmp.append(float(it["score"]))
            if tmp:
                scores = tmp

    elif isinstance(api_resp, list):
        # list of floats or list of dicts
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
        raise RuntimeError(f"Scoring API response does not contain parsable scores. resp={str(api_resp)[:500]}")

    # align length
    if len(scores) < n_docs:
        # allow shorter; pad with -inf
        scores = scores + ([-1e9] * (n_docs - len(scores)))
    elif len(scores) > n_docs:
        scores = scores[:n_docs]
    return scores


# ----------------------------
# UI template
# ----------------------------
HTML_TEMPLATE = """
<!doctype html>
<html lang="zh">
<head>
  <meta charset="utf-8" />
  <title>TwoTower TF-IDF Agent 推荐</title>
  <style>
    body { font-family: "Helvetica Neue", Arial, sans-serif; background: #f5f6fa; }
    .container { max-width: 1400px; margin: 60px auto; background: #fff; padding: 32px; border-radius: 12px; box-shadow: 0 4px 30px rgba(0,0,0,0.08); }
    h1 { text-align: center; margin-bottom: 12px; }
    p.subtitle { text-align: center; color: #555; margin-top: 0; }
    form { display: flex; gap: 12px; justify-content: center; align-items: center; margin-bottom: 24px; }
    input[type=text] { width: 100%; max-width: 760px; padding: 14px 18px; font-size: 16px; border: 1px solid #dcdde1; border-radius: 10px; box-sizing: border-box; }
    button { padding: 14px 22px; font-size: 16px; background: #2d8cf0; color: #fff; border: none; border-radius: 10px; cursor: pointer; }
    button:hover { background: #1d7cd9; }
    .grid { display: grid; grid-template-columns: repeat(4, minmax(260px, 1fr)); gap: 16px; }
    .panel { border: 1px solid #ecf0f1; border-radius: 12px; padding: 14px 14px 4px; background: #fbfcfe; box-shadow: 0 1px 12px rgba(0,0,0,0.03); }
    .panel h2 { margin: 0 0 8px; font-size: 17px; display: flex; align-items: center; gap: 6px; }
    .panel small { color: #777; font-weight: 400; }
    .results ol { padding-left: 18px; margin: 0; }
    .agent-card { border: 1px solid #ecf0f1; border-radius: 10px; padding: 10px 12px; margin-bottom: 10px; background: #fff; }
    .agent-header { display: flex; justify-content: space-between; align-items: center; gap: 8px; }
    .agent-title { font-weight: 600; font-size: 16px; }
    .score { color: #2d8cf0; font-weight: 600; font-size: 14px; }
    .meta { color: #444; margin-top: 4px; line-height: 1.5; font-size: 13px; }
    .tools { margin-top: 6px; color: #555; font-size: 13px; }
    .candidate { border-left: 3px solid #2d8cf0; padding-left: 10px; margin-bottom: 10px; }
    .error { color: #c0392b; text-align: center; margin-bottom: 12px; }
    pre { white-space: pre-wrap; word-break: break-word; background: #f7f8fb; border: 1px solid #ecf0f1; padding: 10px; border-radius: 10px; font-size: 12px; color: #333; }
  </style>
</head>
<body>
  <div class="container">
    <h1>TwoTower TF-IDF 推荐（4列流水线）</h1>
    <p class="subtitle">①LLM rerank ②Tool rerank ③GPT生成候选 ④外部API打分Top10</p>

    <form method="post">
      <input type="text" name="query" placeholder="请输入查询，例如：写一个天气查询助手" value="{{ query|e }}" required />
      <button type="submit">推荐</button>
    </form>

    {% if error %}<div class="error">{{ error }}</div>{% endif %}

    {% if tool_query %}
    <div class="panel" style="margin-bottom:16px;">
        <h2>Tool Query（仅用于② Tools检索）</h2>
        {% if tool_query_error %}<div class="meta" style="color:#c0392b;">{{ tool_query_error }}</div>{% endif %}
        <pre>{{ tool_query }}</pre>
    </div>
    {% endif %}

    {% if suggestion_text %}
      <div class="panel" style="margin-bottom:16px;">
        <h2>Suggestion（由第1/2列拼接）</h2>
        <pre>{{ suggestion_text }}</pre>
      </div>
    {% endif %}

    {% if has_results %}
      <div class="results grid">

        <div class="panel">
        <h2>① BackboneLLMs（Suggestion） <small>Top {{ llm_recs|length }}</small></h2>
        <ol>
        {% for item in llm_recs %}
        <li class="agent-card">
            <div class="agent-header">
              <div class="agent-title">{{ item.llm_name }}</div>
              <div class="score">score={{ '%.4f'|format(item.score) }}</div>
            </div>
            {% if item.desc %}<div class="meta">{{ item.desc }}</div>{% endif %}
        </li>
        {% endfor %}
        </ol>
        </div>

        <div class="panel">
        <h2>② Tools（Suggestion） <small>Top {{ tool_recs|length }}</small></h2>
        <ol>
        {% for item in tool_recs %}
        <li class="agent-card">
            <div class="agent-header">
              <div class="agent-title">{{ item.tool_name }}</div>
              <div class="score">score={{ '%.4f'|format(item.score) }}</div>
            </div>
            {% if item.desc %}<div class="meta">{{ item.desc }}</div>{% endif %}
        </li>
        {% endfor %}
        </ol>
        </div>

        <div class="panel">
          <h2>③ GPT 生成的 Agents <small>Top {{ generated_agents|length }}</small></h2>
          {% if gpt_error %}
            <div class="meta" style="color:#c0392b;">{{ gpt_error }}</div>
          {% endif %}
          <ol>
            {% for a in generated_agents %}
            <li class="agent-card candidate">
              <div class="agent-title">{{ a.M.name }}</div>
              <div class="meta">Tools: {{ a.T.tools | join(', ') }}</div>
            </li>
            {% endfor %}
          </ol>
        </div>

        <div class="panel">
          <h2>④ 外部API打分 Top10 <small>{{ col4_results|length }}</small></h2>
          <div class="meta">Candidate combos: {{ combo_pool_size }} | Scoring API: {{ scoring_url }}</div>
          {% if col4_error %}
            <div class="meta" style="color:#c0392b;">{{ col4_error }}</div>
          {% endif %}
          <ol>
            {% for r in col4_results %}
            <li class="agent-card">
              <div class="agent-header">
                <div class="agent-title">{{ r.display_name }}</div>
                <div class="score">score={{ '%.4f'|format(r.score) }}</div>
              </div>
              {% if r.model_desc %}<div class="meta">模型描述: {{ r.model_desc }}</div>{% endif %}
              {% if r.tools %}
                <div class="tools"><strong>工具:</strong> {{ r.tools | join(', ') }}</div>
              {% endif %}
            </li>
            {% endfor %}
          </ol>
        </div>

      </div>
    {% endif %}
    
      <div class="panel" style="margin-top:18px;">
    <h2>自定义 Agent 预览对比 <small>手动编辑两个 agent，点击 Compare 查看 TwoTower 分数</small></h2>

    <div style="display:flex; gap:14px; flex-wrap:wrap; align-items:flex-start;">
      <!-- Left: Query -->
      <div style="flex: 1 1 320px;">
        <div class="meta" style="font-weight:600; margin-bottom:6px;">Query</div>
        <input id="cmp_query" type="text" value="{{ query|e }}" placeholder="输入 query（默认使用上方查询）"
               style="width:100%; padding:12px 14px; border:1px solid #dcdde1; border-radius:10px;" />
        <div class="meta" style="margin-top:8px; color:#777;">
          选项来自本次检索结果：①Top10 LLM、②Top10 Tools
        </div>
      </div>

      <!-- Agent A -->
      <div style="flex: 1 1 380px; border:1px solid #ecf0f1; border-radius:12px; padding:12px; background:#fff;">
        <div class="agent-header" style="margin-bottom:8px;">
          <div class="agent-title">Agent A</div>
          <div class="score" id="scoreA">score=—</div>
        </div>

        <div class="meta" style="font-weight:600; margin-bottom:6px;">LLM</div>
        <select id="agentA_llm" style="width:100%; padding:10px; border-radius:10px; border:1px solid #dcdde1;"></select>

        <div class="meta" style="font-weight:600; margin:10px 0 6px;">Tools (多选)</div>
        <select id="agentA_tools" multiple size="10"
                style="width:100%; padding:10px; border-radius:10px; border:1px solid #dcdde1;"></select>

        <div class="meta" style="margin-top:8px; color:#777;">
          提示：按住 Ctrl/⌘ 可多选；也可不选 tool
        </div>
      </div>

      <!-- Agent B -->
      <div style="flex: 1 1 380px; border:1px solid #ecf0f1; border-radius:12px; padding:12px; background:#fff;">
        <div class="agent-header" style="margin-bottom:8px;">
          <div class="agent-title">Agent B</div>
          <div class="score" id="scoreB">score=—</div>
        </div>

        <div class="meta" style="font-weight:600; margin-bottom:6px;">LLM</div>
        <select id="agentB_llm" style="width:100%; padding:10px; border-radius:10px; border:1px solid #dcdde1;"></select>

        <div class="meta" style="font-weight:600; margin:10px 0 6px;">Tools (多选)</div>
        <select id="agentB_tools" multiple size="10"
                style="width:100%; padding:10px; border-radius:10px; border:1px solid #dcdde1;"></select>

        <div class="meta" style="margin-top:8px; color:#777;">
          提示：按住 Ctrl/⌘ 可多选；也可不选 tool
        </div>
      </div>
    </div>

    <div style="display:flex; gap:10px; align-items:center; margin-top:12px; flex-wrap:wrap;">
      <button type="button" id="btnCompare">Compare</button>
      <div class="meta" id="cmpStatus" style="color:#777;"></div>
    </div>

    <div id="cmpResult" style="margin-top:12px; display:none;">
      <pre id="cmpResultPre"></pre>
    </div>
  </div>

  <script>
    function getSelectedValues(selectEl) {
      const vals = [];
      for (const opt of selectEl.options) {
        if (opt.selected) vals.push(opt.value);
      }
      return vals;
    }

    function fillSelect(selectEl, options, placeholder) {
      selectEl.innerHTML = "";
      const ph = document.createElement("option");
      ph.value = "";
      ph.textContent = placeholder || "请选择";
      ph.disabled = true;
      ph.selected = true;
      selectEl.appendChild(ph);

      for (const v of options) {
        const opt = document.createElement("option");
        opt.value = v;
        opt.textContent = v;
        selectEl.appendChild(opt);
      }
    }

    // 从页面已有的①②结果抓 top10 名称（避免你再传参）
    function extractTopCandidates() {
      const llms = [];
      const tools = [];

      // ① LLM：取每个卡片的 .agent-title 文本
      const llmPanel = document.querySelectorAll(".results .panel")[0];
      if (llmPanel) {
        const titles = llmPanel.querySelectorAll(".agent-card .agent-title");
        titles.forEach(el => llms.push(el.textContent.trim()));
      }

      // ② Tools：同理
      const toolPanel = document.querySelectorAll(".results .panel")[1];
      if (toolPanel) {
        const titles = toolPanel.querySelectorAll(".agent-card .agent-title");
        titles.forEach(el => tools.push(el.textContent.trim()));
      }

      return { llms: llms.slice(0, 10), tools: tools.slice(0, 10) };
    }

    function initCompareUI() {
      const { llms, tools } = extractTopCandidates();

      const aLLM = document.getElementById("agentA_llm");
      const bLLM = document.getElementById("agentB_llm");
      const aTools = document.getElementById("agentA_tools");
      const bTools = document.getElementById("agentB_tools");

      // 如果还没跑过推荐（results为空），就不给选项
      if (llms.length === 0 && tools.length === 0) {
        document.getElementById("cmpStatus").textContent = "请先在上方提交一次 query，生成①②候选后再对比。";
        return;
      }

      fillSelect(aLLM, llms, "选择 LLM (Top10)");
      fillSelect(bLLM, llms, "选择 LLM (Top10)");

      // Tools 多选不需要 placeholder
      aTools.innerHTML = "";
      bTools.innerHTML = "";
      tools.forEach(t => {
        const oa = document.createElement("option");
        oa.value = t; oa.textContent = t;
        aTools.appendChild(oa);

        const ob = document.createElement("option");
        ob.value = t; ob.textContent = t;
        bTools.appendChild(ob);
      });

      // 预选：A选第1个LLM，B选第2个LLM（如果存在）
      if (llms.length > 0) aLLM.selectedIndex = 1;
      if (llms.length > 1) bLLM.selectedIndex = 2;

      // 预选：A选前2个tool，B选后2个tool（如果存在）
      for (let i = 0; i < aTools.options.length; i++) {
        if (i < 2) aTools.options[i].selected = true;
      }
      for (let i = 0; i < bTools.options.length; i++) {
        if (i >= Math.max(0, bTools.options.length - 2)) bTools.options[i].selected = true;
      }
    }

    async function doCompare() {
      const statusEl = document.getElementById("cmpStatus");
      const resultBox = document.getElementById("cmpResult");
      const resultPre = document.getElementById("cmpResultPre");
      const scoreAEl = document.getElementById("scoreA");
      const scoreBEl = document.getElementById("scoreB");

      const query = (document.getElementById("cmp_query").value || "").trim();
      const llmA = document.getElementById("agentA_llm").value;
      const llmB = document.getElementById("agentB_llm").value;
      const toolsA = getSelectedValues(document.getElementById("agentA_tools"));
      const toolsB = getSelectedValues(document.getElementById("agentB_tools"));

      statusEl.textContent = "Comparing...";
      resultBox.style.display = "none";
      scoreAEl.textContent = "score=—";
      scoreBEl.textContent = "score=—";

      try {
        const resp = await fetch("/api/compare", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            query,
            agentA: { llm_name: llmA, tools: toolsA },
            agentB: { llm_name: llmB, tools: toolsB }
          })
        });

        const data = await resp.json();
        if (!resp.ok) {
          throw new Error(data.error || "compare failed");
        }

        scoreAEl.textContent = "score=" + data.agentA.score.toFixed(4);
        scoreBEl.textContent = "score=" + data.agentB.score.toFixed(4);

        const msg =
`Agent A:
  LLM: ${data.agentA.llm_name}
  Tools: ${(data.agentA.tools || []).join(", ")}
  Score: ${data.agentA.score.toFixed(6)}

Agent B:
  LLM: ${data.agentB.llm_name}
  Tools: ${(data.agentB.tools || []).join(", ")}
  Score: ${data.agentB.score.toFixed(6)}

Delta (A - B): ${data.delta.toFixed(6)}
`;
        resultPre.textContent = msg;
        resultBox.style.display = "block";
        statusEl.textContent = "Done.";
      } catch (e) {
        statusEl.textContent = "Error: " + (e.message || String(e));
      }
    }

    document.addEventListener("DOMContentLoaded", () => {
      initCompareUI();
      const btn = document.getElementById("btnCompare");
      if (btn) btn.addEventListener("click", doCompare);
    });
  </script>


  </div>
</body>
</html>
"""


# ----------------------------
# core inference
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


class TwoTowerInference:
    def __init__(
        self,
        *,
        data_root: str,
        model_path: str,
        device: torch.device,
        max_features: int,
        topk: int,
        scoring_url: str,
    ) -> None:
        self.data_root = data_root
        self.model_path = model_path
        self.device = device
        self.max_features = max_features
        self.topk = topk
        self.scoring_url = scoring_url

        boot = bootstrap_run(
            data_root=data_root,
            exp_name="infer_twotower_tfidf",
            topk=topk,
            seed=1234,
            with_tools=True,
        )
        self.bundle = boot.bundle

        ckpt = _load_checkpoint(model_path, device)
        self.ckpt = ckpt
        self.ckpt_data_sig = ckpt.get("data_sig", boot.data_sig)

        cache_dir = _resolve_feature_cache_dir(data_root, max_features, self.ckpt_data_sig)
        if not feature_cache_exists(cache_dir):
            raise RuntimeError(
                f"未找到特征缓存: {cache_dir}\n"
                "请确认使用相同的数据根目录与max_features训练过TwoTower TF-IDF模型。"
            )
        self.feature_cache = load_feature_cache(cache_dir)

        vecs = load_vectorizers(cache_dir)
        if vecs is None or not hasattr(vecs, "q_vec"):
            raise RuntimeError(
                f"未找到 TF-IDF q_vectorizer: {cache_dir}\n"
                "请确认训练时已保存 q_vectorizer.pkl."
            )
        self.q_vectorizer = vecs.q_vec

        # build agent content view (for exporting agent embeddings)
        flags = ckpt.get("flags", {}) if isinstance(ckpt, dict) else {}
        use_model_content_vector = bool(flags.get("use_model_content_vector", True))
        use_tool_content_vector = bool(flags.get("use_tool_content_vector", True))

        self.agent_content = build_agent_content_view(
            cache=self.feature_cache,
            use_model_content_vector=use_model_content_vector,
            use_tool_content_vector=use_tool_content_vector,
        )

        self.encoder = _build_encoder(ckpt=ckpt, feature_cache=self.feature_cache, device=self.device)
        self.encoder.set_agent_features(self.agent_content)
        self.agent_embeddings = self.encoder.export_agent_embeddings()

        self.agent_ids = list(self.feature_cache.a_ids)
        self.agent_id_to_index = {aid: i for i, aid in enumerate(self.agent_ids)}

        # per-agent parsed info (for candidate reverse index)
        self.agent_llm_ids = list(getattr(self.feature_cache, "llm_ids", []))
        self.agent_tools: List[List[str]] = []
        self.agent_llm_names: List[str] = []
        for aid in self.agent_ids:
            agent = self.bundle.all_agents.get(aid, {}) or {}
            m = (agent.get("M") or {}) if isinstance(agent, dict) else {}
            t = (agent.get("T") or {}) if isinstance(agent, dict) else {}
            self.agent_llm_names.append((m.get("name") or m.get("id") or "").strip())
            self.agent_tools.append(list((t.get("tools") or [])))

        # GLOBAL candidate sets from merge.json (key=name)
        llm_json = load_llms_json(data_root) or {}
        tool_json = load_tools_json(data_root) or {}

        self.llm_candidates: List[str] = [k for k in llm_json.keys() if k and k != UNK_LLM_TOKEN]
        self.tool_candidates: List[str] = [k for k in tool_json.keys() if k and k != UNK_TOOL_TOKEN]

        self.llm_desc_map: Dict[str, str] = {
            k: str((v or {}).get("description", "")).strip()
            for k, v in llm_json.items()
            if k
        }
        self.tool_desc_map: Dict[str, str] = {
            k: str((v or {}).get("description", "")).strip()
            for k, v in tool_json.items()
            if k
        }

        # reverse indices
        self._llm_to_agent_indices: Dict[str, List[int]] = {}
        self._tool_to_agent_indices: Dict[str, List[int]] = {}

        for i in range(len(self.agent_ids)):
            llm_id = (self.agent_llm_ids[i] if i < len(self.agent_llm_ids) else "") or ""
            llm_name = (self.agent_llm_names[i] if i < len(self.agent_llm_names) else "") or ""
            for key in {llm_id.strip(), llm_name.strip()}:
                if not key:
                    continue
                self._llm_to_agent_indices.setdefault(key.lower(), []).append(i)

            for t in (self.agent_tools[i] if i < len(self.agent_tools) else []):
                if not t:
                    continue
                self._tool_to_agent_indices.setdefault(t, []).append(i)

    def _encode_query(self, query: str) -> np.ndarray:
        vec = self.q_vectorizer.transform([query]).toarray().astype(np.float32)
        q = torch.from_numpy(vec).to(self.device)
        q_idx = None
        if getattr(self.encoder, "use_query_id_emb", False):
            q_idx = torch.zeros(1, dtype=torch.long, device=self.device)
        with torch.no_grad():
            qe = self.encoder.encode_q(q, q_idx=q_idx).cpu().numpy()
        return qe

    def score_all_agents(self, query: str) -> np.ndarray:
        query = (query or "").strip()
        if not query:
            raise ValueError("查询不能为空。")
        qe = self._encode_query(query)
        scores = np.dot(qe, self.agent_embeddings.T).reshape(-1)
        return scores

    def score_all_agents_for_tool_query(self, tool_query: str) -> np.ndarray:
        tool_query = (tool_query or "").strip()
        if not tool_query:
            raise ValueError("tool_query 不能为空。")
        qe = self._encode_query(tool_query)
        scores = np.dot(qe, self.agent_embeddings.T).reshape(-1)
        return scores

    # ----------------------------
    # Col1/2: rerank over GLOBAL candidate sets
    # ----------------------------
    def recommend_llms_from_candidates(self, scores: np.ndarray, topk: int = 10) -> List[Dict[str, object]]:
        out: List[Dict[str, object]] = []
        for llm in self.llm_candidates:
            key = (llm or "").strip().lower()
            idxs = self._llm_to_agent_indices.get(key, [])
            if not idxs:
                continue
            best_i = max(idxs, key=lambda i: float(scores[i]))
            best_score = float(scores[best_i])
            out.append(
                {
                    "llm_name": llm,
                    "score": best_score,
                    "desc": self.llm_desc_map.get(llm, "") or "",
                }
            )
        out.sort(key=lambda x: -float(x["score"]))
        return out[:topk]

    def recommend_tools_from_candidates(self, scores: np.ndarray, topk: int = 10) -> List[Dict[str, object]]:
        out: List[Dict[str, object]] = []
        for tool in self.tool_candidates:
            idxs = self._tool_to_agent_indices.get(tool, [])
            if not idxs:
                continue
            best_i = max(idxs, key=lambda i: float(scores[i]))
            best_score = float(scores[best_i])
            out.append(
                {
                    "tool_name": tool,
                    "score": best_score,
                    "desc": self.tool_desc_map.get(tool, "") or "",
                }
            )
        out.sort(key=lambda x: -float(x["score"]))
        return out[:topk]

    # ----------------------------
    # Col3: suggestion -> GPT generate -> parse
    # ----------------------------
    def build_suggestion_text(
        self,
        llm_recs: List[Dict[str, object]],
        tool_recs: List[Dict[str, object]],
        top_llm: int = 10,
        top_tool: int = 10,
    ) -> str:
        llms = []
        for x in llm_recs[:top_llm]:
            name = (x.get("llm_name") or "").strip()
            if not name:
                continue
            llms.append(
                {
                    "name": name,
                    "description": self.llm_desc_map.get(name, "") or (x.get("desc") or ""),
                }
            )

        tools = []
        for x in tool_recs[:top_tool]:
            name = (x.get("tool_name") or "").strip()
            if not name:
                continue
            tools.append(
                {
                    "name": name,
                    "description": self.tool_desc_map.get(name, "") or (x.get("desc") or ""),
                }
            )

        payload = {"BackboneLLMs": llms, "Tools": tools}
        return json.dumps(payload, ensure_ascii=False, indent=2)

    def gpt_generate_agents(self, query: str, suggestion_text: str) -> Tuple[List[Dict[str, object]], str | None]:
        try:
            agents, _prompt, _raw = generate_agents_for_question(query, suggestion_text)
            return agents, None
        except Exception as e:
            return [], str(e)

    # ----------------------------
    # Col4: expand -> build combo docs -> external API scoring -> Top10
    # ----------------------------
    def _expand_candidate_combinations_from_generated(
        self,
        generated_agents: List[Dict[str, object]],
        llm_recs: List[Dict[str, object]],
        tool_recs: List[Dict[str, object]],
        max_candidates: int = 120,
    ) -> List[Dict[str, object]]:
        combos: List[Dict[str, object]] = []
        seen = set()

        llm_pool = [x.get("llm_name") or "" for x in llm_recs[: min(8, len(llm_recs))]]
        llm_pool = [x.strip() for x in llm_pool if x and x.strip()]

        tool_pool = [x.get("tool_name") or "" for x in tool_recs[: min(12, len(tool_recs))]]
        tool_pool = [x.strip() for x in tool_pool if x and x.strip()]

        llm_universe = {x.lower() for x in self.llm_candidates}
        tool_universe = set(self.tool_candidates)

        def add_combo(llm_name: str, tools: Iterable[str]) -> None:
            ln = (llm_name or "").strip()
            if not ln:
                return
            if llm_universe and ln.lower() not in llm_universe:
                return

            tools_set = tuple(sorted({t for t in tools if t and (t in tool_universe)}))
            key = (ln.lower(), tools_set)
            if key in seen or len(combos) >= max_candidates:
                return
            seen.add(key)
            combos.append({"llm_name": ln, "tools": list(tools_set)})

        for a in generated_agents:
            m = (a.get("M") or {})
            t = (a.get("T") or {})
            base_llm = (m.get("name") or "").strip() if isinstance(m, dict) else ""
            base_tools = (t.get("tools") or []) if isinstance(t, dict) else []
            if not isinstance(base_tools, list):
                base_tools = []
            base_tools = [str(x).strip() for x in base_tools if str(x).strip()]

            add_combo(base_llm, base_tools)
            for ln in llm_pool:
                add_combo(ln, base_tools)
            for tp in tool_pool:
                add_combo(base_llm, base_tools + [tp])
            for i in range(len(base_tools)):
                trimmed = [x for j, x in enumerate(base_tools) if j != i]
                add_combo(base_llm, trimmed)

        # if GPT failed and no combos, still make a small deterministic pool
        if not combos:
            # 4* (llm) x 0/1/2 tools from top tools
            llm_pool2 = llm_pool[:4]
            tool_pool2 = tool_pool[:6]
            for ln in llm_pool2:
                add_combo(ln, [])
                for t1 in tool_pool2[:4]:
                    add_combo(ln, [t1])
                for t1 in tool_pool2[:3]:
                    for t2 in tool_pool2[:3]:
                        if t1 != t2:
                            add_combo(ln, [t1, t2])

        return combos

    def _combo_to_document(self, combo: Dict[str, object]) -> Tuple[str, str, str, List[str]]:
        """
        Returns (display_name, model_desc, doc_text, tools_list).
        """
        llm_name = (combo.get("llm_name") or "").strip()
        tools = combo.get("tools") or []
        if not isinstance(tools, list):
            tools = []
        tools = [str(x).strip() for x in tools if str(x).strip()]

        model_desc = (self.llm_desc_map.get(llm_name, "") or "").strip()
        tool_lines = []
        for t in tools:
            td = (self.tool_desc_map.get(t, "") or "").strip()
            if td:
                tool_lines.append(f"- {t}: {td}")
            else:
                tool_lines.append(f"- {t}")

        # scoring doc: stable, concise, self-contained
        parts = [
            f"Backbone LLM: {llm_name}",
        ]
        if model_desc:
            parts.append(f"LLM description: {model_desc}")
        if tools:
            parts.append("Tools:")
            parts.extend(tool_lines)
        else:
            parts.append("Tools: (none)")

        doc = "\n".join(parts).strip()
        display_name = llm_name if not tools else f"{llm_name} + {', '.join(tools)}"
        return display_name, model_desc, doc, tools

    def score_combos_via_api(self, query: str, combos: List[Dict[str, object]], topk: int = 10) -> Tuple[List[Dict[str, object]], str | None]:
        """
        Build documents from combos -> call scoring API -> return TopK combo results.
        """
        if not combos:
            return [], None

        built = [self._combo_to_document(c) for c in combos]
        docs = [x[2] for x in built]

        try:
            resp = _http_post_json(self.scoring_url, {"query": query, "documents": docs}, timeout_s=25.0)
            scores = _extract_scores(resp, n_docs=len(docs))
        except Exception as e:
            return [], str(e)

        scores_np = np.array(scores, dtype=np.float32)
        k = min(topk, len(scores_np))
        idx = np.argpartition(-scores_np, k - 1)[:k]
        ordered = idx[np.argsort(-scores_np[idx])]

        out: List[Dict[str, object]] = []
        for i in ordered:
            display_name, model_desc, _doc, tools = built[i]
            out.append(
                {
                    "display_name": display_name,
                    "llm_name": combos[i].get("llm_name", ""),
                    "tools": tools,
                    "score": float(scores_np[i]),
                    "model_desc": model_desc,
                    "combo_idx": int(i),
                }
            )
        return out, None


def build_app(infer: TwoTowerInference) -> Flask:
    app = Flask(__name__)

    @app.route("/", methods=["GET", "POST"])
    def index():
        error = None
        query = request.form.get("query", "") if request.method == "POST" else ""

        llm_recs: List[Dict[str, object]] = []
        tool_recs: List[Dict[str, object]] = []
        suggestion_text: str = ""
        generated_agents: List[Dict[str, object]] = []
        gpt_error: str | None = None

        tool_query = ""
        tool_query_error: str | None = None

        col4_results: List[Dict[str, object]] = []
        col4_error: str | None = None
        combo_pool_size = 0

        has_results = False

        if request.method == "POST":
            try:
                has_results = True

                # Col1: LLM uses original query
                scores = infer.score_all_agents(query)
                llm_recs = infer.recommend_llms_from_candidates(scores, topk=infer.topk)

                # Col2: Tool uses rewritten tool_query
                tool_query = query
                try:
                    tool_query, _p, _raw = rewrite_tool_query(query, model_name="gpt-5-nano")
                except Exception as e:
                    tool_query_error = str(e)
                    tool_query = query

                tool_scores = infer.score_all_agents_for_tool_query(tool_query)
                tool_recs = infer.recommend_tools_from_candidates(tool_scores, topk=infer.topk)

                # Col3
                suggestion_text = infer.build_suggestion_text(llm_recs, tool_recs, top_llm=infer.topk, top_tool=infer.topk)
                generated_agents, gpt_error = infer.gpt_generate_agents(query, suggestion_text)

                # Col4: expand -> external scoring
                combos = infer._expand_candidate_combinations_from_generated(generated_agents, llm_recs, tool_recs, max_candidates=120)
                combo_pool_size = len(combos)
                col4_results, col4_error = infer.score_combos_via_api(query, combos, topk=infer.topk)

            except Exception as e:  # pragma: no cover
                error = str(e)

        return render_template_string(
            HTML_TEMPLATE,
            query=query,
            error=error,
            has_results=has_results,
            llm_recs=llm_recs,
            tool_recs=tool_recs,
            suggestion_text=suggestion_text,
            generated_agents=generated_agents,
            gpt_error=gpt_error,
            tool_query=tool_query if request.method == "POST" else "",
            tool_query_error=tool_query_error,
            col4_results=col4_results,
            col4_error=col4_error,
            combo_pool_size=combo_pool_size,
            scoring_url=infer.scoring_url,
        )

    @app.route("/api/recommend", methods=["POST"])
    def api_recommend():
        data = request.get_json(force=True, silent=True) or {}
        query = (data.get("query") or "").strip()
        topk = int(data.get("topk", infer.topk))

        if not query:
            return jsonify({"error": "missing query"}), 400

        try:
            scores = infer.score_all_agents(query)
            llm_recs = infer.recommend_llms_from_candidates(scores, topk=topk)

            tool_query = query
            tool_query_error = None
            try:
                tool_query, _p, _raw = rewrite_tool_query(query, model_name="gpt-5-nano")
            except Exception as e:
                tool_query_error = str(e)
                tool_query = query

            tool_scores = infer.score_all_agents_for_tool_query(tool_query)
            tool_recs = infer.recommend_tools_from_candidates(tool_scores, topk=topk)

            suggestion_text = infer.build_suggestion_text(llm_recs, tool_recs, top_llm=topk, top_tool=topk)
            generated_agents, gpt_error = infer.gpt_generate_agents(query, suggestion_text)

            combos = infer._expand_candidate_combinations_from_generated(generated_agents, llm_recs, tool_recs, max_candidates=120)
            col4_results, col4_error = infer.score_combos_via_api(query, combos, topk=topk)

            return jsonify(
                {
                    "query": query,
                    "llm_recs": llm_recs,
                    "tool_query": tool_query,
                    "tool_query_error": tool_query_error,
                    "tool_recs": tool_recs,
                    "suggestion": suggestion_text,
                    "generated_agents": generated_agents,
                    "gpt_error": gpt_error,
                    "combo_pool_size": len(combos),
                    "scoring_url": infer.scoring_url,
                    "col4_results": col4_results,
                    "col4_error": col4_error,
                }
            )
        except Exception as e:  # pragma: no cover
            return jsonify({"error": str(e)}), 400

    

    
    @app.route("/api/compare", methods=["POST"])
    def api_compare():
        data = request.get_json(force=True, silent=True) or {}

        query = (data.get("query") or "").strip()
        a = data.get("agentA") or {}
        b = data.get("agentB") or {}

        def norm_agent(x):
            llm = (x.get("llm_name") or "").strip()
            tools = x.get("tools") or []
            if not isinstance(tools, list):
                tools = []
            tools = [str(t).strip() for t in tools if str(t).strip()]
            return {"llm_name": llm, "tools": tools}

        if not query:
            return jsonify({"error": "query 不能为空"}), 400

        agentA = norm_agent(a)
        agentB = norm_agent(b)

        if not agentA["llm_name"] or not agentB["llm_name"]:
            return jsonify({"error": "两个 agent 都必须选择 LLM"}), 400

        # 复用你 Col4 的 combo->document 逻辑（如果你希望 compare 和 col4 文档格式一致）
        def combo_to_doc(combo: dict) -> tuple[str, str, str, list[str]]:
            llm_name = (combo.get("llm_name") or "").strip()
            tools = combo.get("tools") or []
            if not isinstance(tools, list):
                tools = []
            tools = [str(x).strip() for x in tools if str(x).strip()]

            model_desc = (infer.llm_desc_map.get(llm_name, "") or "").strip()
            tool_lines = []
            for t in tools:
                td = (infer.tool_desc_map.get(t, "") or "").strip()
                tool_lines.append(f"- {t}: {td}" if td else f"- {t}")

            parts = [f"Backbone LLM: {llm_name}"]
            if model_desc:
                parts.append(f"LLM description: {model_desc}")
            if tools:
                parts.append("Tools:")
                parts.extend(tool_lines)
            else:
                parts.append("Tools: (none)")

            doc = "\n".join(parts).strip()
            display_name = llm_name if not tools else f"{llm_name} + {', '.join(tools)}"
            return display_name, model_desc, doc, tools

        try:
            # build 2 docs
            dispA, model_descA, docA, toolsA = combo_to_doc(agentA)
            dispB, model_descB, docB, toolsB = combo_to_doc(agentB)

            # call external scoring service (8001)
            api_resp = _http_post_json(
                infer.scoring_url,  # e.g. http://127.0.0.1:8001/compute_scores
                {"query": query, "documents": [docA, docB]},
                timeout_s=25.0,
            )
            scores = _extract_scores(api_resp, n_docs=2)

            sA = float(scores[0])
            sB = float(scores[1])

            return jsonify(
                {
                    "query": query,
                    "scoring_url": infer.scoring_url,
                    "agentA": {
                        "display_name": dispA,
                        "llm_name": agentA["llm_name"],
                        "tools": toolsA,
                        "model_desc": model_descA,
                        "score": sA,
                    },
                    "agentB": {
                        "display_name": dispB,
                        "llm_name": agentB["llm_name"],
                        "tools": toolsB,
                        "model_desc": model_descB,
                        "score": sB,
                    },
                    "delta": sA - sB,
                }
            )
        except Exception as e:
            return jsonify({"error": str(e)}), 400

    return app

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="TwoTower TF-IDF 推理与Flask服务（Col4用外部API打分）")
    ap.add_argument("--data_root", type=str, required=True, help="数据集根目录 (包含PartI/II/III)")
    ap.add_argument("--model_path", type=str, required=True, help="训练好的TwoTowerTFIDF模型(.pt)")
    ap.add_argument("--max_features", type=int, default=TFIDF_MAX_FEATURES)
    ap.add_argument("--device", type=str, default="cpu", help="设备: cpu / cuda:0 等")
    ap.add_argument("--topk", type=int, default=10, help="TopK")
    ap.add_argument("--host", type=str, default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument(
        "--scoring_url",
        type=str,
        default="http://127.0.0.1:8501/compute_scores",
        help="外部打分服务URL，例如 http://127.0.0.1:8501/compute_scores",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    device = _device_from_arg(args.device)

    infer = TwoTowerInference(
        data_root=args.data_root,
        model_path=args.model_path,
        device=device,
        max_features=args.max_features,
        topk=args.topk,
        scoring_url=args.scoring_url,
    )

    app = build_app(infer)
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
