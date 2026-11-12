"""Shared fixtures for tests/agentic_rag_web.

No working ``OPENAI_API_KEY`` is available in CI/dev for this repo's web
port work, so these fixtures patch only the two OpenAI network boundaries
(embeddings, chat completions) with deterministic fakes and exercise the
**real** compiled graph, the **real** ChromaDB client, and the **real**
LangGraph ``Send``-based fan-out otherwise. This is the same harness used
for the warm-reuse concurrency spike (see
``ARCHITECTURE.md``) and is reused here for the G2
equivalence gate so both are exercised against identical
node behavior.
"""

from __future__ import annotations

import copy
import hashlib
import os
import re

import pytest

os.environ.setdefault("WORKER_QUERY_REFORMULATION_ENABLED", "0")
os.environ.setdefault("OPENAI_API_KEY", "sk-test-not-used")

INITIAL_STATE = {
    "query": "",
    "sub_tasks": [],
    "active_agent": "",
    "pending_agents": [],
    "planned_agents": [],
    "planning_fast_routed": False,
    "planning_fast_route_patterns": [],
    "context_gathered": {},
    "draft_response": "",
    "final_response": "",
    "citations": [],
    "validation_score": 0.0,
    "critic_feedback": "",
    "retry_count": 0,
    "llm_calls": [],
    "retrieval_details": [],
    "worker_errors": [],
}


def fake_embed(texts):
    """Deterministic pseudo-embedding: same text -> same vector, every time."""
    dim = 1536
    out = []
    for t in texts:
        h = hashlib.sha256(t.encode("utf-8")).digest()
        out.append([((h[i % len(h)] / 255.0) - 0.5) for i in range(dim)])
    return out


class FakeResponse:
    def __init__(self, content: str):
        self.content = content
        self.response_metadata = {
            "token_usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
            "model_name": "fake-model",
            "finish_reason": "stop",
        }
        self.usage_metadata = {"input_tokens": 10, "output_tokens": 10, "total_tokens": 20}


class FakeLLM:
    """Discriminates by the SystemMessage (stable per-node), not the
    query-bearing human message (whose overlapping substrings across
    nodes caused an early spike bug -- see session history)."""

    def invoke(self, messages):
        system_text = getattr(messages[0], "content", "") if messages else ""
        human_text = " ".join(getattr(m, "content", "") for m in messages[1:])

        if "conversation context condenser" in system_text.lower():
            m_q = re.search(
                r"FOLLOW-UP USER QUESTION:\s*(.+?)\n\nRewrite",
                human_text,
                re.DOTALL,
            )
            follow_up = m_q.group(1).strip() if m_q else human_text
            m_t1 = re.search(r"Turn 1 User Query:\s*(.+)", human_text)
            prior_q = m_t1.group(1).strip() if m_t1 else ""
            if any(
                w in human_text.lower()
                for w in (
                    "could not find records",
                    "no records were found",
                    "does not appear in the retrieved evidence",
                    "fictitious",
                    "unverified",
                )
            ):
                return FakeResponse(
                    f"Regarding the unverified premise from prior turn ({prior_q}) "
                    f"for which no OSHA or CSB records were found: {follow_up}"
                )
            return FakeResponse(f"{follow_up} (in the context of: {prior_q})")

        if "reviewer" in system_text.lower() or "fact-checker" in system_text.lower():
            return FakeResponse('{"score": 0.9, "critique": "looks fine", "issues": []}')

        if "editor performing targeted revisions" in system_text.lower():
            m = re.search(r"CURRENT DRAFT:\n(.*?)\n\nVALIDATOR FEEDBACK", human_text, re.DOTALL)
            draft = m.group(1) if m else human_text
            return FakeResponse(draft)

        m = re.search(r'USER QUERY:\s*"([^"]+)"', human_text)
        q = m.group(1) if m else "UNKNOWN"
        return FakeResponse(f"## Answer\nThis response is for query: {q}\n\nSources reviewed.")


@pytest.fixture
def patched_llm_boundary():
    """Patch the two OpenAI network boundaries; yield the FakeLLM instance."""
    from unittest.mock import patch

    from src.agentic_rag.workers import TOOLS

    original_embed = {name: tool._embed for name, tool in TOOLS.items()}
    for tool in TOOLS.values():
        tool._embed = fake_embed

    fake_llm = FakeLLM()
    try:
        with patch("src.agentic_rag.supervisor.make_chat_model", return_value=fake_llm), patch(
            "src.agentic_rag.critic.make_chat_model", return_value=fake_llm
        ), patch("src.agentic_rag.workers.make_chat_model", return_value=fake_llm), patch(
            "src.agentic_rag_web.condense.make_chat_model", return_value=fake_llm
        ):
            yield fake_llm
    finally:
        for name, tool in TOOLS.items():
            tool._embed = original_embed[name]


def run_query_sync(query: str, graph) -> dict:
    """Drive one query through a compiled graph, returning the final state dict."""
    init = copy.deepcopy(INITIAL_STATE)
    init["query"] = query
    state = dict(init)
    for ev in graph.stream(init, stream_mode="updates"):
        for node, upd in ev.items():
            if node == "__start__":
                continue
            state.update(upd)
    return state


# Ten queries chosen to span every worker path via the real fast-route
# planner (src.agentic_rag.supervisor._try_fast_route) so PlanNode makes
# zero LLM calls -- see ARCHITECTURE.md
G2_SAMPLE_QUERIES = [
    "What happened in the West Fertilizer explosion?",
    "What are the OSHA regulations for process safety management?",
    "What violations were cited following the incident?",
    "What recommendations were made to prevent future incidents?",
    "What were the root causes identified by the CSB investigation?",
    "How many fatalities occurred in refinery incidents?",
    "Which state has the most violations?",
    "Calculate the permissible exposure limit for benzene.",
    "What CSB incidents relate to OSHA 1910.119 violations?",
    "What is the trend in citation frequency over the past five years?",
]
