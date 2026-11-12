"""Web-server graph assembly.

``build_web_graph()`` re-wires the same node functions ``build_graph()``
(``src/agentic_rag/graph.py``) uses, over ``WebAgentState`` instead of
``AgentState``, and prepends ``CondenseNode`` (``src/agentic_rag_web/condense.py``)
ahead of ``PlanNode`` for multi-turn conversation support (Phase 6). Every
downstream node function and every routing helper is **imported** from
``src.agentic_rag.*`` -- none of it is copied. Only the ~50 lines of
``StateGraph`` wiring (which nodes exist, which edges connect them) are
duplicated, which is the deliberate, accepted middle ground documented in
design doc §3.2: editing ``src/agentic_rag/graph.py`` would violate the
paper/eval freeze in §1; copying the whole package would recreate the drift
problem it exists to avoid.

When ``conversation_history`` is empty or ``None``, ``CondenseNode`` is a pure
no-op passthrough, keeping empty-history behavior byte-identical to
``build_graph()`` for the G2 equivalence gate.
"""

from __future__ import annotations

from langgraph.graph import END, StateGraph

from src.agentic_rag.critic import critic_agent

# Routing/control-flow helpers -- imported, never re-implemented, so there is
# a single source of truth for retry/finalize/critic-routing behavior.
from src.agentic_rag.graph import (
    _finalize_node,
    _retry_node,
    _route_from_critic,
)
from src.agentic_rag.llm_logger import _estimate_cost
from src.agentic_rag.supervisor import (
    _TOKEN_EMITTER,
    route_from_plan,
    supervisor_plan,
    supervisor_revise,
    supervisor_synthesize,
)
from src.agentic_rag.workers import (
    data_analyst_agent,
    incident_retriever_agent,
    osha_enf_agent,
    osha_regs_agent,
    recommendation_specialist_agent,
    root_cause_analyst_agent,
)
from src.agentic_rag_web.condense import condense_query
from src.agentic_rag_web.state import WebAgentState

WORKER_NODES = [
    "OshaRegsWorker",
    "OshaEnfWorker",
    "IncidentWorker",
    "RecommendationsWorker",
    "RootCauseWorker",
    "DataAnalystWorker",
]


def _count_tokens(text: str) -> int:
    if not text:
        return 0
    try:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return max(1, len(text) // 4)


class _StreamUsageProxy:
    """Thin proxy around a LangChain chat model that preserves token usage during ``stream()``."""

    def __init__(self, inner, captured_prompts: list[str]):
        self._inner = inner
        self._captured_prompts = captured_prompts
        if hasattr(inner, "stream_usage"):
            try:
                inner.stream_usage = True
            except Exception:
                pass

    def __getattr__(self, name: str):
        return getattr(self._inner, name)

    def stream(self, input_messages, *args, **kwargs):
        try:
            prompt_text = "\n".join(
                str(getattr(m, "content", "")) for m in (input_messages or [])
            )
            self._captured_prompts.append(prompt_text)
        except Exception:
            pass

        last_chunk = None
        for chunk in self._inner.stream(input_messages, *args, **kwargs):
            last_chunk = chunk
            usage = getattr(chunk, "usage_metadata", None)
            if isinstance(usage, dict) and (
                usage.get("input_tokens") or usage.get("output_tokens")
            ):
                resp_meta = getattr(chunk, "response_metadata", None)
                if isinstance(resp_meta, dict) and not resp_meta.get("token_usage"):
                    pt = int(usage.get("input_tokens") or 0)
                    ct = int(usage.get("output_tokens") or 0)
                    tt = int(usage.get("total_tokens") or (pt + ct))
                    resp_meta["token_usage"] = {
                        "prompt_tokens": pt,
                        "completion_tokens": ct,
                        "total_tokens": tt,
                    }
            yield chunk
        _ = last_chunk


def _web_synthesize_node(state: WebAgentState) -> dict:
    """Web wrapper around ``supervisor_synthesize`` that ensures streamed calls
    record ``prompt_tokens``, ``completion_tokens``, ``total_tokens``, and
    ``estimated_cost_usd`` without modifying ``src/agentic_rag/supervisor.py``.

    When ``_TOKEN_EMITTER.get()`` is ``None`` (non-streaming ``.invoke()``),
    delegates directly to ``supervisor_synthesize`` for Gate G2 equivalence.
    """
    if _TOKEN_EMITTER.get() is None:
        return supervisor_synthesize(state)

    import src.agentic_rag.supervisor as sup_mod

    orig_make = sup_mod.make_chat_model
    captured_prompts: list[str] = []

    def _wrapped_make_chat_model(role: str, **kwargs):
        llm = orig_make(role, **kwargs)
        if not hasattr(llm, "stream"):
            return llm
        return _StreamUsageProxy(llm, captured_prompts)

    sup_mod.make_chat_model = _wrapped_make_chat_model  # type: ignore[assignment]
    try:
        result = supervisor_synthesize(state)
    finally:
        sup_mod.make_chat_model = orig_make  # type: ignore[assignment]

    # If the streaming provider omitted usage chunks despite `stream_usage=True`,
    # backfill token counts and cost from the exact prompt + draft response.
    draft_text = result.get("draft_response") or ""
    for call in result.get("llm_calls") or []:
        if (
            call.get("call_purpose") == "synthesis"
            and call.get("prompt_tokens") is None
            and call.get("completion_tokens") is None
        ):
            prompt_str = (
                captured_prompts[-1]
                if captured_prompts
                else (
                    (call.get("system_prompt_preview") or "")
                    + "\n"
                    + (call.get("user_prompt_preview") or "")
                )
            )
            pt = _count_tokens(prompt_str)
            ct = _count_tokens(draft_text)
            model_used = (
                call.get("model_used") or call.get("model_requested") or "gpt-5-mini"
            )
            call["prompt_tokens"] = pt
            call["completion_tokens"] = ct
            call["total_tokens"] = pt + ct
            call["estimated_cost_usd"] = round(_estimate_cost(model_used, pt, ct), 8)

    return result


def build_web_graph() -> StateGraph:
    """Constructs and compiles the web-server variant of the multi-agent graph.

    Inserts ``CondenseNode`` before ``PlanNode``.
    When ``conversation_history`` is empty, ``CondenseNode`` is a pure no-op
    passthrough so single-turn behavior stays byte-identical to the paper
    graph.
    """
    workflow = StateGraph(WebAgentState)

    workflow.add_node("CondenseNode", condense_query)
    workflow.add_node("PlanNode", supervisor_plan)
    workflow.add_node("OshaRegsWorker", osha_regs_agent)
    workflow.add_node("OshaEnfWorker", osha_enf_agent)
    workflow.add_node("IncidentWorker", incident_retriever_agent)
    workflow.add_node("RecommendationsWorker", recommendation_specialist_agent)
    workflow.add_node("RootCauseWorker", root_cause_analyst_agent)
    workflow.add_node("DataAnalystWorker", data_analyst_agent)
    workflow.add_node("SynthesizeNode", _web_synthesize_node)
    workflow.add_node("RevisionNode", supervisor_revise)
    workflow.add_node("CriticNode", critic_agent)
    workflow.add_node("RetryNode", _retry_node)
    workflow.add_node("FinalizeNode", _finalize_node)

    workflow.set_entry_point("CondenseNode")
    workflow.add_edge("CondenseNode", "PlanNode")

    workflow.add_conditional_edges(
        "PlanNode",
        route_from_plan,
        WORKER_NODES,
    )

    for worker in WORKER_NODES:
        workflow.add_edge(worker, "SynthesizeNode")

    workflow.add_edge("SynthesizeNode", "CriticNode")

    workflow.add_conditional_edges(
        "CriticNode",
        _route_from_critic,
        {
            "FinalizeNode": "FinalizeNode",
            "RetryNode": "RetryNode",
        },
    )

    workflow.add_edge("RetryNode", "RevisionNode")
    workflow.add_edge("RevisionNode", "CriticNode")
    workflow.add_edge("FinalizeNode", END)

    return workflow.compile()
