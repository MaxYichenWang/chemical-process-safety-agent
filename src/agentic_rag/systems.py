"""Comparator / baseline systems for the evaluation ablation study.

The publication plan (Stage E) requires three baselines to contextualise the
full multi-agent system against simpler alternatives:

* ``vanilla`` — naive RAG: retrieve from every collection with the raw
  query (no reformulation, no metadata filters, no routing), union the hits,
  and make a single generation call. No supervisor, workers, critic, or retry.
* ``single_agent`` — one flat agent with all tools and no supervisor
  routing: it makes one reasoning/reformulation call, queries every tool with
  the reformulated query, then makes a single generation call. No critic loop.
* ``no_retrieval`` — parametric knowledge only: a single generation call
  with no retrieved context at all.

Each system exposes an async ``ainvoke({"query": ...})`` returning a state dict
shaped like the LangGraph :class:`~src.agentic_rag.state.AgentState` so the
existing :class:`EndToEndEvaluator` can drive it interchangeably with the full
graph (which it obtains from :func:`build_system` with ``name="full"``).

Both the retrieval tools and the chat-model factory are injectable so the
systems can be unit-tested locally with stubs — unit tests need no corpus or API key and
cannot run them for real (correctness against the corpus is confirmed by the
live generation error rate).
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict, List, Optional, Protocol

from langchain_core.messages import HumanMessage, SystemMessage

from src.agentic_rag.config import (
    HYBRID_VECTOR_TOP_K,
    SUPERVISOR_SYNTHESIS_MAX_TOKENS,
)
from src.agentic_rag.llm_logger import record_llm_call

logger = logging.getLogger(__name__)

FULL = "full"
VANILLA = "vanilla"
SINGLE_AGENT = "single_agent"
NO_RETRIEVAL = "no_retrieval"

COMPARATOR_SYSTEMS = (VANILLA, SINGLE_AGENT, NO_RETRIEVAL)
ALL_SYSTEMS = (FULL,) + COMPARATOR_SYSTEMS

# Baselines deliberately use a plain, un-gated synthesis prompt. The full
# system's confidence-gating / premise-check prompt engineering is part of what
# the ablation measures, so the comparators must NOT inherit it.
_SYNTHESIS_SYSTEM = (
    "You are a Chemical Process Safety analyst. Answer the user's question using "
    "ONLY the retrieved context provided. Cite the source collection and document "
    "for each factual claim. If the context does not support an answer, say so "
    "explicitly rather than speculating."
)

_NO_RETRIEVAL_SYSTEM = (
    "You are a Chemical Process Safety analyst. Answer the user's question from "
    "your own knowledge. You have NO retrieved documents. Do not fabricate "
    "specific incidents, citations, penalties, or dates; if you are not "
    "confident a fact is correct, say so explicitly."
)

_REFORMULATION_SYSTEM = (
    "You rewrite a user's question into a single concise search query for a "
    "vector database of OSHA regulations, OSHA enforcement records, and CSB "
    "incident reports. Return only the rewritten query text, nothing else."
)

# Factory signature: (role, **kwargs) -> chat model with an ``.invoke(messages)``
# method returning an object exposing ``.content`` (and optionally
# ``.response_metadata`` for token accounting).
ChatModelFactory = Callable[..., Any]


class System(Protocol):
    """Minimal interface the evaluator drives (mirrors a compiled graph)."""

    async def ainvoke(self, input: Dict[str, Any]) -> Dict[str, Any]: ...


def _default_chat_model_factory() -> ChatModelFactory:
    from src.agentic_rag.llm_factory import make_chat_model

    return make_chat_model


def _default_tools() -> Dict[str, Any]:
    # Imported lazily: initializing tools connects to ChromaDB at import time.
    from src.agentic_rag.workers import TOOLS

    return TOOLS


def _empty_state(query: str) -> Dict[str, Any]:
    """A state dict pre-populated with the keys the evaluator reads via ``.get``.

    Comparators never route, retry, or run a critic, so those fields stay at
    their neutral defaults (empty plan, no retries, no critic feedback).
    """
    return {
        "query": query,
        "context_gathered": {},
        "citations": [],
        "draft_response": "",
        "final_response": "",
        "validation_score": 0.0,
        "critic_feedback": "",
        "retry_count": 0,
        "planned_agents": [],
        "planning_fast_routed": False,
        "planning_fast_route_patterns": [],
        "retrieval_details": [],
        "worker_errors": [],
        "llm_calls": [],
    }


class _BaseComparator:
    system_name: str = "comparator"

    def __init__(
        self,
        *,
        tools: Optional[Dict[str, Any]] = None,
        chat_model_factory: Optional[ChatModelFactory] = None,
        top_k: int = HYBRID_VECTOR_TOP_K,
    ) -> None:
        self._tools = tools
        self._chat_model_factory = chat_model_factory
        self._top_k = top_k

    # -- lazily-resolved dependencies (so import + --help need no corpus/key) --
    def _get_tools(self) -> Dict[str, Any]:
        if self._tools is None:
            self._tools = _default_tools()
        return self._tools

    def _get_factory(self) -> ChatModelFactory:
        if self._chat_model_factory is None:
            self._chat_model_factory = _default_chat_model_factory()
        return self._chat_model_factory

    def _llm_call(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        node_name: str,
        call_purpose: str,
        existing_calls: List[Dict[str, Any]],
    ) -> tuple[str, List[Dict[str, Any]]]:
        """Make one chat call and append a normalized llm_calls record."""
        llm = self._get_factory()(
            "supervisor",
            max_completion_tokens=SUPERVISOR_SYNTHESIS_MAX_TOKENS,
        )
        t0 = time.time()
        response = llm.invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ]
        )
        latency = time.time() - t0
        model_cfg = getattr(llm, "model_name", None) or getattr(llm, "model", "unknown")
        calls = record_llm_call(
            response=response,
            node_name=node_name,
            call_purpose=call_purpose,
            model_config=str(model_cfg),
            temperature=getattr(llm, "temperature", 0.0) or 0.0,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            latency_s=latency,
            existing_calls=existing_calls,
        )
        return getattr(response, "content", "") or "", calls

    def _retrieve(
        self, query: str, *, tool_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Union retrieval across the given tools (all tools by default).

        No reformulation, no metadata filters — each tool is queried with the
        provided ``query`` string. Returns the ``context_gathered`` /
        ``citations`` / ``retrieval_details`` fragments in a single, index-aligned
        traversal order so ``normalize_graph_state_for_eval`` pairs each citation
        with its formatted text block.
        """
        tools = self._get_tools()
        names = tool_names if tool_names is not None else list(tools.keys())

        context: Dict[str, List[str]] = {}
        citations: List[Dict[str, Any]] = []
        retrieval_details: List[Dict[str, Any]] = []

        for name in names:
            tool = tools.get(name)
            if tool is None:
                continue
            try:
                result_text, new_citations = tool.query(query, top_k=self._top_k)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Comparator retrieval failed on %s: %s", name, exc)
                continue
            context[name] = [result_text]
            citations.extend(new_citations)
            retrieval_details.append(
                {
                    "collection": name,
                    "node_name": self.system_name,
                    "reformulated_query": query,
                    "metadata_filters": None,
                    "chunk_count": len(new_citations),
                    "top_k": self._top_k,
                    "total_chars": len(result_text),
                }
            )

        return {
            "context_gathered": context,
            "citations": citations,
            "retrieval_details": retrieval_details,
        }

    @staticmethod
    def _context_str(context: Dict[str, List[str]]) -> str:
        if not context:
            return "No context retrieved."
        blocks = []
        for name, texts in context.items():
            for text in texts:
                blocks.append(f"--- {name} ---\n{text}")
        return "\n\n".join(blocks)


class VanillaRAGSystem(_BaseComparator):
    """union retriever over all collections + a single generation call."""

    system_name = VANILLA

    async def ainvoke(self, input: Dict[str, Any]) -> Dict[str, Any]:
        query = input.get("query", "")
        state = _empty_state(query)

        retrieved = self._retrieve(query)
        state.update(retrieved)

        answer, calls = self._llm_call(
            system_prompt=_SYNTHESIS_SYSTEM,
            user_prompt=(
                f'USER QUERY: "{query}"\n\nRETRIEVED CONTEXT:\n'
                f"{self._context_str(retrieved['context_gathered'])}"
            ),
            node_name="VanillaRAG",
            call_purpose="synthesis",
            existing_calls=[],
        )
        state["draft_response"] = answer
        state["final_response"] = answer
        state["llm_calls"] = calls
        return state


class SingleAgentSystem(_BaseComparator):
    """one flat agent, all tools, no supervisor routing / critic loop."""

    system_name = SINGLE_AGENT

    async def ainvoke(self, input: Dict[str, Any]) -> Dict[str, Any]:
        query = input.get("query", "")
        state = _empty_state(query)

        # Single reasoning step: reformulate the query into one search string.
        reformulated, calls = self._llm_call(
            system_prompt=_REFORMULATION_SYSTEM,
            user_prompt=query,
            node_name="SingleAgent",
            call_purpose="query_reformulation",
            existing_calls=[],
        )
        reformulated = (reformulated or query).strip() or query

        retrieved = self._retrieve(reformulated)
        state.update(retrieved)

        answer, calls = self._llm_call(
            system_prompt=_SYNTHESIS_SYSTEM,
            user_prompt=(
                f'USER QUERY: "{query}"\n\nRETRIEVED CONTEXT:\n'
                f"{self._context_str(retrieved['context_gathered'])}"
            ),
            node_name="SingleAgent",
            call_purpose="synthesis",
            existing_calls=calls,
        )
        state["draft_response"] = answer
        state["final_response"] = answer
        state["llm_calls"] = calls
        return state


class NoRetrievalSystem(_BaseComparator):
    """parametric knowledge only — a single generation call, no context."""

    system_name = NO_RETRIEVAL

    async def ainvoke(self, input: Dict[str, Any]) -> Dict[str, Any]:
        query = input.get("query", "")
        state = _empty_state(query)

        answer, calls = self._llm_call(
            system_prompt=_NO_RETRIEVAL_SYSTEM,
            user_prompt=f'USER QUERY: "{query}"',
            node_name="NoRetrieval",
            call_purpose="synthesis",
            existing_calls=[],
        )
        state["draft_response"] = answer
        state["final_response"] = answer
        state["llm_calls"] = calls
        return state


def build_system(
    name: str = FULL,
    *,
    tools: Optional[Dict[str, Any]] = None,
    chat_model_factory: Optional[ChatModelFactory] = None,
    top_k: int = HYBRID_VECTOR_TOP_K,
) -> System:
    """Return a runnable system by name.

    ``name="full"`` returns the compiled multi-agent LangGraph. The comparator
    names return the corresponding baseline. ``tools`` / ``chat_model_factory``
    are injection points for tests; they are ignored for the full graph (which
    wires its own dependencies).
    """
    if name == FULL:
        from src.agentic_rag.graph import build_graph

        return build_graph()

    comparators = {
        VANILLA: VanillaRAGSystem,
        SINGLE_AGENT: SingleAgentSystem,
        NO_RETRIEVAL: NoRetrievalSystem,
    }
    cls = comparators.get(name)
    if cls is None:
        raise ValueError(
            f"Unknown system '{name}'. Expected one of: {', '.join(ALL_SYSTEMS)}"
        )
    return cls(tools=tools, chat_model_factory=chat_model_factory, top_k=top_k)
