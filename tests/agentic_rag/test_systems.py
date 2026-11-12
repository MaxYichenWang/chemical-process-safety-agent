"""Unit tests for the Stage-E comparator systems.

unit tests need no corpus or API key, so every test stubs the retrieval tools and the
chat-model factory. The assertions focus on the structural contract each
baseline must satisfy: an eval-compatible state dict, the right number of LLM
calls, and the right retrieval behaviour that distinguishes the three systems.
"""

import asyncio
from unittest.mock import patch

import pytest

from src.agentic_rag.systems import (
    ALL_SYSTEMS,
    NoRetrievalSystem,
    SingleAgentSystem,
    VanillaRAGSystem,
    build_system,
)
from src.evaluation.utils import normalize_graph_state_for_eval


class _StubResponse:
    def __init__(self, content: str):
        self.content = content
        self.response_metadata = {
            "model_name": "stub-model",
            "token_usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
            "finish_reason": "stop",
        }


class _StubChatModel:
    """Records the messages it was invoked with and returns a canned answer."""

    model_name = "stub-model"
    temperature = 0.0

    def __init__(self, reply: str):
        self._reply = reply
        self.invocations = []

    def invoke(self, messages):
        self.invocations.append(messages)
        return _StubResponse(self._reply)


class _FactoryRecorder:
    """A chat_model_factory that hands out one stub model per call."""

    def __init__(self, reply: str = "STUB ANSWER"):
        self._reply = reply
        self.models = []

    def __call__(self, role, **kwargs):
        model = _StubChatModel(self._reply)
        self.models.append(model)
        return model

    @property
    def total_calls(self) -> int:
        return sum(len(m.invocations) for m in self.models)


class _StubTool:
    def __init__(self, name: str):
        self.name = name
        self.queries = []

    def query(self, query_str, metadata_filters=None, top_k=10):
        self.queries.append({"query": query_str, "top_k": top_k})
        text = f"[Source 1] Collection: {self.name} | content about {query_str}"
        citations = [
            {
                "collection": self.name,
                "source": f"{self.name}-doc",
                "location": "p.1",
                "chunk_id": f"{self.name}-chunk-1",
                "similarity_score": 0.9,
            }
        ]
        return text, citations


def _stub_tools():
    return {
        "osha_regulations": _StubTool("osha_regulations"),
        "osha_enf": _StubTool("osha_enf"),
        "csb": _StubTool("csb"),
        "csb_recommendations": _StubTool("csb_recommendations"),
    }


def _run(system, query="What caused the incident?"):
    return asyncio.run(system.ainvoke({"query": query}))


# ---------------------------------------------------------------------------
# no_retrieval
# ---------------------------------------------------------------------------


def test_no_retrieval_makes_one_llm_call_and_no_retrieval():
    factory = _FactoryRecorder("Parametric answer.")
    tools = _stub_tools()
    system = NoRetrievalSystem(tools=tools, chat_model_factory=factory)

    state = _run(system)

    assert factory.total_calls == 1
    assert all(not t.queries for t in tools.values())
    assert state["context_gathered"] == {}
    assert state["citations"] == []
    assert state["final_response"] == "Parametric answer."
    assert state["draft_response"] == "Parametric answer."
    assert len(state["llm_calls"]) == 1


# ---------------------------------------------------------------------------
# vanilla
# ---------------------------------------------------------------------------


def test_vanilla_queries_all_tools_with_raw_query_and_one_gen_call():
    factory = _FactoryRecorder("Vanilla answer.")
    tools = _stub_tools()
    system = VanillaRAGSystem(tools=tools, chat_model_factory=factory)

    state = _run(system, query="raw query text")

    # One generation call only (no reformulation).
    assert factory.total_calls == 1
    # Every tool queried with the RAW query, no reformulation.
    for t in tools.values():
        assert len(t.queries) == 1
        assert t.queries[0]["query"] == "raw query text"
    # Union of all four collections' citations.
    assert len(state["citations"]) == 4
    assert set(state["context_gathered"].keys()) == set(tools.keys())
    assert state["final_response"] == "Vanilla answer."
    assert len(state["retrieval_details"]) == 4


def test_vanilla_state_is_eval_normalizable():
    factory = _FactoryRecorder()
    system = VanillaRAGSystem(tools=_stub_tools(), chat_model_factory=factory)

    state = _run(system)
    docs, agents = normalize_graph_state_for_eval(state)

    # Each citation pairs with a parsed text block; chunk_ids survive.
    assert len(docs) == 4
    assert all(d["chunk_id"].endswith("chunk-1") for d in docs)
    assert agents  # non-empty corpus list


def test_vanilla_top_k_is_forwarded_to_tools():
    system = VanillaRAGSystem(
        tools=_stub_tools(), chat_model_factory=_FactoryRecorder(), top_k=3
    )
    state = _run(system)
    # retrieval_details reflect the configured top_k
    assert all(rd["top_k"] == 3 for rd in state["retrieval_details"])


# ---------------------------------------------------------------------------
# single_agent
# ---------------------------------------------------------------------------


def test_single_agent_reformulates_then_retrieves_then_generates():
    factory = _FactoryRecorder("Reformulated search query")
    tools = _stub_tools()
    system = SingleAgentSystem(tools=tools, chat_model_factory=factory)

    state = _run(system, query="original question")

    # Two LLM calls: reformulation + synthesis.
    assert factory.total_calls == 2
    # Tools queried with the reformulated query, not the raw one.
    for t in tools.values():
        assert len(t.queries) == 1
        assert t.queries[0]["query"] == "Reformulated search query"
    assert len(state["citations"]) == 4
    assert state["final_response"] == "Reformulated search query"
    assert len(state["llm_calls"]) == 2


def test_single_agent_falls_back_to_raw_query_when_reformulation_blank():
    factory = _FactoryRecorder("   ")  # blank reformulation
    tools = _stub_tools()
    system = SingleAgentSystem(tools=tools, chat_model_factory=factory)

    _run(system, query="fallback question")
    for t in tools.values():
        assert t.queries[0]["query"] == "fallback question"


# ---------------------------------------------------------------------------
# build_system factory
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name,cls",
    [
        ("vanilla", VanillaRAGSystem),
        ("single_agent", SingleAgentSystem),
        ("no_retrieval", NoRetrievalSystem),
    ],
)
def test_build_system_returns_expected_comparator(name, cls):
    system = build_system(name, tools=_stub_tools(), chat_model_factory=_FactoryRecorder())
    assert isinstance(system, cls)


def test_build_system_full_delegates_to_build_graph():
    sentinel = object()
    with patch("src.agentic_rag.graph.build_graph", return_value=sentinel) as bg:
        result = build_system("full")
    assert result is sentinel
    bg.assert_called_once()


def test_build_system_unknown_raises():
    with pytest.raises(ValueError, match="Unknown system"):
        build_system("bogus")


def test_all_systems_constant_shape():
    assert ALL_SYSTEMS[0] == "full"
    assert set(ALL_SYSTEMS) == {"full", "vanilla", "single_agent", "no_retrieval"}


# ---------------------------------------------------------------------------
# distinguishing behaviour across the three baselines
# ---------------------------------------------------------------------------


def test_comparators_have_distinct_call_profiles():
    """no_retrieval < vanilla (gen only) < single_agent (reformulate + gen)."""
    profiles = {}
    for name in ("no_retrieval", "vanilla", "single_agent"):
        factory = _FactoryRecorder()
        tools = _stub_tools()
        system = build_system(name, tools=tools, chat_model_factory=factory)
        _run(system)
        n_tool_queries = sum(len(t.queries) for t in tools.values())
        profiles[name] = (factory.total_calls, n_tool_queries)

    assert profiles["no_retrieval"] == (1, 0)
    assert profiles["vanilla"] == (1, 4)
    assert profiles["single_agent"] == (2, 4)
