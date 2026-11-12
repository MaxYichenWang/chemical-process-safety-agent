"""Tests for the ablation flags (CRITIC_ENABLED, WORKER_QUERY_REFORMULATION_ENABLED).

Config-level parsing tests need no LLM/graph deps at all. The critic_agent /
_reformulate_query passthrough tests import the real modules -- on a machine
with langchain_core/langgraph/chromadb installed (the development
machines) they run unconditionally; in a minimal sandbox lacking those
packages they're skipped rather than failing on an unrelated missing dep.
"""

import importlib

import pytest


def _reload_agentic_config(monkeypatch, **env):
    for key in ("CRITIC_ENABLED", "WORKER_QUERY_REFORMULATION_ENABLED"):
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    from src.agentic_rag import config as agentic_config

    return importlib.reload(agentic_config)


def test_critic_enabled_defaults_to_true_when_unset(monkeypatch):
    cfg = _reload_agentic_config(monkeypatch)
    assert cfg.CRITIC_ENABLED is True


def test_critic_enabled_false_when_set_to_zero(monkeypatch):
    cfg = _reload_agentic_config(monkeypatch, CRITIC_ENABLED="0")
    assert cfg.CRITIC_ENABLED is False


def test_worker_query_reformulation_enabled_defaults_to_true_when_unset(monkeypatch):
    cfg = _reload_agentic_config(monkeypatch)
    assert cfg.WORKER_QUERY_REFORMULATION_ENABLED is True


def test_worker_query_reformulation_enabled_false_when_set_to_zero(monkeypatch):
    cfg = _reload_agentic_config(monkeypatch, WORKER_QUERY_REFORMULATION_ENABLED="0")
    assert cfg.WORKER_QUERY_REFORMULATION_ENABLED is False


def _import_critic():
    try:
        from src.agentic_rag import critic
    except ImportError as exc:  # pragma: no cover - depends on local env
        pytest.skip(f"src.agentic_rag.critic not importable here: {exc}")
    return critic


def _import_workers():
    try:
        from src.agentic_rag import workers
    except ImportError as exc:  # pragma: no cover - depends on local env
        pytest.skip(f"src.agentic_rag.workers not importable here: {exc}")
    return workers


def test_critic_agent_passthrough_when_disabled(monkeypatch):
    critic = _import_critic()
    monkeypatch.setattr(critic, "CRITIC_ENABLED", False)

    result = critic.critic_agent({"draft_response": "some draft", "context_gathered": {}, "query": "q"})

    assert result["validation_score"] == 1.0
    assert "CRITIC_ENABLED=0" in result["critic_feedback"]
    assert result["llm_calls"] == []


def test_critic_agent_passthrough_never_triggers_retry_route(monkeypatch):
    """The disabled-critic score must clear VALIDATION_SCORE_THRESHOLD so
    graph._route_from_critic always finalizes rather than retrying."""
    critic = _import_critic()
    from src.agentic_rag.config import VALIDATION_SCORE_THRESHOLD

    monkeypatch.setattr(critic, "CRITIC_ENABLED", False)
    result = critic.critic_agent({"draft_response": "d", "context_gathered": {}, "query": "q"})
    assert result["validation_score"] >= VALIDATION_SCORE_THRESHOLD


def test_reformulate_query_returns_raw_query_when_disabled(monkeypatch):
    workers = _import_workers()
    monkeypatch.setattr(workers, "WORKER_QUERY_REFORMULATION_ENABLED", False)

    reformulated, llm_calls = workers._reformulate_query(
        "what does 1910.119 require?", ["sub task"], "OSHA regulations", existing_llm_calls=[]
    )

    assert reformulated == "what does 1910.119 require?"
    assert llm_calls == []


def test_reformulate_query_preserves_existing_llm_calls_when_disabled(monkeypatch):
    workers = _import_workers()
    monkeypatch.setattr(workers, "WORKER_QUERY_REFORMULATION_ENABLED", False)

    prior_calls = [{"node": "SomeNode", "purpose": "planning"}]
    _reformulated, llm_calls = workers._reformulate_query(
        "q", [], "domain", existing_llm_calls=prior_calls
    )

    assert llm_calls == prior_calls
