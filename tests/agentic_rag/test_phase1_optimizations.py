"""Tests for Phase 1 pipeline optimizations.

Covers:
  1.1 - Metadata filter: full state names (e.g. "Texas" → "TX")
  1.2 - worker_errors field in state / surfacing in supervisor + critic
  1.3 - max_tokens caps on LLM calls
  1.4 - Structured JSON output for planning and critic
  1.5 - print() replaced with logging in graph.py and critic.py
"""

import logging
from unittest.mock import MagicMock, patch

import pytest

from src.agentic_rag.config import (
    CRITIC_EVALUATION_MAX_TOKENS,
    CRITIC_QUANTITATIVE_MAX_TOKENS,
    SUPERVISOR_PLANNING_MAX_TOKENS,
    SUPERVISOR_REVISION_MAX_TOKENS,
    SUPERVISOR_SYNTHESIS_MAX_TOKENS,
    WORKER_QUERY_REFORMULATION_MAX_TOKENS,
)
from src.agentic_rag.workers import (
    _extract_metadata_filters,
    _STATE_NAMES_TO_ABBREV,
    data_analyst_agent,
)
from src.agentic_rag.supervisor import _parse_plan_output, supervisor_agent, supervisor_plan, supervisor_revise
from src.agentic_rag.critic import critic_agent
from src.agentic_rag.graph import _route_from_critic


@pytest.fixture(autouse=True)
def _force_supported_agent_params(monkeypatch):
    monkeypatch.setattr("src.agentic_rag.supervisor.SUPERVISOR_TEMPERATURE", 1.0)
    monkeypatch.setattr("src.agentic_rag.critic.CRITIC_TEMPERATURE", 1.0)
    monkeypatch.setattr("src.agentic_rag.workers.WORKER_TEMPERATURE", 1.0)
    monkeypatch.setattr("src.agentic_rag.workers.REASONING_TEMPERATURE", 1.0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MOCK_RESPONSE_METADATA = {
    "token_usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
    "model_name": "gpt-5-mini",
    "finish_reason": "stop",
    "system_fingerprint": "fp_test",
}


def _make_state(**overrides) -> dict:
    base = {
        "query": "test query",
        "sub_tasks": [],
        "active_agent": "",
        "pending_agents": [],
        "planned_agents": [],
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
    base.update(overrides)
    return base


def _mock_llm_response(content):
    mock = MagicMock()
    mock.content = content
    mock.response_metadata = _MOCK_RESPONSE_METADATA
    return mock


# ===================================================================
# Task 1.1 — Metadata filter for full state names
# ===================================================================


class TestMetadataFilterStateNames:
    """_extract_metadata_filters should handle abbreviations, full names,
    lowercase names, and multi-word state names."""

    def test_uppercase_abbreviation_still_works(self):
        filters = _extract_metadata_filters("violations in TX")
        assert filters is not None
        assert filters["site_state"] == "TX"

    def test_title_case_full_name(self):
        filters = _extract_metadata_filters("OSHA PSM violations in Texas")
        assert filters is not None
        assert filters["site_state"] == "TX"

    def test_lowercase_full_name(self):
        filters = _extract_metadata_filters("OSHA PSM violations in texas")
        assert filters is not None
        assert filters["site_state"] == "TX"

    def test_multiple_states_mixed_format(self):
        filters = _extract_metadata_filters(
            "incidents in California and Texas"
        )
        assert filters is not None
        assert "site_state" in filters
        abbrevs = filters["site_state"]
        assert isinstance(abbrevs, dict)
        assert set(abbrevs["$in"]) == {"CA", "TX"}

    def test_multi_word_state_name(self):
        filters = _extract_metadata_filters("violations in New York refineries")
        assert filters is not None
        assert filters["site_state"] == "NY"

    def test_multi_word_state_lowercase(self):
        filters = _extract_metadata_filters("incidents in north carolina plants")
        assert filters is not None
        assert filters["site_state"] == "NC"

    def test_no_state_returns_none_for_state(self):
        filters = _extract_metadata_filters("general safety improvements")
        assert filters is None

    def test_no_duplicate_when_both_formats_present(self):
        filters = _extract_metadata_filters("violations in TX Texas refineries")
        assert filters is not None
        assert filters["site_state"] == "TX"

    def test_fatality_filter_still_works_alongside_state(self):
        filters = _extract_metadata_filters(
            "fatalities in California refineries"
        )
        assert filters is not None
        assert filters["site_state"] == "CA"
        assert filters["fatality"] is True

    def test_state_names_to_abbrev_has_50_entries(self):
        assert len(_STATE_NAMES_TO_ABBREV) == 50


# ===================================================================
# Task 1.2 — worker_errors field
# ===================================================================


class TestWorkerErrors:
    """worker_errors should be a field on AgentState and surfaced by
    DataAnalystWorker, supervisor synthesis, and critic evaluation."""

    def test_state_has_worker_errors_field(self):
        state = _make_state()
        assert "worker_errors" in state
        assert state["worker_errors"] == []

    @patch("src.agentic_rag.analytical_tools.get_analytical_db")
    def test_data_analyst_populates_worker_errors_on_db_failure(self, mock_db):
        mock_db.side_effect = FileNotFoundError("Data file not found")
        result = data_analyst_agent(_make_state())
        assert "worker_errors" in result
        assert len(result["worker_errors"]) == 1
        err = result["worker_errors"][0]
        assert err["node"] == "DataAnalystWorker"
        assert "Data file not found" in err["error"]
        assert "unavailable" in err["impact"].lower()

    @patch("src.agentic_rag.supervisor.make_chat_model")
    def test_supervisor_synthesis_includes_worker_errors(self, mock_llm_class):
        """Synthesis prompt should include worker error acknowledgment."""
        mock_llm = mock_llm_class.return_value
        mock_response = _mock_llm_response("Synthesized response.")
        mock_llm.invoke.return_value = mock_response

        state = _make_state(
            context_gathered={"csb": ["Some data"]},
            pending_agents=[],
            worker_errors=[{
                "node": "DataAnalystWorker",
                "error": "File not found",
                "impact": "Statistical analysis unavailable.",
            }],
        )
        supervisor_agent(state)

        call_args = mock_llm.invoke.call_args[0][0]
        prompt_text = call_args[1].content
        assert "WORKER FAILURES" in prompt_text
        assert "Statistical analysis unavailable" in prompt_text

    @patch("src.agentic_rag.critic.make_chat_model")
    def test_critic_includes_worker_errors(self, mock_llm_class):
        """Critic evaluation prompt should include worker errors."""
        mock_llm = mock_llm_class.return_value
        mock_response = _mock_llm_response(
            '{"score": 0.7, "critique": "Missing data acknowledgment."}'
        )
        mock_llm.invoke.return_value = mock_response

        state = _make_state(
            context_gathered={"csb": ["Data"]},
            draft_response="Draft without mentioning data gaps.",
            worker_errors=[{
                "node": "DataAnalystWorker",
                "error": "File not found",
                "impact": "Statistical analysis unavailable.",
            }],
        )
        critic_agent(state)

        call_args = mock_llm.invoke.call_args[0][0]
        prompt_text = call_args[1].content
        assert "KNOWN WORKER FAILURES" in prompt_text
        assert "Score lower if it does not" in prompt_text


# ===================================================================
# Task 1.3 — max_tokens caps
# ===================================================================


class TestMaxTokensCaps:
    """Verify max_tokens is set on chat-model instantiations."""

    @patch("src.agentic_rag.supervisor.make_chat_model")
    def test_planning_llm_has_max_tokens(self, mock_llm_class):
        mock_llm = mock_llm_class.return_value
        mock_response = _mock_llm_response(
            '{"sub_tasks": ["A"], "agents_to_call": ["csb_incident"]}'
        )
        mock_llm.invoke.return_value = mock_response

        supervisor_plan(_make_state())
        call_kwargs = mock_llm_class.call_args
        assert call_kwargs.kwargs.get("max_completion_tokens") == SUPERVISOR_PLANNING_MAX_TOKENS

    @patch("src.agentic_rag.supervisor.make_chat_model")
    def test_synthesis_llm_has_max_tokens(self, mock_llm_class):
        mock_llm = mock_llm_class.return_value
        mock_response = _mock_llm_response("Synthesized answer.")
        mock_llm.invoke.return_value = mock_response

        state = _make_state(
            context_gathered={"csb": ["Facts"]},
            pending_agents=[],
        )
        supervisor_agent(state)
        call_kwargs = mock_llm_class.call_args
        assert call_kwargs.kwargs.get("max_completion_tokens") == SUPERVISOR_SYNTHESIS_MAX_TOKENS

    @patch("src.agentic_rag.supervisor.make_chat_model")
    def test_revision_llm_has_max_tokens(self, mock_llm_class):
        mock_llm = mock_llm_class.return_value
        mock_response = _mock_llm_response("Revised answer.")
        mock_llm.invoke.return_value = mock_response

        state = _make_state(
            draft_response="Draft.",
            critic_feedback="Issue found.",
            context_gathered={"csb": ["Facts"]},
        )
        supervisor_revise(state)
        call_kwargs = mock_llm_class.call_args
        assert call_kwargs.kwargs.get("max_completion_tokens") == SUPERVISOR_REVISION_MAX_TOKENS

    @patch("src.agentic_rag.critic.make_chat_model")
    def test_critic_llm_has_max_tokens(self, mock_llm_class):
        mock_llm = mock_llm_class.return_value
        mock_response = _mock_llm_response('{"score": 0.9, "critique": "Good."}')
        mock_llm.invoke.return_value = mock_response

        state = _make_state(
            context_gathered={"csb": ["Facts"]},
            draft_response="Draft.",
        )
        critic_agent(state)
        call_kwargs = mock_llm_class.call_args
        assert call_kwargs.kwargs.get("max_completion_tokens") == CRITIC_EVALUATION_MAX_TOKENS

    @patch("src.agentic_rag.critic.make_chat_model")
    def test_critic_quantitative_llm_has_lower_max_tokens(self, mock_llm_class):
        mock_llm = mock_llm_class.return_value
        mock_response = _mock_llm_response('{"score": 0.9, "critique": "Counts match."}')
        mock_llm.invoke.return_value = mock_response

        state = _make_state(
            query="How many violations in Texas?",
            context_gathered={"osha_enf": ["Records"]},
            draft_response="There were 42 violations.",
        )
        critic_agent(state)
        call_kwargs = mock_llm_class.call_args
        assert call_kwargs.kwargs.get("max_completion_tokens") == CRITIC_QUANTITATIVE_MAX_TOKENS

    @patch("src.agentic_rag.workers.make_chat_model")
    @patch("src.agentic_rag.workers.TOOLS", new_callable=dict)
    def test_reformulation_llm_has_max_tokens(self, mock_tools, mock_llm_class):
        mock_engine = MagicMock()
        mock_engine.query.return_value = ("Results", [{"collection": "osha_enf"}])
        mock_tools["osha_enf"] = mock_engine

        mock_llm = mock_llm_class.return_value
        mock_response = _mock_llm_response("reformulated query")
        mock_llm.invoke.return_value = mock_response

        from src.agentic_rag.workers import osha_enf_agent
        osha_enf_agent(_make_state())

        first_call_kwargs = mock_llm_class.call_args_list[0]
        assert first_call_kwargs.kwargs.get("max_completion_tokens") == WORKER_QUERY_REFORMULATION_MAX_TOKENS


# ===================================================================
# Task 1.4 — Structured JSON output
# ===================================================================


class TestStructuredJsonOutput:
    """Planning and critic should use response_format=json_object."""

    @patch("src.agentic_rag.supervisor.make_chat_model")
    def test_planning_uses_json_response_format(self, mock_llm_class):
        mock_llm = mock_llm_class.return_value
        mock_response = _mock_llm_response(
            '{"sub_tasks": ["A"], "agents_to_call": ["csb_incident"]}'
        )
        mock_llm.invoke.return_value = mock_response

        supervisor_plan(_make_state())
        call_kwargs = mock_llm_class.call_args
        assert call_kwargs.kwargs.get("response_format") == {"type": "json_object"}

    @patch("src.agentic_rag.critic.make_chat_model")
    def test_critic_uses_json_response_format(self, mock_llm_class):
        mock_llm = mock_llm_class.return_value
        mock_response = _mock_llm_response('{"score": 0.9, "critique": "Good."}')
        mock_llm.invoke.return_value = mock_response

        state = _make_state(
            context_gathered={"csb": ["Facts"]},
            draft_response="Draft.",
        )
        critic_agent(state)
        call_kwargs = mock_llm_class.call_args
        assert call_kwargs.kwargs.get("response_format") == {"type": "json_object"}

    def test_parse_plan_output_raw_json(self):
        raw = '{"sub_tasks": ["A"], "agents_to_call": ["osha_regs"]}'
        parsed = _parse_plan_output(raw)
        assert parsed["agents_to_call"] == ["osha_regs"]

    def test_parse_plan_output_fallback_on_invalid(self):
        parsed = _parse_plan_output("not json at all")
        assert parsed["agents_to_call"] == []
        assert parsed["sub_tasks"] == ["Failed to plan"]


# ===================================================================
# Task 1.5 — Logging replaces print()
# ===================================================================


class TestLoggingReplacedPrint:
    """graph.py and critic.py should use logging, not print()."""

    def test_graph_route_from_critic_logs_warning_on_max_retries(self, caplog):
        with caplog.at_level(logging.WARNING, logger="src.agentic_rag.graph"):
            result = _route_from_critic(
                {"validation_score": 0.5, "retry_count": 2}
            )
        assert result == "FinalizeNode"
        assert "Max retries" in caplog.text

    def test_graph_route_from_critic_logs_info_on_retry(self, caplog):
        with caplog.at_level(logging.INFO, logger="src.agentic_rag.graph"):
            result = _route_from_critic(
                {"validation_score": 0.5, "retry_count": 0}
            )
        assert result == "RetryNode"
        assert "Critique failed" in caplog.text

    @patch("src.agentic_rag.critic.make_chat_model")
    def test_critic_logs_error_on_failure(self, mock_llm_class, caplog):
        mock_llm = mock_llm_class.return_value
        mock_llm.invoke.side_effect = RuntimeError("LLM unavailable")

        state = _make_state(
            context_gathered={"csb": ["Facts"]},
            draft_response="Draft.",
        )
        with caplog.at_level(logging.ERROR, logger="src.agentic_rag.critic"):
            result = critic_agent(state)

        assert result["validation_score"] == 0.0
        assert "Critic failed to evaluate" in caplog.text

    def test_no_print_in_graph_module(self):
        """Verify no print() calls in graph.py source."""
        import inspect
        from src.agentic_rag import graph
        source = inspect.getsource(graph)
        # Exclude string literals and comments
        lines = [
            line for line in source.split("\n")
            if line.strip().startswith("print(")
        ]
        assert lines == [], f"Found print() calls in graph.py: {lines}"

    def test_no_print_in_critic_module(self):
        """Verify no print() calls in critic.py source."""
        import inspect
        from src.agentic_rag import critic
        source = inspect.getsource(critic)
        lines = [
            line for line in source.split("\n")
            if line.strip().startswith("print(")
        ]
        assert lines == [], f"Found print() calls in critic.py: {lines}"
