import pytest
from unittest.mock import MagicMock, patch

from src.agentic_rag.critic import _is_quantitative, critic_agent
from src.agentic_rag.config import CRITIC_EVALUATION_MAX_TOKENS, CRITIC_QUANTITATIVE_MAX_TOKENS
from src.agentic_rag.state import AgentState


@pytest.fixture(autouse=True)
def _force_supported_critic_params(monkeypatch):
    monkeypatch.setattr("src.agentic_rag.critic.CRITIC_TEMPERATURE", 1.0)


@pytest.fixture
def initial_state():
    return AgentState(
        query="What caused the explosion?",
        sub_tasks=[],
        active_agent="",
        pending_agents=[],
        planned_agents=[],
        context_gathered={"docs": ["Fact 1", "Fact 2"]},
        draft_response="Draft combining Fact 1 and Fact 2.",
        final_response="",
        citations=[],
        validation_score=0.0,
        critic_feedback="",
        retry_count=0,
        llm_calls=[],
        retrieval_details=[],
        worker_errors=[],
    )


_MOCK_RESPONSE_METADATA = {
    "token_usage": {"prompt_tokens": 200, "completion_tokens": 80, "total_tokens": 280},
    "model_name": "gpt-5-mini",
    "finish_reason": "stop",
    "system_fingerprint": "fp_test",
}


@patch("src.agentic_rag.critic.make_chat_model")
def test_critic_agent_pass(mock_llm_class, initial_state):
    """Test critic parses a passing score and preserves critique text."""
    mock_llm = mock_llm_class.return_value
    mock_response = MagicMock()
    mock_response.content = '{"score": 0.95, "critique": "Well grounded in source documents."}'
    mock_response.response_metadata = _MOCK_RESPONSE_METADATA
    mock_llm.invoke.return_value = mock_response

    state = critic_agent(initial_state)

    assert state["validation_score"] == 0.95
    assert state["critic_feedback"] == "Well grounded in source documents."
    assert "llm_calls" in state
    assert len(state["llm_calls"]) == 1
    assert state["llm_calls"][0]["call_purpose"] == "validation"


@patch("src.agentic_rag.critic.make_chat_model")
def test_critic_agent_fail_with_feedback(mock_llm_class, initial_state):
    """Test critic assigns failing score and provides actionable feedback."""
    mock_llm = mock_llm_class.return_value
    mock_response = MagicMock()
    mock_response.content = (
        '{"score": 0.3, "critique": "The draft fabricates a chemical name not in context."}'
    )
    mock_response.response_metadata = _MOCK_RESPONSE_METADATA
    mock_llm.invoke.return_value = mock_response

    state = critic_agent(initial_state)

    assert state["validation_score"] == 0.3
    assert "fabricates" in state["critic_feedback"]


@patch("src.agentic_rag.critic.make_chat_model")
def test_critic_agent_json_in_codeblock(mock_llm_class, initial_state):
    """Test critic handles JSON wrapped in markdown code block."""
    mock_llm = mock_llm_class.return_value
    mock_response = MagicMock()
    mock_response.content = '```json\n{"score": 0.85, "critique": "Minor issue."}\n```'
    mock_response.response_metadata = _MOCK_RESPONSE_METADATA
    mock_llm.invoke.return_value = mock_response

    state = critic_agent(initial_state)

    assert state["validation_score"] == 0.85
    assert state["critic_feedback"] == "Minor issue."


@patch("src.agentic_rag.critic.make_chat_model")
def test_critic_agent_error_handling(mock_llm_class, initial_state):
    """Test critic sets score to 0 and provides error feedback on parse failure."""
    mock_llm = mock_llm_class.return_value
    mock_response = MagicMock()
    mock_response.content = "I didn't output JSON."
    mock_llm.invoke.return_value = mock_response

    state = critic_agent(initial_state)

    assert state["validation_score"] == 0.0
    assert "failed" in state["critic_feedback"].lower()


# ---------------------------------------------------------------------------
# _is_quantitative classifier tests
# ---------------------------------------------------------------------------


def test_is_quantitative_count():
    assert _is_quantitative("How many OSHA violations in Texas?") is True


def test_is_quantitative_total():
    assert _is_quantitative("What is the total number of incidents?") is True


def test_is_quantitative_trend():
    assert _is_quantitative("Show me the trend of violations over time.") is True


def test_is_quantitative_average():
    assert _is_quantitative("What is the average penalty amount?") is True


def test_is_quantitative_percentage():
    assert _is_quantitative("What percentage of inspections found violations?") is True


def test_is_not_quantitative_qualitative():
    assert _is_quantitative("What caused the BP Texas City explosion?") is False


def test_is_not_quantitative_regulation():
    assert _is_quantitative("What are the PSM requirements for MOC?") is False


# ---------------------------------------------------------------------------
# Adaptive critic — quantitative vs qualitative prompts
# ---------------------------------------------------------------------------


@patch("src.agentic_rag.critic.make_chat_model")
def test_critic_uses_quantitative_prompt_for_count_query(mock_llm_class, initial_state):
    """Quantitative queries use the lighter quantitative prompt."""
    initial_state["query"] = "How many OSHA PSM violations were issued in Texas?"

    mock_llm = mock_llm_class.return_value
    mock_response = MagicMock()
    mock_response.content = '{"score": 0.9, "critique": "Counts match."}'
    mock_response.response_metadata = _MOCK_RESPONSE_METADATA
    mock_llm.invoke.return_value = mock_response

    critic_agent(initial_state)

    call_kwargs = mock_llm_class.call_args
    assert call_kwargs.kwargs.get("max_completion_tokens") == CRITIC_QUANTITATIVE_MAX_TOKENS

    # Verify the quantitative system prompt was used (shorter, data-quality focused)
    call_args = mock_llm.invoke.call_args[0][0]
    system_text = call_args[0].content
    assert "data-quality reviewer" in system_text


@patch("src.agentic_rag.critic.make_chat_model")
def test_critic_uses_qualitative_prompt_for_incident_query(mock_llm_class, initial_state):
    """Qualitative queries use the full validation prompt."""
    initial_state["query"] = "What caused the BP Texas City explosion?"

    mock_llm = mock_llm_class.return_value
    mock_response = MagicMock()
    mock_response.content = '{"score": 0.85, "critique": "Good."}'
    mock_response.response_metadata = _MOCK_RESPONSE_METADATA
    mock_llm.invoke.return_value = mock_response

    critic_agent(initial_state)

    call_kwargs = mock_llm_class.call_args
    assert call_kwargs.kwargs.get("max_completion_tokens") == CRITIC_EVALUATION_MAX_TOKENS

    call_args = mock_llm.invoke.call_args[0][0]
    system_text = call_args[0].content
    assert "fact-checker" in system_text
