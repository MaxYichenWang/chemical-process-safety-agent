from unittest.mock import MagicMock

import pytest

from src.agentic_rag.llm_logger import (
    _estimate_cost,
    record_llm_call,
    validate_model_parameters,
)


def test_estimate_cost_uses_model_pricing():
    """Cost estimate should follow configured per-token rates."""
    cost = _estimate_cost("gpt-5-mini", prompt_tokens=1000, completion_tokens=500)
    assert round(cost, 8) == 0.00125


def test_estimate_cost_handles_versioned_model_name():
    """Versioned model names should resolve by model prefix."""
    cost = _estimate_cost("gpt-5-mini-2026-01-01", prompt_tokens=1000, completion_tokens=500)
    assert round(cost, 8) == 0.00125


def test_record_llm_call_includes_estimated_cost():
    """record_llm_call should persist estimated_cost_usd in call metadata."""
    response = MagicMock()
    response.content = "ok"
    response.response_metadata = {
        "token_usage": {"prompt_tokens": 1000, "completion_tokens": 500, "total_tokens": 1500},
        "model_name": "gpt-5-mini",
        "finish_reason": "stop",
    }

    calls = record_llm_call(
        response=response,
        node_name="TestNode",
        call_purpose="unit_test",
        model_config="gpt-5-mini",
        temperature=0.1,
        system_prompt="system",
        user_prompt="user",
        latency_s=0.12,
        existing_calls=[],
    )

    assert len(calls) == 1
    assert calls[0]["estimated_cost_usd"] == 0.00125


def test_validate_model_parameters_rejects_gpt5_non_default_temperature():
    with pytest.raises(ValueError, match="temperature"):
        validate_model_parameters(model="gpt-5-mini", temperature=0.0)


def test_validate_model_parameters_rejects_gpt5_max_tokens():
    with pytest.raises(ValueError, match="max_tokens"):
        validate_model_parameters(model="gpt-5-mini", max_tokens=128)


def test_validate_model_parameters_allows_gpt5_default_shape():
    validate_model_parameters(
        model="gpt-5-mini",
        temperature=1.0,
        top_p=1.0,
        max_completion_tokens=128,
    )


def test_validate_model_parameters_rejects_o1_invalid_params():
    with pytest.raises(ValueError, match="max_tokens"):
        validate_model_parameters(model="o1-preview", max_tokens=100)

    with pytest.raises(ValueError, match="temperature"):
        validate_model_parameters(model="o1-mini", temperature=0.0)

    with pytest.raises(ValueError, match="top_p"):
        validate_model_parameters(model="o3-mini", top_p=0.9)


def test_validate_model_parameters_allows_o1_valid_shape():
    validate_model_parameters(
        model="o1",
        temperature=1.0,
        top_p=1.0,
        max_completion_tokens=500,
    )


def test_validate_model_parameters_rejects_claude_temperature_out_of_range():
    with pytest.raises(ValueError, match="temperature"):
        validate_model_parameters(model="claude-3-5-sonnet-20241022", temperature=1.5)


def test_validate_model_parameters_allows_claude_valid_shape():
    validate_model_parameters(
        model="claude-3-5-sonnet-20241022",
        temperature=0.0,
        max_completion_tokens=1000,
    )
