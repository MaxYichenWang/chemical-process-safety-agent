"""Tests for the provider-agnostic chat-model factory.

Most of this file exercises resolve_role_config, which is pure and needs no
provider libraries. A handful of make_chat_model dispatch tests at the bottom
patch the lazily-imported langchain_openai/langchain_anthropic classes
directly (requires those packages to be installed, same as the pipeline
machine's requirements.txt) to lock in the new explicit-argument-override
behavior end to end.
"""

import sys
import types
from unittest.mock import patch

import pytest

if "langchain_anthropic" not in sys.modules:
    try:
        import langchain_anthropic
    except ImportError:
        mod = types.ModuleType("langchain_anthropic")
        mod.ChatAnthropic = type("ChatAnthropic", (), {})
        sys.modules["langchain_anthropic"] = mod

from src.agentic_rag.llm_factory import make_chat_model, resolve_role_config, resolve_model_name


def test_defaults_preserve_openai_behavior():
    cfg = resolve_role_config("supervisor", env={})
    assert cfg.provider == "openai"
    assert cfg.model == "gpt-5-mini"
    assert cfg.api_key_env == "OPENAI_API_KEY"
    assert cfg.base_url is None


def test_data_analyst_falls_back_to_worker_model():
    cfg = resolve_role_config("data_analyst", env={"AGENT_WORKER_MODEL": "gpt-5-mini"})
    assert cfg.model == "gpt-5-mini"


def test_anthropic_override():
    env = {
        "AGENT_SUPERVISOR_PROVIDER": "anthropic",
        "AGENT_SUPERVISOR_MODEL": "claude-sonnet-4-6",
        "AGENT_SUPERVISOR_API_KEY_ENV": "ANTHROPIC_API_KEY",
    }
    cfg = resolve_role_config("supervisor", env=env)
    assert cfg.provider == "anthropic"
    assert cfg.model == "claude-sonnet-4-6"
    assert cfg.api_key_env == "ANTHROPIC_API_KEY"


def test_openai_compatible_with_base_url():
    env = {
        "AGENT_WORKER_PROVIDER": "openai_compatible",
        "AGENT_WORKER_MODEL": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        "AGENT_WORKER_BASE_URL": "https://api.together.xyz/v1",
        "AGENT_WORKER_API_KEY_ENV": "TOGETHER_API_KEY",
    }
    cfg = resolve_role_config("worker", env=env)
    assert cfg.provider == "openai_compatible"
    assert cfg.base_url == "https://api.together.xyz/v1"
    assert cfg.api_key_env == "TOGETHER_API_KEY"


def test_temperature_parsed_when_set():
    cfg = resolve_role_config("critic", env={"AGENT_CRITIC_TEMPERATURE": "0.3"})
    assert cfg.temperature == 0.3


def test_unknown_role_raises():
    with pytest.raises(ValueError, match="unknown role"):
        resolve_role_config("bogus", env={})


def test_resolve_model_name_reads_env(monkeypatch):
    monkeypatch.setenv("AGENT_SUPERVISOR_MODEL", "gpt-5")
    assert resolve_model_name("supervisor") == "gpt-5"


# ---------------------------------------------------------------------------
# Explicit-argument overrides: precedence is arg > env var > built-in default,
# without ever mutating os.environ. See llm_factory.py module docstring.
# ---------------------------------------------------------------------------

def test_explicit_provider_and_model_win_over_env():
    env = {"AGENT_SUPERVISOR_MODEL": "gpt-5-mini", "AGENT_SUPERVISOR_PROVIDER": "openai"}
    cfg = resolve_role_config("supervisor", env=env, provider="anthropic", model="claude-sonnet-4-6")
    assert cfg.provider == "anthropic"
    assert cfg.model == "claude-sonnet-4-6"


def test_omitted_args_fall_back_to_env_unchanged():
    cfg = resolve_role_config("supervisor", env={"AGENT_SUPERVISOR_MODEL": "gpt-5-mini"})
    assert cfg.model == "gpt-5-mini"


def test_explicit_base_url_and_api_key_env_win_over_env():
    env = {"AGENT_WORKER_BASE_URL": "https://should-not-be-used", "AGENT_WORKER_API_KEY_ENV": "SHOULD_NOT_BE_USED"}
    cfg = resolve_role_config(
        "worker", env=env, base_url="https://api.together.xyz/v1", api_key_env="TOGETHER_API_KEY"
    )
    assert cfg.base_url == "https://api.together.xyz/v1"
    assert cfg.api_key_env == "TOGETHER_API_KEY"


def test_explicit_args_resolve_openai_compatible_provider_with_empty_env():
    """A caller can fully specify a new-style provider (Gemini, Qwen, ...) purely
    via arguments, with no env vars set at all -- the sweep's in-process path."""
    cfg = resolve_role_config(
        "worker",
        env={},
        provider="openai_compatible",
        model="qwen-plus",
        base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        api_key_env="DASHSCOPE_API_KEY",
    )
    assert cfg.provider == "openai_compatible"
    assert cfg.model == "qwen-plus"
    assert cfg.base_url == "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
    assert cfg.api_key_env == "DASHSCOPE_API_KEY"


def test_resolve_model_name_accepts_explicit_override():
    assert resolve_model_name("supervisor", model="gpt-5") == "gpt-5"


# ---------------------------------------------------------------------------
# eval_judge role: same resolution machinery, EVAL_LLM_* env prefix, default
# model preserved to match src/evaluation/config.py's EVAL_LLM_MODEL default.
# ---------------------------------------------------------------------------

def test_eval_judge_default_model_matches_eval_config_default():
    cfg = resolve_role_config("eval_judge", env={})
    assert cfg.provider == "openai"
    assert cfg.model == "gpt-5.4-mini"
    assert cfg.api_key_env == "OPENAI_API_KEY"


def test_eval_judge_reads_eval_llm_env_prefix():
    cfg = resolve_role_config("eval_judge", env={"EVAL_LLM_MODEL": "gpt-5.4-mini", "EVAL_LLM_TEMPERATURE": "1.0"})
    assert cfg.model == "gpt-5.4-mini"
    assert cfg.temperature == 1.0


def test_eval_judge_explicit_override_to_anthropic():
    cfg = resolve_role_config("eval_judge", env={}, provider="anthropic", model="claude-sonnet-4-6")
    assert cfg.provider == "anthropic"
    assert cfg.model == "claude-sonnet-4-6"
    assert cfg.api_key_env == "ANTHROPIC_API_KEY"


def test_eval_judge_included_in_unknown_role_error():
    with pytest.raises(ValueError, match="eval_judge"):
        resolve_role_config("bogus", env={})


# ---------------------------------------------------------------------------
# make_chat_model dispatch: confirms the new provider/model/base_url/
# api_key_env kwargs actually reach the constructed LangChain chat model, and
# that the default (no-args) path stays byte-for-byte identical to before.
# ---------------------------------------------------------------------------

@patch("langchain_openai.ChatOpenAI")
def test_make_chat_model_default_path_unchanged(mock_chat_openai, monkeypatch):
    for var in ("AGENT_SUPERVISOR_PROVIDER", "AGENT_SUPERVISOR_MODEL", "AGENT_SUPERVISOR_BASE_URL"):
        monkeypatch.delenv(var, raising=False)
    make_chat_model("supervisor", temperature=1.0, max_completion_tokens=99999)
    _, kwargs = mock_chat_openai.call_args
    assert kwargs["model"] == "gpt-5-mini"
    assert kwargs["temperature"] == 1.0
    assert "base_url" not in kwargs


@patch("langchain_anthropic.ChatAnthropic")
def test_make_chat_model_explicit_anthropic_override(mock_chat_anthropic, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    make_chat_model("supervisor", provider="anthropic", model="claude-sonnet-4-6", temperature=1.0)
    _, kwargs = mock_chat_anthropic.call_args
    assert kwargs["model"] == "claude-sonnet-4-6"
    assert kwargs["api_key"] == "sk-ant-test"


@patch("langchain_openai.ChatOpenAI")
def test_make_chat_model_explicit_openai_compatible_override(mock_chat_openai, monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "sk-dashscope-test")
    make_chat_model(
        "worker",
        provider="openai_compatible",
        model="qwen-plus",
        base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        api_key_env="DASHSCOPE_API_KEY",
        temperature=1.0,
    )
    _, kwargs = mock_chat_openai.call_args
    assert kwargs["model"] == "qwen-plus"
    assert kwargs["base_url"] == "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
    assert kwargs["api_key"] == "sk-dashscope-test"


@patch("langchain_openai.ChatOpenAI")
def test_make_chat_model_eval_judge_default_path(mock_chat_openai, monkeypatch):
    for var in ("EVAL_LLM_PROVIDER", "EVAL_LLM_MODEL"):
        monkeypatch.delenv(var, raising=False)
    make_chat_model("eval_judge")
    _, kwargs = mock_chat_openai.call_args
    assert kwargs["model"] == "gpt-5.4-mini"


# ---------------------------------------------------------------------------
# Provider-aware temperature default -- 0.0 unless neither an explicit
# override nor an env var is set AND the resolved model is GPT-5-family
# (which rejects anything but 1.0).
# ---------------------------------------------------------------------------

def test_deepseek_role_defaults_to_temperature_zero():
    cfg = resolve_role_config("supervisor", env={"AGENT_SUPERVISOR_MODEL": "deepseek-v4-flash"})
    assert cfg.temperature == 0.0


def test_gpt5_role_defaults_to_temperature_one():
    cfg = resolve_role_config("supervisor", env={"AGENT_SUPERVISOR_MODEL": "gpt-5-mini"})
    assert cfg.temperature == 1.0


def test_gpt5_variant_models_all_default_to_temperature_one():
    for model in ("gpt-5", "gpt-5-mini", "gpt-5-nano", "gpt-5.4-mini", "gpt-5.6-sol"):
        cfg = resolve_role_config("worker", env={"AGENT_WORKER_MODEL": model})
        assert cfg.temperature == 1.0, model


def test_non_gpt5_models_default_to_temperature_zero():
    for model in ("deepseek-v4-flash", "kimi-k2.6", "MiniMax-M3", "qwen3.7-flash", "glm-5.2", "claude-sonnet-5"):
        cfg = resolve_role_config("worker", env={"AGENT_WORKER_MODEL": model})
        assert cfg.temperature == 0.0, model


def test_explicit_env_temperature_overrides_provider_aware_default():
    """A user who explicitly sets AGENT_<ROLE>_TEMPERATURE still wins, even
    against a model that would otherwise get a different default."""
    cfg = resolve_role_config(
        "supervisor",
        env={"AGENT_SUPERVISOR_MODEL": "deepseek-v4-flash", "AGENT_SUPERVISOR_TEMPERATURE": "0.7"},
    )
    assert cfg.temperature == 0.7


def test_optional_temperature_env_helper_returns_none_when_unset(monkeypatch):
    """agentic_rag/config.py's *_TEMPERATURE constants are optional
    overrides now (None means "let resolve_role_config's provider-aware
    default decide"), not a blanket 1.0. Exercises the parsing helper
    directly rather than reimporting the module (which would re-run
    ``load_dotenv()`` against this machine's real local ``.env`` -- for this
    repo specifically, that file currently still pins
    ``SUPERVISOR_TEMPERATURE=1.0`` etc., so a reload-based test would pass or
    fail depending on the developer's local, untracked ``.env`` contents
    rather than the code under test)."""
    from src.agentic_rag.config import _optional_temperature_env

    monkeypatch.delenv("SOME_ROLE_TEMPERATURE", raising=False)
    assert _optional_temperature_env("SOME_ROLE_TEMPERATURE") is None

    monkeypatch.setenv("SOME_ROLE_TEMPERATURE", "0.7")
    assert _optional_temperature_env("SOME_ROLE_TEMPERATURE") == 0.7

    monkeypatch.setenv("SOME_ROLE_TEMPERATURE", "")
    assert _optional_temperature_env("SOME_ROLE_TEMPERATURE") is None
