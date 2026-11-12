"""Tests for the eval-judge multi-provider factory (src/evaluation/judge_factory.py).

Mirrors the pattern in tests/agentic_rag/test_llm_factory.py: patches the
lazily-imported langchain_openai/langchain_anthropic classes to verify
dispatch, and stubs deepeval.models.base_model.DeepEvalBaseLLM (a small,
stable interface) so the DeepEval-adapter path is testable without requiring
deepeval to be installed.
"""

import sys
import types
from importlib.machinery import ModuleSpec
from unittest.mock import patch

import pytest

if "ragas" not in sys.modules or "ragas.llms" not in sys.modules:
    try:
        import ragas
        import ragas.llms
    except ImportError:
        ragas_mod = sys.modules.get("ragas", types.ModuleType("ragas"))
        ragas_llms_mod = types.ModuleType("ragas.llms")
        ragas_llms_mod.llm_factory = lambda *a, **kw: None
        ragas_mod.llms = ragas_llms_mod
        sys.modules["ragas"] = ragas_mod
        sys.modules["ragas.llms"] = ragas_llms_mod

from src.evaluation.judge_factory import (
    judge_call_stats,
    make_ragas_embeddings,
    make_deepeval_judge_model,
    make_ragas_judge_llm,
)


@pytest.fixture(autouse=True)
def _clean_eval_llm_env(monkeypatch):
    # IMPORTANT: stub anthropic/langchain_anthropic here (fixture-scoped, via
    # monkeypatch) rather than at module level. A module-level
    # `types.ModuleType("anthropic")` leaves `sys.modules["anthropic"]` with
    # `__spec__ = None` for the rest of the pytest process, which makes
    # `importlib.util.find_spec("anthropic")` raise `ValueError` instead of
    # returning None -- and that's exactly what `ragas` -> `instructor` calls
    # during its own import, breaking collection of every test file that
    # imports ragas afterward. Keeping this fixture-scoped (with a real
    # ModuleSpec, cleaned up by monkeypatch after each test) avoids poisoning
    # the rest of the session.
    for var in (
        "EVAL_LLM_PROVIDER",
        "EVAL_LLM_MODEL",
        "EVAL_LLM_BASE_URL",
        "EVAL_LLM_API_KEY_ENV",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "DEEPSEEK_API_KEY",
    ):
        monkeypatch.delenv(var, raising=False)
    if "langchain_anthropic" not in sys.modules:
        langchain_anthropic_mod = types.ModuleType("langchain_anthropic")
        langchain_anthropic_mod.__spec__ = ModuleSpec("langchain_anthropic", loader=None)
        langchain_anthropic_mod.ChatAnthropic = object
        monkeypatch.setitem(sys.modules, "langchain_anthropic", langchain_anthropic_mod)
    if "anthropic" not in sys.modules:
        anthropic_mod = types.ModuleType("anthropic")
        anthropic_mod.__spec__ = ModuleSpec("anthropic", loader=None)
        anthropic_mod.AsyncAnthropic = object
        monkeypatch.setitem(sys.modules, "anthropic", anthropic_mod)


# ---------------------------------------------------------------------------
# make_ragas_judge_llm: RAGAS 0.4 native llm_factory path.
# ---------------------------------------------------------------------------

@patch("ragas.llms.llm_factory")
@patch("openai.AsyncOpenAI")
def test_make_ragas_judge_llm_default_path_uses_ragas_llm_factory(mock_async_openai, mock_llm_factory):
    mock_llm_factory.return_value = object()

    result = make_ragas_judge_llm()

    assert result is mock_llm_factory.return_value
    mock_async_openai.assert_called_once_with()
    mock_llm_factory.assert_called_once()
    args, kwargs = mock_llm_factory.call_args
    assert args == ("gpt-5.4-mini",)
    assert kwargs["provider"] == "openai"
    assert kwargs["client"] is mock_async_openai.return_value


@patch("ragas.llms.llm_factory")
@patch("anthropic.AsyncAnthropic")
def test_make_ragas_judge_llm_explicit_anthropic_override(mock_async_anthropic, mock_llm_factory, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    mock_llm_factory.return_value = object()

    make_ragas_judge_llm(provider="anthropic", model="claude-sonnet-4-6")

    mock_async_anthropic.assert_called_once_with(api_key="sk-ant-test")
    args, kwargs = mock_llm_factory.call_args
    assert args == ("claude-sonnet-4-6",)
    assert kwargs["provider"] == "anthropic"
    assert kwargs["client"] is mock_async_anthropic.return_value


@patch("ragas.llms.llm_factory")
@patch("openai.AsyncOpenAI")
def test_make_ragas_judge_llm_forwards_temperature(mock_async_openai, mock_llm_factory):
    make_ragas_judge_llm(temperature=1.0)
    _, kwargs = mock_llm_factory.call_args
    assert kwargs["temperature"] == 1.0


@patch("ragas.llms.llm_factory")
@patch("openai.AsyncOpenAI")
def test_make_ragas_judge_llm_openai_compatible_uses_openai_client_with_base_url(
    mock_async_openai, mock_llm_factory, monkeypatch
):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-ds-test")
    make_ragas_judge_llm(
        provider="openai_compatible",
        model="deepseek-v4-pro",
        base_url="https://api.deepseek.com/v1",
        api_key_env="DEEPSEEK_API_KEY",
    )

    mock_async_openai.assert_called_once_with(
        api_key="sk-ds-test",
        base_url="https://api.deepseek.com/v1",
    )
    _, kwargs = mock_llm_factory.call_args
    assert kwargs["provider"] == "openai"
    assert kwargs["client"] is mock_async_openai.return_value


def test_make_ragas_embeddings_uses_ragas_embedding_factory(monkeypatch):
    openai_mod = types.ModuleType("openai")
    ragas_mod = types.ModuleType("ragas")
    ragas_embeddings_mod = types.ModuleType("ragas.embeddings")
    captured = {}

    class _FakeAsyncOpenAI:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    def _fake_embedding_factory(**kwargs):
        captured.update(kwargs)
        return "ragas-embeddings"

    openai_mod.AsyncOpenAI = _FakeAsyncOpenAI
    ragas_embeddings_mod.embedding_factory = _fake_embedding_factory
    monkeypatch.setitem(sys.modules, "openai", openai_mod)
    monkeypatch.setitem(sys.modules, "ragas", ragas_mod)
    monkeypatch.setitem(sys.modules, "ragas.embeddings", ragas_embeddings_mod)

    embeddings = make_ragas_embeddings(model="text-embedding-test")

    assert embeddings == "ragas-embeddings"
    assert captured["provider"] == "openai"
    assert captured["model"] == "text-embedding-test"
    assert isinstance(captured["client"], _FakeAsyncOpenAI)


# ---------------------------------------------------------------------------
# make_deepeval_judge_model: openai path returns the bare string (unchanged
# behavior); every other provider must return a DeepEvalBaseLLM-shaped
# adapter, since DeepEval's own string resolution only understands OpenAI ids.
# ---------------------------------------------------------------------------

def test_make_deepeval_judge_model_default_path_returns_bare_string():
    result = make_deepeval_judge_model()
    assert result == "gpt-5.4-mini"
    assert isinstance(result, str)


def test_make_deepeval_judge_model_explicit_openai_override_returns_bare_string():
    result = make_deepeval_judge_model(model="gpt-5.4")
    assert result == "gpt-5.4"
    assert isinstance(result, str)


class _FakeDeepEvalBaseLLM:
    """Stand-in for deepeval.models.base_model.DeepEvalBaseLLM -- same shape
    (a base class judge_factory subclasses and implements 4 methods on)."""


@pytest.fixture
def stub_deepeval(monkeypatch):
    deepeval_mod = types.ModuleType("deepeval")
    deepeval_models_mod = types.ModuleType("deepeval.models")
    deepeval_base_model_mod = types.ModuleType("deepeval.models.base_model")
    deepeval_base_model_mod.DeepEvalBaseLLM = _FakeDeepEvalBaseLLM
    monkeypatch.setitem(sys.modules, "deepeval", deepeval_mod)
    monkeypatch.setitem(sys.modules, "deepeval.models", deepeval_models_mod)
    monkeypatch.setitem(sys.modules, "deepeval.models.base_model", deepeval_base_model_mod)


@patch("langchain_anthropic.ChatAnthropic")
def test_make_deepeval_judge_model_anthropic_returns_adapter(mock_chat_anthropic, monkeypatch, stub_deepeval):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    mock_chat_anthropic.return_value.invoke.return_value = types.SimpleNamespace(content="anthropic-says-hi")

    adapter = make_deepeval_judge_model(provider="anthropic", model="claude-sonnet-4-6")

    assert not isinstance(adapter, str)
    assert isinstance(adapter, _FakeDeepEvalBaseLLM)
    assert adapter.get_model_name() == "claude-sonnet-4-6"
    assert adapter.generate("hi") == "anthropic-says-hi"
    assert adapter.load_model() is mock_chat_anthropic.return_value


@patch("langchain_openai.ChatOpenAI")
def test_make_deepeval_judge_model_openai_compatible_returns_adapter(mock_chat_openai, monkeypatch, stub_deepeval):
    """DeepSeek/Gemini/Qwen/MiniMax/Kimi all resolve to provider=openai_compatible
    -- confirms the adapter path (not the bare-string path) covers all of them,
    not just anthropic."""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-ds-test")

    adapter = make_deepeval_judge_model(
        provider="openai_compatible",
        model="deepseek-v4-flash",
        base_url="https://api.deepseek.com/v1",
        api_key_env="DEEPSEEK_API_KEY",
    )

    assert isinstance(adapter, _FakeDeepEvalBaseLLM)
    assert adapter.get_model_name() == "deepseek-v4-flash"
    _, kwargs = mock_chat_openai.call_args
    assert kwargs["base_url"] == "https://api.deepseek.com/v1"
    assert kwargs["api_key"] == "sk-ds-test"


# ---------------------------------------------------------------------------
# judge_call_stats -- real per-call token/cost tracking for the
# DeepEval adapter path (bare-OpenAI-string judges have no hook and must
# report None, not a fake 0).
# ---------------------------------------------------------------------------


def _fake_response(prompt_tokens=None, completion_tokens=None, total_tokens=None, content="ok"):
    resp_meta = {}
    if prompt_tokens is not None or completion_tokens is not None or total_tokens is not None:
        resp_meta["token_usage"] = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }
    return types.SimpleNamespace(content=content, response_metadata=resp_meta)


def test_judge_call_stats_returns_none_for_bare_openai_judge_string():
    """The openai-native path returns a bare model-name string with no
    call_log attribute -- judge_call_stats must say "not tracked", not 0."""
    bare = make_deepeval_judge_model()
    assert isinstance(bare, str)
    assert judge_call_stats(bare) is None


@patch("langchain_anthropic.ChatAnthropic")
def test_judge_call_stats_sums_real_token_usage_across_calls(mock_chat_anthropic, monkeypatch, stub_deepeval):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    mock_chat_anthropic.return_value.invoke.side_effect = [
        _fake_response(prompt_tokens=100, completion_tokens=20, total_tokens=120),
        _fake_response(prompt_tokens=150, completion_tokens=30, total_tokens=180),
    ]

    adapter = make_deepeval_judge_model(provider="anthropic", model="claude-sonnet-4-6")

    # A trackable judge always has a call_log (even if empty) -- distinct
    # from the bare-string case, which has none at all and reports None.
    assert judge_call_stats(adapter)["n_calls"] == 0
    adapter.generate("first")
    adapter.generate("second")

    stats = judge_call_stats(adapter)
    assert stats["n_calls"] == 2
    assert stats["prompt_tokens"] == 250
    assert stats["completion_tokens"] == 50
    assert stats["total_tokens"] == 300
    assert stats["cost_usd"] >= 0.0


@patch("langchain_anthropic.ChatAnthropic")
def test_judge_call_stats_missing_token_usage_counts_as_zero_not_a_crash(
    mock_chat_anthropic, monkeypatch, stub_deepeval
):
    """A response with no response_metadata at all (as in the existing
    ...returns_adapter test's plain SimpleNamespace) must not raise -- token
    counts fall back to 0 rather than propagating a KeyError/AttributeError."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    mock_chat_anthropic.return_value.invoke.return_value = types.SimpleNamespace(content="hi")

    adapter = make_deepeval_judge_model(provider="anthropic", model="claude-sonnet-4-6")
    adapter.generate("prompt")

    stats = judge_call_stats(adapter)
    assert stats["n_calls"] == 1
    assert stats["prompt_tokens"] == 0
    assert stats["completion_tokens"] == 0
    assert stats["cost_usd"] == 0.0


@patch("langchain_anthropic.ChatAnthropic")
@pytest.mark.asyncio
async def test_judge_call_stats_tracks_async_generate_too(mock_chat_anthropic, monkeypatch, stub_deepeval):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")

    async def fake_ainvoke(prompt):
        return _fake_response(prompt_tokens=10, completion_tokens=5, total_tokens=15)

    mock_chat_anthropic.return_value.ainvoke = fake_ainvoke

    adapter = make_deepeval_judge_model(provider="anthropic", model="claude-sonnet-4-6")
    result = await adapter.a_generate("prompt")

    assert result == "ok"
    stats = judge_call_stats(adapter)
    assert stats["n_calls"] == 1
    assert stats["total_tokens"] == 15


# ---------------------------------------------------------------------------
# Reasoning-model argument remapping
#
# RAGAS 0.4 defaults every judge call to temperature=0.01 / top_p=0.1 /
# max_tokens=1024. It has its own remapping for OpenAI reasoning models, but
# detects them with int(model[4:].split("-")[0]), which raises on a dotted minor
# version -- so this project's default judge, gpt-5.4-mini, silently gets the
# raw defaults and 400s on every call. judge_factory therefore does the
# remapping itself.
# ---------------------------------------------------------------------------

import types as _types

from src.evaluation.judge_factory import (
    _sanitize_reasoning_model_args,
    is_reasoning_model,
)


@pytest.mark.parametrize(
    "model,expected",
    [
        ("gpt-5.4-mini", True),   # the case ragas' own parser misses
        ("gpt-5-mini", True),
        ("gpt-5", True),
        ("GPT-5.4-MINI", True),
        ("gpt-12-turbo", True),
        ("o1", True),
        ("o3-mini", True),
        ("codex-mini", True),
        ("gpt-4o-mini", False),
        ("gpt-4.1", False),
        ("deepseek-v4-flash", False),
        ("claude-sonnet-4-6", False),
        ("", False),
    ],
)
def test_is_reasoning_model(model, expected):
    assert is_reasoning_model(model) is expected


def test_sanitize_remaps_reasoning_model_args():
    llm = _types.SimpleNamespace(
        model_args={"temperature": 0.01, "top_p": 0.1, "max_tokens": 1024}
    )
    _sanitize_reasoning_model_args(llm, "gpt-5.4-mini")

    assert llm.model_args["temperature"] == 1.0
    assert "top_p" not in llm.model_args
    assert "max_tokens" not in llm.model_args
    # 1024 truncates structured output on reasoning models; ragas' docs say 4096+.
    assert llm.model_args["max_completion_tokens"] == 4096


def test_sanitize_preserves_a_larger_explicit_limit():
    llm = _types.SimpleNamespace(
        model_args={"temperature": 0.01, "top_p": 0.1, "max_tokens": 99999}
    )
    _sanitize_reasoning_model_args(llm, "gpt-5.4-mini")
    assert llm.model_args["max_completion_tokens"] == 99999


def test_sanitize_is_idempotent():
    llm = _types.SimpleNamespace(
        model_args={"temperature": 0.01, "top_p": 0.1, "max_tokens": 1024}
    )
    _sanitize_reasoning_model_args(llm, "gpt-5.4-mini")
    first = dict(llm.model_args)
    _sanitize_reasoning_model_args(llm, "gpt-5.4-mini")
    assert llm.model_args == first


def test_sanitize_leaves_non_reasoning_models_alone():
    args = {"temperature": 0.01, "top_p": 0.1, "max_tokens": 1024}
    llm = _types.SimpleNamespace(model_args=dict(args))
    _sanitize_reasoning_model_args(llm, "deepseek-v4-flash")
    assert llm.model_args == args


def test_sanitize_tolerates_a_ragas_build_without_model_args():
    llm = _types.SimpleNamespace()
    assert _sanitize_reasoning_model_args(llm, "gpt-5.4-mini") is llm
