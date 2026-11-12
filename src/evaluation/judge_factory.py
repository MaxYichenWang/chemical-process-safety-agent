"""Provider-agnostic judge-model factory for the evaluation framework.

RAGAS and DeepEval both need an LLM to act as judge. Before this module, every
call site hardcoded ``ChatOpenAI(model=EVAL_LLM_MODEL)`` (RAGAS's ``llm=``
kwarg) or the bare model-name string ``EVAL_LLM_MODEL`` (DeepEval metrics'
``model=`` kwarg) -- see ``src/evaluation/evaluators/faithfulness_evaluator.py``,
``retrieval_evaluator.py``, and ``orchestration_evaluator.py``.

This module resolves the judge through the same role-based factory the
agentic RAG pipeline uses (``src/agentic_rag/llm_factory.py``), under the role
name ``"eval_judge"`` -- so the judge can be pointed at any supported provider
(OpenAI, Anthropic, or any OpenAI-compatible endpoint: DeepSeek, Kimi, MiniMax,
Gemini, Qwen, Together, Groq, ...) the same way the five pipeline roles can,
via ``EVAL_LLM_PROVIDER`` / ``EVAL_LLM_MODEL`` / ``EVAL_LLM_BASE_URL`` /
``EVAL_LLM_API_KEY_ENV`` env vars, or via explicit keyword arguments (which win
over env vars -- see ``llm_factory.resolve_role_config``'s docstring for the
precedence rules).

RAGAS 0.4 deprecates passing LangChain models directly and prefers its native
``ragas.llms.llm_factory()`` with a provider SDK client. ``make_ragas_judge_llm``
therefore returns a native RAGAS LLM, while ``make_deepeval_judge_model`` keeps
DeepEval's expected string/custom-adapter behavior.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Dict, List, Optional

from src.agentic_rag.llm_factory import make_chat_model, resolve_role_config

logger = logging.getLogger(__name__)

_JUDGE_ROLE = "eval_judge"

# OpenAI "reasoning" models (o-series, GPT-5 and newer) reject temperature != 1,
# reject top_p entirely, and require max_completion_tokens instead of max_tokens.
#
# RAGAS 0.4 knows this and remaps the arguments itself -- but its detector does
# ``int(model[4:].split("-")[0])``, which raises on a dotted minor version and
# silently falls through to "not a reasoning model". So this project's default
# judge, ``gpt-5.4-mini``, gets RAGAS's raw defaults (temperature=0.01,
# top_p=0.1, max_tokens=1024) and every judge call 400s. This regex accepts
# dotted/underscored minor versions so the remapping below actually fires.
_REASONING_MODEL_RE = re.compile(
    r"^(?:o[1-9](?:[-_].*)?|gpt-(?:[5-9]|1[0-9])(?:[.\-_].*)?|codex-mini)$"
)

# RAGAS's 1024 default truncates structured output on reasoning models, which
# surfaces as unparseable JSON rather than an obvious error. RAGAS's own docs
# recommend 4096+.
_REASONING_MODEL_MIN_MAX_TOKENS = 4096


def is_reasoning_model(model: str) -> bool:
    """True for OpenAI models with the temperature/top_p/max_tokens constraints."""
    return bool(_REASONING_MODEL_RE.match((model or "").lower()))


def _sanitize_reasoning_model_args(llm: Any, model: str) -> Any:
    """Rewrite a RAGAS ``InstructorLLM``'s model args for reasoning models.

    Applied after construction because the offending values come from RAGAS's
    own ``InstructorModelArgs`` defaults, which cannot be removed by passing
    kwargs. Idempotent, and a no-op both for non-reasoning models and for any
    RAGAS build that does not expose ``model_args``.
    """
    if not is_reasoning_model(model):
        return llm

    model_args = getattr(llm, "model_args", None)
    if not isinstance(model_args, dict):
        logger.debug(
            "RAGAS judge %s exposes no model_args dict; skipping reasoning-model "
            "argument remapping",
            model,
        )
        return llm

    model_args["temperature"] = 1.0
    model_args.pop("top_p", None)
    if "max_tokens" in model_args:
        model_args["max_completion_tokens"] = model_args.pop("max_tokens")
    model_args["max_completion_tokens"] = max(
        int(model_args.get("max_completion_tokens") or 0),
        _REASONING_MODEL_MIN_MAX_TOKENS,
    )
    return llm


def make_ragas_embeddings(model: Optional[str] = None) -> Any:
    """Return native RAGAS v0.4 embeddings for experiment metrics."""
    from openai import AsyncOpenAI
    from ragas.embeddings import embedding_factory

    from src.evaluation.config import EVAL_DEDUP_EMBEDDING_MODEL

    embedding_model = model or os.getenv(
        "EVAL_EMBEDDING_MODEL", EVAL_DEDUP_EMBEDDING_MODEL
    )
    api_key_env = os.getenv("EVAL_EMBEDDING_API_KEY_ENV", "OPENAI_API_KEY")
    api_key = os.getenv(api_key_env)
    base_url = os.getenv("EVAL_EMBEDDING_BASE_URL")

    client_kwargs: dict[str, Any] = {}
    if api_key:
        client_kwargs["api_key"] = api_key
    if base_url:
        client_kwargs["base_url"] = base_url

    return embedding_factory(
        provider="openai",
        client=AsyncOpenAI(**client_kwargs),
        model=embedding_model,
    )


def make_ragas_judge_llm(
    *,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key_env: Optional[str] = None,
    temperature: Optional[float] = None,
    **extra: Any,
):
    """Return a native RAGAS 0.4 judge for RAGAS's ``llm=`` kwarg.

    RAGAS 0.4's ``llm_factory`` requires a native provider client. OpenAI and
    OpenAI-compatible endpoints share the OpenAI SDK client; for compatible
    providers (DeepSeek, Gemini-compatible gateways, Qwen-compatible gateways,
    etc.) we pass the configured ``base_url`` but keep the RAGAS provider as
    ``"openai"`` so RAGAS uses its OpenAI/instructor adapter.
    """
    from ragas.llms import llm_factory

    cfg = resolve_role_config(
        _JUDGE_ROLE,
        provider=provider,
        model=model,
        base_url=base_url,
        api_key_env=api_key_env,
    )
    temp = temperature if temperature is not None else cfg.temperature
    model_kwargs = dict(extra)
    if temp is not None:
        model_kwargs["temperature"] = temp

    # Allow overriding the judge's max output length to avoid provider-side
    # truncation that can break structured JSON responses. Set via
    # EVAL_LLM_MAX_TOKENS (falls back to 8192 if unset).
    max_tokens_raw = os.getenv("EVAL_LLM_MAX_TOKENS", "8192")
    if max_tokens_raw:
        try:
            model_kwargs["max_tokens"] = int(max_tokens_raw)
        except ValueError:
            raise ValueError("EVAL_LLM_MAX_TOKENS must be an integer")

    api_key = os.getenv(cfg.api_key_env)
    if cfg.provider in ("openai", "openai_compatible"):
        from openai import AsyncOpenAI

        client_kwargs: dict[str, Any] = {}
        if api_key:
            client_kwargs["api_key"] = api_key
        if cfg.base_url:
            client_kwargs["base_url"] = cfg.base_url
        llm = llm_factory(
            cfg.model,
            provider="openai",
            client=AsyncOpenAI(**client_kwargs),
            **model_kwargs,
        )
        # Only meaningful for genuine OpenAI reasoning models; an
        # openai_compatible endpoint whose model id happens to match would also
        # be sent temperature=1.0 / max_completion_tokens, both of which are
        # standard OpenAI-compatible fields.
        return _sanitize_reasoning_model_args(llm, cfg.model)

    if cfg.provider == "anthropic":
        try:
            from anthropic import AsyncAnthropic
        except ImportError as exc:
            raise ImportError(
                "anthropic is required for RAGAS eval_judge provider='anthropic'. "
                "Install it with `pip install anthropic`."
            ) from exc

        client_kwargs = {}
        if api_key:
            client_kwargs["api_key"] = api_key
        if cfg.base_url:
            client_kwargs["base_url"] = cfg.base_url
        return llm_factory(
            cfg.model,
            provider="anthropic",
            client=AsyncAnthropic(**client_kwargs),
            **model_kwargs,
        )

    raise ValueError(
        f"unsupported provider '{cfg.provider}' for RAGAS judge. "
        "Expected: openai | openai_compatible | anthropic."
    )


def make_deepeval_judge_model(
    *,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key_env: Optional[str] = None,
) -> Any:
    """Return the ``model=`` value for a DeepEval metric (Faithfulness,
    Hallucination, GEval, TaskCompletion, ContextualPrecision/Recall, ...).

    DeepEval's metrics accept either a bare OpenAI model-name string (which
    DeepEval resolves internally against its own OpenAI model registry) or a
    ``DeepEvalBaseLLM`` instance for any other provider -- passing an
    arbitrary non-OpenAI string (e.g. ``"claude-sonnet-4-6"``) would fail
    DeepEval's internal resolution.

    To stay byte-for-byte compatible with today's behavior, the resolved-openai
    path still returns the plain model string. Any other resolved provider
    (anthropic, or an openai_compatible endpoint such as DeepSeek/Gemini/Qwen/
    MiniMax/Kimi) returns an adapter wrapping a LangChain chat model configured
    with the same provider/model/base URL as the RAGAS judge.
    """
    cfg = resolve_role_config(
        _JUDGE_ROLE,
        provider=provider,
        model=model,
        base_url=base_url,
        api_key_env=api_key_env,
    )
    if cfg.provider == "openai":
        return cfg.model

    max_tokens_raw = os.getenv("EVAL_LLM_MAX_TOKENS", "8192")
    max_tokens = int(max_tokens_raw) if max_tokens_raw else 8192

    chat_model = make_chat_model(
        _JUDGE_ROLE,
        provider=provider,
        model=model,
        base_url=base_url,
        api_key_env=api_key_env,
        max_completion_tokens=max_tokens,
    )
    return _wrap_for_deepeval(chat_model, cfg.model)


def _wrap_for_deepeval(chat_model: Any, model_name: str) -> Any:
    """Wrap a LangChain chat model in DeepEval's ``DeepEvalBaseLLM`` interface.

    ``deepeval`` is imported lazily (inside this function, not at module load
    time) so importing ``judge_factory`` or calling ``make_deepeval_judge_model``
    on the default openai path never requires deepeval to be installed. It only
    matters once a non-openai judge is actually requested.
    """
    from deepeval.models.base_model import DeepEvalBaseLLM

    from src.agentic_rag.llm_logger import _estimate_cost

    class _LangChainDeepEvalModel(DeepEvalBaseLLM):
        def __init__(self) -> None:
            self._chat_model = chat_model
            self._model_name = model_name
            # Every generate()/a_generate() call appends one record here
            # so callers (scoring/deepeval_layer.py, via judge_call_stats())
            # can report real judge_calls/judge_cost_usd instead of a
            # placeholder 0 -- see judge_call_stats' docstring for the one
            # case (bare OpenAI judge string) this still can't cover.
            self.call_log: List[Dict[str, Any]] = []

        def load_model(self) -> Any:
            return self._chat_model

        def _record_call(self, response: Any) -> None:
            resp_meta = getattr(response, "response_metadata", {}) or {}
            token_usage = resp_meta.get("token_usage", {}) or {}
            prompt_tokens = token_usage.get("prompt_tokens") or 0
            completion_tokens = token_usage.get("completion_tokens") or 0
            self.call_log.append(
                {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": token_usage.get("total_tokens")
                    or (prompt_tokens + completion_tokens),
                    "cost_usd": _estimate_cost(
                        self._model_name, prompt_tokens, completion_tokens
                    ),
                }
            )

        def generate(self, prompt: str) -> str:
            response = self._chat_model.invoke(prompt)
            self._record_call(response)
            return response.content

        async def a_generate(self, prompt: str) -> str:
            response = await self._chat_model.ainvoke(prompt)
            self._record_call(response)
            return response.content

        def get_model_name(self) -> str:
            return self._model_name

    return _LangChainDeepEvalModel()


def judge_call_stats(judge_model: Any) -> Optional[Dict[str, Any]]:
    """Real per-call token/cost totals for a DeepEval judge model.

    Returns ``None`` -- never a fake ``0`` -- when the judge is a bare OpenAI
    model-name string (the ``cfg.provider == "openai"`` path in
    ``make_deepeval_judge_model``): DeepEval resolves and calls that model
    through its own internal OpenAI client, which this project doesn't wrap,
    so there is genuinely nothing to sum. Every other provider goes through
    ``_wrap_for_deepeval``'s ``_LangChainDeepEvalModel``, whose ``call_log``
    this reads. No silent zeros: callers must render "not tracked
    for this judge provider", not "$0.00".
    """
    call_log = getattr(judge_model, "call_log", None)
    if call_log is None:
        return None
    return {
        "n_calls": len(call_log),
        "prompt_tokens": sum(c["prompt_tokens"] for c in call_log),
        "completion_tokens": sum(c["completion_tokens"] for c in call_log),
        "total_tokens": sum(c["total_tokens"] for c in call_log),
        "cost_usd": round(sum(c["cost_usd"] for c in call_log), 8),
    }
