"""Provider-agnostic chat-model factory for the agentic RAG pipeline.

Today every node instantiates ``langchain_openai.ChatOpenAI`` directly. This
factory adds a thin indirection so the model *and provider* for each role can be
swapped via environment variables — enabling the model sweep (see
``configs/model_sweep.example.yaml`` and ``src/evaluation/model_sweep.py``) to
compare OpenAI, Anthropic, and any OpenAI-compatible endpoint (Together, Groq,
Fireworks, Azure, local Ollama, ...) per role.

Backward compatible: with no new env vars set, ``make_chat_model("supervisor")``
returns the same ``ChatOpenAI(model=AGENT_SUPERVISOR_MODEL, ...)`` the code builds
today. Provider client libraries are imported lazily, so a missing
``langchain_anthropic`` only matters if a role actually selects ``anthropic``.

Per-role environment variables (ROLE in: SUPERVISOR, WORKER, REASONING, CRITIC,
DATA_ANALYST; also EVAL_LLM for the evaluation judge -- see below):

    AGENT_<ROLE>_PROVIDER     openai | anthropic | openai_compatible   (default: openai)
    AGENT_<ROLE>_MODEL        model name/id                            (existing var, unchanged)
    AGENT_<ROLE>_TEMPERATURE  float                                    (optional; falls back to per-role default)
    AGENT_<ROLE>_BASE_URL     base URL for openai_compatible / proxied OpenAI
    AGENT_<ROLE>_API_KEY_ENV  NAME of the env var holding the API key  (default per provider)

Every one of those can *also* be passed as an explicit keyword argument to
``resolve_role_config`` / ``make_chat_model`` (``provider=``, ``model=``,
``base_url=``, ``api_key_env=``; ``temperature=`` already worked this way).
Precedence is: **explicit argument > env var > built-in default**. Passing
``None`` (the default) for any of these falls through to the env-var behavior
described above -- so existing call sites that don't pass them are unaffected.

The evaluation judge model (``EVAL_LLM_MODEL`` / ``EVAL_LLM_TEMPERATURE`` in
``src/evaluation/config.py``) is resolved through the same machinery under the
role name ``"eval_judge"`` (env prefix ``EVAL_LLM``), so it can be pointed at any
supported provider the same way the five pipeline roles can -- see
``src/evaluation/judge_factory.py``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Optional

# Role -> env-var prefix. We read the *current* env value at call time so
# dotenv / sweep env mutations are respected. Adding a role here (and, if it
# needs a non-default fallback, to _ROLE_DEFAULT_MODEL) is the only step
# needed to extend this resolution machinery to a new use case.
_ROLE_ENV_PREFIX = {
    "supervisor": "AGENT_SUPERVISOR",
    "worker": "AGENT_WORKER",
    "reasoning": "AGENT_REASONING",
    "critic": "AGENT_CRITIC",
    "data_analyst": "AGENT_DATA_ANALYST",
    "eval_judge": "EVAL_LLM",
}

_DEFAULT_MODEL = "gpt-5-mini"

# Per-role default model, for roles whose historical default differs from
# _DEFAULT_MODEL. eval_judge mirrors src/evaluation/config.py's EVAL_LLM_MODEL
# default so this refactor is byte-for-byte backward compatible.
_ROLE_DEFAULT_MODEL = {
    "eval_judge": "gpt-5.4-mini",
}

_DEFAULT_API_KEY_ENV = {
    "openai": "OPENAI_API_KEY",
    "openai_compatible": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "qwen": "ALICLOUD_API_KEY",
    "dashscope": "DASHSCOPE_API_KEY",
    "alicloud": "ALICLOUD_API_KEY",
}


def _default_temperature_for_model(model: str) -> float:
    """Provider-aware temperature default.

    Historically every role defaulted to ``1.0`` everywhere (a GPT-5
    workaround: gpt-5-mini and siblings reject ``temperature=0.0`` with a 400).
    DeepSeek and most other providers accept -- and, for a reproducible eval
    pipeline, want -- ``0.0``. Default to ``0.0``; only GPT-5 and OpenAI reasoning
    families (o1, o3, which require 1.0 or reject 0.0) get 1.0.
    """
    m = (model or "").strip().lower()
    return 1.0 if m.startswith(("gpt-5", "o1", "o3")) else 0.0


@dataclass(frozen=True)
class RoleModelConfig:
    role: str
    provider: str
    model: str
    temperature: Optional[float]
    base_url: Optional[str]
    api_key_env: str


def resolve_role_config(
    role: str,
    env: Optional[dict] = None,
    default_temperature: Optional[float] = None,
    *,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key_env: Optional[str] = None,
) -> RoleModelConfig:
    """Resolve the provider/model config for *role*.

    Resolution precedence for ``provider``/``model``/``base_url``/``api_key_env``
    is: explicit keyword argument (if not ``None``) > env var > built-in default.
    This matches the precedence ``temperature``/``default_temperature`` already
    had -- callers that want to pin a role to a specific provider/model without
    touching ``os.environ`` (e.g. the model sweep, or a test) can now do so directly.

    Pure function (no I/O beyond reading the provided env mapping) so it is unit
    testable without any provider libraries installed.
    """
    role = role.lower()
    if role not in _ROLE_ENV_PREFIX:
        raise ValueError(
            f"unknown role '{role}'; expected one of {sorted(_ROLE_ENV_PREFIX)}"
        )
    env = os.environ if env is None else env
    prefix = _ROLE_ENV_PREFIX[role]

    resolved_model = model or env.get(f"{prefix}_MODEL")
    if not resolved_model and role == "data_analyst":
        resolved_model = env.get(
            "AGENT_WORKER_MODEL"
        )  # mirrors current config.py fallback
    resolved_model = resolved_model or _ROLE_DEFAULT_MODEL.get(role, _DEFAULT_MODEL)

    resolved_provider = (
        (provider or env.get(f"{prefix}_PROVIDER") or "openai").strip().lower()
    )

    temp_raw = env.get(f"{prefix}_TEMPERATURE")
    if temp_raw not in (None, ""):
        temperature = float(temp_raw)
    elif default_temperature is not None:
        temperature = default_temperature
    else:
        # Neither an explicit override nor an env var set this role's
        # temperature -- fall back to a provider-aware default rather than
        # leaving it unset (which used to mean "let the SDK use its own
        # default", silently non-deterministic across providers).
        temperature = _default_temperature_for_model(resolved_model)

    env_provider = (env.get(f"{prefix}_PROVIDER") or "openai").strip().lower()
    provider_overridden = (
        provider is not None and provider.strip().lower() != env_provider
    )

    if base_url:
        resolved_base_url = base_url
    elif provider_overridden:
        resolved_base_url = None
    else:
        resolved_base_url = env.get(f"{prefix}_BASE_URL") or None

    if api_key_env:
        resolved_api_key_env = api_key_env
    elif provider_overridden:
        resolved_api_key_env = _DEFAULT_API_KEY_ENV.get(
            resolved_provider, "OPENAI_API_KEY"
        )
    else:
        resolved_api_key_env = env.get(
            f"{prefix}_API_KEY_ENV"
        ) or _DEFAULT_API_KEY_ENV.get(resolved_provider, "OPENAI_API_KEY")

    return RoleModelConfig(
        role=role,
        provider=resolved_provider,
        model=resolved_model,
        temperature=temperature,
        base_url=resolved_base_url,
        api_key_env=resolved_api_key_env,
    )


def make_chat_model(
    role: str,
    *,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key_env: Optional[str] = None,
    temperature: Optional[float] = None,
    max_completion_tokens: Optional[int] = None,
    response_format: Optional[dict] = None,
    **extra: Any,
):
    """Build a LangChain chat model for *role*, honoring provider env config.

    ``provider``/``model``/``base_url``/``api_key_env`` are optional explicit
    overrides -- when supplied they win over the corresponding env var for this
    call, without mutating ``os.environ`` (see module docstring for precedence).
    Omit them (the default) to keep today's fully env-driven behavior.

    The other parameters mirror the kwargs the nodes pass to ``ChatOpenAI`` today.
    The resolved per-role temperature wins unless an explicit *temperature*
    override is supplied by the caller (matching current call-site behaviour
    where each node passes its own ``*_TEMPERATURE``).
    """
    cfg = resolve_role_config(
        role,
        default_temperature=temperature,
        provider=provider,
        model=model,
        base_url=base_url,
        api_key_env=api_key_env,
    )
    temp = temperature if temperature is not None else cfg.temperature
    api_key = os.getenv(cfg.api_key_env)
    if not api_key and cfg.api_key_env in ("DASHSCOPE_API_KEY", "ALICLOUD_API_KEY"):
        api_key = os.getenv("ALICLOUD_API_KEY") or os.getenv("DASHSCOPE_API_KEY")

    if cfg.provider in ("openai", "openai_compatible"):
        from langchain_openai import ChatOpenAI

        kwargs: dict[str, Any] = {"model": cfg.model}
        if temp is not None:
            kwargs["temperature"] = temp
        if max_completion_tokens is not None:
            kwargs["max_completion_tokens"] = max_completion_tokens
        if cfg.base_url:
            kwargs["base_url"] = cfg.base_url
        if api_key:
            kwargs["api_key"] = api_key
        if response_format is not None:
            kwargs["model_kwargs"] = {"response_format": response_format}
        timeout_env = os.getenv("LLM_REQUEST_TIMEOUT")
        kwargs.setdefault(
            "request_timeout", float(timeout_env) if timeout_env else 90.0
        )
        kwargs.setdefault("max_retries", 3)
        kwargs.update(extra)
        return ChatOpenAI(**kwargs)

    if cfg.provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        kwargs = {"model": cfg.model}
        if temp is not None:
            kwargs["temperature"] = temp
        # Anthropic uses max_tokens (not max_completion_tokens).
        if max_completion_tokens is not None:
            kwargs["max_tokens"] = max_completion_tokens
        if api_key:
            kwargs["api_key"] = api_key
        timeout_env = os.getenv("LLM_REQUEST_TIMEOUT")
        kwargs.setdefault(
            "default_request_timeout", float(timeout_env) if timeout_env else 90.0
        )
        kwargs.setdefault("max_retries", 3)
        kwargs.update(extra)
        # Note: Anthropic has no OpenAI-style response_format; JSON adherence must
        # be enforced via the prompt for anthropic-backed roles.
        return ChatAnthropic(**kwargs)

    raise ValueError(
        f"unsupported provider '{cfg.provider}' for role '{role}'. "
        "Expected: openai | openai_compatible | anthropic."
    )


def resolve_model_name(
    role: str,
    *,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key_env: Optional[str] = None,
) -> str:
    """Return the resolved model id for *role* (for logging / cost attribution)."""
    return resolve_role_config(
        role, provider=provider, model=model, base_url=base_url, api_key_env=api_key_env
    ).model
