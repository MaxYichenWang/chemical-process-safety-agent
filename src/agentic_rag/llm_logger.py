import uuid
from typing import Any, Dict, List, Optional

from src.data_processing.config import MODEL_PRICING

_MODEL_PRICING = MODEL_PRICING


def validate_model_parameters(
    *,
    model: str,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    max_tokens: Optional[int] = None,
    max_completion_tokens: Optional[int] = None,
) -> None:
    """Validate model-specific sampling/token parameters before API calls."""
    model_norm = (model or "").strip().lower()

    if max_tokens is not None and max_completion_tokens is not None:
        raise ValueError(
            "Invalid LLM params: provide only one of max_tokens or max_completion_tokens."
        )
    if max_tokens is not None and max_tokens <= 0:
        raise ValueError("Invalid LLM params: max_tokens must be > 0.")
    if max_completion_tokens is not None and max_completion_tokens <= 0:
        raise ValueError("Invalid LLM params: max_completion_tokens must be > 0.")
    if temperature is not None and not (0.0 <= temperature <= 2.0):
        raise ValueError("Invalid LLM params: temperature must be in [0.0, 2.0].")
    if top_p is not None and not (0.0 < top_p <= 1.0):
        raise ValueError("Invalid LLM params: top_p must be in (0.0, 1.0].")

    # GPT-5 family API compatibility guardrails.
    if model_norm.startswith("gpt-5"):
        if max_tokens is not None:
            raise ValueError(
                f"Invalid LLM params for model '{model}': max_tokens is unsupported; "
                "use max_completion_tokens."
            )
        if temperature is not None and temperature != 1.0:
            raise ValueError(
                f"Invalid LLM params for model '{model}': temperature={temperature} is unsupported; "
                "only 1.0 is allowed."
            )
        if top_p is not None and top_p != 1.0:
            raise ValueError(
                f"Invalid LLM params for model '{model}': top_p={top_p} is unsupported; "
                "only 1.0 is allowed."
            )

    # OpenAI Reasoning models (o1, o3 family)
    if model_norm.startswith(("o1", "o3")):
        if max_tokens is not None:
            raise ValueError(
                f"Invalid LLM params for reasoning model '{model}': max_tokens is unsupported; "
                "use max_completion_tokens."
            )
        if temperature is not None and temperature != 1.0:
            raise ValueError(
                f"Invalid LLM params for reasoning model '{model}': temperature={temperature} is unsupported; "
                "only 1.0 (or default) is allowed."
            )
        if top_p is not None and top_p != 1.0:
            raise ValueError(
                f"Invalid LLM params for reasoning model '{model}': top_p={top_p} is unsupported; "
                "only 1.0 is allowed."
            )

    # Anthropic Claude family guardrails
    if model_norm.startswith("claude"):
        if temperature is not None and not (0.0 <= temperature <= 1.0):
            raise ValueError(
                f"Invalid LLM params for Anthropic model '{model}': temperature must be in [0.0, 1.0], got {temperature}."
            )


def resolve_model_parameters(
    *,
    model: str,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    max_tokens: Optional[int] = None,
    max_completion_tokens: Optional[int] = None,
) -> Dict[str, Any]:
    """Resolve and validate parameter dict for a given model, applying provider defaults."""
    validate_model_parameters(
        model=model,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        max_completion_tokens=max_completion_tokens,
    )
    resolved: Dict[str, Any] = {"model": model}
    if temperature is not None:
        resolved["temperature"] = temperature
    if top_p is not None:
        resolved["top_p"] = top_p
    if max_tokens is not None:
        resolved["max_tokens"] = max_tokens
    if max_completion_tokens is not None:
        resolved["max_completion_tokens"] = max_completion_tokens
    return resolved


def _estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Estimate LLM call cost in USD based on token usage and model pricing.

    Lookup is case-insensitive: pricing dict keys are all lowercase, but
    provider-returned model ids don't always match our casing convention
    (e.g. MiniMax's API may echo back "MiniMax-M2.1" rather than
    "minimax-m2.1"). Normalizing here decouples cost attribution from a
    provider's exact casing, without requiring the pricing dict to carry a
    duplicate entry per casing variant.
    """
    model_key = (model or "").strip().lower()
    pricing = _MODEL_PRICING.get(model_key)
    if pricing is None:
        # Handle versioned model names (e.g., "gpt-5-mini-2025-08-07")
        # Sort by prefix length descending so "gpt-5-mini" matches before "gpt-5"
        for model_prefix, model_pricing in sorted(
            _MODEL_PRICING.items(), key=lambda x: len(x[0]), reverse=True
        ):
            if model_key.startswith(model_prefix):
                pricing = model_pricing
                break
    if pricing is None:
        pricing = _MODEL_PRICING["gpt-5-mini"]

    return (prompt_tokens * pricing["prompt"]) + (
        completion_tokens * pricing["completion"]
    )


def record_llm_call(
    *,
    response: Any,
    node_name: str,
    call_purpose: str,
    model_config: str,
    temperature: float,
    system_prompt: str,
    user_prompt: str,
    latency_s: float,
    existing_calls: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Capture metadata from a LangChain ``ChatOpenAI`` response.

    Parameters
    ----------
    response
        The return value of ``ChatOpenAI.invoke()``.
    node_name
        Graph node that made this call (e.g. ``"Supervisor"``).
    call_purpose
        Short label describing why the call was made
        (``"planning"``, ``"synthesis"``, ``"query_reformulation"``, etc.).
    model_config
        The model name string from the configuration (e.g. ``"gpt-5-mini"``).
    temperature
        Temperature used for this call.
    system_prompt
        The system prompt sent to the model.
    user_prompt
        The user/human prompt sent to the model.
    latency_s
        Wall-clock seconds the ``invoke()`` call took.
    existing_calls
        The current ``llm_calls`` list from state.  If ``None``, a new list is
        created.

    Returns
    -------
    list[dict]
        Updated list with the new call record appended.
    """
    calls = list(existing_calls) if existing_calls else []

    # --- Extract metadata from LangChain response object -----------------
    resp_meta = getattr(response, "response_metadata", {}) or {}
    token_usage = resp_meta.get("token_usage", {}) or {}
    model_used = resp_meta.get("model_name", model_config)
    prompt_tokens = token_usage.get("prompt_tokens") or 0
    completion_tokens = token_usage.get("completion_tokens") or 0
    estimated_cost_usd = _estimate_cost(model_used, prompt_tokens, completion_tokens)

    record: Dict[str, Any] = {
        "call_id": str(uuid.uuid4()),
        "node_name": node_name,
        "call_purpose": call_purpose,
        # Model info
        "model_requested": model_config,
        "model_used": model_used,
        "temperature": temperature,
        # Token usage
        "prompt_tokens": token_usage.get("prompt_tokens"),
        "completion_tokens": token_usage.get("completion_tokens"),
        "total_tokens": token_usage.get("total_tokens"),
        "estimated_cost_usd": round(estimated_cost_usd, 8),
        # Response info
        "finish_reason": resp_meta.get("finish_reason"),
        "system_fingerprint": resp_meta.get("system_fingerprint"),
        "response_id": resp_meta.get("id"),
        # Timing
        "latency_s": round(latency_s, 4),
        # Prompt & response previews (truncated to limit log size)
        "system_prompt_preview": _truncate(system_prompt, 500),
        "user_prompt_preview": _truncate(user_prompt, 500),
        "response_preview": _truncate(getattr(response, "content", ""), 500),
        "response_length": len(getattr(response, "content", "") or ""),
    }

    calls.append(record)
    return calls


def _truncate(text: str, max_chars: int = 500) -> str:
    """Return *text* truncated to *max_chars* with an ellipsis if clipped."""
    if not text:
        return ""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "…"
