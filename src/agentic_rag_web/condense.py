"""Multi-turn query condensation node .

``CondenseNode`` runs ahead of ``PlanNode`` in ``build_web_graph()``:

- **Empty history passthrough (Gate G2):** When ``conversation_history`` is
  empty or ``None``, ``condense_query`` is a pure no-op passthrough that
  returns ``{"query": raw_query, "raw_query": raw_query}`` without making any
  LLM call. This keeps single-turn runs byte-identical to the paper graph.
- **Multi-turn rewrite:** When prior turns are present, ``condense_query``
  uses the ``worker`` model tier (configurable via ``CPS_CONDENSE_ROLE``) to
  resolve pronouns/anaphora and expand ellipsis into a standalone query for
  downstream planning and retrieval while preserving the user's literal input
  in ``raw_query``.
- **Hedging carryover:** Prior turns' assistant
  *responses* are included alongside their queries, and both the condensation
  prompt and a post-check safeguard ensure that unverified/fictitious premises
  refused or hedged in an earlier turn are never laundered into confident
  factual premises on follow-up turns.
"""

from __future__ import annotations

import os
import re
import time
from typing import Dict, List

from langchain_core.messages import HumanMessage, SystemMessage

from src.agentic_rag.config import (
    WORKER_QUERY_REFORMULATION_MAX_TOKENS,
    WORKER_TEMPERATURE,
)
from src.agentic_rag.llm_factory import make_chat_model, resolve_role_config
from src.agentic_rag.llm_logger import record_llm_call, validate_model_parameters
from src.agentic_rag_web.state import WebAgentState

# default to the cheap/fast worker tier; allow override via env var.
CONDENSE_ROLE = os.getenv("CPS_CONDENSE_ROLE", "worker")

# cap history at the last 5 turns and bound per-turn response length so
# multi-hundred-word synthesis answers do not bloat the condensation prompt.
# Head-truncation is deliberate:  refusal/hedging disclaimers always appear
# at the start of an assistant response.
MAX_HISTORY_TURNS = int(os.getenv("CPS_MAX_HISTORY_TURNS", "5"))
MAX_RESPONSE_CHARS = int(os.getenv("CPS_CONDENSE_MAX_RESPONSE_CHARS", "2000"))

CONDENSE_SYSTEM_PROMPT = """You are a conversation context condenser for a Chemical Process Safety RAG system.
Your task is to rewrite a follow-up user question into a single, self-contained, standalone question using the prior conversation history.

RULES:
1. PRONOUN & ANAPHORA RESOLUTION: Replace pronouns and references ("it", "that incident", "that facility", "they", "those violations", "the explosion", "there") with the specific incident names, facility names, chemicals, or regulatory standards (e.g., 29 CFR 1910.119) from the prior turns.
2. ELLIPSIS EXPANSION: Expand brief follow-ups (e.g., "What about in Texas?", "And the root causes?", "How many fatalities?") into complete standalone questions that retain the relevant context from prior turns.
3. STANDALONE PRESERVATION: If the follow-up question is already self-contained and introduces a new topic, return it unchanged.
4. R11 HEDGING & UNCERTAINTY CARRYOVER (CRITICAL SAFETY REQUIREMENT):
   Inspect the prior assistant responses carefully. If a prior assistant response stated that an incident, facility, inspection/citation ID, or event was fictitious, unverified, unverifiable, not found in available OSHA/CSB records, or lacked supporting evidence:
   - You MUST carry forward that uncertainty/refusal framing in the rewritten standalone question.
   - NEVER rewrite the follow-up in a way that treats an unverified, unconfirmed, or fictitious incident/entity as an established fact.
   - Explicitly qualify the entity as unverified or unconfirmed in existing records AND enclose the specific entity/incident name in double quotes (for example: 'Regarding the unverified "2024 Apex Cryogenics explosion" (for which no OSHA or CSB records were found in the prior turn), what caused it and what penalties were issued?').
5. OUTPUT FORMAT: Return ONLY the rewritten standalone question text. Do not include surrounding quotes around the entire sentence, explanations, or preamble."""

# Patterns indicating a prior assistant response flagged an unverified /
# uncorroborated / fictitious premise.
_PRIOR_REFUSAL_RE = re.compile(
    r"\b(?:"
    r"could not find records?(?:\s+matching)?"
    r"|no records?(?:\s+were)?\s+found"
    r"|no records?\s+of\b"
    r"|does not appear in the (?:retrieved|available) (?:evidence|records|sources)"
    r"|does not support the existence"
    r"|no evidence (?:was found|in the retrieved|of any)"
    r"|unverified|unverifiable|unconfirmed|fictitious|fabricated|unsubstantiated"
    r"|cannot verify the existence"
    r"|premise (?:itself )?appears unfounded"
    r")\b",
    re.IGNORECASE,
)

# Markers showing that the condensed query already carries forward
# uncertainty/refusal framing.
_CARRYOVER_HEDGING_RE = re.compile(
    r"\b(?:"
    r"unverified|unverifiable|unconfirmed|alleged|purported|claimed|fictitious|hypothetical"
    r"|no (?:osha|csb|retrieved)?\s*records?"
    r"|could not (?:be )?(?:found|verified)"
    r"|not found in (?:available|osha|csb)"
    r"|lacking evidence"
    r")\b",
    re.IGNORECASE,
)


def _normalize_history(raw_history: object) -> List[Dict[str, str]]:
    """Filter and bound ``conversation_history`` to valid ``{query, response}`` dicts."""
    if not isinstance(raw_history, list) or not raw_history:
        return []
    valid: List[Dict[str, str]] = []
    for item in raw_history:
        if not isinstance(item, dict):
            continue
        q = str(item.get("query") or "").strip()
        r = str(item.get("response") or "").strip()
        if q:
            valid.append({"query": q, "response": r})
    if not valid:
        return []
    return valid[-MAX_HISTORY_TURNS:]


def _format_history_block(turns: List[Dict[str, str]]) -> str:
    """Format prior turns (both query and truncated response) for the condensation prompt."""
    blocks: List[str] = []
    for idx, turn in enumerate(turns, start=1):
        resp = turn["response"]
        if len(resp) > MAX_RESPONSE_CHARS:
            resp = resp[:MAX_RESPONSE_CHARS] + "..."
        blocks.append(
            f"Turn {idx} User Query: {turn['query']}\n"
            f"Turn {idx} Assistant Response: {resp or '(no response recorded)'}"
        )
    return "\n\n".join(blocks)


def _ensure_r11_hedging_carryover(
    rewritten_query: str, turns: List[Dict[str, str]]
) -> str:
    """Deterministic safety net for hedging carryover.

    If any prior turn's assistant response flagged an unverified/fictitious
    premise and the LLM's rewrite omitted uncertainty framing, explicitly
    qualify the rewritten query so downstream planning and synthesis cannot
    mistake a refused premise for an established fact.
    """
    prior_flagged = any(_PRIOR_REFUSAL_RE.search(t.get("response", "")) for t in turns)
    if not prior_flagged:
        return rewritten_query
    if _CARRYOVER_HEDGING_RE.search(rewritten_query):
        return rewritten_query
    return (
        f"Regarding the unverified incident/entity (for which no OSHA or CSB "
        f"records were found in the prior turn): {rewritten_query}"
    )


def condense_query(state: WebAgentState) -> dict:
    """CondenseNode: rewrite follow-up query using conversation history, or no-op if empty."""
    raw_query = state.get("raw_query") or state.get("query", "")
    turns = _normalize_history(state.get("conversation_history"))

    # Gate G2 Passthrough Invariant: empty/None history MUST be a pure no-op
    # passthrough with zero LLM calls and byte-identical query.
    if not turns:
        return {"query": raw_query, "raw_query": raw_query}

    history_block = _format_history_block(turns)
    user_prompt = (
        f"CONVERSATION HISTORY:\n{history_block}\n\n"
        f"FOLLOW-UP USER QUESTION: {raw_query}\n\n"
        "Rewrite the follow-up user question into a standalone question following all system rules:"
    )

    cfg = resolve_role_config(CONDENSE_ROLE, default_temperature=WORKER_TEMPERATURE)
    validate_model_parameters(
        model=cfg.model,
        temperature=cfg.temperature,
        max_completion_tokens=WORKER_QUERY_REFORMULATION_MAX_TOKENS,
    )
    llm = make_chat_model(
        CONDENSE_ROLE,
        temperature=WORKER_TEMPERATURE,
        max_completion_tokens=WORKER_QUERY_REFORMULATION_MAX_TOKENS,
    )

    t0 = time.time()
    response = llm.invoke(
        [
            SystemMessage(content=CONDENSE_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ]
    )
    latency = time.time() - t0

    llm_calls = record_llm_call(
        response=response,
        node_name="CondenseNode",
        call_purpose="condense",
        model_config=cfg.model,
        temperature=cfg.temperature if cfg.temperature is not None else 0.0,
        system_prompt=CONDENSE_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        latency_s=latency,
        existing_calls=[],
    )

    condensed = (getattr(response, "content", "") or "").strip()
    # Strip accidental wrapping quotes if the model added them.
    if (
        len(condensed) >= 2
        and condensed[0] == condensed[-1]
        and condensed[0] in {'"', "'"}
    ):
        condensed = condensed[1:-1].strip()

    if not condensed:
        condensed = raw_query
    else:
        condensed = _ensure_r11_hedging_carryover(condensed, turns)

    return {
        "query": condensed,
        "raw_query": raw_query,
        "llm_calls": llm_calls,
    }
