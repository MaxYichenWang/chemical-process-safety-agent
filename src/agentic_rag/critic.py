import json
import logging
import re
import time

from langchain_core.messages import HumanMessage, SystemMessage

from src.agentic_rag.config import (
    CRITIC_ENABLED,
    CRITIC_EVALUATION_MAX_TOKENS,
    CRITIC_QUANTITATIVE_MAX_TOKENS,
    CRITIC_TEMPERATURE,
)
from src.agentic_rag.llm_factory import make_chat_model
from src.agentic_rag.llm_logger import record_llm_call, validate_model_parameters
from src.agentic_rag.state import AgentState
from src.data_processing.config import AGENT_CRITIC_MODEL

logger = logging.getLogger(__name__)

CRITIC_MODEL = AGENT_CRITIC_MODEL

_QUANTITATIVE_SIGNALS = frozenset(
    {
        "how many",
        "count",
        "total",
        "number of",
        "percentage",
        "average",
        "trend",
        "statistic",
        "ranking",
        "distribution",
        "compare",
        "most common",
        "top",
        "least",
        "median",
        "sum of",
    }
)


def _is_quantitative(query: str) -> bool:
    """Classify a query as quantitative based on keyword signals."""
    q = query.lower()
    return any(signal in q for signal in _QUANTITATIVE_SIGNALS)


def critic_agent(state: AgentState) -> dict:
    """
    Evaluates the draft_response against context_gathered for factual grounding,
    completeness, and domain-specific accuracy. Returns both a validation score
    and actionable critique text so the supervisor can correct on retry.

    Uses a lighter, focused prompt for quantitative queries (counts, trends,
    statistics) and the full validation prompt for qualitative queries.

    Ablation switch: when ``CRITIC_ENABLED`` is false, skips
    the LLM call entirely and always passes the draft straight through
    (score high enough that ``graph._route_from_critic`` never retries) --
    isolating the critic's contribution to hallucination mitigation without
    changing graph topology.
    """
    if not CRITIC_ENABLED:
        return {
            "validation_score": 1.0,
            "critic_feedback": "Critic disabled via CRITIC_ENABLED=0 (ablation).",
            "llm_calls": [],
        }

    draft = state.get("draft_response", "")
    context = state.get("context_gathered", {})
    query = state.get("query", "")
    existing_llm_calls = []
    worker_errors = state.get("worker_errors", [])

    context_str = json.dumps(context, indent=2)

    worker_errors_block = ""
    if worker_errors:
        error_lines = "\n".join(f"- {e['node']}: {e['impact']}" for e in worker_errors)
        worker_errors_block = (
            f"\n\nKNOWN WORKER FAILURES:\n{error_lines}\n"
            "The draft MUST acknowledge these data gaps. "
            "Score lower if it does not."
        )

    quantitative = _is_quantitative(query)

    if quantitative:
        system_prompt = _QUANTITATIVE_SYSTEM_PROMPT
        evaluation_prompt = _build_quantitative_prompt(
            query, context_str, draft, worker_errors_block
        )
        max_tokens = CRITIC_QUANTITATIVE_MAX_TOKENS
    else:
        system_prompt = _QUALITATIVE_SYSTEM_PROMPT
        evaluation_prompt = _build_qualitative_prompt(
            query, context_str, draft, worker_errors_block
        )
        max_tokens = CRITIC_EVALUATION_MAX_TOKENS

    validate_model_parameters(
        model=CRITIC_MODEL,
        temperature=CRITIC_TEMPERATURE,
        max_completion_tokens=max_tokens,
    )
    llm = make_chat_model(
        "critic",
        temperature=CRITIC_TEMPERATURE,
        max_completion_tokens=max_tokens,
        response_format={"type": "json_object"},
    )
    try:
        t0 = time.time()
        response = llm.invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=evaluation_prompt),
            ]
        )
        latency = time.time() - t0

        llm_calls = record_llm_call(
            response=response,
            node_name="CriticNode",
            call_purpose="validation",
            model_config=CRITIC_MODEL,
            temperature=CRITIC_TEMPERATURE,
            system_prompt=system_prompt,
            user_prompt=evaluation_prompt,
            latency_s=latency,
            existing_calls=existing_llm_calls,
        )

        raw_output = response.content
        try:
            result = json.loads(raw_output)
        except json.JSONDecodeError:
            json_match = re.search(r"\{.*\}", raw_output, re.DOTALL)
            json_str = json_match.group(0) if json_match else raw_output
            result = json.loads(json_str)

        score = float(result.get("score", 0.0))
        critique = result.get("critique", "No critique provided.")

    except Exception as e:
        logger.error("Critic failed to evaluate: %s", e)
        score = 0.0
        critique = f"Critic evaluation failed ({e}). Treating as unvalidated."
        llm_calls = existing_llm_calls

    return {
        "validation_score": score,
        "critic_feedback": critique,
        "llm_calls": llm_calls,
    }


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

_QUALITATIVE_SYSTEM_PROMPT = """You are a senior Chemical Process Safety reviewer and fact-checker. \
Your responsibility is to ensure that responses about industrial incidents, OSHA regulations, \
and safety recommendations are STRICTLY grounded in the retrieved evidence.

In this domain, hallucinated facts can lead to incorrect safety decisions and loss of life. \
You must be extremely rigorous.

VALIDATION CRITERIA:
1. FACTUAL GROUNDING: Every claim about incidents, dates, chemicals, equipment, injuries, \
fatalities, regulatory standards, and penalties MUST be directly traceable to the provided context.
2. COMPLETENESS: All parts of the user's original question must be addressed, or the response \
must explicitly acknowledge which parts lack sufficient evidence.
3. CITATION ACCURACY: Regulatory standard numbers (e.g., 1910.119), facility names, chemical \
names, and incident details must exactly match the source documents.
4. NO FABRICATION: The response must not introduce facts, statistics, incident details, or \
regulatory interpretations not present in the retrieved context.
5. APPROPRIATE HEDGING: Claims not fully supported by context should use hedging language \
or be flagged as unsupported."""

_QUANTITATIVE_SYSTEM_PROMPT = """You are a data-quality reviewer for Chemical Process Safety \
statistical responses. Your job is to verify that quantitative claims (counts, totals, \
percentages, rankings, trends) are correctly derived from the retrieved data and properly cited.

Focus on numerical accuracy — not prose style or narrative quality."""


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------


def _build_qualitative_prompt(
    query: str, context_str: str, draft: str, worker_errors_block: str
) -> str:
    return f"""Evaluate the following draft response for factual accuracy against the provided context.

ORIGINAL USER QUERY:
{query}

PROVIDED CONTEXT (retrieved from databases):
{context_str}
{worker_errors_block}

DRAFT RESPONSE TO EVALUATE:
{draft}

EVALUATION INSTRUCTIONS:
1. Score the factual grounding from 0.0 (fabricated/unsupported) to 1.0 (perfectly grounded).
2. Keep critique concise (2-4 sentences max). Focus strictly on concrete grounding errors or unaddressed query aspects.
3. If score < 0.8, your critique MUST contain concrete, actionable revision instructions in 1-2 concise sentences.

OUTPUT FORMAT (strict JSON):
{{
    "score": 0.85,
    "critique": "Brief, actionable critique text (2-4 sentences max)."
}}"""


def _build_quantitative_prompt(
    query: str, context_str: str, draft: str, worker_errors_block: str
) -> str:
    return f"""Evaluate the following statistical response for numerical accuracy.

USER QUERY:
{query}

RETRIEVED DATA:
{context_str}
{worker_errors_block}

DRAFT RESPONSE:
{draft}

CHECK THESE CRITERIA:
1. A specific number or count is present in the response.
2. The number is attributed to a cited source (SQL query result, database record, or document).
3. The response acknowledges any data limitations or gaps.
4. Numerical values match the retrieved data (no arithmetic errors or fabricated figures).

Score from 0.0 to 1.0 and provide a brief critique (under 2 sentences).

OUTPUT FORMAT (strict JSON):
{{
    "score": 0.90,
    "critique": "Brief critique focusing on numerical accuracy issues (1-2 sentences)."
}}"""
