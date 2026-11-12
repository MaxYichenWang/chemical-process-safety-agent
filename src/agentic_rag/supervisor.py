import contextvars
import json
import re
import time
from typing import Any, Dict, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.types import Send

from src.agentic_rag.config import (
    SUPERVISOR_PLANNING_MAX_TOKENS,
    SUPERVISOR_PREMISE_CHECK_ENABLED,
    SUPERVISOR_REVISION_MAX_TOKENS,
    SUPERVISOR_SYNTHESIS_MAX_TOKENS,
    SUPERVISOR_TEMPERATURE,
)
from src.agentic_rag.llm_factory import make_chat_model
from src.agentic_rag.llm_logger import record_llm_call, validate_model_parameters
from src.agentic_rag.state import AgentState
from src.data_processing.config import AGENT_SUPERVISOR_MODEL

_TOKEN_EMITTER: contextvars.ContextVar[Optional[Any]] = contextvars.ContextVar(
    "_TOKEN_EMITTER", default=None
)

SUPERVISOR_MODEL = AGENT_SUPERVISOR_MODEL

VALID_AGENTS = {
    "osha_regs",
    "osha_enf",
    "csb_incident",
    "csb_recs",
    "root_cause",
    "data_analyst",
}

AGENT_TO_NODE = {
    "osha_regs": "OshaRegsWorker",
    "osha_enf": "OshaEnfWorker",
    "csb_incident": "IncidentWorker",
    "csb_recs": "RecommendationsWorker",
    "root_cause": "RootCauseWorker",
    "data_analyst": "DataAnalystWorker",
}

_FAST_ROUTES = [
    {
        # Comparative/aggregation queries (e.g., "Which state has the most violations?")
        # must reach the SQL data analyst even when other patterns also match.
        # Without this route, queries containing words like "violations"/"penalties"
        # are hijacked by the enforcement/violation pattern, bypassing data_analyst.
        # Quantitative questions must reach the analytical worker.
        "description": "analytical/comparative query",
        "pattern": re.compile(
            # Core comparative/aggregation language
            r"\b(most|fewest|highest|lowest|top\s+\d+|bottom\s+\d+"
            r"|rank(?:ed|ing|s)?\b.*?\bby\b|compared?\s+to|distribution\s+of"
            r"|per\s+(?:state|year|industry|naics))\b"
            # Trend phrasing without the literal word "trend"
            r"|\bover\s+the\s+(?:past|last|recent)\s+\w+"
            r"|\bchang(?:ed|es|ing)\s+(?:over|across)\b"
            # Domain-anchored "average" — narrow to dataset-aggregation contexts.
            # Bare \baverage\b is unsafe: it would also match exposure-calculation
            # queries (e.g., "8-hour average concentration") that belong to the
            # calculation/reasoning route, not data_analyst.
            r"|\baverage\s+\w*\s*(?:penalt|violation|inspection|accident"
            r"|fatalit|standard|incident|injur)",
            re.I,
        ),
        "agents": ["data_analyst"],
    },
    {
        "description": "calculation/reasoning query",
        "pattern": re.compile(
            r"\bcalculat\w*\b|\bcompute\b|\bdetermine if\b|\bTWA\b"
            r"|\bexposure limit\b|\bpermissible\b|\bPEL\b|\bTLV\b"
            r"|\btime[\s-]weighted\b|\bconcentration\w*\b.*\bshift\b",
            re.I,
        ),
        "agents": ["osha_regs"],
    },
    {
        "description": "quantitative query",
        "pattern": re.compile(
            r"\bhow many\b|\bcount\b|\btotal\b|\bnumber of\b|\btrend\b|\bstatistic",
            re.I,
        ),
        "agents": ["data_analyst", "osha_enf"],
    },
    {
        "description": "incident facts query",
        "pattern": re.compile(
            r"\bwhat happened\b|\bincidents?\b|\binvestigations?\b|\bexplosion\b|\bfire\b|\brelease\b",
            re.I,
        ),
        "agents": ["csb_incident"],
    },
    {
        # A.7 analysis (2026-05-26): recommendation-focused queries need csb_recs only.
        # Queries that also need csb_incident will separately match "incident facts query".
        "description": "prevention/recommendations query",
        "pattern": re.compile(
            r"\brecommend\w*\b|\bprevention\b|\bsafety improvement\b", re.I
        ),
        "agents": ["csb_recs"],
    },
    {
        "description": "root cause query",
        "pattern": re.compile(
            r"\broot causes?\b|\bwhy did\b|\bsystemic\b|\bfailure analysis\b|\bcontributing factors?\b|\bfailure modes?\b",
            re.I,
        ),
        "agents": ["csb_incident", "root_cause"],
    },
    {
        "description": "regulatory/compliance query",
        "pattern": re.compile(
            r"\bregulation\b|\bcompliance\b|\bstandard\b|\brequirement\b"
            r"|\b29 CFR\b|\b1910\.\d+|\bOSHA\s+\d{4}\.",
            re.I,
        ),
        "agents": ["osha_regs"],
    },
    {
        "description": "enforcement/violation query",
        "pattern": re.compile(
            r"\bviolations?\b|\bpenalties?\b|\bcitations?\b|\benforcement\b"
            r"|\bfine\b|\bfined\b|\baccidents?\b|\bfatalit(?:y|ies)\b"
            r"|\bgeneral\s+duty\b|\b5\(a\)\(1\)\b|\brecognized\s+hazard\b",
            re.I,
        ),
        "agents": ["osha_enf"],
    },
    {
        "description": "cross-corpus CSB+OSHA query",
        "pattern": re.compile(
            r"(?=.*\bCSB\b.*\b(?:OSHA|1910\.\d+|29 CFR|regulation|compliance|standard|requirement)\b)"
            r"|(?=.*\b(?:OSHA|1910\.\d+|29 CFR)\b.*\bCSB\b)"
            r"|(?=.*\bCSB[\s-]identified\b)",
            re.I | re.DOTALL,
        ),
        "agents": ["csb_incident", "osha_regs"],
    },
]


def _try_fast_route(query: str) -> Optional[Dict[str, Any]]:
    """Return fast-route planning metadata for common query patterns."""
    matched_routes = [route for route in _FAST_ROUTES if route["pattern"].search(query)]
    if not matched_routes:
        return None

    merged_agents = []
    for route in matched_routes:
        for agent in route["agents"]:
            if agent in VALID_AGENTS and agent not in merged_agents:
                merged_agents.append(agent)

    if not merged_agents:
        return None

    is_csb_focused = bool(re.search(r"\bCSB\b|Chemical Safety Board", query, re.I))
    has_explicit_osha = bool(
        re.search(r"\bOSHA\b|29 CFR|1910\.|penalt|violation type", query, re.I)
    )

    # P1: Suppress agent bleed-through when enforcement pattern fires alongside
    # other patterns that add irrelevant agents.
    matched_descriptions = {r["description"] for r in matched_routes}
    if "enforcement/violation query" in matched_descriptions:
        if (
            "regulatory/compliance query" in matched_descriptions
            and len(matched_descriptions) > 1
        ):
            merged_agents = [a for a in merged_agents if a != "osha_regs"]
        # Do NOT strip csb_incident or root_cause if the query is CSB-focused
        if "incident facts query" in matched_descriptions and not is_csb_focused:
            merged_agents = [a for a in merged_agents if a != "csb_incident"]
        if "root cause query" in matched_descriptions and not is_csb_focused:
            merged_agents = [a for a in merged_agents if a != "root_cause"]
        if "prevention/recommendations query" in matched_descriptions:
            merged_agents = [a for a in merged_agents if a != "csb_recs"]

        # If CSB-focused and mentions "citation frequency across reports" without mentioning OSHA,
        # strip osha_enf so it doesn't hijack CSB investigation analysis.
        if is_csb_focused and not has_explicit_osha and "osha_enf" in merged_agents:
            if len(merged_agents) > 1:
                merged_agents = [a for a in merged_agents if a != "osha_enf"]

    # Fix 2b (2026-05-27): When recommendation pattern co-fires with incident facts
    # pattern, suppress csb_incident.
    if "prevention/recommendations query" in matched_descriptions:
        if (
            "incident facts query" in matched_descriptions
            and len(merged_agents) > 1
            and not is_csb_focused
        ):
            merged_agents = [a for a in merged_agents if a != "csb_incident"]

    # Fix 3 (2026-05-27): When analytical/comparative or quantitative pattern fires
    # and data_analyst is routed, suppress osha_enf and osha_regs that bleed in from
    # enforcement/quantitative/regulatory/cross-corpus co-fires.
    analytical_patterns = {"analytical/comparative query", "quantitative query"}
    if matched_descriptions & analytical_patterns and "data_analyst" in merged_agents:
        if len(merged_agents) > 1 and (is_csb_focused or not has_explicit_osha):
            merged_agents = [a for a in merged_agents if a != "osha_enf"]
        if len(merged_agents) > 1:
            merged_agents = [a for a in merged_agents if a != "osha_regs"]

    if not merged_agents:
        return None

    route_labels = [route["description"] for route in matched_routes]
    route_desc = " + ".join(route_labels)
    agent_desc = " + ".join(merged_agents)

    return {
        "sub_tasks": [f"Fast-routed: {route_desc} -> {agent_desc}"],
        "planned_agents": merged_agents,
        "planning_fast_routed": True,
        "planning_fast_route_patterns": route_labels,
    }


def _parse_plan_output(raw_output: str) -> dict:
    """Parse the supervisor's JSON plan for sub-tasks and agent routing."""
    try:
        return json.loads(raw_output)
    except json.JSONDecodeError:
        return {"sub_tasks": ["Failed to plan"], "agents_to_call": []}


# ---------------------------------------------------------------------------
# fictional-entity / unverifiable-premise detection
# ---------------------------------------------------------------------------
#
# Adversarial queries reference fabricated incidents, citation/activity/accident
# numbers, or facilities. Retrieval cannot ground them, yet the synthesis
# confidence gate may still answer assertively (it keys on chunk *count*, not
# chunk *relevance*). We detect a specific entity reference that retrieval failed
# to corroborate and, only then, force premise-verification hedging.
#
# Design intent: keep standard-query false positives low. Detection is
# *grounding-gated* — a named entity that retrieval surfaces (the normal case for
# legitimate queries) never triggers. Acceptance is decided by the eval gate:
# adversarial hedging >= 80% AND standard hedging FP <= 30% (Tier A). The
# SUPERVISOR_PREMISE_CHECK_ENABLED toggle lets the sweep A/B this cleanly.

# 6+ digit identifiers: OSHA citation/activity/inspection/accident report numbers.
_ID_TOKEN_RE = re.compile(r"\b\d{6,}\b")

# Quoted spans are usually proper names the user is asserting exist.
_QUOTED_RE = re.compile(r"[\"“”']([^\"“”']{3,})[\"“”']")

# "<Proper Name> incident/explosion/fire/..." (name precedes the event word).
_NAMED_EVENT_RE = re.compile(
    r"\b((?:[A-Z][\w&.\-]+\s+){0,4}[A-Z][\w&.\-]+)\s+"
    r"(?:incident|explosion|fire|disaster|accident|release|spill|catastrophe|tragedy|leak|blast)\b"
)

# "incident/explosion at (the) <Proper Name>" (event word precedes the name).
_EVENT_AT_RE = re.compile(
    r"\b(?:incident|explosion|fire|disaster|accident|release|spill|catastrophe|blast)\s+at\s+"
    r"(?:the\s+)?((?:[A-Z][\w&.\-]+\s+){0,4}[A-Z][\w&.\-]+)"
)

# Generic regulatory/agency tokens that are not "named entities" on their own.
_GENERIC_ENTITY_TOKENS = {
    "osha",
    "csb",
    "epa",
    "cfr",
    "psm",
    "nfpa",
    "api",
    "the",
    "a",
    "an",
    "report",
    "accident",
    "incident",
    "inspection",
    "citation",
    "violation",
    "company",
    "facility",
    "plant",
    "refinery",
    "employee",
    "employer",
}

_STOPWORDS = {"the", "and", "of", "at", "in", "a", "an", "for", "to", "on"}


def _extract_specific_entities(query: str) -> list[str]:
    """Extract specific, assertable entities (IDs and proper names) from *query*."""
    ents: list[str] = []
    ents.extend(_ID_TOKEN_RE.findall(query))
    for rx in (_QUOTED_RE, _NAMED_EVENT_RE, _EVENT_AT_RE):
        ents.extend(m.strip() for m in rx.findall(query))

    out: list[str] = []
    seen: set[str] = set()
    for e in ents:
        key = e.lower()
        if key in seen:
            continue
        # Drop purely-generic captures (e.g. "OSHA accident" → name part "OSHA").
        tokens = [t for t in re.findall(r"[a-z0-9]+", key) if t not in _STOPWORDS]
        if (
            not e[0].isdigit()
            and tokens
            and all(t in _GENERIC_ENTITY_TOKENS for t in tokens)
        ):
            continue
        seen.add(key)
        out.append(e)
    return out


def _entity_grounded(entity: str, context_lower: str) -> bool:
    """True if *entity* (or a strong majority of its distinctive tokens) is in context."""
    e = entity.lower()
    if e in context_lower:
        return True
    if e.isdigit():  # IDs must match exactly — partial digit overlap is meaningless
        return False
    tokens = [t for t in re.findall(r"[a-z0-9]{3,}", e) if t not in _STOPWORDS]
    if not tokens:
        return e in context_lower
    hits = sum(1 for t in tokens if t in context_lower)
    return hits / len(tokens) >= 0.6


def detect_unverifiable_premise(query: str, context_text: str) -> Optional[str]:
    """Return a reason string when the query asserts a specific entity that
    retrieval did not corroborate, else None.

    Flags when:
      - a specific identifier (6+ digit citation/activity/accident number) is
        absent from the retrieved context, OR
      - the query names proper entities and NONE of them are grounded.

    Conservative by design (requires *no* grounded named entity) to minimise
    false positives on legitimate queries.
    """
    entities = _extract_specific_entities(query)
    if not entities:
        return None

    ctx = (context_text or "").lower()
    ids = [e for e in entities if e.isdigit()]
    names = [e for e in entities if not e.isdigit()]

    ungrounded_ids = [i for i in ids if not _entity_grounded(i, ctx)]
    if ungrounded_ids:
        return (
            "the query references identifier(s) "
            + ", ".join(ungrounded_ids[:2])
            + " that do not appear in the retrieved evidence"
        )

    if names and not any(_entity_grounded(n, ctx) for n in names):
        return (
            f"the query references '{names[0]}', which does not appear in the "
            "retrieved evidence"
        )

    return None


PLANNING_SYSTEM_PROMPT = """You are the Orchestrator for a Chemical Process Safety analysis system. \
You decompose complex safety queries into targeted sub-tasks and route them to specialized worker agents.

You must select ALL workers needed to comprehensively answer the user's question. \
Chemical process safety analysis often requires cross-referencing multiple data sources \
(e.g., incident facts AND regulatory standards AND enforcement history).

AVAILABLE WORKERS:
- "osha_regs": Searches OSHA compliance rules, PSM standards (29 CFR 1910.119), and general industry regulations.
- "osha_enf": Searches historical OSHA enforcement actions, citations, penalties, and violation patterns.
- "csb_incident": Searches CSB final investigation reports for factual incident timelines, causes, and equipment involved.
- "csb_recs": Searches CSB safety recommendations, target audiences, and closure statuses.
- "root_cause": Performs deep root-cause analysis by cross-referencing incident facts against PSM regulatory elements.
- "data_analyst": Performs statistical queries (counts, averages, trends, distributions, rankings, comparisons) against structured OSHA enforcement databases (13M+ violations, 5M+ inspections, 165K+ accidents) and CSB incident/recommendation summaries. Use this worker for ANY question involving numbers, quantities, percentages, trends over time, rankings, or comparisons across categories.

ROUTING RULES:
- For questions about "what happened": always include "csb_incident".
- For questions about "why it happened" or root causes: include "csb_incident" AND "root_cause".
- For questions about regulations or compliance: include "osha_regs".
- For questions about violations or penalties: include "osha_enf".
- For questions about recommendations or prevention: include "csb_recs".
- For questions involving counts, trends, statistics, rankings, comparisons, percentages, averages, or "how many/how much": ALWAYS include "data_analyst".
- For hybrid analytical queries (e.g., "What are the most common violations leading to fatalities?"): include BOTH "data_analyst" AND relevant retrieval workers.
- For complex analytical queries: select MULTIPLE workers to gather comprehensive evidence."""


def supervisor_plan(state: AgentState) -> dict:
    """PlanNode: Decompose query into sub-tasks and select worker agents.

    Returns a dict (state update) with sub_tasks and planned_agents.
    The conditional edge ``route_from_plan`` reads ``planned_agents``
    and returns ``Send`` objects for parallel dispatch.
    """
    query = state.get("query", "") or ""
    fast_route_result = _try_fast_route(query)
    if fast_route_result:
        return {
            **fast_route_result,
            "llm_calls": [],
        }

    plan_prompt = f"""Analyze the following user query and create an execution plan.

USER QUERY: "{query}"

Respond with strictly formatted JSON:
{{
    "sub_tasks": ["Description of sub-task 1", "Description of sub-task 2", ...],
    "agents_to_call": ["worker_name_1", "worker_name_2", ...]
}}

The "agents_to_call" list must contain ONLY valid worker names from your available workers. \
Include ALL workers needed to fully answer the query. Order them logically \
(e.g., fact retrieval before analysis)."""

    validate_model_parameters(
        model=SUPERVISOR_MODEL,
        temperature=SUPERVISOR_TEMPERATURE,
        max_completion_tokens=SUPERVISOR_PLANNING_MAX_TOKENS,
    )
    llm = make_chat_model(
        "supervisor",
        temperature=SUPERVISOR_TEMPERATURE,
        max_completion_tokens=SUPERVISOR_PLANNING_MAX_TOKENS,
        response_format={"type": "json_object"},
    )
    t0 = time.time()
    response = llm.invoke(
        [
            SystemMessage(content=PLANNING_SYSTEM_PROMPT),
            HumanMessage(content=plan_prompt),
        ]
    )
    latency = time.time() - t0

    llm_calls = record_llm_call(
        response=response,
        node_name="Supervisor",
        call_purpose="planning",
        model_config=SUPERVISOR_MODEL,
        temperature=SUPERVISOR_TEMPERATURE,
        system_prompt=PLANNING_SYSTEM_PROMPT,
        user_prompt=plan_prompt,
        latency_s=latency,
        existing_calls=[],
    )

    plan = _parse_plan_output(response.content)
    agents = [a for a in plan.get("agents_to_call", []) if a in VALID_AGENTS]

    if not agents:
        agents = ["csb_incident"]

    sub_tasks = plan.get("sub_tasks", [])

    return {
        "sub_tasks": sub_tasks,
        "planned_agents": agents,
        "llm_calls": llm_calls,
        "planning_fast_routed": False,
        "planning_fast_route_patterns": [],
    }


def route_from_plan(state: AgentState) -> list:
    """Conditional edge after PlanNode: fan-out to workers via Send."""
    agents = state.get("planned_agents", [])
    if not agents:
        agents = ["csb_incident"]

    sends = []
    for agent in agents:
        node_name = AGENT_TO_NODE.get(agent)
        if node_name:
            sends.append(Send(node_name, state))

    return sends


def supervisor_synthesize(state: AgentState) -> dict:
    """Phase 3: Synthesize a response from all gathered worker context."""
    query = state.get("query", "")
    context = state.get("context_gathered", {})
    critic_feedback = state.get("critic_feedback", "")
    retry_count = state.get("retry_count", 0)

    context_str = json.dumps(context, indent=2)

    revision_block = ""
    if critic_feedback and retry_count > 0:
        revision_block = f"""
IMPORTANT - PREVIOUS DRAFT WAS REJECTED BY VALIDATOR:
The following issues were identified in your previous draft. You MUST address every one:
{critic_feedback}

Revise your response to fix ALL identified issues. Remove any unsupported claims. \
Add explicit acknowledgments where evidence is lacking."""

    worker_errors = state.get("worker_errors", [])
    error_block = ""
    if worker_errors:
        error_lines = "\n".join(f"- {e['node']}: {e['impact']}" for e in worker_errors)
        error_block = (
            f"\n\nWORKER FAILURES (must acknowledge in response):\n{error_lines}"
        )

    context_chunk_count = 0
    if isinstance(context, dict):
        for value in context.values():
            if isinstance(value, list):
                context_chunk_count += len(value)
            elif value:
                context_chunk_count += 1

    # Confidence gate: determines whether the synthesis prompt uses assertive
    # or cautious language.  On first pass (validation_score == 0.0, no critic
    # yet) we fall back to a chunk-count heuristic — rich context (≥10 chunks)
    # is treated as confident.  On revision passes the critic score is available.
    prior_score = float(state.get("validation_score", 0.0) or 0.0)
    confident_mode = (prior_score >= 0.85 and context_chunk_count >= 5) or (
        prior_score == 0.0 and context_chunk_count >= 10
    )

    # premise verification. If the query asserts a specific entity that
    # retrieval did not corroborate, force cautious mode and inject a verification
    # directive. This catches adversarial / fictional-entity queries that would
    # otherwise be answered assertively because the confidence gate keys on chunk
    # count, not relevance. Grounding-gated → legitimate (grounded) entities are
    # unaffected, keeping standard hedging FP low.
    premise_block = ""
    if SUPERVISOR_PREMISE_CHECK_ENABLED:
        premise_reason = detect_unverifiable_premise(query, context_str)
        if premise_reason:
            confident_mode = False  # never answer an uncorroborated premise assertively
            premise_block = (
                "\n\nPREMISE VERIFICATION (CRITICAL): "
                f"{premise_reason}. Do NOT assume the referenced incident, citation, "
                "facility, or event exists, and do NOT fabricate any details about it. "
                "State plainly that you could not find records matching the referenced "
                "entity in the available OSHA/CSB sources, and answer only with what the "
                "retrieved evidence actually supports. If the premise itself appears "
                "unfounded, say so directly."
            )

    # Rule 3: gap-acknowledgement wording, gated on confidence.
    if confident_mode:
        rule_3 = (
            "3. If a SPECIFIC sub-question has zero relevant chunks, note the gap briefly "
            "in-line (one sentence, e.g., 'No data was retrieved for [aspect].'). "
            "Do NOT add a standalone 'insufficient evidence' paragraph or concluding disclaimer. "
            "The evidence is adequate overall."
        )
    else:
        rule_3 = (
            "3. If the retrieved context does not contain sufficient information to answer "
            "any part of the query, state which SPECIFIC aspect lacks evidence. "
            "Do NOT add blanket disclaimers — be precise about what is missing and what is covered. "
            "Note evidence limitations ONLY if they directly prevent answering what was asked (max 1-2 bullet points). "
            "Do NOT list unrequested topics, unretrieved subsections, or missing databases."
        )

    # Rule 7: tone directive, gated on confidence.
    if confident_mode:
        confidence_block = (
            "7. CONFIDENCE MODE: The retrieved evidence is strong. Write with direct, "
            "authoritative language. Do NOT include phrases like 'insufficient evidence', "
            "'based on available data', 'cannot be determined', or 'evidence suggests'. "
            "If a narrow sub-point has no data, state 'no data was retrieved for [aspect]' "
            "in one sentence and move on — do NOT add concluding disclaimers or caveats."
        )
    else:
        confidence_block = (
            "7. Use cautious language only for claims with weak or no supporting chunks. "
            "Be specific about what lacks evidence — do not add blanket disclaimers."
        )

    synthesis_prompt = f"""Synthesize a comprehensive response to the user's query using ONLY \
the retrieved context below.

USER QUERY: "{query}"

RETRIEVED CONTEXT:
{context_str}
{revision_block}
{error_block}
{premise_block}

FORMATTING REQUIREMENTS:
- Target length: 500–1,500 words (up to 2,000 words for complex multi-incident analyses). Start directly with an executive summary answering the question before deep-dive sections.
- ANTI-REDUNDANCY: Present each finding or root cause ONCE under its most relevant section. Do NOT repeat findings across multiple sections or create duplicate summary lists of the same points.
- Do NOT summarize retrieved chunks that are irrelevant to the specific question asked.
- Use Markdown headings (## for sections, ### for subsections) — NOT plain text lines.
- For year-by-year breakdowns, counts, comparisons, or any tabular data, use a Markdown table:
  | Year | Count |
  |------|-------|
  | 2020 | 5     |
- Use bullet lists (- item) only for short enumerations or qualitative points.
- Use sub-bullets with two-space indent for supporting detail under a bullet.
- Keep paragraphs concise. Separate sections with blank lines.

CONTENT REQUIREMENTS:
1. Base every factual claim (incidents, dates, chemicals, regulations, penalties) EXCLUSIVELY \
on the retrieved context. Do NOT use your general knowledge to fill gaps.
2. For each major claim, include an inline citation referencing the source collection and document \
(e.g., "[CSB Report: Facility Name]" or "[OSHA 1910.119(j)]").
{rule_3}
4. Distinguish between facts directly from sources and analytical inferences drawn from those facts.
5. For statistical claims derived from database queries, cite the exact SQL query and result \
 (e.g., "[Analytical: COUNT of violations WHERE standard_number LIKE '1910.119%' = 4,523]").
6. When combining statistical evidence with retrieved text context, clearly distinguish between \
quantitative facts (from database queries) and qualitative evidence (from document retrieval).
{confidence_block}"""

    synthesis_system = """You are a Chemical Process Safety analyst synthesizing a factual, \
well-cited response from retrieved evidence. You operate under strict evidentiary standards: \
every claim must trace to a source document. Unsupported claims in this domain can lead to \
incorrect safety decisions and loss of life. When evidence is insufficient, say so explicitly \
rather than speculating."""

    validate_model_parameters(
        model=SUPERVISOR_MODEL,
        temperature=SUPERVISOR_TEMPERATURE,
        max_completion_tokens=SUPERVISOR_SYNTHESIS_MAX_TOKENS,
    )
    llm = make_chat_model(
        "supervisor",
        temperature=SUPERVISOR_TEMPERATURE,
        max_completion_tokens=SUPERVISOR_SYNTHESIS_MAX_TOKENS,
    )
    t0 = time.time()
    emitter = _TOKEN_EMITTER.get()
    if emitter is not None and hasattr(llm, "stream"):
        chunks = []
        for chunk in llm.stream(
            [
                SystemMessage(content=synthesis_system),
                HumanMessage(content=synthesis_prompt),
            ]
        ):
            chunks.append(chunk)
            if chunk.content:
                try:
                    emitter(chunk.content)
                except Exception:
                    pass
        response = sum(chunks[1:], chunks[0]) if chunks else AIMessage(content="")
    else:
        response = llm.invoke(
            [
                SystemMessage(content=synthesis_system),
                HumanMessage(content=synthesis_prompt),
            ]
        )
    latency = time.time() - t0

    llm_calls = record_llm_call(
        response=response,
        node_name="Supervisor",
        call_purpose="synthesis",
        model_config=SUPERVISOR_MODEL,
        temperature=SUPERVISOR_TEMPERATURE,
        system_prompt=synthesis_system,
        user_prompt=synthesis_prompt,
        latency_s=latency,
        existing_calls=[],
    )

    return {
        "draft_response": response.content,
        "active_agent": "CRITIC",
        "llm_calls": llm_calls,
    }


def supervisor_revise(state: AgentState) -> dict:
    """RevisionNode: Targeted patch of the draft using only critic feedback.

    Unlike full re-synthesis, this does NOT re-inject the full retrieved context.
    It sends only the draft, the critique, and a summary of cited sources,
    producing a focused revision that is faster and cheaper.
    """
    query = state.get("query", "")
    draft = state.get("draft_response", "")
    critic_feedback = state.get("critic_feedback", "")
    context = state.get("context_gathered", {})

    # Build a compact source summary instead of dumping all context
    source_summary_lines = []
    for collection, chunks in context.items():
        if isinstance(chunks, list):
            source_summary_lines.append(f"- {collection}: {len(chunks)} chunk(s)")
        else:
            source_summary_lines.append(f"- {collection}: (single entry)")
    source_summary = "\n".join(source_summary_lines) or "No sources available."

    revision_prompt = f"""You are revising a Chemical Process Safety response based on validator feedback.

USER QUERY: "{query}"

CURRENT DRAFT:
{draft}

VALIDATOR FEEDBACK (must address every issue):
{critic_feedback}

AVAILABLE SOURCE COLLECTIONS:
{source_summary}

REVISION INSTRUCTIONS:
1. Fix ONLY the specific issues identified in the validator feedback.
2. Keep all correctly-grounded content unchanged — do not rewrite sections that passed validation.
3. Remove or rephrase any claims the validator flagged as unsupported.
4. Add explicit "insufficient evidence" acknowledgments where the validator noted gaps.
5. Preserve the existing Markdown structure (## headings, tables, bullet lists).
6. Do NOT introduce new claims not present in the original draft or source collections.

Return the complete revised response (not just the changed parts)."""

    revision_system = """You are a Chemical Process Safety editor performing targeted revisions. \
You fix only the issues identified by the validator while preserving all correct content. \
Never introduce new unsupported claims during revision."""

    llm = make_chat_model(
        "supervisor",
        temperature=SUPERVISOR_TEMPERATURE,
        max_completion_tokens=SUPERVISOR_REVISION_MAX_TOKENS,
    )
    validate_model_parameters(
        model=SUPERVISOR_MODEL,
        temperature=SUPERVISOR_TEMPERATURE,
        max_completion_tokens=SUPERVISOR_REVISION_MAX_TOKENS,
    )
    t0 = time.time()
    response = llm.invoke(
        [
            SystemMessage(content=revision_system),
            HumanMessage(content=revision_prompt),
        ]
    )
    latency = time.time() - t0

    llm_calls = record_llm_call(
        response=response,
        node_name="RevisionNode",
        call_purpose="revision",
        model_config=SUPERVISOR_MODEL,
        temperature=SUPERVISOR_TEMPERATURE,
        system_prompt=revision_system,
        user_prompt=revision_prompt,
        latency_s=latency,
        existing_calls=[],
    )

    return {
        "draft_response": response.content,
        "active_agent": "CRITIC",
        "llm_calls": llm_calls,
    }


# Keep backward-compatible alias for any code that references supervisor_agent
def supervisor_agent(state: AgentState) -> dict:
    """Legacy three-phase supervisor (kept for backward compatibility).

    New code should use supervisor_plan() and supervisor_synthesize() directly.
    """
    context = state.get("context_gathered", {})
    pending = state.get("pending_agents", [])

    if not context and not pending:
        # Plan phase — but in the new graph this is handled by supervisor_plan
        # which returns Send objects. This path only runs if used with old graph.
        return supervisor_synthesize(state)

    if pending:
        next_agent = pending[0]
        remaining = pending[1:]
        return {
            "pending_agents": remaining,
            "active_agent": next_agent,
        }

    return supervisor_synthesize(state)
