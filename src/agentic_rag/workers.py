import logging
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.messages import HumanMessage, SystemMessage

from src.agentic_rag.config import (
    DATA_ANALYST_SQL_GENERATION_MAX_TOKENS,
    DATA_ANALYST_SQL_VERIFICATION_MAX_TOKENS,
    HYBRID_VECTOR_TOP_K_BY_COLLECTION,
    REASONING_MAX_TOKENS,
    REASONING_TEMPERATURE,
    WORKER_QUERY_REFORMULATION_ENABLED,
    WORKER_QUERY_REFORMULATION_MAX_TOKENS,
    WORKER_TEMPERATURE,
)
from src.agentic_rag.llm_factory import make_chat_model
from src.agentic_rag.llm_logger import record_llm_call, validate_model_parameters
from src.agentic_rag.state import AgentState
from src.agentic_rag.tools import (
    _CSB_INCIDENT_NAME_MAP,
    _extract_citation_ids,
    _extract_enforcement_standard_number,
    _extract_establishment_at_clause,
    get_chroma_tools,
)
from src.data_processing.config import (
    AGENT_DATA_ANALYST_MODEL,
    AGENT_REASONING_MODEL,
    AGENT_WORKER_MODEL,
)

logger = logging.getLogger(__name__)

TOOLS = get_chroma_tools()

DATA_ANALYST_MODEL = AGENT_DATA_ANALYST_MODEL

WORKER_MODEL = AGENT_WORKER_MODEL
REASONING_MODEL = AGENT_REASONING_MODEL

_STATE_ABBREVS = {
    "AL",
    "AK",
    "AZ",
    "AR",
    "CA",
    "CO",
    "CT",
    "DE",
    "FL",
    "GA",
    "HI",
    "ID",
    "IL",
    "IN",
    "IA",
    "KS",
    "KY",
    "LA",
    "ME",
    "MD",
    "MA",
    "MI",
    "MN",
    "MS",
    "MO",
    "MT",
    "NE",
    "NV",
    "NH",
    "NJ",
    "NM",
    "NY",
    "NC",
    "ND",
    "OH",
    "OK",
    "OR",
    "PA",
    "RI",
    "SC",
    "SD",
    "TN",
    "TX",
    "UT",
    "VT",
    "VA",
    "WA",
    "WV",
    "WI",
    "WY",
}

_STATE_NAMES_TO_ABBREV = {
    "Alabama": "AL",
    "Alaska": "AK",
    "Arizona": "AZ",
    "Arkansas": "AR",
    "California": "CA",
    "Colorado": "CO",
    "Connecticut": "CT",
    "Delaware": "DE",
    "Florida": "FL",
    "Georgia": "GA",
    "Hawaii": "HI",
    "Idaho": "ID",
    "Illinois": "IL",
    "Indiana": "IN",
    "Iowa": "IA",
    "Kansas": "KS",
    "Kentucky": "KY",
    "Louisiana": "LA",
    "Maine": "ME",
    "Maryland": "MD",
    "Massachusetts": "MA",
    "Michigan": "MI",
    "Minnesota": "MN",
    "Mississippi": "MS",
    "Missouri": "MO",
    "Montana": "MT",
    "Nebraska": "NE",
    "Nevada": "NV",
    "New Hampshire": "NH",
    "New Jersey": "NJ",
    "New Mexico": "NM",
    "New York": "NY",
    "North Carolina": "NC",
    "North Dakota": "ND",
    "Ohio": "OH",
    "Oklahoma": "OK",
    "Oregon": "OR",
    "Pennsylvania": "PA",
    "Rhode Island": "RI",
    "South Carolina": "SC",
    "South Dakota": "SD",
    "Tennessee": "TN",
    "Texas": "TX",
    "Utah": "UT",
    "Vermont": "VT",
    "Virginia": "VA",
    "Washington": "WA",
    "West Virginia": "WV",
    "Wisconsin": "WI",
    "Wyoming": "WY",
}

_REFORMULATE_SYSTEM = (
    "You are a search query optimizer for chemical process safety databases."
)

# ---------------------------------------------------------------------------
# CSB incident name lookup imported from tools to avoid duplication.
# See src.agentic_rag.tools._CSB_INCIDENT_NAME_MAP
# ---------------------------------------------------------------------------


def _reformulate_query(
    user_query: str,
    sub_tasks: List[str],
    domain_context: str,
    model: str = WORKER_MODEL,
    existing_llm_calls: Optional[List[Dict[str, Any]]] = None,
    node_name: str = "unknown",
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Use an LLM to reformulate the user query into a domain-optimized
    retrieval query for a specific collection.

    Returns (reformulated_query, updated_llm_calls).

    Ablation switch: when
    ``WORKER_QUERY_REFORMULATION_ENABLED`` is false, skips the LLM call
    entirely and returns the raw ``user_query`` unchanged -- every call site
    goes through this one helper, so the ablation applies uniformly across
    all worker/tool query paths without touching each call site.
    """
    if not WORKER_QUERY_REFORMULATION_ENABLED:
        return user_query, list(existing_llm_calls or [])

    sub_tasks_str = (
        "\n".join(f"- {t}" for t in sub_tasks) if sub_tasks else "None specified"
    )

    prompt = f"""Given a user's chemical process safety question and the specific sub-task assigned to you,
generate an optimized search query for retrieving relevant documents from a vector database.

USER QUESTION: {user_query}
SUB-TASKS ASSIGNED: {sub_tasks_str}
YOUR DOMAIN: {domain_context}

INSTRUCTIONS:
- Produce a concise, keyword-rich search query (1-3 sentences) optimized for semantic similarity search.
- Include specific regulatory standard numbers (e.g., 29 CFR 1910.119), chemical names, equipment types,
  or facility identifiers mentioned or implied by the question.
- Do NOT answer the question. Only produce the search query.
- Output ONLY the search query text, nothing else."""

    validate_model_parameters(
        model=model,
        temperature=WORKER_TEMPERATURE,
        max_completion_tokens=WORKER_QUERY_REFORMULATION_MAX_TOKENS,
    )
    llm = make_chat_model(
        "worker",
        temperature=WORKER_TEMPERATURE,
        max_completion_tokens=WORKER_QUERY_REFORMULATION_MAX_TOKENS,
    )
    t0 = time.time()
    response = llm.invoke(
        [
            SystemMessage(content=_REFORMULATE_SYSTEM),
            HumanMessage(content=prompt),
        ]
    )
    latency = time.time() - t0

    llm_calls = record_llm_call(
        response=response,
        node_name=node_name,
        call_purpose="query_reformulation",
        model_config=model,
        temperature=WORKER_TEMPERATURE,
        system_prompt=_REFORMULATE_SYSTEM,
        user_prompt=prompt,
        latency_s=latency,
        existing_calls=existing_llm_calls,
    )

    return response.content.strip(), llm_calls


_FACILITY_SUFFIX_RE = re.compile(
    r"\b([A-Za-z0-9&'./-]+(?:\s+[A-Za-z0-9&'./-]+){0,4}\s+"
    r"(?:refinery|plant|facility|terminal|complex|works|laboratory|lab))\b",
    re.IGNORECASE,
)


def _extract_facility_name(query: str) -> Optional[str]:
    quoted = re.findall(r'"([^"]{3,80})"', query)
    if quoted:
        return quoted[0].strip()

    match = _FACILITY_SUFFIX_RE.search(query)
    if match:
        return re.sub(r"\s+", " ", match.group(1)).strip()
    return None


def _extract_metadata_filters(
    query: str, tool_name: Optional[str] = None
) -> Optional[Dict]:
    """Extract structured metadata filters from the query for ChromaDB where clauses."""
    filters: Dict[str, Any] = {}
    query_lc = query.lower()

    states_found: List[str] = []
    for abbr in sorted(_STATE_ABBREVS):
        # Avoid matching "& CO" as Colorado (CO).
        if re.search(rf"(?<!& )\b{re.escape(abbr)}\b(?:-based)?", query):
            states_found.append(abbr)
    for name, abbrev in _STATE_NAMES_TO_ABBREV.items():
        if re.findall(r"\b" + re.escape(name) + r"\b", query, re.IGNORECASE):
            if abbrev not in states_found:
                states_found.append(abbrev)

    if states_found:
        if len(states_found) == 1:
            filters["site_state"] = states_found[0]
        else:
            filters["site_state"] = {"$in": states_found}

    if re.search(r"\b(fatal|fatality|fatalities|death|deaths|killed)\b", query_lc):
        filters["fatality"] = True

    if tool_name == "osha_enf":
        if re.search(r"\b(accident|report)\b", query_lc):
            accident_ids = re.findall(r"\b\d{8,9}\b", query)
            if accident_ids:
                unique_ids = list(dict.fromkeys(accident_ids))
                filters["summary_nr"] = (
                    unique_ids[0] if len(unique_ids) == 1 else {"$in": unique_ids}
                )
        elif re.search(
            r"\b(violation|citation|inspection|penalty|penalti)\b", query_lc
        ):
            # Violation/citation queries: extract inspection numbers (activity_nr)
            inspection_ids = re.findall(r"\b\d{9,12}\b", query)
            if inspection_ids:
                unique_ids = list(dict.fromkeys(inspection_ids))
                filters["activity_nr"] = (
                    unique_ids[0] if len(unique_ids) == 1 else {"$in": unique_ids}
                )
            citation_ids = _extract_citation_ids(query)
            if citation_ids:
                filters["citation_id"] = (
                    citation_ids[0] if len(citation_ids) == 1 else {"$in": citation_ids}
                )
            standard_number = _extract_enforcement_standard_number(query)
            if standard_number:
                filters["standard_number"] = standard_number
        facility_name = _extract_establishment_at_clause(
            query
        ) or _extract_facility_name(query)
        if facility_name:
            filters["establishment_name"] = facility_name

    if tool_name == "csb":
        incident_name = _extract_facility_name(query)
        if incident_name:
            filters["incident_name"] = incident_name
        else:
            # Fuzzy lookup: match known incident name substrings in the query
            query_lc = query.lower()
            for substring, canonical_name in _CSB_INCIDENT_NAME_MAP.items():
                if substring in query_lc:
                    filters["incident_name"] = canonical_name
                    break

    return filters if filters else None


def _run_hybrid_query(
    state: AgentState,
    tool_name: str,
    domain_context: str,
    node_name: str,
) -> dict:
    """
    Core worker logic: reformulate query, extract metadata filters,
    run hybrid retrieval, and update context with source-attributed results.
    """
    tool = TOOLS.get(tool_name)
    if not tool:
        result_text = f"Error: Tool {tool_name} not found."
        new_citations = []
        reformulated = ""
        metadata_filters = None
        llm_calls = []
    else:
        reformulated, llm_calls = _reformulate_query(
            state["query"],
            state.get("sub_tasks", []),
            domain_context,
            existing_llm_calls=[],
            node_name=node_name,
        )
        metadata_filters = _extract_metadata_filters(
            state["query"], tool_name=tool_name
        )
        top_k = HYBRID_VECTOR_TOP_K_BY_COLLECTION.get(tool_name, 10)

        try:
            result_text, new_citations = tool.query(
                reformulated,
                metadata_filters=metadata_filters,
                top_k=top_k,
            )
            # Fallback: if metadata filters returned 0 results, retry without filters
            if not new_citations and metadata_filters:
                logger.info(
                    "Metadata-filtered query returned 0 results for %s; "
                    "retrying without filters",
                    tool_name,
                )
                metadata_filters = None
                result_text, new_citations = tool.query(
                    reformulated,
                    metadata_filters=None,
                    top_k=top_k,
                )
        except Exception as e:
            result_text = f"Error executing tool {tool_name}: {e}"
            new_citations = []

    context = {tool_name: [result_text]}

    # Capture retrieval details
    similarity_scores = [
        float(c["similarity_score"])
        for c in new_citations
        if isinstance(c, dict) and isinstance(c.get("similarity_score"), (int, float))
    ]
    avg_similarity_score = (
        round(sum(similarity_scores) / len(similarity_scores), 6)
        if similarity_scores
        else None
    )
    min_similarity_score = (
        round(min(similarity_scores), 6) if similarity_scores else None
    )

    retrieval_detail = {
        "collection": tool_name,
        "node_name": node_name,
        "reformulated_query": reformulated,
        "metadata_filters": metadata_filters,
        "chunk_count": len(new_citations),
        "top_k": top_k if tool else None,
        "total_chars": len(result_text),
        "avg_similarity_score": avg_similarity_score,
        "min_similarity_score": min_similarity_score,
    }

    return {
        "context_gathered": context,
        "citations": new_citations,
        "llm_calls": llm_calls,
        "retrieval_details": [retrieval_detail],
    }


# ---------------------------------------------------------------------------
# Worker agent functions
# ---------------------------------------------------------------------------


def osha_regs_agent(state: AgentState) -> dict:
    """Queries OSHA regulations with domain-optimized retrieval."""
    return _run_hybrid_query(
        state,
        "osha_regulations",
        domain_context=(
            "OSHA general industry and process safety management (PSM) regulations. "
            "Relevant standards include 29 CFR 1910.119 (PSM), 1910.147 (LOTO), "
            "1910.146 (permit-required confined spaces), 1910.1200 (HazCom). "
            "Search for specific compliance mandates, definitions, and requirements."
        ),
        node_name="OshaRegsWorker",
    )


def osha_enf_agent(state: AgentState) -> dict:
    """Queries OSHA enforcement history with domain-optimized retrieval."""
    return _run_hybrid_query(
        state,
        "osha_enf",
        domain_context=(
            "Historical OSHA enforcement actions including citations, penalties, "
            "violation types (willful, serious, repeat), inspection results, and "
            "Standard Industrial Classification (SIC) / NAICS codes for facilities. "
            "Search for penalty amounts, citation trends, and violation patterns."
        ),
        node_name="OshaEnfWorker",
    )


def incident_retriever_agent(state: AgentState) -> dict:
    """Retrieves factual CSB incident data with domain-optimized retrieval."""
    return _run_hybrid_query(
        state,
        "csb",
        domain_context=(
            "U.S. Chemical Safety Board (CSB) final investigation reports containing "
            "factual incident timelines, equipment involved, chemicals released, "
            "injuries/fatalities, and sequences of physical events leading to accidents. "
            "Search for specific incident facts, dates, locations, and equipment failures."
        ),
        node_name="IncidentWorker",
    )


def recommendation_specialist_agent(state: AgentState) -> dict:
    """Retrieves CSB safety recommendations with domain-optimized retrieval."""
    return _run_hybrid_query(
        state,
        "csb_recommendations",
        domain_context=(
            "CSB safety recommendations issued to companies, regulators, and industry "
            "organizations following incident investigations. Includes recommendation "
            "text, target audience, and closure status. Search for specific safety "
            "improvement actions and their implementation status."
        ),
        node_name="RecommendationsWorker",
    )


def root_cause_analyst_agent(state: AgentState) -> dict:
    """
    Reasoning-heavy agent using Framework-Injected Prompting.
    Retrieves PSM regulatory definitions and CSB incident facts, then
    performs structured root-cause analysis mapping physical failures
    to systemic PSM compliance gaps.
    """
    llm_calls = []

    psm_tool = TOOLS.get("osha_regulations")
    csb_tool = TOOLS.get("csb")
    recs_tool = TOOLS.get("csb_recommendations")

    psm_defs = "PSM regulatory definitions not available."
    if psm_tool:
        try:
            psm_defs, _ = psm_tool.query(
                "Process Safety Management PSM elements definitions 29 CFR 1910.119 "
                "mechanical integrity management of change process hazard analysis "
                "operating procedures training emergency planning"
            )
        except Exception:
            pass

    csb_facts = "CSB incident facts not available."
    csb_citations = []
    if csb_tool:
        try:
            reformulated, new_calls = _reformulate_query(
                state["query"],
                state.get("sub_tasks", []),
                "CSB incident facts: timelines, equipment failures, chemical releases, causal sequences",
                existing_llm_calls=[],
                node_name="RootCauseWorker",
            )
            llm_calls.extend(new_calls)
            csb_facts, csb_citations = csb_tool.query(reformulated)
        except Exception:
            pass

    historical_context = "No historical precedent data retrieved."
    if recs_tool:
        try:
            reformulated, new_calls = _reformulate_query(
                state["query"],
                state.get("sub_tasks", []),
                "Similar past incidents and safety recommendations for precedent analysis",
                existing_llm_calls=[],
                node_name="RootCauseWorker",
            )
            llm_calls.extend(new_calls)
            historical_context, _ = recs_tool.query(reformulated)
        except Exception:
            pass

    system_prompt = """You are an expert Process Safety Management (PSM) investigator with deep expertise in \
OSHA 29 CFR 1910.119, chemical process hazard analysis, and industrial incident investigation.

Your analysis MUST be rigorous, systematic, and grounded exclusively in the provided evidence. \
You are performing root-cause analysis for a safety-critical domain where inaccurate conclusions \
can lead to loss of life.

STRICT RULES:
- ONLY use facts explicitly stated in the provided context.
- If evidence is insufficient to determine a root cause, state: "Insufficient evidence in retrieved \
context to determine this factor."
- NEVER fabricate incident details, dates, chemical names, regulatory citations, or causal links.
- Distinguish clearly between confirmed facts, reasonable inferences from evidence, and unknowns."""

    analysis_prompt = f"""Analyze the following incident information and determine systemic root causes.

== OSHA 1910.119 PSM ELEMENT DEFINITIONS (from regulatory database) ==
{psm_defs}

== INCIDENT FACTS (from CSB investigation reports) ==
{csb_facts}

== HISTORICAL PRECEDENTS & RECOMMENDATIONS ==
{historical_context}

== ANALYSIS INSTRUCTIONS ==
1. IMMEDIATE CAUSES: Identify the direct physical/mechanical failures (e.g., valve failure, \
overpressure, loss of containment). Cite the specific source document.
2. SYSTEMIC PSM FAILURES: For each immediate cause, trace it back to a specific PSM element \
deficiency using the regulatory definitions above. For example: Was the failed equipment on a \
Mechanical Integrity inspection schedule per 1910.119(j)? Was a Management of Change review \
conducted per 1910.119(l)?
3. ORGANIZATIONAL FACTORS: Identify management system failures, training gaps, or cultural \
issues evidenced in the documents.
4. EVIDENCE GAPS: Explicitly list what information would be needed but was not found in the \
retrieved context.

Format your response with clear section headers for each category above."""

    validate_model_parameters(
        model=REASONING_MODEL,
        temperature=REASONING_TEMPERATURE,
        max_completion_tokens=REASONING_MAX_TOKENS,
    )
    llm = make_chat_model(
        "reasoning",
        temperature=REASONING_TEMPERATURE,
        max_completion_tokens=REASONING_MAX_TOKENS,
    )
    t0 = time.time()
    response = llm.invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=analysis_prompt),
        ]
    )
    latency = time.time() - t0

    llm_calls.extend(
        record_llm_call(
            response=response,
            node_name="RootCauseWorker",
            call_purpose="root_cause_analysis",
            model_config=REASONING_MODEL,
            temperature=REASONING_TEMPERATURE,
            system_prompt=system_prompt,
            user_prompt=analysis_prompt,
            latency_s=latency,
            existing_calls=[],
        )
    )

    context = {"root_cause_analysis": [response.content]}

    retrieval_details = [
        {
            "collection": "root_cause_composite",
            "node_name": "RootCauseWorker",
            "sources_queried": ["osha_regulations", "csb", "csb_recommendations"],
            "csb_citation_count": len(csb_citations),
        }
    ]

    return {
        "context_gathered": context,
        "citations": csb_citations,
        "llm_calls": llm_calls,
        "retrieval_details": retrieval_details,
    }


# ---------------------------------------------------------------------------
# Data Analyst agent – Text-to-SQL over structured OSHA/CSB data
# ---------------------------------------------------------------------------

_TEXT_TO_SQL_SYSTEM = """You are a SQL query generator for chemical process safety databases.
You generate DuckDB-compatible SQL queries from natural language questions.

STRICT RULES:
- Generate ONLY a single SELECT query. No INSERT, UPDATE, DELETE, CREATE, or DROP.
- Use DuckDB SQL syntax (not MySQL or PostgreSQL specific).
- Focus on aggregate statistics (COUNT, SUM, AVG, GROUP BY) or select concise metadata columns (standard_number, title, viol_type, penalty, fatality). NEVER select raw full-text narrative columns (text, abstract_text, event_desc), file paths (pdf_path, url), or ETL metadata (load_dt). Use vector retrieval workers for full-text searches.
- Always include a LIMIT clause (max 100 rows).
- Use TRY_CAST() for safe type conversions.
- For date filtering, use: EXTRACT(YEAR FROM column_name).
- For case-insensitive text matching, use ILIKE.
- CRITICAL DOMAIN TABLE GUIDANCE:
  1. For chemical process safety contributing factors, root causes, failure modes, or operating modes (steady-state/normal operations, startup, maintenance), ALWAYS query `csb_contributing_factors`.
  2. NEVER query `accidents.event_keyword` for chemical process safety failure modes or operating modes (accidents.event_keyword contains OSHA IMIS worker injury codes like 'STRUCK BY', 'PPE', 'PELVIS', not engineering failure modes).
  3. For OSHA violations, citations, and standard numbers, ALWAYS use `v_violations_parsed` (which has pre-parsed `standard_number` like '1910.119' and `psm_paragraph` like 'j', 'd', 'e', 'f', 'l'). Join with `inspections` on `activity_nr` to filter NAICS ('325%') or SIC ('28%').
- Output ONLY the SQL query, nothing else."""

_DUCKDB_SYNTAX_GUIDE = """DuckDB SYNTAX CHEAT SHEET:
- Date extraction: EXTRACT(YEAR FROM TRY_CAST(date_col AS DATE))
- Case-insensitive match: column ILIKE '%pattern%'
- Safe cast: TRY_CAST(col AS INTEGER), TRY_CAST(col AS DOUBLE)
- String concat: col1 || ' ' || col2
- NULL handling: COALESCE(col, default_value)
- Top-N: ORDER BY col DESC LIMIT N
- Aggregation: COUNT(*), SUM(col), AVG(col), MIN(col), MAX(col)
- Grouping: GROUP BY col1, col2 ... ORDER BY aggregate DESC"""

_FEW_SHOT_EXAMPLES = """EXAMPLE QUERIES:

Q: Query the OSHA enforcement records for NAICS 325 (2010–2019) to determine the top 5 most frequently cited standards.
SQL: SELECT v.standard_number, COUNT(*) AS citation_count
     FROM v_violations_parsed v
     JOIN inspections i ON v.activity_nr = i.activity_nr
     WHERE (i.naics_code LIKE '325%' OR i.sic_code LIKE '28%')
       AND EXTRACT(YEAR FROM TRY_CAST(v.issuance_date AS DATE)) BETWEEN 2010 AND 2019
       AND v.standard_number IS NOT NULL
     GROUP BY v.standard_number
     ORDER BY citation_count DESC
     LIMIT 5;

Q: For chemical manufacturing establishments (NAICS 325) between 2010 and 2019, calculate the total violation count for 29 CFR 1910.119 and rank the top cited paragraphs (j, d, f, e).
SQL: SELECT v.psm_paragraph, COUNT(*) AS violation_count
     FROM v_violations_parsed v
     JOIN inspections i ON v.activity_nr = i.activity_nr
     WHERE v.standard_number = '1910.119'
       AND v.psm_paragraph IS NOT NULL
       AND (i.naics_code LIKE '325%' OR i.sic_code LIKE '28%')
       AND EXTRACT(YEAR FROM TRY_CAST(v.issuance_date AS DATE)) BETWEEN 2010 AND 2019
     GROUP BY v.psm_paragraph
     ORDER BY violation_count DESC;

Q: Rank the primary contributing factors identified in CSB chemical plant investigations (2010–2019) by citation frequency across reports.
SQL: SELECT factor_name, COUNT(DISTINCT incident_name) AS incident_count
     FROM csb_contributing_factors
     WHERE (naics_325_flag = true OR naics_code LIKE '325%')
       AND incident_year BETWEEN 2010 AND 2019
     GROUP BY factor_name
     ORDER BY incident_count DESC;

Q: Analyze the distribution of chemical process incidents across operational phases (steady-state, startup, shutdown, maintenance).
SQL: SELECT operating_mode AS phase,
            COUNT(DISTINCT incident_name) AS incident_count,
            ROUND(100.0 * COUNT(DISTINCT incident_name) / SUM(COUNT(DISTINCT incident_name)) OVER (), 1) AS pct
     FROM csb_contributing_factors
     GROUP BY operating_mode
     ORDER BY incident_count DESC;

Q: Do higher severity chemical process incidents exhibit more multi-factor breakdowns, and which 4-digit NAICS subsector has the highest incidence?
SQL: SELECT severity_group,
            COUNT(DISTINCT incident_name) AS incident_count,
            ROUND(AVG(factor_count), 2) AS avg_multi_factor_breakdowns
     FROM (
         SELECT incident_name, severity_group, COUNT(*) AS factor_count
         FROM csb_contributing_factors
         GROUP BY incident_name, severity_group
     ) sub
     GROUP BY severity_group
     ORDER BY avg_multi_factor_breakdowns DESC;

Q: How many willful violations were issued in Texas in 2023?
SQL: SELECT COUNT(*) AS willful_count
     FROM violations v
     JOIN inspections i ON v.activity_nr = i.activity_nr
     WHERE v.viol_type = 'W'
       AND i.site_state = 'TX'
       AND EXTRACT(YEAR FROM TRY_CAST(v.issuance_date AS DATE)) = 2023;

Q: What are the top 5 companies by total penalty amount for serious violations?
SQL: SELECT i.estab_name, SUM(TRY_CAST(v.current_penalty AS DOUBLE)) AS total_penalty
     FROM violations v
     JOIN inspections i ON v.activity_nr = i.activity_nr
     WHERE v.viol_type = 'S'
     GROUP BY i.estab_name
     ORDER BY total_penalty DESC
     LIMIT 5;"""

_VERIFICATION_SYSTEM = """You are a SQL result verifier for chemical process safety analytics.
Given a user question, the SQL query executed, and the query results, verify that the SQL
correctly answers the user's question.

Respond with ONLY one of:
- "VERIFIED" if the SQL output correctly answers the question.
- "MISMATCH: <brief explanation of what is wrong>" if there is a problem."""


def _extract_sql_from_response(text: str) -> str:
    """Extract SQL from an LLM response, stripping markdown code fences."""
    match = re.match(r".*?```(?:sql)?\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


def data_analyst_agent(state: AgentState) -> dict:
    """Translates analytical queries into SQL against structured OSHA/CSB data.

    Returns statistical results as formatted context with full provenance.
    """
    from src.agentic_rag.analytical_tools import (
        execute_analytical_query,
        get_db,
        get_join_relationships,
        get_registered_views,
        get_schema_description,
    )

    query = state.get("query", "")
    sub_tasks = state.get("sub_tasks", [])
    sub_tasks_str = (
        "\n".join(f"- {t}" for t in sub_tasks) if sub_tasks else "None specified"
    )

    # Get DB connection (lazy singleton with graceful degradation)
    try:
        conn = get_db()
    except Exception as e:
        return {
            "context_gathered": {
                "analytical_results": [
                    f"Error: Could not connect to analytical database: {e}"
                ]
            },
            "citations": [],
            "llm_calls": [],
            "retrieval_details": [],
            "worker_errors": [
                {
                    "node": "DataAnalystWorker",
                    "error": str(e),
                    "impact": "Statistical/quantitative analysis unavailable. "
                    "Results will rely solely on vector-retrieved text chunks.",
                }
            ],
        }

    # Check if any views were registered
    registered = get_registered_views()
    if not registered:
        return {
            "context_gathered": {
                "analytical_results": [
                    "Error: No data files are available for analytical queries."
                ]
            },
            "citations": [],
            "llm_calls": [],
            "retrieval_details": [],
            "worker_errors": [
                {
                    "node": "DataAnalystWorker",
                    "error": "No data files available",
                    "impact": "Statistical/quantitative analysis unavailable. "
                    "No data files found on disk for any view definitions.",
                }
            ],
        }

    # Build prompt with schema and join info
    try:
        schema_desc = get_schema_description(conn)
    except Exception:
        schema_desc = "(Schema unavailable)"

    join_rels = get_join_relationships()
    available_tables = ", ".join(sorted(registered))

    user_prompt = f"""Given the following database schema and a user question, generate a single
DuckDB-compatible SQL query.

AVAILABLE TABLES: {available_tables}
Only use tables listed above. Do NOT reference any other tables.

DATABASE SCHEMA:
{schema_desc}

{join_rels}

{_DUCKDB_SYNTAX_GUIDE}

{_FEW_SHOT_EXAMPLES}

USER QUESTION: {query}
SUB-TASKS: {sub_tasks_str}

Generate ONLY the SQL query. No explanation."""

    # Text-to-SQL LLM call
    validate_model_parameters(
        model=DATA_ANALYST_MODEL,
        temperature=WORKER_TEMPERATURE,
        max_completion_tokens=DATA_ANALYST_SQL_GENERATION_MAX_TOKENS,
    )
    llm = make_chat_model(
        "data_analyst",
        temperature=WORKER_TEMPERATURE,
        max_completion_tokens=DATA_ANALYST_SQL_GENERATION_MAX_TOKENS,
    )
    t0 = time.time()
    response = llm.invoke(
        [
            SystemMessage(content=_TEXT_TO_SQL_SYSTEM),
            HumanMessage(content=user_prompt),
        ]
    )
    latency = time.time() - t0

    llm_calls = record_llm_call(
        response=response,
        node_name="DataAnalystWorker",
        call_purpose="text_to_sql",
        model_config=DATA_ANALYST_MODEL,
        temperature=WORKER_TEMPERATURE,
        system_prompt=_TEXT_TO_SQL_SYSTEM,
        user_prompt=user_prompt,
        latency_s=latency,
        existing_calls=[],
    )

    # Extract and execute SQL
    generated_sql = _extract_sql_from_response(response.content)
    result_text, metadata = execute_analytical_query(conn, generated_sql)

    # capture the first-attempt outcome before any self-correction so retry
    # reporting can separate initial SQL success from post-recovery success. The
    # headline "SQL success" figure conflated a first-try slip that was cleanly
    # recovered with a genuine unrecoverable failure.
    first_attempt_sql = generated_sql
    first_attempt_error = "error" in metadata
    first_attempt_error_msg = metadata.get("error", "") if first_attempt_error else ""
    retry_triggered = False
    retry_reasons: list[str] = []

    # Self-correction path 1 — the first SQL raised a DuckDB execution error.
    # Previously verification (and therefore any retry) only ran when the first
    # attempt executed cleanly, so a recoverable syntax slip counted as a hard
    # failure. Feed the error back and let the model correct itself once.
    if first_attempt_error:
        retry_triggered = True
        retry_reasons.append("execution_error")
        error_correction_prompt = f"""{user_prompt}

PREVIOUS ATTEMPT FAILED TO EXECUTE:
Previous SQL: {generated_sql}
DuckDB error: {first_attempt_error_msg}

Generate a corrected DuckDB-compatible SQL query that executes successfully."""

        t0 = time.time()
        error_retry_response = llm.invoke(
            [
                SystemMessage(content=_TEXT_TO_SQL_SYSTEM),
                HumanMessage(content=error_correction_prompt),
            ]
        )
        error_retry_latency = time.time() - t0

        llm_calls = record_llm_call(
            response=error_retry_response,
            node_name="DataAnalystWorker",
            call_purpose="text_to_sql_error_retry",
            model_config=DATA_ANALYST_MODEL,
            temperature=WORKER_TEMPERATURE,
            system_prompt=_TEXT_TO_SQL_SYSTEM,
            user_prompt=error_correction_prompt,
            latency_s=error_retry_latency,
            existing_calls=llm_calls,
        )

        generated_sql = _extract_sql_from_response(error_retry_response.content)
        result_text, metadata = execute_analytical_query(conn, generated_sql)

    # Verification step: check if result answers the question
    if "error" not in metadata:
        verification_prompt = f"""USER QUESTION: {query}
SQL EXECUTED: {generated_sql}
QUERY RESULTS:
{result_text}

Does the SQL output correctly answer the user's question?"""

        t0 = time.time()
        validate_model_parameters(
            model=DATA_ANALYST_MODEL,
            temperature=WORKER_TEMPERATURE,
            max_completion_tokens=DATA_ANALYST_SQL_VERIFICATION_MAX_TOKENS,
        )
        verify_llm = make_chat_model(
            "data_analyst",
            temperature=WORKER_TEMPERATURE,
            max_completion_tokens=DATA_ANALYST_SQL_VERIFICATION_MAX_TOKENS,
        )
        verify_response = verify_llm.invoke(
            [
                SystemMessage(content=_VERIFICATION_SYSTEM),
                HumanMessage(content=verification_prompt),
            ]
        )
        verify_latency = time.time() - t0

        llm_calls = record_llm_call(
            response=verify_response,
            node_name="DataAnalystWorker",
            call_purpose="result_verification",
            model_config=DATA_ANALYST_MODEL,
            temperature=WORKER_TEMPERATURE,
            system_prompt=_VERIFICATION_SYSTEM,
            user_prompt=verification_prompt,
            latency_s=verify_latency,
            existing_calls=llm_calls,
        )

        # If mismatch detected, retry once with correction feedback
        if "MISMATCH" in verify_response.content.upper():
            retry_triggered = True
            retry_reasons.append("verification_mismatch")
            correction_prompt = f"""{user_prompt}

PREVIOUS ATTEMPT FAILED VERIFICATION:
Previous SQL: {generated_sql}
Verification feedback: {verify_response.content}

Generate a corrected SQL query that properly answers the question."""

            t0 = time.time()
            retry_response = llm.invoke(
                [
                    SystemMessage(content=_TEXT_TO_SQL_SYSTEM),
                    HumanMessage(content=correction_prompt),
                ]
            )
            retry_latency = time.time() - t0

            llm_calls = record_llm_call(
                response=retry_response,
                node_name="DataAnalystWorker",
                call_purpose="text_to_sql_retry",
                model_config=DATA_ANALYST_MODEL,
                temperature=WORKER_TEMPERATURE,
                system_prompt=_TEXT_TO_SQL_SYSTEM,
                user_prompt=correction_prompt,
                latency_s=retry_latency,
                existing_calls=llm_calls,
            )

            generated_sql = _extract_sql_from_response(retry_response.content)
            result_text, metadata = execute_analytical_query(conn, generated_sql)

    # Handle empty results
    if metadata.get("row_count", 0) == 0 and "error" not in metadata:
        result_text = f"No matching records found for the query: {generated_sql}"

    # Assemble context — return only new data
    context = {"analytical_results": [result_text]}

    new_citations = [
        {
            "collection": "analytical",
            "source": "DuckDB query",
            "location": metadata.get("sql_executed", generated_sql),
            "chunk_id": "n/a",
        }
    ]

    retrieval_details = [
        {
            "collection": "analytical",
            "node_name": "DataAnalystWorker",
            "sql_executed": metadata.get("sql_executed", generated_sql),
            "row_count": metadata.get("row_count", 0),
            "columns": metadata.get("columns", []),
            # Structured result rows, so `sql_correctness` can score the 60%
            # cell-value component. Absent from every run before this, which
            # capped that metric at 0.40 by construction.
            "sample_values": metadata.get("sample_values", []),
            # self-correction instrumentation. `first_attempt_error` marks a
            # first-try DuckDB failure; `recovered` is True when a retry produced
            # a clean final result, so downstream metrics can report a first-pass
            # success rate and a post-self-correction success rate separately.
            "sql_first_attempt": first_attempt_sql,
            "first_attempt_error": first_attempt_error,
            "first_attempt_error_msg": first_attempt_error_msg,
            "retry_triggered": retry_triggered,
            "retry_reason": "+".join(retry_reasons),
            "recovered": retry_triggered and "error" not in metadata,
            "final_error": "error" in metadata,
        }
    ]

    return {
        "context_gathered": context,
        "citations": new_citations,
        "llm_calls": llm_calls,
        "retrieval_details": retrieval_details,
    }
