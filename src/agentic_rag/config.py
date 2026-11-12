"""
Centralized configuration for all agentic RAG LLM calls.

Adjust these values to balance response quality against cost/latency.
Higher limits produce more complete outputs but increase token spend.
All values can be overridden via environment variables.
"""

import os

from dotenv import load_dotenv

load_dotenv()

# --- Supervisor ---
SUPERVISOR_PLANNING_MAX_TOKENS = int(
    os.getenv("SUPERVISOR_PLANNING_MAX_TOKENS", "99999")
)
SUPERVISOR_SYNTHESIS_MAX_TOKENS = int(
    os.getenv("SUPERVISOR_SYNTHESIS_MAX_TOKENS", "2500")
)
SUPERVISOR_REVISION_MAX_TOKENS = int(
    os.getenv("SUPERVISOR_REVISION_MAX_TOKENS", "2500")
)

# --- Workers ---
WORKER_QUERY_REFORMULATION_MAX_TOKENS = int(
    os.getenv("WORKER_QUERY_REFORMULATION_MAX_TOKENS", "99999")
)
REASONING_MAX_TOKENS = int(os.getenv("REASONING_MAX_TOKENS", "3000"))

# --- Data Analyst (text-to-SQL) ---
DATA_ANALYST_SQL_GENERATION_MAX_TOKENS = int(
    os.getenv("DATA_ANALYST_SQL_GENERATION_MAX_TOKENS", "99999")
)
DATA_ANALYST_SQL_VERIFICATION_MAX_TOKENS = int(
    os.getenv("DATA_ANALYST_SQL_VERIFICATION_MAX_TOKENS", "99999")
)

# --- Critic ---
CRITIC_EVALUATION_MAX_TOKENS = int(os.getenv("CRITIC_EVALUATION_MAX_TOKENS", "400"))
CRITIC_QUANTITATIVE_MAX_TOKENS = int(os.getenv("CRITIC_QUANTITATIVE_MAX_TOKENS", "300"))

# --- Graph Control ---
MAX_CRITIC_RETRIES = int(os.getenv("MAX_CRITIC_RETRIES", "2"))
VALIDATION_SCORE_THRESHOLD = float(os.getenv("VALIDATION_SCORE_THRESHOLD", "0.8"))

# --- Premise verification (fictional-entity / unverifiable-premise hedging) ---
# When enabled, the synthesizer detects specific entities in the query (incident
# names, citation/activity/accident IDs, named facilities) that retrieval failed to
# corroborate, and forces premise-verification hedging instead of an assertive
# answer. Grounding-gated to keep standard-query false positives low. Toggle exists
# so the model/feature sweep can A/B it against the hedging-FP gate (≤30%).
SUPERVISOR_PREMISE_CHECK_ENABLED = os.getenv(
    "SUPERVISOR_PREMISE_CHECK_ENABLED", "1"
).strip().lower() in {"1", "true", "yes", "on"}

# --- Ablation flags ---
# Both default to enabled (today's behavior). Set to 0 to run the
# corresponding no_critic / no_query_reformulation ablation generation --
# each flag is read once per node call, so it can be flipped per-process via
# env var without touching graph topology.
CRITIC_ENABLED = os.getenv("CRITIC_ENABLED", "1").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
WORKER_QUERY_REFORMULATION_ENABLED = os.getenv(
    "WORKER_QUERY_REFORMULATION_ENABLED", "1"
).strip().lower() in {"1", "true", "yes", "on"}

# --- Retrieval Tuning ---
HYBRID_VECTOR_TOP_K = int(os.getenv("HYBRID_VECTOR_TOP_K", "10"))
HYBRID_KEYWORD_TOP_K = int(os.getenv("HYBRID_KEYWORD_TOP_K", "5"))
RRF_K = int(os.getenv("RRF_K", "60"))
HYBRID_RERANK_ENABLED = os.getenv("HYBRID_RERANK_ENABLED", "1").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
HYBRID_RERANK_MODEL = os.getenv("HYBRID_RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
HYBRID_RERANK_TOP_N = int(os.getenv("HYBRID_RERANK_TOP_N", "10"))
HYBRID_VECTOR_TOP_K_BY_COLLECTION = {
    "osha_regulations": int(
        os.getenv("HYBRID_VECTOR_TOP_K_OSHA_REGULATIONS", str(HYBRID_VECTOR_TOP_K))
    ),
    "osha_enf": int(os.getenv("HYBRID_VECTOR_TOP_K_OSHA_ENF", "25")),
    "csb": int(os.getenv("HYBRID_VECTOR_TOP_K_CSB", str(HYBRID_VECTOR_TOP_K))),
    "csb_recommendations": int(
        os.getenv("HYBRID_VECTOR_TOP_K_CSB_RECOMMENDATIONS", "10")
    ),
}


# --- LLM Temperature Defaults ---
# These are *optional* overrides now, not blanket 1.0s. When unset
# (None), llm_factory.resolve_role_config() picks a provider-aware default:
# 1.0 for GPT-5-family models (gpt-5-mini and siblings only accept
# temperature=1 -- setting 0.0 causes a 400 Bad Request), 0.0 for everything
# else (DeepSeek and most other providers accept 0.0, and a reproducible eval
# pipeline wants it). Set one of these env vars only to pin a role's
# temperature regardless of which model it resolves to.
def _optional_temperature_env(name: str) -> float | None:
    raw = os.getenv(name)
    return float(raw) if raw not in (None, "") else None


SUPERVISOR_TEMPERATURE = _optional_temperature_env("SUPERVISOR_TEMPERATURE")
CRITIC_TEMPERATURE = _optional_temperature_env("CRITIC_TEMPERATURE")
WORKER_TEMPERATURE = _optional_temperature_env("WORKER_TEMPERATURE")
REASONING_TEMPERATURE = _optional_temperature_env("REASONING_TEMPERATURE")
