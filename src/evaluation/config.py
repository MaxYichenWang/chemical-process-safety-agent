"""
Configuration module for the evaluation framework.

Centralizes all evaluation-related settings: paths, model parameters,
metric thresholds, and bootstrap configuration. Imports shared model
constants from the data_processing config to stay consistent with the
agentic RAG pipeline.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

from src.data_processing.config import (
    CHROMADB_DIR as CHROMADB_DIR,
    CHUNKS_DIR as LOADED_CHUNKS_DIR,
    CSB_DATA_DIR,
    KEY_OSHA_STANDARDS as KEY_OSHA_STANDARDS,
    OSHA_DATA_DIR,
)

load_dotenv()


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


# ---------------------------------------------------------------------------
# Evaluation outputs resolve from the working directory. Corpus paths above
# share processing configuration, including its absolute environment overrides.
# ---------------------------------------------------------------------------
EVAL_DATASET_DIR = Path(os.getenv("EVAL_DATASET_DIR", "data/evaluation"))
EVAL_RESULTS_DIR = EVAL_DATASET_DIR / "results"
EVAL_LOGS_DIR = EVAL_RESULTS_DIR / "logs"
EVAL_HISTORY_DIR = EVAL_DATASET_DIR / "improvement_history"
PROCESSED_CSB_DIR = CSB_DATA_DIR
PROCESSED_OSHA_DIR = OSHA_DATA_DIR

# ---------------------------------------------------------------------------
# Chroma / LangGraph tool collection names -> EvalEntry ``expected_agents`` names
# (see ``src/agentic_rag/tools.py`` get_chroma_tools and ``schemas.VALID_AGENTS``).
# ---------------------------------------------------------------------------
COLLECTION_TO_AGENT: dict[str, str] = {
    "osha_regulations": "osha_regs",
    "osha_enf": "osha_enf",
    "csb": "csb_incident",
    "csb_recommendations": "csb_recommendations",
    "root_cause_analysis": "root_cause",
    "analytical": "data_analyst",
}

# ---------------------------------------------------------------------------
# Dataset files
# ---------------------------------------------------------------------------
EVAL_DATASET_FILE = Path(
    os.getenv("EVAL_DATASET_FILE", str(EVAL_DATASET_DIR / "eval_dataset_v1.json"))
)
ADVERSARIAL_DATASET_FILE = EVAL_DATASET_DIR / "eval_dataset_adversarial.json"

# ---------------------------------------------------------------------------
# Evaluation LLM (used by RAGAS and DeepEval as judge)
# ---------------------------------------------------------------------------
EVAL_LLM_MODEL = os.getenv("EVAL_LLM_MODEL", "gpt-5.4-mini")
# gpt-5-mini and gpt-5.4-mini only accept temperature=1 (the API default).
# Using 0.0 causes a 400 Bad Request from the OpenAI API.
EVAL_LLM_TEMPERATURE = float(os.getenv("EVAL_LLM_TEMPERATURE", "1.0"))
# Maximum tokens for judge evaluations (defaults to 8192 to allow reasoning models
# like DeepSeek sufficient headroom for thinking tokens + output payload).
EVAL_LLM_MAX_TOKENS = int(os.getenv("EVAL_LLM_MAX_TOKENS", "8192"))

# RAGAS defaults its internal LLM args to temperature=0.01 / top_p=0.1 /
# max_tokens=1024, all three of which GPT-5 family models reject. RAGAS 0.4 has
# its own remapping for "reasoning models" but its version parser only handles
# integer versions, so it misses "gpt-5.4-mini" -- judge_factory therefore
# applies the remapping itself (see _sanitize_reasoning_model_args there).
#
# RAGAS is still off by default on GPT-5 judges: DeepEval already covers
# faithfulness/hallucination and is what gating actually uses, so RAGAS is
# opt-in extra judge spend rather than a required signal. Set
# EVAL_ENABLE_RAGAS=1 to turn it on -- as of 2026-07-31 that path works instead
# of silently scoring every row 0.0.
EVAL_ENABLE_RAGAS = _env_bool(
    "EVAL_ENABLE_RAGAS",
    default=not EVAL_LLM_MODEL.startswith("gpt-5"),
)

# Citation accuracy credits an answer that cites a parent paragraph when the
# retrieved context establishes a child paragraph (context 1910.1001(j)(2)(iii)
# supports an answer citing 1910.1001(j)). Set to 0 for strict exact matching.
EVAL_CITATION_CREDIT_ANCESTORS = _env_bool(
    "EVAL_CITATION_CREDIT_ANCESTORS", default=True
)

# ---------------------------------------------------------------------------
# Metric thresholds – minimum acceptable values unless noted otherwise
# ---------------------------------------------------------------------------
THRESHOLDS = {
    "context_precision": float(os.getenv("EVAL_THRESHOLD_CONTEXT_PRECISION", "0.85")),
    "context_recall": float(os.getenv("EVAL_THRESHOLD_CONTEXT_RECALL", "0.75")),
    "hit_rate_at_5": float(os.getenv("EVAL_THRESHOLD_HIT_RATE_AT_5", "0.80")),
    "mrr": float(os.getenv("EVAL_THRESHOLD_MRR", "0.70")),
    "ndcg_at_5": float(os.getenv("EVAL_THRESHOLD_NDCG_AT_5", "0.75")),
    "faithfulness": float(os.getenv("EVAL_THRESHOLD_FAITHFULNESS", "0.90")),
    "hallucination_rate": float(
        os.getenv("EVAL_THRESHOLD_HALLUCINATION_RATE", "0.05")
    ),  # max acceptable
    "answer_correctness": float(os.getenv("EVAL_THRESHOLD_ANSWER_CORRECTNESS", "0.80")),
    "factual_correctness": float(
        os.getenv("EVAL_THRESHOLD_FACTUAL_CORRECTNESS", "0.85")
    ),
    "citation_accuracy": float(os.getenv("EVAL_THRESHOLD_CITATION_ACCURACY", "0.95")),
    "goal_accuracy": float(os.getenv("EVAL_THRESHOLD_GOAL_ACCURACY", "0.85")),
    "tool_call_accuracy": float(os.getenv("EVAL_THRESHOLD_TOOL_CALL_ACCURACY", "0.90")),
    "topic_adherence": float(os.getenv("EVAL_THRESHOLD_TOPIC_ADHERENCE", "0.95")),
    "critic_score_mean": float(os.getenv("EVAL_THRESHOLD_CRITIC_SCORE_MEAN", "0.80")),
    "retry_rate": float(
        os.getenv("EVAL_THRESHOLD_RETRY_RATE", "0.20")
    ),  # max acceptable
    "sql_execution_success": float(
        os.getenv("EVAL_THRESHOLD_SQL_EXECUTION_SUCCESS", "0.95")
    ),
    "sql_correctness": float(os.getenv("EVAL_THRESHOLD_SQL_CORRECTNESS", "0.85")),
    "sql_correctness_relaxed": float(
        os.getenv("EVAL_THRESHOLD_SQL_CORRECTNESS_RELAXED", "0.90")
    ),
    "numerical_fidelity": float(os.getenv("EVAL_THRESHOLD_NUMERICAL_FIDELITY", "0.98")),
    "hybrid_synthesis_quality": float(
        os.getenv("EVAL_THRESHOLD_HYBRID_SYNTHESIS_QUALITY", "0.80")
    ),
    # Hedging detection gates (Q4 decision: 80% adversarial target, 25% FP ceiling)
    "adversarial_hedge_rate": float(
        os.getenv("EVAL_THRESHOLD_ADVERSARIAL_HEDGE_RATE", "0.80")
    ),  # min acceptable
    "standard_hedge_rate": float(
        os.getenv("EVAL_THRESHOLD_STANDARD_HEDGE_RATE", "0.25")
    ),  # max acceptable (FP)
}

# ---------------------------------------------------------------------------
# Bootstrap settings for confidence intervals
# ---------------------------------------------------------------------------
BOOTSTRAP_ITERATIONS = int(os.getenv("EVAL_BOOTSTRAP_ITERATIONS", "1000"))
CONFIDENCE_LEVEL = float(os.getenv("EVAL_CONFIDENCE_LEVEL", "0.95"))

# ---------------------------------------------------------------------------
# Generator Defaults (shared across QA generators)
# ---------------------------------------------------------------------------
GENERATOR_SEED = int(os.getenv("EVAL_GENERATOR_SEED", "42"))
GENERATOR_MIN_TOKEN_COUNT = int(os.getenv("EVAL_GENERATOR_MIN_TOKEN_COUNT", "50"))
GENERATOR_CHECKPOINT_EVERY = int(os.getenv("EVAL_GENERATOR_CHECKPOINT_EVERY", "10"))

# ---------------------------------------------------------------------------
# RAGAS / DeepEval Runtime
# ---------------------------------------------------------------------------
EVAL_MAX_WORKERS = int(os.getenv("EVAL_MAX_WORKERS", "4"))
EVAL_WORKER_TIMEOUT = int(os.getenv("EVAL_WORKER_TIMEOUT", "120"))
EVAL_MAX_CONCURRENT = int(os.getenv("EVAL_MAX_CONCURRENT", "5"))

# Process-wide ceiling on concurrent judge (LLM-as-judge) calls, shared
# by every evaluator/thread/event-loop -- see judge_concurrency.py. Bounds the
# *product* of evaluator-level, DeepEval-worker-level, and intra-entry-metric
# concurrency, which each look bounded in isolation but stack multiplicatively.
EVAL_JUDGE_MAX_CONCURRENT = int(os.getenv("EVAL_JUDGE_MAX_CONCURRENT", "16"))

# ---------------------------------------------------------------------------
# Retrieval Evaluation
# ---------------------------------------------------------------------------
EVAL_RETRIEVAL_K = int(os.getenv("EVAL_RETRIEVAL_K", "5"))

# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------
EVAL_DEDUP_SIMILARITY_THRESHOLD = float(
    os.getenv("EVAL_DEDUP_SIMILARITY_THRESHOLD", "0.95")
)
EVAL_DEDUP_EMBEDDING_MODEL = os.getenv(
    "EVAL_DEDUP_EMBEDDING_MODEL", "text-embedding-3-small"
)

# ---------------------------------------------------------------------------
# Loaded chunk parquet paths (all generators read exclusively from here)
# ---------------------------------------------------------------------------
REGULATIONS_CHUNKS_PARQUET = LOADED_CHUNKS_DIR / "osha_regulations_chunks.parquet"
ACCIDENTS_CHUNKS_PARQUET = LOADED_CHUNKS_DIR / "osha_accidents_chunks.parquet"
VIOLATIONS_SAMPLE_CHUNKS_PARQUET = (
    LOADED_CHUNKS_DIR / "osha_violations_sample_chunks.parquet"
)
GENDUTY_SAMPLE_CHUNKS_PARQUET = LOADED_CHUNKS_DIR / "osha_genduty_sample_chunks.parquet"
CSB_REPORTS_CHUNKS_PARQUET = LOADED_CHUNKS_DIR / "csb_final_reports_chunks.parquet"
CSB_RECS_CHUNKS_PARQUET = LOADED_CHUNKS_DIR / "csb_recommendations_chunks.parquet"
INSPECTION_PARQUET = LOADED_CHUNKS_DIR / "inspection.parquet"
