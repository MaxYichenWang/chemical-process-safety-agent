"""
Configuration module for data processing.

This module centralizes all configuration settings for data loading,
embedding generation, and storage, backed by environment variables.
"""

import json as _json
import os
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Project Root
# Assuming this file is in src/data_processing/config.py
# Root is ../../..
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()

# OpenAI API Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Embedding Configuration
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
EMBEDDING_DIMENSIONS = int(os.getenv("EMBEDDING_DIMENSIONS", "1024"))

# Agentic RAG Model Configuration
AGENT_WORKER_MODEL = os.getenv("AGENT_WORKER_MODEL", "gpt-5-mini")
AGENT_REASONING_MODEL = os.getenv("AGENT_REASONING_MODEL", "gpt-5-mini")
AGENT_CRITIC_MODEL = os.getenv("AGENT_CRITIC_MODEL", "gpt-5-mini")
AGENT_SUPERVISOR_MODEL = os.getenv("AGENT_SUPERVISOR_MODEL", "gpt-5-mini")

# Data Analyst Agent Configuration
AGENT_DATA_ANALYST_MODEL = os.getenv("AGENT_DATA_ANALYST_MODEL", AGENT_WORKER_MODEL)
SQL_MAX_RESULT_ROWS = int(os.getenv("SQL_MAX_RESULT_ROWS", "1000"))
SQL_QUERY_TIMEOUT_S = int(os.getenv("SQL_QUERY_TIMEOUT_S", "30"))

# Chunking Configuration
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))
MAX_TOKENS_PER_CHUNK = int(os.getenv("MAX_TOKENS_PER_CHUNK", "800"))

# Database and Storage Paths
# Resolve relative processing paths against PROJECT_ROOT.
DATA_ROOT = Path(os.getenv("DATA_ROOT", "data")).resolve()

CHROMADB_DIR = (PROJECT_ROOT / os.getenv("CHROMADB_DIR", "./chromadb")).resolve()

EMBEDDINGS_CACHE_DB = (
    PROJECT_ROOT / os.getenv("EMBEDDINGS_CACHE_DB", "data/embeddings_cache.duckdb")
).resolve()

CHUNKS_DIR = (PROJECT_ROOT / os.getenv("CHUNKS_DIR", "data/loaded_chunks")).resolve()

BATCH_FILES_DIR = (
    PROJECT_ROOT / os.getenv("BATCH_FILES_DIR", "data/batch_files")
).resolve()

EMBEDDINGS_OUTPUT_DIR = (
    PROJECT_ROOT / os.getenv("EMBEDDINGS_OUTPUT_DIR", "data/embeddings_output")
).resolve()

# Data Source Paths
CSB_DATA_DIR = (
    PROJECT_ROOT / os.getenv("CSB_DATA_DIR", "data/processed/csb_olmocr")
).resolve()

OSHA_DATA_DIR = (
    PROJECT_ROOT / os.getenv("OSHA_DATA_DIR", "data/processed/osha")
).resolve()

# OSHA Enforcement Sample Chunk Filenames
VIOLATIONS_SAMPLE_CHUNKS = "osha_violations_sample_chunks.parquet"
GENDUTY_SAMPLE_CHUNKS = "osha_genduty_sample_chunks.parquet"

# Batch Processing
OPENAI_BATCH_SIZE = int(os.getenv("OPENAI_BATCH_SIZE", "500"))

# ---------------------------------------------------------------------------
# OSHA standards of interest (shared across data processing and evaluation)
# ---------------------------------------------------------------------------
KEY_OSHA_STANDARDS = [
    "1910.119",
    "1910.147",
    "1910.146",
    "1910.1200",
    "1910.134",
    "1910.120",
    "1910.106",
    "1910.38",
]

# ---------------------------------------------------------------------------
# Batch Embedding Pipeline
# ---------------------------------------------------------------------------
BATCH_MAX_REQUESTS = int(os.getenv("BATCH_MAX_REQUESTS", "5000"))
BATCH_MAX_CONCURRENT = int(os.getenv("BATCH_MAX_CONCURRENT", "3"))
BATCH_POLL_INTERVAL_S = int(os.getenv("BATCH_POLL_INTERVAL_S", "60"))
BATCH_TIMEOUT_MINUTES = int(os.getenv("BATCH_TIMEOUT_MINUTES", "120"))
BATCH_GLOBAL_TIMEOUT_MINUTES = int(os.getenv("BATCH_GLOBAL_TIMEOUT_MINUTES", "2880"))
BATCH_RETRY_WAIT_S = int(os.getenv("BATCH_RETRY_WAIT_S", "120"))
BATCH_RETRY_MAX = int(os.getenv("BATCH_RETRY_MAX", "5"))
BATCH_COMPLETION_WINDOW = os.getenv("BATCH_COMPLETION_WINDOW", "24h")
EMBEDDING_PRICE_PER_1K_TOKENS = float(
    os.getenv("EMBEDDING_PRICE_PER_1K_TOKENS", "0.000065")
)

# ---------------------------------------------------------------------------
# Chunk Validation Thresholds
# ---------------------------------------------------------------------------
CHUNK_TARGET_MIN_TOKENS = int(os.getenv("CHUNK_TARGET_MIN_TOKENS", "300"))
CHUNK_TARGET_MAX_TOKENS = int(os.getenv("CHUNK_TARGET_MAX_TOKENS", "800"))

# ---------------------------------------------------------------------------
# ChromaDB Indexing
# ---------------------------------------------------------------------------
CHROMADB_BATCH_SIZE = int(os.getenv("CHROMADB_BATCH_SIZE", "1000"))
CHROMADB_DEFAULT_N_RESULTS = int(os.getenv("CHROMADB_DEFAULT_N_RESULTS", "10"))

# ---------------------------------------------------------------------------
# Embeddings Metadata
# ---------------------------------------------------------------------------
EMBEDDINGS_VERSION = os.getenv("EMBEDDINGS_VERSION", "openai.t3l.1024@2025-11-12")
EMBEDDINGS_SCHEMA_VERSION = int(os.getenv("EMBEDDINGS_SCHEMA_VERSION", "1"))

# ---------------------------------------------------------------------------
# Model Pricing (for cost estimation, USD per token)
#
# Comprehensive pricing for all supported LLM providers.
# Prices are per-token (i.e. per-1M-token price / 1,000,000).
# Override any model's pricing via the MODEL_PRICING_JSON env var
# (a JSON dict of {"model_name": {"prompt": X, "completion": Y}, ...}).
#
# Pricing sources (verified 2026-07-27):
#   OpenAI:   https://openai.com/api/pricing/
#   Claude:   https://docs.anthropic.com/en/docs/about-claude/pricing
#   Gemini:   https://ai.google.dev/gemini-api/docs/pricing
#   MiniMax:  https://platform.minimax.io/docs/guides/pricing-paygo
#   DeepSeek: https://api-docs.deepseek.com/quick_start/pricing
#   Kimi:     https://platform.kimi.ai/docs/pricing/chat (+ per-model chat-k3/chat-k26 pages)
#   Qwen:     https://www.alibabacloud.com/help/en/model-studio/models (model ids
#             only, no $ pricing on this page -- Qwen figures below are
#             third-party-sourced; see note in that section)
#   GLM:      https://docs.z.ai (base_url/model id confirmed; pricing is third-party)
# ---------------------------------------------------------------------------
_PER_M = 1_000_000  # constant for readability

# Default pricing dict — all values are USD per token
_DEFAULT_MODEL_PRICING = {
    # --- OpenAI -----------------------------------------------------------
    # gpt-5.6 (Sol/Terra/Luna) reached GA 2026-07-09 and is the current
    # flagship tier, replacing the old flagship/mini/nano naming. Confirmed
    # directly against openai.com/api/pricing (2026-07-27).
    "gpt-5.6-sol": {"prompt": 5.00 / _PER_M, "completion": 30.00 / _PER_M},
    "gpt-5.6-terra": {"prompt": 2.50 / _PER_M, "completion": 15.00 / _PER_M},
    "gpt-5.6-luna": {"prompt": 1.00 / _PER_M, "completion": 6.00 / _PER_M},
    # gpt-5.4 and earlier: prior generation. No longer shown on the current
    # pricing page; kept for historical cost-log compatibility and because
    # gpt-5-mini / gpt-5.4-mini remain llm_factory.py's built-in defaults.
    "gpt-5.4": {"prompt": 2.50 / _PER_M, "completion": 15.00 / _PER_M},
    "gpt-5.4-mini": {"prompt": 0.75 / _PER_M, "completion": 4.50 / _PER_M},
    "gpt-5.3-codex": {"prompt": 1.75 / _PER_M, "completion": 14.00 / _PER_M},
    "gpt-5.2-codex": {"prompt": 1.75 / _PER_M, "completion": 14.00 / _PER_M},
    "gpt-5.2": {"prompt": 1.25 / _PER_M, "completion": 10.00 / _PER_M},
    "gpt-5.1": {"prompt": 1.25 / _PER_M, "completion": 10.00 / _PER_M},
    "gpt-5": {"prompt": 1.25 / _PER_M, "completion": 10.00 / _PER_M},
    "gpt-5-mini": {"prompt": 0.25 / _PER_M, "completion": 2.00 / _PER_M},
    "gpt-5-nano": {"prompt": 0.05 / _PER_M, "completion": 0.40 / _PER_M},
    "gpt-4.1": {"prompt": 2.00 / _PER_M, "completion": 8.00 / _PER_M},
    "gpt-4.1-mini": {"prompt": 0.40 / _PER_M, "completion": 1.60 / _PER_M},
    "gpt-4.1-nano": {"prompt": 0.10 / _PER_M, "completion": 0.40 / _PER_M},
    "gpt-4o": {"prompt": 2.50 / _PER_M, "completion": 10.00 / _PER_M},
    "gpt-4o-mini": {"prompt": 0.15 / _PER_M, "completion": 0.60 / _PER_M},
    "o3": {"prompt": 2.00 / _PER_M, "completion": 8.00 / _PER_M},
    "o4-mini": {"prompt": 1.10 / _PER_M, "completion": 4.40 / _PER_M},
    # --- Anthropic Claude -------------------------------------------------
    # Current model line, confirmed against platform.claude.com/.../pricing
    # (2026-07-27). Claude Sonnet 5 is on introductory pricing ($2/$10)
    # through 2026-08-31; it reverts to $3/$15 on 2026-09-01 -- update then.
    "claude-fable-5": {"prompt": 10.00 / _PER_M, "completion": 50.00 / _PER_M},
    "claude-opus-5": {"prompt": 5.00 / _PER_M, "completion": 25.00 / _PER_M},
    "claude-sonnet-5": {"prompt": 2.00 / _PER_M, "completion": 10.00 / _PER_M},
    "claude-haiku-4-5": {"prompt": 1.00 / _PER_M, "completion": 5.00 / _PER_M},
    # Prior generation, kept for historical cost-log compatibility.
    "claude-opus-4": {"prompt": 5.00 / _PER_M, "completion": 25.00 / _PER_M},
    "claude-sonnet-4": {"prompt": 3.00 / _PER_M, "completion": 15.00 / _PER_M},
    "claude-haiku-4": {"prompt": 1.00 / _PER_M, "completion": 5.00 / _PER_M},
    # --- Google Gemini ----------------------------------------------------
    # Gemini 3.x is the only line shown on ai.google.dev/gemini-api/docs/pricing
    # as of 2026-07-27; 2.5 models are no longer listed there (kept below for
    # historical cost-log compatibility -- treat as likely deprecated).
    "gemini-3.6-flash": {"prompt": 1.50 / _PER_M, "completion": 7.50 / _PER_M},
    "gemini-3.5-flash": {"prompt": 1.50 / _PER_M, "completion": 9.00 / _PER_M},
    "gemini-3.5-flash-lite": {"prompt": 0.30 / _PER_M, "completion": 2.50 / _PER_M},
    "gemini-3.1-pro-preview": {"prompt": 2.00 / _PER_M, "completion": 12.00 / _PER_M},
    "gemini-3.1-flash-lite": {"prompt": 0.25 / _PER_M, "completion": 1.50 / _PER_M},
    "gemini-2.5-pro": {"prompt": 1.25 / _PER_M, "completion": 10.00 / _PER_M},
    "gemini-2.5-flash": {"prompt": 0.30 / _PER_M, "completion": 2.50 / _PER_M},
    "gemini-2.5-flash-lite": {"prompt": 0.10 / _PER_M, "completion": 0.40 / _PER_M},
    # --- MiniMax ----------------------------------------------------------
    # Confirmed against platform.minimax.io/docs/guides/pricing-paygo
    # (2026-07-27). M3 (new flagship) and all M2.x tiers share the same
    # <=512k-input-token rate; m2.1's price is corrected below (was too low).
    "minimax-m3": {"prompt": 0.30 / _PER_M, "completion": 1.20 / _PER_M},
    "minimax-m2.7": {"prompt": 0.30 / _PER_M, "completion": 1.20 / _PER_M},
    "minimax-m2.5": {"prompt": 0.30 / _PER_M, "completion": 1.20 / _PER_M},
    "minimax-m2.1": {"prompt": 0.30 / _PER_M, "completion": 1.20 / _PER_M},
    # --- DeepSeek -----------------------------------------------------------
    # Confirmed against api-docs.deepseek.com/quick_start/pricing (2026-07-27).
    # The deepseek-chat/deepseek-reasoner deprecation (previously scheduled
    # for 2026-07-24) has now taken effect: both aliases route to
    # deepseek-v4-flash (non-thinking / thinking mode respectively). v3.x/r1
    # rows kept below only for historical cost-log compatibility.
    "deepseek-v4-flash": {"prompt": 0.14 / _PER_M, "completion": 0.28 / _PER_M},
    "deepseek-v4-pro": {"prompt": 0.435 / _PER_M, "completion": 0.87 / _PER_M},
    "deepseek-v3.2": {"prompt": 0.28 / _PER_M, "completion": 0.42 / _PER_M},
    "deepseek-v3": {"prompt": 0.28 / _PER_M, "completion": 0.42 / _PER_M},
    "deepseek-r1": {"prompt": 0.55 / _PER_M, "completion": 2.19 / _PER_M},
    # --- Kimi (Moonshot AI) -----------------------------------------------
    # Confirmed against platform.kimi.ai/docs/pricing/chat-k3 and chat-k26
    # (2026-07-27). K3 is now the flagship and K2.6 the current mid-tier;
    # K2.5/K2 no longer appear on the current pricing page but are kept below
    # for historical cost-log compatibility.
    "kimi-k3": {"prompt": 3.00 / _PER_M, "completion": 15.00 / _PER_M},
    "kimi-k2.6": {"prompt": 0.95 / _PER_M, "completion": 4.00 / _PER_M},
    "kimi-k2.5": {"prompt": 0.60 / _PER_M, "completion": 2.50 / _PER_M},
    "kimi-k2": {"prompt": 0.60 / _PER_M, "completion": 2.50 / _PER_M},
    # --- Qwen (Alibaba Cloud) --------------------------------------------
    # alibabacloud.com/help/en/model-studio/models lists current model ids
    # but no $ pricing table for most of these. qwen3.7-max is third-party-
    # sourced (LOW-MEDIUM confidence) -- re-check via the Model Studio console
    # before relying on cost estimates; its rate reflects a stated "50%
    # Confirmed against Alibaba Cloud Model Studio console (2026-08-02).
    # qwen3.7-flash: $0.03/M input, $0.13/M output; 1M context window;
    # OpenAI-compatible base URL: https://ws-48q2hywrewc4xw5p.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1
    "qwen3.7-max": {"prompt": 1.25 / _PER_M, "completion": 3.75 / _PER_M},
    "qwen3.7-flash": {"prompt": 0.03 / _PER_M, "completion": 0.13 / _PER_M},
    "qwen3-max": {"prompt": 1.20 / _PER_M, "completion": 6.00 / _PER_M},
    "qwen-plus": {"prompt": 0.40 / _PER_M, "completion": 1.20 / _PER_M},
    "qwen3-coder": {"prompt": 0.30 / _PER_M, "completion": 1.50 / _PER_M},
    # --- GLM (Zhipu AI / z.ai) --------------------------------------------
    # Pricing from third-party aggregators, not docs.z.ai directly (as of
    # 2026-07-27) -- MEDIUM confidence, re-check before relying on cost
    # estimates for this provider. base_url/model id ARE confirmed directly
    # against https://docs.z.ai/api-reference/introduction (HIGH confidence).
    # glm-5.2 corrected 2026-08-01: the previous $1.40/$4.40 figures
    # overstated the live OpenRouter rate by ~3.3x. $0.42/$1.32 is the
    # verified rate as of a 2026-08-01 judge-selection cost comparison.
    # 5.1/5 kept as-is below (not part of that verification).
    "glm-5.2": {"prompt": 0.42 / _PER_M, "completion": 1.32 / _PER_M},
    "glm-5.1": {"prompt": 0.966 / _PER_M, "completion": 3.04 / _PER_M},
    "glm-5": {"prompt": 0.60 / _PER_M, "completion": 1.92 / _PER_M},
}

# Allow full override via a single JSON env var
_pricing_override = os.getenv("MODEL_PRICING_JSON")
if _pricing_override:
    try:
        _user_pricing = _json.loads(_pricing_override)
        _DEFAULT_MODEL_PRICING.update(_user_pricing)
    except (ValueError, TypeError):
        pass  # silently ignore malformed JSON; use defaults

MODEL_PRICING = _DEFAULT_MODEL_PRICING
