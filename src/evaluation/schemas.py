"""
Unified evaluation entry schema and validation.

Defines the canonical shape for every evaluation dataset entry and provides
validation helpers used by generators, the dataset builder, and the
load/save utilities.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, TypedDict

# ---------------------------------------------------------------------------
# Allowed enumeration values
# ---------------------------------------------------------------------------

VALID_SOURCES = frozenset(
    {
        "auto_regulatory",
        "auto_incident",
        "auto_enforcement",
        "auto_violation",
        "auto_genduty",
        "auto_statistical",
        "external_osha",
        "external_csb",
        "adversarial",
        "synthetic",
    }
)

VALID_CATEGORIES = frozenset(
    {
        "regulatory_factual",
        "incident_factual",
        "enforcement_factual",
        "enforcement_violation",
        "enforcement_genduty",
        "enforcement_analytical",
        "cross_corpus",
        "adversarial",
        "synthetic_simple",
        "synthetic_reasoning",
        "synthetic_multi_context",
        "statistical_factual",
        "statistical_comparative",
        "statistical_trend",
        "hybrid_analytical",
    }
)

VALID_COMPLEXITIES = frozenset({"simple", "reasoning", "multi_context"})

VALID_AGENTS = frozenset(
    {
        "osha_regs",
        "osha_enf",
        "csb_incident",
        "csb_recs",  # canonical agent name (pipeline + eval dataset)
        "csb_recommendations",  # legacy alias — kept for backward compat with generators
        "root_cause",
        "data_analyst",
    }
)


# ---------------------------------------------------------------------------
# TypedDict – for type-checker support and documentation
# ---------------------------------------------------------------------------


class KeyClaim(TypedDict, total=False):
    claim: str
    source_chunk: str


class EvalEntry(TypedDict, total=False):
    """Canonical evaluation dataset entry.

    Required keys: ``query``, ``expected_answer``, ``ground_truth_chunks``,
    ``source``, ``category``.
    """

    # Required
    query: str
    expected_answer: str
    ground_truth_chunks: list[str]
    source: str
    category: str

    # Optional
    key_claims: list[KeyClaim]
    expected_agents: list[str]
    complexity: str
    metadata: dict[str, Any]

    # Track F: Statistical Aggregation (optional)
    expected_sql: str
    ground_truth_result: dict

    # Metric applicability flags
    l1_applicable: bool  # False => excluded from Layer-1 retrieval metrics (zero-GT)
    expected_behavior: str  # e.g. "refuse" for adversarial / no-retrieval entries


# ---------------------------------------------------------------------------
# Entry-level validation
# ---------------------------------------------------------------------------

_REQUIRED_KEYS = {
    "query",
    "expected_answer",
    "ground_truth_chunks",
    "source",
    "category",
}


def validate_eval_entry(entry: dict[str, Any]) -> list[str]:
    """Return a list of validation error strings for *entry*.

    An empty list means the entry is valid.
    """
    errors: list[str] = []

    # --- required keys ---
    missing = _REQUIRED_KEYS - set(entry.keys())
    if missing:
        errors.append(f"Missing required key(s): {sorted(missing)}")
        return errors  # remaining checks would be meaningless

    # --- type checks on required fields ---
    if not isinstance(entry["query"], str) or not entry["query"].strip():
        errors.append("'query' must be a non-empty string")

    if (
        not isinstance(entry["expected_answer"], str)
        or not entry["expected_answer"].strip()
    ):
        errors.append("'expected_answer' must be a non-empty string")

    if not isinstance(entry["ground_truth_chunks"], list):
        errors.append("'ground_truth_chunks' must be a list")
    elif not all(isinstance(c, str) for c in entry["ground_truth_chunks"]):
        errors.append("Every item in 'ground_truth_chunks' must be a string")

    if not isinstance(entry["source"], str):
        errors.append("'source' must be a string")
    elif entry["source"] not in VALID_SOURCES:
        errors.append(
            f"'source' must be one of {sorted(VALID_SOURCES)}, got '{entry['source']}'"
        )

    if not isinstance(entry["category"], str):
        errors.append("'category' must be a string")
    elif entry["category"] not in VALID_CATEGORIES:
        errors.append(
            f"'category' must be one of {sorted(VALID_CATEGORIES)}, got '{entry['category']}'"
        )

    # --- optional field checks ---
    if "key_claims" in entry:
        if not isinstance(entry["key_claims"], list):
            errors.append("'key_claims' must be a list")
        else:
            for i, claim in enumerate(entry["key_claims"]):
                if not isinstance(claim, dict):
                    errors.append(f"key_claims[{i}] must be a dict")
                elif "claim" not in claim:
                    errors.append(f"key_claims[{i}] missing 'claim' key")

    if "expected_agents" in entry:
        if not isinstance(entry["expected_agents"], list):
            errors.append("'expected_agents' must be a list")
        else:
            for agent in entry["expected_agents"]:
                if agent not in VALID_AGENTS:
                    errors.append(
                        f"Unknown agent '{agent}' in 'expected_agents'; "
                        f"valid agents: {sorted(VALID_AGENTS)}"
                    )

    if "complexity" in entry:
        if entry["complexity"] not in VALID_COMPLEXITIES:
            errors.append(
                f"'complexity' must be one of {sorted(VALID_COMPLEXITIES)}, "
                f"got '{entry['complexity']}'"
            )

    if "metadata" in entry and not isinstance(entry["metadata"], dict):
        errors.append("'metadata' must be a dict")

    if "expected_sql" in entry:
        if (
            not isinstance(entry["expected_sql"], str)
            or not entry["expected_sql"].strip()
        ):
            errors.append("'expected_sql' must be a non-empty string")

    if "ground_truth_result" in entry:
        if not isinstance(entry["ground_truth_result"], dict):
            errors.append("'ground_truth_result' must be a dict")

    if "l1_applicable" in entry and not isinstance(entry["l1_applicable"], bool):
        errors.append("'l1_applicable' must be a bool")

    if "expected_behavior" in entry and not isinstance(entry["expected_behavior"], str):
        errors.append("'expected_behavior' must be a string")

    return errors


# ---------------------------------------------------------------------------
# Dataset-level validation
# ---------------------------------------------------------------------------


def validate_eval_dataset(entries: list[dict[str, Any]]) -> dict[str, Any]:
    """Validate a full evaluation dataset and return a summary report.

    Returns a dict with keys:
        ``total``, ``valid``, ``invalid``, ``errors`` (list of per-entry
        error dicts), ``duplicate_queries`` (list of repeated query strings),
        ``source_counts``, ``category_counts``, ``complexity_counts``.
    """
    entry_errors: list[dict[str, Any]] = []
    valid_count = 0
    queries: list[str] = []
    source_counter: Counter[str] = Counter()
    category_counter: Counter[str] = Counter()
    complexity_counter: Counter[str] = Counter()

    for idx, entry in enumerate(entries):
        errs = validate_eval_entry(entry)
        if errs:
            entry_errors.append({"index": idx, "errors": errs})
        else:
            valid_count += 1
            queries.append(entry["query"])
            source_counter[entry["source"]] += 1
            category_counter[entry["category"]] += 1
            if "complexity" in entry:
                complexity_counter[entry["complexity"]] += 1

    seen: set[str] = set()
    duplicates: list[str] = []
    for q in queries:
        normalized = q.strip().lower()
        if normalized in seen and q not in duplicates:
            duplicates.append(q)
        seen.add(normalized)

    return {
        "total": len(entries),
        "valid": valid_count,
        "invalid": len(entry_errors),
        "errors": entry_errors,
        "duplicate_queries": duplicates,
        "source_counts": dict(source_counter),
        "category_counts": dict(category_counter),
        "complexity_counts": dict(complexity_counter),
    }
