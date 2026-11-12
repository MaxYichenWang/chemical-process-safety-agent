"""
Shared utilities for the evaluation framework.

Provides dataset I/O, statistical helpers, and text extraction functions
used across all evaluator modules.
"""

import json
import logging
import random
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from src.evaluation.config import COLLECTION_TO_AGENT, EVAL_LOGS_DIR, EVAL_RESULTS_DIR

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Failure isolation
# ---------------------------------------------------------------------------


def finalize_pipeline_results(
    dataset: List[Dict[str, Any]],
    results: List[Dict[str, Any]],
    *,
    stopped_early_error: Optional[str] = None,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Preserve dataset row positions and describe operational completion.

    A stopped batch returns a prefix. Explicit placeholders retain the remaining
    questions without treating them as completed model calls.
    """
    if len(results) > len(dataset):
        raise ValueError("Pipeline returned more results than dataset entries")
    processed = len(results)
    failed = sum("error" in result for result in results)
    unprocessed = len(dataset) - processed
    reasons = []
    if not dataset:
        reasons.append("Dataset is empty")
    if unprocessed:
        reasons.append(f"Pipeline stopped with {unprocessed} unprocessed entries")
    if stopped_early_error:
        reasons.append(str(stopped_early_error))
    if failed:
        reasons.append(f"{failed}/{processed} processed pipeline entries failed")
    if unprocessed or stopped_early_error:
        status = "incomplete"
    elif not dataset or failed == processed:
        status = "failed"
    elif failed:
        status = "complete_with_errors"
    else:
        status = "complete"
    completion = {
        "status": status,
        "expected_entries": len(dataset),
        "processed_entries": processed,
        "successful_entries": processed - failed,
        "failed_entries": failed,
        "unprocessed_entries": unprocessed,
    }
    if reasons:
        completion["error"] = "; ".join(reasons)
    aligned = list(results) + [
        {
            "query": entry.get("query", ""),
            "error": "Not processed: pipeline stopped before this entry",
            "not_processed": True,
            "final_response": "",
            "context_gathered": [],
            "agents_called": [],
            "llm_calls": [],
        }
        for entry in dataset[processed:]
    ]
    return aligned, completion


def metric_failure_paths(metrics: Dict[str, Any]) -> List[str]:
    """Find operational failures without failing on disabled/inapplicable metrics."""
    failures = []

    def walk(node: Any, path: str) -> None:
        if not isinstance(node, dict):
            return
        if (
            "skipped" in node
            or node.get("status") == "no_applicable_entries"
            or node.get("error") == "no applicable entries (n=0)"
            or node.get("n_rows") == 0
        ):
            return
        for key, value in node.items():
            child = f"{path}.{key}" if path else key
            if (
                (key == "error" or key.endswith("_error"))
                and isinstance(value, str) and value
            ) or (
                (key == "n_failed" or key.endswith("_n_failed"))
                and isinstance(value, (int, float)) and value > 0
            ):
                failures.append(child)
            elif isinstance(value, dict):
                walk(value, child)

    walk(metrics, "")
    return failures


def guarded(name: str, fn, *args, **kwargs) -> Any:
    """Call ``fn`` and turn any exception into ``{"error": ...}``.

    Evaluator ``evaluate()`` methods assemble a dict of independent sub-metrics.
    Without this, one sub-metric raising loses every *sibling* metric in the
    same section: on 2026-07-31 a ragas constructor TypeError took out the
    entire orchestration block -- pipeline error rate, routing accuracy, cost,
    retries -- and the report rendered all of them as a confident ``0.0%``.
    Every sub-metric is therefore isolated at its own boundary.
    """
    try:
        return fn(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001 - isolation boundary, see docstring
        logger.error("metric %s failed: %s", name, exc, exc_info=True)
        return {"error": f"{type(exc).__name__}: {exc}"}


# ---------------------------------------------------------------------------
# Dataset I/O
# ---------------------------------------------------------------------------

_REQUIRED_DATASET_KEYS = {"query", "expected_answer", "ground_truth_chunks"}


def load_eval_dataset(
    path: Path,
    *,
    validate: bool = False,
) -> list[dict[str, Any]]:
    """Load and validate a JSON evaluation dataset.

    Args:
        path: Path to a JSON file containing a list of evaluation entries.
        validate: If ``True``, run full schema validation via
            :func:`~src.evaluation.schemas.validate_eval_entry` and raise on
            any invalid entries.  When ``False`` (default), only the minimal
            required-key check is performed for backward compatibility.

    Returns:
        List of validated evaluation entry dicts.

    Raises:
        FileNotFoundError: If *path* does not exist.
        json.JSONDecodeError: If the file is not valid JSON.
        ValueError: If the top-level structure is not a list, if any entry
            is missing required keys, or (when *validate=True*) if any entry
            fails full schema validation.
    """
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    if not isinstance(data, list):
        raise ValueError(
            f"Expected a JSON array at top level, got {type(data).__name__}"
        )

    if validate:
        from src.evaluation.schemas import validate_eval_entry

        all_errors: list[str] = []
        for idx, entry in enumerate(data):
            errs = validate_eval_entry(entry)
            if errs:
                all_errors.append(f"Entry {idx}: {'; '.join(errs)}")
        if all_errors:
            raise ValueError("Schema validation failed:\n" + "\n".join(all_errors))
    else:
        for idx, entry in enumerate(data):
            missing = _REQUIRED_DATASET_KEYS - set(entry.keys())
            if missing:
                raise ValueError(
                    f"Entry {idx} is missing required key(s): {sorted(missing)}"
                )

    return data


def save_eval_results(results: dict[str, Any], run_id: str) -> Path:
    """Persist evaluation results as a timestamped JSON file.

    Args:
        results: Arbitrary results dict (must be JSON-serializable).
        run_id: Identifier for this evaluation run (used in the filename).

    Returns:
        Path to the written JSON file.
    """
    EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = EVAL_RESULTS_DIR / f"{run_id}_results.json"
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, default=str)
    return output_path


def sample_dataset_per_source(
    dataset: List[Dict[str, Any]],
    sample_size_per_source: int,
    seed: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Stratified-by-source sample, backfilled so every category keeps >=1 entry.

    Extracted from ``EndToEndEvaluator.evaluate()``'s inline sampling block
    (unchanged there for compatibility) so ``run.py generate`` can reuse
    the exact same logic instead of a second copy that could silently drift.
    """
    from src.evaluation.config import GENERATOR_SEED

    full_dataset = list(dataset)
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for entry in dataset:
        grouped.setdefault(entry.get("source", "unknown"), []).append(entry)

    rng = random.Random(seed if seed is not None else GENERATOR_SEED)

    sampled: List[Dict[str, Any]] = []
    for _source, items in grouped.items():
        if len(items) > sample_size_per_source:
            sampled.extend(rng.sample(items, sample_size_per_source))
        else:
            sampled.extend(items)

    required_categories = {entry.get("category", "unknown") for entry in full_dataset}
    sampled_categories = {entry.get("category", "unknown") for entry in sampled}
    missing_categories = sorted(required_categories - sampled_categories)
    for category in missing_categories:
        candidates = [
            entry
            for entry in full_dataset
            if entry.get("category", "unknown") == category and entry not in sampled
        ]
        if candidates:
            sampled.append(rng.choice(candidates))

    return sampled


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------


def bootstrap_ci(
    scores: list[float],
    n_iterations: int = 1000,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Compute a bootstrap confidence interval for the mean of *scores*.

    Args:
        scores: Observed metric values.
        n_iterations: Number of bootstrap resamples.
        confidence: Confidence level (e.g. 0.95 for a 95 % CI).

    Returns:
        ``(lower_bound, upper_bound)`` of the confidence interval.

    Raises:
        ValueError: If *scores* is empty.
    """
    if not scores:
        raise ValueError("Cannot compute bootstrap CI on an empty list")

    rng = np.random.default_rng(seed=42)
    arr = np.asarray(scores, dtype=np.float64)
    bootstrap_means = np.empty(n_iterations, dtype=np.float64)

    for i in range(n_iterations):
        sample = rng.choice(arr, size=len(arr), replace=True)
        bootstrap_means[i] = sample.mean()

    alpha = 1.0 - confidence
    lower = float(np.percentile(bootstrap_means, 100 * alpha / 2))
    upper = float(np.percentile(bootstrap_means, 100 * (1 - alpha / 2)))
    return (lower, upper)


# ---------------------------------------------------------------------------
# Text extraction helpers
# ---------------------------------------------------------------------------

_OSHA_STANDARD_RE = re.compile(r"\b((?:29\s*CFR\s*)?1910\.\d+(?:\([a-zA-Z0-9]+\))*)")


def extract_osha_standards(text: str) -> list[str]:
    """Extract OSHA standard references from *text*.

    Recognises patterns such as ``1910.119``, ``1910.119(j)``,
    ``1910.119(j)(4)(i)``, and optionally prefixed with ``29 CFR``.
    Duplicates are removed; original order is preserved.

    Args:
        text: Arbitrary text that may mention OSHA standards.

    Returns:
        De-duplicated list of matched standard strings.
    """
    matches = _OSHA_STANDARD_RE.findall(text)
    seen: set[str] = set()
    result: list[str] = []
    for m in matches:
        normalized = re.sub(r"\s+", " ", m).strip()
        if normalized not in seen:
            seen.add(normalized)
            result.append(normalized)
    return result


_CHEMICAL_KEYWORDS = [
    "ammonium nitrate",
    "chlorine",
    "hydrogen fluoride",
    "methyl isocyanate",
    "sulfuric acid",
    "hydrochloric acid",
    "ammonia",
    "propane",
    "butane",
    "ethylene oxide",
    "vinyl chloride",
    "phosgene",
    "hydrogen sulfide",
    "methane",
    "nitrogen dioxide",
    "carbon monoxide",
    "formaldehyde",
    "benzene",
    "toluene",
    "xylene",
    "sodium hydroxide",
    "nitric acid",
    "phosphoric acid",
    "acetylene",
    "ethanol",
    "methanol",
    "natural gas",
    "crude oil",
    "diesel fuel",
    "gasoline",
]

# Facility keywords are restricted to distinctive, named-facility-type descriptors.
# Generic English common-nouns ("plant", "facility", "lab", "station", "works",
# "complex", "mill", "terminal") were removed: they matched
# boilerplate prose in both responses and context, so incident_fidelity flipped
# purely with context size rather than with grounding. See domain_evaluator
# incident_fidelity for the scoring implementation.
_FACILITY_KEYWORDS = [
    "refinery",
    "warehouse",
    "factory",
    "laboratory",
]


def extract_entity_names(text: str, entity_type: str) -> list[str]:
    """Extract entity names from *text* by keyword matching.

    Args:
        text: Source text to search.
        entity_type: ``"chemical"`` or ``"facility"``.

    Returns:
        De-duplicated list of matched entity strings (lower-cased).

    Raises:
        ValueError: If *entity_type* is not ``"chemical"`` or ``"facility"``.
    """
    entity_type = entity_type.lower()
    if entity_type == "chemical":
        keywords = _CHEMICAL_KEYWORDS
    elif entity_type == "facility":
        keywords = _FACILITY_KEYWORDS
    else:
        raise ValueError(
            f"entity_type must be 'chemical' or 'facility', got '{entity_type}'"
        )

    text_lower = text.lower()
    found: list[str] = []
    for kw in keywords:
        # Word-boundary match: naive substring matching let short
        # keywords match inside unrelated words and inflated false positives. Boundary
        # matching keeps entity detection token-precise.
        if kw not in found and re.search(r"\b" + re.escape(kw) + r"\b", text_lower):
            found.append(kw)
    return found


# ---------------------------------------------------------------------------
# LangGraph state normalization for evaluators
# ---------------------------------------------------------------------------


def _parse_retrieval_text_blocks(result_text: str) -> list[str]:
    """Split a hybrid tool formatted string into per-document text bodies."""
    if not result_text or "no relevant documents found" in result_text.lower():
        return []
    bodies: list[str] = []
    for block in result_text.split("\n\n---\n\n"):
        block = block.strip()
        if not block:
            continue
        lines = block.split("\n", 1)
        body = lines[1].strip() if len(lines) > 1 else ""
        bodies.append(body)
    return bodies


def normalize_graph_state_for_eval(
    final_state: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Normalize LangGraph ``AgentState`` into evaluator-friendly structures.

    Workers store ``context_gathered`` as ``dict[tool_name, list[formatted_str]]``.
    Evaluators expect a flat ``list[dict]`` with ``chunk_id``, ``text``, ``corpus``, etc.

    Pairs structured ``citations`` (chunk_id, collection, …) with parsed retrieval text
    blocks in traversal order. Root-cause LLM narratives are appended as separate docs.

    If ``context_gathered`` is already a ``list`` (tests / legacy), it is passed through.
    """
    cg = final_state.get("context_gathered")
    if cg is None:
        return [], []
    if isinstance(cg, list):
        agents_called: list[str] = []
        for doc in cg:
            if not isinstance(doc, dict):
                continue
            c = doc.get("corpus", "unknown")
            if c not in agents_called:
                agents_called.append(c)
        return cg, agents_called
    if not isinstance(cg, dict):
        return [], []

    citations: list[dict[str, Any]] = list(final_state.get("citations") or [])

    retrieval_texts: list[str] = []
    for tool_name, texts in cg.items():
        if tool_name in ("root_cause_analysis", "analytical_results"):
            continue
        if not isinstance(texts, list):
            continue
        for result_text in texts:
            if not isinstance(result_text, str):
                continue
            retrieval_texts.extend(_parse_retrieval_text_blocks(result_text))

    # When lengths differ (e.g. root-cause adds citations without matching formatted
    # blocks in traversal order), we still attach chunk_ids; missing bodies get "".

    docs: list[dict[str, Any]] = []
    text_idx = 0
    for cit in citations:
        if not isinstance(cit, dict):
            continue
        coll = str(cit.get("collection") or "")
        if coll == "analytical":
            continue
        text = retrieval_texts[text_idx] if text_idx < len(retrieval_texts) else ""
        text_idx += 1
        agent = COLLECTION_TO_AGENT.get(coll, coll)
        docs.append(
            {
                "chunk_id": cit.get("chunk_id", ""),
                "text": text,
                "corpus": agent,
                "collection": coll,
                "source": cit.get("source", ""),
                "location": cit.get("location", ""),
            }
        )

    rc_list = cg.get("root_cause_analysis")
    if isinstance(rc_list, list):
        for narrative in rc_list:
            if not isinstance(narrative, str) or not narrative.strip():
                continue
            docs.append(
                {
                    "chunk_id": "",
                    "text": narrative,
                    "corpus": "root_cause",
                    "collection": "root_cause_analysis",
                    "source": "",
                    "location": "",
                }
            )

    analytical_list = cg.get("analytical_results")
    if isinstance(analytical_list, list):
        for entry in analytical_list:
            if not isinstance(entry, str) or not entry.strip():
                continue
            docs.append(
                {
                    "chunk_id": "n/a",
                    "text": entry,
                    "corpus": "data_analyst",
                    "collection": "analytical",
                    "source": "DuckDB query",
                    "location": "",
                }
            )

    agents_called: list[str] = []
    for d in docs:
        c = d.get("corpus") or "unknown"
        if c not in agents_called:
            agents_called.append(c)

    return docs, agents_called


# ---------------------------------------------------------------------------
# Evaluation run file logging
# ---------------------------------------------------------------------------


def configure_evaluation_file_logging(
    log_path: Path,
    *,
    level: int = logging.INFO,
    format_str: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
) -> logging.Handler:
    """Attach a UTF-8 file handler to the root logger for a full eval run.

    All log records at *level* and above from any logger propagate to the root
    logger and are written to *log_path*. Idempotent for the same *log_path*:
    an existing file handler for that path is not added twice.

    Returns:
        The ``FileHandler`` instance (new or existing).
    """
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    for h in root.handlers:
        if isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", None):
            try:
                if Path(h.baseFilename).resolve() == log_path.resolve():
                    return h
            except OSError:
                pass

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(logging.Formatter(format_str))
    root.addHandler(fh)
    # Ensure INFO records reach the file when root was left at WARNING (no basicConfig).
    if root.level > level or root.level == logging.NOTSET:
        root.setLevel(level)
    return fh


def default_eval_log_path(run_id: str) -> Path:
    """Default path for ``--save-logs`` under ``data/evaluation/results/logs/``."""
    return EVAL_LOGS_DIR / f"{run_id}.log"


# ---------------------------------------------------------------------------
# Run ID generation
# ---------------------------------------------------------------------------


def generate_run_id() -> str:
    """Produce a timestamped run identifier.

    Format: ``eval_YYYY-MM-DD_v1``  (date in UTC).

    Returns:
        A string like ``"eval_2026-03-18_v1"``.
    """
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return f"eval_{date_str}_v1"


# ---------------------------------------------------------------------------
# Pre-evaluation ground-truth validation
# ---------------------------------------------------------------------------


def validate_ground_truth_chunks(
    dataset: list[dict[str, Any]],
    chromadb_path: Path,
    collections: list[str] | None = None,
) -> dict[str, Any]:
    """Validate that ground_truth_chunks in the dataset exist in ChromaDB.

    Issues warnings for missing or fuzzy-matched chunk IDs. Does NOT raise —
    this is a non-blocking pre-flight check to surface stale chunk IDs before
    a long evaluation run.

    Args:
        dataset: Loaded evaluation dataset entries.
        chromadb_path: Path to the ChromaDB persistent directory.
        collections: Collection names to check. Defaults to all four standard
            collections.

    Returns:
        Dict with keys ``total``, ``exact``, ``fuzzy``, ``missing``,
        ``alignment_rate``, and ``missing_details`` (list of dicts).
    """
    import chromadb as _chromadb

    _collections = collections or [
        "osha_regulations",
        "osha_enf",
        "csb",
        "csb_recommendations",
    ]

    logger = logging.getLogger(__name__)

    all_chunk_ids: set[str] = set()
    all_normalized: dict[str, list[str]] = {}

    try:
        client = _chromadb.PersistentClient(path=str(chromadb_path))
        for coll_name in _collections:
            try:
                coll = client.get_collection(coll_name)
                ids = coll.get(include=[])["ids"]
                for cid in ids:
                    all_chunk_ids.add(cid)
                    norm = re.sub(r"_chunk_\d+$", "", cid)
                    all_normalized.setdefault(norm, []).append(cid)
            except Exception as exc:
                logger.warning(
                    "Could not read ChromaDB collection '%s': %s", coll_name, exc
                )
    except Exception as exc:
        logger.warning(
            "ChromaDB unavailable for ground-truth validation (%s). Skipping.", exc
        )
        return {
            "total": 0,
            "exact": 0,
            "fuzzy": 0,
            "missing": 0,
            "alignment_rate": None,
            "missing_details": [],
        }

    total = exact = fuzzy = missing = 0
    missing_details: list[dict[str, Any]] = []

    for i, entry in enumerate(dataset):
        for chunk_id in entry.get("ground_truth_chunks", []):
            total += 1
            if chunk_id in all_chunk_ids:
                exact += 1
            else:
                norm = re.sub(r"_chunk_\d+$", "", chunk_id)
                if norm in all_normalized:
                    fuzzy += 1
                else:
                    missing += 1
                    missing_details.append(
                        {
                            "entry_index": i,
                            "category": entry.get("category", "unknown"),
                            "chunk_id": chunk_id,
                            "query": entry.get("query", "")[:80],
                        }
                    )

    alignment_rate = (exact + fuzzy) / total if total > 0 else None

    if missing_details:
        logger.warning(
            "Pre-eval ground-truth audit: %d/%d chunk IDs not found in ChromaDB "
            "(alignment %.1f%%). Run `python scripts/reusable/audit_ground_truth.py` for details.",
            missing,
            total,
            (alignment_rate or 0.0) * 100,
        )
    else:
        logger.info(
            "Pre-eval ground-truth audit: all %d chunk IDs verified (exact=%d, fuzzy=%d).",
            total,
            exact,
            fuzzy,
        )

    return {
        "total": total,
        "exact": exact,
        "fuzzy": fuzzy,
        "missing": missing,
        "alignment_rate": alignment_rate,
        "missing_details": missing_details,
    }
