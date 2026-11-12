"""Stage 3 -- DeepEval judge scoring layer.

Consumes ``generation.json`` (``dataset`` + ``pipeline_results``) only -- no
dependency on ChromaDB or the live pipeline. Unlike the legacy
``FaithfulnessEvaluator.evaluate_deepeval`` (which drives DeepEval's own
``ThreadPoolExecutor.map`` over the whole dataset with no per-entry
persistence), this module loops per-entry and checks/writes a
:class:`~src.evaluation.scoring.cache.ScoringCache` after every entry, so a
killed/interrupted ``run.py score`` resumes by only (re-)scoring entries
still missing from the cache instead of the whole ~1h pass.

``measure_fn`` is the test injection point: production callers leave it as
:func:`default_measure` (which lazily imports ``deepeval`` so importing this
module never requires the package installed); the mock-judge unit tests
pass a stub instead.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional, Tuple

from src.evaluation.judge_concurrency import judge_call_slot
from src.evaluation.scoring.cache import ScoringCache
from src.evaluation.utils import bootstrap_ci

logger = logging.getLogger(__name__)

LAYER_NAME = "deepeval"
METRICS = ("faithfulness", "hallucination", "correctness")

MeasureFn = Callable[[Dict[str, Any], Dict[str, Any], Any], Dict[str, Optional[float]]]


_MAX_EVAL_CONTEXT_DOC_CHARS = 20_000


def default_measure(
    entry: Dict[str, Any], result: Dict[str, Any], judge_model: Any
) -> Dict[str, Optional[float]]:
    """Real DeepEval measurement for one ``(entry, result)`` pair.

    Mirrors ``FaithfulnessEvaluator.evaluate_deepeval``'s per-entry metric
    construction (FaithfulnessMetric / HallucinationMetric / GEval
    correctness), but measures one entry at a time so the caller can persist
    each metric to the cache immediately.
    """
    from deepeval.metrics import FaithfulnessMetric, GEval, HallucinationMetric
    from deepeval.test_case import LLMTestCase, LLMTestCaseParams

    retrieved_contexts = [
        doc.get("text", "")[:_MAX_EVAL_CONTEXT_DOC_CHARS]
        for doc in result.get("context_gathered", [])
    ]
    test_case = LLMTestCase(
        input=entry.get("query", ""),
        actual_output=result.get("final_response", ""),
        retrieval_context=retrieved_contexts,
        context=retrieved_contexts,
        expected_output=entry.get("expected_answer", ""),
    )

    out: Dict[str, Optional[float]] = {}

    try:
        metric = FaithfulnessMetric(model=judge_model, threshold=0.9)
        with judge_call_slot():
            metric.measure(test_case)
        out["faithfulness"] = metric.score
    except Exception as exc:  # noqa: BLE001 - isolate per-metric failures
        logger.warning("deepeval faithfulness measure failed: %s", exc)
        out["faithfulness"] = None

    try:
        metric = HallucinationMetric(model=judge_model, threshold=0.05)
        with judge_call_slot():
            metric.measure(test_case)
        out["hallucination"] = metric.score
    except Exception as exc:  # noqa: BLE001
        logger.warning("deepeval hallucination measure failed: %s", exc)
        out["hallucination"] = None

    try:
        metric = GEval(
            name="Correctness",
            criteria="Determine whether the actual output is factually correct based on the expected output.",
            evaluation_params=[
                LLMTestCaseParams.INPUT,
                LLMTestCaseParams.ACTUAL_OUTPUT,
                LLMTestCaseParams.EXPECTED_OUTPUT,
            ],
            model=judge_model,
        )
        with judge_call_slot():
            metric.measure(test_case)
        out["correctness"] = metric.score
    except Exception as exc:  # noqa: BLE001
        logger.warning("deepeval correctness measure failed: %s", exc)
        out["correctness"] = None

    return out


def score_deepeval(
    dataset: List[Dict[str, Any]],
    results: List[Dict[str, Any]],
    *,
    judge_model: Any = None,
    judge_id: str = "unknown",
    generation_sha: str = "",
    cache: Optional[ScoringCache] = None,
    measure_fn: MeasureFn = default_measure,
    max_concurrency: int = 8,
) -> Dict[str, Any]:
    """Score one generation's DeepEval metrics, per-entry, cache-aware.

    Returns ``{"per_entry", "aggregates", "n_scored", "n_failed", "failures"}``
    -- the payload :func:`~src.evaluation.scoring.contract.build_score_file`
    expects, not yet wrapped in the full score-file envelope (that happens in
    ``run.py score``, which also knows wall-clock/judge-cost bookkeeping).
    """
    per_entry: Dict[str, Any] = {}
    failures: List[Dict[str, Any]] = []
    n_scored = 0
    n_failed = 0

    def _process_entry(
        i: int, entry: Dict[str, Any], result: Dict[str, Any]
    ) -> Tuple[str, Optional[Dict[str, Optional[float]]], Optional[str]]:
        entry_id = str(i)
        if "error" in result:
            return entry_id, None, None

        if cache is not None and cache.has_all(
            generation_sha, entry_id, METRICS, judge_id
        ):
            scores = {
                m: cache.get(generation_sha, entry_id, m, judge_id) for m in METRICS
            }
            return entry_id, scores, None

        try:
            scores = measure_fn(entry, result, judge_model)
            if cache is not None:
                for metric_name in METRICS:
                    if scores.get(metric_name) is not None:
                        cache.put(
                            generation_sha,
                            entry_id,
                            metric_name,
                            judge_id,
                            scores[metric_name],
                        )
            return entry_id, scores, None
        except Exception as exc:  # noqa: BLE001
            return entry_id, None, f"{type(exc).__name__}: {exc}"

    pairs = list(enumerate(zip(dataset, results)))
    if max_concurrency > 1:
        with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
            futs = [
                executor.submit(_process_entry, i, entry, result)
                for i, (entry, result) in pairs
            ]
            processed_results = [fut.result() for fut in futs]
    else:
        processed_results = [
            _process_entry(i, entry, result) for i, (entry, result) in pairs
        ]

    for entry_id, scores, error_reason in processed_results:
        if error_reason is not None:
            n_failed += 1
            failures.append({"entry_id": entry_id, "reason": error_reason})
            continue
        if scores is None:
            continue  # pipeline-error stub: nothing to judge

        per_entry[entry_id] = scores
        if any(v is not None for v in scores.values()):
            n_scored += 1
        else:
            n_failed += 1
            failures.append(
                {"entry_id": entry_id, "reason": "all deepeval metrics returned None"}
            )

    aggregates: Dict[str, Any] = {"n_rows": len(per_entry)}
    for metric_name in METRICS:
        values = [v.get(metric_name) for v in per_entry.values()]
        scored = [v for v in values if v is not None]
        if scored:
            mean_val = sum(scored) / len(scored)
            ci_lower, ci_upper = (
                bootstrap_ci(scored) if len(scored) > 1 else (mean_val, mean_val)
            )
            aggregates[metric_name] = {
                "mean": mean_val,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "n_scored": len(scored),
            }
            if len(scored) != len(values):
                aggregates[metric_name]["n_failed"] = len(values) - len(scored)
        else:
            aggregates[metric_name] = {
                "error": (
                    f"no score returned for any of {len(values)} row(s)"
                    if values
                    else "no rows to score"
                )
            }

    # Real judge call/cost accounting, if the judge model is trackable
    # (see judge_call_stats' docstring for the one case -- a bare OpenAI
    # judge string -- it can't cover).
    from src.evaluation.judge_factory import judge_call_stats

    call_stats = judge_call_stats(judge_model)

    return {
        "per_entry": per_entry,
        "aggregates": aggregates,
        "n_scored": n_scored,
        "n_failed": n_failed,
        "failures": failures,
        "judge_calls": call_stats["n_calls"] if call_stats else None,
        "judge_cost_usd": call_stats["cost_usd"] if call_stats else None,
    }
