"""Stage 4 -- RAGAS judge scoring layer.

Consumes ``generation.json`` (``dataset`` + ``pipeline_results``) only.
Wraps the existing RAGAS v0.4 driver code
(``src.evaluation.ragas_experiments.MetricScorer`` / ``score_rows``, shared
with the legacy evaluators) to produce the standard score-file
``per_entry``/``failures`` structure rather than the legacy aggregate-only
shape.

``metrics_factory`` is the test injection point: production callers leave it
as :func:`default_metrics_factory` (which lazily imports ``ragas`` so
importing this module never requires the package installed); tests pass a
stub factory instead (see the empty-content instrumentation in
``ragas_experiments.MetricScorer``, which is exercised transitively here).
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence

from src.evaluation.ragas_experiments import (
    MetricScorer,
    aggregate_metric_records,
    score_rows,
)

LAYER_NAME = "ragas"
DEFAULT_METRICS = ("faithfulness", "answer_correctness")

MetricsFactory = Callable[[], Dict[str, Any]]

# Each RAGAS metric's ``ascore()`` takes a different kwarg shape -- see
# tests/evaluation/test_ragas_contract.py for the authoritative signatures.
_KWARGS_BY_METRIC: Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]] = {
    "faithfulness": lambda row: {
        "user_input": row["user_input"],
        "response": row["response"],
        "retrieved_contexts": row["retrieved_contexts"],
    },
    "answer_correctness": lambda row: {
        "user_input": row["user_input"],
        "response": row["response"],
        "reference": row["reference"],
    },
}


def default_metrics_factory() -> Dict[str, Any]:
    """Real RAGAS metric objects, built from the configured eval_judge."""
    from ragas.metrics.collections import AnswerCorrectness, Faithfulness

    from src.evaluation.judge_factory import make_ragas_embeddings, make_ragas_judge_llm

    llm = make_ragas_judge_llm()
    embeddings = make_ragas_embeddings()
    return {
        "faithfulness": Faithfulness(llm=llm),
        "answer_correctness": AnswerCorrectness(llm=llm, embeddings=embeddings),
    }


_MAX_EVAL_CONTEXT_DOC_CHARS = 20_000


def _build_row(entry: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
    gt_text = " ".join(entry.get("ground_truth_chunks", []))
    return {
        "user_input": entry.get("query", ""),
        "response": result.get("final_response", ""),
        "retrieved_contexts": [
            doc.get("text", "")[:_MAX_EVAL_CONTEXT_DOC_CHARS]
            for doc in result.get("context_gathered", [])
        ],
        "reference": entry.get("expected_answer", gt_text),
    }


def score_ragas(
    dataset: List[Dict[str, Any]],
    results: List[Dict[str, Any]],
    *,
    metrics_factory: MetricsFactory = default_metrics_factory,
    metric_names: Sequence[str] = DEFAULT_METRICS,
    max_concurrency: int = 4,
    timeout_s: Optional[float] = None,
) -> Dict[str, Any]:
    """Score one generation's RAGAS metrics, entry-indexed, standard score-file shaped."""
    metric_names = list(metric_names)
    ok_pairs = [
        (i, e, r) for i, (e, r) in enumerate(zip(dataset, results)) if "error" not in r
    ]

    if not ok_pairs:
        return {
            "per_entry": {},
            "aggregates": {"n_rows": 0},
            "n_scored": 0,
            "n_failed": 0,
            "failures": [],
        }

    entry_ids = [str(i) for i, _e, _r in ok_pairs]
    rows = [_build_row(e, r) for _i, e, r in ok_pairs]

    metrics = metrics_factory()
    unknown = set(metric_names) - set(_KWARGS_BY_METRIC)
    if unknown:
        raise ValueError(f"no kwarg mapping for ragas metric(s): {sorted(unknown)}")

    scorer = MetricScorer()

    async def score_row(row: Dict[str, Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for name in metric_names:
            out[name] = await scorer.score(
                name, metrics[name], **_KWARGS_BY_METRIC[name](row)
            )
        return out

    records = score_rows(
        score_row, rows, max_concurrency=max_concurrency, timeout_s=timeout_s
    )

    per_entry: Dict[str, Any] = {
        eid: (rec or {name: None for name in metric_names})
        for eid, rec in zip(entry_ids, records)
    }

    failures: List[Dict[str, Any]] = []
    n_scored = 0
    n_failed = 0
    for eid, rec in per_entry.items():
        if any(v is not None for v in rec.values()):
            n_scored += 1
        else:
            n_failed += 1
        for name in metric_names:
            if rec.get(name) is None:
                reason = scorer.errors.get(name, ["no usable value returned"])[-1]
                failures.append({"entry_id": eid, "metric": name, "reason": reason})

    aggregates = aggregate_metric_records(records, metric_names, errors=scorer.errors)

    return {
        "per_entry": per_entry,
        "aggregates": aggregates,
        "n_scored": n_scored,
        "n_failed": n_failed,
        "failures": failures,
    }
