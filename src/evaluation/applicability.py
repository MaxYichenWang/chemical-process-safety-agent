"""Entry applicability — which entries may legitimately enter which metric.

Background
----------
Not every benchmark entry can support every metric. Three separate reasons, and
conflating them is how a benchmark reports a number it has not earned:

``l1_applicable: false``
    The entry has no usable ground-truth chunks, so retrieval metrics (Hit@5,
    MRR, NDCG) cannot be computed. Honoured at *scoring* time by
    :mod:`src.evaluation.evaluators.retrieval_evaluator`.

``answer_gradable: false``
    The entry's *reference answer* is unusable — a raw SQL string, or an answer
    that states facts belonging to a different incident (Stage C-bis). Metrics
    that grade the response **against the reference** are meaningless here.
    Metrics that grade the response against the **retrieved context**
    (faithfulness, hallucination) remain perfectly valid, because they never
    consult the reference. Keeping that distinction is the whole point of this
    module: quarantining a bad reference answer must not silently discard the
    faithfulness signal too.

``domain_relevance``
    Descriptive only. Supports reporting results over the whole benchmark *and*
    over its process-safety core. **Never filters anything.**

Why applicability is applied at aggregation time
------------------------------------------------
Exclusions are computed from the *per-entry* scores already stored in each
score file, rather than by skipping entries during measurement. That choice
matters operationally: 72 committed score files, representing roughly $60 of
judge calls, stay valid. Correcting a denominator becomes a free re-aggregate
instead of a paid re-score.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

#: Metrics that grade a response *against the reference answer*. These are the
#: only ones invalidated by ``answer_gradable: false``.
REFERENCE_DEPENDENT_METRICS = frozenset(
    {
        "correctness",
        "answer_correctness",
        "factual_correctness",
        "answer_relevancy",
    }
)

#: Metrics graded against retrieved context, never against the reference. They
#: stay valid even when the reference answer is quarantined.
CONTEXT_DEPENDENT_METRICS = frozenset({"faithfulness", "hallucination"})

UNKNOWN_DOMAIN = "unclassified"


def exclude_from_reference_metrics(entry: Mapping[str, Any]) -> bool:
    """True when the entry's reference answer cannot support grading.

    Only an explicit ``answer_gradable: false`` excludes. Absence of the flag
    means "not assessed", which must never be treated as "excluded" — that
    would silently shrink denominators on datasets that predate the flag.
    """
    return entry.get("answer_gradable") is False


def exclude_from_retrieval_metrics(entry: Mapping[str, Any]) -> bool:
    """True when the entry has no ground truth usable for retrieval metrics.

    Mirrors the authoritative check in ``RetrievalEvaluator`` for the explicit
    flag only; category heuristics stay in the evaluator that owns them.
    """
    return entry.get("l1_applicable") is False


def domain_bucket(entry: Mapping[str, Any]) -> str:
    return entry.get("domain_relevance") or UNKNOWN_DOMAIN


def defect_codes(entry: Mapping[str, Any]) -> List[str]:
    defects = entry.get("defects")
    if not isinstance(defects, list):
        return []
    return sorted(
        {d.get("code") for d in defects if isinstance(d, dict) and d.get("code")}
    )


def dataset_applicability(dataset: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Summarise how many entries each metric family may legitimately use."""
    total = len(dataset)
    retrieval_excluded = [
        i for i, e in enumerate(dataset) if exclude_from_retrieval_metrics(e)
    ]
    reference_excluded = [
        i for i, e in enumerate(dataset) if exclude_from_reference_metrics(e)
    ]

    domains: Dict[str, int] = {}
    codes: Dict[str, int] = {}
    for entry in dataset:
        bucket = domain_bucket(entry)
        domains[bucket] = domains.get(bucket, 0) + 1
        for code in defect_codes(entry):
            codes[code] = codes.get(code, 0) + 1

    return {
        "n_entries": total,
        "retrieval_denominator": total - len(retrieval_excluded),
        "retrieval_excluded_indices": retrieval_excluded,
        "reference_denominator": total - len(reference_excluded),
        "reference_excluded_indices": reference_excluded,
        "domain_relevance": dict(sorted(domains.items())),
        "defect_codes": dict(sorted(codes.items())),
        "entries_with_defects": sum(1 for e in dataset if defect_codes(e)),
    }


def _mean_ci(values: Sequence[float]) -> Dict[str, Any]:
    from src.evaluation.utils import bootstrap_ci

    mean_value = sum(values) / len(values)
    if len(values) > 1:
        ci_lower, ci_upper = bootstrap_ci(list(values))
    else:
        ci_lower, ci_upper = mean_value, mean_value
    return {
        "mean": mean_value,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "n_scored": len(values),
    }


def recompute_reference_metrics(
    per_entry: Mapping[str, Mapping[str, Any]],
    aggregates: Mapping[str, Any],
    dataset: Sequence[Mapping[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Recompute reference-dependent metric aggregates with quarantine applied.

    ``per_entry`` is keyed by the entry's dataset index as a string, the layout
    written by the scoring layers. Returns ``None`` when nothing is quarantined,
    so callers can leave existing aggregates untouched and report no adjustment
    rather than an adjustment of zero.
    """
    excluded = {i for i, e in enumerate(dataset) if exclude_from_reference_metrics(e)}
    if not excluded:
        return None

    adjusted: Dict[str, Any] = {}
    for metric in sorted(set(aggregates) & REFERENCE_DEPENDENT_METRICS):
        kept: List[float] = []
        dropped = 0
        for key, scores in per_entry.items():
            try:
                index = int(key)
            except (TypeError, ValueError):
                continue
            if not isinstance(scores, Mapping):
                continue
            value = scores.get(metric)
            if value is None:
                continue
            if index in excluded:
                dropped += 1
                continue
            kept.append(float(value))

        if not kept:
            # Refuse to emit a number computed from nothing. §3.3's honest
            # reporting contract: absent is absent, never 0.0.
            adjusted[metric] = {
                "mean": None,
                "n_scored": 0,
                "n_excluded": dropped,
                "reason": "every scored entry for this metric is answer_gradable=false",
            }
            continue

        entry_stats = _mean_ci(kept)
        entry_stats["n_excluded"] = dropped
        previous = aggregates.get(metric)
        if isinstance(previous, Mapping) and previous.get("mean") is not None:
            entry_stats["mean_before_exclusion"] = previous["mean"]
            entry_stats["delta"] = entry_stats["mean"] - previous["mean"]
        adjusted[metric] = entry_stats

    if not adjusted:
        return None

    return {
        "policy": "answer_gradable=false entries excluded from reference-dependent metrics only",
        "reference_dependent_metrics_adjusted": sorted(adjusted),
        "context_dependent_metrics_unchanged": sorted(
            set(aggregates) & CONTEXT_DEPENDENT_METRICS
        ),
        "n_excluded_entries": len(excluded),
        "metrics": adjusted,
    }


def domain_breakdown(
    per_entry: Mapping[str, Mapping[str, Any]],
    dataset: Sequence[Mapping[str, Any]],
    metrics: Iterable[str],
) -> Dict[str, Dict[str, Any]]:
    """Per-domain means for the requested metrics. Descriptive, never filtering.

    Entries quarantined by ``answer_gradable`` are omitted from
    reference-dependent metrics here too, so the breakdown and the headline
    number are computed on the same population.
    """
    wanted = list(metrics)
    buckets: Dict[str, Dict[str, List[float]]] = {}

    for key, scores in per_entry.items():
        try:
            index = int(key)
        except (TypeError, ValueError):
            continue
        if not (0 <= index < len(dataset)) or not isinstance(scores, Mapping):
            continue
        entry = dataset[index]
        bucket = buckets.setdefault(domain_bucket(entry), {})
        for metric in wanted:
            value = scores.get(metric)
            if value is None:
                continue
            if (
                metric in REFERENCE_DEPENDENT_METRICS
                and exclude_from_reference_metrics(entry)
            ):
                continue
            bucket.setdefault(metric, []).append(float(value))

    return {
        name: {
            metric: {"mean": sum(values) / len(values), "n_scored": len(values)}
            for metric, values in sorted(metric_values.items())
        }
        for name, metric_values in sorted(buckets.items())
    }
