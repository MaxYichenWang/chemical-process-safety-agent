"""Stage 2 -- deterministic (non-judge, non-corpus) scoring layer.

Every metric computed here is a **pure function** of ``(dataset,
pipeline_results)``: no LLM judge call, no ChromaDB connection, no network
access, no API key. Given a committed ``generation.json``, this layer is
bit-reproducible and free -- separating it out this way means metric logic
can be defined, fixed, and validated with no judge key and no corpus for
roughly half the metric surface.

Deliberately does **not** import ``FaithfulnessEvaluator``: that module
imports ``scipy.stats`` at module level purely to support
``evaluate_cross_framework`` (a judge-vs-judge comparison this layer never
touches). Citation accuracy is computed directly via
``src.evaluation.citation_utils.score_citation_accuracy`` instead, so this
module's only third-party import is ``numpy`` (transitively, via
``src.evaluation.utils.bootstrap_ci``). Confirmed import-clean with
``EVAL_LLM_*`` unset and no ChromaDB directory present.

This module is the single implementation; ``EndToEndEvaluator`` (the legacy
all-in-one evaluator) delegates its per-category rollup to
:func:`score_per_category` rather than keeping a second copy.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from src.evaluation import citation_utils
from src.evaluation.evaluators.analytical_evaluator import AnalyticalEvaluator
from src.evaluation.evaluators.domain_evaluator import DomainEvaluator
from src.evaluation.evaluators.orchestration_evaluator import OrchestrationEvaluator
from src.evaluation.evaluators.retrieval_evaluator import RetrievalEvaluator
from src.evaluation.utils import guarded

LAYER_NAME = "deterministic"

# The complete set of top-level metric sections this layer produces. Used by
# ``aggregate`` to know which report rows must render "N/A -- layer
# absent" rather than a false zero when this score file is missing.
SECTIONS = (
    "retrieval",
    "faithfulness",
    "orchestration",
    "domain",
    "analytical",
    "per_category",
)


def _split_ok(
    dataset: List[Dict[str, Any]], results: List[Dict[str, Any]]
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Partition into ``(ok_dataset, ok_results)``, dropping pipeline-error stubs.

    Mirrors ``EndToEndEvaluator.evaluate()``'s partitioning: quality metrics
    must never see a forced ``validation_score=0.0`` / ``final_response=""``
    stub, which would deflate their distributions without reflecting
    anything about actual answer quality. Orchestration metrics take the
    *full*, unpartitioned lists instead -- see :func:`score_orchestration`.
    """
    pairs = [(d, r) for d, r in zip(dataset, results) if "error" not in r]
    if not pairs:
        return [], []
    ok_dataset, ok_results = (list(x) for x in zip(*pairs))
    return ok_dataset, ok_results


def score_retrieval(
    dataset: List[Dict[str, Any]], results: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Layer 1 custom metrics: Hit@5 / MRR / NDCG@5, overall and per-worker."""
    ok_dataset, ok_results = _split_ok(dataset, results)
    retrieval_eval = RetrievalEvaluator()
    return {
        "custom_metrics": guarded(
            "retrieval.custom",
            retrieval_eval.evaluate_batch_custom,
            ok_dataset,
            ok_results,
        ),
        "per_worker_metrics": guarded(
            "retrieval.per_worker",
            retrieval_eval.evaluate_per_worker,
            ok_dataset,
            ok_results,
        ),
    }


def score_faithfulness(
    dataset: List[Dict[str, Any]], results: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Regex-only (OSHA-standard-number) citation accuracy -- the non-judge half of Layer 2."""
    _ok_dataset, ok_results = _split_ok(dataset, results)
    return {
        "citation_accuracy": guarded(
            "faithfulness.citation_accuracy",
            citation_utils.score_citation_accuracy,
            ok_results,
            include_enforcement_ids=False,
        ),
    }


def score_orchestration(
    dataset: List[Dict[str, Any]], results: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Layer 3 orchestration metrics (all deterministic; ragas/deepeval goal-accuracy excluded).

    Receives the *full*, unpartitioned ``dataset``/``results`` (including
    pipeline-error stubs): ``pipeline_error_rate`` needs the error count, and
    every other sub-metric here already gates on the ``"error"`` key
    internally (see ``OrchestrationEvaluator``).
    """
    orch = OrchestrationEvaluator()
    return {
        "pipeline_error_rate": guarded(
            "orchestration.pipeline_error_rate",
            orch.evaluate_pipeline_error_rate,
            results,
        ),
        "tool_call_accuracy": guarded(
            "orchestration.tool_call_accuracy",
            orch.evaluate_tool_call_accuracy,
            dataset,
            results,
        ),
        "plan_accuracy": guarded(
            "orchestration.plan_accuracy", orch.evaluate_plan_accuracy, dataset, results
        ),
        "fast_route_coverage": guarded(
            "orchestration.fast_route_coverage",
            orch.evaluate_fast_route_coverage,
            dataset,
            results,
        ),
        "cost": guarded("orchestration.cost", orch.evaluate_cost, results),
        "critic_distribution": guarded(
            "orchestration.critic_distribution",
            orch.evaluate_critic_distribution,
            results,
        ),
        "retry_rate": guarded(
            "orchestration.retry_rate", orch.evaluate_retry_rate, results
        ),
    }


def score_domain(
    dataset: List[Dict[str, Any]], results: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Domain-aware citation accuracy (OSHA std + enforcement IDs), incident fidelity, hedging."""
    ok_dataset, ok_results = _split_ok(dataset, results)
    domain = DomainEvaluator()
    return {
        "citation_accuracy": guarded(
            "domain.citation_accuracy", domain.evaluate_citation_accuracy, ok_results
        ),
        "incident_fidelity": guarded(
            "domain.incident_fidelity", domain.evaluate_incident_fidelity, ok_results
        ),
        "hedging_detection": guarded(
            "domain.hedging_detection",
            domain.evaluate_hedging_detection,
            ok_dataset,
            ok_results,
        ),
        "safety_claim_detection": guarded(
            "domain.safety_claim_detection",
            domain.evaluate_safety_claim_detection,
            ok_results,
        ),
    }


def score_analytical(
    dataset: List[Dict[str, Any]], results: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Layer 4 (Statistical Aggregation): fully deterministic already -- see ``AnalyticalEvaluator``."""
    ok_dataset, ok_results = _split_ok(dataset, results)
    return AnalyticalEvaluator().evaluate(ok_dataset, ok_results)


def score_per_category(
    dataset: List[Dict[str, Any]], results: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Deterministic per-category rollups (Hit@5, citation acc., tool exact, cost, ... per category)."""
    retrieval_eval = RetrievalEvaluator()
    orch = OrchestrationEvaluator()
    domain = DomainEvaluator()
    analytical_eval = AnalyticalEvaluator()

    categories = sorted({entry.get("category", "unknown") for entry in dataset})
    output: Dict[str, Any] = {}

    for category in categories:
        cat_pairs = [
            (entry, result)
            for entry, result in zip(dataset, results)
            if entry.get("category", "unknown") == category
        ]
        cat_dataset = [d for d, _ in cat_pairs]
        cat_results = [r for _, r in cat_pairs]

        clean_pairs = [
            (entry, result) for entry, result in cat_pairs if "error" not in result
        ]
        clean_dataset = [d for d, _ in clean_pairs]
        clean_results = [r for _, r in clean_pairs]

        output[category] = {
            "n_total": len(cat_results),
            "n_ok": len(clean_results),
            "n_error": len(cat_results) - len(clean_results),
            "retrieval": {
                "custom_metrics": retrieval_eval.evaluate_batch_custom(
                    clean_dataset, clean_results
                )
            },
            "faithfulness": {
                "citation_accuracy": citation_utils.score_citation_accuracy(
                    clean_results, include_enforcement_ids=False
                )
            },
            "orchestration": {
                "pipeline_error_rate": orch.evaluate_pipeline_error_rate(cat_results),
                "tool_call_accuracy": orch.evaluate_tool_call_accuracy(
                    cat_dataset, cat_results
                ),
                "plan_accuracy": orch.evaluate_plan_accuracy(cat_dataset, cat_results),
                "fast_route_coverage": orch.evaluate_fast_route_coverage(
                    cat_dataset, cat_results
                ),
                "critic_distribution": orch.evaluate_critic_distribution(cat_results),
                "retry_rate": orch.evaluate_retry_rate(cat_results),
                "cost": orch.evaluate_cost(cat_results),
            },
            "domain": {
                "citation_accuracy": domain.evaluate_citation_accuracy(clean_results),
                "incident_fidelity": domain.evaluate_incident_fidelity(clean_results),
                "hedging_detection": domain.evaluate_hedging_detection(
                    clean_dataset, clean_results
                ),
            },
            "analytical": analytical_eval.evaluate(clean_dataset, clean_results),
        }

    return output


def score_deterministic(
    dataset: List[Dict[str, Any]], results: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Compute every Stage-2 (non-judge, non-corpus) metric for a generation.

    Pure function: identical ``(dataset, results)`` input always produces
    byte-identical output -- there is no
    randomness, wall-clock dependency, or external I/O anywhere in this call
    graph. No network, no API key, no ChromaDB.
    """
    return {
        "retrieval": score_retrieval(dataset, results),
        "faithfulness": score_faithfulness(dataset, results),
        "orchestration": score_orchestration(dataset, results),
        "domain": score_domain(dataset, results),
        "analytical": score_analytical(dataset, results),
        "per_category": guarded("per_category", score_per_category, dataset, results),
    }
