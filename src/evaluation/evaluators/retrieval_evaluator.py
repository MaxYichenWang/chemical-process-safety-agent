import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List

try:
    from src.evaluation.config import (
        CHROMADB_DIR,
        EVAL_RETRIEVAL_K,
        EVAL_WORKER_TIMEOUT,
    )
except ImportError:
    CHROMADB_DIR = Path("chromadb")
    EVAL_RETRIEVAL_K = 5
    EVAL_WORKER_TIMEOUT = 120

import re

from src.evaluation.judge_concurrency import judge_call_slot

# Regex to strip trailing _chunk_N suffix for fuzzy matching
_CHUNK_SUFFIX_RE = re.compile(r"_chunk_\d+$")


def _normalize_chunk_id(cid: str) -> str:
    """Strip trailing ``_chunk_N`` suffix so IDs with and without it match."""
    return _CHUNK_SUFFIX_RE.sub("", cid)


def _chunk_id_matches(retrieved_id: str, ground_truth_id: str) -> bool:
    """Compare two chunk IDs, tolerating a ``_chunk_N`` suffix on either side.

    This handles the common mismatch where ground-truth IDs lack the
    ``_chunk_0`` suffix appended by ``generate_chunk_id`` during indexing
    (or vice-versa).
    """
    if retrieved_id == ground_truth_id:
        return True
    return _normalize_chunk_id(retrieved_id) == _normalize_chunk_id(ground_truth_id)


def _any_gt_in_retrieved(retrieved: List[str], ground_truth: List[str]) -> bool:
    """Return True if any ground-truth ID matches a retrieved ID (fuzzy)."""
    return any(_chunk_id_matches(r, gt) for gt in ground_truth for r in retrieved)


def hit_rate_at_k(retrieved: List[str], ground_truth: List[str], k: int = 5) -> float:
    if not ground_truth:
        return 0.0
    return 1.0 if _any_gt_in_retrieved(retrieved[:k], ground_truth) else 0.0


def mean_reciprocal_rank(retrieved: List[str], ground_truth: List[str]) -> float:
    if not ground_truth:
        return 0.0
    for rank, chunk_id in enumerate(retrieved, 1):
        if any(_chunk_id_matches(chunk_id, gt) for gt in ground_truth):
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: List[str], ground_truth: List[str], k: int = 5) -> float:
    if not ground_truth:
        return 0.0

    dcg = 0.0
    for i, chunk_id in enumerate(retrieved[:k]):
        if any(_chunk_id_matches(chunk_id, gt) for gt in ground_truth):
            dcg += 1.0 / math.log2(
                i + 2
            )  # i+2 because i is 0-indexed, rank is i+1 -> log2(rank+1) = log2((i+1)+1) = log2(i+2)

    idcg = 0.0
    num_ideal = min(k, len(ground_truth))
    for i in range(num_ideal):
        idcg += 1.0 / math.log2(i + 2)

    return dcg / idcg if idcg > 0 else 0.0


_DEEPEVAL_MAX_RETRIES = 3
_DEEPEVAL_TIMEOUT_S = EVAL_WORKER_TIMEOUT
_DEEPEVAL_PARALLELISM = 4


class RetrievalEvaluator:
    def __init__(self, chromadb_dir: Path = CHROMADB_DIR):
        self.chromadb_dir = chromadb_dir

    def evaluate_single(
        self, query: str, retrieved_chunks: List[str], ground_truth_chunks: List[str]
    ) -> Dict[str, float]:
        """Compute Layer 1 retrieval metrics for a single query."""
        if not ground_truth_chunks:
            return {"hit_rate_at_5": 0.0, "mrr": 0.0, "ndcg_at_5": 0.0}

        return {
            "hit_rate_at_5": hit_rate_at_k(
                retrieved_chunks, ground_truth_chunks, EVAL_RETRIEVAL_K
            ),
            "mrr": mean_reciprocal_rank(retrieved_chunks, ground_truth_chunks),
            "ndcg_at_5": ndcg_at_k(
                retrieved_chunks, ground_truth_chunks, EVAL_RETRIEVAL_K
            ),
        }

    @staticmethod
    def _exclude_from_retrieval_metrics(entry: Dict[str, Any]) -> bool:
        """Return True if entry should be excluded from retrieval metrics.

        an explicit ``l1_applicable: false`` flag is authoritative and
        excludes the entry regardless of category — this is how the dataset
        marks the 36 zero-GT entries (24 adversarial + 12 statistical/hybrid)
        so retrieval denominators are honest. Previously ``hybrid_analytical``
        zero-GT entries leaked in and scored 0%, depressing Hit@5/NDCG.

        R3: Adversarial entries have empty ground_truth_chunks by design.
        R6: Pure-SQL categories (statistical_*) use DuckDB, not ChromaDB retrieval.
        These entries always score 0% on Hit@5/NDCG regardless of pipeline quality.
        """
        # explicit flag wins over category heuristics.
        if entry.get("l1_applicable") is False:
            return True
        cat = entry.get("category", "")
        src = entry.get("source", "")
        # R3: adversarial
        if "adversarial" in cat or "adversarial" in src:
            return True
        # R6: pure-SQL categories (no retrieval to evaluate)
        if cat in (
            "statistical_factual",
            "statistical_comparative",
            "statistical_trend",
        ):
            return True
        return False

    def evaluate_batch_custom(
        self, dataset: List[Dict[str, Any]], results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Evaluate custom retrieval metrics for a batch.

        Excludes adversarial (R3) and pure-SQL (R6) entries from retrieval
        metrics since they have no ground_truth_chunks or no retrieval path.
        """
        from src.evaluation.utils import bootstrap_ci

        metrics = {"hit_rate_at_5": [], "mrr": [], "ndcg_at_5": []}

        for entry, result in zip(dataset, results):
            if self._exclude_from_retrieval_metrics(entry):
                continue
            gt = entry.get("ground_truth_chunks", [])
            retrieved = [
                doc.get("chunk_id", doc.get("doc_id", ""))
                for doc in result.get("context_gathered", [])
            ]

            scores = self.evaluate_single(entry.get("query", ""), retrieved, gt)

            for key, val in scores.items():
                metrics[key].append(val)

        # Aggregate with CI
        aggregated = {}
        for key, vals in metrics.items():
            if vals:
                mean_val = sum(vals) / len(vals)
                ci_lower, ci_upper = (
                    bootstrap_ci(vals) if len(vals) > 1 else (mean_val, mean_val)
                )
                aggregated[key] = {
                    "mean": mean_val,
                    "ci_lower": ci_lower,
                    "ci_upper": ci_upper,
                    # Lets the report print "64.4% (132/205)" instead of
                    # a bare percentage with no denominator.
                    "n": len(vals),
                }
            else:
                # No applicable entries (every entry excluded as
                # adversarial/pure-SQL) is not the same as "0% hit
                # rate" -- reporting a numeric here would be indistinguishable
                # from a real zero.
                aggregated[key] = {
                    "status": "no_applicable_entries",
                    "reason": "no applicable entries (n=0)",
                    "n": 0,
                }

        return aggregated

    def evaluate_batch_ragas(
        self, dataset: List[Dict[str, Any]], results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Compute RAGAS v0.4 context_precision and context_recall."""
        from src.evaluation.config import EVAL_ENABLE_RAGAS

        if not EVAL_ENABLE_RAGAS:
            return {"skipped": "ragas disabled for current judge model configuration"}

        try:
            from ragas.metrics.collections import (
                ContextPrecisionWithReference,
                ContextRecall,
            )
        except (ImportError, TypeError):
            return {"error": "ragas not installed"}

        from src.evaluation.config import EVAL_MAX_WORKERS, EVAL_WORKER_TIMEOUT
        from src.evaluation.ragas_experiments import (
            MetricScorer,
            aggregate_metric_records,
            score_rows,
        )

        rows: List[Dict[str, Any]] = []
        for entry, result in zip(dataset, results):
            gt_text = " ".join(entry.get("ground_truth_chunks", []))
            rows.append(
                {
                    "user_input": entry.get("query", ""),
                    "retrieved_contexts": [
                        doc.get("text", "")
                        for doc in result.get("context_gathered", [])
                    ],
                    "reference": entry.get("expected_answer", gt_text),
                }
            )

        try:
            from src.evaluation.judge_factory import make_ragas_judge_llm

            llm = make_ragas_judge_llm()
            # Neither of these takes an ``embeddings`` argument in ragas 0.4;
            # passing one is silently swallowed by BaseMetric's **kwargs.
            context_precision_metric = ContextPrecisionWithReference(llm=llm)
            context_recall_metric = ContextRecall(llm=llm)
        except Exception as exc:  # noqa: BLE001
            return {"error": f"ragas judge/metric setup failed: {exc}"}

        scorer = MetricScorer()

        async def score_row(row: Dict[str, Any]) -> Dict[str, Any]:
            return {
                "context_precision": await scorer.score(
                    "context_precision",
                    context_precision_metric,
                    user_input=row["user_input"],
                    reference=row["reference"],
                    retrieved_contexts=row["retrieved_contexts"],
                ),
                "context_recall": await scorer.score(
                    "context_recall",
                    context_recall_metric,
                    user_input=row["user_input"],
                    retrieved_contexts=row["retrieved_contexts"],
                    reference=row["reference"],
                ),
            }

        try:
            records = score_rows(
                score_row,
                rows,
                max_concurrency=EVAL_MAX_WORKERS,
                timeout_s=EVAL_WORKER_TIMEOUT,
            )
        except Exception as exc:  # noqa: BLE001
            return {"error": f"ragas execution failed: {exc}"}

        return aggregate_metric_records(
            records, ["context_precision", "context_recall"], errors=scorer.errors
        )

    def evaluate_batch_deepeval(
        self, dataset: List[Dict[str, Any]], results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Compute DeepEval ContextualPrecisionMetric and ContextualRecallMetric."""
        try:
            from deepeval.metrics import (
                ContextualPrecisionMetric,
                ContextualRecallMetric,
            )
            from deepeval.test_case import LLMTestCase

            from src.evaluation.judge_factory import make_deepeval_judge_model
        except (ImportError, TypeError):
            return {"error": "deepeval not installed"}

        def _build_metric(metric_cls: Any, **kwargs: Any) -> Any:
            full_kwargs = {
                **kwargs,
                "max_retries": _DEEPEVAL_MAX_RETRIES,
                "timeout": _DEEPEVAL_TIMEOUT_S,
            }
            try:
                return metric_cls(**full_kwargs)
            except TypeError:
                return metric_cls(**kwargs)

        def _evaluate_entry(
            entry: Dict[str, Any], result: Dict[str, Any]
        ) -> tuple[Any, Any]:
            retrieved_contexts = [
                doc.get("text", "") for doc in result.get("context_gathered", [])
            ]
            test_case = LLMTestCase(
                input=entry.get("query", ""),
                actual_output=result.get("final_response", ""),
                retrieval_context=retrieved_contexts,
                expected_output=entry.get("expected_answer", ""),
            )

            contextual_precision_metric = _build_metric(
                ContextualPrecisionMetric,
                model=make_deepeval_judge_model(),
                threshold=0.7,
            )
            contextual_recall_metric = _build_metric(
                ContextualRecallMetric, model=make_deepeval_judge_model(), threshold=0.7
            )

            def _measure_with_retries(metric: Any) -> bool:
                for _ in range(_DEEPEVAL_MAX_RETRIES):
                    try:
                        with judge_call_slot():
                            metric.measure(test_case)
                        return True
                    except Exception:
                        continue
                return False

            precision = (
                contextual_precision_metric.score
                if _measure_with_retries(contextual_precision_metric)
                else None
            )
            recall = (
                contextual_recall_metric.score
                if _measure_with_retries(contextual_recall_metric)
                else None
            )
            return precision, recall

        precisions: List[float] = []
        recalls: List[float] = []
        with ThreadPoolExecutor(max_workers=_DEEPEVAL_PARALLELISM) as executor:
            for precision, recall in executor.map(_evaluate_entry, dataset, results):
                if precision is not None:
                    precisions.append(precision)
                if recall is not None:
                    recalls.append(recall)

        metrics = {}
        from src.evaluation.utils import bootstrap_ci

        if precisions:
            mean_val = sum(precisions) / len(precisions)
            ci_l, ci_u = (
                bootstrap_ci(precisions)
                if len(precisions) > 1
                else (mean_val, mean_val)
            )
            metrics["contextual_precision"] = {
                "mean": mean_val,
                "ci_lower": ci_l,
                "ci_upper": ci_u,
            }

        if recalls:
            mean_val = sum(recalls) / len(recalls)
            ci_l, ci_u = (
                bootstrap_ci(recalls) if len(recalls) > 1 else (mean_val, mean_val)
            )
            metrics["contextual_recall"] = {
                "mean": mean_val,
                "ci_lower": ci_l,
                "ci_upper": ci_u,
            }

        return metrics

    def evaluate_per_worker(
        self, dataset: List[Dict[str, Any]], results: List[Dict[str, Any]]
    ) -> Dict[str, dict]:
        """Group by expected_agents and compute per-worker metrics."""
        worker_groups = {}
        for entry, result in zip(dataset, results):
            agents = entry.get("expected_agents", ["unknown"])
            for agent in agents:
                if agent not in worker_groups:
                    worker_groups[agent] = {"dataset": [], "results": []}
                worker_groups[agent]["dataset"].append(entry)
                worker_groups[agent]["results"].append(result)

        worker_metrics = {}
        for agent, data in worker_groups.items():
            worker_metrics[agent] = self.evaluate_batch_custom(
                data["dataset"], data["results"]
            )

        return worker_metrics

    def evaluate(
        self, dataset: List[Dict[str, Any]], results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Full evaluation: aggregate + per-worker + both frameworks.

        Sub-metrics are isolated so a judge-framework failure cannot blank out
        the deterministic custom/per-worker metrics alongside it.
        """
        from src.evaluation.utils import guarded

        return {
            "custom_metrics": guarded(
                "retrieval.custom", self.evaluate_batch_custom, dataset, results
            ),
            "ragas_metrics": guarded(
                "retrieval.ragas", self.evaluate_batch_ragas, dataset, results
            ),
            "deepeval_metrics": guarded(
                "retrieval.deepeval", self.evaluate_batch_deepeval, dataset, results
            ),
            "per_worker_metrics": guarded(
                "retrieval.per_worker", self.evaluate_per_worker, dataset, results
            ),
        }
