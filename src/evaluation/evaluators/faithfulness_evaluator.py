import math
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List

import scipy.stats

from src.evaluation.config import EVAL_WORKER_TIMEOUT
from src.evaluation.judge_concurrency import judge_call_slot

_DEEPEVAL_MAX_RETRIES = 3
_DEEPEVAL_TIMEOUT_S = EVAL_WORKER_TIMEOUT
_DEEPEVAL_PARALLELISM = 4
_MAX_EVAL_CONTEXT_DOC_CHARS = 20_000


class FaithfulnessEvaluator:
    def __init__(self):
        pass

    def evaluate_citation_accuracy(
        self, results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """OSHA-standard-number citation accuracy (the "regex-only" variant).

        Extraction and matching live in ``src.evaluation.citation_utils`` and are
        shared with ``DomainEvaluator``; see that module for the context-metadata
        and squashed-encoding bugs this used to have.
        """
        from src.evaluation.citation_utils import score_citation_accuracy

        return score_citation_accuracy(results, include_enforcement_ids=False)

    def evaluate_cross_framework(
        self, ragas_scores: Dict[str, Any], deepeval_scores: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Compute Pearson and Spearman correlation between frameworks
        Using scipy.stats
        """
        ragas_faith = ragas_scores.get("faithfulness_list", [])
        deep_faith = deepeval_scores.get("faithfulness_list", [])

        if not ragas_faith or not deep_faith or len(ragas_faith) != len(deep_faith):
            return {"pearson": 0.0, "spearman": 0.0}

        # RAGAS lists carry ``None`` in slots where the judge call failed (see
        # ragas_experiments). Drop those pairs rather than correlating against a
        # substituted zero, which would manufacture a spurious correlation.
        pairs = [
            (r, d)
            for r, d in zip(ragas_faith, deep_faith)
            if r is not None and d is not None
        ]
        if len(pairs) < 2:
            return {"pearson": 0.0, "spearman": 0.0, "n_pairs": len(pairs)}
        ragas_faith = [r for r, _ in pairs]
        deep_faith = [d for _, d in pairs]

        pearson_corr, _ = scipy.stats.pearsonr(ragas_faith, deep_faith)
        spearman_corr, _ = scipy.stats.spearmanr(ragas_faith, deep_faith)

        if math.isnan(pearson_corr):
            pearson_corr = 0.0
        if math.isnan(spearman_corr):
            spearman_corr = 0.0

        return {
            "pearson": float(pearson_corr),
            "spearman": float(spearman_corr),
            "n_pairs": len(pairs),
        }

    def evaluate_ragas(
        self, dataset: List[Dict[str, Any]], results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Compute RAGAS v0.4 faithfulness / answer_correctness / factual_correctness.

        Metric construction is inside the ``try`` on purpose: a constructor that
        raises used to escape this method and let ``end_to_end_evaluator``'s
        catch-all blank out the *entire* enclosing metrics section (that is how
        the 2026-07-31 run lost all of Layer 3). Nothing in here may propagate.
        """
        from src.evaluation.config import EVAL_ENABLE_RAGAS

        if not EVAL_ENABLE_RAGAS:
            return {"skipped": "ragas disabled for current judge model configuration"}

        try:
            from ragas.metrics.collections import AnswerCorrectness, Faithfulness

            try:
                from ragas.metrics.collections import FactualCorrectness
            except ImportError:
                FactualCorrectness = None
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
                    "response": result.get("final_response", ""),
                    "retrieved_contexts": [
                        doc.get("text", "")[:_MAX_EVAL_CONTEXT_DOC_CHARS]
                        for doc in result.get("context_gathered", [])
                    ],
                    "reference": entry.get("expected_answer", gt_text),
                }
            )

        try:
            from src.evaluation.judge_factory import (
                make_ragas_embeddings,
                make_ragas_judge_llm,
            )

            llm = make_ragas_judge_llm()
            embeddings = make_ragas_embeddings()
            faithfulness_metric = Faithfulness(llm=llm)
            answer_correctness_metric = AnswerCorrectness(
                llm=llm, embeddings=embeddings
            )
            factual_correctness_metric = (
                FactualCorrectness(llm=llm) if FactualCorrectness else None
            )
        except Exception as exc:  # noqa: BLE001
            return {"error": f"ragas judge/metric setup failed: {exc}"}

        scorer = MetricScorer()
        metric_names = ["faithfulness", "answer_correctness"]
        if factual_correctness_metric is not None:
            metric_names.append("factual_correctness")

        import asyncio

        async def score_row(row: Dict[str, Any]) -> Dict[str, Any]:
            f_task = scorer.score(
                "faithfulness",
                faithfulness_metric,
                user_input=row["user_input"],
                response=row["response"],
                retrieved_contexts=row["retrieved_contexts"],
            )
            ac_task = scorer.score(
                "answer_correctness",
                answer_correctness_metric,
                user_input=row["user_input"],
                response=row["response"],
                reference=row["reference"],
            )
            tasks = [f_task, ac_task]
            if factual_correctness_metric is not None:
                tasks.append(
                    scorer.score(
                        "factual_correctness",
                        factual_correctness_metric,
                        response=row["response"],
                        reference=row["reference"],
                    )
                )

            res = await asyncio.gather(*tasks)
            scores: Dict[str, Any] = {
                "faithfulness": res[0],
                "answer_correctness": res[1],
            }
            if factual_correctness_metric is not None:
                scores["factual_correctness"] = res[2]
            return scores

        try:
            records = score_rows(
                score_row,
                rows,
                max_concurrency=EVAL_MAX_WORKERS,
                timeout_s=EVAL_WORKER_TIMEOUT,
            )
        except Exception as exc:  # noqa: BLE001
            return {"error": f"ragas execution failed: {exc}"}

        return aggregate_metric_records(records, metric_names, errors=scorer.errors)

    def evaluate_deepeval(
        self, dataset: List[Dict[str, Any]], results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Compute FaithfulnessMetric, HallucinationMetric, GEval(correctness)"""
        try:
            from deepeval.metrics import FaithfulnessMetric, GEval, HallucinationMetric
            from deepeval.test_case import LLMTestCase, LLMTestCaseParams

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
        ) -> tuple[Any, Any, Any]:
            retrieved_contexts = [
                doc.get("text", "") for doc in result.get("context_gathered", [])
            ]
            test_case = LLMTestCase(
                input=entry.get("query", ""),
                actual_output=result.get("final_response", ""),
                retrieval_context=retrieved_contexts,
                context=retrieved_contexts,
                expected_output=entry.get("expected_answer", ""),
            )

            judge_model = make_deepeval_judge_model()
            faithfulness_metric = _build_metric(
                FaithfulnessMetric, model=judge_model, threshold=0.9
            )
            hallucination_metric = _build_metric(
                HallucinationMetric, model=judge_model, threshold=0.05
            )
            correctness_metric = _build_metric(
                GEval,
                name="Correctness",
                criteria="Determine whether the actual output is factually correct based on the expected output.",
                evaluation_params=[
                    LLMTestCaseParams.INPUT,
                    LLMTestCaseParams.ACTUAL_OUTPUT,
                    LLMTestCaseParams.EXPECTED_OUTPUT,
                ],
                model=judge_model,
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

            # Intra-entry parallel metric measurement
            with ThreadPoolExecutor(max_workers=3) as metric_executor:
                f_fut = metric_executor.submit(
                    _measure_with_retries, faithfulness_metric
                )
                h_fut = metric_executor.submit(
                    _measure_with_retries, hallucination_metric
                )
                c_fut = metric_executor.submit(
                    _measure_with_retries, correctness_metric
                )
                f_ok = f_fut.result()
                h_ok = h_fut.result()
                c_ok = c_fut.result()

            faithfulness = faithfulness_metric.score if f_ok else None
            hallucination = hallucination_metric.score if h_ok else None
            correctness = correctness_metric.score if c_ok else None
            return faithfulness, hallucination, correctness

        f_scores: List[float] = []
        h_scores: List[float] = []
        c_scores: List[float] = []
        with ThreadPoolExecutor(max_workers=_DEEPEVAL_PARALLELISM) as executor:
            for faithfulness, hallucination, correctness in executor.map(
                lambda pair: _evaluate_entry(pair[0], pair[1]), zip(dataset, results)
            ):
                if faithfulness is not None:
                    f_scores.append(faithfulness)
                if hallucination is not None:
                    h_scores.append(hallucination)
                if correctness is not None:
                    c_scores.append(correctness)

        from src.evaluation.utils import bootstrap_ci

        metrics = {"faithfulness_list": f_scores, "hallucination_list": h_scores}

        if f_scores:
            mean_val = sum(f_scores) / len(f_scores)
            ci_low, ci_high = (
                bootstrap_ci(f_scores) if len(f_scores) > 1 else (mean_val, mean_val)
            )
            metrics["faithfulness"] = {
                "mean": mean_val,
                "ci_lower": ci_low,
                "ci_upper": ci_high,
            }

        if h_scores:
            mean_val = sum(h_scores) / len(h_scores)
            ci_low, ci_high = (
                bootstrap_ci(h_scores) if len(h_scores) > 1 else (mean_val, mean_val)
            )
            metrics["hallucination"] = {
                "mean": mean_val,
                "ci_lower": ci_low,
                "ci_upper": ci_high,
            }

        if c_scores:
            mean_val = sum(c_scores) / len(c_scores)
            ci_low, ci_high = (
                bootstrap_ci(c_scores) if len(c_scores) > 1 else (mean_val, mean_val)
            )
            metrics["correctness"] = {
                "mean": mean_val,
                "ci_lower": ci_low,
                "ci_upper": ci_high,
            }

        return metrics

    def evaluate(
        self, dataset: List[Dict[str, Any]], results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Full evaluation: RAGAS + DeepEval + citation accuracy + cross-framework.

        Each sub-metric is isolated so one failing framework cannot blank out
        the others (see ``src.evaluation.utils.guarded``).
        """
        from src.evaluation.utils import guarded

        with ThreadPoolExecutor(max_workers=2) as executor:
            ragas_future = executor.submit(
                guarded, "faithfulness.ragas", self.evaluate_ragas, dataset, results
            )
            deepeval_future = executor.submit(
                guarded,
                "faithfulness.deepeval",
                self.evaluate_deepeval,
                dataset,
                results,
            )
            ragas_scores = ragas_future.result()
            deepeval_scores = deepeval_future.result()

        return {
            "citation_accuracy": guarded(
                "faithfulness.citation_accuracy",
                self.evaluate_citation_accuracy,
                results,
            ),
            "ragas_metrics": ragas_scores,
            "deepeval_metrics": deepeval_scores,
            "cross_framework": guarded(
                "faithfulness.cross_framework",
                self.evaluate_cross_framework,
                ragas_scores,
                deepeval_scores,
            ),
        }
