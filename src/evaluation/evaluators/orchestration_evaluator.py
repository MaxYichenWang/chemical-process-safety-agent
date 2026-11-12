import math
from typing import Any, Dict, List

from src.evaluation.judge_concurrency import judge_call_slot


def _normalize_agent_name(name: str) -> str:
    """Map legacy agent names to canonical form.

    Canonical name is ``csb_recs`` (matches pipeline output and eval dataset
    after P2 rename).  Legacy references to ``csb_recommendations`` are
    normalised so comparisons still work if any slip through.
    """
    aliases = {
        "csb_recommendations": "csb_recs",
    }
    return aliases.get(name, name)


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _percentile_nearest_rank(values: List[float], percentile: float) -> float:
    if not values:
        return 0.0
    if percentile <= 0:
        return min(values)
    if percentile >= 100:
        return max(values)

    sorted_vals = sorted(values)
    k = math.ceil((percentile / 100.0) * len(sorted_vals))
    k = min(max(k, 1), len(sorted_vals))
    return sorted_vals[k - 1]


def _is_adversarial(entry: Dict[str, Any]) -> bool:
    """Return True if entry is an adversarial query (R3: excluded from routing metrics)."""
    cat = entry.get("category", "")
    src = entry.get("source", "")
    return "adversarial" in cat or "adversarial" in src


class OrchestrationEvaluator:
    def __init__(self):
        pass

    def evaluate_tool_call_accuracy(
        self, dataset: List[Dict[str, Any]], results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Compare result["agents_called"] against entry["expected_agents"].

        Skips pipeline-error stubs and adversarial entries (R3: adversarial
        expected_agents are invalid placeholders, not real agent names).
        """
        exact_matches = 0
        jaccard_scores = []
        exact_match_flags: list[float] = []
        evaluated = 0

        for entry, result in zip(dataset, results):
            if "error" in result:
                continue
            if _is_adversarial(entry):
                continue
            evaluated += 1
            expected = {
                _normalize_agent_name(a) for a in entry.get("expected_agents", [])
            }
            called = {_normalize_agent_name(a) for a in result.get("agents_called", [])}

            if not expected and not called:
                exact_matches += 1
                jaccard_scores.append(1.0)
                exact_match_flags.append(1.0)
                continue
            if not expected or not called:
                jaccard_scores.append(0.0)
                exact_match_flags.append(0.0)
                continue

            if expected == called:
                exact_matches += 1
                exact_match_flags.append(1.0)
            else:
                exact_match_flags.append(0.0)

            intersection = len(expected.intersection(called))
            union = len(expected.union(called))
            jaccard_scores.append(intersection / union)

        total = evaluated
        if total == 0:
            # No applicable entries (every entry was a pipeline error or
            # adversarial) is not "0% exact match" -- those render identically
            # as a numeric 0.0, which is exactly the failure mode this guards
            # against.
            return {"error": "no applicable entries (n=0)", "n": 0}

        exact_match_rate = exact_matches / total

        from src.evaluation.utils import bootstrap_ci

        j_mean = sum(jaccard_scores) / len(jaccard_scores)
        j_l, j_u = (
            bootstrap_ci(jaccard_scores)
            if len(jaccard_scores) > 1
            else (j_mean, j_mean)
        )
        # CI + per-query flags on the exact-match rate (L3), so the
        # headline routing metric carries a 95% CI and can feed the paired
        # significance test.
        e_l, e_u = (
            bootstrap_ci(exact_match_flags)
            if len(exact_match_flags) > 1
            else (exact_match_rate, exact_match_rate)
        )

        return {
            "exact_match_rate": exact_match_rate,
            "exact_match_ci_lower": e_l,
            "exact_match_ci_upper": e_u,
            "exact_match_rate_list": exact_match_flags,
            # Denominator for "Tool Call Exact Match 90.2% (193/214)".
            "n": total,
            "jaccard": {"mean": j_mean, "ci_lower": j_l, "ci_upper": j_u},
        }

    def evaluate_goal_accuracy_ragas(
        self, dataset: List[Dict[str, Any]], results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        return {
            "skipped": (
                "RAGAS AgentGoalAccuracy is not migrated to the v0.4 collections "
                "experiment API; using deepeval_goal_accuracy instead"
            )
        }

    def evaluate_topic_adherence(
        self, dataset: List[Dict[str, Any]], results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Answer-relevancy proxy for on-topic-ness, reported as ``topic_adherence``.

        The underlying metric is ragas ``AnswerRelevancy``; the key name is kept
        for continuity with earlier reports. Every failure path returns a dict --
        an exception escaping here previously wiped the whole orchestration
        section (2026-07-31 run), taking pipeline error rate, routing accuracy
        and cost down with it.
        """
        from src.evaluation.config import EVAL_ENABLE_RAGAS

        if not EVAL_ENABLE_RAGAS:
            return {"skipped": "ragas disabled for current judge model configuration"}

        try:
            from ragas.metrics.collections import AnswerRelevancy
        except (ImportError, TypeError):
            return {"error": "ragas not installed or metrics missing"}

        from src.evaluation.config import (
            EVAL_LLM_TEMPERATURE,
            EVAL_MAX_WORKERS,
            EVAL_WORKER_TIMEOUT,
        )
        from src.evaluation.ragas_experiments import (
            MetricScorer,
            aggregate_metric_records,
            score_rows,
        )

        rows = [
            {
                "user_input": entry.get("query", ""),
                "response": result.get("final_response", ""),
            }
            for entry, result in zip(dataset, results)
        ]

        try:
            from src.evaluation.judge_factory import (
                make_ragas_embeddings,
                make_ragas_judge_llm,
            )

            metric = AnswerRelevancy(
                llm=make_ragas_judge_llm(temperature=EVAL_LLM_TEMPERATURE),
                embeddings=make_ragas_embeddings(),
            )
        except Exception as exc:  # noqa: BLE001
            return {"error": f"ragas judge/metric setup failed: {exc}"}

        scorer = MetricScorer()

        async def score_row(row: Dict[str, Any]) -> Dict[str, Any]:
            return {
                "topic_adherence": await scorer.score(
                    "topic_adherence",
                    metric,
                    user_input=row["user_input"],
                    response=row["response"],
                )
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

        output = aggregate_metric_records(
            records, ["topic_adherence"], errors=scorer.errors
        )
        output.setdefault("metric", "ragas.AnswerRelevancy")
        return output

    def evaluate_goal_accuracy_deepeval(
        self, dataset: List[Dict[str, Any]], results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Use DeepEval TaskCompletionMetric"""
        try:
            from deepeval.metrics import TaskCompletionMetric
            from deepeval.test_case import LLMTestCase

            from src.evaluation.judge_factory import make_deepeval_judge_model
        except (ImportError, TypeError):
            return {"error": "deepeval not installed"}

        try:
            metric = TaskCompletionMetric(
                model=make_deepeval_judge_model(), threshold=0.7
            )
        except Exception:
            return {"error": "TaskCompletionMetric unsupported"}

        scores = []
        for entry, result in zip(dataset, results):
            test_case = LLMTestCase(
                input=entry.get("query", ""),
                actual_output=result.get("final_response", ""),
                expected_output=entry.get("expected_answer", ""),
            )
            try:
                with judge_call_slot():
                    metric.measure(test_case)
                scores.append(metric.score)
            except Exception:
                pass

        if not scores:
            return {"mean": 0.0}

        from src.evaluation.utils import bootstrap_ci

        mean_val = sum(scores) / len(scores)
        ci_low, ci_high = (
            bootstrap_ci(scores) if len(scores) > 1 else (mean_val, mean_val)
        )
        return {"mean": mean_val, "ci_lower": ci_low, "ci_upper": ci_high}

    def evaluate_critic_distribution(
        self, results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze validation_score from AgentState.

        Skips pipeline-error stubs (results with an ``"error"`` key) so a
        forced ``validation_score=0.0`` from a failed run does not pull
        down the distribution.
        """
        scores = [
            res.get("validation_score", 0.0)
            for res in results
            if "validation_score" in res and "error" not in res
        ]

        if not scores:
            return {"mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0}

        scores.sort()
        n = len(scores)
        mean_val = sum(scores) / n
        median_val = (
            scores[n // 2]
            if n % 2 != 0
            else (scores[n // 2 - 1] + scores[n // 2]) / 2.0
        )

        bins = {"0.0-0.2": 0, "0.2-0.4": 0, "0.4-0.6": 0, "0.6-0.8": 0, "0.8-1.0": 0}
        for s in scores:
            if s < 0.2:
                bins["0.0-0.2"] += 1
            elif s < 0.4:
                bins["0.2-0.4"] += 1
            elif s < 0.6:
                bins["0.4-0.6"] += 1
            elif s < 0.8:
                bins["0.6-0.8"] += 1
            else:
                bins["0.8-1.0"] += 1

        return {
            "mean": mean_val,
            "median": median_val,
            "min": scores[0],
            "max": scores[-1],
            "histogram": bins,
        }

    def evaluate_retry_rate(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute: count(retry_count > 0) / total_queries.

        Errored pipeline stubs are excluded so a failed run (forced to
        ``retry_count=0``) does not artificially deflate the retry rate.
        """
        clean = [res for res in results if "error" not in res]
        if not clean:
            return {"retry_rate": 0.0, "mean_retries": 0.0}

        retries = [res.get("retry_count", 0) for res in clean]
        any_retries = sum(1 for r in retries if r > 0)

        return {
            "retry_rate": any_retries / len(clean),
            "mean_retries": sum(retries) / len(clean),
        }

    def evaluate_pipeline_error_rate(
        self, results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Fraction of queries where the pipeline raised a non-retryable error.

        Errored results are stubbed by ``EndToEndEvaluator.run_pipeline_async``
        with ``{"error": ..., "validation_score": 0.0, ...}``; this metric is
        the only place such stubs are counted.
        """
        if not results:
            return {"error_rate": 0.0, "error_count": 0, "total": 0, "errors": []}

        errored = [res for res in results if "error" in res]
        return {
            "error_rate": len(errored) / len(results),
            "error_count": len(errored),
            "total": len(results),
            # Truncated stack-message previews for quick triage in the report
            "errors": [str(res.get("error", ""))[:160] for res in errored][:20],
        }

    def evaluate_cost(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate LLM-call cost and usage from result['llm_calls']."""
        if not results:
            return {
                "total_usd": 0.0,
                "mean_per_query_usd": 0.0,
                "p95_per_query_usd": 0.0,
                "mean_calls_per_query": 0.0,
                "total_prompt_tokens": 0,
                "total_completion_tokens": 0,
                "breakdown_by_node": {},
                "breakdown_by_purpose": {},
            }

        per_query_costs: List[float] = []
        per_query_calls: List[int] = []
        total_prompt_tokens = 0
        total_completion_tokens = 0
        breakdown_by_node: Dict[str, Dict[str, Any]] = {}
        breakdown_by_purpose: Dict[str, Dict[str, Any]] = {}

        for result in results:
            calls = result.get("llm_calls", []) or []
            query_cost = 0.0
            per_query_calls.append(len(calls))

            for call in calls:
                call_cost = _safe_float(call.get("estimated_cost_usd", 0.0))
                query_cost += call_cost

                p_tokens = int(_safe_float(call.get("prompt_tokens", 0)))
                c_tokens = int(_safe_float(call.get("completion_tokens", 0)))
                total_prompt_tokens += p_tokens
                total_completion_tokens += c_tokens

                node = str(call.get("node_name", "unknown") or "unknown")
                purpose = str(call.get("call_purpose", "unknown") or "unknown")
                latency = _safe_float(call.get("latency_s", 0.0))

                node_entry = breakdown_by_node.setdefault(
                    node,
                    {
                        "calls": 0,
                        "cost_usd": 0.0,
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "latency_s_total": 0.0,
                    },
                )
                node_entry["calls"] += 1
                node_entry["cost_usd"] += call_cost
                node_entry["prompt_tokens"] += p_tokens
                node_entry["completion_tokens"] += c_tokens
                node_entry["latency_s_total"] += latency

                purpose_entry = breakdown_by_purpose.setdefault(
                    purpose,
                    {
                        "calls": 0,
                        "cost_usd": 0.0,
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "latency_s_total": 0.0,
                    },
                )
                purpose_entry["calls"] += 1
                purpose_entry["cost_usd"] += call_cost
                purpose_entry["prompt_tokens"] += p_tokens
                purpose_entry["completion_tokens"] += c_tokens
                purpose_entry["latency_s_total"] += latency

            per_query_costs.append(query_cost)

        total_usd = sum(per_query_costs)
        mean_per_query = total_usd / len(results)
        p95_per_query = _percentile_nearest_rank(per_query_costs, 95.0)
        mean_calls = sum(per_query_calls) / len(results)

        for bucket in (breakdown_by_node, breakdown_by_purpose):
            for key, data in bucket.items():
                calls = data["calls"]
                data["cost_usd"] = round(float(data["cost_usd"]), 8)
                data["latency_s_mean"] = (
                    data["latency_s_total"] / calls if calls > 0 else 0.0
                )
                data["latency_s_total"] = round(float(data["latency_s_total"]), 4)

        return {
            "total_usd": round(total_usd, 8),
            "mean_per_query_usd": round(mean_per_query, 8),
            "p95_per_query_usd": round(p95_per_query, 8),
            "mean_calls_per_query": round(mean_calls, 4),
            "total_prompt_tokens": total_prompt_tokens,
            "total_completion_tokens": total_completion_tokens,
            "breakdown_by_node": breakdown_by_node,
            "breakdown_by_purpose": breakdown_by_purpose,
        }

    def evaluate_plan_accuracy(
        self, dataset: List[Dict[str, Any]], results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Compare planned_agents against expected_agents (planner-routing quality).

        Skips adversarial entries (R3: invalid expected_agents).
        """
        exact_matches = 0
        jaccard_scores = []
        evaluated = 0

        for entry, result in zip(dataset, results):
            if "error" in result:
                continue
            if _is_adversarial(entry):
                continue

            expected = {
                _normalize_agent_name(a) for a in entry.get("expected_agents", [])
            }
            planned = {
                _normalize_agent_name(a) for a in result.get("planned_agents", [])
            }
            evaluated += 1

            if not expected and not planned:
                exact_matches += 1
                jaccard_scores.append(1.0)
                continue
            if not expected or not planned:
                jaccard_scores.append(0.0)
                continue

            if expected == planned:
                exact_matches += 1

            intersection = len(expected.intersection(planned))
            union = len(expected.union(planned))
            jaccard_scores.append(intersection / union)

        if not jaccard_scores:
            return {
                "exact_match_rate": 0.0,
                "jaccard": {"mean": 0.0, "ci_lower": 0.0, "ci_upper": 0.0},
                "n": 0,
            }

        from src.evaluation.utils import bootstrap_ci

        mean_val = sum(jaccard_scores) / len(jaccard_scores)
        ci_lower, ci_upper = (
            bootstrap_ci(jaccard_scores)
            if len(jaccard_scores) > 1
            else (mean_val, mean_val)
        )

        return {
            "exact_match_rate": exact_matches / evaluated if evaluated > 0 else 0.0,
            "jaccard": {"mean": mean_val, "ci_lower": ci_lower, "ci_upper": ci_upper},
            "n": evaluated,
        }

    def evaluate_fast_route_coverage(
        self, dataset: List[Dict[str, Any]], results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Measure fast-route usage and planner overlap quality by routing path."""
        clean_pairs = [
            (entry, result)
            for entry, result in zip(dataset, results)
            if "error" not in result
        ]
        total = len(clean_pairs)
        if total == 0:
            return {
                "fast_route_rate": 0.0,
                "fast_route_count": 0,
                "llm_plan_count": 0,
                "fast_route_plan_jaccard_mean": 0.0,
                "llm_plan_jaccard_mean": 0.0,
                "fast_route_pattern_counts": {},
            }

        fast_pairs = []
        llm_pairs = []
        pattern_counts: Dict[str, int] = {}

        for entry, result in clean_pairs:
            is_fast = bool(result.get("planning_fast_routed", False))
            if is_fast:
                fast_pairs.append((entry, result))
                for pattern in result.get("planning_fast_route_patterns", []) or []:
                    key = str(pattern)
                    pattern_counts[key] = pattern_counts.get(key, 0) + 1
            else:
                llm_pairs.append((entry, result))

        def mean_jaccard(pairs: List[Any]) -> float:
            if not pairs:
                return 0.0
            scores = []
            for entry, result in pairs:
                expected = {
                    _normalize_agent_name(a) for a in entry.get("expected_agents", [])
                }
                planned = {
                    _normalize_agent_name(a) for a in result.get("planned_agents", [])
                }
                if not expected and not planned:
                    scores.append(1.0)
                    continue
                if not expected or not planned:
                    scores.append(0.0)
                    continue
                inter = len(expected.intersection(planned))
                union = len(expected.union(planned))
                scores.append(inter / union)
            return sum(scores) / len(scores) if scores else 0.0

        return {
            "fast_route_rate": len(fast_pairs) / total,
            "fast_route_count": len(fast_pairs),
            "llm_plan_count": len(llm_pairs),
            "fast_route_plan_jaccard_mean": mean_jaccard(fast_pairs),
            "llm_plan_jaccard_mean": mean_jaccard(llm_pairs),
            "fast_route_pattern_counts": pattern_counts,
        }

    def evaluate(
        self, dataset: List[Dict[str, Any]], results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Full orchestration evaluation.

        Receives full ``dataset`` and ``results`` (including pipeline-error
        stubs) so ``pipeline_error_rate`` can be computed before downstream
        evaluators see only the cleaned subset.

        Every sub-metric is isolated: the deterministic ones here (error rate,
        routing accuracy, fast-route coverage, cost) are the section's most
        important output and must survive a judge-framework failure.
        """
        from src.evaluation.utils import guarded

        return {
            "pipeline_error_rate": guarded(
                "orchestration.pipeline_error_rate",
                self.evaluate_pipeline_error_rate,
                results,
            ),
            "tool_call_accuracy": guarded(
                "orchestration.tool_call_accuracy",
                self.evaluate_tool_call_accuracy,
                dataset,
                results,
            ),
            "plan_accuracy": guarded(
                "orchestration.plan_accuracy",
                self.evaluate_plan_accuracy,
                dataset,
                results,
            ),
            "fast_route_coverage": guarded(
                "orchestration.fast_route_coverage",
                self.evaluate_fast_route_coverage,
                dataset,
                results,
            ),
            "cost": guarded("orchestration.cost", self.evaluate_cost, results),
            "critic_distribution": guarded(
                "orchestration.critic_distribution",
                self.evaluate_critic_distribution,
                results,
            ),
            "retry_rate": guarded(
                "orchestration.retry_rate", self.evaluate_retry_rate, results
            ),
            "ragas_goal_accuracy": guarded(
                "orchestration.ragas_goal_accuracy",
                self.evaluate_goal_accuracy_ragas,
                dataset,
                results,
            ),
            "ragas_topic_adherence": guarded(
                "orchestration.ragas_topic_adherence",
                self.evaluate_topic_adherence,
                dataset,
                results,
            ),
            "deepeval_goal_accuracy": guarded(
                "orchestration.deepeval_goal_accuracy",
                self.evaluate_goal_accuracy_deepeval,
                dataset,
                results,
            ),
        }
