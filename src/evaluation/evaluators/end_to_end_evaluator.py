import asyncio
import json
import logging
import random
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
)

from src.agentic_rag.graph import build_graph
from src.evaluation.config import (
    ADVERSARIAL_DATASET_FILE,
    CHROMADB_DIR,
    EVAL_DATASET_FILE,
    EVAL_MAX_CONCURRENT,
    EVAL_RESULTS_DIR,
)
from src.evaluation.reporting.report_generator import (
    generate_json_results,
    generate_report,
)
from src.evaluation.utils import (
    configure_evaluation_file_logging,
    default_eval_log_path,
    finalize_pipeline_results,
    generate_run_id,
    load_eval_dataset,
    metric_failure_paths,
    normalize_graph_state_for_eval,
    validate_ground_truth_chunks,
)

from .analytical_evaluator import AnalyticalEvaluator
from .domain_evaluator import DomainEvaluator
from .faithfulness_evaluator import FaithfulnessEvaluator
from .orchestration_evaluator import OrchestrationEvaluator
from .retrieval_evaluator import RetrievalEvaluator

logger = logging.getLogger(__name__)

warnings.filterwarnings(
    "ignore",
    message=r".*Mixing V1 models and V2 models.*BasePromptTemplate.*",
    category=UserWarning,
)


def should_retry(exception):
    err_str = str(exception).lower()
    return (
        "429" in err_str
        or "rate limit" in err_str
        or "ratelimit" in err_str
        or "timeout" in err_str
    )


class EndToEndEvaluator:
    def __init__(self, chromadb_dir: Path = CHROMADB_DIR, system: str = "full"):
        self.chromadb_dir = chromadb_dir
        self._last_fail_fast_error: Optional[str] = None
        self.system = system

        # Instantiate the runnable system: the full multi-agent graph by
        # default, or one of the Stage-E comparator baselines (vanilla /
        # single_agent / no_retrieval). All expose ``.ainvoke({"query": ...})``.
        if system == "full":
            self.graph = build_graph()
        else:
            from src.agentic_rag.systems import build_system

            self.graph = build_system(system)

        # Instantiate evaluators
        self.retrieval_eval = RetrievalEvaluator(chromadb_dir)
        self.faithfulness_eval = FaithfulnessEvaluator()
        self.orchestration_eval = OrchestrationEvaluator()
        self.domain_eval = DomainEvaluator()
        self.analytical_eval = AnalyticalEvaluator()

    @staticmethod
    def _write_partial_checkpoint(
        run_id: str, results: List[Dict[str, Any]], total_queries: int
    ) -> Path:
        path = EVAL_RESULTS_DIR / f"{run_id}_partial.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "run_id": run_id,
            "processed_queries": len(results),
            "total_queries": total_queries,
            "pipeline_results": results,
        }
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, default=str)
        return path

    @staticmethod
    def _aggregate_llm_call_stats(llm_calls: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Sum the per-call token/cost figures already recorded by
        ``llm_logger.record_llm_call`` (``prompt_tokens``/``completion_tokens``/
        ``total_tokens``/``estimated_cost_usd``) into a per-query total, so
        latency and cost tables don't need to re-walk ``llm_calls`` in
        every downstream consumer. Missing/``None`` per-call fields (a model
        that didn't return usage metadata) count as 0, not a fatal error."""
        return {
            "n_llm_calls": len(llm_calls),
            "prompt_tokens": sum(c.get("prompt_tokens") or 0 for c in llm_calls),
            "completion_tokens": sum(
                c.get("completion_tokens") or 0 for c in llm_calls
            ),
            "total_tokens": sum(c.get("total_tokens") or 0 for c in llm_calls),
            "total_cost_usd": round(
                sum(c.get("estimated_cost_usd") or 0.0 for c in llm_calls), 8
            ),
        }

    @retry(
        wait=wait_exponential(multiplier=2, min=4, max=60),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    async def run_pipeline_async(
        self, query: str, semaphore: asyncio.Semaphore
    ) -> Dict[str, Any]:
        """Runs the LangGraph Agentic logic, wrapped in tenacity for safe concurrency."""
        async with semaphore:
            logger.info(f"Running pipeline for query: {query[:50]}...")
            wall_start = time.monotonic()
            try:
                final_state = await self.graph.ainvoke({"query": query})
                wall_seconds = time.monotonic() - wall_start

                context_flat, agents_called = normalize_graph_state_for_eval(
                    final_state
                )
                llm_calls = final_state.get("llm_calls", [])

                result = {
                    "query": query,
                    "final_state": final_state,
                    "context_gathered": context_flat,
                    "draft_response": final_state.get("draft_response", ""),
                    "final_response": final_state.get("final_response", ""),
                    "validation_score": final_state.get("validation_score", 0.0),
                    "critic_feedback": final_state.get("critic_feedback", ""),
                    "retry_count": final_state.get("retry_count", 0),
                    "citations": final_state.get("citations", []),
                    "agents_called": agents_called,
                    "retrieval_details": final_state.get("retrieval_details", []),
                    "worker_errors": final_state.get("worker_errors", []),
                    "llm_calls": llm_calls,
                    "planned_agents": final_state.get("planned_agents", []),
                    "planning_fast_routed": final_state.get(
                        "planning_fast_routed", False
                    ),
                    "planning_fast_route_patterns": final_state.get(
                        "planning_fast_route_patterns", []
                    ),
                    # Per-query wall-clock + token/cost totals.
                    "wall_seconds": round(wall_seconds, 4),
                    **self._aggregate_llm_call_stats(llm_calls),
                }
                return result
            except Exception as e:
                # Decide if we retry or fail outright
                if should_retry(e):
                    logger.warning(f"Rate limit / timeout hit, retrying: {e}")
                    raise e
                else:
                    wall_seconds = time.monotonic() - wall_start
                    logger.error(f"Pipeline failed for query: {query}. Error: {e}")
                    # Return mock dict so we don't crash whole batch
                    return {
                        "query": query,
                        "error": str(e),
                        "context_gathered": [],
                        "draft_response": "",
                        "final_response": "",
                        "validation_score": 0.0,
                        "critic_feedback": "",
                        "retry_count": 0,
                        "citations": [],
                        "agents_called": [],
                        # No final_state reached on a hard failure, so
                        # llm_calls/token totals are honestly 0 -- not hidden,
                        # just genuinely nothing to aggregate.
                        "wall_seconds": round(wall_seconds, 4),
                        **self._aggregate_llm_call_stats([]),
                    }

    async def run_batch_async(
        self,
        dataset: List[Dict[str, Any]],
        max_concurrent: int = EVAL_MAX_CONCURRENT,
        run_id: Optional[str] = None,
        checkpoint_every: int = 10,
    ) -> List[Dict[str, Any]]:
        semaphore = asyncio.Semaphore(max_concurrent)
        final_results: List[Dict[str, Any]] = []
        fail_fast_error: Optional[str] = None

        consecutive_error_count = 0
        last_error_message: Optional[str] = None
        fail_fast_after = 3

        for i in range(0, len(dataset), max_concurrent):
            batch = dataset[i : i + max_concurrent]
            tasks = [
                self.run_pipeline_async(entry.get("query", ""), semaphore)
                for entry in batch
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for entry, res in zip(batch, results):
                if isinstance(res, Exception):
                    logger.error(f"Unhandled exception in batch: {res}")
                    normalized = str(res).strip()
                    stub = {
                        "query": entry.get("query", ""),
                        "error": normalized,
                        "context_gathered": [],
                        "final_response": "",
                        "validation_score": 0.0,
                        "retry_count": 0,
                        "agents_called": [],
                    }
                    final_results.append(stub)
                    current_error = normalized
                else:
                    final_results.append(res)
                    current_error = (
                        str(res.get("error", "")).strip() if "error" in res else None
                    )

                if current_error:
                    if current_error == last_error_message:
                        consecutive_error_count += 1
                    else:
                        last_error_message = current_error
                        consecutive_error_count = 1
                else:
                    consecutive_error_count = 0
                    last_error_message = None

                if consecutive_error_count >= fail_fast_after and not fail_fast_error:
                    fail_fast_error = (
                        "Fail-fast triggered after 3 consecutive identical pipeline errors: "
                        f"{last_error_message}"
                    )
                    # gather() has already completed this entire batch. Keep
                    # every paid result before deciding whether to launch more.

                if (
                    run_id
                    and checkpoint_every > 0
                    and len(final_results) % checkpoint_every == 0
                ):
                    self._write_partial_checkpoint(run_id, final_results, len(dataset))

            if fail_fast_error:
                if len(final_results) < len(dataset):
                    logger.error(fail_fast_error)
                    break
                fail_fast_error = None

        self._last_fail_fast_error = fail_fast_error
        return final_results

    def run_batch(
        self,
        dataset: List[Dict[str, Any]],
        max_concurrent: int = EVAL_MAX_CONCURRENT,
        run_id: Optional[str] = None,
        checkpoint_every: int = 10,
    ) -> List[Dict[str, Any]]:
        """Synchronous wrapper for batch execution"""
        return asyncio.run(
            self.run_batch_async(dataset, max_concurrent, run_id, checkpoint_every)
        )

    def _evaluate_per_category(
        self, dataset: List[Dict[str, Any]], results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Compute per-category rollups using deterministic/custom metrics only."""
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
                    "custom_metrics": self.retrieval_eval.evaluate_batch_custom(
                        clean_dataset, clean_results
                    )
                },
                "faithfulness": {
                    "citation_accuracy": self.faithfulness_eval.evaluate_citation_accuracy(
                        clean_results
                    )
                },
                "orchestration": {
                    "pipeline_error_rate": self.orchestration_eval.evaluate_pipeline_error_rate(
                        cat_results
                    ),
                    "tool_call_accuracy": self.orchestration_eval.evaluate_tool_call_accuracy(
                        cat_dataset, cat_results
                    ),
                    "plan_accuracy": self.orchestration_eval.evaluate_plan_accuracy(
                        cat_dataset, cat_results
                    ),
                    "fast_route_coverage": self.orchestration_eval.evaluate_fast_route_coverage(
                        cat_dataset, cat_results
                    ),
                    "critic_distribution": self.orchestration_eval.evaluate_critic_distribution(
                        cat_results
                    ),
                    "retry_rate": self.orchestration_eval.evaluate_retry_rate(
                        cat_results
                    ),
                    "cost": self.orchestration_eval.evaluate_cost(cat_results),
                },
                "domain": {
                    "citation_accuracy": self.domain_eval.evaluate_citation_accuracy(
                        clean_results
                    ),
                    "incident_fidelity": self.domain_eval.evaluate_incident_fidelity(
                        clean_results
                    ),
                    "hedging_detection": self.domain_eval.evaluate_hedging_detection(
                        clean_dataset, clean_results
                    ),
                },
                "analytical": self.analytical_eval.evaluate(
                    clean_dataset, clean_results
                ),
            }

        return output

    def evaluate(
        self,
        dataset_path: Path = EVAL_DATASET_FILE,
        include_adversarial: bool = False,
        max_concurrent: int = EVAL_MAX_CONCURRENT,
        sample_size_per_source: int = None,
        save_logs: bool = False,
        log_file: Optional[Path] = None,
        _run_id: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Loads dataset, evaluates graphs, runs custom evaluators, generates reports.

        Args:
            save_logs: If True, attach a file handler so all INFO+ logs for this process
                are also written to disk (see ``log_file``).
            log_file: Destination log file path. Defaults to
                ``data/evaluation/results/logs/{run_id}.log`` when ``save_logs`` is True.
            _run_id: Pre-generated run identifier. When provided (e.g. from ``__main__``
                so the log file and run artefacts share the same ID), it is used as-is;
                otherwise a fresh ID is generated here.
        """
        run_id = _run_id or generate_run_id()
        run_start = time.monotonic()
        log_path: Optional[Path] = None
        if save_logs:
            log_path = (
                Path(log_file)
                if log_file is not None
                else default_eval_log_path(run_id)
            )
            configure_evaluation_file_logging(log_path)
            logger.info("File logging enabled: %s", log_path)

        logger.info(
            "Run config | run_id=%s dataset=%s include_adversarial=%s "
            "max_concurrent=%s sample_per_source=%s",
            run_id,
            dataset_path,
            include_adversarial,
            max_concurrent,
            sample_size_per_source,
        )

        logger.info(f"Loading dataset from {dataset_path}")
        dataset = load_eval_dataset(dataset_path)

        if include_adversarial:
            try:
                adv_dataset = load_eval_dataset(ADVERSARIAL_DATASET_FILE)
                dataset.extend(adv_dataset)
                logger.info(f"Added {len(adv_dataset)} adversarial entries.")
            except Exception as e:
                logger.warning(f"Could not load adversarial dataset: {e}")

        if sample_size_per_source is not None:
            full_dataset = list(dataset)
            grouped = {}
            for entry in dataset:
                s = entry.get("source", "unknown")
                grouped.setdefault(s, []).append(entry)

            from src.evaluation.config import GENERATOR_SEED

            rng = random.Random(seed if seed is not None else GENERATOR_SEED)

            sampled_dataset = []
            for s, items in grouped.items():
                if len(items) > sample_size_per_source:
                    sampled_dataset.extend(rng.sample(items, sample_size_per_source))
                else:
                    sampled_dataset.extend(items)

            required_categories = {
                entry.get("category", "unknown") for entry in full_dataset
            }
            sampled_categories = {
                entry.get("category", "unknown") for entry in sampled_dataset
            }
            missing_categories = sorted(required_categories - sampled_categories)
            for category in missing_categories:
                candidates = [
                    entry
                    for entry in full_dataset
                    if entry.get("category", "unknown") == category
                    and entry not in sampled_dataset
                ]
                if candidates:
                    sampled_dataset.append(rng.choice(candidates))

            dataset = sampled_dataset
            logger.info(
                "Sampled dataset (seed=%s). Total entries reduced to %d "
                "(capped at %d per source).",
                seed if seed is not None else GENERATOR_SEED,
                len(dataset),
                sample_size_per_source,
            )

        if not dataset:
            raise ValueError("Dataset is empty; evaluation requires at least one entry")
        logger.info(
            f"Running pipeline on {len(dataset)} entries with max_concurrent={max_concurrent}..."
        )

        # Pre-eval ground-truth audit (non-blocking warn-only)
        validate_ground_truth_chunks(dataset, self.chromadb_dir)

        results = self.run_batch(
            dataset, max_concurrent, run_id=run_id, checkpoint_every=10
        )
        fail_fast_error = self._last_fail_fast_error
        if fail_fast_error:
            logger.error("Evaluation stopped early due to fail-fast guard.")
        results, completion = finalize_pipeline_results(
            dataset, results, stopped_early_error=fail_fast_error
        )

        # Partition results into "ok" (no pipeline error) and "errored" subsets.
        # Quality evaluators (retrieval / faithfulness / domain / analytical)
        # only see the ok subset so a forced ``validation_score=0.0`` stub
        # cannot deflate their distributions. The orchestration evaluator
        # receives the full lists because it computes pipeline_error_rate
        # from the errored count and its other metrics already gate on the
        # ``"error"`` key internally.
        ok_pairs = [(d, r) for d, r in zip(dataset, results) if "error" not in r]
        if ok_pairs:
            ok_dataset, ok_results = map(list, zip(*ok_pairs))
        else:
            ok_dataset, ok_results = [], []
        errored_count = len(results) - len(ok_results)
        if errored_count:
            logger.warning(
                "Excluding %d pipeline-error result(s) from quality evaluators "
                "(visible in orchestration.pipeline_error_rate).",
                errored_count,
            )

        logger.info("Running Evaluators in parallel...")
        eval_results: Dict[str, Any] = {}
        eval_tasks = [
            ("retrieval", lambda: self.retrieval_eval.evaluate(ok_dataset, ok_results)),
            (
                "faithfulness",
                lambda: self.faithfulness_eval.evaluate(ok_dataset, ok_results),
            ),
            (
                "orchestration",
                lambda: self.orchestration_eval.evaluate(dataset, results),
            ),
            ("domain", lambda: self.domain_eval.evaluate(ok_dataset, ok_results)),
            (
                "analytical",
                lambda: self.analytical_eval.evaluate(ok_dataset, ok_results),
            ),
        ]

        from concurrent.futures import ThreadPoolExecutor

        def _run_eval(eval_name: str, eval_fn: Any) -> tuple[str, Any]:
            try:
                return eval_name, eval_fn()
            except Exception as exc:
                logger.error(
                    f"{eval_name} evaluator raised an unhandled exception: {exc}",
                    exc_info=True,
                )
                return eval_name, {"error": str(exc)}

        with ThreadPoolExecutor(max_workers=len(eval_tasks)) as executor:
            futures = [executor.submit(_run_eval, name, fn) for name, fn in eval_tasks]
            for future in futures:
                name, res = future.result()
                eval_results[name] = res

        # Deterministic per-category rollups. Keep this separate from evaluator
        # calls to avoid re-running expensive judge-based metrics per slice.
        try:
            eval_results["per_category"] = self._evaluate_per_category(dataset, results)
        except Exception as exc:
            logger.error("per_category rollup failed: %s", exc, exc_info=True)
            eval_results["per_category"] = {"error": str(exc)}

        evaluator_errors = metric_failure_paths(eval_results)
        failure_reasons = []
        if completion.get("error"):
            failure_reasons.append(completion["error"])
        if evaluator_errors:
            failure_reasons.append(
                "Evaluation metrics failed: " + ", ".join(evaluator_errors)
            )
        run_status = {
            **completion,
            "evaluator_errors": evaluator_errors,
            "status": "failed" if failure_reasons else "complete",
        }
        if failure_reasons:
            run_status["error"] = "; ".join(failure_reasons)

        logger.info("Generating Report and JSON...")
        report_md = generate_report(eval_results, run_id)
        if failure_reasons:
            report_md = f"**Run failed:** {run_status['error']}\n\n" + report_md
        report_path = EVAL_RESULTS_DIR / f"{run_id}_report.md"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w") as f:
            f.write(report_md)

        json_path = generate_json_results(
            {
                "dataset": dataset, "pipeline_results": results,
                "metrics": eval_results, "completion": run_status,
            },
            run_id,
        )

        out: Dict[str, Any] = {
            "run_id": run_id,
            "metrics": eval_results,
            "report_path": str(report_path),
            "json_path": str(json_path),
            "dataset_size": len(dataset),
            "completion": run_status,
        }
        if failure_reasons:
            out["error"] = run_status["error"]
        if fail_fast_error or completion["unprocessed_entries"]:
            out["stopped_early"] = True
        if log_path is not None:
            out["log_path"] = str(log_path)
        partial_path = EVAL_RESULTS_DIR / f"{run_id}_partial.json"
        if partial_path.exists():
            out["partial_json_path"] = str(partial_path)

        elapsed = time.monotonic() - run_start
        pipeline_errors = sum(1 for r in results if "error" in r)
        logger.info(
            "=== Run complete | run_id=%s elapsed=%.1fs dataset_size=%d "
            "pipeline_errors=%d evaluator_errors=%s report=%s json=%s ===",
            run_id,
            elapsed,
            len(dataset),
            pipeline_errors,
            evaluator_errors or "none",
            report_path,
            json_path,
        )

        return out


def main() -> int:
    import argparse
    import multiprocessing
    import os
    import platform
    import sys
    import warnings

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    # macOS + Python 3.12 defaults to "fork" for multiprocessing, but forking
    # after SSL/asyncio initialisation causes segfaults on Apple Silicon (the
    # child inherits C-level state that isn't fork-safe).  ChromaDB's telemetry
    # layer (posthog) can trigger a fork; switching to "spawn" avoids the crash
    # and the leaked-semaphore warning from resource_tracker.
    try:
        multiprocessing.set_start_method("spawn")
    except RuntimeError:
        pass  # already set (e.g. inside a test runner)

    # LangChain emits a noisy UserWarning when max_completion_tokens is passed to ChatOpenAI.
    warnings.filterwarnings(
        "ignore",
        message=r".*max_completion_tokens.*",
        category=UserWarning,
    )

    # Suppress the resource_tracker leaked-semaphore warning that sometimes
    # fires during interpreter shutdown even after a clean run.
    warnings.filterwarnings(
        "ignore",
        message=r".*resource_tracker.*leaked.*",
        category=UserWarning,
    )

    parser = argparse.ArgumentParser(description="Run End-to-End Evaluation")
    parser.add_argument(
        "--run-id",
        help="Use this run identifier for result, report, checkpoint, and log files.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=str(EVAL_DATASET_FILE),
        help="Path to main dataset",
    )
    parser.add_argument(
        "--include-adversarial", action="store_true", help="Include adversarial cases"
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=EVAL_MAX_CONCURRENT,
        help="Max concurrent requests",
    )
    parser.add_argument(
        "--sample-per-source",
        type=int,
        default=None,
        help="Number of samples to take per source to run a quick test",
    )
    parser.add_argument(
        "--save-logs",
        action="store_true",
        help="Also write all INFO+ logs to a UTF-8 file under data/evaluation/results/logs/ (or --log-file)",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Log file path when --save-logs is set (default: results/logs/<run_id>.log)",
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=None,
        help=(
            "RNG seed for --sample-per-source. Defaults to EVAL_GENERATOR_SEED (42) for "
            "reproducibility. Named 'sample' because it only seeds dataset sampling -- "
            "DeepSeek's native API exposes no inference seed, so pipeline generation itself "
            "is not made reproducible by this flag."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Deprecated alias for --sample-seed. Will be removed in a future version.",
    )
    parser.add_argument(
        "--judge-provider",
        type=str,
        default=None,
        help=(
            "Override EVAL_LLM_PROVIDER for this run only (openai | anthropic | "
            "openai_compatible). Equivalent to exporting the env var before "
            "invoking this script, except: EVAL_ENABLE_RAGAS's auto-enable "
            "heuristic is computed at import time from the EVAL_LLM_MODEL env "
            "var, so it will NOT reflect a --judge-model override passed here -- "
            "export EVAL_ENABLE_RAGAS explicitly (or set EVAL_LLM_MODEL as a real "
            "env var before launching) if that matters for your comparison."
        ),
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default=None,
        help="Override EVAL_LLM_MODEL for this run only. See --judge-provider's help for a caveat re: EVAL_ENABLE_RAGAS.",
    )
    parser.add_argument(
        "--judge-base-url",
        type=str,
        default=None,
        help="Override EVAL_LLM_BASE_URL for this run only (required when --judge-provider=openai_compatible).",
    )
    parser.add_argument(
        "--judge-api-key-env",
        type=str,
        default=None,
        help=(
            "Override EVAL_LLM_API_KEY_ENV for this run only (required when "
            "--judge-provider=openai_compatible, else the judge silently looks "
            "for OPENAI_API_KEY -- see llm_factory._DEFAULT_API_KEY_ENV)."
        ),
    )

    args = parser.parse_args()

    # --seed is a deprecated alias for --sample-seed (it only ever seeded
    # dataset sampling, never pipeline generation -- see the help text above).
    # --sample-seed wins if both are somehow passed.
    if args.seed is not None:
        logger.warning(
            "--seed is deprecated; use --sample-seed instead (it only seeds dataset "
            "sampling -- DeepSeek's native API exposes no inference seed, so pipeline "
            "generation itself is not made reproducible by this flag)."
        )
        if args.sample_seed is None:
            args.sample_seed = args.seed

    # Apply judge overrides as env vars *before* the evaluator is constructed --
    # src.evaluation.judge_factory / llm_factory.resolve_role_config read
    # EVAL_LLM_* live from os.environ at call time (inside evaluate()), so this
    # is sufficient even though it happens after src.evaluation.config's own
    # module-level EVAL_LLM_MODEL/EVAL_ENABLE_RAGAS were already computed at
    # import time (hence the caveat in --judge-provider's help above).
    for _flag, _env_name in (
        (args.judge_provider, "EVAL_LLM_PROVIDER"),
        (args.judge_model, "EVAL_LLM_MODEL"),
        (args.judge_base_url, "EVAL_LLM_BASE_URL"),
        (args.judge_api_key_env, "EVAL_LLM_API_KEY_ENV"),
    ):
        if _flag is not None:
            os.environ[_env_name] = _flag

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Pre-generate the run_id so the log file, report, and JSON all share the same identifier,
    # and so the file handler can be attached *before* the evaluator is constructed (capturing
    # any import / ChromaDB errors that occur before evaluate() is called).
    run_id = args.run_id or generate_run_id()
    log_path: Optional[Path] = None
    if args.save_logs:
        log_path = (
            Path(args.log_file) if args.log_file else default_eval_log_path(run_id)
        )
        configure_evaluation_file_logging(log_path)

    # ---- Log run header -------------------------------------------------------
    logger.info("=" * 72)
    logger.info("Evaluation run started | run_id=%s", run_id)
    logger.info("Command: %s", " ".join(sys.argv))
    logger.info("Python:  %s", sys.version.replace("\n", " "))
    logger.info("Platform: %s", platform.platform())
    logger.info(
        "Args: dataset=%s include_adversarial=%s max_concurrent=%s "
        "sample_per_source=%s save_logs=%s log_file=%s",
        args.dataset,
        args.include_adversarial,
        args.max_concurrent,
        args.sample_per_source,
        args.save_logs,
        log_path,
    )
    if any(
        [
            args.judge_provider,
            args.judge_model,
            args.judge_base_url,
            args.judge_api_key_env,
        ]
    ):
        logger.info(
            "Judge override | EVAL_LLM_PROVIDER=%s EVAL_LLM_MODEL=%s "
            "EVAL_LLM_BASE_URL=%s EVAL_LLM_API_KEY_ENV=%s",
            os.environ.get("EVAL_LLM_PROVIDER"),
            os.environ.get("EVAL_LLM_MODEL"),
            os.environ.get("EVAL_LLM_BASE_URL"),
            os.environ.get("EVAL_LLM_API_KEY_ENV"),
        )
    logger.info("=" * 72)
    # ---------------------------------------------------------------------------

    wall_start = time.monotonic()
    res: Optional[Dict[str, Any]] = None
    exit_code = 0
    try:
        evaluator = EndToEndEvaluator(CHROMADB_DIR)

        dataset_path = Path(args.dataset)
        res = evaluator.evaluate(
            dataset_path=dataset_path,
            include_adversarial=args.include_adversarial,
            max_concurrent=args.max_concurrent,
            sample_size_per_source=args.sample_per_source,
            save_logs=args.save_logs,
            log_file=log_path,
            _run_id=run_id,
            seed=args.sample_seed,
        )
        if res.get("error"):
            exit_code = 1
    except Exception:
        exit_code = 1
        logger.critical(
            "Unhandled exception – evaluation aborted after %.1fs",
            time.monotonic() - wall_start,
            exc_info=True,
        )
    finally:
        logger.info("=" * 72)
        logger.info(
            "Evaluation finished | run_id=%s wall_time=%.1fs exit_code=%d",
            run_id,
            time.monotonic() - wall_start,
            exit_code,
        )
        if res is not None:
            logger.info("  report  → %s", res.get("report_path"))
            logger.info("  json    → %s", res.get("json_path"))
            logger.info("  log     → %s", res.get("log_path") or log_path or "N/A")
        logger.info("=" * 72)

    if res is not None:
        label = "Evaluation failed. Diagnostic report" if exit_code else "Done! Report"
        print(f"\n{label} saved to {res['report_path']}")
        print(f"JSON Results saved to {res['json_path']}")
        if res.get("log_path"):
            print(f"Logs saved to {res['log_path']}")
    elif log_path:
        print(f"\nEvaluation failed. See logs at {log_path}", file=sys.stderr)

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
