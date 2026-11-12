import pytest
import asyncio
import json
from unittest.mock import patch, MagicMock, AsyncMock
from pathlib import Path

from src.evaluation.evaluators.end_to_end_evaluator import EndToEndEvaluator, should_retry


@pytest.fixture
def operational_evaluator(tmp_path, monkeypatch):
    """Keep real batch/evaluate/CLI code; replace only model and metric I/O."""
    from src.evaluation.evaluators import end_to_end_evaluator as runner
    from src.evaluation.reporting import report_generator

    evaluator = object.__new__(EndToEndEvaluator)
    evaluator._last_fail_fast_error = None
    evaluator.chromadb_dir = tmp_path / "index"
    evaluator.graph = MagicMock()
    evaluator.graph.ainvoke = AsyncMock(return_value={"final_response": "Synthetic answer"})
    for name in ["retrieval_eval", "faithfulness_eval", "orchestration_eval", "domain_eval", "analytical_eval"]:
        metric = MagicMock()
        metric.evaluate.return_value = {"score": 0.0}
        setattr(evaluator, name, metric)
    evaluator._evaluate_per_category = MagicMock(return_value={})
    monkeypatch.setattr(runner, "EndToEndEvaluator", lambda *a, **kw: evaluator)
    monkeypatch.setattr(runner, "validate_ground_truth_chunks", lambda *a: {})
    monkeypatch.setattr(runner, "EVAL_RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(report_generator, "EVAL_RESULTS_DIR", tmp_path / "results")
    return evaluator


@pytest.mark.parametrize(
    "failure",
    ["pipeline", "fail_fast", "mixed_pipeline", "evaluator", "nested_metric", "metric_rows", "empty", "disabled", "inapplicable", None],
)
def test_operational_failures_propagate_through_actual_evaluate_and_cli(
    operational_evaluator, tmp_path, monkeypatch, capsys, failure
):
    import sys
    from src.evaluation.evaluators import end_to_end_evaluator as runner

    evaluator = operational_evaluator
    size = 4 if failure == "fail_fast" else 2
    if failure == "empty":
        size = 0
    dataset = [{"query": f"Question {i}", "expected_answer": "Synthetic", "ground_truth_chunks": []} for i in range(size)]
    path = tmp_path / "dataset.json"
    path.write_text(json.dumps(dataset))
    if failure in {"pipeline", "fail_fast"}:
        evaluator.graph.ainvoke.side_effect = ValueError("Synthetic provider failure")
    elif failure == "mixed_pipeline":
        evaluator.graph.ainvoke.side_effect = [ValueError("Synthetic provider failure"), {"final_response": "Synthetic"}]
    elif failure == "evaluator":
        evaluator.faithfulness_eval.evaluate.side_effect = RuntimeError("Synthetic judge failure")
    elif failure == "nested_metric":
        evaluator.faithfulness_eval.evaluate.return_value = {"ragas_metrics": {"faithfulness_error": "Synthetic judge failure"}}
    elif failure == "metric_rows":
        evaluator.faithfulness_eval.evaluate.return_value = {"ragas_metrics": {"faithfulness": 0.5, "faithfulness_n_failed": 1}}
    elif failure == "disabled":
        evaluator.faithfulness_eval.evaluate.return_value = {"ragas_metrics": {"skipped": "disabled"}}
    elif failure == "inapplicable":
        evaluator.orchestration_eval.evaluate.return_value = {"tool_call_accuracy": {"error": "no applicable entries (n=0)", "n": 0}}
    monkeypatch.setattr(sys, "argv", ["evaluator", "--dataset", str(path), "--run-id", "operational", "--max-concurrent", "1"])

    failed = failure not in {None, "disabled", "inapplicable"}
    assert runner.main() == int(failed)
    output = capsys.readouterr().out
    if failure == "empty":
        evaluator.graph.ainvoke.assert_not_called()
        assert not (tmp_path / "results" / "operational_results.json").exists()
        return
    payload = json.loads((tmp_path / "results" / "operational_results.json").read_text())
    assert payload["completion"]["status"] == ("failed" if failed else "complete")
    assert len(payload["dataset"]) == len(payload["pipeline_results"]) == size
    if failure == "fail_fast":
        assert payload["completion"]["processed_entries"] == 3
        assert payload["pipeline_results"][-1]["not_processed"] is True
    if failed:
        assert "Evaluation failed." in output
        assert "Run failed:" in (tmp_path / "results" / "operational_report.md").read_text()


@pytest.mark.parametrize("failure", [None, "partial", "exception"])
def test_cli_preserves_run_id_and_reports_failure(monkeypatch, failure):
    import multiprocessing
    import sys
    import warnings

    from src.evaluation.evaluators import end_to_end_evaluator as runner

    evaluator = MagicMock()
    evaluator.evaluate.return_value = {
        "report_path": "test_run_report.md",
        "json_path": "test_run_results.json",
    }
    if failure == "partial":
        evaluator.evaluate.return_value["error"] = "Stopped after pipeline failures"
    elif failure == "exception":
        evaluator.evaluate.side_effect = RuntimeError("Pipeline failed")
    monkeypatch.setattr(runner, "EndToEndEvaluator", lambda *_: evaluator)
    monkeypatch.setattr(
        multiprocessing, "set_start_method", lambda *_: None,
    )
    monkeypatch.setenv("TOKENIZERS_PARALLELISM", "false")
    monkeypatch.setattr(
        sys, "argv",
        ["end_to_end_evaluator.py", "--run-id", "test_run", "--dataset", "custom.json"],
    )

    with warnings.catch_warnings():
        status = runner.main()

    assert status == (1 if failure else 0)
    assert evaluator.evaluate.call_args.kwargs["_run_id"] == "test_run"
    assert evaluator.evaluate.call_args.kwargs["dataset_path"] == Path("custom.json")

def test_should_retry():
    assert should_retry(Exception("429 Too Many Requests")) is True
    assert should_retry(Exception("Rate limit exceeded")) is True
    assert should_retry(Exception("timeout")) is True
    assert should_retry(Exception("ValueError: Bad input")) is False

@pytest.mark.asyncio
async def test_run_pipeline_async_success():
    # Mock all the components
    with patch("src.evaluation.evaluators.end_to_end_evaluator.build_graph") as mock_build_graph, \
         patch("src.evaluation.evaluators.end_to_end_evaluator.RetrievalEvaluator"), \
         patch("src.evaluation.evaluators.end_to_end_evaluator.FaithfulnessEvaluator"), \
         patch("src.evaluation.evaluators.end_to_end_evaluator.OrchestrationEvaluator"), \
         patch("src.evaluation.evaluators.end_to_end_evaluator.DomainEvaluator"):

        # Setup mock graph
        mock_graph = MagicMock()
        # Real graph stores context_gathered as dict[tool, list[str]] + citations
        fmt = (
            "[Source 1] Collection: osha_regulations | Document: d | Location: l\n"
            "chunk text"
        )
        expected_state = {
            "context_gathered": {"osha_regulations": [fmt]},
            "draft_response": "draft",
            "final_response": "final",
            "validation_score": 0.9,
            "retry_count": 0,
            "citations": [
                {
                    "chunk_id": "cid1",
                    "collection": "osha_regulations",
                    "source": "d",
                    "location": "l",
                }
            ],
        }
        mock_graph.ainvoke = AsyncMock(return_value=expected_state)
        mock_build_graph.return_value = mock_graph

        evaluator = EndToEndEvaluator(Path("dummy"))
        semaphore = asyncio.Semaphore(1)

        result = await evaluator.run_pipeline_async("test query", semaphore)

        assert result["query"] == "test query"
        assert result["agents_called"] == ["osha_regs"]
        assert len(result["context_gathered"]) == 1
        assert result["context_gathered"][0]["chunk_id"] == "cid1"
        assert result["context_gathered"][0]["text"] == "chunk text"
        assert result["final_response"] == "final"
        assert result["validation_score"] == 0.9

@pytest.mark.asyncio
async def test_run_pipeline_async_records_wall_clock_and_token_totals():
    """Per-query wall-clock + token/cost totals must land directly on
    pipeline_results[i], summed from final_state's llm_calls."""
    with patch("src.evaluation.evaluators.end_to_end_evaluator.build_graph") as mock_build_graph, \
         patch("src.evaluation.evaluators.end_to_end_evaluator.RetrievalEvaluator"), \
         patch("src.evaluation.evaluators.end_to_end_evaluator.FaithfulnessEvaluator"), \
         patch("src.evaluation.evaluators.end_to_end_evaluator.OrchestrationEvaluator"), \
         patch("src.evaluation.evaluators.end_to_end_evaluator.DomainEvaluator"):

        mock_graph = MagicMock()
        expected_state = {
            "context_gathered": {},
            "draft_response": "draft",
            "final_response": "final",
            "validation_score": 0.9,
            "retry_count": 0,
            "citations": [],
            "llm_calls": [
                {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150, "estimated_cost_usd": 0.0001},
                {"prompt_tokens": 200, "completion_tokens": 80, "total_tokens": 280, "estimated_cost_usd": 0.0002},
            ],
        }

        async def slow_ainvoke(*_a, **_k):
            await asyncio.sleep(0.02)
            return expected_state

        mock_graph.ainvoke = slow_ainvoke
        mock_build_graph.return_value = mock_graph

        evaluator = EndToEndEvaluator(Path("dummy"))
        semaphore = asyncio.Semaphore(1)

        result = await evaluator.run_pipeline_async("test query", semaphore)

        assert result["wall_seconds"] >= 0.02
        assert result["n_llm_calls"] == 2
        assert result["prompt_tokens"] == 300
        assert result["completion_tokens"] == 130
        assert result["total_tokens"] == 430
        assert abs(result["total_cost_usd"] - 0.0003) < 1e-9


@pytest.mark.asyncio
async def test_run_pipeline_async_non_retryable_failure_still_records_wall_clock():
    """A hard pipeline failure has no final_state to sum llm_calls from,
    so token/cost totals are honestly 0 -- but wall_seconds is still recorded
    (it timed a real, if failed, attempt)."""
    with patch("src.evaluation.evaluators.end_to_end_evaluator.build_graph") as mock_build_graph, \
         patch("src.evaluation.evaluators.end_to_end_evaluator.RetrievalEvaluator"), \
         patch("src.evaluation.evaluators.end_to_end_evaluator.FaithfulnessEvaluator"), \
         patch("src.evaluation.evaluators.end_to_end_evaluator.OrchestrationEvaluator"), \
         patch("src.evaluation.evaluators.end_to_end_evaluator.DomainEvaluator"):

        mock_graph = MagicMock()

        async def raising_ainvoke(*_a, **_k):
            await asyncio.sleep(0.01)
            raise ValueError("Bad input")

        mock_graph.ainvoke = raising_ainvoke
        mock_build_graph.return_value = mock_graph

        evaluator = EndToEndEvaluator(Path("dummy"))
        semaphore = asyncio.Semaphore(1)

        result = await evaluator.run_pipeline_async("test query", semaphore)

        assert "error" in result
        assert result["wall_seconds"] >= 0.01
        assert result["n_llm_calls"] == 0
        assert result["total_cost_usd"] == 0.0


@pytest.mark.asyncio
async def test_run_pipeline_async_non_retryable_failure():
    with patch("src.evaluation.evaluators.end_to_end_evaluator.build_graph") as mock_build_graph, \
         patch("src.evaluation.evaluators.end_to_end_evaluator.RetrievalEvaluator"), \
         patch("src.evaluation.evaluators.end_to_end_evaluator.FaithfulnessEvaluator"), \
         patch("src.evaluation.evaluators.end_to_end_evaluator.OrchestrationEvaluator"), \
         patch("src.evaluation.evaluators.end_to_end_evaluator.DomainEvaluator"):

        mock_graph = MagicMock()
        mock_graph.ainvoke = AsyncMock(side_effect=ValueError("Bad input"))
        mock_build_graph.return_value = mock_graph

        evaluator = EndToEndEvaluator(Path("dummy"))
        semaphore = asyncio.Semaphore(1)

        result = await evaluator.run_pipeline_async("test query", semaphore)

        assert result["query"] == "test query"
        assert "Bad input" in result["error"]
        assert result["agents_called"] == []

@pytest.mark.asyncio
async def test_run_batch_async():
    with patch("src.evaluation.evaluators.end_to_end_evaluator.build_graph"), \
         patch("src.evaluation.evaluators.end_to_end_evaluator.RetrievalEvaluator"), \
         patch("src.evaluation.evaluators.end_to_end_evaluator.FaithfulnessEvaluator"), \
         patch("src.evaluation.evaluators.end_to_end_evaluator.OrchestrationEvaluator"), \
         patch("src.evaluation.evaluators.end_to_end_evaluator.DomainEvaluator"):

        evaluator = EndToEndEvaluator(Path("dummy"))

        # Mock run_pipeline_async to return immediately
        evaluator.run_pipeline_async = AsyncMock(side_effect=[{"query": "1"}, {"query": "2"}, Exception("Timeout")])

        dataset = [{"query": "1"}, {"query": "2"}, {"query": "3"}]
        results = await evaluator.run_batch_async(dataset, max_concurrent=2)

        assert len(results) == 3
        assert results[0]["query"] == "1"
        assert results[1]["query"] == "2"
        assert "Timeout" in results[2]["error"]


@pytest.mark.asyncio
async def test_run_batch_async_fail_fast_on_identical_errors():
    with patch("src.evaluation.evaluators.end_to_end_evaluator.build_graph"), \
         patch("src.evaluation.evaluators.end_to_end_evaluator.RetrievalEvaluator"), \
         patch("src.evaluation.evaluators.end_to_end_evaluator.FaithfulnessEvaluator"), \
         patch("src.evaluation.evaluators.end_to_end_evaluator.OrchestrationEvaluator"), \
         patch("src.evaluation.evaluators.end_to_end_evaluator.DomainEvaluator"):

        evaluator = EndToEndEvaluator(Path("dummy"))
        evaluator.run_pipeline_async = AsyncMock(side_effect=[
            {"query": "1", "error": "Unsupported value: temperature"},
            {"query": "2", "error": "Unsupported value: temperature"},
            {"query": "3", "error": "Unsupported value: temperature"},
            {"query": "4", "final_response": "should not run"},
        ])

        dataset = [{"query": "1"}, {"query": "2"}, {"query": "3"}, {"query": "4"}]
        results = await evaluator.run_batch_async(dataset, max_concurrent=1)

        assert len(results) == 3
        assert evaluator._last_fail_fast_error is not None
        assert "3 consecutive identical pipeline errors" in evaluator._last_fail_fast_error


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [4, 5])
async def test_fail_fast_preserves_every_result_in_completed_concurrent_batch(
    operational_evaluator, size
):
    evaluator = operational_evaluator
    evaluator.run_pipeline_async = AsyncMock(side_effect=[
        {"query": str(i), "error": "Synthetic failure"} if i < 3
        else {"query": str(i), "final_response": "Completed paid answer"}
        for i in range(size)
    ])
    dataset = [{"query": str(i)} for i in range(size)]
    results = await evaluator.run_batch_async(dataset, max_concurrent=4)
    assert len(results) == 4
    assert results[-1]["final_response"] == "Completed paid answer"
    assert evaluator.run_pipeline_async.await_count == 4
    assert bool(evaluator._last_fail_fast_error) == (size > 4)


@pytest.mark.asyncio
async def test_run_batch_async_writes_partial_checkpoints(tmp_path):
    with patch("src.evaluation.evaluators.end_to_end_evaluator.build_graph"), \
         patch("src.evaluation.evaluators.end_to_end_evaluator.RetrievalEvaluator"), \
         patch("src.evaluation.evaluators.end_to_end_evaluator.FaithfulnessEvaluator"), \
         patch("src.evaluation.evaluators.end_to_end_evaluator.OrchestrationEvaluator"), \
         patch("src.evaluation.evaluators.end_to_end_evaluator.DomainEvaluator"), \
         patch("src.evaluation.evaluators.end_to_end_evaluator.EVAL_RESULTS_DIR", tmp_path):
        evaluator = EndToEndEvaluator(Path("dummy"))
        evaluator.run_pipeline_async = AsyncMock(
            side_effect=[{"query": "1"}, {"query": "2"}, {"query": "3"}]
        )
        dataset = [{"query": "1"}, {"query": "2"}, {"query": "3"}]
        results = await evaluator.run_batch_async(
            dataset,
            max_concurrent=1,
            run_id="run_123",
            checkpoint_every=2,
        )
        assert len(results) == 3
        assert (tmp_path / "run_123_partial.json").exists()

@patch("src.evaluation.evaluators.end_to_end_evaluator.generate_run_id", return_value="run_123")
@patch("src.evaluation.evaluators.end_to_end_evaluator.generate_report", return_value="# summary")
@patch("src.evaluation.evaluators.end_to_end_evaluator.generate_json_results", return_value=Path("dummy.json"))
@patch("src.evaluation.evaluators.end_to_end_evaluator.load_eval_dataset")
def test_evaluate_synchronous(mock_load, mock_gen_json, mock_gen_repo, mock_gen_id):
    with patch("src.evaluation.evaluators.end_to_end_evaluator.build_graph"), \
         patch("src.evaluation.evaluators.end_to_end_evaluator.RetrievalEvaluator") as MockRet, \
         patch("src.evaluation.evaluators.end_to_end_evaluator.FaithfulnessEvaluator") as MockFaith, \
         patch("src.evaluation.evaluators.end_to_end_evaluator.OrchestrationEvaluator") as MockOrch, \
         patch("src.evaluation.evaluators.end_to_end_evaluator.DomainEvaluator") as MockDom, \
         patch("builtins.open", MagicMock()):

        mock_load.return_value = [{"query": "1"}, {"query": "2"}]

        evaluator = EndToEndEvaluator(Path("dummy"))
        # Mock batch to avoid real asyncio loop internally
        evaluator.run_batch = MagicMock(return_value=[{"query": "1"}, {"query": "2"}])

        # Mock sub evaluators
        MockRet.return_value.evaluate.return_value = {"r": 1}
        MockFaith.return_value.evaluate.return_value = {"f": 1}
        MockOrch.return_value.evaluate.return_value = {"o": 1}
        MockDom.return_value.evaluate.return_value = {"d": 1}

        # We need to mock Path.mkdir to prevent side effects
        with patch.object(Path, "mkdir"):
            result = evaluator.evaluate(Path("dummy.json"), include_adversarial=False)

            assert result["run_id"] == "run_123"
            assert result["metrics"]["retrieval"] == {"r": 1}
            assert "per_category" in result["metrics"]
            assert result["dataset_size"] == 2
            assert "log_path" not in result


@patch("src.evaluation.evaluators.end_to_end_evaluator.generate_run_id", return_value="run_123")
@patch("src.evaluation.evaluators.end_to_end_evaluator.generate_report", return_value="# summary")
@patch("src.evaluation.evaluators.end_to_end_evaluator.generate_json_results", return_value=Path("dummy.json"))
@patch("src.evaluation.evaluators.end_to_end_evaluator.load_eval_dataset")
@patch("src.evaluation.evaluators.end_to_end_evaluator.configure_evaluation_file_logging")
def test_evaluate_save_logs_configures_file(
    mock_configure, mock_load, mock_gen_json, mock_gen_repo, mock_gen_id, tmp_path
):
    with patch("src.evaluation.evaluators.end_to_end_evaluator.build_graph"), \
         patch("src.evaluation.evaluators.end_to_end_evaluator.RetrievalEvaluator") as MockRet, \
         patch("src.evaluation.evaluators.end_to_end_evaluator.FaithfulnessEvaluator") as MockFaith, \
         patch("src.evaluation.evaluators.end_to_end_evaluator.OrchestrationEvaluator") as MockOrch, \
         patch("src.evaluation.evaluators.end_to_end_evaluator.DomainEvaluator") as MockDom, \
         patch("builtins.open", MagicMock()):

        mock_load.return_value = [{"query": "1"}]
        MockRet.return_value.evaluate.return_value = {}
        MockFaith.return_value.evaluate.return_value = {}
        MockOrch.return_value.evaluate.return_value = {}
        MockDom.return_value.evaluate.return_value = {}

        log_file = tmp_path / "custom.log"
        evaluator = EndToEndEvaluator(Path("dummy"))
        evaluator.run_batch = MagicMock(return_value=[{}])

        with patch.object(Path, "mkdir"):
            result = evaluator.evaluate(
                Path("dummy.json"),
                save_logs=True,
                log_file=log_file,
            )

        mock_configure.assert_called_once_with(log_file)
        assert result["log_path"] == str(log_file)


@patch("src.evaluation.evaluators.end_to_end_evaluator.generate_run_id", return_value="run_123")
@patch("src.evaluation.evaluators.end_to_end_evaluator.generate_report", return_value="# summary")
@patch("src.evaluation.evaluators.end_to_end_evaluator.generate_json_results", return_value=Path("dummy.json"))
@patch("src.evaluation.evaluators.end_to_end_evaluator.load_eval_dataset")
def test_evaluate_sample_per_source_guarantees_category_coverage(
    mock_load, mock_gen_json, mock_gen_repo, mock_gen_id
):
    with patch("src.evaluation.evaluators.end_to_end_evaluator.build_graph"), \
         patch("src.evaluation.evaluators.end_to_end_evaluator.RetrievalEvaluator") as MockRet, \
         patch("src.evaluation.evaluators.end_to_end_evaluator.FaithfulnessEvaluator") as MockFaith, \
         patch("src.evaluation.evaluators.end_to_end_evaluator.OrchestrationEvaluator") as MockOrch, \
         patch("src.evaluation.evaluators.end_to_end_evaluator.DomainEvaluator") as MockDom, \
         patch("builtins.open", MagicMock()):

        # Two sources where per-source cap can accidentally drop category "analytical".
        mock_load.return_value = [
            {"query": "s1-c1", "source": "s1", "category": "incident"},
            {"query": "s1-c2", "source": "s1", "category": "analytical"},
            {"query": "s2-c1", "source": "s2", "category": "incident"},
            {"query": "s2-c2", "source": "s2", "category": "incident"},
        ]
        MockRet.return_value.evaluate.return_value = {}
        MockFaith.return_value.evaluate.return_value = {}
        MockOrch.return_value.evaluate.return_value = {}
        MockDom.return_value.evaluate.return_value = {}

        evaluator = EndToEndEvaluator(Path("dummy"))
        evaluator.run_batch = MagicMock(return_value=[{}, {}, {}])

        with patch.object(Path, "mkdir"):
            result = evaluator.evaluate(
                Path("dummy.json"),
                include_adversarial=False,
                sample_size_per_source=1,
                seed=1,
            )

        assert result["dataset_size"] == 3
