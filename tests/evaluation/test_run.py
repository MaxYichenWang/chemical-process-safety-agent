"""Tests for the staged-eval CLI (`src.evaluation.run`)."""

import argparse
import json
import os

import pytest

from src.evaluation import run as run_module
from src.evaluation.run import (
    build_parser,
    generation_path,
    main,
    results_path,
    scores_dir,
    skip_gt_validation_reason,
)
from src.evaluation.scoring.contract import sha256_json


def _subparser_help(parser: argparse.ArgumentParser, name: str) -> str:
    for action in parser._subparsers._group_actions:  # type: ignore[attr-defined]
        choices = getattr(action, "choices", None)
        if choices and name in choices:
            return choices[name].format_help()
    raise AssertionError(f"no subparser named {name!r}")


def _small_dataset_and_results():
    dataset = [
        {
            "query": "What does 1910.119 require?",
            "expected_answer": "PSM applies.",
            "ground_truth_chunks": ["osha_regulations_1910119_chunk_0"],
            "source": "auto_regulatory",
            "category": "regulatory_factual",
            "expected_agents": ["osha_regs"],
        },
        {
            "query": "adversarial: what happened at Fictional Plant 999?",
            "expected_answer": "No record found.",
            "ground_truth_chunks": [],
            "source": "adversarial",
            "category": "adversarial",
            "expected_agents": ["root_cause"],
        },
    ]
    results = [
        {
            "final_response": "1910.119 requires a written PSM program.",
            "context_gathered": [
                {
                    "chunk_id": "osha_regulations_1910119_chunk_0",
                    "text": "1910.119 requires...",
                    "corpus": "osha_regs",
                    "collection": "osha_regulations",
                    "source": "1910.119",
                    "location": "",
                }
            ],
            "agents_called": ["osha_regs"],
            "planned_agents": ["osha_regs"],
            "validation_score": 0.9,
            "retry_count": 0,
            "llm_calls": [],
        },
        {
            "final_response": "There is no record of that facility in the retrieved context.",
            "context_gathered": [],
            "agents_called": [],
            "planned_agents": [],
            "validation_score": 0.85,
            "retry_count": 0,
            "llm_calls": [],
        },
    ]
    return dataset, results


def _write_generation(run_id, dataset, results, *, dataset_path):
    run_metadata = {
        "run_id": run_id,
        "dataset_path": str(dataset_path),
        "dataset_sha256": sha256_json(dataset),
        "include_adversarial": False,
        "sample_size_per_source": None,
        "sample_seed": None,
    }
    gen_path = generation_path(run_id)
    gen_path.parent.mkdir(parents=True, exist_ok=True)
    with open(gen_path, "w", encoding="utf-8") as fh:
        json.dump({"run_metadata": run_metadata, "pipeline_results": results}, fh)
    return gen_path


@pytest.fixture(autouse=True)
def _isolated_runs_dir(tmp_path, monkeypatch):
    """Every test gets its own runs/ dir -- never touches the real repo's data/evaluation/runs."""
    monkeypatch.setattr(run_module, "RUNS_DIR", tmp_path / "runs")
    yield


# ---------------------------------------------------------------------------
# --help documents all three subcommands
# ---------------------------------------------------------------------------


def test_top_level_help_lists_all_three_subcommands():
    parser = build_parser()
    help_text = parser.format_help()
    assert "generate" in help_text
    assert "score" in help_text
    assert "aggregate" in help_text


def test_generate_help_documents_dry_run_and_skip_gt_validation():
    help_text = _subparser_help(build_parser(), "generate")
    assert "--dry-run" in help_text
    assert "--skip-gt-validation" in help_text
    assert "--sample-seed" in help_text


def test_score_help_documents_layer_choices_and_judge():
    help_text = _subparser_help(build_parser(), "score")
    assert "--layer" in help_text
    assert "--judge" in help_text
    assert "deterministic" in help_text and "deepeval" in help_text and "ragas" in help_text


def test_aggregate_help_documents_run_id():
    help_text = _subparser_help(build_parser(), "aggregate")
    assert "--run-id" in help_text


# ---------------------------------------------------------------------------
# dry-run resolves paths without running the pipeline
# ---------------------------------------------------------------------------


def test_generate_dry_run_resolves_paths_and_writes_nothing(tmp_path, capsys):
    dataset, _results = _small_dataset_and_results()
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text(json.dumps(dataset))

    rc = main(["generate", "--dataset", str(dataset_path), "--run-id", "dryrun1", "--dry-run"])

    assert rc == 0
    assert not generation_path("dryrun1").exists()
    out = capsys.readouterr().out
    assert "dryrun1" in out
    assert "2 entries" in out


def test_generate_dry_run_reports_auto_skipped_gt_validation(tmp_path, monkeypatch, capsys):
    # Point CHROMADB_DIR at a path that definitely doesn't exist, regardless
    # of whether this checkout happens to have a real chromadb/ directory,
    # so the auto-skip branch is exercised deterministically.
    monkeypatch.setattr(run_module, "CHROMADB_DIR", tmp_path / "no-such-chromadb")

    dataset, _results = _small_dataset_and_results()
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text(json.dumps(dataset))

    rc = main(["generate", "--dataset", str(dataset_path), "--run-id", "dryrun2", "--dry-run"])

    assert rc == 0
    out = capsys.readouterr().out
    assert "skipped" in out


@pytest.mark.parametrize("dry_run", [False, True])
def test_empty_generation_stops_before_constructing_pipeline(tmp_path, monkeypatch, dry_run):
    from unittest.mock import Mock
    from src.evaluation.evaluators import end_to_end_evaluator as runner

    path = tmp_path / "empty.json"
    path.write_text("[]")
    factory = Mock()
    monkeypatch.setattr(runner, "EndToEndEvaluator", factory)
    args = ["generate", "--dataset", str(path), "--run-id", "empty", "--no-save-logs"]
    assert main(args + (["--dry-run"] if dry_run else [])) == 1
    factory.assert_not_called()
    assert not generation_path("empty").exists()


@pytest.mark.parametrize(
    "mode,size,status,processed,score_status",
    [
        ("all_failed", 4, "incomplete", 3, 1),
        ("all_failed", 2, "failed", 2, 1),
        ("mixed", 2, "complete_with_errors", 2, 0),
        ("success", 2, "complete", 2, 0),
    ],
)
def test_actual_batch_completion_through_generation_and_scoring(
    tmp_path, monkeypatch, mode, size, status, processed, score_status
):
    from unittest.mock import AsyncMock, Mock
    from src.evaluation.evaluators import end_to_end_evaluator as runner

    dataset = [dict(_small_dataset_and_results()[0][0], query=f"Question {i}") for i in range(size)]
    path = tmp_path / "dataset.json"
    path.write_text(json.dumps(dataset))
    evaluator = object.__new__(runner.EndToEndEvaluator)
    evaluator._last_fail_fast_error = None
    evaluator.graph = Mock()

    async def invoke(state):
        if mode == "all_failed" or (mode == "mixed" and state["query"] == "Question 0"):
            raise ValueError("Synthetic provider failure")
        return {"final_response": "Synthetic answer", "draft_response": "Synthetic answer"}

    evaluator.graph.ainvoke = AsyncMock(side_effect=invoke)
    monkeypatch.setattr(runner, "EndToEndEvaluator", lambda *a, **kw: evaluator)
    rc = main(["generate", "--dataset", str(path), "--run-id", "completion", "--max-concurrent", "1", "--skip-gt-validation", "--no-save-logs"])
    assert rc == (0 if mode == "success" else 1)
    generation = json.loads(generation_path("completion").read_text())
    completion = generation["run_metadata"]["completion"]
    assert completion["status"] == status
    assert completion["processed_entries"] == processed
    assert completion["unprocessed_entries"] == size - processed
    assert len(generation["pipeline_results"]) == size
    assert generation["run_metadata"]["dataset_sha256"] == sha256_json(dataset)
    assert [r["query"] for r in generation["pipeline_results"]] == [d["query"] for d in dataset]
    if processed < size:
        assert generation["pipeline_results"][-1]["not_processed"] is True
    assert main(["score", "--run-id", "completion", "--layer", "deterministic"]) == score_status
    if score_status:
        assert main(["aggregate", "--run-id", "completion"]) == 1
        assert not (scores_dir("completion") / "deterministic.json").exists()


def test_skip_gt_validation_reason_forced(tmp_path):
    assert skip_gt_validation_reason(tmp_path, True) is not None


def test_skip_gt_validation_reason_auto_skips_when_corpus_absent(tmp_path):
    missing = tmp_path / "does-not-exist"
    reason = skip_gt_validation_reason(missing, False)
    assert reason is not None
    assert "no corpus" in reason


def test_skip_gt_validation_reason_none_when_corpus_present_and_not_forced(tmp_path):
    assert skip_gt_validation_reason(tmp_path, False) is None


# ---------------------------------------------------------------------------
# score --layer deterministic needs no judge/key/corpus
# ---------------------------------------------------------------------------


def test_score_deterministic_matches_direct_call(tmp_path):
    dataset, results = _small_dataset_and_results()
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text(json.dumps(dataset))
    gen_path = _write_generation("run1", dataset, results, dataset_path=dataset_path)

    rc = main(["score", "--run-id", "run1", "--layer", "deterministic"])
    assert rc == 0

    score_path = scores_dir("run1") / "deterministic.json"
    assert score_path.exists()
    score_file = json.loads(score_path.read_text())

    assert score_file["layer"] == "deterministic"
    assert score_file["generation_sha256"] == sha256_json(json.loads(gen_path.read_text()))
    assert score_file["judge"] is None
    assert score_file["n_scored"] == 2
    assert score_file["n_failed"] == 0

    from src.evaluation.scoring.deterministic import score_deterministic

    assert score_file["aggregates"] == score_deterministic(dataset, results)


def test_score_dry_run_writes_no_score_file(tmp_path):
    dataset, results = _small_dataset_and_results()
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text(json.dumps(dataset))
    _write_generation("run2", dataset, results, dataset_path=dataset_path)

    rc = main(["score", "--run-id", "run2", "--layer", "deterministic", "--dry-run"])
    assert rc == 0
    assert not (scores_dir("run2") / "deterministic.json").exists()


def test_score_missing_generation_errors(capsys):
    rc = main(["score", "--run-id", "does-not-exist", "--layer", "deterministic"])
    assert rc == 1
    assert "no generation.json" in capsys.readouterr().err


def test_score_requires_judge_for_deepeval(tmp_path, capsys):
    dataset, results = _small_dataset_and_results()
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text(json.dumps(dataset))
    _write_generation("run3", dataset, results, dataset_path=dataset_path)

    rc = main(["score", "--run-id", "run3", "--layer", "deepeval"])
    assert rc == 1
    assert "--judge is required" in capsys.readouterr().err


def test_score_dataset_length_mismatch_errors(tmp_path, capsys):
    dataset, results = _small_dataset_and_results()
    dataset_path = tmp_path / "dataset.json"
    # Only persist ONE entry on disk, but pipeline_results has two --
    # simulates the dataset file changing after generation.
    dataset_path.write_text(json.dumps(dataset[:1]))
    _write_generation("run4", dataset, results, dataset_path=dataset_path)

    rc = main(["score", "--run-id", "run4", "--layer", "deterministic"])
    assert rc == 1
    assert "dataset_sha256 mismatch" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# aggregate + integrity contract hard-fail
# ---------------------------------------------------------------------------


def test_aggregate_merges_deterministic_layer(tmp_path):
    dataset, results = _small_dataset_and_results()
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text(json.dumps(dataset))
    _write_generation("run5", dataset, results, dataset_path=dataset_path)
    assert main(["score", "--run-id", "run5", "--layer", "deterministic"]) == 0

    rc = main(["aggregate", "--run-id", "run5"])
    assert rc == 0

    results_json = json.loads(results_path("run5").read_text())
    assert results_json["layers_present"] == ["deterministic"]
    assert results_json["layers_missing"] == []
    assert "retrieval" in results_json["metrics"]

    report_text = run_module.report_path("run5").read_text()
    assert "run5" in report_text
    assert "Layer 1: Retrieval Quality" in report_text


def _write_fake_judge_score_file(run_id, *, layer, judge_id, generation_sha256, judge_calls, judge_cost_usd):
    """Write a judge score file directly (bypassing `main(["score",...])`,
    which needs a real/mocked judge model) so aggregate's cost_summary can be
    tested against known judge_calls/judge_cost_usd values."""
    from src.evaluation.scoring.contract import build_score_file

    score_file = build_score_file(
        layer=layer,
        generation_sha256=generation_sha256,
        dataset_sha256=json.loads(generation_path(run_id).read_text())["run_metadata"]["dataset_sha256"],
        per_entry={},
        aggregates={},
        started_utc="2026-08-01T00:00:00Z",
        wall_seconds=1.0,
        n_scored=0,
        n_failed=0,
        judge={"provider": "test", "model": judge_id, "temperature": 0.0, "same_family_as_pipeline": False},
        judge_calls=judge_calls,
        judge_cost_usd=judge_cost_usd,
    )
    out_path = run_module.scores_dir(run_id) / run_module.score_file_name(layer, judge_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(score_file))
    return out_path


def test_aggregate_cost_summary_sums_pipeline_and_judge_cost(tmp_path):
    dataset, results = _small_dataset_and_results()
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text(json.dumps(dataset))

    run_metadata = {
        "run_id": "run_cost1",
        "dataset_path": str(dataset_path),
        "dataset_sha256": sha256_json(dataset),
        "include_adversarial": False,
        "sample_size_per_source": None,
        "sample_seed": None,
        "pipeline_cost_usd": 0.8186,
    }
    gen_path = generation_path("run_cost1")
    gen_path.parent.mkdir(parents=True, exist_ok=True)
    gen_path.write_text(json.dumps({"run_metadata": run_metadata, "pipeline_results": results}))
    generation_sha256 = sha256_json({"run_metadata": run_metadata, "pipeline_results": results})

    assert main(["score", "--run-id", "run_cost1", "--layer", "deterministic"]) == 0
    _write_fake_judge_score_file(
        "run_cost1", layer="deepeval", judge_id="qwen3.7-flash",
        generation_sha256=generation_sha256, judge_calls=10, judge_cost_usd=0.05,
    )
    _write_fake_judge_score_file(
        "run_cost1", layer="ragas", judge_id="qwen3.7-flash",
        generation_sha256=generation_sha256, judge_calls=5, judge_cost_usd=0.02,
    )

    assert main(["aggregate", "--run-id", "run_cost1"]) == 0
    results_json = json.loads(results_path("run_cost1").read_text())
    cost = results_json["cost_summary"]
    assert cost["pipeline_cost_usd"] == 0.8186
    assert abs(cost["judge_cost_usd"] - 0.07) < 1e-9
    assert cost["judge_calls"] == 15
    assert abs(cost["total_cost_usd"] - 0.8886) < 1e-9


def test_aggregate_cost_summary_judge_cost_none_when_no_judge_layer(tmp_path):
    dataset, results = _small_dataset_and_results()
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text(json.dumps(dataset))
    _write_generation("run_cost2", dataset, results, dataset_path=dataset_path)

    assert main(["score", "--run-id", "run_cost2", "--layer", "deterministic"]) == 0
    assert main(["aggregate", "--run-id", "run_cost2"]) == 0

    results_json = json.loads(results_path("run_cost2").read_text())
    cost = results_json["cost_summary"]
    assert cost["judge_cost_usd"] is None
    assert cost["judge_calls"] is None
    assert cost["total_cost_usd"] is None


def test_aggregate_dry_run_lists_score_files_without_writing(tmp_path):
    dataset, results = _small_dataset_and_results()
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text(json.dumps(dataset))
    _write_generation("run6", dataset, results, dataset_path=dataset_path)
    assert main(["score", "--run-id", "run6", "--layer", "deterministic"]) == 0

    rc = main(["aggregate", "--run-id", "run6", "--dry-run"])
    assert rc == 0
    assert not results_path("run6").exists()


def test_aggregate_missing_deterministic_layer_is_reported(tmp_path):
    dataset, results = _small_dataset_and_results()
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text(json.dumps(dataset))
    _write_generation("run7", dataset, results, dataset_path=dataset_path)
    # No score command run -- scores/ dir doesn't even exist.

    rc = main(["aggregate", "--run-id", "run7"])
    assert rc == 0
    results_json = json.loads(results_path("run7").read_text())
    assert results_json["layers_present"] == []
    assert results_json["layers_missing"] == ["deterministic"]


def test_score_uses_embedded_dataset_for_migrated_style_generation(tmp_path):
    """A migrated legacy run embeds `dataset` directly in generation.json
    (its dataset_path/sampling params aren't reliably reproducible against the
    *current* dataset file) -- score must use the embedded copy, not resolve_dataset."""
    dataset, results = _small_dataset_and_results()
    gen_path = generation_path("legacy1")
    gen_path.parent.mkdir(parents=True, exist_ok=True)
    generation = {
        "run_metadata": {
            "run_id": "legacy1",
            "dataset_path": "<migrated from some_old_file.json -- not reliably re-resolvable>",
            "dataset_sha256": sha256_json(dataset),
            "include_adversarial": None,
            "sample_size_per_source": None,
            "sample_seed": None,
            "provenance": "reconstructed",
        },
        "pipeline_results": results,
        "dataset": dataset,
    }
    gen_path.write_text(json.dumps(generation))

    rc = main(["score", "--run-id", "legacy1", "--layer", "deterministic"])
    assert rc == 0

    score_file = json.loads((scores_dir("legacy1") / "deterministic.json").read_text())
    from src.evaluation.scoring.deterministic import score_deterministic

    assert score_file["aggregates"] == score_deterministic(dataset, results)


def test_load_generation_falls_back_to_gzip(tmp_path):
    import gzip

    dataset, results = _small_dataset_and_results()
    generation = {
        "run_metadata": {"run_id": "gz1", "dataset_path": "x", "dataset_sha256": sha256_json(dataset)},
        "pipeline_results": results,
        "dataset": dataset,
    }
    gz_path = generation_path("gz1").with_suffix(".json.gz")
    gz_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(gz_path, "wt", encoding="utf-8") as fh:
        json.dump(generation, fh)

    assert not generation_path("gz1").exists()
    loaded, path_used = run_module.load_generation("gz1")
    assert path_used == gz_path
    assert loaded == generation


def test_load_generation_prefers_plain_json_over_gzip(tmp_path):
    import gzip

    dataset, results = _small_dataset_and_results()
    plain = {"run_metadata": {"run_id": "both1"}, "pipeline_results": results}
    stale_gz = {"run_metadata": {"run_id": "both1", "stale": True}, "pipeline_results": []}

    gen_path = generation_path("both1")
    gen_path.parent.mkdir(parents=True, exist_ok=True)
    gen_path.write_text(json.dumps(plain))
    with gzip.open(gen_path.with_suffix(".json.gz"), "wt", encoding="utf-8") as fh:
        json.dump(stale_gz, fh)

    loaded, path_used = run_module.load_generation("both1")
    assert path_used == gen_path
    assert loaded == plain


def test_load_generation_missing_run_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        run_module.load_generation("no-such-run-anywhere")


def test_score_reads_gzip_generation_end_to_end(tmp_path):
    import gzip

    dataset, results = _small_dataset_and_results()
    generation = {
        "run_metadata": {
            "run_id": "gz2",
            "dataset_path": "x",
            "dataset_sha256": sha256_json(dataset),
            "include_adversarial": None,
            "sample_size_per_source": None,
            "sample_seed": None,
        },
        "pipeline_results": results,
        "dataset": dataset,
    }
    gz_path = generation_path("gz2").with_suffix(".json.gz")
    gz_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(gz_path, "wt", encoding="utf-8") as fh:
        json.dump(generation, fh)

    rc = main(["score", "--run-id", "gz2", "--layer", "deterministic"])
    assert rc == 0
    score_file = json.loads((scores_dir("gz2") / "deterministic.json").read_text())
    assert score_file["generation_sha256"] == sha256_json(generation)


def test_aggregate_hard_fails_on_generation_sha_mismatch(tmp_path, capsys):
    """A score file computed against a different generation must never
    be silently merged -- nonzero exit, no results.json written."""
    dataset, results = _small_dataset_and_results()
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text(json.dumps(dataset))
    _write_generation("run8", dataset, results, dataset_path=dataset_path)
    assert main(["score", "--run-id", "run8", "--layer", "deterministic"]) == 0

    score_path = scores_dir("run8") / "deterministic.json"
    score_file = json.loads(score_path.read_text())
    score_file["generation_sha256"] = "0" * 64
    score_path.write_text(json.dumps(score_file))

    assert not results_path("run8").exists()
    rc = main(["aggregate", "--run-id", "run8"])
    assert rc == 1
    assert "generation_sha256 mismatch" in capsys.readouterr().err
    assert not results_path("run8").exists()


# ---------------------------------------------------------------------------
# judge metadata / env-var override plumbing (no judge package import needed --
# resolve_role_config is a pure function over os.environ)
# ---------------------------------------------------------------------------


def test_apply_judge_overrides_sets_env_vars():
    saved = {
        k: os.environ.get(k)
        for k in ("EVAL_LLM_MODEL", "EVAL_LLM_PROVIDER", "EVAL_LLM_BASE_URL", "EVAL_LLM_API_KEY_ENV")
    }
    try:
        ns = argparse.Namespace(
            judge="qwen3.7-flash",
            judge_provider="openai_compatible",
            judge_base_url="https://example.test/v1",
            judge_api_key_env="QWEN_API_KEY",
        )
        run_module._apply_judge_overrides(ns)
        assert os.environ["EVAL_LLM_MODEL"] == "qwen3.7-flash"
        assert os.environ["EVAL_LLM_PROVIDER"] == "openai_compatible"
        assert os.environ["EVAL_LLM_BASE_URL"] == "https://example.test/v1"
        assert os.environ["EVAL_LLM_API_KEY_ENV"] == "QWEN_API_KEY"
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_apply_judge_overrides_sets_temperature():
    saved = os.environ.get("EVAL_LLM_TEMPERATURE")
    try:
        ns = argparse.Namespace(judge=None, judge_temperature=0.0)
        run_module._apply_judge_overrides(ns)
        assert os.environ["EVAL_LLM_TEMPERATURE"] == "0.0"
        # None must not touch the env var
        os.environ["EVAL_LLM_TEMPERATURE"] = "sentinel"
        run_module._apply_judge_overrides(argparse.Namespace(judge=None, judge_temperature=None))
        assert os.environ["EVAL_LLM_TEMPERATURE"] == "sentinel"
    finally:
        if saved is None:
            os.environ.pop("EVAL_LLM_TEMPERATURE", None)
        else:
            os.environ["EVAL_LLM_TEMPERATURE"] = saved


def test_tag_disambiguates_score_file_and_cache_names():
    assert run_module.score_file_name("deepeval", "qwen3.7-flash") == "deepeval.qwen3-7-flash.json"
    assert (
        run_module.score_file_name("deepeval", "qwen3.7-flash", "temp0")
        == "deepeval.qwen3-7-flash.temp0.json"
    )
    # same judge, different tag -> different resume cache, so a temperature
    # change never resumes from the previous setting's cached scores
    c_untagged = run_module.cache_file_path("R", "deepeval", "qwen3.7-flash").name
    c_tagged = run_module.cache_file_path("R", "deepeval", "qwen3.7-flash", "temp0").name
    assert c_untagged != c_tagged


def test_aggregate_keeps_tagged_variants_separately(tmp_path, monkeypatch):
    from src.evaluation.scoring.contract import build_score_file, sha256_json

    monkeypatch.setattr(run_module, "RUNS_DIR", tmp_path)
    run_id = "_tagtest"
    base = tmp_path / run_id
    (base / "scores").mkdir(parents=True)
    gen = {
        "run_metadata": {"pipeline_cost_usd": 0.1},
        "pipeline_results": [{"a": 1}, {"a": 2}],
        "dataset": [{"id": "0"}, {"id": "1"}],
    }
    (base / "generation.json").write_text(json.dumps(gen))
    gsha = sha256_json(gen)
    dsha = sha256_json(gen["dataset"])
    for tag, temp in ((None, 1.0), ("temp0", 0.0)):
        sf = build_score_file(
            layer="deepeval",
            generation_sha256=gsha,
            dataset_sha256=dsha,
            per_entry={"0": {"faithfulness": 0.9}},
            aggregates={"faithfulness": {"mean": 0.9}},
            started_utc="t",
            wall_seconds=1,
            n_scored=2,
            n_failed=0,
            judge={"model": "qwen3.7-flash", "temperature": temp},
            tag=tag,
        )
        name = "deepeval.qwen3-7-flash.json" if tag is None else "deepeval.qwen3-7-flash.temp0.json"
        (base / "scores" / name).write_text(json.dumps(sf))

    rc = run_module.cmd_aggregate(argparse.Namespace(run_id=run_id, dry_run=False, no_archive=False))
    assert rc == 0
    results = json.loads((base / "results.json").read_text())
    assert "deepeval.qwen3-7-flash" in results["layers_present"]
    assert "deepeval.qwen3-7-flash.temp0" in results["layers_present"]

    # re-aggregating snapshots the prior results.json instead of silently losing it
    run_module.cmd_aggregate(argparse.Namespace(run_id=run_id, dry_run=False, no_archive=False))
    assert list((base / "results_history").glob("results.*.json"))


def test_aggregate_promotes_ragas_crossframework_and_bias(tmp_path, monkeypatch):
    from src.evaluation.scoring.contract import build_score_file, sha256_json

    monkeypatch.setattr(run_module, "RUNS_DIR", tmp_path)
    run_id = "_promote"
    base = tmp_path / run_id
    (base / "scores").mkdir(parents=True)
    gen = {
        "run_metadata": {"pipeline_cost_usd": 0.1},
        "pipeline_results": [{"a": 1}, {"a": 2}],
        "dataset": [{"id": "0"}, {"id": "1"}],
    }
    (base / "generation.json").write_text(json.dumps(gen))
    gsha = sha256_json(gen)
    dsha = sha256_json(gen["dataset"])

    def write(layer, judge, per_entry, aggregates):
        sf = build_score_file(
            layer=layer,
            generation_sha256=gsha,
            dataset_sha256=dsha,
            per_entry=per_entry,
            aggregates=aggregates,
            started_utc="t",
            wall_seconds=1,
            n_scored=2,
            n_failed=0,
            judge={"model": judge, "same_family_as_pipeline": judge == "deepseek"},
        )
        name = run_module.score_file_name(layer, judge)
        (base / "scores" / name).write_text(json.dumps(sf))

    write("deepeval", "qwen", {"0": {"faithfulness": 1.0}, "1": {"faithfulness": 0.6}}, {})
    write(
        "ragas",
        "qwen",
        {"0": {"faithfulness": 0.9}, "1": {"faithfulness": 0.5}},
        {"faithfulness": 0.7, "faithfulness_n_scored": 2},
    )
    # deepseek sorts before qwen on disk; primary selection must still pick qwen
    write(
        "ragas",
        "deepseek",
        {"0": {"faithfulness": 0.6}, "1": {"faithfulness": 0.4}},
        {"faithfulness": 0.5, "faithfulness_n_scored": 2},
    )

    rc = run_module.cmd_aggregate(argparse.Namespace(run_id=run_id, dry_run=False, no_archive=True))
    assert rc == 0
    results = json.loads((base / "results.json").read_text())
    faith = results["metrics"]["faithfulness"]
    # headline RAGAS metrics come from the primary (qwen) judge, not disk order
    assert faith["ragas_metrics"]["faithfulness"] == 0.7
    cross = faith["cross_framework"]
    assert cross["n"] == 2 and cross["deepeval_judge"] == "qwen" and cross["ragas_judge"] == "qwen"
    bias = results["metrics"]["judge_comparison"][0]
    assert bias["primary_judge"] == "qwen" and bias["secondary_judge"] == "deepseek"
    assert bias["metrics"]["faithfulness"]["delta"] > 0



    monkeypatch.setenv("EVAL_LLM_MODEL", "deepseek-v4-flash")
    monkeypatch.setenv("EVAL_LLM_PROVIDER", "openai_compatible")
    monkeypatch.setenv("AGENT_WORKER_PROVIDER", "openai_compatible")
    monkeypatch.setenv("AGENT_WORKER_MODEL", "deepseek-v4-flash")

    meta = run_module._resolve_judge_metadata("deepseek-v4-flash")
    assert meta["model"] == "deepseek-v4-flash"
    assert meta["same_family_as_pipeline"] is True


def test_resolve_judge_metadata_flags_cross_family(monkeypatch):
    monkeypatch.setenv("EVAL_LLM_MODEL", "qwen3.7-flash")
    monkeypatch.setenv("EVAL_LLM_PROVIDER", "openai_compatible")
    monkeypatch.setenv("AGENT_WORKER_PROVIDER", "openai_compatible")
    monkeypatch.setenv("AGENT_WORKER_MODEL", "deepseek-v4-flash")

    meta = run_module._resolve_judge_metadata("qwen3.7-flash")
    assert meta["same_family_as_pipeline"] is False


# ---------------------------------------------------------------------------
# ablation flag snapshotting (more flags may be added later; this documents today's set)
# ---------------------------------------------------------------------------


def test_collect_ablation_flags_reads_existing_flags():
    flags = run_module._collect_ablation_flags()
    assert "SUPERVISOR_PREMISE_CHECK_ENABLED" in flags
    assert "HYBRID_RERANK_ENABLED" in flags
    assert isinstance(flags["SUPERVISOR_PREMISE_CHECK_ENABLED"], bool)


# ---------------------------------------------------------------------------
# resolve_dataset
# ---------------------------------------------------------------------------


def test_resolve_dataset_with_no_sampling_returns_full_dataset(tmp_path):
    dataset, _results = _small_dataset_and_results()
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text(json.dumps(dataset))

    resolved = run_module.resolve_dataset(
        dataset_path, include_adversarial=False, sample_size_per_source=None, sample_seed=None
    )
    assert resolved == dataset


def test_resolve_dataset_is_deterministic_with_a_seed(tmp_path):
    dataset = [
        {
            "query": f"q{i}",
            "expected_answer": "a",
            "ground_truth_chunks": [],
            "source": "auto_regulatory",
            "category": "regulatory_factual",
        }
        for i in range(10)
    ]
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text(json.dumps(dataset))

    a = run_module.resolve_dataset(
        dataset_path, include_adversarial=False, sample_size_per_source=3, sample_seed=7
    )
    b = run_module.resolve_dataset(
        dataset_path, include_adversarial=False, sample_size_per_source=3, sample_seed=7
    )
    assert a == b
    assert len(a) == 3


# ---------------------------------------------------------------------------
# Embedded-dataset re-binding (--rebind-dataset)
#
# A migrated run embeds its dataset snapshot, which freezes it against later
# dataset repairs. Re-binding maps the snapshot onto today's entries by query.
# The safety properties matter more than the happy path: a partial or reordered
# re-bind would score a response against a different question's reference.
# ---------------------------------------------------------------------------


def _entry(query, expected_answer="a", **extra):
    base = {
        "query": query,
        "expected_answer": expected_answer,
        "ground_truth_chunks": [],
        "source": "auto_regulatory",
        "category": "regulatory_factual",
    }
    base.update(extra)
    return base


def test_rebind_returns_current_entries_in_snapshot_order():
    snapshot = [_entry("q2"), _entry("q0"), _entry("q1")]
    current = [
        _entry("q0", "repaired-0", defects=[{"code": "gt_stub_only"}]),
        _entry("q1", "repaired-1"),
        _entry("q2", "repaired-2"),
    ]

    rebound = run_module.rebind_embedded_dataset(snapshot, current)

    assert [e["query"] for e in rebound] == ["q2", "q0", "q1"]
    assert [e["expected_answer"] for e in rebound] == [
        "repaired-2",
        "repaired-0",
        "repaired-1",
    ]
    # The whole point: repair annotations reach the migrated run.
    assert rebound[1]["defects"] == [{"code": "gt_stub_only"}]


def test_rebind_rejects_a_changed_question_set():
    snapshot = [_entry("q0"), _entry("q1")]
    current = [_entry("q0"), _entry("q-renamed")]

    with pytest.raises(ValueError, match="absent from the current dataset"):
        run_module.rebind_embedded_dataset(snapshot, current)


def test_rebind_rejects_a_length_mismatch():
    with pytest.raises(ValueError, match="different question sets"):
        run_module.rebind_embedded_dataset([_entry("q0")], [_entry("q0"), _entry("q1")])


def test_rebind_rejects_ambiguous_duplicate_queries():
    snapshot = [_entry("dup"), _entry("q1")]
    current = [_entry("dup"), _entry("dup")]

    with pytest.raises(ValueError, match="duplicate query"):
        run_module.rebind_embedded_dataset(snapshot, current)


def test_dataset_for_generation_returns_frozen_snapshot_by_default():
    generation = {"dataset": [_entry("q0", "stale")]}

    resolved = run_module.dataset_for_generation(generation, {})

    assert resolved[0]["expected_answer"] == "stale"


def test_dataset_for_generation_rebinds_when_asked(tmp_path, monkeypatch):
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text(json.dumps([_entry("q0", "repaired")]))
    monkeypatch.setattr(run_module, "EVAL_DATASET_FILE", str(dataset_path))

    generation = {"dataset": [_entry("q0", "stale")]}
    run_metadata = {"dataset_path": "<migrated from ...; not resolvable>"}

    resolved = run_module.dataset_for_generation(generation, run_metadata, rebind=True)

    assert resolved[0]["expected_answer"] == "repaired"


def test_rebind_is_a_noop_for_runs_that_resolve_from_run_metadata(tmp_path):
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text(json.dumps([_entry("q0")]))
    run_metadata = {"dataset_path": str(dataset_path), "include_adversarial": False}

    without = run_module.dataset_for_generation({}, run_metadata)
    with_flag = run_module.dataset_for_generation({}, run_metadata, rebind=True)

    assert without == with_flag


def test_score_and_aggregate_help_document_rebind_dataset():
    parser = build_parser()
    assert "--rebind-dataset" in _subparser_help(parser, "score")
    assert "--rebind-dataset" in _subparser_help(parser, "aggregate")


@pytest.mark.parametrize("command", ["score", "aggregate"])
def test_changed_external_dataset_is_rejected_before_writing(tmp_path, capsys, command):
    dataset, results = _small_dataset_and_results()
    path = tmp_path / "dataset.json"
    path.write_text(json.dumps(dataset))
    _write_generation("changed", dataset, results, dataset_path=path)
    assert main(["score", "--run-id", "changed", "--layer", "deterministic"]) == 0
    score_path = scores_dir("changed") / "deterministic.json"
    original_score = score_path.read_bytes()
    path.write_text(json.dumps(dataset[::-1]))

    args = [command, "--run-id", "changed"]
    if command == "score":
        args += ["--layer", "deterministic"]
    assert main(args) == 1
    assert "dataset_sha256 mismatch" in capsys.readouterr().err
    assert score_path.read_bytes() == original_score
    assert not results_path("changed").exists()


def test_aggregate_rejects_score_from_another_dataset(tmp_path, capsys):
    dataset, results = _small_dataset_and_results()
    path = tmp_path / "dataset.json"
    path.write_text(json.dumps(dataset))
    _write_generation("mixed", dataset, results, dataset_path=path)
    assert main(["score", "--run-id", "mixed", "--layer", "deterministic"]) == 0
    score_path = scores_dir("mixed") / "deterministic.json"
    score = json.loads(score_path.read_text())
    score["dataset_sha256"] = sha256_json(dataset[::-1])
    score["layer"] = "ragas"
    score["judge"] = {"model": "local"}
    score_path.with_name("ragas.local.json").write_text(json.dumps(score))
    assert main(["aggregate", "--run-id", "mixed"]) == 1
    assert "dataset_sha256 mismatch" in capsys.readouterr().err
    assert not results_path("mixed").exists()


def test_rebound_dataset_requires_consistent_score_and_aggregate_flags(tmp_path, capsys):
    dataset, results = _small_dataset_and_results()
    path = tmp_path / "dataset.json"
    generation = {"dataset": dataset, "pipeline_results": results,
                  "run_metadata": {"dataset_path": str(path), "dataset_sha256": sha256_json(dataset)}}
    gen_path = generation_path("rebound")
    gen_path.parent.mkdir(parents=True)
    gen_path.write_text(json.dumps(generation))
    current = [dict(entry, answer_gradable=False) for entry in dataset[::-1]]
    path.write_text(json.dumps(current))

    assert main(["score", "--run-id", "rebound", "--layer", "deterministic", "--rebind-dataset"]) == 0
    assert main(["aggregate", "--run-id", "rebound"]) == 1
    assert "dataset_sha256 mismatch" in capsys.readouterr().err
    assert main(["aggregate", "--run-id", "rebound", "--rebind-dataset"]) == 0
    merged = json.loads(results_path("rebound").read_text())
    assert merged["dataset_sha256"] == sha256_json(current[::-1])
    assert gen_path.read_text() == json.dumps(generation)


def test_aggregate_existing_scores_when_external_dataset_is_archived(tmp_path):
    dataset, results = _small_dataset_and_results()
    path = tmp_path / "dataset.json"
    path.write_text(json.dumps(dataset))
    _write_generation("archived", dataset, results, dataset_path=path)
    assert main(["score", "--run-id", "archived", "--layer", "deterministic"]) == 0
    path.unlink()
    assert main(["aggregate", "--run-id", "archived"]) == 0


def test_deepeval_resume_cache_isolated_by_effective_dataset(monkeypatch):
    from src.evaluation import judge_factory
    from src.evaluation.scoring import deepeval_layer

    dataset, results = _small_dataset_and_results()
    measured = []

    def measure(entry, result, judge):
        measured.append(entry["expected_answer"])
        return {"faithfulness": 1.0, "hallucination": 0.0, "correctness": 1.0}

    score = deepeval_layer.score_deepeval
    monkeypatch.setattr(judge_factory, "make_deepeval_judge_model", lambda: object())
    monkeypatch.setattr(deepeval_layer, "score_deepeval", lambda *a, **kw: score(*a, measure_fn=measure, **kw))

    def run(entries):
        return run_module._score_deepeval_payload(
            entries, results, run_id="cache-rebind", judge_id="local", generation_sha256="fixed-generation"
        )

    run(dataset)
    run(dataset)
    assert len(measured) == 2
    updated = [dict(entry, expected_answer="Updated reference") for entry in dataset]
    run(updated)
    run(updated)
    assert measured == [entry["expected_answer"] for entry in dataset] + ["Updated reference"] * 2


def test_ragas_layer_scores_through_cli_with_local_metrics(tmp_path, monkeypatch):
    from src.evaluation.scoring import ragas_layer

    class LocalMetric:
        async def ascore(self, **kwargs):
            return 0.75

    score = ragas_layer.score_ragas
    monkeypatch.setattr(ragas_layer, "score_ragas", lambda *a, **kw: score(
        *a, metrics_factory=lambda: {name: LocalMetric() for name in ragas_layer.DEFAULT_METRICS}, **kw
    ))
    dataset, results = _small_dataset_and_results()
    path = tmp_path / "dataset.json"
    path.write_text(json.dumps(dataset))
    _write_generation("ragas-local", dataset, results, dataset_path=path)
    assert main(["score", "--run-id", "ragas-local", "--layer", "ragas", "--judge", "local"]) == 0
    scored = json.loads((scores_dir("ragas-local") / "ragas.local.json").read_text())
    assert scored["n_scored"] == 2
    assert scored["n_failed"] == 0
    assert scored["per_entry"]["0"]["faithfulness"] == 0.75
    assert main(["aggregate", "--run-id", "ragas-local"]) == 0
