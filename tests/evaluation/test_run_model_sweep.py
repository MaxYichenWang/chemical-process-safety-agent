"""Tests for the model-sweep execution wrapper."""

import os
import json
import subprocess
import sys
from unittest.mock import MagicMock, call

import pytest
import yaml

from scripts.reusable import run_model_sweep
from scripts.reusable.run_model_sweep import _eval_cmd, _resolve_eval_defaults
from tests.evaluation.evaluators.test_end_to_end_evaluator import operational_evaluator  # noqa: F401


def test_cli_tier_override_resets_sample_per_source_to_tier_default():
    defaults = {"tier": "T1", "sample_per_source": 3, "seed": 42}

    resolved = _resolve_eval_defaults(defaults, tier_override="T0")

    assert resolved["tier"] == "T0"
    assert resolved["sample_per_source"] == 1
    assert _eval_cmd(resolved)[-2:] == ["--sample-per-source", "1"]


def test_cli_tier_t2_omits_sample_per_source_for_full_dataset():
    defaults = {"tier": "T1", "sample_per_source": 3, "seed": 42}

    resolved = _resolve_eval_defaults(defaults, tier_override="T2")

    assert resolved["tier"] == "T2"
    assert resolved["sample_per_source"] is None
    assert "--sample-per-source" not in _eval_cmd(resolved)


def test_cli_sample_override_wins_over_tier_default():
    defaults = {"tier": "T1", "sample_per_source": 3, "seed": 42}

    resolved = _resolve_eval_defaults(
        defaults,
        tier_override="T0",
        sample_override=2,
    )

    assert resolved["tier"] == "T0"
    assert resolved["sample_per_source"] == 2
    assert _eval_cmd(resolved)[-2:] == ["--sample-per-source", "2"]


def test_include_adversarial_default_adds_evaluator_flag():
    defaults = {
        "tier": "T1",
        "sample_per_source": 3,
        "seed": 42,
        "include_adversarial": True,
    }

    assert "--include-adversarial" in _eval_cmd(defaults)


def test_cli_include_adversarial_override_wins_over_config_default():
    defaults = {"tier": "T1", "sample_per_source": 3, "seed": 42}

    resolved = _resolve_eval_defaults(
        defaults,
        include_adversarial_override=True,
    )

    assert resolved["include_adversarial"] is True
    assert "--include-adversarial" in _eval_cmd(resolved)


def test_cli_max_concurrent_override_wins():
    defaults = {"tier": "T1", "sample_per_source": 3, "seed": 42}

    resolved = _resolve_eval_defaults(
        defaults,
        max_concurrent_override=12,
    )

    assert resolved["max_concurrent"] == 12
    cmd = _eval_cmd(resolved)
    idx = cmd.index("--max-concurrent")
    assert cmd[idx + 1] == "12"


@pytest.fixture
def sweep_invocation(tmp_path, monkeypatch):
    baseline = tmp_path / "baseline.json"
    baseline.write_text("{}")
    config = tmp_path / "sweep.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "defaults": {"baseline_results": str(baseline)},
                "runs": [{"name": "first"}, {"name": "second"}],
            }
        )
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(run_model_sweep, "PROJECT_ROOT", tmp_path)
    monkeypatch.delenv("EVAL_DATASET_DIR", raising=False)
    monkeypatch.setattr(sys, "argv", ["run_model_sweep.py", "--config", str(config)])
    analyzer = MagicMock()
    analyzer.generate_report.return_value = "Comparison report"
    monkeypatch.setattr(
        "src.evaluation.analyzers.improvement_analyzer.ImprovementAnalyzer",
        lambda: analyzer,
    )
    return baseline, analyzer


@pytest.mark.parametrize("path_source", ["environment", "dotenv"])
def test_sweep_compares_its_own_results_in_custom_directory(
    sweep_invocation,
    tmp_path,
    monkeypatch,
    path_source,
):
    baseline, analyzer = sweep_invocation
    dataset_dir = tmp_path / "custom-evaluation"
    if path_source == "environment":
        monkeypatch.setenv("EVAL_DATASET_DIR", str(dataset_dir))
    else:
        (tmp_path / ".env").write_text(f"EVAL_DATASET_DIR={dataset_dir}\n")
        monkeypatch.setattr(os, "environ", os.environ.copy())
    results_dir = dataset_dir / "results"
    results_dir.mkdir(parents=True)
    stale = results_dir / "stale_results.json"
    stale.write_text("{}")
    os.utime(stale, (4102444800, 4102444800))
    produced = []

    def evaluate(cmd, env):
        assert env["EVAL_DATASET_DIR"] == str(dataset_dir)
        run_id = cmd[cmd.index("--run-id") + 1]
        path = results_dir / f"{run_id}_results.json"
        path.write_text("{}")
        produced.append(path)
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(run_model_sweep.subprocess, "run", evaluate)

    assert run_model_sweep.main() == 0
    assert len(set(produced)) == 2
    assert analyzer.compare_runs.call_args_list == [
        call(baseline, produced[0]),
        call(baseline, produced[1]),
    ]
    assert (results_dir / "sweep_first_vs_baseline.md").is_file()
    assert (results_dir / "sweep_second_vs_baseline.md").is_file()


@pytest.mark.parametrize("failure", ["exit", "missing-result", "launch"])
def test_sweep_returns_failure_and_continues_other_runs(
    sweep_invocation,
    monkeypatch,
    failure,
):
    _, analyzer = sweep_invocation
    invocations = []

    def evaluate(cmd, env):
        invocations.append(cmd)
        if failure == "launch":
            raise OSError("Could not start evaluator")
        return subprocess.CompletedProcess(cmd, 17 if failure == "exit" else 0)

    monkeypatch.setattr(run_model_sweep.subprocess, "run", evaluate)

    assert run_model_sweep.main() == 1
    assert len(invocations) == 2
    analyzer.compare_runs.assert_not_called()


def test_sweep_rejects_missing_baseline_before_evaluation(
    sweep_invocation,
    monkeypatch,
):
    baseline, _ = sweep_invocation
    baseline.unlink()
    evaluate = MagicMock()
    monkeypatch.setattr(run_model_sweep.subprocess, "run", evaluate)

    with pytest.raises(SystemExit) as exc:
        run_model_sweep.main()

    assert exc.value.code == 2
    evaluate.assert_not_called()


def test_sweep_propagates_actual_evaluator_pipeline_failure(
    operational_evaluator, sweep_invocation, tmp_path, monkeypatch
):
    """A real evaluate() failure must not reach baseline comparison as success."""
    from src.evaluation.evaluators import end_to_end_evaluator as runner

    _, analyzer = sweep_invocation
    dataset = tmp_path / "dataset.json"
    dataset.write_text(json.dumps([
        {"query": f"Synthetic {i}", "expected_answer": "Synthetic", "ground_truth_chunks": []}
        for i in range(2)
    ]))
    operational_evaluator.graph.ainvoke.side_effect = ValueError("Synthetic provider failure")
    original_command = run_model_sweep._eval_cmd
    monkeypatch.setattr(run_model_sweep, "_eval_cmd", lambda defaults: original_command(defaults) + ["--dataset", str(dataset)])
    statuses = []

    def evaluate(command, env):
        with monkeypatch.context() as child:
            child.setattr(sys, "argv", [command[2], *command[3:]])
            status = runner.main()
        statuses.append(status)
        return subprocess.CompletedProcess(command, status)

    monkeypatch.setattr(run_model_sweep.subprocess, "run", evaluate)
    assert run_model_sweep.main() == 1
    assert statuses == [1, 1]
    analyzer.compare_runs.assert_not_called()
