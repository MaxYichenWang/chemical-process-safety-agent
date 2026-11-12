"""Tests for src/web/runs.py."""

import json

from src.web.runs import get_run, list_runs


def _write_run(log_dir, run_id, ts, query, score=0.9, final_response="Answer text"):
    log_dir.mkdir(parents=True, exist_ok=True)
    data = {
        "run_id": run_id,
        "timestamp": ts,
        "query": query,
        "total_elapsed_s": 12.3,
        "final_state": {
            "final_response": final_response,
            "validation_score": score,
            "retry_count": 0,
        },
    }
    (log_dir / f"{ts}_{query[:10]}.json").write_text(json.dumps(data))


def test_list_runs_reads_both_new_and_legacy_log_dirs(tmp_path):
    _write_run(tmp_path / "logs" / "runs", "run-new", "2026-01-02T00-00-00", "new query")
    _write_run(tmp_path / "logs" / "cli", "run-legacy", "2026-01-01T00-00-00", "legacy query")

    result = list_runs(str(tmp_path), limit=10)
    assert result["total"] == 2
    run_ids = {r["run_id"] for r in result["runs"]}
    assert run_ids == {"run-new", "run-legacy"}


def test_list_runs_is_newest_first(tmp_path):
    _write_run(tmp_path / "logs" / "runs", "older", "2026-01-01T00-00-00", "q1")
    _write_run(tmp_path / "logs" / "runs", "newer", "2026-01-02T00-00-00", "q2")

    result = list_runs(str(tmp_path))
    assert [r["run_id"] for r in result["runs"]] == ["newer", "older"]


def test_list_runs_paginates(tmp_path):
    for i in range(5):
        _write_run(tmp_path / "logs" / "runs", f"run{i}", f"2026-01-0{i+1}T00-00-00", f"q{i}")

    page1 = list_runs(str(tmp_path), limit=2, offset=0)
    page2 = list_runs(str(tmp_path), limit=2, offset=2)
    assert page1["total"] == 5
    assert len(page1["runs"]) == 2
    assert len(page2["runs"]) == 2
    assert page1["runs"] != page2["runs"]


def test_get_run_finds_run_by_id_across_dirs(tmp_path):
    _write_run(tmp_path / "logs" / "cli", "target-run", "2026-01-01T00-00-00", "find me")
    result = get_run(str(tmp_path), "target-run")
    assert result is not None
    assert result["query"] == "find me"


def test_get_run_returns_none_when_missing(tmp_path):
    assert get_run(str(tmp_path), "does-not-exist") is None


def test_list_runs_skips_unreadable_files_without_crashing(tmp_path):
    log_dir = tmp_path / "logs" / "runs"
    log_dir.mkdir(parents=True)
    (log_dir / "2026-01-01T00-00-00_bad.json").write_text("{not valid json")
    _write_run(log_dir, "good-run", "2026-01-02T00-00-00", "ok query")

    result = list_runs(str(tmp_path))
    assert result["total"] == 2  # both files counted
    assert len(result["runs"]) == 1  # only the readable one summarized
    assert result["runs"][0]["run_id"] == "good-run"
