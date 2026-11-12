import json

import pytest

from scripts.demo_agentic_rag import (
    _console_emit,
    _trunc_ctx,
    run_streaming,
)


class _FakeGraph:
    """Minimal stand-in for the compiled LangGraph app's ``.stream()`` API."""

    def __init__(self, updates):
        self._updates = updates

    def stream(self, init, stream_mode="updates"):
        yield {"__start__": {}}
        for node, upd in self._updates:
            yield {node: upd}


def _sample_updates():
    return [
        ("PlanNode", {"planned_agents": ["osha_regs"], "sub_tasks": ["find the standard"]}),
        (
            "OshaRegsWorker",
            {"context_gathered": {"osha_regulations": ["chunk one", "chunk two"]}},
        ),
        (
            "CriticNode",
            {"validation_score": 0.91, "final_response": "The answer is 42.", "retry_count": 0},
        ),
    ]


def test_run_streaming_emits_expected_event_sequence(tmp_path):
    events = []
    final_state = run_streaming(
        "test query",
        project_root=str(tmp_path),
        emit=events.append,
        log_dir=tmp_path / "logs",
        build_graph_fn=lambda: _FakeGraph(_sample_updates()),
    )

    kinds = [e["event"] for e in events]
    assert kinds == [
        "graph_ready",
        "step_start", "step_complete",
        "step_start", "step_complete",
        "step_start", "step_complete",
        "pipeline_complete",
    ]
    assert final_state["validation_score"] == 0.91
    assert final_state["final_response"] == "The answer is 42."


def test_run_streaming_truncates_context_in_step_events(tmp_path):
    events = []
    run_streaming(
        "test query",
        project_root=str(tmp_path),
        emit=events.append,
        max_context_preview=4,
        log_dir=tmp_path / "logs",
        build_graph_fn=lambda: _FakeGraph(_sample_updates()),
    )
    worker_event = next(
        e for e in events if e.get("event") == "step_complete" and e.get("node") == "OshaRegsWorker"
    )
    ctx = worker_event["update"]["context_gathered"]["osha_regulations"]
    assert ctx["chunk_count"] == 2
    assert ctx["preview"] == "chun"  # truncated to max_context_preview=4


def test_run_streaming_writes_log_and_analytics_csv(tmp_path):
    log_dir = tmp_path / "logs"
    run_streaming(
        "test query",
        project_root=str(tmp_path),
        emit=lambda e: None,
        log_dir=log_dir,
        build_graph_fn=lambda: _FakeGraph(_sample_updates()),
    )

    json_logs = list(log_dir.glob("*.json"))
    assert len(json_logs) == 1
    saved = json.loads(json_logs[0].read_text())
    assert saved["metadata"]["validation_score"] == 0.91
    assert saved["query"] == "test query"

    csv_path = log_dir / "run_analytics.csv"
    assert csv_path.exists()
    rows = csv_path.read_text().strip().splitlines()
    assert len(rows) == 2  # header + one run


def test_run_streaming_persists_error_log_on_pipeline_failure(tmp_path):
    class _FailingGraph:
        def stream(self, init, stream_mode="updates"):
            yield {"PlanNode": {"planned_agents": []}}
            raise RuntimeError("boom")

    log_dir = tmp_path / "logs"
    with pytest.raises(RuntimeError):
        run_streaming(
            "q2",
            project_root=str(tmp_path),
            emit=lambda e: None,
            log_dir=log_dir,
            build_graph_fn=lambda: _FailingGraph(),
        )

    # one stable path for the run's whole lifecycle -- no separate
    # `error_*.json` file; the same path the in-progress record was written
    # to gets overwritten with `status: "error"` once the pipeline raises.
    json_logs = list(log_dir.glob("*.json"))
    assert len(json_logs) == 1
    payload = json.loads(json_logs[0].read_text())
    assert payload["status"] == "error"
    assert "boom" in payload["error"]


def test_run_streaming_leaves_in_progress_record_if_killed_mid_run(tmp_path):
    """a killed process must leave a readable, explicitly-incomplete record."""

    class _HangingGraph:
        def stream(self, init, stream_mode="updates"):
            yield {"PlanNode": {"planned_agents": ["osha_regs"]}}
            # Simulate the process dying here (SIGKILL) -- no further code in
            # run_streaming ever executes, so nothing after this point can
            # write a file. Only the incremental write from the PlanNode
            # step above should be on disk.
            raise SystemExit("simulated SIGKILL")

    log_dir = tmp_path / "logs"
    with pytest.raises(SystemExit):
        run_streaming(
            "q3",
            project_root=str(tmp_path),
            emit=lambda e: None,
            log_dir=log_dir,
            build_graph_fn=lambda: _HangingGraph(),
        )

    json_logs = list(log_dir.glob("*.json"))
    assert len(json_logs) == 1
    payload = json.loads(json_logs[0].read_text())
    assert payload["status"] == "in_progress"
    assert payload["metadata"]["finalized"] is False
    assert payload["metadata"]["node_names"] == ["PlanNode"]



def test_trunc_ctx_reports_chunk_count_and_total_chars():
    out = _trunc_ctx({"osha_regulations": ["abcdef", "gh"]}, max_preview=3)
    assert out["osha_regulations"] == {
        "chunk_count": 2,
        "total_chars": 8,
        "preview": "abc",
    }


def test_console_emit_does_not_raise(capsys):
    _console_emit({"event": "graph_ready", "elapsed_ms": 12})
    _console_emit(
        {
            "event": "step_complete",
            "node": "PlanNode",
            "elapsed_ms": 34,
            "update": {"planned_agents": ["osha_regs"]},
        }
    )
    _console_emit(
        {
            "event": "pipeline_complete",
            "elapsed_ms": 100,
            "state": {"validation_score": 0.9, "retry_count": 0, "final_response": "ok"},
        }
    )
    _console_emit({"event": "error", "message": "boom"})
    out = capsys.readouterr().out
    assert "Graph built" in out
    assert "ok" in out
