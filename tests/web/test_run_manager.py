"""Tests for src/web/run_manager.py."""

from __future__ import annotations

import asyncio

import pytest

from src.web.run_manager import Run, RunManager, render_markdown_safely


def test_render_markdown_safely_escapes_script_tags():
    """ acceptance: HTML in model output must be escaped, not passed through."""
    raw = "Here is a finding: <script>alert('xss')</script> and **bold** text."
    html_out = render_markdown_safely(raw)
    assert "<script>" not in html_out
    assert "&lt;script&gt;" in html_out
    assert "<strong>bold</strong>" in html_out


def test_render_markdown_safely_renders_tables_and_headings():
    raw = "# Title\n\n| a | b |\n|---|---|\n| 1 | 2 |\n"
    html_out = render_markdown_safely(raw)
    assert "<h1>Title</h1>" in html_out
    assert "<table>" in html_out


@pytest.mark.asyncio
async def test_start_run_assigns_sequential_ids_and_retains_events(monkeypatch, tmp_path):
    """POST /api/query's underlying flow: run_streaming's emitted events get
    sequential IDs and are retained for SSE replay."""

    def fake_run_streaming(query, *, project_root, emit, log_dir=None, build_graph_fn=None, run_id=None):
        assert build_graph_fn is not None
        assert run_id is not None
        emit({"event": "graph_ready", "elapsed_ms": 1})
        emit({"event": "step_start", "node": "PlanNode"})
        emit(
            {
                "event": "step_complete",
                "node": "PlanNode",
                "elapsed_ms": 5,
                "update": {"planned_agents": ["osha_regs"]},
            }
        )
        emit({"event": "pipeline_complete", "elapsed_ms": 10, "state": {"final_response": "**ok**"}})
        return {"final_response": "**ok**"}

    monkeypatch.setattr("scripts.demo_agentic_rag.run_streaming", fake_run_streaming)

    manager = RunManager(
        project_root=str(tmp_path),
        build_graph_fn=lambda: object(),
        log_dir=tmp_path / "logs",
        max_concurrency=3,
    )

    run_id = await manager.start_run("what is osha 1910?")
    assert run_id

    # Give the background task (spawned via asyncio.create_task) a moment to finish.
    for _ in range(50):
        run = manager.get(run_id)
        if run.done:
            break
        await asyncio.sleep(0.02)

    run = manager.get(run_id)
    assert run.done
    ids = [e["id"] for e in run.events]
    assert ids == sorted(ids)
    assert ids == list(range(1, len(ids) + 1))

    kinds = [e["event"] for e in run.events]
    assert kinds == ["graph_ready", "step_start", "step_complete", "pipeline_complete"]

    # pipeline_complete must carry both raw markdown and rendered HTML.
    final_event = run.events[-1]
    assert final_event["state"]["final_response"] == "**ok**"
    assert "<strong>ok</strong>" in final_event["state"]["final_response_html"]
    assert manager.active_run_ids() == []  # done runs are not "active"


@pytest.mark.asyncio
async def test_stream_from_does_not_redeliver_events_already_in_the_queue():
    """Regression test for a race found via TestClient: if a client connects
    after some events have already been recorded (present in both
    `run.events` and still sitting, undrained, in `run.queue`), the queue
    must not redeliver what the events-list replay already yielded."""
    run = Run("run-x", "some query")

    def publish(event):
        record = {"id": run.next_seq(), **event}
        run.events.append(record)
        run.queue.put_nowait(record)
        return record

    publish({"event": "graph_ready"})
    publish({"event": "step_start", "node": "PlanNode"})
    # Note: neither event has been drained from `run.queue` yet -- this is
    # exactly the state a freshly-connecting client observes.

    delivered = []
    async for record in run.stream_from(0, is_disconnected=None):
        delivered.append(record)
        if len(delivered) == 2:
            # Stand in for the client disconnecting once it has what it needs;
            # avoids blocking forever waiting on more queue items in this test.
            break

    ids = [r["id"] for r in delivered]
    assert ids == [1, 2]  # not [1, 2, 1, 2] -- no redelivery


@pytest.mark.asyncio
async def test_stream_from_tails_new_events_after_replay():
    run = Run("run-y", "q")

    def publish(event):
        record = {"id": run.next_seq(), **event}
        run.events.append(record)
        run.queue.put_nowait(record)
        return record

    publish({"event": "graph_ready"})

    collected = []

    async def consume():
        async for record in run.stream_from(0, is_disconnected=None):
            collected.append(record)
            if record.get("event") in ("pipeline_complete", "error"):
                return

    task = asyncio.create_task(consume())
    await asyncio.sleep(0.05)  # let it replay id=1 and start tailing the queue
    publish({"event": "pipeline_complete", "state": {}})
    await asyncio.wait_for(task, timeout=2.0)

    assert [r["id"] for r in collected] == [1, 2]
    assert collected[-1]["event"] == "pipeline_complete"


@pytest.mark.parametrize("terminal", ["pipeline_complete", "error"])
@pytest.mark.asyncio
async def test_terminal_event_published_during_replay_is_delivered_once(terminal):
    run = Run("replay-race", "q")
    first = {"id": run.next_seq(), "event": "step_complete"}
    run.events.append(first)
    run.queue.put_nowait(first)
    stream = run.stream_from(0)
    assert await anext(stream) == first

    final = {"id": run.next_seq(), "event": terminal}
    run.events.append(final)
    run.queue.put_nowait(final)
    run.done = True

    assert [record async for record in stream] == [final]
    assert [record async for record in run.stream_from(first["id"])] == [final]
    assert [record async for record in run.stream_from(final["id"])] == []
