"""Tests for parallel worker activation and malformed-event resilience."""

from src.web.reducer import initial_state, reduce_event


def test_plan_node_fan_out_activates_all_workers_under_one_step_number():
    state = initial_state()
    state = reduce_event(state, {"event": "graph_ready", "elapsed_ms": 5})
    state = reduce_event(
        state,
        {
            "event": "step_complete",
            "node": "PlanNode",
            "elapsed_ms": 120,
            "update": {"planned_agents": ["osha_regs", "csb_incident", "data_analyst"]},
        },
    )

    assert state["active_step"] is None
    assert len(state["active_workers"]) == 3
    nodes = {w["node"] for w in state["active_workers"]}
    assert nodes == {"OshaRegsWorker", "IncidentWorker", "DataAnalystWorker"}
    assert all(w["status"] == "active" for w in state["active_workers"])
    # All three share one step-group number, not three distinct step numbers.
    assert state["worker_group_step"] > 0
    assert len(state["completed_steps"]) == 1
    assert state["completed_steps"][0]["node"] == "PlanNode"


def test_workers_complete_independently_then_advance_to_synthesize_node():
    state = initial_state()
    state = reduce_event(
        state,
        {
            "event": "step_complete",
            "node": "PlanNode",
            "elapsed_ms": 100,
            "update": {"planned_agents": ["osha_regs", "csb_incident"]},
        },
    )

    # First worker completes -- not all done yet, no next-active prediction.
    state = reduce_event(
        state,
        {"event": "step_complete", "node": "OshaRegsWorker", "elapsed_ms": 200, "update": {}},
    )
    assert state["active_step"] is None
    statuses = {w["node"]: w["status"] for w in state["active_workers"]}
    assert statuses["OshaRegsWorker"] == "done"
    assert statuses["IncidentWorker"] == "active"

    # Second worker completes -- all done, SynthesizeNode predicted next.
    state = reduce_event(
        state,
        {"event": "step_complete", "node": "IncidentWorker", "elapsed_ms": 210, "update": {}},
    )
    assert all(w["status"] == "done" for w in state["active_workers"])
    assert state["active_step"]["node"] == "SynthesizeNode"


def test_step_start_does_not_duplicate_an_already_tracked_worker():
    state = initial_state()
    state = reduce_event(
        state,
        {
            "event": "step_complete",
            "node": "PlanNode",
            "elapsed_ms": 100,
            "update": {"planned_agents": ["osha_regs"]},
        },
    )
    before = state["_step_count"]
    state = reduce_event(state, {"event": "step_start", "node": "OshaRegsWorker"})
    assert state["_step_count"] == before  # unchanged -- already tracked


def test_pipeline_complete_sets_final_state_and_completes_phase():
    state = initial_state()
    state = reduce_event(
        state, {"event": "pipeline_complete", "elapsed_ms": 5000, "state": {"final_response": "ok"}}
    )
    assert state["phase"] == "complete"
    assert state["active_step"] is None
    assert state["final_state"] == {"final_response": "ok"}


def test_error_event_sets_error_phase_and_message():
    state = initial_state()
    state = reduce_event(state, {"event": "error", "message": "boom"})
    assert state["phase"] == "error"
    assert state["error_message"] == "boom"
    assert state["active_step"] is None


def test_malformed_or_partial_events_do_not_crash_the_reducer():
    state = initial_state()

    # None event (e.g. a blank/garbled SSE frame).
    state = reduce_event(state, None)
    assert state["phase"] == "idle"

    # Missing "event" key entirely.
    state = reduce_event(state, {"node": "PlanNode"})
    assert state["phase"] == "idle"

    # Unknown event name.
    state = reduce_event(state, {"event": "not_a_real_event", "foo": "bar"})
    assert state["phase"] == "idle"

    # step_complete with a missing "update" key -- must default rather than raise.
    state = reduce_event(state, {"event": "step_complete", "node": "PlanNode", "elapsed_ms": 1})
    assert state["phase"] == "idle"
    assert state["completed_steps"][-1]["update"] == {}

    # Should still be able to process a well-formed event afterward.
    state = reduce_event(state, {"event": "graph_ready", "elapsed_ms": 1})
    assert state["phase"] == "running"
