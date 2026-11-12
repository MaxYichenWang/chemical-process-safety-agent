"""Reference reducer for the pipeline UI state machine.

The browser implementation in src/web/static/app.js mirrors this Python reference.
Tests cover parallel worker activation after planning, malformed events, and
next-node prediction. Both reducers use the event contract in src/web/events.py."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from src.web.events import AGENT_KEY_TO_NODE, WORKER_NODES, predict_next_node


def initial_state() -> Dict[str, Any]:
    return {
        "phase": "idle",
        "completed_steps": [],
        "active_step": None,
        "active_workers": [],
        "worker_group_step": 0,
        "final_state": None,
        "error_message": None,
        # Internal counters (JS keeps these in refs; a pure Python
        # function keeps them in state instead -- same semantics).
        "_step_count": 0,
        "_last_elapsed_ms": 0,
    }


def _now_ms() -> int:
    return int(time.time() * 1000)


def reduce_event(
    state: Dict[str, Any], event: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    """Apply one backend event to the reducer state. Never raises.

    Malformed or partial event dicts (missing ``event``, unknown
    ``event`` value, ``None``) are returned unchanged -- mirrors
    ``usePipeline.ts``'s ``default: return prev`` branch, which is what
    keeps a dropped/garbled SSE frame from crashing the UI.
    """
    if not isinstance(event, dict):
        return state

    kind = event.get("event")
    now = _now_ms()

    if kind == "graph_ready":
        return {**state, "phase": "running"}

    if kind == "step_start":
        node = event.get("node")
        if any(w.get("node") == node for w in state["active_workers"]):
            return state
        if state.get("active_step") and state["active_step"].get("node") == node:
            return state
        step_count = state["_step_count"] + 1
        return {
            **state,
            "_step_count": step_count,
            "active_step": {"stepNumber": step_count, "node": node, "startedAtMs": now},
        }

    if kind == "step_complete":
        return _reduce_step_complete(state, event, now)

    if kind == "pipeline_complete":
        return {
            **state,
            "phase": "complete",
            "active_step": None,
            "final_state": event.get("state"),
        }

    if kind == "error":
        return {
            **state,
            "phase": "error",
            "active_step": None,
            "error_message": event.get("message"),
        }

    return state


def _reduce_step_complete(
    state: Dict[str, Any], event: Dict[str, Any], now: int
) -> Dict[str, Any]:
    node = event.get("node")
    update = event.get("update", {}) or {}
    elapsed_ms = event.get("elapsed_ms", 0)
    step_duration = elapsed_ms - state["_last_elapsed_ms"]

    # --- PlanNode fan-out special case ---
    planned = update.get("planned_agents")
    if node == "PlanNode" and planned:
        workers: List[Dict[str, Any]] = [
            {
                "node": AGENT_KEY_TO_NODE.get(agent_key, agent_key),
                "status": "active",
                "startedAtMs": now,
            }
            for agent_key in planned
        ]
        completed = {
            "stepNumber": (state["active_step"] or {}).get(
                "stepNumber", state["_step_count"]
            ),
            "node": node,
            "elapsedMs": elapsed_ms,
            "stepDurationMs": step_duration,
            "update": update,
            "errors": event.get("errors"),
        }
        curr_step = completed["stepNumber"]
        return {
            **state,
            "_step_count": curr_step,
            "_last_elapsed_ms": elapsed_ms,
            "completed_steps": state["completed_steps"] + [completed],
            "active_step": None,
            "active_workers": workers,
            "worker_group_step": curr_step + 1,
        }

    # --- Worker completion ---
    is_worker = node in WORKER_NODES
    if is_worker and state["active_workers"]:
        worker_duration = event.get("step_duration_s")
        if worker_duration is not None:
            worker_duration_ms = int(worker_duration * 1000)
        else:
            worker_duration_ms = max(0, elapsed_ms - state["_last_elapsed_ms"])

        updated_workers = [
            (
                {
                    **w,
                    "status": "done",
                    "completedAtMs": now,
                    "durationMs": now - w["startedAtMs"],
                    "update": update,
                    "errors": event.get("errors"),
                }
                if w["node"] == node
                else w
            )
            for w in state["active_workers"]
        ]
        all_done = all(w["status"] == "done" for w in updated_workers)

        step_count = state["_step_count"] + 1
        worker_completed = {
            "stepNumber": step_count,
            "node": node,
            "elapsedMs": elapsed_ms,
            "stepDurationMs": worker_duration_ms,
            "update": update,
            "errors": event.get("errors"),
        }

        next_active = None
        if all_done:
            step_count += 1
            next_active = {
                "stepNumber": step_count,
                "node": "SynthesizeNode",
                "startedAtMs": now,
            }

        return {
            **state,
            "_step_count": step_count,
            "_last_elapsed_ms": elapsed_ms,
            "completed_steps": state["completed_steps"] + [worker_completed],
            "active_workers": updated_workers,
            "active_step": next_active,
        }

    # --- Normal, linear step completion ---
    completed = {
        "stepNumber": (state["active_step"] or {}).get(
            "stepNumber", state["_step_count"]
        ),
        "node": node,
        "elapsedMs": elapsed_ms,
        "stepDurationMs": step_duration,
        "update": update,
        "errors": event.get("errors"),
    }

    next_node = predict_next_node(node, update)
    next_active = None
    step_count = state["_step_count"]
    if next_node:
        step_count += 1
        next_active = {"stepNumber": step_count, "node": next_node, "startedAtMs": now}

    return {
        **state,
        "_step_count": step_count,
        "_last_elapsed_ms": elapsed_ms,
        "completed_steps": state["completed_steps"] + [completed],
        "active_step": next_active,
    }
