#!/usr/bin/env python3
"""Shared entry point over ``build_graph()`` (``src/agentic_rag/graph.py``).

Run the pipeline from a terminal::

    PYTHONPATH=. python scripts/demo_agentic_rag.py --query "..."

prints a readable trace (node-by-node progress, retrieved-context sizes,
the critic's validation score, and the final answer) to stdout.

The same ``run_streaming()`` entry point also backs ``src/web``'s FastAPI
server, so the CLI and the web
UI can no longer drift apart.

Only depends on ``src.agentic_rag.graph`` (the intelligence layer) plus
stdlib -- no ChromaDB/API key is touched until ``build_graph()`` is called
inside :func:`run_streaming`, so ``--help`` and argument parsing work with
no corpus or credentials present.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import re
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

Event = Dict[str, Any]
EmitFn = Callable[[Event], None]

# Overridable via env vars so the web server (src/web/config.py) and console
# callers can both tune truncation without code changes.
DEFAULT_MAX_CONTEXT_PREVIEW = int(os.environ.get("CPS_MAX_CONTEXT_PREVIEW", "500"))
DEFAULT_MAX_LOG_CONTEXT_CHARS = int(os.environ.get("CPS_MAX_LOG_CONTEXT_CHARS", "2000"))

INITIAL_STATE: Dict[str, Any] = {
    "query": "",
    "sub_tasks": [],
    "active_agent": "",
    "pending_agents": [],
    "planned_agents": [],
    "planning_fast_routed": False,
    "planning_fast_route_patterns": [],
    "context_gathered": {},
    "draft_response": "",
    "final_response": "",
    "citations": [],
    "validation_score": 0.0,
    "critic_feedback": "",
    "retry_count": 0,
    "llm_calls": [],
    "retrieval_details": [],
    "worker_errors": [],
}


def _safe(obj: Any) -> Any:
    """Recursively coerce *obj* into something ``json.dumps`` can always handle."""
    if isinstance(obj, dict):
        return {k: _safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_safe(i) for i in obj]
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    return str(obj)


def _apply_state_update(state: Dict[str, Any], update: Dict[str, Any]) -> None:
    """Merge a single node's ``stream_mode='updates'`` delta into ``state``.

    Mirrors ``AgentState``'s ``operator.add`` and ``_merge_context`` reducers
    so ``state`` accumulates ``llm_calls``, ``retrieval_details``,
    ``citations``, ``worker_errors``, and ``context_gathered`` across steps
    rather than overwriting earlier nodes' entries when a downstream node
    returns its own per-step delta (e.g. ``PlanNode`` fast-routing with
    ``llm_calls: []`` after ``CondenseNode`` recorded a condensation call).
    """
    for k, v in update.items():
        if k == "context_gathered" and isinstance(v, dict):
            merged = dict(state.get("context_gathered") or {})
            for coll, chunks in v.items():
                if (
                    coll in merged
                    and isinstance(merged[coll], list)
                    and isinstance(chunks, list)
                ):
                    merged[coll] = merged[coll] + chunks
                else:
                    merged[coll] = chunks
            state["context_gathered"] = merged
        elif k in ("llm_calls", "retrieval_details", "citations", "worker_errors") and isinstance(v, list):
            existing = list(state.get(k) or [])
            for item in v:
                if item not in existing:
                    existing.append(item)
            state[k] = existing
        else:
            state[k] = v


def _trunc_ctx(ctx: Dict[str, Any], max_preview: int) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for coll, chunks in ctx.items():
        if isinstance(chunks, list):
            total_chars = sum(len(str(c)) for c in chunks)
            out[coll] = {
                "chunk_count": len(chunks),
                "total_chars": total_chars,
                "preview": str(chunks[0])[:max_preview] if chunks else "",
            }
        else:
            out[coll] = chunks
    return out


def _trunc_update(update: Dict[str, Any], max_preview: int) -> Dict[str, Any]:
    out = dict(update)
    if "context_gathered" in out and isinstance(out["context_gathered"], dict):
        out["context_gathered"] = _trunc_ctx(out["context_gathered"], max_preview)
    return out


def _trunc_state(state: Dict[str, Any], max_preview: int) -> Dict[str, Any]:
    out = dict(state)
    if "context_gathered" in out and isinstance(out["context_gathered"], dict):
        out["context_gathered"] = _trunc_ctx(out["context_gathered"], max_preview)
    if len(out.get("draft_response", "")) > 1000:
        out["draft_response"] = out["draft_response"][:1000] + "..."
    return out


def _sum_llm_latency(llm_calls: List[Dict[str, Any]], call_purpose: str) -> float:
    return round(
        sum(
            (c.get("latency_s") or 0.0)
            for c in llm_calls
            if c.get("call_purpose") == call_purpose
        ),
        4,
    )


def _append_run_analytics_csv(
    csv_path: Path,
    *,
    run_id: str,
    timestamp: str,
    query: str,
    total_elapsed_s: float,
    total_tokens: int,
    total_cost_usd: float,
    retry_count: int,
    validation_score: float,
    worker_count: int,
    worker_errors_count: int,
    planning_latency_s: float,
    synthesis_latency_s: float,
    critic_latency_s: float,
) -> None:
    header = [
        "run_id",
        "timestamp",
        "query",
        "total_elapsed_s",
        "total_tokens",
        "total_cost_usd",
        "retry_count",
        "validation_score",
        "worker_count",
        "worker_errors_count",
        "planning_latency_s",
        "synthesis_latency_s",
        "critic_latency_s",
    ]
    row = [
        run_id,
        timestamp,
        query,
        round(total_elapsed_s, 3),
        total_tokens,
        round(total_cost_usd, 8),
        retry_count,
        validation_score,
        worker_count,
        worker_errors_count,
        planning_latency_s,
        synthesis_latency_s,
        critic_latency_s,
    ]
    is_new_file = not csv_path.exists()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if is_new_file:
            writer.writerow(header)
        writer.writerow(row)


def _make_log_path(log_dir: Path, run_id: str, query: str, ts: datetime) -> Path:
    """One stable filename for a run's entire lifecycle (checklist W28).

    Computed once at the start of ``run_streaming`` and reused for every
    incremental write plus the final success/error write, so a run's record
    lives at a single path throughout -- no ``error_`` prefix distinguishing
    outcomes by filename; the JSON body's ``status`` field does that instead
    (design doc §12.4/§12.5 measure 4). The ``run_id`` prefix keeps two runs
    of the same query started within the same second from colliding, which
    matters once queries can run concurrently (checklist W20).
    """
    slug = re.sub(r"[^a-z0-9]+", "_", query.lower()).strip("_")[:40]
    return log_dir / f"{ts.strftime('%Y-%m-%dT%H-%M-%S')}_{run_id[:8]}_{slug}.json"


def _write_incremental_log(
    fp: Path,
    run_id: str,
    query: str,
    ts: datetime,
    steps: List[Dict[str, Any]],
    state: Dict[str, Any],
    max_log_context_chars: int,
    *,
    session_id: Optional[str] = None,
) -> None:
    """Atomically save an ``in_progress`` snapshot after each step.

    Interrupted runs retain a readable partial record.
    """
    fp.parent.mkdir(parents=True, exist_ok=True)
    trunc_state = {
        k: v
        for k, v in _trunc_state(state, max_log_context_chars).items()
        if k not in ("llm_calls", "retrieval_details")
    }
    data = {
        "run_id": run_id,
        "session_id": session_id or run_id,
        "timestamp": ts.isoformat(),
        "status": "in_progress",
        "query": query,
        "steps": _safe(steps),
        "final_state": _safe(trunc_state),
        "metadata": {
            "step_count": len(steps),
            "node_names": list(dict.fromkeys(s["node_name"] for s in steps)),
            "finalized": False,
        },
    }
    tmp_fp = fp.with_suffix(".json.tmp")
    tmp_fp.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    tmp_fp.replace(fp)


def _save_log(
    log_dir: Path,
    query: str,
    steps: List[Dict[str, Any]],
    state: Dict[str, Any],
    total_s: float,
    run_id: str,
    max_log_context_chars: int,
    *,
    fp: Path,
    ts: datetime,
    session_id: Optional[str] = None,
) -> None:
    from src.data_processing.config import (
        AGENT_CRITIC_MODEL,
        AGENT_REASONING_MODEL,
        AGENT_SUPERVISOR_MODEL,
        AGENT_WORKER_MODEL,
        EMBEDDING_DIMENSIONS,
        EMBEDDING_MODEL,
    )

    log_dir.mkdir(parents=True, exist_ok=True)

    log_ctx: Dict[str, Any] = {}
    for coll, chunks in state.get("context_gathered", {}).items():
        if isinstance(chunks, list):
            total_chars = sum(len(str(c)) for c in chunks)
            log_ctx[coll] = {
                "chunk_count": len(chunks),
                "total_chars": total_chars,
                "preview": str(chunks[0])[:max_log_context_chars] if chunks else "",
            }
        else:
            log_ctx[coll] = chunks
    log_state = dict(state)
    log_state["context_gathered"] = log_ctx
    log_state.pop("llm_calls", None)
    log_state.pop("retrieval_details", None)

    llm_calls = state.get("llm_calls", [])
    retrieval_details = state.get("retrieval_details", [])

    total_prompt_tokens = sum(c.get("prompt_tokens") or 0 for c in llm_calls)
    total_completion_tokens = sum(c.get("completion_tokens") or 0 for c in llm_calls)
    total_tokens = sum(c.get("total_tokens") or 0 for c in llm_calls)
    total_estimated_cost_usd = sum(c.get("estimated_cost_usd") or 0 for c in llm_calls)
    planning_latency_s = _sum_llm_latency(llm_calls, "planning")
    synthesis_latency_s = _sum_llm_latency(llm_calls, "synthesis")
    critic_latency_s = _sum_llm_latency(llm_calls, "validation")
    worker_node_names = {
        s["node_name"] for s in steps if s.get("node_name", "").endswith("Worker")
    }

    data = {
        "run_id": run_id,
        "session_id": session_id or run_id,
        "timestamp": ts.isoformat(),
        "status": "complete",
        "query": query,
        "steps": steps,
        "llm_calls": _safe(llm_calls),
        "retrieval_details": _safe(retrieval_details),
        "final_state": _safe(log_state),
        "total_elapsed_s": round(total_s, 3),
        "environment": {
            "supervisor_model": AGENT_SUPERVISOR_MODEL,
            "worker_model": AGENT_WORKER_MODEL,
            "reasoning_model": AGENT_REASONING_MODEL,
            "critic_model": AGENT_CRITIC_MODEL,
            "embedding_model": EMBEDDING_MODEL,
            "embedding_dimensions": EMBEDDING_DIMENSIONS,
        },
        "metadata": {
            "step_count": len(steps),
            "node_names": list(dict.fromkeys(s["node_name"] for s in steps)),
            "total_llm_calls": len(llm_calls),
            "total_prompt_tokens": total_prompt_tokens,
            "total_completion_tokens": total_completion_tokens,
            "total_tokens": total_tokens,
            "total_estimated_cost_usd": round(total_estimated_cost_usd, 8),
            "retry_count": state.get("retry_count", 0),
            "validation_score": state.get("validation_score", 0.0),
            "finalized": True,
        },
    }
    fp.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    _append_run_analytics_csv(
        log_dir / "run_analytics.csv",
        run_id=run_id,
        timestamp=ts.isoformat(),
        query=query,
        total_elapsed_s=total_s,
        total_tokens=total_tokens,
        total_cost_usd=total_estimated_cost_usd,
        retry_count=state.get("retry_count", 0),
        validation_score=state.get("validation_score", 0.0),
        worker_count=len(worker_node_names),
        worker_errors_count=len(state.get("worker_errors", [])),
        planning_latency_s=planning_latency_s,
        synthesis_latency_s=synthesis_latency_s,
        critic_latency_s=critic_latency_s,
    )


def _save_error_log(
    log_dir: Path,
    query: str,
    steps: List[Dict[str, Any]],
    state: Optional[Dict[str, Any]],
    error_msg: str,
    error_tb: str,
    run_id: str,
    *,
    fp: Path,
    ts: datetime,
    session_id: Optional[str] = None,
) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_state = _safe(state) if state else {}
    llm_calls = state.get("llm_calls", []) if state else []
    data = {
        "run_id": run_id,
        "session_id": session_id or run_id,
        "timestamp": ts.isoformat(),
        "status": "error",
        "query": query,
        "error": error_msg,
        "traceback": error_tb,
        "steps": _safe(steps) if steps else [],
        "llm_calls": _safe(llm_calls),
        "final_state": log_state,
        "metadata": {
            "step_count": len(steps) if steps else 0,
            "node_names": (
                list(dict.fromkeys(s["node_name"] for s in steps)) if steps else []
            ),
            "total_llm_calls": len(llm_calls),
            "finalized": False,
        },
    }
    fp.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


def run_streaming(
    query: str,
    *,
    project_root: str,
    emit: EmitFn,
    max_context_preview: int = DEFAULT_MAX_CONTEXT_PREVIEW,
    max_log_context_chars: int = DEFAULT_MAX_LOG_CONTEXT_CHARS,
    log_dir: Optional[Path] = None,
    build_graph_fn: Optional[Callable[[], Any]] = None,
    run_id: Optional[str] = None,
    history: Optional[List[Dict[str, str]]] = None,
    session_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Run the agentic graph on *query*, calling ``emit(event)`` per step.

    Shared by the plain-Python console demo in :func:`main` and the FastAPI
    web server (``src/web/run_manager.py``) -- identical event shapes,
    identical logging/analytics side effects, so the two front ends can
    never silently diverge.

    ``run_id``: optional caller-supplied identifier for this run's log file
    and analytics row. The web server (``src/web/run_manager.py``) passes
    its own run ID here so the ID used for ``GET /api/stream/{run_id}`` and
    ``GET /api/runs/{run_id}`` matches the one written to disk -- without
    this, the server and the log file would each mint an independent UUID
    and history lookups could never join the two. Defaults to a fresh
    ``uuid.uuid4()`` for the CLI/console callers that don't care.

    Returns the final pipeline state dict (already truncated for display).
    Raises on pipeline failure, after persisting an error log -- callers
    decide how to present that to their respective UI.
    """
    if build_graph_fn is None:
        from src.agentic_rag.graph import (
            build_graph as build_graph_fn,  # type: ignore[assignment]
        )

    log_dir = log_dir if log_dir is not None else Path(project_root) / "logs" / "cli"

    run_id = run_id if run_id is not None else str(uuid.uuid4())
    # Wall clock is used only for human-readable timestamps/filenames, never
    # for measuring elapsed time -- see design doc §12.2/§12.5 measure 3.
    # `time.time()` can jump (NTP steps, DST, laptop sleep/wake), which
    # silently corrupted every duration in `run_analytics.csv` and the SSE
    # `elapsed_ms` fields; `time.monotonic()` cannot go backwards or jump.
    ts_start = datetime.now(timezone.utc)
    fp = _make_log_path(log_dir, run_id, query, ts_start)

    t0 = time.monotonic() * 1000
    app = build_graph_fn()
    emit({"event": "graph_ready", "elapsed_ms": round(time.monotonic() * 1000 - t0)})

    init = copy.deepcopy(INITIAL_STATE)
    init["query"] = query
    init["raw_query"] = query
    init["conversation_history"] = list(history) if history else []
    state = dict(init)
    steps: List[Dict[str, Any]] = []
    log_t0 = time.monotonic()
    last_step_elapsed_s = 0.0

    from src.agentic_rag.supervisor import _TOKEN_EMITTER

    def _stream_token(token_str: str) -> None:
        emit({"event": "synthesis_token", "token": token_str})

    token_ctx = _TOKEN_EMITTER.set(_stream_token)
    try:
        for ev in app.stream(init, stream_mode="updates"):
            for node, upd in ev.items():
                if node == "__start__":
                    continue
                ems = round(time.monotonic() * 1000 - t0)
                emit({"event": "step_start", "node": node})
                s = _safe(upd)
                _apply_state_update(state, s)
                elapsed_s = round(time.monotonic() - log_t0, 3)
                step_duration_s = round(max(elapsed_s - last_step_elapsed_s, 0.0), 3)
                last_step_elapsed_s = elapsed_s
                step_data = {
                    "node_name": node,
                    "elapsed_s": elapsed_s,
                    "step_duration_s": step_duration_s,
                    "update_keys": list(s.keys()),
                    "update": s,
                }
                # Strip bulky logging fields from step updates to keep events lean.
                trunc_s = {
                    k: v
                    for k, v in s.items()
                    if k not in ("llm_calls", "retrieval_details")
                }
                steps.append(step_data)
                node_errors = [
                    e for e in state.get("worker_errors", []) if e.get("node") == node
                ]
                step_event: Dict[str, Any] = {
                    "event": "step_complete",
                    "node": node,
                    "elapsed_ms": ems,
                    "step_duration_s": step_duration_s,
                    "update": _trunc_update(trunc_s, max_context_preview),
                }
                if node_errors:
                    step_event["errors"] = node_errors
                emit(step_event)
                # W28: leaves a readable, explicitly-incomplete record if the
                # process dies before this run concludes (design doc §12.4).
                _write_incremental_log(
                    fp,
                    run_id,
                    query,
                    ts_start,
                    steps,
                    state,
                    max_log_context_chars,
                    session_id=session_id,
                )
    except Exception as stream_exc:
        import traceback as _tb

        err_msg = str(stream_exc)
        err_tb = _tb.format_exc()
        _save_error_log(
            log_dir,
            query,
            steps,
            state,
            err_msg,
            err_tb,
            run_id,
            fp=fp,
            ts=ts_start,
            session_id=session_id,
        )
        raise
    finally:
        _TOKEN_EMITTER.reset(token_ctx)

    total_s = time.monotonic() - log_t0
    _save_log(
        log_dir,
        query,
        steps,
        state,
        total_s,
        run_id,
        max_log_context_chars,
        fp=fp,
        ts=ts_start,
        session_id=session_id,
    )
    final_state = _safe(_trunc_state(state, max_context_preview))
    emit(
        {
            "event": "pipeline_complete",
            "elapsed_ms": round(time.monotonic() * 1000 - t0),
            "state": final_state,
        }
    )
    return final_state


def _console_emit(event: Dict[str, Any]) -> None:
    """Human-readable console renderer for :func:`main`'s ``--query`` mode."""
    kind = event.get("event")
    if kind == "graph_ready":
        print(f"[{event['elapsed_ms']:>6} ms] Graph built. Starting pipeline...")
    elif kind == "step_start":
        print(f"\n--> {event['node']}")
    elif kind == "step_complete":
        update = event.get("update", {})
        detail_bits = []
        if "planned_agents" in update:
            detail_bits.append(f"planned_agents={update['planned_agents']}")
        if "context_gathered" in update:
            counts = {
                coll: info.get("chunk_count")
                for coll, info in update["context_gathered"].items()
                if isinstance(info, dict)
            }
            if counts:
                detail_bits.append(f"chunks={counts}")
        if "validation_score" in update:
            detail_bits.append(f"validation_score={update['validation_score']}")
        detail = (" | " + ", ".join(detail_bits)) if detail_bits else ""
        print(f"[{event['elapsed_ms']:>6} ms] {event['node']} done{detail}")
        for err in event.get("errors", []):
            print(f"    ! worker error: {err}")
    elif kind == "pipeline_complete":
        state = event.get("state", {})
        print("\n" + "=" * 72)
        print(f"Validation score: {state.get('validation_score')}")
        print(f"Retries: {state.get('retry_count')}")
        print("-" * 72)
        print(state.get("final_response", "(no response)"))
        print("=" * 72)
        print(f"\nTotal elapsed: {event['elapsed_ms']} ms")
    elif kind == "error":
        print(f"\nERROR: {event.get('message')}", file=sys.stderr)
        if event.get("traceback"):
            print(event["traceback"], file=sys.stderr)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the agentic RAG pipeline (src/agentic_rag/graph.py) on a single query."
    )
    parser.add_argument(
        "--query", type=str, required=True, help="Question to ask the pipeline."
    )
    args = parser.parse_args(argv)

    project_root = os.environ.get(
        "CPS_PROJECT_ROOT", str(Path.cwd())
    )

    try:
        run_streaming(args.query, project_root=project_root, emit=_console_emit)
    except Exception as exc:  # noqa: BLE001 - console mode: report and exit non-zero
        print(f"\nPipeline failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
