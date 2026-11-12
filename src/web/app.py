"""FastAPI application.

Wires together the warm backend (``src.agentic_rag_web``), the ported
event contract (``src.web.events``), the background run executor
(``src.web.run_manager``), and run history (``src.web.runs``) behind a
small HTTP/SSE surface. Serves the static frontend (``src/web/static/``)
built in Phase 2's browser tasks.
"""

from __future__ import annotations

import json
import ipaddress
import logging
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.agentic_rag.analytical_tools import validate_data_availability
from src.agentic_rag.config import VALIDATION_SCORE_THRESHOLD
from src.agentic_rag_web.graph import build_web_graph
from src.agentic_rag_web.warm import warm_up
from src.web import runs as runs_module
from src.web.config import (
    WEB_ACCESS_TOKEN,
    WEB_DAILY_SPEND_CEILING_USD,
    WEB_HOST,
    WEB_MAX_CONCURRENCY,
    WEB_REQUIRE_TOKEN,
)
from src.web.events import (
    AGENT_KEY_TO_NODE,
    NODE_DESCRIPTIONS,
    NODE_LABELS,
    WORKER_NODES,
)
from src.web.run_manager import RunManager, render_markdown_safely
from src.web.spend import check_spend_ceiling

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(os.getenv("CPS_PROJECT_ROOT", str(Path.cwd()))).resolve()
STATIC_DIR = Path(__file__).resolve().parent / "static"
LOG_DIR = PROJECT_ROOT / "logs" / "runs"

ACCESS_TOKEN_COOKIE = "cps_web_token"

# Hosts that mean "this machine only" -- binding to any of these is exactly
# the pre-Phase-5 posture where a shared token would just be friction.
_LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}


def _token_required(request: Optional[Request] = None) -> bool:
    """Gate configured LAN access and requests arriving on a non-loopback socket."""
    if WEB_REQUIRE_TOKEN or WEB_HOST not in _LOCAL_HOSTS:
        return True
    if request is not None and request.scope.get("server"):
        try:
            address = ipaddress.ip_address(request.scope["server"][0])
        except ValueError:
            # ASGI test transports can use a symbolic server name.
            return False
        if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped:
            address = address.ipv4_mapped
        return not address.is_loopback
    return False


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Pay every startup cost once and hold the compiled graph + run manager."""
    if _token_required() and not WEB_ACCESS_TOKEN:
        raise RuntimeError(
            "Set CPS_WEB_ACCESS_TOKEN before enabling token-protected access"
        )
    warm_status = warm_up()
    graph = build_web_graph()
    app.state.warm_status = warm_status
    app.state.graph = graph
    app.state.run_manager = RunManager(
        project_root=str(PROJECT_ROOT),
        build_graph_fn=lambda: graph,
        log_dir=LOG_DIR,
        max_concurrency=WEB_MAX_CONCURRENCY,
    )
    logger.info("Web server warmed up: %s", warm_status)
    if _token_required():
        logger.info(
            "Access token required. Use the token configured in CPS_WEB_ACCESS_TOKEN."
        )
    yield


app = FastAPI(title="Chemical Process Safety LLM — Web UI", lifespan=lifespan)


@app.middleware("http")
async def disable_static_cache(request: Request, call_next):
    """Ensure browser always revalidates `/` and `/static/*` assets."""
    response = await call_next(request)
    path = request.url.path
    if path == "/" or path.startswith("/static/"):
        response.headers["Cache-Control"] = "no-cache, must-revalidate"
    return response


@app.middleware("http")
async def enforce_access_token(request: Request, call_next):
    """Check the shared token through a cookie, bearer header, or initial URL."""
    if not _token_required(request):
        return await call_next(request)

    if not WEB_ACCESS_TOKEN:
        return JSONResponse(
            status_code=503, content={"detail": "Access token is not configured"}
        )

    if request.cookies.get(ACCESS_TOKEN_COOKIE) == WEB_ACCESS_TOKEN:
        return await call_next(request)

    auth_header = request.headers.get("authorization", "")
    if (
        auth_header.startswith("Bearer ")
        and auth_header[len("Bearer ") :] == WEB_ACCESS_TOKEN
    ):
        return await call_next(request)

    query_token = request.query_params.get("token")
    if query_token == WEB_ACCESS_TOKEN:
        response = await call_next(request)
        response.set_cookie(
            ACCESS_TOKEN_COOKIE, WEB_ACCESS_TOKEN, httponly=True, samesite="lax"
        )
        return response

    return JSONResponse(
        status_code=401,
        content={
            "detail": "Missing or invalid access token. Use the join URL shared by the operator."
        },
    )


class QueryRequest(BaseModel):
    query: str
    history: Optional[List[Dict[str, str]]] = None
    session_id: Optional[str] = None


def _run_manager(request: Request) -> RunManager:
    return request.app.state.run_manager


@app.get("/api/health")
async def health(request: Request) -> Dict[str, Any]:
    """warm status, data availability, in-flight count, spend-vs-ceiling."""
    manager: RunManager = request.app.state.run_manager
    ok, spent_today = check_spend_ceiling(LOG_DIR, WEB_DAILY_SPEND_CEILING_USD)
    return {
        "status": "ok",
        "warm": request.app.state.warm_status,
        "data_availability": validate_data_availability(),
        "in_flight": manager.in_flight_count,
        "max_concurrency": WEB_MAX_CONCURRENCY,
        "daily_spend_usd": round(spent_today, 4),
        "daily_spend_ceiling_usd": WEB_DAILY_SPEND_CEILING_USD,
        "spend_ok": ok,
    }


@app.get("/api/meta")
async def meta() -> Dict[str, Any]:
    """Node labels/descriptions/agent-key mapping the browser needs (avoids a 2nd copy)."""
    return {
        "node_labels": NODE_LABELS,
        "node_descriptions": NODE_DESCRIPTIONS,
        "agent_key_to_node": AGENT_KEY_TO_NODE,
        "worker_nodes": WORKER_NODES,
        "validation_score_threshold": VALIDATION_SCORE_THRESHOLD,
    }


@app.post("/api/query")
async def post_query(body: QueryRequest, request: Request) -> Dict[str, Any]:
    """register a run and return immediately; the pipeline runs in the background."""
    query = body.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="query must not be empty")

    ok, spent_today = check_spend_ceiling(LOG_DIR, WEB_DAILY_SPEND_CEILING_USD)
    if not ok:
        raise HTTPException(
            status_code=429,
            detail=(
                f"Daily spend ceiling reached (${spent_today:.2f} of "
                f"${WEB_DAILY_SPEND_CEILING_USD:.2f}). Try again tomorrow."
            ),
        )

    manager: RunManager = request.app.state.run_manager
    run_id = await manager.start_run(
        query, history=body.history, session_id=body.session_id
    )
    run = manager.get(run_id)
    return {
        "run_id": run_id,
        "session_id": run.session_id if run is not None else run_id,
    }


def _sse_format(record: Dict[str, Any]) -> str:
    return f"id: {record['id']}\ndata: {json.dumps(record)}\n\n"


@app.get("/api/stream/{run_id}")
async def stream_run(run_id: str, request: Request) -> StreamingResponse:
    """SSE stream, replaying missed events after ``Last-Event-ID`` (reconnect)."""
    manager: RunManager = request.app.state.run_manager
    run = manager.get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="unknown run_id")

    last_id = 0
    header_val = request.headers.get("last-event-id")
    if header_val is not None:
        try:
            last_id = int(header_val)
        except ValueError:
            last_id = 0

    async def event_generator() -> AsyncIterator[str]:
        async def _disconnected() -> bool:
            return await request.is_disconnected()

        async for record in run.stream_from(last_id, is_disconnected=_disconnected):
            if record.get("event") == "_keep_alive":
                yield ": keep-alive\n\n"
                continue
            yield _sse_format(record)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@app.get("/api/runs/active")
async def active_runs(request: Request) -> Dict[str, Any]:
    """lets a reloaded page discover in-flight runs to re-attach to."""
    manager: RunManager = request.app.state.run_manager
    return {"run_ids": manager.active_run_ids()}


@app.get("/api/runs")
async def list_runs(
    request: Request, limit: int = 50, offset: int = 0
) -> Dict[str, Any]:
    """paginated run history sidebar, merging in-flight runs."""
    # LOG_DIR is `<project_root>/logs/runs`; derive project_root from it
    # (rather than the module-level PROJECT_ROOT constant) so tests that
    # redirect LOG_DIR to a tmp_path see their own run history back out.
    history = runs_module.list_runs(
        str(LOG_DIR.parent.parent), limit=limit, offset=offset
    )
    manager: RunManager = getattr(request.app.state, "run_manager", None)
    if offset == 0 and manager is not None:
        existing_runs = history.get("runs", [])
        disk_run_ids: set[str] = set()
        session_by_id: Dict[str, Dict[str, Any]] = {}
        for r in existing_runs:
            if r.get("run_id"):
                disk_run_ids.add(r["run_id"])
            for trid in r.get("turn_run_ids") or []:
                disk_run_ids.add(trid)
            sid = r.get("session_id") or r.get("run_id")
            if sid:
                session_by_id[sid] = r

        active_summaries = []
        for run_id in reversed(manager.active_run_ids()):
            if run_id not in disk_run_ids:
                run = manager.get(run_id)
                if run is not None:
                    sid = run.session_id or run.run_id
                    if sid in session_by_id:
                        existing = session_by_id[sid]
                        existing["status"] = "in_progress"
                        existing["turn_count"] = (existing.get("turn_count") or 1) + 1
                        existing.setdefault("turn_run_ids", []).append(run.run_id)
                    else:
                        active_summaries.append(
                            {
                                "run_id": sid,
                                "session_id": sid,
                                "timestamp": datetime.fromtimestamp(
                                    run.created_at, timezone.utc
                                ).isoformat(),
                                "query": run.query,
                                "validation_score": None,
                                "retry_count": 0,
                                "total_elapsed_s": round(
                                    time.time() - run.created_at, 1
                                ),
                                "final_response_preview": "",
                                "status": (
                                    "error"
                                    if run.error
                                    else ("complete" if run.done else "in_progress")
                                ),
                                "source_file": None,
                                "turn_count": 1,
                                "turn_run_ids": [run.run_id],
                            }
                        )
        if active_summaries:
            history["runs"] = active_summaries + history["runs"]
            history["total"] = history.get("total", 0) + len(active_summaries)
    return history


def _augment_stored_run_html(stored: Dict[str, Any]) -> Dict[str, Any]:
    final_state = stored.get("final_state")
    if isinstance(final_state, dict) and final_state.get("final_response"):
        final_state.setdefault(
            "final_response_html", render_markdown_safely(final_state["final_response"])
        )
    return stored


def _collect_session_turns(
    project_root: str, manager: RunManager, session_id: str
) -> List[Dict[str, Any]]:
    """Return all turns belonging to *session_id* (disk + in-memory) in chronological order."""
    stored_turns = runs_module.get_session_runs(project_root, session_id)
    turns_by_id: Dict[str, Dict[str, Any]] = {}
    ordered_ids: List[str] = []

    for st in stored_turns:
        rid = st.get("run_id")
        if not rid:
            continue
        _augment_stored_run_html(st)
        turns_by_id[rid] = {
            "run_id": rid,
            "session_id": st.get("session_id") or session_id,
            "query": st.get("query"),
            "done": st.get("status") in ("complete", "error"),
            "error": st.get("error"),
            "final_state": st.get("final_state"),
            "steps": st.get("steps"),
        }
        ordered_ids.append(rid)

    for live in manager.get_session_runs(session_id):
        if live.run_id not in turns_by_id:
            turns_by_id[live.run_id] = {
                "run_id": live.run_id,
                "session_id": live.session_id,
                "query": live.query,
                "done": live.done,
                "error": live.error,
                "events": live.events,
            }
            ordered_ids.append(live.run_id)
        elif not live.done:
            turns_by_id[live.run_id]["done"] = False
            turns_by_id[live.run_id]["events"] = live.events

    return [turns_by_id[rid] for rid in ordered_ids]


@app.get("/api/runs/{run_id}")
async def get_run(run_id: str, request: Request) -> Dict[str, Any]:
    """Full detail for one run: in-flight (from RunManager) or historical (from disk)."""
    manager: RunManager = request.app.state.run_manager
    project_root = str(LOG_DIR.parent.parent)
    live_run = manager.get(run_id)
    if live_run is not None:
        sid = live_run.session_id or live_run.run_id
        turns = _collect_session_turns(project_root, manager, sid)
        return {
            "run_id": live_run.run_id,
            "session_id": sid,
            "query": live_run.query,
            "done": live_run.done,
            "error": live_run.error,
            "events": live_run.events,
            "turns": turns,
        }

    stored = runs_module.get_run(project_root, run_id)
    if stored is None:
        raise HTTPException(status_code=404, detail="unknown run_id")

    _augment_stored_run_html(stored)
    sid = stored.get("session_id") or stored.get("run_id") or run_id
    stored["session_id"] = sid
    stored["turns"] = _collect_session_turns(project_root, manager, sid)
    return stored


@app.delete("/api/runs/{run_id}")
async def delete_run(run_id: str, request: Request) -> Dict[str, Any]:
    """Delete a completed session (all its turns on disk and in memory)."""
    manager: RunManager = request.app.state.run_manager
    if manager.has_active_session_run(run_id):
        raise HTTPException(
            status_code=409,
            detail="cannot delete a session while a query is still running",
        )
    project_root = str(LOG_DIR.parent.parent)
    deleted_disk = runs_module.delete_session_runs(project_root, run_id)
    deleted_mem = manager.remove_session(run_id)
    if deleted_disk == 0 and deleted_mem == 0:
        raise HTTPException(status_code=404, detail="unknown run_id")
    return {
        "deleted": True,
        "session_id": run_id,
        "deleted_runs": deleted_disk or deleted_mem,
    }


if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def index() -> FileResponse:
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=503, detail="frontend not built yet")
    return FileResponse(str(index_path))
