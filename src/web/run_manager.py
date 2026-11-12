"""Per-run SSE state and background execution.

``RunManager`` owns every in-flight and completed run for the process
lifetime of the server: it assigns sequential per-run event IDs,
retains the full event list so a reconnecting client can replay everything
after ``Last-Event-ID``, and runs the pipeline via
``anyio.to_thread.run_sync`` behind a concurrency semaphore so
``POST /api/query`` returns immediately while the actual pipeline
executes detached from the request.
"""

from __future__ import annotations

import asyncio
import copy
import html
import logging
import re
import time
import uuid
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Dict, List, Optional

import anyio
import markdown as markdown_module

logger = logging.getLogger(__name__)

# Matches the design doc's daily-spend accounting: reuse the per-call cost
# data _save_log() already records rather than inventing a second source.
_SCRIPT_TAG_RE = re.compile(r"<\s*script", re.I)
_BARE_BLOCK_MARKER_RE = re.compile(r"^\s*(?:#{1,6}|[-*+]|\d+\.)\s*$")
_TABLE_SEP_PARTIAL_RE = re.compile(r"^\s*\|[\s\-:|]*$")
_FENCE_RE = re.compile(r"^```", re.M)


def _count_table_cols(header_line: str) -> int:
    s = header_line.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|"):
        s = s[:-1]
    cols = len(s.split("|")) if s else 1
    return max(1, cols)


def stabilize_streaming_markdown(raw_markdown: str) -> str:
    """Normalize incomplete trailing Markdown constructs during live token streaming.

    Prevents transient raw Markdown syntax (such as partial table headers
    ``| Col1 | Col2 |\\n|---|--``, bare ``## `` markers, unclosed ``**bold``,
    or unclosed ```` ``` ```` code fences) from flashing as plain paragraph text
    before enough tokens arrive for Python ``markdown`` to parse the block.
    """
    if not raw_markdown:
        return ""

    md = raw_markdown
    if len(_FENCE_RE.findall(md)) % 2 == 1:
        return md + "\n```"

    lines = md.split("\n")
    while lines and (
        not lines[-1].strip()
        or _BARE_BLOCK_MARKER_RE.match(lines[-1])
        or lines[-1].strip() == "|"
    ):
        lines.pop()

    if not lines:
        return ""

    if (
        len(lines) >= 2
        and _TABLE_SEP_PARTIAL_RE.match(lines[-1])
        and lines[-2].lstrip().startswith("|")
    ):
        prev_is_table = len(lines) >= 3 and lines[-3].lstrip().startswith("|")
        if not prev_is_table:
            ncols = _count_table_cols(lines[-2])
            lines[-1] = "|" + "|".join(["---"] * ncols) + "|"
            if (
                len(lines) >= 3
                and lines[-3].strip()
                and not lines[-3].lstrip().startswith("#")
            ):
                lines.insert(-2, "")
    elif lines[-1].lstrip().startswith("|"):
        prev_is_table = len(lines) >= 2 and lines[-2].lstrip().startswith("|")
        if not prev_is_table:
            header = lines[-1]
            if not header.rstrip().endswith("|"):
                header = header + " |"
            ncols = _count_table_cols(header)
            lines[-1] = header
            if (
                len(lines) >= 2
                and lines[-2].strip()
                and not lines[-2].lstrip().startswith("#")
            ):
                lines.insert(-1, "")
            lines.append("|" + "|".join(["---"] * ncols) + "|")

    md = "\n".join(lines)

    last_nl = md.rfind("\n")
    last_line = md[last_nl + 1 :]
    if last_line.count("**") % 2 == 1:
        open_idx = last_line.rfind("**")
        after = last_line[open_idx + 2 :]
        if not after.strip():
            cut_pos = (last_nl + 1) + open_idx
            md = md[:cut_pos]
        else:
            trailing_ws_len = len(md) - len(md.rstrip(" \t"))
            if trailing_ws_len:
                md = md[:-trailing_ws_len] + "**" + md[-trailing_ws_len:]
            else:
                md = md + "**"

    return md


def render_markdown_safely(raw_markdown: str) -> str:
    """Render markdown to HTML, escaping any raw HTML in the model output.

    The content is LLM-generated text, not author-controlled (design doc
    §5) -- ``pipeline_complete`` must never pass model output through as
    live HTML. ``markdown.markdown(..., safe_mode=...)`` was removed
    upstream, so raw model text is HTML-escaped *before* conversion; the
    result contains no live tags except the ones the markdown renderer
    itself produces from the escaped text.
    """
    escaped = html.escape(raw_markdown)
    return markdown_module.markdown(
        escaped, extensions=["tables", "fenced_code", "nl2br"]
    )


class Run:
    """One query's full event history plus its live SSE fan-out queue."""

    __slots__ = (
        "run_id",
        "session_id",
        "query",
        "history",
        "created_at",
        "events",
        "queue",
        "done",
        "error",
        "_next_seq",
        "_streaming_md",
    )

    def __init__(
        self,
        run_id: str,
        query: str,
        history: Optional[List[Dict[str, str]]] = None,
        session_id: Optional[str] = None,
    ):
        self.run_id = run_id
        self.session_id = (session_id or "").strip() or run_id
        self.query = query
        self.history = history
        self.created_at = time.time()
        self.events: List[Dict[str, Any]] = []
        self.queue: "asyncio.Queue[Dict[str, Any]]" = asyncio.Queue()
        self.done = False
        self.error: Optional[str] = None
        self._next_seq = 0
        self._streaming_md = ""

    def next_seq(self) -> int:
        self._next_seq += 1
        return self._next_seq

    async def stream_from(
        self, last_id: int, *, is_disconnected: Optional[Callable[[], Any]] = None
    ) -> AsyncIterator[Dict[str, Any]]:
        """Replay events after ``last_id``, then tail live ones. Never double-delivers.

        Two sources feed this: the retained ``events`` list (for replay) and
        the live ``queue`` (for tailing). Both are populated by the same
        ``_record_and_publish()`` call, so an event can be sitting in
        *both* at once -- if a client connects after the run has already
        produced some events but before anything has drained the queue,
        the naive approach (replay the list, then drain whatever the
        queue yields) redelivers those same events a second time. Tracking
        a running cursor and filtering the queue against it as well is
        what prevents that.
        """
        cursor = max(0, last_id)
        while True:
            # A producer can append the terminal record while a replay yield is
            # suspended. Re-read the retained log before testing completion.
            # IDs start at one and the retained list is never truncated.
            for record in self.events[cursor:]:
                if record["id"] > cursor:
                    cursor = record["id"]
                    yield record
                    if record.get("event") in ("pipeline_complete", "error"):
                        return
            if self.done:
                if self.events and self.events[-1]["id"] > cursor:
                    continue
                return
            if is_disconnected is not None and await is_disconnected():
                return
            try:
                record = await asyncio.wait_for(self.queue.get(), timeout=15.0)
            except asyncio.TimeoutError:
                yield {"event": "_keep_alive"}
                continue
            if record["id"] <= cursor:
                # Already delivered via the events-list replay above.
                continue
            yield record
            cursor = record["id"]
            if record.get("event") in ("pipeline_complete", "error"):
                return


class RunManager:
    def __init__(
        self,
        *,
        project_root: str,
        build_graph_fn: Callable[[], Any],
        log_dir: Path,
        max_concurrency: int = 3,
    ):
        self.project_root = project_root
        self.build_graph_fn = build_graph_fn
        self.log_dir = log_dir
        self._runs: Dict[str, Run] = {}
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self.in_flight_count = 0

    def get(self, run_id: str) -> Optional[Run]:
        return self._runs.get(run_id)

    def get_session_runs(self, session_id: str) -> List[Run]:
        """Return all in-memory runs belonging to *session_id* in chronological order."""
        runs = [
            r
            for r in self._runs.values()
            if r.session_id == session_id or r.run_id == session_id
        ]
        runs.sort(key=lambda r: r.created_at)
        return runs

    def has_active_session_run(self, session_id: str) -> bool:
        """Return True if any run in *session_id* is still executing."""
        target_sid = session_id
        if session_id in self._runs:
            target_sid = self._runs[session_id].session_id or session_id
        return any(
            not r.done
            and (
                r.session_id == target_sid
                or r.run_id == session_id
                or r.session_id == session_id
            )
            for r in self._runs.values()
        )

    def remove_session(self, session_id: str) -> int:
        """Remove completed in-memory runs belonging to *session_id*."""
        target_sid = session_id
        if session_id in self._runs:
            target_sid = self._runs[session_id].session_id or session_id
        to_delete = [
            rid
            for rid, r in self._runs.items()
            if r.done
            and (
                r.session_id == target_sid
                or r.run_id == session_id
                or r.session_id == session_id
            )
        ]
        for rid in to_delete:
            self._runs.pop(rid, None)
        return len(to_delete)

    def active_run_ids(self) -> List[str]:
        return [rid for rid, r in self._runs.items() if not r.done]

    async def start_run(
        self,
        query: str,
        *,
        history: Optional[List[Dict[str, str]]] = None,
        session_id: Optional[str] = None,
    ) -> str:
        """Register a run and kick off its execution. Returns immediately."""
        run_id = str(uuid.uuid4())
        run = Run(run_id, query, history=history, session_id=session_id)
        self._runs[run_id] = run
        self._loop = asyncio.get_running_loop()
        asyncio.create_task(self._execute(run))
        return run_id

    def _record_and_publish(self, run: Run, event: Dict[str, Any]) -> None:
        """Append `event` to the run's replay log and push it to its live queue.

        Must run on the event loop thread -- called via
        ``call_soon_threadsafe`` from the worker thread that runs the
        pipeline.
        """
        record = {"id": run.next_seq(), **event}
        run.events.append(record)
        run.queue.put_nowait(record)
        if event.get("event") in ("pipeline_complete", "error"):
            run.done = True

    def _emit(self, run: Run, event: Dict[str, Any]) -> None:
        """The `emit` callback passed into `run_streaming`; called from a worker thread."""
        kind = event.get("event")
        if kind == "synthesis_token":
            token = event.get("token") or ""
            run._streaming_md += token
            stabilized = stabilize_streaming_markdown(run._streaming_md)
            event = {
                **event,
                "html": render_markdown_safely(stabilized) if stabilized else "",
            }
        elif kind == "step_complete" and event.get("node") in (
            "SynthesizeNode",
            "RevisionNode",
        ):
            upd = event.get("update") or {}
            draft = upd.get("draft_response")
            if draft:
                event = copy.deepcopy(event)
                event["update"]["draft_response_html"] = render_markdown_safely(draft)
            # Reset accumulator AFTER step_complete (not on step_start, which in
            # LangGraph `stream_mode="updates"` fires after all tokens streamed).
            run._streaming_md = ""
        elif kind == "pipeline_complete":
            event = _augment_pipeline_complete(event)
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._record_and_publish, run, event)
        else:  # pragma: no cover - defensive; loop always set by start_run
            self._record_and_publish(run, event)

    async def _execute(self, run: Run) -> None:
        async with self._semaphore:
            self.in_flight_count += 1
            try:
                from scripts.demo_agentic_rag import run_streaming

                def emit(event: Dict[str, Any]) -> None:
                    self._emit(run, event)

                def blocking() -> None:
                    kwargs: Dict[str, Any] = {
                        "project_root": self.project_root,
                        "emit": emit,
                        "log_dir": self.log_dir,
                        "build_graph_fn": self.build_graph_fn,
                        "run_id": run.run_id,
                    }
                    if run.history is not None:
                        kwargs["history"] = run.history
                    if run.session_id and run.session_id != run.run_id:
                        kwargs["session_id"] = run.session_id
                    run_streaming(run.query, **kwargs)

                await anyio.to_thread.run_sync(blocking)
            except Exception as exc:  # pragma: no cover - defensive fallback
                logger.exception("Run %s failed", run.run_id)
                run.error = str(exc)
                if self._loop is not None:
                    self._loop.call_soon_threadsafe(
                        self._record_and_publish,
                        run,
                        {"event": "error", "message": str(exc)},
                    )
            finally:
                self.in_flight_count -= 1
                # Publish thread callbacks already queued by _emit before closing replay.
                asyncio.get_running_loop().call_soon(setattr, run, "done", True)


def _augment_pipeline_complete(event: Dict[str, Any]) -> Dict[str, Any]:
    """Add rendered HTML alongside the raw markdown final response."""
    event = copy.deepcopy(event)
    state = event.get("state") or {}
    final_response = state.get("final_response", "")
    if final_response:
        state["final_response_html"] = render_markdown_safely(final_response)
    event["state"] = state
    return event
