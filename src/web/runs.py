"""List, read, and delete stored web and CLI session history.

Both ``logs/runs/`` and ``logs/cli/`` are scanned so earlier CLI sessions
remain available to the browser interface.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _iter_log_dirs(project_root: str) -> List[Path]:
    root = Path(project_root)
    return [root / "logs" / "runs", root / "logs" / "cli"]


def _load_run_file(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        # An incrementally-written record may be read mid-write, or a
        # killed process may leave a partial/corrupt file. Skip rather than
        # 500 the whole history endpoint over one bad file.
        logger.warning("Skipping unreadable run log %s: %s", path, exc)
        return None


def _effective_run_id(data: Dict[str, Any], path: Path) -> str:
    """Return stored ``run_id`` or fall back to filename stem for legacy CLI logs."""
    return str(data.get("run_id") or path.stem)


def _summarize(data: Dict[str, Any], path: Path) -> Dict[str, Any]:
    final_state = data.get("final_state", {})
    run_id = _effective_run_id(data, path)
    session_id = str(data.get("session_id") or run_id)
    history_len = len(final_state.get("conversation_history") or [])
    return {
        "run_id": run_id,
        "session_id": session_id,
        "timestamp": data.get("timestamp"),
        "query": data.get("query"),
        "validation_score": final_state.get("validation_score"),
        "retry_count": final_state.get("retry_count"),
        "total_elapsed_s": data.get("total_elapsed_s"),
        "final_response_preview": (final_state.get("final_response") or "")[:280],
        "status": data.get(
            "status", "complete" if final_state.get("final_response") else "error"
        ),
        "source_file": path.name,
        "turn_count": 1,
        "turn_run_ids": [run_id] if run_id else [],
        "_history_len": history_len,
    }


def _turn_sort_key(item: Dict[str, Any], session_id: str) -> tuple:
    is_root = 0 if item.get("run_id") == session_id else 1
    hist_len = int(item.get("_history_len") or 0)
    ts = str(item.get("timestamp") or "")
    src = str(item.get("source_file") or "")
    return (is_root, hist_len, ts, src)


def list_runs(project_root: str, *, limit: int = 50, offset: int = 0) -> Dict[str, Any]:
    """Newest-first, paginated summary list across both log directories.

    Multi-turn runs sharing a ``session_id`` are collapsed into a single
    session summary anchored by the session's initial query and the latest
    turn's timestamp/status.
    """
    files: List[Path] = []
    for d in _iter_log_dirs(project_root):
        if d.exists():
            files.extend(d.glob("*.json"))

    files.sort(key=lambda p: p.name, reverse=True)

    # Check if any files share a multi-turn session_id. To preserve exact
    # backward-compatibility with single-turn pagination and unreadable-file
    # accounting, group by session_id across loaded summaries.
    all_summaries: List[Dict[str, Any]] = []
    unreadable_count = 0
    for path in files:
        data = _load_run_file(path)
        if data is None:
            unreadable_count += 1
        else:
            all_summaries.append(_summarize(data, path))

    grouped_order: List[str] = []
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for idx, item in enumerate(all_summaries):
        sid = str(item.get("session_id") or item.get("run_id") or f"_anon_{idx}")
        if sid not in groups:
            grouped_order.append(sid)
            groups[sid] = []
        groups[sid].append(item)

    merged_sessions: List[Dict[str, Any]] = []
    for sid in grouped_order:
        turns = groups[sid]
        if len(turns) == 1:
            single = dict(turns[0])
            single.pop("_history_len", None)
            merged_sessions.append(single)
        else:
            turns_oldest_first = sorted(
                turns, key=lambda item: _turn_sort_key(item, sid)
            )
            first_turn = turns_oldest_first[0]
            latest_turn = dict(turns_oldest_first[-1])
            latest_turn.pop("_history_len", None)
            merged_sessions.append(
                {
                    **latest_turn,
                    "run_id": sid,
                    "session_id": sid,
                    "query": first_turn.get("query") or latest_turn.get("query"),
                    "turn_count": len(turns_oldest_first),
                    "turn_run_ids": [
                        t["run_id"] for t in turns_oldest_first if t.get("run_id")
                    ],
                }
            )

    total = len(merged_sessions) + unreadable_count
    runs = merged_sessions[offset : offset + limit]

    return {"runs": runs, "total": total, "limit": limit, "offset": offset}


def get_run(project_root: str, run_id: str) -> Optional[Dict[str, Any]]:
    """Full stored detail for one run, searched across both log directories."""
    for d in _iter_log_dirs(project_root):
        if not d.exists():
            continue
        for path in d.glob("*.json"):
            data = _load_run_file(path)
            if data is not None and _effective_run_id(data, path) == run_id:
                data.setdefault("run_id", run_id)
                return data
    return None


def get_session_runs(project_root: str, session_id: str) -> List[Dict[str, Any]]:
    """Return all stored runs belonging to *session_id* in chronological order."""
    matched: List[tuple[tuple, Dict[str, Any]]] = []
    for d in _iter_log_dirs(project_root):
        if not d.exists():
            continue
        for path in d.glob("*.json"):
            data = _load_run_file(path)
            if data is None:
                continue
            rid = _effective_run_id(data, path)
            data.setdefault("run_id", rid)
            sid = str(data.get("session_id") or rid)
            if sid == session_id or rid == session_id:
                is_root = 0 if rid == session_id else 1
                hist_len = len(
                    ((data.get("final_state") or {}).get("conversation_history")) or []
                )
                sort_key = (
                    is_root,
                    hist_len,
                    str(data.get("timestamp") or ""),
                    path.name,
                )
                matched.append((sort_key, data))
    matched.sort(key=lambda pair: pair[0])
    return [data for _, data in matched]


def delete_session_runs(project_root: str, session_id: str) -> int:
    """Delete all stored JSON run logs belonging to *session_id* (or *run_id*).

    Returns the number of deleted log files across ``logs/runs`` and ``logs/cli``.
    """
    candidates: List[tuple[Path, str, str]] = []
    target_sid = session_id

    for d in _iter_log_dirs(project_root):
        if not d.exists():
            continue
        for path in d.glob("*.json"):
            data = _load_run_file(path)
            if data is None:
                continue
            rid = _effective_run_id(data, path)
            sid = str(data.get("session_id") or rid)
            candidates.append((path, rid, sid))
            if rid == session_id and sid:
                target_sid = sid

    deleted = 0
    for path, rid, sid in candidates:
        if sid == target_sid or rid == session_id:
            try:
                path.unlink(missing_ok=True)
                deleted += 1
            except OSError as exc:
                logger.warning("Failed to delete run log %s: %s", path, exc)
    return deleted
