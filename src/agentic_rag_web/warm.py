"""Process-lifetime warm state for the web server.

Two costs the CLI pays per-query and the server must pay exactly once:

1. ``get_chroma_tools()`` (``src/agentic_rag/tools.py``) constructs a
   ``chromadb.PersistentClient`` and four ``VectorStoreIndex`` objects.
2. Nothing else needs warming -- ``build_web_graph()`` (``graph.py`` in this
   package) is compiled once by the FastAPI lifespan and held on
   ``app.state``.

This module also carries the one genuinely new piece of concurrency-safety
work: the shared DuckDB connection used by
``DataAnalystWorker`` is not safe for concurrent use from multiple threads.
See ``ARCHITECTURE.md`` for the full writeup.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Dict

from src.agentic_rag import analytical_tools
from src.agentic_rag import workers as _workers_module
from src.agentic_rag.tools import HybridChromaQueryEngine

logger = logging.getLogger(__name__)

_tools_lock = threading.Lock()
_cached_tools: Dict[str, HybridChromaQueryEngine] | None = None


def get_cached_tools() -> Dict[str, HybridChromaQueryEngine]:
    """Return the process-lifetime singleton backing ``get_chroma_tools()``.

    ``src.agentic_rag.workers`` already builds this once at import time
    (``TOOLS = get_chroma_tools()``, module scope) -- Python's module cache
    means every subsequent ``import src.agentic_rag.workers`` anywhere in
    this process returns the same module object and the same ``TOOLS``
    dict. This function exposes that cache to the rest of the web layer.
    Tests assert that repeated calls return the identical object and do not
    construct another ``PersistentClient``.
    """
    global _cached_tools
    with _tools_lock:
        if _cached_tools is None:
            _cached_tools = _workers_module.TOOLS
            logger.info(
                "Warmed %d Chroma tool(s) for process lifetime: %s",
                len(_cached_tools),
                sorted(_cached_tools.keys()),
            )
        return _cached_tools


_analytical_patch_lock = threading.Lock()
_analytical_patch_applied = False
_thread_local_db = threading.local()


def patch_analytical_concurrency() -> None:
    """Install one DuckDB cursor per thread to protect concurrent queries.

    The worker resolves ``get_db`` at call time and picks up the wrapper.
    Repeated calls are safe.
    """
    global _analytical_patch_applied
    with _analytical_patch_lock:
        if _analytical_patch_applied:
            return

        original_get_db = analytical_tools.get_db

        def _thread_safe_get_db() -> Any:
            if not hasattr(_thread_local_db, "conn"):
                _thread_local_db.conn = original_get_db().cursor()
            return _thread_local_db.conn

        analytical_tools.get_db = _thread_safe_get_db  # type: ignore[assignment]
        _analytical_patch_applied = True
        logger.info(
            "Patched analytical_tools.get_db for thread-safe concurrent access "
            "(per-thread DuckDB cursor)."
        )


def is_analytical_concurrency_patched() -> bool:
    return _analytical_patch_applied


def warm_up() -> Dict[str, Any]:
    """Pay every per-process startup cost once. Called from the FastAPI lifespan.

    Returns a small status dict suitable for surfacing on ``/api/health``.
    """
    patch_analytical_concurrency()
    tools = get_cached_tools()
    data_availability = analytical_tools.validate_data_availability()
    return {
        "tools_warmed": sorted(tools.keys()),
        "analytical_concurrency_patched": _analytical_patch_applied,
        "data_availability": data_availability,
    }
