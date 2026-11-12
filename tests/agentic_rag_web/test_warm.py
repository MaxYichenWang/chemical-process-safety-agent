"""Tests for src/agentic_rag_web/warm.py."""

from __future__ import annotations

import threading

from src.agentic_rag import analytical_tools
from src.agentic_rag_web.warm import (
    get_cached_tools,
    patch_analytical_concurrency,
    warm_up,
)


def test_get_cached_tools_returns_identical_object_across_calls():
    first = get_cached_tools()
    second = get_cached_tools()
    assert first is second


def test_get_cached_tools_is_the_workers_tools_singleton():
    from src.agentic_rag.workers import TOOLS

    assert get_cached_tools() is TOOLS


def test_patch_analytical_concurrency_is_idempotent():
    patch_analytical_concurrency()
    patched_once = analytical_tools.get_db
    patch_analytical_concurrency()
    assert analytical_tools.get_db is patched_once


def test_patch_analytical_concurrency_gives_each_thread_its_own_cursor():
    patch_analytical_concurrency()

    n = 4
    seen = {}
    lock = threading.Lock()
    # Force all threads to hold their connection object simultaneously
    # before any of them can exit and be garbage-collected -- otherwise
    # id() reuse across non-overlapping thread lifetimes would produce a
    # false negative (two *different* objects can share an id() once the
    # first is freed).
    barrier = threading.Barrier(n)

    def worker(name):
        conn = analytical_tools.get_db()
        with lock:
            seen[name] = id(conn)
        barrier.wait(timeout=5)

    threads = [threading.Thread(target=worker, args=(f"t{i}",)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Each thread must get its own connection object (no sharing).
    assert len(set(seen.values())) == len(seen)


def test_patch_analytical_concurrency_survives_concurrent_queries():
    """Regression test for the spike finding: concurrent execute_analytical_query()
    calls against the shared connection used to corrupt results. After the
    patch, concurrent threads must see the same result as sequential execution."""
    patch_analytical_concurrency()
    from src.agentic_rag.analytical_tools import execute_analytical_query, get_db

    sql = "SELECT COUNT(*) AS n FROM csb_incident_meta"

    def run_once():
        conn = get_db()
        return execute_analytical_query(conn, sql)

    sequential = [run_once() for _ in range(3)]

    results = [None] * 6
    errors = []

    def worker(i):
        try:
            results[i] = run_once()
        except Exception as exc:  # pragma: no cover - failure path
            errors.append((i, repr(exc)))

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"concurrent execution raised: {errors}"
    baseline_row_count = sequential[0][1].get("row_count")
    for r in results:
        assert r[1].get("row_count") == baseline_row_count


def test_warm_up_returns_status_dict():
    status = warm_up()
    assert "tools_warmed" in status
    assert "analytical_concurrency_patched" in status
    assert status["analytical_concurrency_patched"] is True
    assert "data_availability" in status
    assert status["data_availability"]["status"] in {"full", "partial", "none"}
