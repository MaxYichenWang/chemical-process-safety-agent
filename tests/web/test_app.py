"""End-to-end tests for src/web/app.py.

Drives the real FastAPI app, the real compiled web graph, and the real
`run_streaming()` bridge -- only the two OpenAI network boundaries
(embeddings, chat completions) are faked, via the same
`patched_llm_boundary` harness used for the G2 equivalence gate. Log
output is redirected to a tmp_path so this test never touches the real
`logs/` directory.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

import src.web.app as app_module
from tests.agentic_rag_web.conftest import patched_llm_boundary  # noqa: F401 (used as a fixture)


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(app_module, "LOG_DIR", tmp_path / "logs" / "runs")
    monkeypatch.setattr(app_module, "WEB_ACCESS_TOKEN", "test-access-token")
    with TestClient(app_module.app) as c:
        yield c


def _parse_sse(text: str):
    events = []
    for block in text.strip().split("\n\n"):
        if not block.strip():
            continue
        data_line = [ln for ln in block.splitlines() if ln.startswith("data: ")]
        if data_line:
            events.append(json.loads(data_line[0][len("data: ") :]))
    return events


def test_health_endpoint_reports_warm_status(client):
    resp = client.get("/api/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "tools_warmed" in body["warm"]
    assert body["warm"]["analytical_concurrency_patched"] is True
    assert body["max_concurrency"] >= 1
    assert body["spend_ok"] is True


def test_meta_endpoint_exposes_node_tables(client):
    resp = client.get("/api/meta")
    assert resp.status_code == 200
    body = resp.json()
    assert body["node_labels"]["PlanNode"] == "Supervisor (Plan)"
    assert "DataAnalystWorker" in body["worker_nodes"]
    assert 0 < body["validation_score_threshold"] <= 1


def test_query_rejects_empty_string(client):
    resp = client.post("/api/query", json={"query": "   "})
    assert resp.status_code == 400


def test_unknown_run_id_stream_returns_404(client):
    resp = client.get("/api/stream/does-not-exist")
    assert resp.status_code == 404


def test_full_query_and_stream_roundtrip(client, patched_llm_boundary):
    resp = client.post("/api/query", json={"query": "What are the OSHA regulations for process safety management?"})
    assert resp.status_code == 200
    run_id = resp.json()["run_id"]
    assert run_id

    stream_resp = client.get(f"/api/stream/{run_id}")
    assert stream_resp.status_code == 200
    events = _parse_sse(stream_resp.text)

    assert events, "expected at least one SSE event"
    kinds = [e["event"] for e in events]
    assert "graph_ready" in kinds
    assert "pipeline_complete" in kinds
    ids = [e["id"] for e in events]
    assert ids == sorted(ids)

    final_event = events[-1]
    assert final_event["event"] == "pipeline_complete"
    assert "final_response_html" in final_event["state"]

    # Reconnect semantics: replay only events after Last-Event-ID.
    resume_resp = client.get(
        f"/api/stream/{run_id}", headers={"Last-Event-ID": str(ids[0])}
    )
    resumed_events = _parse_sse(resume_resp.text)
    assert [e["id"] for e in resumed_events] == ids[1:]

    # Run detail endpoint should reflect the completed run.
    detail_resp = client.get(f"/api/runs/{run_id}")
    assert detail_resp.status_code == 200
    detail = detail_resp.json()
    assert detail["done"] is True
    assert detail["query"].startswith("What are the OSHA")

    # active runs should no longer include a completed run.
    active_resp = client.get("/api/runs/active")
    assert run_id not in active_resp.json()["run_ids"]


def test_run_history_lists_completed_runs(client, patched_llm_boundary):
    resp = client.post("/api/query", json={"query": "What violations were cited following the incident?"})
    run_id = resp.json()["run_id"]
    # Drain the stream synchronously to ensure the run (and its log file) is complete.
    client.get(f"/api/stream/{run_id}")

    history_resp = client.get("/api/runs")
    assert history_resp.status_code == 200
    history = history_resp.json()
    run_ids_in_history = {r["run_id"] for r in history["runs"]}
    assert run_id in run_ids_in_history


def test_run_history_includes_in_flight_runs(client, patched_llm_boundary):
    resp = client.post("/api/query", json={"query": "In-flight query test"})
    run_id = resp.json()["run_id"]

    history_resp = client.get("/api/runs")
    assert history_resp.status_code == 200
    history = history_resp.json()
    matching = [r for r in history["runs"] if r["run_id"] == run_id]
    assert matching, "expected in-flight run to be included in /api/runs"
    assert matching[0]["query"] == "In-flight query test"
    assert matching[0]["status"] in ("in_progress", "complete")

    # Clean up by draining the stream
    client.get(f"/api/stream/{run_id}")


def test_get_run_augments_historical_final_state_with_html(client, patched_llm_boundary):
    """the history view must never need a second markdown renderer.

    The on-disk log (written by ``_save_log()``) predates the in-memory
    ``final_response_html`` augmentation that only happens for live SSE
    events, so ``GET /api/runs/{run_id}`` must add it back in for
    already-completed runs looked up from disk.
    """
    resp = client.post("/api/query", json={"query": "What are the recommendations from CSB reports?"})
    run_id = resp.json()["run_id"]
    client.get(f"/api/stream/{run_id}")  # drain to completion

    # Force the lookup through the on-disk path, not the in-memory RunManager.
    app_module.app.state.run_manager._runs.pop(run_id, None)

    detail_resp = client.get(f"/api/runs/{run_id}")
    assert detail_resp.status_code == 200
    detail = detail_resp.json()
    assert "final_response_html" in detail["final_state"]
    assert detail["final_state"]["final_response_html"]


def test_index_page_serves_frontend(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "app.js" in resp.text


def test_static_assets_are_served(client):
    js_resp = client.get("/static/app.js")
    assert js_resp.status_code == 200
    assert "EventSource" in js_resp.text

    css_resp = client.get("/static/styles.css")
    assert css_resp.status_code == 200
    assert "--color-worker" in css_resp.text


# --- shared access token, required once LAN-reachable ---


def test_token_not_required_on_default_localhost_config(client):
    """Baseline (127.0.0.1, WEB_REQUIRE_TOKEN=False): no token needed, as today."""
    resp = client.get("/api/health")
    assert resp.status_code == 200


def test_token_required_once_bound_to_lan_host(client, monkeypatch):
    monkeypatch.setattr(app_module, "WEB_HOST", "0.0.0.0")
    resp = client.get("/api/health")
    assert resp.status_code == 401
    assert "token" in resp.json()["detail"].lower()


def test_token_required_when_explicitly_forced_on_localhost(client, monkeypatch):
    monkeypatch.setattr(app_module, "WEB_REQUIRE_TOKEN", True)
    resp = client.get("/api/health")
    assert resp.status_code == 401


def test_query_param_token_grants_access_and_sets_cookie(client, monkeypatch):
    monkeypatch.setattr(app_module, "WEB_HOST", "0.0.0.0")
    monkeypatch.setattr(app_module, "WEB_ACCESS_TOKEN", "secret-token-123")

    resp = client.get("/api/health?token=secret-token-123")
    assert resp.status_code == 200
    assert resp.cookies.get("cps_web_token") == "secret-token-123"

    # Subsequent request relies on the cookie alone, no query param needed.
    resp2 = client.get("/api/health")
    assert resp2.status_code == 200


def test_wrong_query_param_token_is_rejected(client, monkeypatch):
    monkeypatch.setattr(app_module, "WEB_HOST", "0.0.0.0")
    monkeypatch.setattr(app_module, "WEB_ACCESS_TOKEN", "secret-token-123")
    resp = client.get("/api/health?token=wrong")
    assert resp.status_code == 401


def test_bearer_header_token_grants_access(client, monkeypatch):
    monkeypatch.setattr(app_module, "WEB_HOST", "0.0.0.0")
    monkeypatch.setattr(app_module, "WEB_ACCESS_TOKEN", "secret-token-123")
    resp = client.get("/api/health", headers={"Authorization": "Bearer secret-token-123"})
    assert resp.status_code == 200


def test_token_mode_requires_a_configured_token_before_warmup(monkeypatch):
    monkeypatch.setattr(app_module, "WEB_HOST", "0.0.0.0")
    monkeypatch.setattr(app_module, "WEB_ACCESS_TOKEN", "")
    with pytest.raises(RuntimeError, match="CPS_WEB_ACCESS_TOKEN"):
        with TestClient(app_module.app):
            pass


def test_actual_non_loopback_socket_requires_token(client, monkeypatch):
    monkeypatch.setattr(app_module, "WEB_HOST", "127.0.0.1")
    with TestClient(app_module.app, base_url="http://192.0.2.5:8000") as simulated_lan:
        assert simulated_lan.get("/api/health").status_code == 401
        assert simulated_lan.get("/api/health", headers={
            "Authorization": "Bearer test-access-token"
        }).status_code == 200


def test_token_is_absent_from_startup_output(tmp_path, monkeypatch, caplog, capsys):
    monkeypatch.setattr(app_module, "LOG_DIR", tmp_path / "logs" / "runs")
    monkeypatch.setattr(app_module, "WEB_REQUIRE_TOKEN", True)
    monkeypatch.setattr(app_module, "WEB_ACCESS_TOKEN", "private-test-access-token")
    with TestClient(app_module.app) as client:
        assert client.get("/api/health").status_code == 401
    assert "private-test-access-token" not in caplog.text
    assert "private-test-access-token" not in capsys.readouterr().out


def test_server_launcher_uses_app_configuration_without_access_logs(monkeypatch):
    from src.web import __main__ as server
    from unittest.mock import Mock

    run = Mock()
    monkeypatch.setattr(server.uvicorn, "run", run)
    monkeypatch.setattr(server, "WEB_HOST", "127.0.0.1")
    monkeypatch.setattr(server, "WEB_PORT", 8012)
    server.main()
    run.assert_called_once_with("src.web.app:app", host="127.0.0.1", port=8012, access_log=False)


def test_two_concurrent_lan_clients_do_not_interleave_or_leak_state(
    client, patched_llm_boundary, monkeypatch
):
    """ (single-process approximation): two token-gated 'clients' -- one
    with a valid cookie, one anonymous -- hit the server concurrently and
    each gets exactly its own correct outcome, with no cross-talk. True
    multi-machine verification needs a live LAN and is out of scope for an
    automated test; this exercises the same middleware + run-manager code
    path two real browsers would hit.
    """
    monkeypatch.setattr(app_module, "WEB_HOST", "0.0.0.0")
    monkeypatch.setattr(app_module, "WEB_ACCESS_TOKEN", "secret-token-123")

    authed = TestClient(app_module.app, cookies={"cps_web_token": "secret-token-123"})
    anon = TestClient(app_module.app)

    import threading

    results: dict = {}

    def _authed_query():
        r = authed.post("/api/query", json={"query": "authed query"})
        results["authed_status"] = r.status_code
        results["authed_run_id"] = r.json().get("run_id") if r.status_code == 200 else None

    def _anon_query():
        r = anon.post("/api/query", json={"query": "anon query"})
        results["anon_status"] = r.status_code

    t1 = threading.Thread(target=_authed_query)
    t2 = threading.Thread(target=_anon_query)
    t1.start()
    t2.start()
    t1.join(timeout=10)
    t2.join(timeout=10)

    assert results["authed_status"] == 200
    assert results["authed_run_id"]
    assert results["anon_status"] == 401
