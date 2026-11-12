"""Tests for Phase 6 multi-turn condensation (src/agentic_rag_web/condense.py).

Covers:
  a. Empty/None history passthrough (Gate G2 byte-identical query, zero LLM calls).
  b. Anaphora/pronoun resolution and ellipsis expansion, plus N=5 turn window cap.
  c. hedging carryover (refusal/uncertainty in turn 1 is preserved in turn 2).
  d. End-to-end POST /api/query with history through RunManager and build_web_graph().
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

import src.web.app as app_module
from src.agentic_rag_web.condense import (
    CONDENSE_SYSTEM_PROMPT,
    MAX_HISTORY_TURNS,
    condense_query,
)
from tests.agentic_rag_web.conftest import (
    FakeResponse,
    patched_llm_boundary,  # noqa: F401
)


# ---------------------------------------------------------------------------
# a. Empty history passthrough (Gate G2)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "history_value",
    [
        [],
        None,
        [{"query": "   ", "response": ""}],
    ],
)
def test_condense_empty_history_is_pure_noop_passthrough(history_value):
    """When conversation_history is empty or None, CondenseNode must return
    byte-identical query and raw_query without invoking make_chat_model."""
    raw = "What are the OSHA regulations for process safety management (29 CFR 1910.119)?"
    state = {
        "query": raw,
        "raw_query": raw,
        "conversation_history": history_value,
    }
    with patch("src.agentic_rag_web.condense.make_chat_model") as mock_factory:
        result = condense_query(state)  # type: ignore[arg-type]
        mock_factory.assert_not_called()

    assert result == {"query": raw, "raw_query": raw}
    assert result["query"] is raw or result["query"] == raw


def test_condense_passthrough_falls_back_to_query_when_raw_query_omitted():
    """Single-turn callers (e.g. G2 equivalence harness) that only set
    `state['query']` must also get a byte-identical passthrough."""
    raw = "How many fatalities occurred in refinery incidents?"
    state = {"query": raw}
    with patch("src.agentic_rag_web.condense.make_chat_model") as mock_factory:
        result = condense_query(state)  # type: ignore[arg-type]
        mock_factory.assert_not_called()

    assert result == {"query": raw, "raw_query": raw}


# ---------------------------------------------------------------------------
# b. Anaphora / pronoun resolution & ellipsis expansion
# ---------------------------------------------------------------------------


def test_condense_resolves_pronoun_reference_with_history():
    """Turn 2 'What caused it?' resolves 'it' against Turn 1's incident and
    preserves `raw_query` as the literal user follow-up."""
    captured_messages = []

    mock_llm = MagicMock()

    def _invoke(messages):
        captured_messages.extend(messages)
        return FakeResponse(
            "What were the root causes of the 2013 West Fertilizer explosion?"
        )

    mock_llm.invoke.side_effect = _invoke

    state = {
        "query": "What caused it?",
        "raw_query": "What caused it?",
        "conversation_history": [
            {
                "query": "What happened in the West Fertilizer explosion?",
                "response": (
                    "On April 17, 2013, a fire and ammonium nitrate explosion "
                    "occurred at the West Fertilizer Company storage facility in West, Texas."
                ),
            }
        ],
    }

    with patch("src.agentic_rag_web.condense.make_chat_model", return_value=mock_llm) as mock_factory:
        result = condense_query(state)  # type: ignore[arg-type]

    mock_factory.assert_called_once()
    assert mock_factory.call_args.args[0] == "worker"
    assert result["raw_query"] == "What caused it?"
    assert result["query"] == "What were the root causes of the 2013 West Fertilizer explosion?"
    assert len(result["llm_calls"]) == 1
    assert result["llm_calls"][0]["node_name"] == "CondenseNode"
    assert result["llm_calls"][0]["call_purpose"] == "condense"

    # Both prior query AND prior response must be passed into the prompt.
    assert len(captured_messages) == 2
    assert captured_messages[0].content == CONDENSE_SYSTEM_PROMPT
    human_prompt = captured_messages[1].content
    assert "West Fertilizer explosion" in human_prompt
    assert "April 17, 2013" in human_prompt
    assert "What caused it?" in human_prompt


def test_condense_caps_history_to_last_n_turns():
    """Only the most recent MAX_HISTORY_TURNS (5) turns are sent in the prompt."""
    captured_prompts = []
    mock_llm = MagicMock()
    mock_llm.invoke.side_effect = lambda msgs: (
        captured_prompts.append(msgs[1].content)
        or FakeResponse("Standalone question from recent turns")
    )

    history = [
        {"query": f"Question number {i}", "response": f"Response number {i}"}
        for i in range(1, 9)
    ]
    state = {
        "query": "And what about the penalties?",
        "raw_query": "And what about the penalties?",
        "conversation_history": history,
    }

    with patch("src.agentic_rag_web.condense.make_chat_model", return_value=mock_llm):
        condense_query(state)  # type: ignore[arg-type]

    assert MAX_HISTORY_TURNS == 5
    prompt_text = captured_prompts[0]
    # Turns 1..3 should be dropped; turns 4..8 should be present.
    assert "Question number 1" not in prompt_text
    assert "Question number 3" not in prompt_text
    assert "Question number 4" in prompt_text
    assert "Question number 8" in prompt_text


# ---------------------------------------------------------------------------
# c. hedging carryover (Safety Requirement, Checklist )
# ---------------------------------------------------------------------------


def test_condense_preserves_r11_hedging_when_llm_includes_uncertainty():
    """When Turn 1 refused/hedged a fictional incident and the LLM carries
    forward the uncertainty framing, the hedged rewrite is preserved."""
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = FakeResponse(
        "Regarding the unverified 2024 Apex Cryogenics explosion (for which no "
        "OSHA or CSB records were found in the prior turn), what caused it?"
    )

    state = {
        "query": "What caused it?",
        "raw_query": "What caused it?",
        "conversation_history": [
            {
                "query": "What happened in the 2024 Apex Cryogenics explosion?",
                "response": (
                    "I could not find records matching '2024 Apex Cryogenics explosion' "
                    "in the available OSHA or CSB sources. The premise appears unverified."
                ),
            }
        ],
    }

    with patch("src.agentic_rag_web.condense.make_chat_model", return_value=mock_llm):
        result = condense_query(state)  # type: ignore[arg-type]

    assert "unverified" in result["query"].lower()
    assert "no osha or csb records" in result["query"].lower()
    assert result["raw_query"] == "What caused it?"


def test_condense_r11_safeguard_prevents_premise_laundering_if_llm_omits_hedging():
    """Even if the underlying model naively rewrites Turn 2 as a factual
    question about the fictional incident, the deterministic  carryover
    safeguard must inject uncertainty framing so the false premise is never
    laundered into an established fact."""
    mock_llm = MagicMock()
    # Simulate an uncooperative LLM that laundered the refused premise into fact:
    mock_llm.invoke.return_value = FakeResponse(
        "What caused the 2024 Apex Cryogenics explosion?"
    )

    state = {
        "query": "What caused it?",
        "raw_query": "What caused it?",
        "conversation_history": [
            {
                "query": "What happened in the 2024 Apex Cryogenics explosion?",
                "response": (
                    "I could not find records matching the 2024 Apex Cryogenics "
                    "explosion in the available OSHA/CSB sources."
                ),
            }
        ],
    }

    with patch("src.agentic_rag_web.condense.make_chat_model", return_value=mock_llm):
        result = condense_query(state)  # type: ignore[arg-type]

    rewritten_lower = result["query"].lower()
    assert "unverified" in rewritten_lower
    assert "no osha or csb records were found" in rewritten_lower
    assert "2024 apex cryogenics explosion" in rewritten_lower
    assert result["raw_query"] == "What caused it?"


# ---------------------------------------------------------------------------
# d. End-to-end API & Web Graph multi-turn flow
# ---------------------------------------------------------------------------


def test_api_query_accepts_history_and_executes_condense_node(
    tmp_path,
    monkeypatch,
    patched_llm_boundary,  # noqa: F811
):
    """POST /api/query with `history` threads `conversation_history` into
    `run_streaming()` and `CondenseNode`, rewriting `query` while preserving
    `raw_query`."""
    monkeypatch.setattr(app_module, "LOG_DIR", tmp_path / "logs" / "runs")
    with TestClient(app_module.app) as client:
        history = [
            {
                "query": "What happened in the West Fertilizer explosion?",
                "response": "A fire and detonation occurred at West Fertilizer in 2013.",
            }
        ]
        resp = client.post(
            "/api/query",
            json={"query": "What were the root causes of it?", "history": history},
        )
        assert resp.status_code == 200
        run_id = resp.json()["run_id"]

        stream_resp = client.get(f"/api/stream/{run_id}")
        assert stream_resp.status_code == 200

        events = []
        for block in stream_resp.text.strip().split("\n\n"):
            data_lines = [
                ln for ln in block.splitlines() if ln.startswith("data: ")
            ]
            if data_lines:
                events.append(json.loads(data_lines[0][len("data: ") :]))

        condense_complete = next(
            e
            for e in events
            if e.get("event") == "step_complete" and e.get("node") == "CondenseNode"
        )
        assert condense_complete["update"]["raw_query"] == "What were the root causes of it?"
        assert "West Fertilizer explosion" in condense_complete["update"]["query"]

        final_event = events[-1]
        assert final_event["event"] == "pipeline_complete"
        assert final_event["state"]["raw_query"] == "What were the root causes of it?"
        assert "West Fertilizer explosion" in final_event["state"]["query"]
        assert len(final_event["state"]["conversation_history"]) == 1

        # Verify on-disk run log preserves CondenseNode's LLM call even after
        # PlanNode fast-routes with `llm_calls: []`.
        log_files = list((tmp_path / "logs" / "runs").glob("*.json"))
        assert len(log_files) == 1
        saved_log = json.loads(log_files[0].read_text(encoding="utf-8"))
        condense_calls = [
            c for c in saved_log["llm_calls"] if c.get("node_name") == "CondenseNode"
        ]
        assert len(condense_calls) == 1
        assert condense_calls[0]["call_purpose"] == "condense"


def test_condense_apex_cryogenics_live_wording_and_downstream_premise_check():
    """Regression test for the live Turn 1 / Turn 2 Apex Cryogenics wording:
    Turn 1 response says 'No record of an \"Apex Cryogenics\" explosion exists in
    the retrieved evidence.' and Turn 2 asks 'What caused it and what penalties
    were issued?'. Verifies  carryover and downstream `detect_unverifiable_premise`."""
    from src.agentic_rag.supervisor import detect_unverifiable_premise

    mock_llm = MagicMock()
    mock_llm.invoke.return_value = FakeResponse(
        'What caused the "2024 Apex Cryogenics explosion" and what penalties were issued?'
    )

    state = {
        "query": "What caused it and what penalties were issued?",
        "raw_query": "What caused it and what penalties were issued?",
        "conversation_history": [
            {
                "query": "What happened in the 2024 Apex Cryogenics explosion?",
                "response": (
                    '## Executive Summary\n\n**No record of an "Apex Cryogenics" '
                    "explosion exists in the retrieved evidence.** The premise "
                    "appears unfounded as stated, or at minimum unverifiable."
                ),
            }
        ],
    }

    with patch("src.agentic_rag_web.condense.make_chat_model", return_value=mock_llm):
        result = condense_query(state)  # type: ignore[arg-type]

    rewritten = result["query"]
    assert "unverified" in rewritten.lower()
    assert "apex cryogenics" in rewritten.lower()
    # Downstream SynthesizeNode premise detector must flag the ungrounded entity:
    premise_reason = detect_unverifiable_premise(
        rewritten, "unrelated OSHA citation 1910.119 context"
    )
    assert premise_reason is not None
    assert "Apex Cryogenics" in premise_reason


def test_static_cache_busting_and_frontend_multiturn_wiring(
    patched_llm_boundary,  # noqa: F811
):
    """Ensure `/` and `/static/app.js` disable stale browser caching and wire
    multi-turn history extraction, CondenseNode UI rendering, session export,
    and native `<dialog>` session deletion."""
    with TestClient(app_module.app) as client:
        index_resp = client.get("/")
        assert index_resp.status_code == 200
        assert "no-cache" in index_resp.headers.get("cache-control", "")
        assert "/static/app.js?v=phase6_4" in index_resp.text
        assert 'id="export-session-btn"' in index_resp.text
        assert 'id="delete-session-dialog"' in index_resp.text
        assert 'closedby="any"' in index_resp.text
        assert 'aria-labelledby="delete-dialog-title"' in index_resp.text
        assert 'role="dialog"' not in index_resp.text
        assert "aria-modal" not in index_resp.text

        js_resp = client.get("/static/app.js")
        assert js_resp.status_code == 200
        assert "no-cache" in js_resp.headers.get("cache-control", "")
        assert "getSessionHistory" in js_resp.text
        assert "session_id: currentSessionId" in js_resp.text
        assert "fs.conversation_history" in js_resp.text
        assert 'step.node === "CondenseNode"' in js_resp.text
        assert "turn.streamingHtml" in js_resp.text
        assert "exportCurrentSessionMarkdown" in js_resp.text
        assert "deleteConfirmedSession" in js_resp.text
        assert 'deleteDialog.returnValue = ""' in js_resp.text


def test_multiturn_session_grouping_in_sidebar_and_turn_reconstruction(
    patched_llm_boundary,  # noqa: F811
    tmp_path,
    monkeypatch,
):
    """Step 1: Verify multi-turn runs sharing a `session_id` collapse into a single
    sidebar session entry in `GET /api/runs` (`turn_count == 2`) and return all
    chronological turns in `GET /api/runs/{session_id}`."""
    monkeypatch.setattr(app_module, "LOG_DIR", tmp_path / "logs" / "runs")
    with TestClient(app_module.app) as client:
        # Turn 1 (starts new session)
        r1 = client.post(
            "/api/query",
            json={"query": "What happened in the 2005 BP Texas City explosion?"},
        )
        assert r1.status_code == 200
        r1_data = r1.json()
        run1_id = r1_data["run_id"]
        session_id = r1_data["session_id"]
        assert session_id == run1_id

        # Drain Turn 1 SSE stream so run completes and persists to disk
        s1 = client.get(f"/api/stream/{run1_id}")
        assert s1.status_code == 200

        # Turn 2 (continues same session)
        r2 = client.post(
            "/api/query",
            json={
                "query": "What recommendations were made?",
                "session_id": session_id,
                "history": [
                    {
                        "query": "What happened in the 2005 BP Texas City explosion?",
                        "response": "A hydrocarbon vapor cloud explosion occurred.",
                    }
                ],
            },
        )
        assert r2.status_code == 200
        r2_data = r2.json()
        run2_id = r2_data["run_id"]
        assert r2_data["session_id"] == session_id
        assert run2_id != run1_id

        # Drain Turn 2 SSE stream
        s2 = client.get(f"/api/stream/{run2_id}")
        assert s2.status_code == 200

        # GET /api/runs should group both runs into a single session entry
        runs_resp = client.get("/api/runs")
        assert runs_resp.status_code == 200
        runs_list = runs_resp.json()["runs"]
        matching = [
            item
            for item in runs_list
            if (item.get("session_id") or item.get("run_id")) == session_id
        ]
        assert len(matching) == 1
        grouped = matching[0]
        assert grouped["turn_count"] == 2
        assert grouped["query"] == "What happened in the 2005 BP Texas City explosion?"
        assert set(grouped["turn_run_ids"]) == {run1_id, run2_id}

        # GET /api/runs/{session_id} should return chronological turns with rendered HTML
        detail_resp = client.get(f"/api/runs/{session_id}")
        assert detail_resp.status_code == 200
        detail = detail_resp.json()
        assert detail["session_id"] == session_id
        turns = detail["turns"]
        assert len(turns) == 2
        assert turns[0]["run_id"] == run1_id
        assert turns[0]["query"] == "What happened in the 2005 BP Texas City explosion?"
        assert turns[0]["final_state"].get("final_response_html")
        assert turns[1]["run_id"] == run2_id
        assert turns[1]["query"] == "What recommendations were made?"
        assert turns[1]["final_state"].get("final_response_html")


def test_synthesis_token_and_step_complete_stream_formatted_html(tmp_path):
    """Step 3: Verify `RunManager._emit` attaches stabilized progressive HTML to
    `synthesis_token` events (zero raw Markdown pipes `|`, `##`, or `**` even on
    partial table headers) and `draft_response_html` to `SynthesizeNode`
    `step_complete` events when `step_start` arrives after tokens (`stream_mode='updates'`)."""
    from src.web.run_manager import Run, RunManager

    mgr = RunManager(
        project_root=app_module.PROJECT_ROOT,
        build_graph_fn=lambda: None,
        log_dir=str(tmp_path),
    )
    run = Run(run_id="r-html-test", query="test", session_id="r-html-test")

    # Simulate exact LangGraph `stream_mode="updates"` event order:
    # 1. Tokens arrive FIRST while SynthesizeNode runs
    mgr._emit(run, {"event": "synthesis_token", "token": "## "})
    mgr._emit(run, {"event": "synthesis_token", "token": "Heading\n\n"})
    mgr._emit(run, {"event": "synthesis_token", "token": "**Bold "})
    mgr._emit(run, {"event": "synthesis_token", "token": "text**\n\n"})
    mgr._emit(run, {"event": "synthesis_token", "token": "| Col1 | Col2"})
    mgr._emit(run, {"event": "synthesis_token", "token": " |\n|---|"})
    mgr._emit(run, {"event": "synthesis_token", "token": "---|\n| Val1 | Val2"})
    # 2. `step_start` and `step_complete` arrive AFTER tokens finish
    mgr._emit(run, {"event": "step_start", "node": "SynthesizeNode", "step": 6})
    full_draft = "## Heading\n\n**Bold text**\n\n| Col1 | Col2 |\n|---|---|\n| Val1 | Val2 |"
    mgr._emit(
        run,
        {
            "event": "step_complete",
            "node": "SynthesizeNode",
            "step": 6,
            "update": {"draft_response": full_draft},
        },
    )

    token_events = [e for e in run.events if e.get("event") == "synthesis_token"]
    assert len(token_events) == 7
    for ev in token_events:
        assert "|" not in ev["html"], f"Raw pipe leaked into HTML: {ev['html']!r}"
        assert "**" not in ev["html"], f"Raw bold leaked into HTML: {ev['html']!r}"
        assert "##" not in ev["html"], f"Raw heading leaked into HTML: {ev['html']!r}"

    # Partial bold renders as <strong> immediately before closing `**` arrives
    assert "<strong>Bold</strong>" in token_events[2]["html"]
    # Partial table header renders as <table> immediately before `|---|---|` completes
    assert "<table>" in token_events[4]["html"]
    assert "<th>Col1</th>" in token_events[4]["html"]
    assert "<table>" in token_events[5]["html"]
    assert "<td>Val1</td>" in token_events[6]["html"]

    step_complete_ev = [e for e in run.events if e.get("event") == "step_complete"][0]
    assert "<table>" in step_complete_ev["update"]["draft_response_html"]
    assert "<td>Val2</td>" in step_complete_ev["update"]["draft_response_html"]
    assert run._streaming_md == ""


def test_web_synthesize_node_records_streaming_token_usage():
    """Step 2: Verify `_web_synthesize_node` populates non-null `prompt_tokens`,
    `completion_tokens`, `total_tokens`, and `estimated_cost_usd` during streaming
    even when the LLM provider omits token usage on streamed chunks."""
    from langchain_core.messages import AIMessageChunk

    from src.agentic_rag.supervisor import _TOKEN_EMITTER
    from src.agentic_rag_web.graph import _web_synthesize_node

    class _StreamingStubLLM:
        model_name = "deepseek-v4-flash"

        def stream(self, messages, *args, **kwargs):
            yield AIMessageChunk(
                content="## Summary\n\n",
                response_metadata={"model_name": "deepseek-v4-flash"},
            )
            yield AIMessageChunk(
                content="Grounded synthesis [Source 1].",
                response_metadata={"model_name": "deepseek-v4-flash"},
            )

    emitted_tokens: list[str] = []
    tok = _TOKEN_EMITTER.set(emitted_tokens.append)
    try:
        with patch(
            "src.agentic_rag.supervisor.make_chat_model",
            return_value=_StreamingStubLLM(),
        ):
            state = {
                "query": "What caused the BP Texas City explosion?",
                "raw_query": "What caused the BP Texas City explosion?",
                "context_gathered": [
                    "[Source 1: CSB Report] Raffinate splitter tower overfilled."
                ],
                "sub_queries": ["BP Texas City explosion cause"],
                "critic_feedback": None,
                "critic_passed": None,
                "iteration_count": 1,
                "draft_response": None,
                "final_response": None,
                "llm_calls": [],
            }
            out = _web_synthesize_node(state)  # type: ignore[arg-type]
    finally:
        _TOKEN_EMITTER.reset(tok)

    assert "".join(emitted_tokens) == "## Summary\n\nGrounded synthesis [Source 1]."
    calls = out.get("llm_calls") or []
    assert len(calls) == 1
    synth_call = calls[0]
    assert synth_call["call_purpose"] == "synthesis"
    assert isinstance(synth_call["prompt_tokens"], int) and synth_call["prompt_tokens"] > 0
    assert (
        isinstance(synth_call["completion_tokens"], int)
        and synth_call["completion_tokens"] > 0
    )
    assert (
        synth_call["total_tokens"]
        == synth_call["prompt_tokens"] + synth_call["completion_tokens"]
    )
    assert isinstance(synth_call["estimated_cost_usd"], float)
    assert synth_call["estimated_cost_usd"] > 0.0


def test_delete_session_removes_all_turns_and_returns_404_after(
    patched_llm_boundary,  # noqa: F811
    tmp_path,
    monkeypatch,
):
    """Step 4: Verify `DELETE /api/runs/{session_id}` deletes all stored logs and
    in-memory runs for a multi-turn session, returns 404 for unknown sessions,
    and returns 409 if an active run is in-flight."""
    from src.web.run_manager import Run

    runs_dir = tmp_path / "logs" / "runs"
    runs_dir.mkdir(parents=True)
    monkeypatch.setattr(app_module, "LOG_DIR", runs_dir)

    with TestClient(app_module.app) as client:
        # Unknown session -> 404
        missing_resp = client.delete("/api/runs/does-not-exist")
        assert missing_resp.status_code == 404

        # Active in-flight run -> 409
        manager = app_module.app.state.run_manager
        active_run = Run("active-123", "Running query", session_id="active-123")
        active_run.done = False
        manager._runs["active-123"] = active_run
        conflict_resp = client.delete("/api/runs/active-123")
        assert conflict_resp.status_code == 409
        manager._runs.pop("active-123", None)

        # Create a 2-turn completed session
        r1 = client.post(
            "/api/query",
            json={"query": "What happened in the 2005 BP Texas City explosion?"},
        )
        assert r1.status_code == 200
        run1_id = r1.json()["run_id"]
        session_id = r1.json()["session_id"]
        assert client.get(f"/api/stream/{run1_id}").status_code == 200

        r2 = client.post(
            "/api/query",
            json={
                "query": "What recommendations were made?",
                "session_id": session_id,
                "history": [
                    {
                        "query": "What happened in the 2005 BP Texas City explosion?",
                        "response": "A hydrocarbon vapor cloud explosion occurred.",
                    }
                ],
            },
        )
        assert r2.status_code == 200
        run2_id = r2.json()["run_id"]
        assert client.get(f"/api/stream/{run2_id}").status_code == 200

        # Both JSON logs exist on disk before deletion
        assert len(list(runs_dir.glob("*.json"))) == 2

        # Delete session
        del_resp = client.delete(f"/api/runs/{session_id}")
        assert del_resp.status_code == 200
        del_data = del_resp.json()
        assert del_data["deleted"] is True
        assert del_data["session_id"] == session_id
        assert del_data["deleted_runs"] == 2

        # Disk logs and memory entries are gone
        assert list(runs_dir.glob("*.json")) == []
        assert client.get(f"/api/runs/{session_id}").status_code == 404
        assert client.get(f"/api/runs/{run2_id}").status_code == 404
        runs_after = client.get("/api/runs").json()["runs"]
        assert all(item["run_id"] != session_id for item in runs_after)

        # Legacy CLI logs without a `run_id` field use filename stem as ID and delete cleanly
        legacy_dir = tmp_path / "logs" / "cli"
        legacy_dir.mkdir(parents=True)
        legacy_stem = "2026-03-07T06-30-18_legacy_psm_query"
        (legacy_dir / f"{legacy_stem}.json").write_text(
            json.dumps(
                {
                    "timestamp": "2026-03-07T06:28:28+00:00",
                    "query": "What are the PSM requirements for mechanical integrity?",
                    "final_state": {"final_response": "Legacy answer"},
                }
            ),
            encoding="utf-8",
        )
        listed = client.get("/api/runs").json()["runs"]
        assert any(item["run_id"] == legacy_stem for item in listed)
        assert client.get(f"/api/runs/{legacy_stem}").status_code == 200
        del_legacy = client.delete(f"/api/runs/{legacy_stem}")
        assert del_legacy.status_code == 200
        assert del_legacy.json()["deleted_runs"] == 1
        assert list(legacy_dir.glob("*.json")) == []
