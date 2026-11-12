import pytest
from unittest.mock import MagicMock, patch

from src.agentic_rag.state import AgentState
from src.agentic_rag.supervisor import (
    _parse_plan_output,
    detect_unverifiable_premise,
    route_from_plan,
    supervisor_agent,
    supervisor_plan,
    supervisor_revise,
    supervisor_synthesize,
)


@pytest.fixture(autouse=True)
def _force_supported_supervisor_params(monkeypatch):
    monkeypatch.setattr("src.agentic_rag.supervisor.SUPERVISOR_TEMPERATURE", 1.0)


def test_parse_plan_output_json_block():
    """With response_format=json_object, code blocks are not expected.
    If one arrives anyway, the parser returns a safe fallback."""
    output = '```json\n{"sub_tasks": ["A"], "agents_to_call": ["csb_incident"]}\n```'
    parsed = _parse_plan_output(output)
    assert parsed["agents_to_call"] == []


def test_parse_plan_output_raw_json():
    """Test parsing raw JSON without code block."""
    output = '{"sub_tasks": ["A", "B"], "agents_to_call": ["osha_regs", "csb_incident"]}'
    parsed = _parse_plan_output(output)
    assert len(parsed["agents_to_call"]) == 2


def test_parse_plan_output_fallback():
    """Test fallback on malformed output."""
    output = "I couldn't write JSON."
    parsed = _parse_plan_output(output)
    assert parsed["agents_to_call"] == []


@pytest.fixture
def initial_state():
    return AgentState(
        query="What are the causes of pressure vessel failures?",
        sub_tasks=[],
        active_agent="",
        pending_agents=[],
        planned_agents=[],
        planning_fast_routed=False,
        planning_fast_route_patterns=[],
        context_gathered={},
        draft_response="",
        final_response="",
        citations=[],
        validation_score=0.0,
        critic_feedback="",
        retry_count=0,
        llm_calls=[],
        retrieval_details=[],
        worker_errors=[],
    )


_MOCK_RESPONSE_METADATA = {
    "token_usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
    "model_name": "gpt-5-mini",
    "finish_reason": "stop",
    "system_fingerprint": "fp_test",
}


# ---------------------------------------------------------------------------
# supervisor_plan tests
# ---------------------------------------------------------------------------


@patch("src.agentic_rag.supervisor.make_chat_model")
def test_supervisor_plan_returns_dict(mock_llm_class, initial_state):
    """supervisor_plan returns a dict with sub_tasks and planned_agents."""
    mock_llm = mock_llm_class.return_value
    mock_response = MagicMock()
    mock_response.content = (
        '{"sub_tasks": ["Get CSB facts", "Get OSHA regs"], '
        '"agents_to_call": ["csb_incident", "osha_regs"]}'
    )
    mock_response.response_metadata = _MOCK_RESPONSE_METADATA
    mock_llm.invoke.return_value = mock_response

    result = supervisor_plan(initial_state)
    assert isinstance(result, dict)
    assert result["planned_agents"] == ["csb_incident", "osha_regs"]
    assert len(result["sub_tasks"]) == 2
    assert len(result["llm_calls"]) == 1
    assert result["planning_fast_routed"] is False


@patch("src.agentic_rag.supervisor.make_chat_model")
def test_supervisor_plan_defaults_to_csb(mock_llm_class, initial_state):
    """supervisor_plan defaults to csb_incident when no valid agents returned."""
    mock_llm = mock_llm_class.return_value
    mock_response = MagicMock()
    mock_response.content = '{"sub_tasks": ["Something"], "agents_to_call": ["unknown_agent"]}'
    mock_response.response_metadata = _MOCK_RESPONSE_METADATA
    mock_llm.invoke.return_value = mock_response

    result = supervisor_plan(initial_state)
    assert result["planned_agents"] == ["csb_incident"]
    assert result["planning_fast_routed"] is False


@patch("src.agentic_rag.supervisor.make_chat_model")
def test_supervisor_plan_fast_routes_without_llm(mock_llm_class, initial_state):
    """Quantitative query should fast-route and skip planning LLM call."""
    initial_state["query"] = "How many OSHA PSM violations occurred in Texas?"

    result = supervisor_plan(initial_state)

    # Stage I remediation (bda8196, 2026-09-09) superseded Fix 3: osha_enf is
    # retained when the query names OSHA explicitly. See
    # test_fast_route_quantitative_without_osha_suppresses_osha_enf for the
    # surviving Fix 3 behavior.
    assert result["planned_agents"] == ["data_analyst", "osha_enf"]
    assert result["planning_fast_routed"] is True
    assert result["planning_fast_route_patterns"] == [
        "quantitative query",
        "enforcement/violation query",
    ]
    assert result["llm_calls"] == []
    assert "Fast-routed:" in result["sub_tasks"][0]
    mock_llm_class.assert_not_called()


@patch("src.agentic_rag.supervisor.make_chat_model")
def test_supervisor_plan_fast_routes_multiple_patterns(mock_llm_class, initial_state):
    """Overlapping patterns should resolve to data_analyst after Fix 3 suppression."""
    initial_state["query"] = "How many violations happened after the fire incident?"

    result = supervisor_plan(initial_state)

    # P1 suppresses csb_incident and Fix 3 suppresses osha_enf for quantitative route.
    assert result["planned_agents"] == ["data_analyst"]
    assert result["planning_fast_routed"] is True
    assert "quantitative query" in result["planning_fast_route_patterns"]
    assert "incident facts query" in result["planning_fast_route_patterns"]
    assert "enforcement/violation query" in result["planning_fast_route_patterns"]
    mock_llm_class.assert_not_called()


def test_route_from_plan_returns_send_objects(initial_state):
    """route_from_plan reads planned_agents and returns Send objects."""
    initial_state["planned_agents"] = ["csb_incident", "osha_regs"]
    sends = route_from_plan(initial_state)
    assert isinstance(sends, list)
    assert len(sends) == 2
    node_names = [s.node for s in sends]
    assert "IncidentWorker" in node_names
    assert "OshaRegsWorker" in node_names


def test_route_from_plan_defaults_to_csb(initial_state):
    """route_from_plan defaults to csb_incident when planned_agents is empty."""
    initial_state["planned_agents"] = []
    sends = route_from_plan(initial_state)
    assert len(sends) == 1
    assert sends[0].node == "IncidentWorker"


# ---------------------------------------------------------------------------
# supervisor_synthesize tests
# ---------------------------------------------------------------------------


@patch("src.agentic_rag.supervisor.make_chat_model")
def test_supervisor_synthesize_produces_draft(mock_llm_class, initial_state):
    """supervisor_synthesize creates draft_response and routes to CRITIC."""
    initial_state["context_gathered"] = {"csb": ["Fact A"], "osha_regulations": ["Reg B"]}

    mock_llm = mock_llm_class.return_value
    mock_response = MagicMock()
    mock_response.content = "Based on Fact A and Reg B, this is the answer."
    mock_response.response_metadata = _MOCK_RESPONSE_METADATA
    mock_llm.invoke.return_value = mock_response

    result = supervisor_synthesize(initial_state)
    assert "Based on Fact A" in result["draft_response"]
    assert result["active_agent"] == "CRITIC"
    assert len(result["llm_calls"]) == 1
    assert result["llm_calls"][0]["call_purpose"] == "synthesis"


@patch("src.agentic_rag.supervisor.make_chat_model")
def test_supervisor_synthesize_includes_critic_feedback(mock_llm_class, initial_state):
    """supervisor_synthesize incorporates critic feedback on retry."""
    initial_state["context_gathered"] = {"csb": ["Fact A"]}
    initial_state["critic_feedback"] = "The draft fabricated a chemical name."
    initial_state["retry_count"] = 1

    mock_llm = mock_llm_class.return_value
    mock_response = MagicMock()
    mock_response.content = "Revised response."
    mock_response.response_metadata = _MOCK_RESPONSE_METADATA
    mock_llm.invoke.return_value = mock_response

    result = supervisor_synthesize(initial_state)

    call_args = mock_llm.invoke.call_args[0][0]
    prompt_text = call_args[1].content
    assert "fabricated a chemical name" in prompt_text
    assert result["active_agent"] == "CRITIC"


@patch("src.agentic_rag.supervisor.make_chat_model")
def test_supervisor_synthesize_returns_only_new_llm_calls(mock_llm_class, initial_state):
    """supervisor_synthesize returns only its own LLM calls, not accumulated."""
    initial_state["context_gathered"] = {"csb": ["Fact A"]}
    initial_state["llm_calls"] = [{"call_purpose": "previous", "node_name": "PlanNode"}]

    mock_llm = mock_llm_class.return_value
    mock_response = MagicMock()
    mock_response.content = "Answer."
    mock_response.response_metadata = _MOCK_RESPONSE_METADATA
    mock_llm.invoke.return_value = mock_response

    result = supervisor_synthesize(initial_state)
    assert len(result["llm_calls"]) == 1
    assert result["llm_calls"][0]["call_purpose"] == "synthesis"


@patch("src.agentic_rag.supervisor.make_chat_model")
def test_supervisor_synthesize_confidence_mode_prompt(mock_llm_class, initial_state):
    """High-score + enough context should enable confident synthesis guidance."""
    initial_state["validation_score"] = 0.95
    initial_state["context_gathered"] = {
        "csb": ["a", "b", "c"],
        "osha_enf": ["d", "e"],
    }

    mock_llm = mock_llm_class.return_value
    mock_response = MagicMock()
    mock_response.content = "Answer."
    mock_response.response_metadata = _MOCK_RESPONSE_METADATA
    mock_llm.invoke.return_value = mock_response

    supervisor_synthesize(initial_state)
    call_args = mock_llm.invoke.call_args[0][0]
    prompt_text = call_args[1].content
    assert "CONFIDENCE MODE" in prompt_text


# ---------------------------------------------------------------------------
# Legacy supervisor_agent tests
# ---------------------------------------------------------------------------


@patch("src.agentic_rag.supervisor.make_chat_model")
def test_supervisor_dispatch_phase(mock_llm_class, initial_state):
    """Test supervisor dispatches to next pending worker."""
    initial_state["context_gathered"] = {"csb": ["Some facts"]}
    initial_state["pending_agents"] = ["osha_regs", "root_cause"]

    state = supervisor_agent(initial_state)

    assert state["active_agent"] == "osha_regs"
    assert state["pending_agents"] == ["root_cause"]


@patch("src.agentic_rag.supervisor.make_chat_model")
def test_supervisor_synthesis_phase(mock_llm_class, initial_state):
    """Test supervisor synthesizes when all workers are done."""
    initial_state["context_gathered"] = {"csb": ["Fact A"], "osha_regulations": ["Reg B"]}
    initial_state["pending_agents"] = []

    mock_llm = mock_llm_class.return_value
    mock_response = MagicMock()
    mock_response.content = "Based on Fact A and Reg B, this is the answer."
    mock_response.response_metadata = _MOCK_RESPONSE_METADATA
    mock_llm.invoke.return_value = mock_response

    state = supervisor_agent(initial_state)

    assert "Based on Fact A" in state["draft_response"]
    assert state["active_agent"] == "CRITIC"
    assert "llm_calls" in state
    assert state["llm_calls"][0]["call_purpose"] == "synthesis"


# ---------------------------------------------------------------------------
# supervisor_revise tests
# ---------------------------------------------------------------------------


@patch("src.agentic_rag.supervisor.make_chat_model")
def test_supervisor_revise_produces_revised_draft(mock_llm_class, initial_state):
    """supervisor_revise returns a revised draft_response."""
    initial_state["draft_response"] = "Original draft with a fabricated claim."
    initial_state["critic_feedback"] = "The claim about chemical X is not in the context."
    initial_state["context_gathered"] = {"csb": ["Fact A", "Fact B"]}

    mock_llm = mock_llm_class.return_value
    mock_response = MagicMock()
    mock_response.content = "Revised draft without the fabricated claim."
    mock_response.response_metadata = _MOCK_RESPONSE_METADATA
    mock_llm.invoke.return_value = mock_response

    result = supervisor_revise(initial_state)
    assert result["draft_response"] == "Revised draft without the fabricated claim."
    assert result["active_agent"] == "CRITIC"
    assert len(result["llm_calls"]) == 1
    assert result["llm_calls"][0]["call_purpose"] == "revision"
    assert result["llm_calls"][0]["node_name"] == "RevisionNode"


@patch("src.agentic_rag.supervisor.make_chat_model")
def test_supervisor_revise_includes_feedback_in_prompt(mock_llm_class, initial_state):
    """supervisor_revise sends critic feedback in the revision prompt."""
    initial_state["draft_response"] = "Draft."
    initial_state["critic_feedback"] = "Missing citation for penalty amount."
    initial_state["context_gathered"] = {"osha_enf": ["Record 1"]}

    mock_llm = mock_llm_class.return_value
    mock_response = MagicMock()
    mock_response.content = "Revised."
    mock_response.response_metadata = _MOCK_RESPONSE_METADATA
    mock_llm.invoke.return_value = mock_response

    supervisor_revise(initial_state)

    call_args = mock_llm.invoke.call_args[0][0]
    prompt_text = call_args[1].content
    assert "Missing citation for penalty amount" in prompt_text
    assert "Draft." in prompt_text


@patch("src.agentic_rag.supervisor.make_chat_model")
def test_supervisor_revise_uses_source_summary_not_full_context(mock_llm_class, initial_state):
    """supervisor_revise sends a compact source summary, not full context."""
    initial_state["draft_response"] = "Draft."
    initial_state["critic_feedback"] = "Issue."
    initial_state["context_gathered"] = {
        "csb": ["chunk1", "chunk2", "chunk3"],
        "osha_enf": ["record1"],
    }

    mock_llm = mock_llm_class.return_value
    mock_response = MagicMock()
    mock_response.content = "Revised."
    mock_response.response_metadata = _MOCK_RESPONSE_METADATA
    mock_llm.invoke.return_value = mock_response

    supervisor_revise(initial_state)

    call_args = mock_llm.invoke.call_args[0][0]
    prompt_text = call_args[1].content
    # Source summary should show chunk counts, not actual content
    assert "csb: 3 chunk(s)" in prompt_text
    assert "osha_enf: 1 chunk(s)" in prompt_text
    # Full context should NOT be in the prompt
    assert "chunk1" not in prompt_text
    assert "record1" not in prompt_text


@patch("src.agentic_rag.supervisor.make_chat_model")
def test_supervisor_revise_returns_only_new_llm_calls(mock_llm_class, initial_state):
    """supervisor_revise returns only its own LLM calls, not accumulated."""
    initial_state["draft_response"] = "Draft."
    initial_state["critic_feedback"] = "Issue."
    initial_state["llm_calls"] = [{"call_purpose": "previous"}]

    mock_llm = mock_llm_class.return_value
    mock_response = MagicMock()
    mock_response.content = "Revised."
    mock_response.response_metadata = _MOCK_RESPONSE_METADATA
    mock_llm.invoke.return_value = mock_response

    result = supervisor_revise(initial_state)
    assert len(result["llm_calls"]) == 1
    assert result["llm_calls"][0]["call_purpose"] == "revision"


# ---------------------------------------------------------------------------
# Fast-route regression tests for the accident/fatality routing bug fix
# (supervisor.py commit: remove 'accident' from CSB pattern, add it to enf pattern)
# ---------------------------------------------------------------------------


@patch("src.agentic_rag.supervisor.make_chat_model")
def test_fast_route_osha_accident_routes_to_enf_not_csb(mock_llm_class, initial_state):
    """OSHA accident queries must route to osha_enf.

    Regression test for: 'accident' was in the CSB incident pattern, causing
    OSHA fatality/accident queries to be fast-routed only to csb_incident,
    missing osha_enf entirely.

    Note: queries containing both 'accident' AND words like 'what happened'
    (which match the CSB incident pattern) will correctly route to BOTH agents
    via pattern merging. This test uses a pure accident/fatality query that
    should NOT match the CSB pattern at all.
    """
    initial_state["query"] = (
        "Describe the fatal workplace accident recorded by OSHA "
        "at the chemical plant in 2006."
    )
    result = supervisor_plan(initial_state)

    assert result["planning_fast_routed"] is True
    assert "osha_enf" in result["planned_agents"], "OSHA accident must route to osha_enf"
    assert "csb_incident" not in result["planned_agents"], (
        "A pure OSHA accident query (no 'incident'/'fire'/'explosion'/'release'/'what happened') "
        "must NOT route to csb_incident"
    )
    mock_llm_class.assert_not_called()


@patch("src.agentic_rag.supervisor.make_chat_model")
def test_fast_route_csb_explosion_still_routes_to_csb(mock_llm_class, initial_state):
    """Pure CSB explosion queries must still reach csb_incident after the fix."""
    initial_state["query"] = "What caused the explosion at the Texas City refinery?"
    result = supervisor_plan(initial_state)

    assert result["planning_fast_routed"] is True
    assert "csb_incident" in result["planned_agents"]
    mock_llm_class.assert_not_called()


@patch("src.agentic_rag.supervisor.make_chat_model")
def test_fast_route_incident_and_accident_routes_to_enf(mock_llm_class, initial_state):
    """P1: enforcement+incident overlap suppresses csb_incident in favor of osha_enf.

    Updated for P1 (post-T2 fast-route overlap suppression). Previously, 'incident'
    + 'accident' in the same query would merge both csb_incident and osha_enf.
    P1 suppresses csb_incident when the enforcement/violation pattern fires alongside
    incident facts, fixing enforcement_factual routing (27.8% → 100% Tool Exact).

    Tradeoff: genuine cross-corpus queries that need both should use explicit
    CSB/OSHA keywords to trigger the cross-corpus CSB+OSHA pattern instead.
    """
    initial_state["query"] = (
        "What were the contributing factors to the chemical plant incident "
        "that led to the OSHA-recorded accident?"
    )
    result = supervisor_plan(initial_state)

    assert result["planning_fast_routed"] is True
    assert "osha_enf" in result["planned_agents"]
    # P1: csb_incident suppressed when enforcement/violation fires alongside incident facts
    assert "csb_incident" not in result["planned_agents"]
    mock_llm_class.assert_not_called()


@patch("src.agentic_rag.supervisor.make_chat_model")
def test_fast_route_fatality_routes_to_osha_enf(mock_llm_class, initial_state):
    """Quantitative fatality query naming OSHA keeps osha_enf alongside data_analyst.

    Stage I remediation (bda8196) superseded Fix 3 here: suppression of osha_enf
    now applies only when the query does not name OSHA explicitly.
    """
    initial_state["query"] = "How many worker fatalities were recorded under OSHA in 2019?"
    result = supervisor_plan(initial_state)

    assert result["planning_fast_routed"] is True
    assert "data_analyst" in result["planned_agents"]
    assert "osha_enf" in result["planned_agents"]
    mock_llm_class.assert_not_called()


@patch("src.agentic_rag.supervisor.make_chat_model")
def test_fast_route_comparative_query_includes_data_analyst(mock_llm_class, initial_state):
    """Comparative/aggregation queries must reach the SQL data analyst.

    Regression test for / in T1 v2.5, "Which state has the most
    OSHA violations?" was hijacked by the enforcement/violation fast-route
    (whose pluralized pattern matches "violations") and never reached
    data_analyst, dropping statistical_comparative routing accuracy.
    With the analytical/comparative route added, data_analyst must be in
    the merged agent list whenever the query expresses aggregation intent
    via "most/fewest/highest/lowest/top N/etc.".
    """
    initial_state["query"] = "Which state has the most OSHA violations?"
    result = supervisor_plan(initial_state)

    assert result["planning_fast_routed"] is True
    assert "data_analyst" in result["planned_agents"], (
        "Comparative analytical query must route to data_analyst (was hijacked "
        "by enforcement fast-route in v2.5; see P3b.6)"
    )
    assert "analytical/comparative query" in result["planning_fast_route_patterns"]
    mock_llm_class.assert_not_called()


@patch("src.agentic_rag.supervisor.make_chat_model")
def test_fast_route_top_n_pattern_matches(mock_llm_class, initial_state):
    """'top 5' / 'top 10' phrasing should trigger the analytical route."""
    initial_state["query"] = "What are the top 5 most common OSHA violation types by frequency?"
    result = supervisor_plan(initial_state)

    assert result["planning_fast_routed"] is True
    assert "data_analyst" in result["planned_agents"]
    mock_llm_class.assert_not_called()


@patch("src.agentic_rag.supervisor.make_chat_model")
def test_fast_route_analytical_route_does_not_swallow_unrelated_most(mock_llm_class, initial_state):
    """'almost' must not trigger the analytical route via stray 'most' substring.

    Word boundaries in the regex should prevent false positives on words
    like 'almost', 'foremost', 'utmost' that contain 'most' as a substring
    but are not aggregation intents.
    """
    initial_state["query"] = "What caused the explosion at the Texas City refinery?"
    result = supervisor_plan(initial_state)

    # Must not trigger the analytical route on this incident query.
    assert "analytical/comparative query" not in result.get(
        "planning_fast_route_patterns", []
    )


@patch("src.agentic_rag.supervisor.make_chat_model")
def test_fast_route_domain_anchored_average(mock_llm_class, initial_state):
    """'average penalty' style queries must route to data_analyst.

    Broadening of the analytical pattern catches
    dataset-aggregation queries that use 'average' anchored to a
    domain noun (penalty/violation/inspection/accident/etc.).
    """
    initial_state["query"] = "What is the average penalty amount for OSHA violations?"
    result = supervisor_plan(initial_state)

    assert result["planning_fast_routed"] is True
    assert "data_analyst" in result["planned_agents"]
    assert "analytical/comparative query" in result["planning_fast_route_patterns"]


@patch("src.agentic_rag.supervisor.make_chat_model")
def test_fast_route_temporal_change_routes_to_data_analyst(mock_llm_class, initial_state):
    """Trend questions using 'changed over' / 'over the past N years' must route to data_analyst.

    These are temporal-aggregation queries that previously slipped through
    both the quantitative pattern (no 'trend' word) and the original
    analytical pattern (no 'most/highest/etc'.).
    """
    initial_state["query"] = "How have OSHA penalty amounts changed over the past 5 years?"
    result = supervisor_plan(initial_state)

    assert result["planning_fast_routed"] is True
    assert "data_analyst" in result["planned_agents"]
    assert "analytical/comparative query" in result["planning_fast_route_patterns"]


@patch("src.agentic_rag.supervisor.make_chat_model")
def test_fast_route_bare_average_in_exposure_calc_does_not_hit_data_analyst(mock_llm_class, initial_state):
    """The 'average' broadening must be domain-anchored, not match exposure queries.

    'Average concentration' or '8-hour average exposure' belongs to the
    calculation/reasoning route, not data_analyst. The domain-anchored
    pattern requires 'average' near a dataset noun (penalty/violation/etc.).
    """
    initial_state["query"] = "What is the 8-hour average concentration of benzene during this shift?"
    result = supervisor_plan(initial_state)

    # Calculation route should fire (it has 'concentration ... shift' and 'TWA'-ish intent).
    # Analytical route must NOT fire — there's no domain anchor word.
    assert "analytical/comparative query" not in result.get(
        "planning_fast_route_patterns", []
    )
    assert "data_analyst" not in result["planned_agents"]


# ---------------------------------------------------------------------------
# P1: Fast-route overlap suppression tests
# ---------------------------------------------------------------------------


@patch("src.agentic_rag.supervisor.make_chat_model")
def test_fast_route_enforcement_violation_suppresses_osha_regs(mock_llm_class, initial_state):
    """enforcement_violation + regulatory/compliance overlap must drop osha_regs.

    P1 fix: enforcement queries mention 29 CFR / OSHA standard, triggering both
    'enforcement/violation query' and 'regulatory/compliance query'. The merged
    set incorrectly included osha_regs. The fix suppresses osha_regs when the
    enforcement pattern fires alongside regulatory/compliance.
    """
    initial_state["query"] = (
        "What penalty was issued for violation of 29 CFR 1910.119 cited in OSHA citation 1632714?"
    )
    result = supervisor_plan(initial_state)

    assert result["planning_fast_routed"] is True
    assert "osha_enf" in result["planned_agents"], "enforcement query must route to osha_enf"
    assert "osha_regs" not in result["planned_agents"], (
        "P1: osha_regs must be suppressed when enforcement/violation pattern fires "
        "alongside regulatory/compliance pattern"
    )
    mock_llm_class.assert_not_called()


@patch("src.agentic_rag.supervisor.make_chat_model")
def test_fast_route_enforcement_factual_suppresses_csb_incident(mock_llm_class, initial_state):
    """enforcement_violation + incident facts overlap must drop csb_incident.

    P1 fix: some enforcement_factual queries mention 'incident' or 'fire', triggering
    both 'enforcement/violation query' and 'incident facts query'. The merged set
    incorrectly included csb_incident. The fix suppresses csb_incident when the
    enforcement pattern fires alongside incident facts.
    """
    initial_state["query"] = (
        "What OSHA violations and penalties were cited after the fire incident at the facility?"
    )
    result = supervisor_plan(initial_state)

    assert result["planning_fast_routed"] is True
    assert "osha_enf" in result["planned_agents"], "enforcement query must route to osha_enf"
    assert "csb_incident" not in result["planned_agents"], (
        "P1: csb_incident must be suppressed when enforcement/violation pattern fires "
        "alongside incident facts pattern"
    )
    mock_llm_class.assert_not_called()


@patch("src.agentic_rag.supervisor.make_chat_model")
def test_fast_route_enforcement_violation_cites_cfr_suppresses_osha_regs(mock_llm_class, initial_state):
    """Enforcement query citing a standard number must route only to osha_enf."""
    initial_state["query"] = (
        "What enforcement actions were taken for OSHA standard 1910.147 violations?"
    )
    result = supervisor_plan(initial_state)

    assert result["planning_fast_routed"] is True
    assert "osha_enf" in result["planned_agents"]
    assert "osha_regs" not in result["planned_agents"], (
        "P1: osha_regs suppressed even when standard number (1910.xxx) is cited in violation query"
    )
    mock_llm_class.assert_not_called()


@patch("src.agentic_rag.supervisor.make_chat_model")
def test_fast_route_pure_regulatory_query_still_routes_to_osha_regs(mock_llm_class, initial_state):
    """A query that matches only regulatory/compliance (no enforcement) must still route to osha_regs.

    P1 suppression is conditional: osha_regs is only dropped when the enforcement
    pattern ALSO fires. A pure regulatory query must still reach osha_regs.
    """
    initial_state["query"] = (
        "What does 29 CFR 1910.119 require for process hazard analysis?"
    )
    result = supervisor_plan(initial_state)

    assert result["planning_fast_routed"] is True
    assert "osha_regs" in result["planned_agents"], (
        "P1 must NOT suppress osha_regs when enforcement pattern did NOT fire"
    )
    mock_llm_class.assert_not_called()


@patch("src.agentic_rag.supervisor.make_chat_model")
def test_fast_route_enforcement_only_no_suppression_needed(mock_llm_class, initial_state):
    """Enforcement-only query (no regulatory/compliance or incident patterns) routes as-is."""
    initial_state["query"] = "What penalties were issued for OSHA citations in 2020?"
    result = supervisor_plan(initial_state)

    assert result["planning_fast_routed"] is True
    assert "osha_enf" in result["planned_agents"]
    # No suppression should occur (nothing to suppress), and osha_regs was never added
    assert "osha_regs" not in result["planned_agents"]
    mock_llm_class.assert_not_called()


# ---------------------------------------------------------------------------
# Fix 2 tests: recommendation pattern routes to csb_recs only (2026-05-27)
# ---------------------------------------------------------------------------


@patch("src.agentic_rag.supervisor.make_chat_model")
def test_fast_route_recommendation_query_routes_csb_recs_only(mock_llm_class, initial_state):
    """Recommendation-focused queries route to csb_recs only, not csb_incident.

    Per A.7 analysis (2026-05-26): 29/35 incident_factual queries are
    recommendation-focused and don't need csb_incident.  The prevention/
    recommendations pattern agents changed from [csb_recs, csb_incident]
    to [csb_recs] only.
    """
    initial_state["query"] = "What did the CSB recommend developing for hazardous gas detection?"
    result = supervisor_plan(initial_state)

    assert result["planning_fast_routed"] is True
    assert "csb_recs" in result["planned_agents"]
    assert "csb_incident" not in result["planned_agents"], (
        "Recommendation-only queries should not dispatch to csb_incident (Fix 2)"
    )
    mock_llm_class.assert_not_called()


@patch("src.agentic_rag.supervisor.make_chat_model")
def test_fast_route_recommendation_plus_incident_suppresses_csb_incident(mock_llm_class, initial_state):
    """Rec-focused query mentioning 'explosion' suppresses csb_incident (Fix 2b).

    Per A.7 analysis: recommendation-focused queries don't need csb_incident even
    when incident keywords are present.  The csb_recs collection already contains
    sufficient incident context within the recommendation documents.
    """
    initial_state["query"] = "What safety recommendations came from the explosion investigation?"
    result = supervisor_plan(initial_state)

    assert result["planning_fast_routed"] is True
    assert "csb_recs" in result["planned_agents"]
    assert "csb_incident" not in result["planned_agents"], (
        "Rec-focused queries should suppress csb_incident even when incident "
        "facts pattern co-fires (Fix 2b, A.7 conclusion)"
    )
    mock_llm_class.assert_not_called()


# ---------------------------------------------------------------------------
# Fix 3 tests: analytical/data_analyst suppresses osha_enf (2026-05-27)
# ---------------------------------------------------------------------------


@patch("src.agentic_rag.supervisor.make_chat_model")
def test_fast_route_statistical_query_retains_osha_enf_when_osha_named(
    mock_llm_class, initial_state
):
    """Statistical 'how many' queries that name OSHA keep osha_enf.

    Fix 3 (2026-05-27) originally suppressed osha_enf on any analytical co-fire.
    Stage I remediation (bda8196, 2026-09-09) narrowed that to queries which do
    not name OSHA explicitly, so an explicitly-OSHA aggregation keeps both the
    DuckDB path and enforcement retrieval.
    """
    initial_state["query"] = "How many OSHA violations are recorded in the database?"
    result = supervisor_plan(initial_state)

    assert result["planning_fast_routed"] is True
    assert "data_analyst" in result["planned_agents"]
    assert "osha_enf" in result["planned_agents"], (
        "Query names OSHA explicitly, so Stage I routing retains osha_enf"
    )
    mock_llm_class.assert_not_called()


@patch("src.agentic_rag.supervisor.make_chat_model")
def test_fast_route_trend_query_retains_osha_enf_when_osha_named(mock_llm_class, initial_state):
    """Trend/temporal queries naming OSHA keep osha_enf (Stage I, supersedes Fix 3)."""
    initial_state["query"] = "What is the trend in OSHA inspection frequency over the last decade?"
    result = supervisor_plan(initial_state)

    assert result["planning_fast_routed"] is True
    assert "data_analyst" in result["planned_agents"]
    assert "osha_enf" in result["planned_agents"]
    mock_llm_class.assert_not_called()


@patch("src.agentic_rag.supervisor.make_chat_model")
def test_fast_route_top_n_violations_retains_osha_enf_when_osha_named(
    mock_llm_class, initial_state
):
    """'Top 5 OSHA violation types' keeps osha_enf (Stage I, supersedes Fix 3)."""
    initial_state["query"] = "What are the top 5 most common OSHA violation types by frequency?"
    result = supervisor_plan(initial_state)

    assert result["planning_fast_routed"] is True
    assert "data_analyst" in result["planned_agents"]
    assert "osha_enf" in result["planned_agents"]
    mock_llm_class.assert_not_called()


@patch("src.agentic_rag.supervisor.make_chat_model")
def test_fast_route_quantitative_without_osha_suppresses_osha_enf(mock_llm_class, initial_state):
    """Fix 3 survives for queries that do not name OSHA: data_analyst only.

    This is the retained half of Fix 3 after the Stage I remediation (bda8196)
    gated suppression on ``not has_explicit_osha``. Without this case, the
    suppression branch would have no coverage at all.
    """
    initial_state["query"] = "How many citations were issued in Texas refineries in 2019?"
    result = supervisor_plan(initial_state)

    assert result["planning_fast_routed"] is True
    assert result["planned_agents"] == ["data_analyst"]
    assert "osha_enf" not in result["planned_agents"], (
        "Query does not name OSHA, so Fix 3 suppression still applies"
    )
    mock_llm_class.assert_not_called()


@patch("src.agentic_rag.supervisor.make_chat_model")
def test_fast_route_csb_focused_query_strips_osha_enf(mock_llm_class, initial_state):
    """CSB-focused analytical queries keep CSB agents and drop osha_enf.

    Covers the ``is_csb_focused`` branch added by the Stage I remediation, which
    prevents osha_enf from hijacking CSB investigation analysis.
    """
    initial_state["query"] = (
        "What does the CSB say about the most common contributing factors across reports?"
    )
    result = supervisor_plan(initial_state)

    assert result["planning_fast_routed"] is True
    assert "osha_enf" not in result["planned_agents"]
    assert "csb_incident" in result["planned_agents"]
    assert "root_cause" in result["planned_agents"]
    mock_llm_class.assert_not_called()


@patch("src.agentic_rag.supervisor.make_chat_model")
def test_fast_route_hybrid_analytical_suppresses_osha_enf_and_regs(mock_llm_class, initial_state):
    """Hybrid analytical queries (CSB + data_analyst) suppress osha_enf and osha_regs.

    Query like 'Compare average OSHA penalty... across states with most CSB incidents'
    should route to data_analyst + csb_incident, not additionally to osha_enf/osha_regs.
    """
    initial_state["query"] = (
        "Compare the average OSHA penalty for serious vs willful violations "
        "across states with the most CSB-investigated incidents"
    )
    result = supervisor_plan(initial_state)

    assert result["planning_fast_routed"] is True
    assert "data_analyst" in result["planned_agents"]
    assert "csb_incident" in result["planned_agents"]
    assert "osha_enf" not in result["planned_agents"], (
        "Hybrid analytical queries should not dispatch to osha_enf (Fix 3)"
    )
    assert "osha_regs" not in result["planned_agents"], (
        "Hybrid analytical queries should not dispatch to osha_regs (Fix 3)"
    )
    mock_llm_class.assert_not_called()


@patch("src.agentic_rag.supervisor.make_chat_model")
def test_fast_route_cross_corpus_without_analytical_keeps_osha_regs(mock_llm_class, initial_state):
    """Cross-corpus CSB+OSHA query without analytical trigger keeps osha_regs.

    Regression: Fix 3 must not suppress osha_regs for non-analytical cross-corpus
    queries that legitimately need regulation retrieval.
    """
    initial_state["query"] = "How do CSB investigation findings relate to OSHA 1910.119 compliance?"
    result = supervisor_plan(initial_state)

    assert result["planning_fast_routed"] is True
    assert "csb_incident" in result["planned_agents"]
    assert "osha_regs" in result["planned_agents"], (
        "Non-analytical cross-corpus queries should keep osha_regs"
    )
    mock_llm_class.assert_not_called()


# ---------------------------------------------------------------------------
# Confidence-gated synthesis prompt tests (pre-T2 hedging fix §4a)
# ---------------------------------------------------------------------------


@patch("src.agentic_rag.supervisor.make_chat_model")
def test_confident_mode_threshold_085(mock_llm_class, initial_state):
    """Score 0.85 + ≥5 chunks should activate confident mode."""
    initial_state["validation_score"] = 0.85
    initial_state["context_gathered"] = {
        "osha_regs": [f"chunk{i}" for i in range(10)],
    }

    mock_llm = mock_llm_class.return_value
    mock_response = MagicMock()
    mock_response.content = "Answer."
    mock_response.response_metadata = _MOCK_RESPONSE_METADATA
    mock_llm.invoke.return_value = mock_response

    supervisor_synthesize(initial_state)

    prompt_text = mock_llm.invoke.call_args[0][0][1].content
    assert "CONFIDENCE MODE" in prompt_text


@patch("src.agentic_rag.supervisor.make_chat_model")
def test_confident_mode_first_pass_rich_context(mock_llm_class, initial_state):
    """First pass (score=0.0) with ≥10 chunks should activate confident mode."""
    initial_state["validation_score"] = 0.0
    initial_state["context_gathered"] = {
        "csb": [f"chunk{i}" for i in range(8)],
        "osha_regs": ["reg1", "reg2", "reg3"],
    }

    mock_llm = mock_llm_class.return_value
    mock_response = MagicMock()
    mock_response.content = "Answer."
    mock_response.response_metadata = _MOCK_RESPONSE_METADATA
    mock_llm.invoke.return_value = mock_response

    supervisor_synthesize(initial_state)

    prompt_text = mock_llm.invoke.call_args[0][0][1].content
    assert "CONFIDENCE MODE" in prompt_text
    assert "Do NOT include phrases like" in prompt_text


@patch("src.agentic_rag.supervisor.make_chat_model")
def test_confident_mode_first_pass_sparse_context(mock_llm_class, initial_state):
    """First pass (score=0.0) with <10 chunks should NOT activate confident mode."""
    initial_state["validation_score"] = 0.0
    initial_state["context_gathered"] = {
        "csb": ["chunk1", "chunk2", "chunk3"],
    }

    mock_llm = mock_llm_class.return_value
    mock_response = MagicMock()
    mock_response.content = "Answer."
    mock_response.response_metadata = _MOCK_RESPONSE_METADATA
    mock_llm.invoke.return_value = mock_response

    supervisor_synthesize(initial_state)

    prompt_text = mock_llm.invoke.call_args[0][0][1].content
    assert "CONFIDENCE MODE" not in prompt_text
    assert "Use cautious language" in prompt_text


@patch("src.agentic_rag.supervisor.make_chat_model")
def test_synthesis_prompt_confident_suppresses_blanket_disclaimer(mock_llm_class, initial_state):
    """In confident mode, rule 3 must suppress standalone 'insufficient evidence' paragraphs."""
    initial_state["validation_score"] = 0.90
    initial_state["context_gathered"] = {
        "osha_regs": [f"chunk{i}" for i in range(6)],
    }

    mock_llm = mock_llm_class.return_value
    mock_response = MagicMock()
    mock_response.content = "Answer."
    mock_response.response_metadata = _MOCK_RESPONSE_METADATA
    mock_llm.invoke.return_value = mock_response

    supervisor_synthesize(initial_state)

    prompt_text = mock_llm.invoke.call_args[0][0][1].content
    # Confident rule 3 should suppress the old boilerplate
    assert "Based on the available retrieved data, insufficient evidence was found" not in prompt_text
    # Should instead have the brief in-line gap instruction
    assert "note the gap briefly" in prompt_text


@patch("src.agentic_rag.supervisor.make_chat_model")
def test_synthesis_prompt_non_confident_requires_specific_gaps(mock_llm_class, initial_state):
    """In non-confident mode, rule 3 must require specific gap identification, not blanket disclaimers."""
    initial_state["validation_score"] = 0.0
    initial_state["context_gathered"] = {
        "csb": ["chunk1", "chunk2"],
    }

    mock_llm = mock_llm_class.return_value
    mock_response = MagicMock()
    mock_response.content = "Answer."
    mock_response.response_metadata = _MOCK_RESPONSE_METADATA
    mock_llm.invoke.return_value = mock_response

    supervisor_synthesize(initial_state)

    prompt_text = mock_llm.invoke.call_args[0][0][1].content
    # Old unconditional boilerplate must be gone
    assert "Based on the available retrieved data, insufficient evidence was found" not in prompt_text
    # New wording should require specificity
    assert "SPECIFIC aspect" in prompt_text
    assert "blanket disclaimers" in prompt_text


# ---------------------------------------------------------------------------
# unverifiable-premise / fictional-entity detection
# ---------------------------------------------------------------------------


class TestDetectUnverifiablePremise:
    """Unit tests for detect_unverifiable_premise (pure, no LLM)."""

    def test_fabricated_id_flagged(self):
        reason = detect_unverifiable_premise(
            "What penalty was issued for OSHA citation 999999111?",
            "context about scaffolding violations, no matching number",
        )
        assert reason is not None and "999999111" in reason

    def test_grounded_id_not_flagged(self):
        reason = detect_unverifiable_premise(
            "What happened in OSHA accident report 200673143 on 2001-02-14?",
            "OSHA accident report 200673143 occurred on 2001-02-14 at the plant",
        )
        assert reason is None

    def test_fictional_named_incident_flagged(self):
        reason = detect_unverifiable_premise(
            "Describe the 2019 Brframton Chemical explosion that killed 12 workers.",
            "CSB report on the Bayer CropScience incident at Institute, WV",
        )
        assert reason is not None and "Brframton" in reason

    def test_grounded_named_incident_not_flagged(self):
        reason = detect_unverifiable_premise(
            "What were the systemic causes of the Bayer CropScience explosion?",
            "CSB final report: Bayer CropScience incident at Institute, West Virginia",
        )
        assert reason is None

    def test_generic_regulatory_query_not_flagged(self):
        # No specific assertable entity → never flagged (keeps standard FP low).
        assert detect_unverifiable_premise(
            "What does 29 CFR 1910.119 require for process safety management?",
            "OSHA regulation 1910.119 PSM text",
        ) is None

    def test_generic_agency_tokens_not_treated_as_entities(self):
        # "OSHA accident" must not be captured as a named entity on its own.
        assert detect_unverifiable_premise(
            "According to the accident report, what happened to the employee?",
            "unrelated context",
        ) is None


@patch("src.agentic_rag.supervisor.make_chat_model")
def test_synthesis_injects_premise_block_for_fabricated_entity(mock_llm_class, initial_state):
    """When the query asserts an uncorroborated entity, the synthesis prompt must
    include the premise-verification directive and drop confident mode."""
    initial_state["query"] = "Summarize the findings of the incident at Zorptech Refinery."
    initial_state["validation_score"] = 0.0
    # Rich but irrelevant context (would otherwise trigger confident mode).
    initial_state["context_gathered"] = {
        "osha_enf": [f"unrelated enforcement chunk {i}" for i in range(12)],
    }

    mock_llm = mock_llm_class.return_value
    mock_response = MagicMock()
    mock_response.content = "Answer."
    mock_response.response_metadata = _MOCK_RESPONSE_METADATA
    mock_llm.invoke.return_value = mock_response

    supervisor_synthesize(initial_state)

    prompt_text = mock_llm.invoke.call_args[0][0][1].content
    assert "PREMISE VERIFICATION" in prompt_text
    assert "Zorptech Refinery" in prompt_text
    # Premise check overrides the chunk-count confidence gate.
    assert "CONFIDENCE MODE" not in prompt_text


@patch("src.agentic_rag.supervisor.make_chat_model")
def test_synthesis_no_premise_block_for_grounded_query(mock_llm_class, initial_state):
    """A grounded/normal query must NOT get the premise-verification block."""
    initial_state["query"] = "What are the causes of pressure vessel failures?"
    initial_state["context_gathered"] = {
        "csb": [f"pressure vessel failure chunk {i}" for i in range(6)],
    }

    mock_llm = mock_llm_class.return_value
    mock_response = MagicMock()
    mock_response.content = "Answer."
    mock_response.response_metadata = _MOCK_RESPONSE_METADATA
    mock_llm.invoke.return_value = mock_response

    supervisor_synthesize(initial_state)

    prompt_text = mock_llm.invoke.call_args[0][0][1].content
    assert "PREMISE VERIFICATION" not in prompt_text


def test_fast_route_general_duty_clause_routes_to_osha_enf_only(initial_state):
    """P03: General duty clause 5(a)(1) queries should route to osha_enf and suppress root_cause/csb_recs."""
    initial_state["query"] = "What recognized hazards were cited under Section 5(a)(1) general duty clause?"
    state = supervisor_plan(initial_state)
    assert state["planned_agents"] == ["osha_enf"]
    assert "root_cause" not in state["planned_agents"]
    assert "osha_regs" not in state["planned_agents"]
