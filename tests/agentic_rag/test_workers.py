import pytest
from unittest.mock import MagicMock, patch

from src.agentic_rag.state import AgentState
from src.agentic_rag.workers import (
    _extract_metadata_filters,
    _reformulate_query,
    incident_retriever_agent,
    osha_enf_agent,
    osha_regs_agent,
    recommendation_specialist_agent,
    root_cause_analyst_agent,
)


@pytest.fixture(autouse=True)
def _force_supported_worker_params(monkeypatch):
    monkeypatch.setattr("src.agentic_rag.workers.WORKER_TEMPERATURE", 1.0)
    monkeypatch.setattr("src.agentic_rag.workers.REASONING_TEMPERATURE", 1.0)


@pytest.fixture
def initial_state():
    return AgentState(
        query="Test query about pressure vessels",
        sub_tasks=["Find incident facts"],
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


def test_extract_metadata_filters_state():
    """Test state abbreviation extraction."""
    filters = _extract_metadata_filters("pressure vessel failures in TX refineries")
    assert filters is not None
    assert filters["site_state"] == "TX"


def test_extract_metadata_filters_fatality():
    """Test fatality keyword extraction."""
    filters = _extract_metadata_filters("incidents with fatalities in chemical plants")
    assert filters is not None
    assert filters["fatality"] is True


def test_extract_metadata_filters_none():
    """Test returns None when no filters found."""
    filters = _extract_metadata_filters("general safety improvements")
    assert filters is None


def test_extract_metadata_filters_enforcement_citation_company():
    query = (
        "What violation was cited in citation 01001 under standard 19100212 A03 II "
        "at MORRIS BEAN & CO in Yellville, AR?"
    )
    filters = _extract_metadata_filters(query, tool_name="osha_enf")
    assert filters is not None
    assert filters["citation_id"] == "01001"
    assert filters["standard_number"] == "19100212 A03 II"
    assert filters["establishment_name"] == "MORRIS BEAN & CO"
    assert filters.get("site_state") == "AR"


def test_extract_metadata_filters_enforcement_entities():
    filters = _extract_metadata_filters(
        'What happened in OSHA accident report 14217376 at "AB Specialty Silicones" plant?',
        tool_name="osha_enf",
    )
    assert filters is not None
    assert filters["summary_nr"] == "14217376"
    assert "AB Specialty Silicones" in filters["establishment_name"]


def test_extract_metadata_filters_incident_name():
    filters = _extract_metadata_filters(
        "Summarize the incident at BP Texas City refinery.",
        tool_name="csb",
    )
    assert filters is not None
    assert "incident_name" in filters
    assert "Texas City refinery" in filters["incident_name"]


@patch("src.agentic_rag.workers.make_chat_model")
def test_reformulate_query(mock_llm_class):
    """Test query reformulation produces a search-optimized query."""
    mock_llm = mock_llm_class.return_value
    mock_response = MagicMock()
    mock_response.content = "PSM mechanical integrity pressure vessel inspection 1910.119(j)"
    mock_response.response_metadata = _MOCK_RESPONSE_METADATA
    mock_llm.invoke.return_value = mock_response

    result, llm_calls = _reformulate_query(
        "What causes pressure vessel failures?",
        ["Find incident data"],
        "CSB incident reports",
    )
    assert "1910.119" in result
    mock_llm.invoke.assert_called_once()
    assert len(llm_calls) == 1
    assert llm_calls[0]["call_purpose"] == "query_reformulation"


@patch("src.agentic_rag.workers.make_chat_model")
@patch("src.agentic_rag.workers.TOOLS", new_callable=dict)
def test_osha_regs_agent(mock_tools, mock_llm_class, initial_state):
    """Test OSHA regs worker with hybrid query."""
    mock_engine = MagicMock()
    mock_engine.query.return_value = (
        "Mock OSHA regulation text",
        [
            {"collection": "osha_regulations", "similarity_score": 0.8},
            {"collection": "osha_regulations", "similarity_score": 0.6},
        ],
    )
    mock_tools["osha_regulations"] = mock_engine

    mock_llm = mock_llm_class.return_value
    mock_response = MagicMock()
    mock_response.content = "reformulated query"
    mock_response.response_metadata = _MOCK_RESPONSE_METADATA
    mock_llm.invoke.return_value = mock_response

    state = osha_regs_agent(initial_state)
    assert "osha_regulations" in state["context_gathered"]
    assert len(state["citations"]) > 0
    assert "llm_calls" in state
    assert "retrieval_details" in state
    assert state["retrieval_details"][0]["collection"] == "osha_regulations"
    assert state["retrieval_details"][0]["avg_similarity_score"] == 0.7
    assert state["retrieval_details"][0]["min_similarity_score"] == 0.6


@patch("src.agentic_rag.workers.make_chat_model")
@patch("src.agentic_rag.workers.TOOLS", new_callable=dict)
def test_adaptive_top_k_for_enforcement(mock_tools, mock_llm_class, initial_state):
    mock_engine = MagicMock()
    mock_engine.query.return_value = ("Mock enforcement data", [{"collection": "osha_enf"}])
    mock_tools["osha_enf"] = mock_engine

    mock_llm = mock_llm_class.return_value
    mock_response = MagicMock()
    mock_response.content = "reformulated enforcement query"
    mock_response.response_metadata = _MOCK_RESPONSE_METADATA
    mock_llm.invoke.return_value = mock_response

    osha_enf_agent(initial_state)
    kwargs = mock_engine.query.call_args.kwargs
    assert kwargs["top_k"] == 25


@patch("src.agentic_rag.workers.make_chat_model")
@patch("src.agentic_rag.workers.TOOLS", new_callable=dict)
def test_adaptive_top_k_for_csb_recommendations(mock_tools, mock_llm_class, initial_state):
    mock_engine = MagicMock()
    mock_engine.query.return_value = ("Mock recommendation data", [{"collection": "csb_recommendations"}])
    mock_tools["csb_recommendations"] = mock_engine

    mock_llm = mock_llm_class.return_value
    mock_response = MagicMock()
    mock_response.content = "reformulated recommendations query"
    mock_response.response_metadata = _MOCK_RESPONSE_METADATA
    mock_llm.invoke.return_value = mock_response

    recommendation_specialist_agent(initial_state)
    kwargs = mock_engine.query.call_args.kwargs
    assert kwargs["top_k"] == 10


@patch("src.agentic_rag.workers.make_chat_model")
@patch("src.agentic_rag.workers.TOOLS", new_callable=dict)
def test_incident_retriever_agent(mock_tools, mock_llm_class, initial_state):
    """Test CSB incident worker with hybrid query."""
    mock_engine = MagicMock()
    mock_engine.query.return_value = ("Mock CSB incident data", [{"collection": "csb"}])
    mock_tools["csb"] = mock_engine

    mock_llm = mock_llm_class.return_value
    mock_response = MagicMock()
    mock_response.content = "reformulated query"
    mock_response.response_metadata = _MOCK_RESPONSE_METADATA
    mock_llm.invoke.return_value = mock_response

    state = incident_retriever_agent(initial_state)
    assert "csb" in state["context_gathered"]
    assert "Mock CSB incident data" in state["context_gathered"]["csb"]


@patch("src.agentic_rag.workers.make_chat_model")
@patch("src.agentic_rag.workers.TOOLS", new_callable=dict)
def test_root_cause_analyst(mock_tools, mock_llm_class, initial_state):
    """Test root-cause analyst with framework-injected prompting."""
    mock_osha = MagicMock()
    mock_osha.query.return_value = ("PSM definitions text", [])
    mock_csb = MagicMock()
    mock_csb.query.return_value = ("Incident facts text", [{"collection": "csb"}])
    mock_recs = MagicMock()
    mock_recs.query.return_value = ("Historical recommendations", [])

    mock_tools["osha_regulations"] = mock_osha
    mock_tools["csb"] = mock_csb
    mock_tools["csb_recommendations"] = mock_recs

    mock_llm = mock_llm_class.return_value
    mock_response = MagicMock()
    mock_response.content = "Root cause analysis: valve failure traced to MI deficiency"
    mock_response.response_metadata = _MOCK_RESPONSE_METADATA
    mock_llm.invoke.return_value = mock_response

    state = root_cause_analyst_agent(initial_state)

    assert "root_cause_analysis" in state["context_gathered"]
    assert "valve failure" in state["context_gathered"]["root_cause_analysis"][0]
    assert len(state["citations"]) > 0


# ---------------------------------------------------------------------------
# P3.2 — CSB incident-name fuzzy filter
# ---------------------------------------------------------------------------


def test_extract_metadata_filters_csb_watson_grinding():
    """CSB fuzzy lookup should match 'Watson Grinding' to canonical incident_name."""
    filters = _extract_metadata_filters(
        "What caused the explosion at Watson Grinding?",
        tool_name="csb",
    )
    assert filters is not None
    assert filters.get("incident_name") == "Watson_Grinding_and_Manufacturing"


def test_extract_metadata_filters_csb_ab_specialty_fuzzy():
    """CSB fuzzy lookup should match 'AB Specialty Silicones' by its full substring."""
    filters = _extract_metadata_filters(
        "Describe the AB Specialty Silicones fire investigation.",
        tool_name="csb",
    )
    assert filters is not None
    assert filters.get("incident_name") == "AB_Specialty_Silicones,_LLC"


def test_extract_metadata_filters_csb_ab_specialty_short():
    """CSB fuzzy lookup should match 'AB Specialty' (shorter substring)."""
    filters = _extract_metadata_filters(
        "What did CSB find at AB Specialty?",
        tool_name="csb",
    )
    assert filters is not None
    assert "AB_Specialty" in filters.get("incident_name", "")


def test_extract_metadata_filters_csb_facility_name_takes_precedence():
    """Quoted facility name extraction should override fuzzy CSB lookup."""
    filters = _extract_metadata_filters(
        'Incident at "Some Unknown Plant" — any CSB report?',
        tool_name="csb",
    )
    # Facility name regex extracts the quoted string; fuzzy map won't match
    assert filters is not None
    assert filters.get("incident_name") == "Some Unknown Plant"


def test_extract_metadata_filters_csb_no_match():
    """Unknown CSB facility name returns no incident_name filter."""
    filters = _extract_metadata_filters(
        "Tell me about a random fire.",
        tool_name="csb",
    )
    assert filters is None or "incident_name" not in (filters or {})
