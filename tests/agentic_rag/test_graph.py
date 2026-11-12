from unittest.mock import patch

from src.agentic_rag.graph import (
    MAX_CRITIC_RETRIES,
    _finalize_node,
    _retry_node,
    _route_from_critic,
    build_graph,
)


def test_route_from_critic_pass():
    """Test critic routes to finalize on passing score."""
    result = _route_from_critic({"validation_score": 0.9, "retry_count": 0})
    assert result == "FinalizeNode"


def test_route_from_critic_fail_retry():
    """Test critic routes to retry on failing score within budget."""
    result = _route_from_critic({"validation_score": 0.5, "retry_count": 0})
    assert result == "RetryNode"


def test_route_from_critic_max_retries():
    """Test critic routes to finalize when retry budget exhausted."""
    result = _route_from_critic({"validation_score": 0.5, "retry_count": MAX_CRITIC_RETRIES})
    assert result == "FinalizeNode"


def test_retry_node_increments_counter():
    """Test retry node increments retry_count."""
    state = {"retry_count": 0}
    result = _retry_node(state)
    assert result["retry_count"] == 1


def test_finalize_node_passing():
    """Test finalize moves draft to final without disclaimer."""
    state = {"draft_response": "Good answer.", "validation_score": 0.9}
    result = _finalize_node(state)
    assert result["final_response"] == "Good answer."
    assert "Note:" not in result["final_response"]


def test_finalize_node_low_confidence():
    """Test finalize adds disclaimer when score is below threshold."""
    state = {"draft_response": "Uncertain answer.", "validation_score": 0.5}
    result = _finalize_node(state)
    assert "Note:" in result["final_response"]
    assert "confidence score" in result["final_response"]
    assert "Uncertain answer." in result["final_response"]


@patch("src.agentic_rag.graph.incident_retriever_agent")
@patch("src.agentic_rag.graph.osha_regs_agent")
@patch("src.agentic_rag.graph.osha_enf_agent")
@patch("src.agentic_rag.graph.recommendation_specialist_agent")
@patch("src.agentic_rag.graph.root_cause_analyst_agent")
@patch("src.agentic_rag.graph.data_analyst_agent")
@patch("src.agentic_rag.graph.critic_agent")
@patch("src.agentic_rag.graph.route_from_plan")
@patch("src.agentic_rag.graph.supervisor_plan")
@patch("src.agentic_rag.graph.supervisor_revise")
@patch("src.agentic_rag.graph.supervisor_synthesize")
def test_build_graph(mock_synth, mock_revise, mock_plan, mock_route, mock_critic, mock_da, mock_rc, mock_recs, mock_enf, mock_regs, mock_inc):
    """Test that the state graph compiles with all expected nodes."""
    workflow = build_graph()

    expected_nodes = [
        "PlanNode", "SynthesizeNode", "RevisionNode",
        "OshaRegsWorker", "OshaEnfWorker",
        "IncidentWorker", "RecommendationsWorker", "RootCauseWorker",
        "DataAnalystWorker",
        "CriticNode", "RetryNode", "FinalizeNode",
    ]
    for node in expected_nodes:
        assert node in workflow.nodes, f"Missing node: {node}"
