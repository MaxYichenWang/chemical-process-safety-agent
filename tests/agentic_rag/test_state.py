from src.agentic_rag.state import AgentState, _last_writer_wins, _merge_context


def test_agent_state_initialization():
    """Test that AgentState can be initialized with all fields."""
    state: AgentState = {
        "query": "What happened?",
        "sub_tasks": ["task 1", "task 2"],
        "active_agent": "supervisor",
        "pending_agents": ["osha_regs", "csb_incident"],
        "planned_agents": ["osha_regs", "csb_incident"],
        "planning_fast_routed": False,
        "planning_fast_route_patterns": [],
        "context_gathered": {"csb": [{"doc": "info"}]},
        "draft_response": "Draft",
        "final_response": "",
        "citations": [{"source": "csb", "text": "info"}],
        "validation_score": 0.95,
        "critic_feedback": "",
        "retry_count": 0,
        "llm_calls": [],
        "retrieval_details": [],
        "worker_errors": [],
    }

    assert state["query"] == "What happened?"
    assert len(state["sub_tasks"]) == 2
    assert state["active_agent"] == "supervisor"
    assert "csb" in state["context_gathered"]
    assert state["validation_score"] == 0.95
    assert len(state["pending_agents"]) == 2
    assert state["critic_feedback"] == ""
    assert state["retry_count"] == 0
    assert state["llm_calls"] == []
    assert state["retrieval_details"] == []
    assert state["worker_errors"] == []


def test_merge_context_disjoint():
    """Merge context dicts with disjoint keys."""
    a = {"csb": ["doc1"]}
    b = {"osha": ["doc2"]}
    merged = _merge_context(a, b)
    assert merged == {"csb": ["doc1"], "osha": ["doc2"]}


def test_merge_context_overlapping():
    """Merge context dicts with overlapping keys extends lists."""
    a = {"csb": ["doc1"]}
    b = {"csb": ["doc2", "doc3"]}
    merged = _merge_context(a, b)
    assert merged == {"csb": ["doc1", "doc2", "doc3"]}


def test_merge_context_does_not_mutate():
    """Merge context should not mutate the original dicts."""
    a = {"csb": ["doc1"]}
    b = {"csb": ["doc2"]}
    _merge_context(a, b)
    assert a == {"csb": ["doc1"]}


def test_last_writer_wins():
    """Last writer wins reducer returns the update value."""
    assert _last_writer_wins("old", "new") == "new"
    assert _last_writer_wins(0.5, 0.9) == 0.9
