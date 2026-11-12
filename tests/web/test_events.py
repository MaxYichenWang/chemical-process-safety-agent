"""Tests for pipeline event types, labels, and next-node prediction."""

from src.web.events import (
    AGENT_KEY_TO_NODE,
    NODE_DESCRIPTIONS,
    NODE_LABELS,
    WORKER_NODES,
    detect_supervisor_phase,
    predict_next_node,
)


def test_node_labels_has_a_label_for_every_expected_node():
    expected = [
        "Supervisor",
        "OshaRegsWorker",
        "OshaEnfWorker",
        "IncidentWorker",
        "RecommendationsWorker",
        "RootCauseWorker",
        "CriticNode",
        "RetryNode",
        "FinalizeNode",
    ]
    for n in expected:
        assert isinstance(NODE_LABELS.get(n), str) and NODE_LABELS[n]


def test_node_descriptions_has_a_description_for_every_node_in_labels():
    for key in NODE_LABELS:
        assert isinstance(NODE_DESCRIPTIONS.get(key), str) and NODE_DESCRIPTIONS[key]


def test_agent_key_to_node_maps_known_agent_keys():
    assert AGENT_KEY_TO_NODE["osha_regs"] == "OshaRegsWorker"
    assert AGENT_KEY_TO_NODE["osha_enf"] == "OshaEnfWorker"
    assert AGENT_KEY_TO_NODE["csb_incident"] == "IncidentWorker"
    assert AGENT_KEY_TO_NODE["csb_recs"] == "RecommendationsWorker"
    assert AGENT_KEY_TO_NODE["root_cause"] == "RootCauseWorker"
    assert AGENT_KEY_TO_NODE["CRITIC"] == "CriticNode"


def test_worker_nodes_contains_exactly_the_six_worker_nodes():
    assert len(WORKER_NODES) == 6
    for n in (
        "OshaRegsWorker",
        "OshaEnfWorker",
        "IncidentWorker",
        "RecommendationsWorker",
        "RootCauseWorker",
        "DataAnalystWorker",
    ):
        assert n in WORKER_NODES


class TestDetectSupervisorPhase:
    def test_plan_when_sub_tasks_present(self):
        assert detect_supervisor_phase({"sub_tasks": ["task1", "task2"]}) == "Plan"

    def test_synthesize_when_draft_response_present(self):
        assert detect_supervisor_phase({"draft_response": "Some draft text"}) == "Synthesize"

    def test_dispatch_otherwise(self):
        assert detect_supervisor_phase({}) == "Dispatch"
        assert detect_supervisor_phase({"active_agent": "osha_regs"}) == "Dispatch"

    def test_prefers_plan_over_synthesize_when_both_present(self):
        assert detect_supervisor_phase({"sub_tasks": ["t"], "draft_response": "draft"}) == "Plan"


class TestPredictNextNodeFromSupervisor:
    def test_returns_critic_node_when_active_agent_is_critic(self):
        assert predict_next_node("Supervisor", {"active_agent": "CRITIC"}) == "CriticNode"

    def test_returns_none_when_active_agent_is_finish(self):
        assert predict_next_node("Supervisor", {"active_agent": "FINISH"}) is None

    def test_returns_none_when_active_agent_is_empty(self):
        assert predict_next_node("Supervisor", {"active_agent": ""}) is None

    def test_resolves_known_agent_keys(self):
        assert predict_next_node("Supervisor", {"active_agent": "osha_regs"}) == "OshaRegsWorker"
        assert predict_next_node("Supervisor", {"active_agent": "root_cause"}) == "RootCauseWorker"

    def test_returns_none_for_unknown_agent_key(self):
        assert predict_next_node("Supervisor", {"active_agent": "unknown_key"}) is None


class TestPredictNextNodeFromWorkers:
    def test_returns_synthesize_node_after_every_worker(self):
        for worker in WORKER_NODES:
            assert predict_next_node(worker, {}) == "SynthesizeNode"


class TestPredictNextNodeFromCritic:
    def test_returns_finalize_node_when_score_at_or_above_threshold(self):
        assert predict_next_node("CriticNode", {"validation_score": 0.8}) == "FinalizeNode"
        assert predict_next_node("CriticNode", {"validation_score": 1.0}) == "FinalizeNode"

    def test_returns_retry_node_when_score_below_threshold(self):
        assert predict_next_node("CriticNode", {"validation_score": 0.79}) == "RetryNode"
        assert predict_next_node("CriticNode", {}) == "RetryNode"


def test_predict_next_node_from_retry_node_returns_revision_node():
    assert predict_next_node("RetryNode", {}) == "RevisionNode"


def test_predict_next_node_from_finalize_node_returns_none():
    assert predict_next_node("FinalizeNode", {}) is None
