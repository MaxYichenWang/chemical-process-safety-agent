"""Tests for DataAnalystWorker graph integration.

Verifies that data_analyst is properly wired into the supervisor routing
and LangGraph topology.
"""

from unittest.mock import patch


from src.agentic_rag.graph import build_graph
from src.agentic_rag.supervisor import VALID_AGENTS, AGENT_TO_NODE


# ---------------------------------------------------------------------------
# Supervisor routing
# ---------------------------------------------------------------------------


class TestSupervisorRouting:

    def test_valid_agents_includes_data_analyst(self):
        """data_analyst is in VALID_AGENTS."""
        assert "data_analyst" in VALID_AGENTS

    def test_agent_to_node_maps_data_analyst(self):
        """AGENT_TO_NODE maps data_analyst to DataAnalystWorker."""
        assert AGENT_TO_NODE["data_analyst"] == "DataAnalystWorker"

    def test_existing_routes_unchanged(self):
        """All agent-to-node mappings are correct."""
        expected = {
            "osha_regs": "OshaRegsWorker",
            "osha_enf": "OshaEnfWorker",
            "csb_incident": "IncidentWorker",
            "csb_recs": "RecommendationsWorker",
            "root_cause": "RootCauseWorker",
            "data_analyst": "DataAnalystWorker",
        }
        for agent, expected_node in expected.items():
            assert AGENT_TO_NODE[agent] == expected_node


# ---------------------------------------------------------------------------
# Graph topology
# ---------------------------------------------------------------------------


class TestGraphTopology:

    @patch("src.agentic_rag.workers.get_chroma_tools", return_value={})
    def test_build_graph_includes_data_analyst_node(self, mock_tools):
        """DataAnalystWorker is a node in the compiled graph."""
        graph = build_graph()
        node_names = list(graph.get_graph().nodes.keys())
        assert "DataAnalystWorker" in node_names

    @patch("src.agentic_rag.workers.get_chroma_tools", return_value={})
    def test_data_analyst_worker_in_workers_list(self, mock_tools):
        """DataAnalystWorker exists as a node."""
        graph = build_graph()
        graph_repr = graph.get_graph()
        da_node = graph_repr.nodes.get("DataAnalystWorker")
        assert da_node is not None

    @patch("src.agentic_rag.workers.get_chroma_tools", return_value={})
    def test_build_graph_compiles_without_error(self, mock_tools):
        """Graph compiles successfully with all worker nodes."""
        graph = build_graph()
        assert graph is not None

    @patch("src.agentic_rag.workers.get_chroma_tools", return_value={})
    def test_all_nodes_present(self, mock_tools):
        """All expected nodes are present in the new topology."""
        graph = build_graph()
        node_names = list(graph.get_graph().nodes.keys())
        expected_nodes = [
            "PlanNode",
            "SynthesizeNode",
            "RevisionNode",
            "OshaRegsWorker",
            "OshaEnfWorker",
            "IncidentWorker",
            "RecommendationsWorker",
            "RootCauseWorker",
            "DataAnalystWorker",
            "CriticNode",
            "RetryNode",
            "FinalizeNode",
        ]
        for node in expected_nodes:
            assert node in node_names, f"Missing node: {node}"
