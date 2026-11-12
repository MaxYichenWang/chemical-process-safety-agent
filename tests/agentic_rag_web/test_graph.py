from src.agentic_rag_web.graph import WORKER_NODES, build_web_graph


def test_build_web_graph_compiles_all_expected_nodes():
    app = build_web_graph()
    expected_nodes = [
        "CondenseNode",
        "PlanNode",
        "SynthesizeNode",
        "RevisionNode",
        *WORKER_NODES,
        "CriticNode",
        "RetryNode",
        "FinalizeNode",
    ]
    for node in expected_nodes:
        assert node in app.get_graph().nodes, f"Missing node: {node}"


def test_build_web_graph_matches_paper_graph_node_set():
    """The web graph's node set must equal the paper graph's plus CondenseNode (Phase 6)."""
    from src.agentic_rag.graph import build_graph

    paper_nodes = set(build_graph().get_graph().nodes)
    web_nodes = set(build_web_graph().get_graph().nodes)
    assert web_nodes == paper_nodes | {"CondenseNode"}
