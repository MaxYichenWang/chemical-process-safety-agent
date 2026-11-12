"""Gate G2 — single-turn equivalence.

``build_web_graph()`` (empty conversation history, i.e. every single-turn
run today) must produce the same routing and answers as ``build_graph()``
across a sample of eval queries spanning different worker paths. A failure
here is a correctness bug in the web graph's wiring, never a threshold to
loosen (design doc §3.2, checklist Phase 0 note).
"""

from __future__ import annotations

from src.agentic_rag.graph import build_graph
from src.agentic_rag_web.graph import build_web_graph
from tests.agentic_rag_web.conftest import G2_SAMPLE_QUERIES, run_query_sync


def test_g2_equivalence_over_sample_queries(patched_llm_boundary):
    """`build_web_graph()` must match `build_graph()` for every sample query."""
    paper_graph = build_graph()
    web_graph = build_web_graph()

    assert len(G2_SAMPLE_QUERIES) >= 10, "G2 requires a sample of >= 10 queries"

    mismatches = []
    seen_workers = set()
    for query in G2_SAMPLE_QUERIES:
        paper_state = run_query_sync(query, paper_graph)
        web_state = run_query_sync(query, web_graph)

        seen_workers.update(paper_state.get("planned_agents", []))

        if paper_state.get("planned_agents") != web_state.get("planned_agents"):
            mismatches.append(
                f"{query!r}: planned_agents differ "
                f"(paper={paper_state.get('planned_agents')}, web={web_state.get('planned_agents')})"
            )
            continue

        # validation_score must be within noise (the fake critic always
        # returns a fixed score, so this should be exact here; a small
        # tolerance keeps the test meaningful if the fake ever varies).
        score_delta = abs(
            paper_state.get("validation_score", 0.0) - web_state.get("validation_score", 0.0)
        )
        if score_delta > 1e-6:
            mismatches.append(f"{query!r}: validation_score differs by {score_delta}")

        if paper_state.get("citations") != web_state.get("citations"):
            mismatches.append(f"{query!r}: citations differ")

    assert not mismatches, "G2 equivalence failures:\n" + "\n".join(mismatches)

    # Sanity: the sample must actually exercise every worker at least once,
    # otherwise this test could pass trivially without covering all paths.
    expected_workers = {
        "osha_regs",
        "osha_enf",
        "csb_incident",
        "csb_recs",
        "root_cause",
        "data_analyst",
    }
    missing = expected_workers - seen_workers
    assert not missing, f"sample queries never dispatched: {missing}"
