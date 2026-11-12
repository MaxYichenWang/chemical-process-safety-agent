import logging

from langgraph.graph import END, StateGraph

from src.agentic_rag.config import MAX_CRITIC_RETRIES, VALIDATION_SCORE_THRESHOLD
from src.agentic_rag.critic import critic_agent
from src.agentic_rag.state import AgentState
from src.agentic_rag.supervisor import (
    route_from_plan,
    supervisor_plan,
    supervisor_revise,
    supervisor_synthesize,
)
from src.agentic_rag.workers import (
    data_analyst_agent,
    incident_retriever_agent,
    osha_enf_agent,
    osha_regs_agent,
    recommendation_specialist_agent,
    root_cause_analyst_agent,
)

logger = logging.getLogger(__name__)


def _route_from_critic(state: AgentState) -> str:
    """
    Route based on validation score and retry budget.
    Returns a node name -- never mutates state.
    """
    score = state.get("validation_score", 0.0)
    retry_count = state.get("retry_count", 0)

    if score >= VALIDATION_SCORE_THRESHOLD:
        return "FinalizeNode"

    if retry_count >= MAX_CRITIC_RETRIES:
        logger.warning(
            "Max retries (%d) reached with score %s. "
            "Returning best-effort response with disclaimer.",
            MAX_CRITIC_RETRIES,
            score,
        )
        return "FinalizeNode"

    logger.info(
        "Critique failed with score %s, retry %d/%d.",
        score,
        retry_count + 1,
        MAX_CRITIC_RETRIES,
    )
    return "RetryNode"


def _retry_node(state: AgentState) -> dict:
    """Increment retry counter."""
    return {
        "retry_count": state.get("retry_count", 0) + 1,
    }


def _finalize_node(state: AgentState) -> dict:
    """
    Move the draft to final_response. If the validation score is below threshold,
    prepend a confidence disclaimer.
    """
    draft = state.get("draft_response", "")
    score = state.get("validation_score", 0.0)

    if score < VALIDATION_SCORE_THRESHOLD:
        disclaimer = (
            "**Note:** This response could not be fully validated against the retrieved "
            f"source documents (confidence score: {score:.2f}/{1.0}). Some claims may "
            "lack sufficient evidentiary support. Please verify critical details against "
            "primary sources before making safety decisions.\n\n---\n\n"
        )
        draft = disclaimer + draft

    return {"final_response": draft}


def build_graph() -> StateGraph:
    """Constructs and compiles the multi-agent StateGraph with parallel worker fan-out."""
    workflow = StateGraph(AgentState)

    workflow.add_node("PlanNode", supervisor_plan)
    workflow.add_node("OshaRegsWorker", osha_regs_agent)
    workflow.add_node("OshaEnfWorker", osha_enf_agent)
    workflow.add_node("IncidentWorker", incident_retriever_agent)
    workflow.add_node("RecommendationsWorker", recommendation_specialist_agent)
    workflow.add_node("RootCauseWorker", root_cause_analyst_agent)
    workflow.add_node("DataAnalystWorker", data_analyst_agent)
    workflow.add_node("SynthesizeNode", supervisor_synthesize)
    workflow.add_node("RevisionNode", supervisor_revise)
    workflow.add_node("CriticNode", critic_agent)
    workflow.add_node("RetryNode", _retry_node)
    workflow.add_node("FinalizeNode", _finalize_node)

    workflow.set_entry_point("PlanNode")

    # PlanNode → conditional edge that fans out to workers via Send
    worker_nodes = [
        "OshaRegsWorker",
        "OshaEnfWorker",
        "IncidentWorker",
        "RecommendationsWorker",
        "RootCauseWorker",
        "DataAnalystWorker",
    ]
    workflow.add_conditional_edges(
        "PlanNode",
        route_from_plan,
        worker_nodes,
    )

    # Workers return to SynthesizeNode (LangGraph auto-collects Send results)
    for worker in worker_nodes:
        workflow.add_edge(worker, "SynthesizeNode")

    workflow.add_edge("SynthesizeNode", "CriticNode")

    workflow.add_conditional_edges(
        "CriticNode",
        _route_from_critic,
        {
            "FinalizeNode": "FinalizeNode",
            "RetryNode": "RetryNode",
        },
    )

    # Retry routes to RevisionNode for targeted draft patching (not full re-synthesis)
    workflow.add_edge("RetryNode", "RevisionNode")
    workflow.add_edge("RevisionNode", "CriticNode")
    workflow.add_edge("FinalizeNode", END)

    return workflow.compile()
