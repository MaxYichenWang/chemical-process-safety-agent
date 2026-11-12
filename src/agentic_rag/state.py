import operator
from typing import Annotated, Any, Dict, List, TypedDict


def _merge_context(current: dict, update: dict) -> dict:
    """Merge context_gathered dicts by extending list values."""
    merged = dict(current)
    for k, v in update.items():
        if k in merged and isinstance(merged[k], list) and isinstance(v, list):
            merged[k] = merged[k] + v
        else:
            merged[k] = v
    return merged


def _last_writer_wins(current, update):
    """Use the latest value (for scalar fields written by multiple nodes)."""
    return update


class AgentState(TypedDict):
    """
    The state of the agentic RAG system passed between LangGraph nodes.
    """

    query: str
    sub_tasks: List[str]
    active_agent: Annotated[str, _last_writer_wins]
    pending_agents: List[str]
    planned_agents: List[str]
    planning_fast_routed: Annotated[bool, _last_writer_wins]
    planning_fast_route_patterns: Annotated[List[str], _last_writer_wins]
    context_gathered: Annotated[Dict[str, List[Any]], _merge_context]
    draft_response: Annotated[str, _last_writer_wins]
    final_response: str
    citations: Annotated[List[Dict[str, str]], operator.add]
    validation_score: Annotated[float, _last_writer_wins]
    critic_feedback: Annotated[str, _last_writer_wins]
    retry_count: Annotated[int, _last_writer_wins]
    llm_calls: Annotated[List[Dict[str, Any]], operator.add]
    retrieval_details: Annotated[List[Dict[str, Any]], operator.add]
    worker_errors: Annotated[List[Dict[str, str]], operator.add]
