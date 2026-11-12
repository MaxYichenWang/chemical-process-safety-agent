"""The pipeline event contract.

Defines node labels, worker mappings, event payloads, and next-node prediction.
The shared execution callback and SSE endpoint emit these events. Browser metadata
is served from this module so the frontend does not maintain a second label map."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, TypedDict, Union

from src.agentic_rag.config import VALIDATION_SCORE_THRESHOLD

NodeName = Literal[
    "CondenseNode",
    "PlanNode",
    "SynthesizeNode",
    "RevisionNode",
    "Supervisor",
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

SupervisorPhase = Literal["Plan", "Dispatch", "Synthesize"]

PipelinePhase = Literal["idle", "initializing", "running", "complete", "error"]

WorkerStatus = Literal["active", "done", "error"]


class Citation(TypedDict):
    collection: str
    source: str
    page: Union[int, str]
    chunk_id: str


class StepUpdate(TypedDict, total=False):
    sub_tasks: List[str]
    active_agent: str
    pending_agents: List[str]
    planned_agents: List[str]
    context_gathered: Dict[str, Any]
    draft_response: str
    final_response: str
    citations: List[Citation]
    validation_score: float
    critic_feedback: str
    retry_count: int


class WorkerError(TypedDict):
    node: str
    error: str
    impact: str


class WorkerProgress(TypedDict, total=False):
    node: NodeName
    status: WorkerStatus
    startedAtMs: int
    completedAtMs: int
    durationMs: int
    update: StepUpdate
    errors: List[WorkerError]


class PipelineState(TypedDict, total=False):
    query: str
    sub_tasks: List[str]
    active_agent: str
    pending_agents: List[str]
    context_gathered: Dict[str, Any]
    draft_response: str
    final_response: str
    citations: List[Citation]
    validation_score: float
    critic_feedback: str
    retry_count: int


class GraphReadyEvent(TypedDict):
    event: Literal["graph_ready"]
    elapsed_ms: int


class StepStartEvent(TypedDict):
    event: Literal["step_start"]
    node: NodeName


class StepCompleteEvent(TypedDict, total=False):
    event: Literal["step_complete"]
    node: NodeName
    elapsed_ms: int
    update: StepUpdate
    errors: List[WorkerError]


class PipelineCompleteEvent(TypedDict):
    event: Literal["pipeline_complete"]
    elapsed_ms: int
    state: PipelineState


class ErrorEvent(TypedDict, total=False):
    event: Literal["error"]
    message: str
    traceback: str


class SynthesisTokenEvent(TypedDict):
    event: Literal["synthesis_token"]
    token: str


BackendEvent = Union[
    GraphReadyEvent,
    StepStartEvent,
    StepCompleteEvent,
    PipelineCompleteEvent,
    ErrorEvent,
    SynthesisTokenEvent,
]


class CompletedStep(TypedDict, total=False):
    stepNumber: int
    node: NodeName
    elapsedMs: int
    stepDurationMs: int
    update: StepUpdate
    errors: List[WorkerError]


class ActiveStep(TypedDict):
    stepNumber: int
    node: NodeName
    startedAtMs: int


NODE_LABELS: Dict[NodeName, str] = {
    "CondenseNode": "Context Condenser",
    "PlanNode": "Supervisor (Plan)",
    "SynthesizeNode": "Supervisor (Synthesize)",
    "RevisionNode": "Supervisor (Revise)",
    "Supervisor": "Supervisor",
    "OshaRegsWorker": "OSHA Regulations Worker",
    "OshaEnfWorker": "OSHA Enforcement Worker",
    "IncidentWorker": "CSB Incidents Worker",
    "RecommendationsWorker": "CSB Recommendations Worker",
    "RootCauseWorker": "Root Cause Analysis Worker",
    "DataAnalystWorker": "Data Analyst Worker",
    "CriticNode": "Critic",
    "RetryNode": "Retry",
    "FinalizeNode": "Finalize",
}

NODE_DESCRIPTIONS: Dict[NodeName, str] = {
    "CondenseNode": "Resolving conversation context...",
    "PlanNode": "Planning worker dispatch...",
    "SynthesizeNode": "Synthesizing response...",
    "RevisionNode": "Revising draft from feedback...",
    "Supervisor": "Orchestrating pipeline...",
    "OshaRegsWorker": "Searching OSHA regulations...",
    "OshaEnfWorker": "Searching OSHA enforcement records...",
    "IncidentWorker": "Retrieving CSB incident reports...",
    "RecommendationsWorker": "Retrieving CSB recommendations...",
    "RootCauseWorker": "Performing root cause analysis...",
    "DataAnalystWorker": "Running statistical analysis...",
    "CriticNode": "Validating response accuracy...",
    "RetryNode": "Preparing retry...",
    "FinalizeNode": "Finalizing response...",
}

AGENT_KEY_TO_NODE: Dict[str, NodeName] = {
    "osha_regs": "OshaRegsWorker",
    "osha_enf": "OshaEnfWorker",
    "csb_incident": "IncidentWorker",
    "csb_recs": "RecommendationsWorker",
    "root_cause": "RootCauseWorker",
    "data_analyst": "DataAnalystWorker",
    "CRITIC": "CriticNode",
}

WORKER_NODES: List[NodeName] = [
    "OshaRegsWorker",
    "OshaEnfWorker",
    "IncidentWorker",
    "RecommendationsWorker",
    "RootCauseWorker",
    "DataAnalystWorker",
]


def detect_supervisor_phase(update: StepUpdate) -> SupervisorPhase:
    if update.get("sub_tasks"):
        return "Plan"
    if update.get("draft_response"):
        return "Synthesize"
    return "Dispatch"


def predict_next_node(node_name: NodeName, update: StepUpdate) -> Optional[NodeName]:
    """Optimistically predict the node that will start next.

    Lets the browser mark the next node active before its ``step_start``
    arrives, which is what makes the pipeline tree feel live rather than
    lagging visibly behind the pipeline (design doc §4.2).
    """
    if node_name == "CondenseNode":
        return "PlanNode"

    # PlanNode dispatches to workers in parallel -- no single next node.
    if node_name == "PlanNode":
        return None

    if node_name in WORKER_NODES:
        return "SynthesizeNode"

    if node_name == "SynthesizeNode":
        return "CriticNode"

    if node_name == "Supervisor":
        active = update.get("active_agent") or ""
        if active == "CRITIC":
            return "CriticNode"
        if active in ("FINISH", ""):
            return None
        return AGENT_KEY_TO_NODE.get(active)

    if node_name == "CriticNode":
        score = update.get("validation_score") or 0
        return "FinalizeNode" if score >= VALIDATION_SCORE_THRESHOLD else "RetryNode"

    if node_name == "RetryNode":
        return "RevisionNode"

    if node_name == "RevisionNode":
        return "CriticNode"

    return None
