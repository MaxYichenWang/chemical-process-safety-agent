"""Web-server state extension.

``WebAgentState`` extends the paper's ``AgentState`` with the two fields
multi-turn conversation needs (Phase 6, ``CondenseNode``). It is defined now
so ``graph.py``'s wiring is written once rather than reshaped later; both
new fields are unused until Phase 6 wires ``CondenseNode`` in.
"""

from __future__ import annotations

from typing import List

from src.agentic_rag.state import AgentState


class WebAgentState(AgentState):
    """``AgentState`` plus the fields ``CondenseNode`` needs.

    - ``raw_query``: the practitioner's literal turn-N question, before any
      condensation. ``query`` (inherited from ``AgentState``) becomes the
      *condensed*, standalone question that ``PlanNode`` actually plans
      against -- for turn 1 (or any single-turn run) the two are identical.
    - ``conversation_history``: prior turns in this browser session, oldest
      first. Empty for every single-turn run, which is what keeps
      ``build_web_graph()`` byte-identical to ``build_graph()`` for the G2
      equivalence gate.
    """

    raw_query: str
    conversation_history: List[dict]
