"""Metric evaluators and end-to-end pipeline evaluation."""

from .domain_evaluator import DomainEvaluator
from .faithfulness_evaluator import FaithfulnessEvaluator
from .orchestration_evaluator import OrchestrationEvaluator
from .retrieval_evaluator import RetrievalEvaluator

__all__ = [
    "DomainEvaluator",
    "EndToEndEvaluator",
    "FaithfulnessEvaluator",
    "OrchestrationEvaluator",
    "RetrievalEvaluator",
    "should_retry",
]


def __getattr__(name):
    if name in {"EndToEndEvaluator", "should_retry"}:
        from .end_to_end_evaluator import EndToEndEvaluator, should_retry

        return {"EndToEndEvaluator": EndToEndEvaluator, "should_retry": should_retry}[
            name
        ]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
