"""Shared RAGAS test doubles for the evaluator unit tests.

Deliberately does **not** provide ``ragas.experiment``. The evaluators used to
depend on that decorator and the old stubs faked it with a version that iterated
the row list -- which is precisely what the real ``ExperimentWrapper.__call__``
does *not* do. The mock passed, every real run failed. Omitting it here means a
reintroduced dependency fails loudly in tests.

Signature fidelity is covered separately by
``tests/evaluation/test_ragas_contract.py``, which introspects the real
installed ragas classes.
"""

from __future__ import annotations

import sys
import types
from typing import Any, Callable, List, Optional, Sequence, Tuple

Call = Tuple[str, dict]


def install_fake_ragas(
    monkeypatch,
    metric_names: Sequence[str],
    *,
    score: float = 1.0,
    on_call: Optional[Callable[[str, dict], Any]] = None,
) -> List[Call]:
    """Install stub ``ragas.metrics.collections`` metrics and a stub judge.

    ``on_call(metric_name, kwargs)`` runs before each score is returned; raise
    from it to simulate a judge failure, or return a float to override the score.
    Returns the accumulating list of ``(metric_class_name, ascore_kwargs)``.
    """
    calls: List[Call] = []

    class _Metric:
        def __init__(self, llm, embeddings=None, **kwargs):
            self.llm = llm
            self.embeddings = embeddings

        async def ascore(self, **kwargs):
            name = type(self).__name__
            calls.append((name, kwargs))
            override = on_call(name, kwargs) if on_call else None
            value = override if isinstance(override, (int, float)) else score
            return types.SimpleNamespace(value=value)

    ragas_mod = types.ModuleType("ragas")
    ragas_metrics_mod = types.ModuleType("ragas.metrics")
    collections_mod = types.ModuleType("ragas.metrics.collections")
    for metric_name in metric_names:
        setattr(collections_mod, metric_name, type(metric_name, (_Metric,), {}))

    monkeypatch.setitem(sys.modules, "ragas", ragas_mod)
    monkeypatch.setitem(sys.modules, "ragas.metrics", ragas_metrics_mod)
    monkeypatch.setitem(sys.modules, "ragas.metrics.collections", collections_mod)

    import src.evaluation.config as eval_config
    import src.evaluation.judge_factory as judge_factory

    monkeypatch.setattr(eval_config, "EVAL_ENABLE_RAGAS", True)
    monkeypatch.setattr(
        judge_factory, "make_ragas_judge_llm", lambda **kwargs: "judge", raising=False
    )
    monkeypatch.setattr(
        judge_factory, "make_ragas_embeddings", lambda **kwargs: "embeddings",
        raising=False,
    )
    return calls
