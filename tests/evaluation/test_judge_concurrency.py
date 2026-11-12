"""Tests for the global judge-call concurrency ceiling.

``EVAL_JUDGE_MAX_CONCURRENT`` bounds the *product* of evaluator-level,
DeepEval-worker-level, and intra-entry-metric concurrency, which each look
locally bounded but stack multiplicatively (see judge_concurrency.py's module
docstring). These tests verify the ceiling is actually enforced -- for both
the sync (``judge_call_slot``, used by DeepEval's ``metric.measure()``) and
async (``judge_call_slot_async``, used by RAGAS's ``metric.ascore()``) call
sites -- not just that the config value parses.
"""

import asyncio
import importlib
import threading
import time
from concurrent.futures import ThreadPoolExecutor


def _reload_judge_concurrency(monkeypatch, ceiling=None):
    if ceiling is None:
        monkeypatch.delenv("EVAL_JUDGE_MAX_CONCURRENT", raising=False)
    else:
        monkeypatch.setenv("EVAL_JUDGE_MAX_CONCURRENT", str(ceiling))

    from src.evaluation import config as eval_config

    importlib.reload(eval_config)

    from src.evaluation import judge_concurrency

    return importlib.reload(judge_concurrency)


def test_default_ceiling_is_16_when_unset(monkeypatch):
    jc = _reload_judge_concurrency(monkeypatch)
    from src.evaluation import config as eval_config

    assert eval_config.EVAL_JUDGE_MAX_CONCURRENT == 16
    assert jc._judge_semaphore._value == 16


def test_ceiling_respects_env_var_override(monkeypatch):
    jc = _reload_judge_concurrency(monkeypatch, ceiling=3)
    from src.evaluation import config as eval_config

    assert eval_config.EVAL_JUDGE_MAX_CONCURRENT == 3
    assert jc._judge_semaphore._value == 3


def test_ceiling_clamps_to_at_least_one(monkeypatch):
    jc = _reload_judge_concurrency(monkeypatch, ceiling=0)
    assert jc._judge_semaphore._value == 1


def test_judge_call_slot_bounds_concurrent_sync_callers(monkeypatch):
    """Spin up more sync workers than the ceiling and confirm peak concurrent
    holders of judge_call_slot() never exceeds the configured limit."""
    jc = _reload_judge_concurrency(monkeypatch, ceiling=2)

    lock = threading.Lock()
    state = {"current": 0, "peak": 0}

    def worker():
        with jc.judge_call_slot():
            with lock:
                state["current"] += 1
                state["peak"] = max(state["peak"], state["current"])
            time.sleep(0.05)
            with lock:
                state["current"] -= 1

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(lambda _: worker(), range(8)))

    assert state["peak"] == 2, f"expected peak concurrency of exactly 2, got {state['peak']}"


def test_judge_call_slot_async_bounds_concurrent_callers(monkeypatch):
    """Same guarantee via the async entry point used by ragas_experiments."""
    jc = _reload_judge_concurrency(monkeypatch, ceiling=2)

    state = {"current": 0, "peak": 0}
    lock = asyncio.Lock()

    async def worker():
        async with jc.judge_call_slot_async():
            async with lock:
                state["current"] += 1
                state["peak"] = max(state["peak"], state["current"])
            await asyncio.sleep(0.05)
            async with lock:
                state["current"] -= 1

    async def run_all():
        await asyncio.gather(*(worker() for _ in range(8)))

    asyncio.run(run_all())

    assert state["peak"] == 2, f"expected peak concurrency of exactly 2, got {state['peak']}"


def test_ragas_metric_scorer_still_scores_correctly_through_the_semaphore(monkeypatch):
    """Regression check: wrapping MetricScorer.score()'s ascore() call in the
    global semaphore must not change its return value or error handling."""
    _reload_judge_concurrency(monkeypatch, ceiling=4)
    # Re-import after reload so ragas_experiments binds the reloaded semaphore
    # module-level object (it imported judge_call_slot_async by reference at
    # its own import time, which happened once at test-collection time).
    import src.evaluation.ragas_experiments as ragas_experiments
    importlib.reload(ragas_experiments)

    class _OkMetric:
        async def ascore(self, **kwargs):
            return 0.75

    scorer = ragas_experiments.MetricScorer()
    result = asyncio.run(scorer.score("faithfulness", _OkMetric()))
    assert result == 0.75
    assert "faithfulness" not in scorer.errors
