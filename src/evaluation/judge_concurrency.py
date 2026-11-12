"""Process-wide ceiling on concurrent judge (LLM-as-judge) calls.

The scoring code has three independent, *locally* bounded concurrency knobs
that stack multiplicatively instead of adding:

* multiple evaluators (retrieval / faithfulness / orchestration / ...) may run
  side-by-side under ``end_to_end_evaluator.py --max-concurrent``;
* each DeepEval batch fans entries out across ``EVAL_MAX_WORKERS`` /
  ``_DEEPEVAL_PARALLELISM`` threads;
* some entries fan out *again* into one thread per intra-entry metric
  (``FaithfulnessMetric`` / ``HallucinationMetric`` / ``GEval`` correctness
  all measured concurrently in ``faithfulness_evaluator.py``).

Each level looks bounded in isolation, but nothing bounds the product, so the
judge provider can see far more simultaneous requests than any single knob
suggests (worst case at defaults: 5 evaluators x 4 workers x 3 metrics = 60).

This module is one process-wide semaphore that every judge call site --
DeepEval's ``metric.measure()`` (sync, legacy evaluators/*.py and
``scoring/deepeval_layer.py``) and RAGAS's ``metric.ascore()`` (async,
``ragas_experiments.py``, shared by both legacy and staged-CLI code) -- acquires
immediately before invoking the judge model, regardless of which evaluator,
thread, or event loop initiated the call. Default ceiling: 16 concurrent
judge calls (``EVAL_JUDGE_MAX_CONCURRENT``).

A plain ``threading.Semaphore`` is used (not ``asyncio.Semaphore``) because it
is the one primitive that is safe to share across both worker threads and
whatever event loop(s) ``ragas_experiments.run_async`` happens to spin up --
an ``asyncio.Semaphore`` is bound to a single event loop and breaks if awaited
from a different one, which happens here since each evaluator/thread can spin
up its own loop via ``asyncio.run``.
"""

from __future__ import annotations

import asyncio
import threading
from contextlib import contextmanager
from typing import Iterator

from src.evaluation.config import EVAL_JUDGE_MAX_CONCURRENT

_judge_semaphore = threading.Semaphore(max(1, EVAL_JUDGE_MAX_CONCURRENT))


@contextmanager
def judge_call_slot() -> Iterator[None]:
    """Block until a global judge-call slot is free, then hold it for one call.

    Use around a single synchronous judge invocation, e.g.::

        with judge_call_slot():
            metric.measure(test_case)
    """
    _judge_semaphore.acquire()
    try:
        yield
    finally:
        _judge_semaphore.release()


class _AsyncJudgeSlot:
    """Async-friendly acquire/release for the same process-wide semaphore.

    Tries a non-blocking acquire first, so the common case (a slot is free)
    completes synchronously with no thread hand-off and no extra suspension
    point -- this keeps coroutine scheduling order predictable when the
    semaphore isn't actually contended, which is the normal case outside of
    a true concurrency pile-up. Only when the semaphore is exhausted does it
    fall back to a blocking ``acquire()`` in the default executor, so the
    event loop is never stalled while genuinely waiting for a slot.
    """

    async def __aenter__(self) -> "_AsyncJudgeSlot":
        if _judge_semaphore.acquire(blocking=False):
            return self
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _judge_semaphore.acquire)
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        _judge_semaphore.release()


def judge_call_slot_async() -> _AsyncJudgeSlot:
    """Async counterpart of :func:`judge_call_slot`, for ``await metric.ascore()``.

    Use as::

        async with judge_call_slot_async():
            raw = await metric.ascore(**kwargs)
    """
    return _AsyncJudgeSlot()
