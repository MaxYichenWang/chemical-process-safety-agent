"""Helpers for running RAGAS v0.4 collection metrics from the sync evaluator API.

Call metric ``ascore()`` directly to avoid experiment storage.
Failed metrics stay missing and are counted separately.

* a metric that raises for a row yields ``None`` for that row, and the row's
  other metrics are still scored;
* means are taken over successful rows only, alongside ``n_scored`` /
  ``n_failed`` so a reader can see the denominator;
* a metric that fails on *every* row reports ``<name>_error`` rather than a
  score, and if no metric scored at all the whole block is returned as
  ``{"error": ...}``;
* ``<name>_list`` stays aligned with the input rows (``None`` in failed slots)
  so cross-framework correlation can drop pairs rather than silently misalign.
"""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Optional, Sequence

from src.evaluation.judge_concurrency import judge_call_slot_async

logger = logging.getLogger(__name__)

# Per-metric error strings kept in the results JSON. Enough to diagnose, not
# enough to bloat a 238-row T2 artifact.
MAX_ERROR_SAMPLES = 3

Row = Dict[str, Any]
RowScores = Dict[str, Optional[float]]
ScoreRow = Callable[[Row], Awaitable[RowScores]]


def run_async(coro: Any) -> Any:
    """Run a coroutine from sync code, even when a loop is already running."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    with ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(lambda: asyncio.run(coro)).result()


# Backwards-compatible alias: the v0.4 migration exported this name.
run_async_experiment = run_async


def metric_value(result: Any) -> Optional[float]:
    """Extract a float from a RAGAS ``MetricResult``-like object.

    Returns ``None`` -- never ``0.0`` -- when there is no usable score, so a
    failure can't be mistaken for a bad score downstream.
    """
    value = getattr(result, "value", result)
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        logger.warning("RAGAS metric returned non-numeric value %r", value)
        return None
    if number != number:  # NaN
        return None
    return number


def _safe_repr(obj: Any, max_len: int = 300) -> str:
    """Truncated ``repr()`` safe to embed in a score-file failure reason."""
    try:
        text = repr(obj)
    except (
        Exception
    ):  # noqa: BLE001 - repr() itself must never be allowed to raise here
        text = f"<unreprable {type(obj).__name__}>"
    return text if len(text) <= max_len else text[:max_len] + "...(truncated)"


class MetricScorer:
    """Scores individual metrics, recording failures instead of raising.

    One instance per evaluator call; ``errors`` accumulates ``{metric: [msg]}``
    across all rows and is handed to :func:`aggregate_metric_records`.
    """

    def __init__(self) -> None:
        self.errors: Dict[str, List[str]] = {}

    async def score(self, name: str, metric: Any, **kwargs: Any) -> Optional[float]:
        try:
            async with judge_call_slot_async():
                raw = await metric.ascore(**kwargs)
        except (
            Exception
        ) as exc:  # noqa: BLE001 - deliberately broad, see module docstring
            message = f"{type(exc).__name__}: {exc}"
            bucket = self.errors.setdefault(name, [])
            if message not in bucket:
                bucket.append(message)
            logger.warning("RAGAS metric %s failed for one row: %s", name, message)
            return None

        value = metric_value(raw)
        if value is None:
            # ``metric.ascore()`` raised nothing but still produced no
            # usable value -- tests the "DeepSeek json_object mode can return
            # empty content -> ragas parses it to None -> silently dropped"
            # hypothesis. Previously this path recorded nothing at all, so a
            # judge failing on every row rendered as an unexplained ``0.0``
            # with no diagnostic trail.
            # Persisting the raw response turns "why did RAGAS fail here" from
            # a guess into evidence.
            message = f"empty/unparseable judge response: raw={_safe_repr(raw)}"
            bucket = self.errors.setdefault(name, [])
            if message not in bucket:
                bucket.append(message)
            logger.warning(
                "RAGAS metric %s returned no usable value for one row (raw=%r)",
                name,
                raw,
            )
        return value


def score_rows(
    score_row: ScoreRow,
    rows: Sequence[Row],
    *,
    max_concurrency: int = 4,
    timeout_s: Optional[float] = None,
) -> List[Optional[RowScores]]:
    """Run ``score_row`` for every row with bounded concurrency.

    A row that raises (or times out) contributes ``None`` and does not abort the
    batch. Returns a list positionally aligned with ``rows``.
    """
    if not rows:
        return []

    async def _runner() -> List[Optional[RowScores]]:
        semaphore = asyncio.Semaphore(max(1, max_concurrency))

        async def _one(index: int, row: Row) -> Optional[RowScores]:
            async with semaphore:
                try:
                    awaitable = score_row(row)
                    if timeout_s:
                        return await asyncio.wait_for(awaitable, timeout_s)
                    return await awaitable
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "RAGAS row %d failed: %s: %s", index, type(exc).__name__, exc
                    )
                    return None

        return await asyncio.gather(
            *(_one(index, row) for index, row in enumerate(rows))
        )

    return run_async(_runner())


def aggregate_metric_records(
    records: Sequence[Optional[RowScores]],
    metric_names: Iterable[str],
    *,
    errors: Optional[Dict[str, List[str]]] = None,
) -> Dict[str, Any]:
    """Aggregate per-row scores into the reporting shape.

    See the module docstring for the failure-handling contract this implements.
    """
    metric_names = list(metric_names)
    records = list(records or [])
    errors = errors or {}

    output: Dict[str, Any] = {"n_rows": len(records)}
    any_scored = False

    for name in metric_names:
        values = [
            record.get(name) if isinstance(record, dict) else None for record in records
        ]
        scored = [value for value in values if value is not None]
        output[f"{name}_list"] = values

        if scored:
            any_scored = True
            output[name] = sum(scored) / len(scored)
            output[f"{name}_n_scored"] = len(scored)
            if len(scored) != len(records):
                output[f"{name}_n_failed"] = len(records) - len(scored)
        else:
            output[f"{name}_error"] = _failure_reason(name, errors, len(records))

    if errors:
        output["errors"] = {
            name: messages[:MAX_ERROR_SAMPLES] for name, messages in errors.items()
        }

    if not any_scored:
        return {
            "error": (
                "no RAGAS metric produced a score for any row "
                f"({len(records)} row(s) attempted)"
            ),
            "n_rows": len(records),
            "errors": output.get("errors", {}),
        }

    return output


def _failure_reason(name: str, errors: Dict[str, List[str]], n_rows: int) -> str:
    messages = errors.get(name)
    if messages:
        return f"failed on all {n_rows} row(s): {messages[0]}"
    if n_rows == 0:
        return "no rows to score"
    return f"no score returned for any of {n_rows} row(s)"
