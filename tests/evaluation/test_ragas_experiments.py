"""Tests for src.evaluation.ragas_experiments, focused on the instrumentation
of the RAGAS "silent None" path (empty/unparseable judge response)."""

import asyncio
import math

from src.evaluation.ragas_experiments import MetricScorer, aggregate_metric_records


class _RaisingMetric:
    async def ascore(self, **kwargs):
        raise ValueError("boom")


class _EmptyContentMetric:
    """Simulates DeepSeek's json_object mode returning empty content: ragas's
    ascore() completes without raising, but produces a MetricResult whose
    value is None (or a bare None, depending on ragas version)."""

    def __init__(self, raw=None):
        self._raw = raw

    async def ascore(self, **kwargs):
        return self._raw


class _FakeMetricResult:
    def __init__(self, value, reason="parsed empty response"):
        self.value = value
        self.reason = reason

    def __repr__(self):
        return f"FakeMetricResult(value={self.value!r}, reason={self.reason!r})"


class _OkMetric:
    def __init__(self, value):
        self._value = value

    async def ascore(self, **kwargs):
        return self._value


def test_score_records_exception_message_in_errors():
    scorer = MetricScorer()
    result = asyncio.run(scorer.score("faithfulness", _RaisingMetric()))
    assert result is None
    assert "ValueError: boom" in scorer.errors["faithfulness"][0]


def test_score_records_diagnostic_for_bare_none_with_no_exception():
    """metric.ascore() returns bare None (no exception) -> must be
    recorded, not silently dropped."""
    scorer = MetricScorer()
    result = asyncio.run(scorer.score("faithfulness", _EmptyContentMetric(raw=None)))
    assert result is None
    assert scorer.errors["faithfulness"], "empty-content path must populate errors"
    assert "empty/unparseable judge response" in scorer.errors["faithfulness"][0]


def test_score_records_raw_response_repr_for_metric_result_with_none_value():
    """metric.ascore() returns a MetricResult-like object whose .value is
    None -- the raw response repr must be captured for root-cause diagnosis."""
    scorer = MetricScorer()
    fake_result = _FakeMetricResult(value=None)
    result = asyncio.run(scorer.score("faithfulness", _EmptyContentMetric(raw=fake_result)))
    assert result is None
    message = scorer.errors["faithfulness"][0]
    assert "empty/unparseable judge response" in message
    assert "FakeMetricResult" in message  # raw repr embedded, not just "None"


def test_score_does_not_record_anything_on_success():
    scorer = MetricScorer()
    result = asyncio.run(scorer.score("faithfulness", _OkMetric(_FakeMetricResult(0.9))))
    assert result == 0.9
    assert "faithfulness" not in scorer.errors


def test_aggregate_surfaces_empty_content_diagnostic_when_metric_fails_every_row():
    """A judge that returns empty content on every row for one metric, but
    scores another metric fine, must not report a silent 0.0 for the failed
    metric -- its per-metric error message should carry the diagnostic."""
    scorer = MetricScorer()
    faithfulness_scores = [
        asyncio.run(scorer.score("faithfulness", _EmptyContentMetric(raw=None)))
        for _ in range(3)
    ]
    correctness_scores = [
        asyncio.run(scorer.score("answer_correctness", _OkMetric(_FakeMetricResult(0.8))))
        for _ in range(3)
    ]
    records = [
        {"faithfulness": f, "answer_correctness": c}
        for f, c in zip(faithfulness_scores, correctness_scores)
    ]
    out = aggregate_metric_records(records, ["faithfulness", "answer_correctness"], errors=scorer.errors)

    assert out["faithfulness_error"].startswith("failed on all 3 row(s)")
    assert "empty/unparseable judge response" in out["faithfulness_error"]
    assert math.isclose(out["answer_correctness"], 0.8)


def test_aggregate_reports_whole_block_error_when_every_metric_fails_empty_content():
    """When *no* metric scores anything, the whole-block error must still be
    diagnosable via the per-metric ``errors`` it carries forward."""
    scorer = MetricScorer()
    records = []
    for _ in range(3):
        val = asyncio.run(scorer.score("faithfulness", _EmptyContentMetric(raw=None)))
        records.append({"faithfulness": val})

    out = aggregate_metric_records(records, ["faithfulness"], errors=scorer.errors)
    assert "error" in out
    assert out["errors"]["faithfulness"]
    assert "empty/unparseable judge response" in out["errors"]["faithfulness"][0]
