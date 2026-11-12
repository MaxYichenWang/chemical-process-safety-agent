"""Mock-judge tests for the RAGAS scoring layer."""

from src.evaluation.scoring.ragas_layer import DEFAULT_METRICS, score_ragas


class _StubMetric:
    """Stand-in RAGAS metric: returns a fixed value, or raises/empties on demand."""

    def __init__(self, value=0.8, empty_content=False, raises=False):
        self._value = value
        self._empty_content = empty_content
        self._raises = raises

    async def ascore(self, **kwargs):
        if self._raises:
            raise RuntimeError("judge unreachable")
        if self._empty_content:
            return None  # simulates DeepSeek json_object empty-content -> None
        return self._value


def _fixture():
    dataset = [
        {"query": "q0", "expected_answer": "a0", "ground_truth_chunks": ["c0"]},
        {"query": "q1", "expected_answer": "a1", "ground_truth_chunks": ["c1"]},
        {"query": "q2", "expected_answer": "a2", "ground_truth_chunks": []},
    ]
    results = [
        {"final_response": "r0", "context_gathered": [{"text": "ctx0"}]},
        {"final_response": "r1", "context_gathered": [{"text": "ctx1"}]},
        {"error": "pipeline blew up", "final_response": "", "context_gathered": []},
    ]
    return dataset, results


def test_score_ragas_skips_pipeline_error_stubs():
    dataset, results = _fixture()
    factory = lambda: {"faithfulness": _StubMetric(0.9), "answer_correctness": _StubMetric(0.7)}
    out = score_ragas(dataset, results, metrics_factory=factory)
    assert set(out["per_entry"].keys()) == {"0", "1"}
    assert out["n_scored"] == 2
    assert out["n_failed"] == 0


def test_score_ragas_all_metrics_score_successfully():
    dataset, results = _fixture()
    factory = lambda: {"faithfulness": _StubMetric(0.9), "answer_correctness": _StubMetric(0.7)}
    out = score_ragas(dataset, results, metrics_factory=factory)
    assert out["per_entry"]["0"]["faithfulness"] == 0.9
    assert out["per_entry"]["0"]["answer_correctness"] == 0.7
    assert out["failures"] == []


def test_score_ragas_empty_content_recorded_as_entry_level_failure():
    """A judge returning empty content (None, no exception) must surface
    as an entry-level failure with a diagnosable reason, not a silent 0.0."""
    dataset, results = _fixture()
    factory = lambda: {
        "faithfulness": _StubMetric(empty_content=True),
        "answer_correctness": _StubMetric(0.7),
    }
    out = score_ragas(dataset, results, metrics_factory=factory)

    assert out["per_entry"]["0"]["faithfulness"] is None
    assert out["per_entry"]["0"]["answer_correctness"] == 0.7
    # still "scored" overall since answer_correctness came through
    assert out["n_scored"] == 2

    faithfulness_failures = [f for f in out["failures"] if f["metric"] == "faithfulness"]
    assert len(faithfulness_failures) == 2  # entries "0" and "1"
    assert all("empty/unparseable judge response" in f["reason"] for f in faithfulness_failures)
    assert {f["entry_id"] for f in faithfulness_failures} == {"0", "1"}


def test_score_ragas_metric_raising_is_isolated_per_metric():
    dataset, results = _fixture()
    factory = lambda: {
        "faithfulness": _StubMetric(raises=True),
        "answer_correctness": _StubMetric(0.7),
    }
    out = score_ragas(dataset, results, metrics_factory=factory)
    assert out["per_entry"]["0"]["faithfulness"] is None
    assert out["per_entry"]["0"]["answer_correctness"] == 0.7
    reasons = [f["reason"] for f in out["failures"] if f["metric"] == "faithfulness"]
    assert any("judge unreachable" in r for r in reasons)


def test_score_ragas_no_ok_entries_returns_empty_shape():
    dataset = [{"query": "q0", "expected_answer": "a0", "ground_truth_chunks": []}]
    results = [{"error": "boom", "final_response": "", "context_gathered": []}]
    factory = lambda: {"faithfulness": _StubMetric(), "answer_correctness": _StubMetric()}
    out = score_ragas(dataset, results, metrics_factory=factory)
    assert out == {"per_entry": {}, "aggregates": {"n_rows": 0}, "n_scored": 0, "n_failed": 0, "failures": []}


def test_default_metrics_matches_documented_set():
    assert set(DEFAULT_METRICS) == {"faithfulness", "answer_correctness"}


def test_score_ragas_rejects_unknown_metric_names():
    dataset, results = _fixture()
    factory = lambda: {"topic_adherence": _StubMetric()}
    try:
        score_ragas(dataset, results, metrics_factory=factory, metric_names=["topic_adherence"])
        raise AssertionError("expected ValueError for unmapped metric name")
    except ValueError as exc:
        assert "topic_adherence" in str(exc)


def test_build_row_truncation():
    from src.evaluation.scoring.ragas_layer import _build_row

    long_text = "B" * 30_000
    entry = {"query": "q", "expected_answer": "a", "ground_truth_chunks": []}
    result = {"final_response": "r", "context_gathered": [{"text": long_text}]}

    row = _build_row(entry, result)
    assert len(row["retrieved_contexts"][0]) == 20_000
