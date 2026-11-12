"""Mock-judge tests for the DeepEval scoring layer."""

import pytest

from src.evaluation.scoring.cache import ScoringCache
from src.evaluation.scoring.deepeval_layer import METRICS, score_deepeval


def _fixture():
    dataset = [
        {"query": "q0", "expected_answer": "a0"},
        {"query": "q1", "expected_answer": "a1"},
        {"query": "q2", "expected_answer": "a2"},
    ]
    results = [
        {"final_response": "r0", "context_gathered": []},
        {"final_response": "r1", "context_gathered": []},
        {"error": "pipeline blew up", "final_response": "", "context_gathered": []},
    ]
    return dataset, results


def _mock_measure(entry, result, judge_model):
    # Deterministic stand-in for a real judge call -- keyed off the query so
    # tests can assert on specific values without a network call.
    n = int(entry["query"][1:])
    return {"faithfulness": 0.9 - 0.1 * n, "hallucination": 0.05, "correctness": 0.8}


def test_score_deepeval_skips_pipeline_error_stubs():
    dataset, results = _fixture()
    out = score_deepeval(dataset, results, measure_fn=_mock_measure, judge_id="mock")
    assert set(out["per_entry"].keys()) == {"0", "1"}  # entry 2 has an "error" stub
    assert out["n_scored"] == 2
    assert out["n_failed"] == 0


def test_score_deepeval_aggregates_have_ci_and_n_scored():
    dataset, results = _fixture()
    out = score_deepeval(dataset, results, measure_fn=_mock_measure, judge_id="mock")
    agg = out["aggregates"]["faithfulness"]
    assert agg["n_scored"] == 2
    assert 0.7 <= agg["mean"] <= 0.9
    assert "ci_lower" in agg and "ci_upper" in agg


def test_score_deepeval_records_failure_when_measure_fn_raises():
    dataset, results = _fixture()

    def _raising_measure(entry, result, judge_model):
        raise RuntimeError("judge API down")

    out = score_deepeval(dataset, results, measure_fn=_raising_measure, judge_id="mock")
    assert out["n_scored"] == 0
    assert out["n_failed"] == 2
    reasons = [f["reason"] for f in out["failures"]]
    assert any("judge API down" in r for r in reasons)


def test_score_deepeval_is_cache_aware_and_resumable(tmp_path):
    dataset, results = _fixture()
    cache = ScoringCache(tmp_path / "cache.json")
    calls = {"count": 0}

    def _counting_measure(entry, result, judge_model):
        calls["count"] += 1
        return _mock_measure(entry, result, judge_model)

    out1 = score_deepeval(
        dataset, results, measure_fn=_counting_measure, judge_id="mock",
        generation_sha="gensha", cache=cache,
    )
    assert calls["count"] == 2  # 2 non-error entries measured

    # "Resume": a fresh score_deepeval call over the same generation/judge
    # must reuse the cache instead of re-measuring.
    out2 = score_deepeval(
        dataset, results, measure_fn=_counting_measure, judge_id="mock",
        generation_sha="gensha", cache=cache,
    )
    assert calls["count"] == 2  # unchanged -- nothing re-measured
    assert out1["per_entry"] == out2["per_entry"]


def test_score_deepeval_cache_keyed_by_generation_and_judge(tmp_path):
    dataset, results = _fixture()
    cache = ScoringCache(tmp_path / "cache.json")
    calls = {"count": 0}

    def _counting_measure(entry, result, judge_model):
        calls["count"] += 1
        return _mock_measure(entry, result, judge_model)

    score_deepeval(dataset, results, measure_fn=_counting_measure, judge_id="qwen",
                   generation_sha="gensha-A", cache=cache)
    assert calls["count"] == 2

    # Different judge_id -> cache miss -> re-measured.
    score_deepeval(dataset, results, measure_fn=_counting_measure, judge_id="deepseek",
                   generation_sha="gensha-A", cache=cache)
    assert calls["count"] == 4

    # Different generation -> cache miss -> re-measured.
    score_deepeval(dataset, results, measure_fn=_counting_measure, judge_id="qwen",
                   generation_sha="gensha-B", cache=cache)
    assert calls["count"] == 6


def test_metrics_constant_matches_documented_set():
    assert set(METRICS) == {"faithfulness", "hallucination", "correctness"}


# ---------------------------------------------------------------------------
# score_deepeval must surface real judge_calls/judge_cost_usd from
# judge_factory.judge_call_stats(), not a hardcoded 0/0.0 -- and report None
# (not 0) when the judge model isn't trackable (bare OpenAI judge string).
# ---------------------------------------------------------------------------


class _FakeTrackableJudge:
    """Stands in for judge_factory's _LangChainDeepEvalModel adapter."""

    def __init__(self, call_log):
        self.call_log = call_log


def test_score_deepeval_reports_none_judge_cost_when_judge_model_untrackable():
    dataset, results = _fixture()
    out = score_deepeval(
        dataset, results, measure_fn=_mock_measure, judge_id="mock",
        judge_model="gpt-5.4-mini",  # bare string -- no call_log attribute
    )
    assert out["judge_calls"] is None
    assert out["judge_cost_usd"] is None


def test_score_deepeval_reports_real_judge_calls_and_cost_when_trackable():
    dataset, results = _fixture()
    fake_judge = _FakeTrackableJudge(call_log=[
        {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120, "cost_usd": 0.001},
        {"prompt_tokens": 200, "completion_tokens": 40, "total_tokens": 240, "cost_usd": 0.002},
    ])
    out = score_deepeval(
        dataset, results, measure_fn=_mock_measure, judge_id="mock",
        judge_model=fake_judge,
    )
    assert out["judge_calls"] == 2
    assert abs(out["judge_cost_usd"] - 0.003) < 1e-9


def test_context_doc_truncation(monkeypatch):
    """default_measure constructs an LLMTestCase with retrieved_contexts."""
    pytest.importorskip("deepeval")
    import deepeval.metrics
    from src.evaluation.scoring import deepeval_layer

    captured = {}

    class StubMetric:

        def __init__(self, **kwargs):
            self.score = 0.9

        def measure(self, test_case):
            captured["test_case"] = test_case
            return self.score

    monkeypatch.setattr(deepeval.metrics, "FaithfulnessMetric", StubMetric)
    monkeypatch.setattr(deepeval.metrics, "HallucinationMetric", StubMetric)
    monkeypatch.setattr(deepeval.metrics, "GEval", StubMetric)

    long_text = "A" * 30_000
    entry = {"query": "q", "expected_answer": "e"}
    result = {"final_response": "r", "context_gathered": [{"text": long_text}]}

    deepeval_layer.default_measure(entry, result, judge_model=None)
    tc = captured.get("test_case")
    assert tc is not None
    assert len(tc.retrieval_context[0]) == 20_000


def test_default_measure_constructs_llm_test_case_with_context():
    pytest.importorskip("deepeval")
    from unittest.mock import MagicMock, patch
    from src.evaluation.scoring.deepeval_layer import default_measure

    mock_faith_inst = MagicMock(score=0.9)
    mock_hallu_inst = MagicMock(score=0.01)
    mock_geval_inst = MagicMock(score=0.85)

    with (
        patch("deepeval.metrics.FaithfulnessMetric", return_value=mock_faith_inst),
        patch("deepeval.metrics.HallucinationMetric", return_value=mock_hallu_inst),
        patch("deepeval.metrics.GEval", return_value=mock_geval_inst),
    ):
        entry = {"query": "q", "expected_answer": "ans"}
        result = {
            "final_response": "resp",
            "context_gathered": [{"text": "ctx1"}, {"text": "ctx2"}],
        }

        out = default_measure(entry, result, judge_model="mock")

        assert out == {
            "faithfulness": 0.9,
            "hallucination": 0.01,
            "correctness": 0.85,
        }

        # Verify LLMTestCase passed to HallucinationMetric.measure had context set
        tc_hallu = mock_hallu_inst.measure.call_args[0][0]
        assert tc_hallu.context == ["ctx1", "ctx2"]
        assert tc_hallu.retrieval_context == ["ctx1", "ctx2"]
