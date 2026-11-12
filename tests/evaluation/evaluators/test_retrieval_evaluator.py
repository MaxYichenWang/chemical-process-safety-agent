from tests.evaluation.ragas_stubs import install_fake_ragas
from src.evaluation.evaluators.retrieval_evaluator import (
    hit_rate_at_k,
    mean_reciprocal_rank,
    ndcg_at_k,
    RetrievalEvaluator
)

class TestRetrievalEvaluator:
    def test_hit_rate_perfect(self):
        assert hit_rate_at_k(["a", "b", "c"], ["a"], k=5) == 1.0

    def test_hit_rate_miss(self):
        assert hit_rate_at_k(["a", "b", "c"], ["d"], k=5) == 0.0

    def test_hit_rate_at_k_cutoff(self):
        assert hit_rate_at_k(["a", "b", "c"], ["c"], k=2) == 0.0
        assert hit_rate_at_k(["a", "b", "c"], ["c"], k=3) == 1.0

    def test_mrr_first_position(self):
        assert mean_reciprocal_rank(["a", "b", "c"], ["a"]) == 1.0

    def test_mrr_third_position(self):
        assert mean_reciprocal_rank(["a", "b", "c"], ["c"]) == 1/3

    def test_mrr_miss(self):
        assert mean_reciprocal_rank(["a", "b", "c"], ["d"]) == 0.0

    def test_mrr_empty_ground_truth(self):
        assert mean_reciprocal_rank(["a", "b", "c"], []) == 0.0

    def test_ndcg_perfect_ranking(self):
        assert ndcg_at_k(["a", "b"], ["a", "b"], k=5) == 1.0

    def test_ndcg_imperfect(self):
        score = ndcg_at_k(["a", "b"], ["b"], k=5)
        # dcg = 1/log2(3) approx 0.6309
        # idcg = 1/log2(2) = 1.0
        assert 0.63 < score < 0.64

    def test_ndcg_empty_ground_truth(self):
        assert ndcg_at_k(["a", "b"], [], k=5) == 0.0

    def test_evaluate_single(self):
        evaluator = RetrievalEvaluator()
        result = evaluator.evaluate_single("query", ["a", "b"], ["b"])
        assert result["hit_rate_at_5"] == 1.0
        assert result["mrr"] == 0.5
        assert 0.63 < result["ndcg_at_5"] < 0.64

    def test_evaluate_single_empty_gt(self):
        evaluator = RetrievalEvaluator()
        result = evaluator.evaluate_single("query", ["a", "b"], [])
        assert result["hit_rate_at_5"] == 0.0
        assert result["mrr"] == 0.0
        assert result["ndcg_at_5"] == 0.0

    def test_evaluate_per_worker(self):
        evaluator = RetrievalEvaluator()
        dataset = [
            {"query": "q1", "ground_truth_chunks": ["a"], "expected_agents": ["worker_A"]},
            {"query": "q2", "ground_truth_chunks": ["b"], "expected_agents": ["worker_B"]}
        ]
        results = [
            {"context_gathered": [{"chunk_id": "a"}]},
            {"context_gathered": [{"chunk_id": "c"}]} # miss
        ]
        worker_metrics = evaluator.evaluate_per_worker(dataset, results)

        assert "worker_A" in worker_metrics
        assert "worker_B" in worker_metrics

        assert worker_metrics["worker_A"]["hit_rate_at_5"]["mean"] == 1.0
        assert worker_metrics["worker_B"]["hit_rate_at_5"]["mean"] == 0.0

    def test_evaluate_batch_custom_with_empty(self):
        evaluator = RetrievalEvaluator()
        dataset = [{"query": "q", "ground_truth_chunks": ["a"]}]
        results = [{"context_gathered": [{"chunk_id": "a"}]}]
        metrics = evaluator.evaluate_batch_custom(dataset, results)
        assert metrics["hit_rate_at_5"]["mean"] == 1.0
        assert metrics["hit_rate_at_5"]["ci_lower"] == 1.0

    def test_evaluate_batch_ragas_uses_v04_collection_metrics(self, monkeypatch):
        calls = install_fake_ragas(
            monkeypatch,
            ["ContextPrecisionWithReference", "ContextRecall"],
            score=0.6,
        )

        result = RetrievalEvaluator().evaluate_batch_ragas(
            [{"query": "q", "expected_answer": "a"}],
            [{"final_response": "a", "context_gathered": [{"text": "ctx"}]}],
        )

        assert result["context_precision"] == 0.6
        assert result["context_recall"] == 0.6
        assert result["context_precision_list"] == [0.6]

        # Neither metric accepts ``response`` in ragas 0.4; passing it raises.
        by_metric = {name: kwargs for name, kwargs in calls}
        assert set(by_metric["ContextPrecisionWithReference"]) == {
            "user_input", "reference", "retrieved_contexts",
        }
        assert set(by_metric["ContextRecall"]) == {
            "user_input", "retrieved_contexts", "reference",
        }

    def test_evaluate_batch_ragas_reports_error_when_all_rows_fail(self, monkeypatch):
        def _fail(name, kwargs):
            raise RuntimeError("judge unavailable")

        install_fake_ragas(
            monkeypatch,
            ["ContextPrecisionWithReference", "ContextRecall"],
            on_call=_fail,
        )

        result = RetrievalEvaluator().evaluate_batch_ragas(
            [{"query": "q", "expected_answer": "a"}],
            [{"final_response": "a", "context_gathered": [{"text": "ctx"}]}],
        )
        assert "error" in result
        assert "context_precision" not in result

    def test_evaluate_isolates_a_failing_subsection(self, monkeypatch):
        """A ragas/deepeval blowup must not take the custom metrics with it."""
        evaluator = RetrievalEvaluator()
        monkeypatch.setattr(
            evaluator, "evaluate_batch_ragas",
            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")),
        )
        monkeypatch.setattr(evaluator, "evaluate_batch_deepeval", lambda *a, **k: {})

        out = evaluator.evaluate(
            [{"query": "q", "ground_truth_chunks": ["a"]}],
            [{"context_gathered": [{"chunk_id": "a"}]}],
        )

        assert out["custom_metrics"]["hit_rate_at_5"]["mean"] == 1.0
        assert "boom" in out["ragas_metrics"]["error"]
