from src.evaluation.evaluators.faithfulness_evaluator import FaithfulnessEvaluator
from tests.evaluation.ragas_stubs import install_fake_ragas
from unittest.mock import patch
import sys

class TestFaithfulnessEvaluator:
    def test_evaluate_citation_accuracy_perfect(self):
        evaluator = FaithfulnessEvaluator()
        results = [
            {
                "final_response": "Per OSHA standard 1910.119(j) and 1910.147...",
                "context_gathered": [
                    {"text": "Under 1910.119(j), employers must..."},
                    {"text": "Section 1910.147 applies to lockout/tagout..."}
                ]
            }
        ]
        acc = evaluator.evaluate_citation_accuracy(results)
        assert acc["mean"] == 1.0

    def test_evaluate_citation_accuracy_miss(self):
        evaluator = FaithfulnessEvaluator()
        results = [
            {
                "final_response": "OSHA 1910.1200 requires hazard communication.",
                "context_gathered": [
                    {"text": "1910.119 covers process safety management."}
                ]
            }
        ]
        acc = evaluator.evaluate_citation_accuracy(results)
        assert acc["mean"] == 0.0

    def test_evaluate_citation_accuracy_partial(self):
        evaluator = FaithfulnessEvaluator()
        results = [
            {
                "final_response": "See 1910.119 and 1910.120",
                "context_gathered": [{"text": "1910.119 details PSM."}]
            }
        ]
        acc = evaluator.evaluate_citation_accuracy(results)
        assert acc["mean"] == 0.5

    def test_evaluate_citation_accuracy_no_citations(self):
        evaluator = FaithfulnessEvaluator()
        results = [
            {
                "final_response": "General safety is important.",
                "context_gathered": [{"text": "1910.119 details PSM."}]
            }
        ]
        acc = evaluator.evaluate_citation_accuracy(results)
        assert acc["mean"] == 0.0

    def test_evaluate_cross_framework(self):
        evaluator = FaithfulnessEvaluator()
        ragas = {"faithfulness_list": [0.8, 0.9, 0.4]}
        deepeval = {"faithfulness_list": [0.7, 0.85, 0.5]}

        corr = evaluator.evaluate_cross_framework(ragas, deepeval)
        assert "pearson" in corr
        assert "spearman" in corr
        assert corr["pearson"] > 0.8 # should be highly correlated

    def test_evaluate_cross_framework_empty(self):
        evaluator = FaithfulnessEvaluator()
        corr = evaluator.evaluate_cross_framework({}, {})
        assert corr["pearson"] == 0.0

    @patch("src.evaluation.evaluators.faithfulness_evaluator.FaithfulnessEvaluator.evaluate_ragas")
    def test_evaluate_full(self, mock_evaluate_ragas):
        evaluator = FaithfulnessEvaluator()
        mock_evaluate_ragas.return_value = {"faithfulness": 0.8}

        # Override evaluate_deepeval for test determinism
        def mock_evaluate_deepeval(d, r):
            return {"faithfulness": {"mean": 0.9}}

        evaluator.evaluate_deepeval = mock_evaluate_deepeval

        dataset = [{"query": "What is PSM?", "ground_truth_chunks": ["chunk1"]}]
        results = [{"final_response": "It's process safety.", "context_gathered": [{"text": "PSM context"}]}]

        res = evaluator.evaluate(dataset, results)
        assert "citation_accuracy" in res
        assert "ragas_metrics" in res
        assert "deepeval_metrics" in res
        assert "cross_framework" in res

    def test_evaluate_ragas_uses_v04_collection_metrics(self, monkeypatch):
        calls = install_fake_ragas(
            monkeypatch,
            ["Faithfulness", "AnswerCorrectness", "FactualCorrectness"],
            score=0.75,
        )

        result = FaithfulnessEvaluator().evaluate_ragas(
            [{"query": "q", "expected_answer": "a"}],
            [{"final_response": "a", "context_gathered": [{"text": "ctx"}]}],
        )

        assert result["faithfulness"] == 0.75
        assert result["answer_correctness"] == 0.75
        assert result["factual_correctness"] == 0.75
        assert result["faithfulness_list"] == [0.75]
        assert result["n_rows"] == 1

        # Guards the exact ragas 0.4 ascore signatures. The migration originally
        # passed ``response=`` to metrics that do not accept it.
        by_metric = {name: kwargs for name, kwargs in calls}
        assert set(by_metric["Faithfulness"]) == {
            "user_input", "response", "retrieved_contexts",
        }
        assert set(by_metric["AnswerCorrectness"]) == {
            "user_input", "response", "reference",
        }
        assert set(by_metric["FactualCorrectness"]) == {"response", "reference"}

    def test_evaluate_ragas_scores_rows_individually(self, monkeypatch):
        """A row function must receive one row, never the whole list.

        The v0.4 migration called ``ExperimentWrapper.__call__(rows)``, handing
        the entire list to a function expecting a single row; every real run
        died with ``'list' object has no attribute 'user_input'`` while the
        mocked test passed.
        """
        seen = []

        calls = install_fake_ragas(
            monkeypatch, ["Faithfulness", "AnswerCorrectness"], score=1.0,
            on_call=lambda name, kwargs: seen.append(kwargs.get("user_input")),
        )
        # The evaluators must not reach for the experiment API any more.
        assert not hasattr(sys.modules["ragas"], "experiment")

        dataset = [{"query": f"q{i}", "expected_answer": "a"} for i in range(3)]
        results = [
            {"final_response": "a", "context_gathered": [{"text": "ctx"}]}
            for _ in range(3)
        ]
        out = FaithfulnessEvaluator().evaluate_ragas(dataset, results)

        assert out["n_rows"] == 3
        assert list(dict.fromkeys(s for s in seen if s is not None)) == ["q0", "q1", "q2"]
        assert all(isinstance(kwargs.get("user_input", ""), str) for _, kwargs in calls)

    def test_evaluate_ragas_isolates_per_row_failures(self, monkeypatch):
        """One bad row must not zero out or abort the rest of the batch."""
        state = {"n": 0}

        def _flaky(name, kwargs):
            state["n"] += 1
            if state["n"] == 2:
                raise RuntimeError("judge 400")

        install_fake_ragas(
            monkeypatch, ["Faithfulness", "AnswerCorrectness"], score=0.5,
            on_call=_flaky,
        )

        dataset = [{"query": f"q{i}", "expected_answer": "a"} for i in range(2)]
        results = [
            {"final_response": "a", "context_gathered": [{"text": "ctx"}]}
            for _ in range(2)
        ]
        out = FaithfulnessEvaluator().evaluate_ragas(dataset, results)

        assert out["faithfulness"] == 0.5
        assert out["faithfulness_list"] == [0.5, 0.5]
        # The 2nd call is answer_correctness on row 0 -> that metric loses one row.
        assert out["answer_correctness_n_failed"] == 1
        assert None in out["answer_correctness_list"]
        assert "judge 400" in out["errors"]["answer_correctness"][0]

    def test_evaluate_ragas_reports_error_instead_of_zero_when_all_rows_fail(
        self, monkeypatch
    ):
        """Total judge failure must not be published as a 0.0 score.

        This is the exact regression that produced "RAGAS faithfulness 0.0 on
        every query" across two DeepSeek runs via ``df.fillna(0)``.
        """
        def _always_fail(name, kwargs):
            raise RuntimeError("400 temperature not supported")

        install_fake_ragas(
            monkeypatch, ["Faithfulness", "AnswerCorrectness"], score=0.9,
            on_call=_always_fail,
        )

        out = FaithfulnessEvaluator().evaluate_ragas(
            [{"query": "q", "expected_answer": "a"}],
            [{"final_response": "a", "context_gathered": [{"text": "ctx"}]}],
        )

        assert "error" in out
        assert out.get("faithfulness") is None
        assert "temperature not supported" in str(out["errors"])

    def test_cross_framework_drops_failed_ragas_rows(self):
        evaluator = FaithfulnessEvaluator()
        corr = evaluator.evaluate_cross_framework(
            {"faithfulness_list": [0.8, None, 0.4]},
            {"faithfulness_list": [0.7, 0.9, 0.5]},
        )
        assert corr["n_pairs"] == 2

    def test_citation_accuracy_credits_context_metadata(self):
        """Section numbers that live only in chunk metadata must count."""
        evaluator = FaithfulnessEvaluator()
        results = [
            {
                "final_response": "See 1910.1001(j).",
                "context_gathered": [
                    {
                        "text": "Communication of hazards to employees.",
                        "source": "1910.1001 §1910.1001(j)",
                        "chunk_id": "osha_reg_1910.1001(j)",
                    }
                ],
            }
        ]
        assert evaluator.evaluate_citation_accuracy(results)["mean"] == 1.0
