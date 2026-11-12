import pytest
from src.evaluation.evaluators.orchestration_evaluator import OrchestrationEvaluator
from tests.evaluation.ragas_stubs import install_fake_ragas

class TestOrchestrationEvaluator:
    def test_tool_call_accuracy(self):
        evaluator = OrchestrationEvaluator()
        dataset = [
            {"expected_agents": ["agentA", "agentB"]},
            {"expected_agents": ["agentC"]}
        ]
        results = [
            {"agents_called": ["agentA", "agentB"]}, # Exact match
            {"agents_called": ["agentC", "agentD"]}  # Partial match, jaccard = 1/2
        ]

        acc = evaluator.evaluate_tool_call_accuracy(dataset, results)
        assert acc["exact_match_rate"] == 0.5
        assert acc["jaccard"]["mean"] == 0.75 # (1.0 + 0.5) / 2

    def test_evaluate_critic_distribution(self):
        evaluator = OrchestrationEvaluator()
        results = [
            {"validation_score": 0.1},
            {"validation_score": 0.5},
            {"validation_score": 0.9}
        ]
        dist = evaluator.evaluate_critic_distribution(results)
        assert dist["mean"] == 0.5
        assert dist["min"] == 0.1
        assert dist["max"] == 0.9
        assert dist["histogram"]["0.0-0.2"] == 1
        assert dist["histogram"]["0.4-0.6"] == 1
        assert dist["histogram"]["0.8-1.0"] == 1

    def test_evaluate_retry_rate(self):
        evaluator = OrchestrationEvaluator()
        results = [
            {"retry_count": 0},
            {"retry_count": 1},
            {"retry_count": 3}
        ]
        rates = evaluator.evaluate_retry_rate(results)
        assert rates["retry_rate"] == 2/3
        assert rates["mean_retries"] == 4/3

    def test_evaluate_pipeline_error_rate(self):
        evaluator = OrchestrationEvaluator()
        results = [
            {"error": "boom"},
            {"retry_count": 0},
            {"error": "timeout"},
        ]
        out = evaluator.evaluate_pipeline_error_rate(results)
        assert out["error_count"] == 2
        assert out["total"] == 3
        assert out["error_rate"] == pytest.approx(2 / 3)

    def test_evaluate_cost(self):
        evaluator = OrchestrationEvaluator()
        results = [
            {
                "llm_calls": [
                    {
                        "estimated_cost_usd": 0.001,
                        "prompt_tokens": 100,
                        "completion_tokens": 50,
                        "node_name": "Supervisor",
                        "call_purpose": "planning",
                        "latency_s": 0.5,
                    },
                    {
                        "estimated_cost_usd": 0.002,
                        "prompt_tokens": 200,
                        "completion_tokens": 100,
                        "node_name": "Supervisor",
                        "call_purpose": "synthesis",
                        "latency_s": 1.5,
                    },
                ]
            },
            {
                "llm_calls": [
                    {
                        "estimated_cost_usd": 0.004,
                        "prompt_tokens": 300,
                        "completion_tokens": 150,
                        "node_name": "Critic",
                        "call_purpose": "validation",
                        "latency_s": 1.0,
                    }
                ]
            },
        ]

        out = evaluator.evaluate_cost(results)
        assert out["total_usd"] == pytest.approx(0.007)
        assert out["mean_per_query_usd"] == pytest.approx(0.0035)
        assert out["p95_per_query_usd"] == pytest.approx(0.004)
        assert out["mean_calls_per_query"] == pytest.approx(1.5)
        assert out["total_prompt_tokens"] == 600
        assert out["total_completion_tokens"] == 300
        assert out["breakdown_by_node"]["Supervisor"]["calls"] == 2
        assert out["breakdown_by_purpose"]["planning"]["calls"] == 1

    def test_evaluate_plan_accuracy_with_alias(self):
        evaluator = OrchestrationEvaluator()
        dataset = [
            {"expected_agents": ["csb_recommendations", "osha_enf"]},
            {"expected_agents": ["data_analyst"]},
        ]
        results = [
            {"planned_agents": ["csb_recs", "osha_enf"]},
            {"planned_agents": ["osha_enf"]},
        ]

        out = evaluator.evaluate_plan_accuracy(dataset, results)
        assert out["exact_match_rate"] == 0.5
        assert out["jaccard"]["mean"] == pytest.approx(0.5)
        assert out["n"] == 2

    def test_evaluate_fast_route_coverage(self):
        evaluator = OrchestrationEvaluator()
        dataset = [
            {"expected_agents": ["osha_enf"]},
            {"expected_agents": ["data_analyst"]},
        ]
        results = [
            {
                "planning_fast_routed": True,
                "planning_fast_route_patterns": ["enforcement/violation query"],
                "planned_agents": ["osha_enf"],
            },
            {
                "planning_fast_routed": False,
                "planned_agents": ["osha_enf"],
            },
        ]

        out = evaluator.evaluate_fast_route_coverage(dataset, results)
        assert out["fast_route_rate"] == 0.5
        assert out["fast_route_count"] == 1
        assert out["llm_plan_count"] == 1
        assert out["fast_route_plan_jaccard_mean"] == 1.0
        assert out["llm_plan_jaccard_mean"] == 0.0
        assert out["fast_route_pattern_counts"]["enforcement/violation query"] == 1

    def test_ragas_goal_accuracy_reports_not_migrated(self):
        out = OrchestrationEvaluator().evaluate_goal_accuracy_ragas([], [])
        assert "skipped" in out
        assert "not migrated" in out["skipped"]

    def test_evaluate_topic_adherence_uses_v04_collection_metric(self, monkeypatch):
        calls = install_fake_ragas(monkeypatch, ["AnswerRelevancy"], score=0.8)

        out = OrchestrationEvaluator().evaluate_topic_adherence(
            [{"query": "q"}],
            [{"final_response": "a"}],
        )

        assert out["topic_adherence"] == 0.8
        assert out["topic_adherence_list"] == [0.8]
        assert out["metric"] == "ragas.AnswerRelevancy"
        assert calls == [("AnswerRelevancy", {"user_input": "q", "response": "a"})]

    def test_topic_adherence_setup_failure_does_not_escape(self, monkeypatch):
        """A metric constructor raising must return a dict, not propagate.

        On 2026-07-31 ``AnswerRelevancy(...)`` was constructed outside the
        try/except; the resulting TypeError escaped ``evaluate()`` and
        ``end_to_end_evaluator``'s catch-all replaced the whole orchestration
        section with a single error string, losing pipeline error rate, routing
        accuracy and cost from both the JSON and the report.
        """
        install_fake_ragas(monkeypatch, ["AnswerRelevancy"])
        import src.evaluation.judge_factory as judge_factory

        def _boom(**kwargs):
            raise TypeError("missing 1 required positional argument: 'embeddings'")

        monkeypatch.setattr(judge_factory, "make_ragas_embeddings", _boom)

        out = OrchestrationEvaluator().evaluate_topic_adherence(
            [{"query": "q"}], [{"final_response": "a"}]
        )
        assert "error" in out
        assert "embeddings" in out["error"]

    def test_evaluate_keeps_deterministic_metrics_when_a_judge_metric_fails(
        self, monkeypatch
    ):
        evaluator = OrchestrationEvaluator()
        monkeypatch.setattr(
            evaluator, "evaluate_topic_adherence",
            lambda *a, **k: (_ for _ in ()).throw(TypeError("ragas exploded")),
        )
        monkeypatch.setattr(evaluator, "evaluate_goal_accuracy_deepeval", lambda *a, **k: {})

        dataset = [{"query": "q", "expected_agents": ["osha_enf"]}]
        results = [{"agents_called": ["osha_enf"], "llm_calls": [], "final_response": "x"}]
        out = evaluator.evaluate(dataset, results)

        assert out["pipeline_error_rate"]["error_count"] == 0
        assert out["tool_call_accuracy"]["exact_match_rate"] == 1.0
        assert "ragas exploded" in out["ragas_topic_adherence"]["error"]
