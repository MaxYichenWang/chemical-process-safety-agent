import json
from src.evaluation.reporting.report_generator import generate_report, generate_json_results, format_metric, format_ci, safe_get

def test_format_metric():
    assert format_metric(0.852) == "85.2%"
    assert format_metric(0.852, is_percentage=False) == "0.852"

def test_format_ci():
    assert format_ci(0.81, 0.89) == "[81.0% - 89.0%]"
    assert format_ci(0.81, 0.89, is_percentage=False) == "[0.810 - 0.890]"

def test_safe_get():
    d = {"a": {"b": {"c": 1.0}}}
    assert safe_get(d, "a", "b", "c") == 1.0
    assert safe_get(d, "a", "x", "c", default=0.5) == 0.5
    assert safe_get(d, "x", default=0.0) == 0.0

def test_generate_report():
    mock_results = {
        "retrieval": {
            "custom_metrics": {
                "hit_rate_at_5": {"mean": 0.8, "ci_lower": 0.7, "ci_upper": 0.9},
                "ndcg_at_5": {"mean": 0.75, "ci_lower": 0.65, "ci_upper": 0.85}
            }
        },
        "faithfulness": {
            "deepeval_metrics": {
                "faithfulness": {"mean": 0.9},
                "hallucination": {"mean": 0.05}
            },
            "citation_accuracy": {"mean": 0.85}
        },
        "orchestration": {
            "pipeline_error_rate": {"error_rate": 0.1, "error_count": 1, "total": 10},
            "tool_call_accuracy": {"exact_match_rate": 0.9, "jaccard": {"mean": 0.85}},
            "plan_accuracy": {"exact_match_rate": 0.8, "jaccard": {"mean": 0.75}},
            "fast_route_coverage": {"fast_route_rate": 0.3, "fast_route_count": 3, "llm_plan_count": 7},
            "cost": {"mean_per_query_usd": 0.0123, "p95_per_query_usd": 0.02, "total_usd": 0.123},
            "retry_rate": {"retry_rate": 0.1}
        },
        "domain": {
            "incident_fidelity": {"mean": 0.95},
            "hedging_detection": {"adversarial_hedge_rate": 0.9, "standard_hedge_rate": 0.05}
        },
        "per_category": {
            "regulatory_factual": {
                "n_total": 12,
                "retrieval": {"custom_metrics": {"hit_rate_at_5": {"mean": 0.9}}},
                "faithfulness": {"citation_accuracy": {"mean": 0.8}},
                "orchestration": {
                    "tool_call_accuracy": {"exact_match_rate": 0.75},
                    "cost": {"mean_per_query_usd": 0.01}
                }
            }
        }
    }

    report_md = generate_report(mock_results, "test_run")
    assert "Evaluation Run: test_run" in report_md
    assert "80.0%" in report_md  # hit rate mean
    assert "[70.0% - 90.0%]" in report_md # hit rate CI
    assert "90.0%" in report_md # faithfulness mean
    assert "5.0%" in report_md # hallucination
    assert "Plan Exact Match" in report_md
    assert "Fast-Route Rate" in report_md
    assert "$0.0123" in report_md
    assert "Per-Category Summary" in report_md
    assert "regulatory_factual" in report_md

def test_generate_json_results(tmp_path):
    mock_data = {"test": "data", "nested": {"list": [1, 2, 3]}}
    # Monkeypatch EVAL_RESULTS_DIR in the module
    import src.evaluation.reporting.report_generator as rg
    old_dir = rg.EVAL_RESULTS_DIR
    rg.EVAL_RESULTS_DIR = tmp_path

    try:
        out_path = generate_json_results(mock_data, "test_json_run")
        assert out_path.exists()
        with open(out_path, "r") as f:
            data = json.load(f)
        assert data["test"] == "data"
        assert data["nested"]["list"] == [1, 2, 3]
    finally:
        rg.EVAL_RESULTS_DIR = old_dir


# ---------------------------------------------------------------------------
# Data-quality surfacing
#
# Failed metric sections render as 0.0% in every table below, which is
# indistinguishable from a real zero. That is how a broken RAGAS judge was
# published as "0.0% faithful on every query" and a crashed orchestration
# evaluator as 0% routing accuracy at $0.0000 cost. Section 0 makes those
# impossible to miss.
# ---------------------------------------------------------------------------

def test_collect_metric_issues_finds_errors_and_skips():
    from src.evaluation.reporting.report_generator import collect_metric_issues

    issues = collect_metric_issues({
        "orchestration": {"error": "AnswerRelevancy.__init__() missing 'embeddings'"},
        "faithfulness": {
            "ragas_metrics": {"skipped": "ragas disabled"},
            "citation_accuracy": {"mean": 0.5},
        },
    })
    by_path = {path: (kind, msg) for path, kind, msg in issues}
    assert by_path["orchestration"][0] == "error"
    assert by_path["faithfulness.ragas_metrics"][0] == "skipped"


def test_collect_metric_issues_ignores_per_category_duplicates():
    from src.evaluation.reporting.report_generator import collect_metric_issues

    issues = collect_metric_issues({
        "per_category": {"regulatory_factual": {"orchestration": {"error": "noise"}}},
    })
    assert issues == []


def test_report_flags_a_failed_metric_section():
    report_md = generate_report(
        {"orchestration": {"error": "AnswerRelevancy.__init__() missing 'embeddings'"}},
        "broken_run",
    )
    assert "## 0. Data Quality" in report_md
    assert "metric section(s) failed" in report_md
    assert "`orchestration`" in report_md


def test_report_says_so_when_everything_computed():
    report_md = generate_report({"retrieval": {"custom_metrics": {}}}, "clean_run")
    assert "All metric sections computed successfully." in report_md


# ---------------------------------------------------------------------------
# Denominators, no silent zeros for failed/partial sections, and promoting
# previously-computed-but-unrendered metrics.
# ---------------------------------------------------------------------------

from src.evaluation.reporting.report_generator import (  # noqa: E402
    collect_metric_issues,
    format_currency_safe,
    format_ci_safe,
    format_rate_with_n,
    safe_get_rate,
)


def test_format_rate_with_n_renders_denominator():
    # 132/205 = 0.6439024390243903, matching the flagship run's Hit@5 number.
    d = {"mean": 0.6439024390243903, "n": 205}
    assert format_rate_with_n(d) == "64.4% (132/205)"


def test_format_rate_with_n_falls_back_to_plain_percentage_without_n():
    assert format_rate_with_n({"mean": 0.5}) == "50.0%"


def test_format_rate_with_n_returns_na_for_error():
    assert format_rate_with_n({"error": "judge unreachable"}) == "N/A"


def test_format_rate_with_n_returns_na_for_skipped():
    assert format_rate_with_n({"skipped": "ragas disabled"}) == "N/A"


def test_format_rate_with_n_returns_na_when_value_missing():
    assert format_rate_with_n({}) == "N/A"


def test_format_rate_with_n_supports_custom_value_and_n_keys():
    d = {"exact_match_rate": 0.9018691588785047, "n": 214}
    assert format_rate_with_n(d, value_key="exact_match_rate") == "90.2% (193/214)"


def test_format_rate_with_n_as_fraction_false_shows_sample_size_not_fabricated_numerator():
    """Regression test: SQL correctness is a 0.4/0.6-weighted hybrid score
    per query (continuous, not binary), so mean=16/45 with n=9 (a *different*
    run) must never render as "(3/9)" -- round(mean * n) is not a real count
    when the underlying per-item scores aren't strictly 0.0/1.0."""
    d = {"mean": 0.3555555555555555, "n": 9}
    assert format_rate_with_n(d, as_fraction=False) == "35.6% (n=9)"
    assert "(3/9)" not in format_rate_with_n(d, as_fraction=False)


def test_report_does_not_fabricate_numerator_for_continuous_metrics():
    """NDCG, SQL correctness, numerical fidelity, and hybrid synthesis are all
    continuous/multi-valued per-item scores -- the report must show sample
    size (n=N), never an (x/N) fraction that implies a literal hit-count."""
    mock_results = {
        "retrieval": {"custom_metrics": {"ndcg_at_5": {"mean": 0.6, "n": 10}}},
        "analytical": {
            "sql_correctness": {"mean": 0.3555555555555555, "n": 9},
            "numerical_fidelity": {"mean": 0.504, "n": 13},
            "hybrid_synthesis": {"mean": 0.5, "n": 3},
        },
    }
    report_md = generate_report(mock_results, "continuous_metric_run")
    assert "(n=10)" in report_md  # NDCG
    assert "(n=9)" in report_md  # SQL correctness
    assert "(n=13)" in report_md  # numerical fidelity
    assert "(n=3)" in report_md  # hybrid synthesis
    assert "(6/10)" not in report_md
    assert "(3/9)" not in report_md
    assert "(7/13)" not in report_md
    assert "(2/3)" not in report_md


def test_format_ci_safe_returns_na_for_error():
    assert format_ci_safe({"error": "boom"}) == "N/A"


def test_format_ci_safe_renders_normally_when_present():
    assert format_ci_safe({"ci_lower": 0.7, "ci_upper": 0.9}) == "[70.0% - 90.0%]"


def test_format_currency_safe_returns_na_for_error():
    assert format_currency_safe({"error": "boom"}, "mean_per_query_usd") == "N/A"


def test_format_currency_safe_returns_na_when_key_missing():
    assert format_currency_safe({}, "mean_per_query_usd") == "N/A"


def test_safe_get_rate_returns_none_when_path_dead_ends_on_error_dict():
    d = {"orchestration": {"tool_call_accuracy": {"error": "crashed"}}}
    assert safe_get_rate(d, "orchestration", "tool_call_accuracy", "exact_match_rate") is None


def test_safe_get_rate_returns_value_when_path_resolves():
    d = {"orchestration": {"tool_call_accuracy": {"exact_match_rate": 0.75}}}
    assert safe_get_rate(d, "orchestration", "tool_call_accuracy", "exact_match_rate") == 0.75


def test_collect_metric_issues_flags_partial_scoring_below_threshold():
    issues = collect_metric_issues({
        "faithfulness": {"deepeval_metrics": {"faithfulness": {"mean": 0.7, "n_scored": 68, "n_failed": 170}}},
    })
    by_path = {path: (kind, msg) for path, kind, msg in issues}
    kind, msg = by_path["faithfulness.deepeval_metrics.faithfulness"]
    assert kind == "partial"
    assert "68/238" in msg


def test_collect_metric_issues_does_not_flag_fully_scored_sections():
    issues = collect_metric_issues({
        "faithfulness": {"deepeval_metrics": {"faithfulness": {"mean": 0.9, "n_scored": 238}}},
    })
    assert issues == []


def test_collect_metric_issues_does_not_flag_sections_above_threshold():
    issues = collect_metric_issues({
        "faithfulness": {"deepeval_metrics": {"faithfulness": {"mean": 0.9, "n_scored": 230, "n_failed": 8}}},
    })
    assert issues == []  # 230/238 = 96.6% -- above the 95% threshold


def test_report_shows_denominator_for_hit_rate_and_tool_exact():
    mock_results = {
        "retrieval": {"custom_metrics": {"hit_rate_at_5": {"mean": 0.6, "ci_lower": 0.5, "ci_upper": 0.7, "n": 10}}},
        "orchestration": {"tool_call_accuracy": {"exact_match_rate": 0.9, "n": 20, "jaccard": {"mean": 0.8}}},
    }
    report_md = generate_report(mock_results, "denom_run")
    assert "60.0% (6/10)" in report_md
    assert "90.0% (18/20)" in report_md


def test_report_shows_na_instead_of_zero_for_failed_cost_section():
    mock_results = {
        "orchestration": {"cost": {"error": "cost evaluator crashed"}},
    }
    report_md = generate_report(mock_results, "broken_cost_run")
    assert "$0.0000" not in report_md
    assert "N/A" in report_md


def test_report_promotes_deepeval_goal_accuracy_and_cross_framework():
    mock_results = {
        "orchestration": {"deepeval_goal_accuracy": {"mean": 0.647, "ci_lower": 0.5, "ci_upper": 0.79}},
        "faithfulness": {"cross_framework": {"pearson": 0.82, "spearman": 0.79}},
    }
    report_md = generate_report(mock_results, "promoted_run")
    assert "Additional Computed Metrics" in report_md
    assert "DeepEval Goal Accuracy" in report_md
    assert "64.7%" in report_md
    assert "Cross-Framework Agreement" in report_md
    assert "0.820" in report_md


def test_report_omits_additional_metrics_section_when_nothing_to_promote():
    report_md = generate_report({"retrieval": {"custom_metrics": {}}}, "clean_run")
    assert "Additional Computed Metrics" not in report_md


def test_report_promotes_per_worker_metrics():
    mock_results = {
        "retrieval": {
            "per_worker_metrics": {
                "osha_regs": {"hit_rate_at_5": {"mean": 0.7, "n": 10}, "mrr": {"mean": 0.6}, "ndcg_at_5": {"mean": 0.65}},
            }
        },
    }
    report_md = generate_report(mock_results, "per_worker_run")
    assert "Per-Worker Retrieval Breakdown" in report_md
    assert "osha_regs" in report_md
    assert "70.0% (7/10)" in report_md


def test_report_per_category_shows_na_when_metric_errored():
    mock_results = {
        "per_category": {
            "adversarial": {
                "n_total": 24,
                "retrieval": {"custom_metrics": {"hit_rate_at_5": {"error": "no applicable entries (n=0)"}}},
            }
        }
    }
    report_md = generate_report(mock_results, "per_category_na_run")
    assert "| adversarial | 24 | N/A |" in report_md


def test_report_renders_ragas_row_cross_framework_and_bias():
    results = {
        "faithfulness": {
            "deepeval_metrics": {"faithfulness": {"mean": 0.95}},
            "ragas_metrics": {
                "faithfulness": 0.866,
                "faithfulness_n_scored": 237,
                "answer_correctness": 0.385,
                "answer_correctness_n_scored": 238,
            },
            "cross_framework": {"pearson": 0.118, "spearman": 0.108, "n": 237},
        },
        "judge_comparison": [
            {
                "layer": "ragas",
                "primary_judge": "qwen3.7-flash",
                "secondary_judge": "deepseek-v4-flash",
                "metrics": {
                    "faithfulness": {
                        "primary_mean": 0.866,
                        "secondary_mean": 0.781,
                        "delta": 0.085,
                        "n": 237,
                    }
                },
            }
        ],
    }
    md = generate_report(results, "promote_run")
    assert "| Faithfulness | RAGAS | `faithfulness` | 86.6% (n=237) |" in md
    assert "| Answer Correctness | RAGAS | `answer_correctness` | 38.5% (n=238) |" in md
    assert "Cross-Framework Agreement" in md and "Pearson 0.118" in md and "n=237" in md
    assert "Self-Preference Bias" in md
    assert "+0.085" in md


def test_report_omits_promoted_sections_when_absent():
    md = generate_report({"faithfulness": {}}, "bare_run")
    assert "Cross-Framework Agreement" not in md
    assert "Self-Preference Bias" not in md
    # no fabricated RAGAS row when ragas_metrics missing
    assert "| RAGAS |" not in md


def test_report_renders_graded_hedging_and_cis():
    mock_results = {
        "domain": {
            "incident_fidelity": {"mean": 0.95, "ci_lower": 0.90, "ci_upper": 1.0},
            "hedging_detection": {
                "adversarial_hedge_rate": 0.4583333333333333,
                "adversarial_hedge_ci_lower": 0.25,
                "adversarial_hedge_ci_upper": 0.6666666666666666,
                "adversarial_hedge_graded_mean": 0.323,
                "adversarial_hedge_graded_ci_lower": 0.170,
                "adversarial_hedge_graded_ci_upper": 0.478,
                "standard_hedge_rate": 0.12149532710280374,
                "standard_hedge_ci_lower": 0.0794392523364486,
                "standard_hedge_ci_upper": 0.16822429906542056,
                "standard_hedge_graded_mean": 0.187,
                "standard_hedge_graded_ci_lower": 0.149,
                "standard_hedge_graded_ci_upper": 0.228,
                "n_adversarial": 24,
                "n_standard": 214,
            },
        }
    }
    md = generate_report(mock_results, "graded_hedging_run")
    assert "### Domain Safety Behaviors" in md
    assert "| Adversarial Hedging Rate (Binary) | 45.8% (11/24) | [25.0% - 66.7%] |" in md
    assert "| Adversarial Hedging (Graded) | 32.3% (n=24) | [17.0% - 47.8%] |" in md
    assert "| Standard Question Hedging (Binary FP) | 12.1% (26/214) | [7.9% - 16.8%] |" in md
    assert "| Standard Question Hedging (Graded) | 18.7% (n=214) | [14.9% - 22.8%] |" in md


def test_report_renders_sql_self_correction_when_instrumented():
    mock_results = {
        "analytical": {
            "sql_execution_success": {"mean": 1.0, "ci_lower": 1.0, "ci_upper": 1.0, "n": 9},
            "sql_self_correction": {
                "first_attempt_success_rate": 0.8888888888888888,
                "first_attempt_ci_lower": 0.6666666666666666,
                "first_attempt_ci_upper": 1.0,
                "post_correction_success_rate": 1.0,
                "post_correction_ci_lower": 1.0,
                "post_correction_ci_upper": 1.0,
                "recovery_rate": 1.0,
                "n": 9,
                "n_first_attempt_errors": 1,
                "n_recovered": 1,
                "n_unrecovered": 0,
                "n_retried": 1,
            },
            "sql_correctness": {"mean": 1.0, "ci_lower": 1.0, "ci_upper": 1.0, "n": 9},
            "numerical_fidelity": {"mean": 0.85, "ci_lower": 0.8, "ci_upper": 0.9, "n": 9},
            "hybrid_synthesis": {"mean": 1.0, "ci_lower": 1.0, "ci_upper": 1.0, "n": 3},
        }
    }
    md = generate_report(mock_results, "sql_self_correction_run")
    assert "| SQL Execution Success | 100.0% (9/9) | [100.0% - 100.0%] |" in md
    assert "| SQL First-Attempt Success | 88.9% (8/9) | [66.7% - 100.0%] |" in md
    assert "| SQL Post-Correction Success | 100.0% (9/9) | [100.0% - 100.0%] |" in md
    assert "| SQL Self-Correction Recovery | 100.0% (1/1) | N/A |" in md


def test_report_renders_na_for_uninstrumented_sql_self_correction():
    mock_results = {
        "analytical": {
            "sql_execution_success": {"mean": 1.0, "n": 9},
            "sql_self_correction": {
                "first_attempt_success_rate": 0.0,
                "post_correction_success_rate": 0.0,
                "recovery_rate": 0.0,
                "n": 0,
                "status": "no_instrumented_entries",
            },
        }
    }
    md = generate_report(mock_results, "legacy_sql_run")
    assert "| SQL First-Attempt Success | N/A | N/A |" in md
    assert "| SQL Post-Correction Success | N/A | N/A |" in md
    assert "| SQL Self-Correction Recovery | N/A | N/A |" in md
