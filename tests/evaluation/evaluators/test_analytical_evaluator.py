"""Tests for AnalyticalEvaluator."""

import pytest

from src.evaluation.evaluators.analytical_evaluator import (
    AnalyticalEvaluator,
    _compare_sample_values,
    _extract_numbers,
    _is_analytical_entry,
    _values_match,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_analytical_entry(**overrides):
    """Return a minimal analytical evaluation entry."""
    base = {
        "query": "How many violations?",
        "expected_answer": "42",
        "ground_truth_chunks": [],
        "source": "auto_statistical",
        "category": "statistical_factual",
        "expected_agents": ["data_analyst"],
        "expected_sql": "SELECT COUNT(*) FROM violations",
    }
    base.update(overrides)
    return base


def _make_result(**overrides):
    """Return a minimal pipeline result."""
    base = {
        "final_response": "There are 42 violations.",
        "context_gathered": {
            "analytical_results": ["| count |\n|---|\n| 42 |"]
        },
        "retrieval_details": [{
            "collection": "analytical",
            "node_name": "DataAnalystWorker",
            "sql_executed": "SELECT COUNT(*) FROM violations",
            "row_count": 1,
            "columns": ["count"],
        }],
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Helper tests
# ---------------------------------------------------------------------------


class TestExtractNumbers:

    def test_integers(self):
        assert 42 in _extract_numbers("There are 42 items")

    def test_decimals(self):
        assert 3.14 in _extract_numbers("Pi is 3.14")

    def test_comma_separated(self):
        assert 1000000 in _extract_numbers("Count: 1,000,000")

    def test_empty(self):
        assert _extract_numbers("") == []

    def test_no_numbers(self):
        assert _extract_numbers("no numbers here") == []

    # The regression below: SQL result tables render numbers without thousands
    # separators, so an uncapped 4+ digit run must parse as ONE number. The old
    # `\d{1,3}(?:,\d{3})*` lead alternative split "3110.27" into [311.0, 0.27],
    # which made every well-formatted prose answer score as a fidelity miss.
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("3110.27", [3110.27]),
            ("20209", [20209.0]),
            ("81796", [81796.0]),
            ("1234567.89", [1234567.89]),
        ],
    )
    def test_unformatted_long_numbers_are_not_split(self, text, expected):
        assert _extract_numbers(text) == expected

    def test_comma_formatted_and_raw_forms_agree(self):
        assert _extract_numbers("81,796 inspections") == _extract_numbers("81796")

    def test_multiple_numbers_still_separate(self):
        assert _extract_numbers("rows 1234 and 56") == [1234.0, 56.0]

    def test_scientific_notation(self):
        assert _extract_numbers("1.3153867e+07 violations") == [13153867.0]
        assert _extract_numbers("5.14e6 inspections") == [5140000.0]

    def test_magnitude_units(self):
        assert _extract_numbers("~$229.1 million in penalties") == [229100000.0]
        assert _extract_numbers("2.5 billion dollars") == [2500000000.0]
        assert _extract_numbers("81.8k inspections") == [81800.0]

    def test_strip_list_markers(self):
        text = "Top items:\n1. Violations: 500\n2. Inspections: 200"
        assert _extract_numbers(text, strip_list_markers=True) == [500.0, 200.0]



class TestIsAnalyticalEntry:

    def test_with_data_analyst_agent(self):
        assert _is_analytical_entry({"expected_agents": ["data_analyst"]})

    def test_statistical_category(self):
        assert _is_analytical_entry({"category": "statistical_factual"})

    def test_hybrid_category(self):
        assert _is_analytical_entry({"category": "hybrid_analytical"})

    def test_non_analytical(self):
        assert not _is_analytical_entry({"category": "regulatory_factual"})


# ---------------------------------------------------------------------------
# Evaluator tests
# ---------------------------------------------------------------------------


class TestEvaluateSqlExecutionSuccess:

    def test_all_pass(self):
        evaluator = AnalyticalEvaluator()
        dataset = [_make_analytical_entry()]
        results = [_make_result()]
        metrics = evaluator.evaluate_sql_execution_success(dataset, results)
        assert metrics["mean"] == 1.0
        assert metrics["n"] == 1

    def test_with_errors(self):
        evaluator = AnalyticalEvaluator()
        dataset = [_make_analytical_entry()]
        results = [_make_result(retrieval_details=[{
            "collection": "analytical",
            "sql_executed": "error: column not found",
            "row_count": 0,
        }])]
        metrics = evaluator.evaluate_sql_execution_success(dataset, results)
        assert metrics["mean"] == 0.0

    def test_no_analytical_queries(self):
        evaluator = AnalyticalEvaluator()
        dataset = [{"category": "regulatory_factual", "expected_agents": ["osha_regs"]}]
        results = [_make_result()]
        metrics = evaluator.evaluate_sql_execution_success(dataset, results)
        assert metrics["n"] == 0


class TestValuesMatch:

    def test_exact_int(self):
        assert _values_match(42, 42)

    def test_float_tolerance(self):
        assert _values_match(100.0, 100.0000001)

    def test_float_mismatch(self):
        assert not _values_match(100.0, 101.0)

    def test_zero_comparison(self):
        assert _values_match(0, 0.0)
        assert not _values_match(0, 1.0)

    def test_string_case_insensitive(self):
        assert _values_match("OSHA", "osha")

    def test_string_mismatch(self):
        assert not _values_match("OSHA", "CSB")

    def test_numeric_string(self):
        assert _values_match("42", 42)


class TestCompareSampleValues:

    def test_perfect_match(self):
        gt = [{"count": 42, "label": "violations"}]
        actual = [{"count": 42, "label": "violations"}]
        assert _compare_sample_values(gt, actual) == 1.0

    def test_partial_match(self):
        gt = [{"count": 42, "label": "violations"}]
        actual = [{"count": 42, "label": "inspections"}]
        assert _compare_sample_values(gt, actual) == 0.5

    def test_no_actual(self):
        gt = [{"count": 42}]
        assert _compare_sample_values(gt, None) == 0.0
        assert _compare_sample_values(gt, []) == 0.0

    def test_multi_row(self):
        gt = [{"a": 1}, {"a": 2}]
        actual = [{"a": 1}, {"a": 99}]
        assert _compare_sample_values(gt, actual) == 0.5

    def test_missing_column_in_actual(self):
        gt = [{"a": 1, "b": 2}]
        actual = [{"a": 1}]
        # b is missing → 1 match out of 2 cells
        assert _compare_sample_values(gt, actual) == 0.5

    def test_alias_invariance_single_scalar(self):
        """Single scalar value query with differing column aliases matches 1.0."""
        gt = [{"total_violations": 13153867}]
        actual = [{"violation_count": 13153867}]
        assert _compare_sample_values(gt, actual) == 1.0

    def test_alias_invariance_multicol(self):
        """Multi-column query where one alias differs positionally matches 1.0."""
        gt = [{"violation_type": "O", "count": 6775764}]
        actual = [{"violation_type": "O", "frequency": 6775764}]
        assert _compare_sample_values(gt, actual) == 1.0

    def test_order_invariance_reverse_trend(self):
        """Trend queries sorted DESC vs ASC match 1.0."""
        gt = [{"year": 2025, "val": 10}, {"year": 2024, "val": 20}]
        actual = [{"year": 2024, "val": 20}, {"year": 2025, "val": 10}]
        assert _compare_sample_values(gt, actual) == 1.0



class TestEvaluateSqlCorrectness:

    def test_row_count_only_match(self):
        """Legacy: ground truth has row_count but no sample_values."""
        evaluator = AnalyticalEvaluator()
        dataset = [_make_analytical_entry(ground_truth_result={"row_count": 1, "columns": ["count"]})]
        results = [_make_result()]
        metrics = evaluator.evaluate_sql_correctness(dataset, results)
        assert metrics["mean"] == 1.0

    def test_row_count_only_mismatch(self):
        evaluator = AnalyticalEvaluator()
        dataset = [_make_analytical_entry(ground_truth_result={"row_count": 5, "columns": ["count"]})]
        results = [_make_result()]
        metrics = evaluator.evaluate_sql_correctness(dataset, results)
        assert metrics["mean"] == 0.0

    def test_no_ground_truth(self):
        evaluator = AnalyticalEvaluator()
        dataset = [_make_analytical_entry()]
        results = [_make_result()]
        metrics = evaluator.evaluate_sql_correctness(dataset, results)
        assert metrics["n"] == 0

    def test_empty_ground_truth_result(self):
        """Empty dict ground_truth_result should be skipped."""
        evaluator = AnalyticalEvaluator()
        dataset = [_make_analytical_entry(ground_truth_result={})]
        results = [_make_result()]
        metrics = evaluator.evaluate_sql_correctness(dataset, results)
        assert metrics["n"] == 0

    def test_hybrid_perfect_score(self):
        """Both row_count and sample_values match → 1.0."""
        evaluator = AnalyticalEvaluator()
        dataset = [_make_analytical_entry(ground_truth_result={
            "row_count": 1,
            "sample_values": [{"count": 42}],
        })]
        results = [_make_result(retrieval_details=[{
            "collection": "analytical",
            "sql_executed": "SELECT COUNT(*) FROM violations",
            "row_count": 1,
            "columns": ["count"],
            "sample_values": [{"count": 42}],
        }])]
        metrics = evaluator.evaluate_sql_correctness(dataset, results)
        assert metrics["mean"] == 1.0

    def test_hybrid_values_wrong(self):
        """Row count matches but values don't → 0.4 (only row_count component)."""
        evaluator = AnalyticalEvaluator()
        dataset = [_make_analytical_entry(ground_truth_result={
            "row_count": 1,
            "sample_values": [{"count": 42}],
        })]
        results = [_make_result(retrieval_details=[{
            "collection": "analytical",
            "sql_executed": "SELECT COUNT(*) FROM violations",
            "row_count": 1,
            "columns": ["count"],
            "sample_values": [{"count": 999}],
        }])]
        metrics = evaluator.evaluate_sql_correctness(dataset, results)
        assert metrics["mean"] == pytest.approx(0.4)

    def test_hybrid_both_wrong(self):
        """Row count and values both wrong → 0.0."""
        evaluator = AnalyticalEvaluator()
        dataset = [_make_analytical_entry(ground_truth_result={
            "row_count": 1,
            "sample_values": [{"count": 42}],
        })]
        results = [_make_result(retrieval_details=[{
            "collection": "analytical",
            "sql_executed": "SELECT COUNT(*) FROM violations",
            "row_count": 5,
            "columns": ["count"],
            "sample_values": [{"count": 999}],
        }])]
        metrics = evaluator.evaluate_sql_correctness(dataset, results)
        assert metrics["mean"] == pytest.approx(0.0)

    def test_hybrid_no_actual_values(self):
        """GT has sample_values but pipeline doesn't → value_score=0, score=0.4 if row_count matches."""
        evaluator = AnalyticalEvaluator()
        dataset = [_make_analytical_entry(ground_truth_result={
            "row_count": 1,
            "sample_values": [{"count": 42}],
        })]
        results = [_make_result()]  # default result has no sample_values
        metrics = evaluator.evaluate_sql_correctness(dataset, results)
        assert metrics["mean"] == pytest.approx(0.4)


class TestEvaluateNumericalFidelity:

    def test_all_numbers_match(self):
        evaluator = AnalyticalEvaluator()
        dataset = [_make_analytical_entry()]
        results = [_make_result(
            final_response="There are 42 violations total.",
            context_gathered={"analytical_results": ["Count: 42"]},
        )]
        metrics = evaluator.evaluate_numerical_fidelity(dataset, results)
        assert metrics["mean"] == 1.0

    def test_missing_numbers(self):
        evaluator = AnalyticalEvaluator()
        dataset = [_make_analytical_entry()]
        results = [_make_result(
            final_response="There are 42 violations and 100 inspections.",
            context_gathered={"analytical_results": ["Count: 42"]},
        )]
        metrics = evaluator.evaluate_numerical_fidelity(dataset, results)
        assert metrics["mean"] < 1.0

    def test_hybrid_context_numbers_match(self):
        """Numbers from both analytical table and CSB report docs count as faithful."""
        evaluator = AnalyticalEvaluator()
        dataset = [_make_analytical_entry(category="hybrid_analytical")]
        results = [_make_result(
            final_response="Found 42 violations. CSB report R3 from 2019 noted 3 recommendations.",
            context_gathered=[
                {"collection": "analytical", "corpus": "data_analyst", "text": "count: 42"},
                {"collection": "csb_recommendations", "corpus": "csb_recs", "text": "Recommendation R3 issued in 2019 with total 3 recommendations."},
            ],
        )]
        metrics = evaluator.evaluate_numerical_fidelity(dataset, results)
        assert metrics["mean"] == 1.0

    def test_no_analytical_queries(self):
        evaluator = AnalyticalEvaluator()
        dataset = [{"category": "regulatory_factual", "expected_agents": ["osha_regs"]}]
        results = [_make_result(final_response="No numbers here")]
        metrics = evaluator.evaluate_numerical_fidelity(dataset, results)
        assert metrics["n"] == 0


class TestEvaluateHybridSynthesis:

    def test_both_sources_present(self):
        evaluator = AnalyticalEvaluator()
        dataset = [_make_analytical_entry(category="hybrid_analytical")]
        results = [_make_result(
            final_response="There are 42 violations. Based on CSB investigation reports, the primary causes include equipment failure and inadequate maintenance procedures."
        )]
        metrics = evaluator.evaluate_hybrid_synthesis(dataset, results)
        assert metrics["mean"] == 1.0

    def test_missing_statistical(self):
        evaluator = AnalyticalEvaluator()
        dataset = [_make_analytical_entry(category="hybrid_analytical")]
        results = [_make_result(
            final_response="Based on CSB investigation reports, the primary causes include equipment failure."
        )]
        metrics = evaluator.evaluate_hybrid_synthesis(dataset, results)
        assert metrics["mean"] == 0.5

    def test_no_hybrid_queries(self):
        evaluator = AnalyticalEvaluator()
        dataset = [_make_analytical_entry(category="statistical_factual")]
        results = [_make_result()]
        metrics = evaluator.evaluate_hybrid_synthesis(dataset, results)
        assert metrics["n"] == 0


class TestEvaluateSqlErrorTaxonomy:

    def test_mixed_errors(self):
        evaluator = AnalyticalEvaluator()
        results = [
            _make_result(retrieval_details=[{
                "collection": "analytical",
                "sql_executed": "error: column xyz not found",
            }]),
            _make_result(retrieval_details=[{
                "collection": "analytical",
                "sql_executed": "error: timeout exceeded",
            }]),
            _make_result(),  # success
        ]
        taxonomy = evaluator.evaluate_sql_error_taxonomy(results)
        assert taxonomy["wrong_column"] == 1
        assert taxonomy["timeout"] == 1

    def test_no_errors(self):
        evaluator = AnalyticalEvaluator()
        results = [_make_result()]
        taxonomy = evaluator.evaluate_sql_error_taxonomy(results)
        assert all(v == 0 for v in taxonomy.values())


class TestEvaluateReturnsAllKeys:

    def test_returns_all_keys(self):
        evaluator = AnalyticalEvaluator()
        dataset = [_make_analytical_entry()]
        results = [_make_result()]
        output = evaluator.evaluate(dataset, results)
        expected_keys = {
            "sql_execution_success",
            "sql_self_correction",
            "sql_correctness",
            "numerical_fidelity",
            "hybrid_synthesis",
            "sql_error_taxonomy",
        }
        assert set(output.keys()) == expected_keys


class TestEvaluateSqlSelfCorrection:
    """first-attempt vs post-self-correction SQL success split."""

    def _detail(self, **kw):
        base = {
            "collection": "analytical",
            "node_name": "DataAnalystWorker",
            "sql_executed": "SELECT COUNT(*) FROM violations",
            "row_count": 1,
            "columns": ["count"],
            "sql_first_attempt": "SELECT COUNT(*) FROM violations",
            "first_attempt_error": False,
            "first_attempt_error_msg": "",
            "retry_triggered": False,
            "retry_reason": "",
            "recovered": False,
            "final_error": False,
        }
        base.update(kw)
        return base

    def test_clean_first_attempt(self):
        evaluator = AnalyticalEvaluator()
        dataset = [_make_analytical_entry()]
        results = [_make_result(retrieval_details=[self._detail()])]
        m = evaluator.evaluate_sql_self_correction(dataset, results)
        assert m["first_attempt_success_rate"] == 1.0
        assert m["post_correction_success_rate"] == 1.0
        assert m["n"] == 1
        assert m["n_first_attempt_errors"] == 0
        # No first-attempt failures -> recovery_rate defaults to 1.0
        assert m["recovery_rate"] == 1.0

    def test_recovered_after_error(self):
        evaluator = AnalyticalEvaluator()
        dataset = [_make_analytical_entry()]
        results = [_make_result(retrieval_details=[self._detail(
            first_attempt_error=True,
            first_attempt_error_msg="Binder Error: no such column",
            retry_triggered=True,
            retry_reason="execution_error",
            recovered=True,
            final_error=False,
        )])]
        m = evaluator.evaluate_sql_self_correction(dataset, results)
        assert m["first_attempt_success_rate"] == 0.0
        assert m["post_correction_success_rate"] == 1.0
        assert m["n_first_attempt_errors"] == 1
        assert m["n_recovered"] == 1
        assert m["n_unrecovered"] == 0
        assert m["recovery_rate"] == 1.0
        assert m["n_retried"] == 1

    def test_unrecovered_error(self):
        evaluator = AnalyticalEvaluator()
        dataset = [_make_analytical_entry()]
        results = [_make_result(retrieval_details=[self._detail(
            first_attempt_error=True,
            retry_triggered=True,
            retry_reason="execution_error",
            recovered=False,
            final_error=True,
        )])]
        m = evaluator.evaluate_sql_self_correction(dataset, results)
        assert m["first_attempt_success_rate"] == 0.0
        assert m["post_correction_success_rate"] == 0.0
        assert m["n_recovered"] == 0
        assert m["n_unrecovered"] == 1
        assert m["recovery_rate"] == 0.0

    def test_mixed_batch_recovery_rate(self):
        evaluator = AnalyticalEvaluator()
        dataset = [_make_analytical_entry() for _ in range(3)]
        results = [
            _make_result(retrieval_details=[self._detail()]),  # clean
            _make_result(retrieval_details=[self._detail(
                first_attempt_error=True, retry_triggered=True,
                recovered=True, final_error=False)]),  # recovered
            _make_result(retrieval_details=[self._detail(
                first_attempt_error=True, retry_triggered=True,
                recovered=False, final_error=True)]),  # unrecovered
        ]
        m = evaluator.evaluate_sql_self_correction(dataset, results)
        assert m["n"] == 3
        assert m["n_first_attempt_errors"] == 2
        assert m["recovery_rate"] == 0.5
        assert abs(m["first_attempt_success_rate"] - 1 / 3) < 1e-9
        assert abs(m["post_correction_success_rate"] - 2 / 3) < 1e-9

    def test_legacy_uninstrumented_results_are_skipped(self):
        evaluator = AnalyticalEvaluator()
        dataset = [_make_analytical_entry()]
        # No first_attempt_error field -> not counted (avoids scoring old runs).
        results = [_make_result()]
        m = evaluator.evaluate_sql_self_correction(dataset, results)
        assert m["n"] == 0
        assert m["status"] == "no_instrumented_entries"
