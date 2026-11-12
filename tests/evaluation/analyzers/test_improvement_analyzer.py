"""Tests for improvement_analyzer.py.

All tests use synthetic in-memory result fixtures — no file I/O needed
(files are written to tmp_path for the path-based APIs).
"""
from __future__ import annotations

import json
import pytest
from pathlib import Path

from src.evaluation.analyzers.improvement_analyzer import (
    ComparisonReport,
    ImprovementAnalyzer,
    _cis_overlap,
    _deep_get,
    _extract_mean_ci,
)


# ---------------------------------------------------------------------------
# Unit helpers
# ---------------------------------------------------------------------------


def test_deep_get_simple():
    d = {"a": {"b": {"c": 42}}}
    assert _deep_get(d, "a.b.c") == 42


def test_deep_get_missing():
    d = {"a": {"b": 1}}
    assert _deep_get(d, "a.x.y", default="MISS") == "MISS"


def test_extract_mean_ci_full():
    blob = {"mean": 0.5, "ci_lower": 0.4, "ci_upper": 0.6}
    mean, ci = _extract_mean_ci(blob)
    assert mean == pytest.approx(0.5)
    assert ci == pytest.approx((0.4, 0.6))


def test_extract_mean_ci_error_rate():
    blob = {"error_rate": 0.03, "error_count": 2, "total": 60}
    mean, ci = _extract_mean_ci(blob)
    assert mean == pytest.approx(0.03)
    assert ci is None


def test_extract_mean_ci_scalar():
    mean, ci = _extract_mean_ci(0.75)
    assert mean == pytest.approx(0.75)
    assert ci is None


def test_extract_mean_ci_none():
    mean, ci = _extract_mean_ci(None)
    assert mean is None


def test_cis_overlap_overlapping():
    assert _cis_overlap((0.3, 0.6), (0.5, 0.8)) is True


def test_cis_overlap_non_overlapping():
    assert _cis_overlap((0.3, 0.5), (0.6, 0.8)) is False


def test_cis_overlap_one_none():
    assert _cis_overlap(None, (0.5, 0.8)) is True


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_result(
    n: int = 10,
    tool_exact: float = 0.4,
    hit_at_5: float = 0.4,
    error_rate: float = 0.0,
    adversarial_hedge: float = 0.2,
    citation_acc: float = 0.5,
    agents_override: list | None = None,
) -> dict:
    """Build a minimal result JSON structure."""
    ds_entry = {"query": "Test query?", "category": "enforcement_violation", "expected_agents": ["osha_enf"]}
    planned = agents_override or ["osha_enf"]
    pr_entry = {"planned_agents": planned, "agents_called": planned}
    dataset = [dict(ds_entry) for _ in range(n)]
    pipeline = [dict(pr_entry) for _ in range(n)]
    n_exact = int(n * tool_exact)
    for i in range(n_exact):
        pipeline[i]["planned_agents"] = dataset[i].get("expected_agents", ["osha_enf"])

    return {
        "dataset": dataset,
        "pipeline_results": pipeline,
        "metrics": {
            "retrieval": {
                "custom_metrics": {
                    "hit_rate_at_5": {"mean": hit_at_5, "ci_lower": hit_at_5 - 0.05, "ci_upper": hit_at_5 + 0.05},
                    "mrr": {"mean": 0.3, "ci_lower": 0.25, "ci_upper": 0.35},
                    "ndcg_at_5": {"mean": 0.3, "ci_lower": 0.25, "ci_upper": 0.35},
                },
                "deepeval_metrics": {
                    "contextual_precision": {"mean": 0.5, "ci_lower": 0.45, "ci_upper": 0.55},
                    "contextual_recall": {"mean": 0.7, "ci_lower": 0.65, "ci_upper": 0.75},
                },
            },
            "faithfulness": {
                "citation_accuracy": {"mean": citation_acc, "ci_lower": citation_acc - 0.05, "ci_upper": citation_acc + 0.05},
                "deepeval_metrics": {"faithfulness_list": [1.0] * n},
            },
            "orchestration": {
                "pipeline_error_rate": {"error_rate": error_rate, "error_count": 0, "total": n},
                "tool_call_accuracy": {
                    "exact_match_rate": tool_exact,
                    "jaccard": {"mean": 0.7, "ci_lower": 0.65, "ci_upper": 0.75},
                },
                "plan_accuracy": {"exact_match_rate": tool_exact, "n": n},
            },
            "domain": {
                "citation_accuracy": {"mean": citation_acc},
                "incident_fidelity": {"mean": 0.8},
                "hedging_detection": {
                    "adversarial_hedge_rate": adversarial_hedge,
                    "standard_hedge_rate": 0.1,
                },
                "safety_claim_detection": {"claim_traceability": 1.0},
            },
            "analytical": {
                "sql_execution_success": {"mean": 1.0, "ci_lower": 1.0, "ci_upper": 1.0, "n": 5},
                "sql_correctness": {"mean": 0.0, "ci_lower": 0.0, "ci_upper": 0.0, "n": 0},
                "numerical_fidelity": {"mean": 0.5, "ci_lower": 0.3, "ci_upper": 0.7, "n": 5},
            },
            "per_category": {
                "enforcement_violation": {
                    "n_total": n,
                    "n_ok": n,
                    "n_error": 0,
                    "orchestration": {
                        "tool_call_accuracy": {
                            "exact_match_rate": tool_exact,
                            "jaccard": {"mean": 0.7},
                        }
                    },
                }
            },
        },
    }


@pytest.fixture
def baseline_file(tmp_path) -> Path:
    p = tmp_path / "baseline.json"
    data = _make_result(n=20, tool_exact=0.4, hit_at_5=0.4)
    p.write_text(json.dumps(data))
    return p


@pytest.fixture
def candidate_file(tmp_path) -> Path:
    p = tmp_path / "candidate.json"
    data = _make_result(n=20, tool_exact=0.7, hit_at_5=0.4)
    p.write_text(json.dumps(data))
    return p


@pytest.fixture
def regression_file(tmp_path) -> Path:
    p = tmp_path / "regression.json"
    data = _make_result(n=20, tool_exact=0.2, hit_at_5=0.4)
    p.write_text(json.dumps(data))
    return p


# ---------------------------------------------------------------------------
# compare_runs tests
# ---------------------------------------------------------------------------


def test_compare_returns_report(baseline_file, candidate_file):
    analyzer = ImprovementAnalyzer()
    report = analyzer.compare_runs(baseline_file, candidate_file)
    assert isinstance(report, ComparisonReport)
    assert report.n_baseline == 20
    assert report.n_candidate == 20


def test_compare_metric_deltas_populated(baseline_file, candidate_file):
    analyzer = ImprovementAnalyzer()
    report = analyzer.compare_runs(baseline_file, candidate_file)
    # Should have at least the core metrics
    names = {md.name for md in report.metric_deltas}
    assert "Tool Exact Match" in names
    assert "Hit@5" in names
    assert "Pipeline Error Rate" in names


def test_compare_tool_exact_delta(baseline_file, candidate_file):
    """Candidate has 0.7 tool exact vs baseline 0.4 → delta +0.3."""
    analyzer = ImprovementAnalyzer()
    report = analyzer.compare_runs(baseline_file, candidate_file)
    tool_exact_delta = next(md for md in report.metric_deltas if md.name == "Tool Exact Match")
    assert tool_exact_delta.delta == pytest.approx(0.3, abs=1e-9)
    assert tool_exact_delta.candidate == pytest.approx(0.7, abs=1e-9)


def test_compare_regression_detected(baseline_file, regression_file):
    """When candidate has lower tool exact, regression is flagged."""
    analyzer = ImprovementAnalyzer()
    report = analyzer.compare_runs(baseline_file, regression_file)
    assert len(report.regressions) > 0
    reg_names = {r.name for r in report.regressions}
    assert "Tool Exact Match" in reg_names


def test_compare_no_regression_when_improved(baseline_file, candidate_file):
    analyzer = ImprovementAnalyzer()
    report = analyzer.compare_runs(baseline_file, candidate_file)
    reg_names = {r.name for r in report.regressions}
    assert "Tool Exact Match" not in reg_names


def test_compare_improvements_detected(baseline_file, candidate_file):
    analyzer = ImprovementAnalyzer()
    report = analyzer.compare_runs(baseline_file, candidate_file)
    imp_names = {md.name for md in report.improvements}
    assert "Tool Exact Match" in imp_names


def test_compare_ci_significance(tmp_path):
    """Non-overlapping CIs should be flagged as significant via the CI fallback.

    hit_rate_at_5 carries no per-query list here, so the analyzer falls back to
    the bootstrap-CI non-overlap test and labels the method accordingly.
    """
    b_path = tmp_path / "b.json"
    c_path = tmp_path / "c.json"
    # Baseline CI: (0.3, 0.4), Candidate CI: (0.6, 0.7) — no overlap
    b_data = _make_result(n=20, hit_at_5=0.35)
    c_data = _make_result(n=20, hit_at_5=0.65)
    b_data["metrics"]["retrieval"]["custom_metrics"]["hit_rate_at_5"] = {
        "mean": 0.35, "ci_lower": 0.30, "ci_upper": 0.40
    }
    c_data["metrics"]["retrieval"]["custom_metrics"]["hit_rate_at_5"] = {
        "mean": 0.65, "ci_lower": 0.60, "ci_upper": 0.70
    }
    b_path.write_text(json.dumps(b_data))
    c_path.write_text(json.dumps(c_data))
    analyzer = ImprovementAnalyzer()
    report = analyzer.compare_runs(b_path, c_path)
    hit_at_5_delta = next(md for md in report.metric_deltas if md.name == "Hit@5")
    assert hit_at_5_delta.significant is True
    assert hit_at_5_delta.significance_method == "ci_overlap"
    assert hit_at_5_delta.p_value is None


# ---------------------------------------------------------------------------
# real significance test (paired bootstrap + Holm-Bonferroni)
# ---------------------------------------------------------------------------


def _add_perquery(data: dict, path: str, values: list) -> None:
    """Attach a per-query "<leaf>_list" sibling under a dotted metric path."""
    keys = path.split(".")
    cur = data["metrics"]
    for k in keys[:-1]:
        cur = cur.setdefault(k, {})
    cur[keys[-1] + "_list"] = values


def test_paired_test_used_when_perquery_lists_present(tmp_path):
    """A large, consistent per-query difference is flagged via paired bootstrap."""
    b_path = tmp_path / "b.json"
    c_path = tmp_path / "c.json"
    n = 40
    b_data = _make_result(n=n, citation_acc=0.2)
    c_data = _make_result(n=n, citation_acc=0.9)
    _add_perquery(b_data, "faithfulness.citation_accuracy", [0.0] * n)
    _add_perquery(c_data, "faithfulness.citation_accuracy", [1.0] * n)
    b_path.write_text(json.dumps(b_data))
    c_path.write_text(json.dumps(c_data))

    report = ImprovementAnalyzer().compare_runs(b_path, c_path)
    md = next(
        m for m in report.metric_deltas
        if m.name == "Citation Accuracy (Faithfulness)"
    )
    assert md.significance_method == "holm_bootstrap"
    assert md.p_value is not None
    assert md.p_value_adjusted is not None
    assert md.significant is True


def test_paired_test_not_significant_for_tiny_perquery_difference(tmp_path):
    b_path = tmp_path / "b.json"
    c_path = tmp_path / "c.json"
    n = 20
    b_vals = [0.5, 0.6, 0.4, 0.55, 0.45] * 4
    c_vals = [0.51, 0.59, 0.41, 0.54, 0.46] * 4  # negligible shift
    b_data = _make_result(n=n)
    c_data = _make_result(n=n)
    _add_perquery(b_data, "faithfulness.citation_accuracy", b_vals)
    _add_perquery(c_data, "faithfulness.citation_accuracy", c_vals)
    b_path.write_text(json.dumps(b_data))
    c_path.write_text(json.dumps(c_data))

    report = ImprovementAnalyzer().compare_runs(b_path, c_path)
    md = next(
        m for m in report.metric_deltas
        if m.name == "Citation Accuracy (Faithfulness)"
    )
    assert md.significance_method == "holm_bootstrap"
    assert md.significant is False


def test_holm_correction_notes_partial_coverage(tmp_path):
    """When only some metrics have per-query data, a coverage note is emitted."""
    b_path = tmp_path / "b.json"
    c_path = tmp_path / "c.json"
    n = 30
    b_data = _make_result(n=n)
    c_data = _make_result(n=n)
    _add_perquery(b_data, "faithfulness.citation_accuracy", [0.0] * n)
    _add_perquery(c_data, "faithfulness.citation_accuracy", [1.0] * n)
    b_path.write_text(json.dumps(b_data))
    c_path.write_text(json.dumps(c_data))

    report = ImprovementAnalyzer().compare_runs(b_path, c_path)
    assert any("paired bootstrap" in note for note in report.notes)


def test_report_renders_pvalue_and_method_columns(tmp_path):
    b_path = tmp_path / "b.json"
    c_path = tmp_path / "c.json"
    n = 40
    b_data = _make_result(n=n)
    c_data = _make_result(n=n)
    _add_perquery(b_data, "faithfulness.citation_accuracy", [0.0] * n)
    _add_perquery(c_data, "faithfulness.citation_accuracy", [1.0] * n)
    b_path.write_text(json.dumps(b_data))
    c_path.write_text(json.dumps(c_data))

    analyzer = ImprovementAnalyzer()
    report = analyzer.compare_runs(b_path, c_path)
    md_report = analyzer.generate_report(report)
    assert "p (adj)" in md_report
    assert "boot+Holm" in md_report



def test_compare_dataset_size_mismatch_note(tmp_path):
    b_path = tmp_path / "b.json"
    c_path = tmp_path / "c.json"
    b_path.write_text(json.dumps(_make_result(n=10)))
    c_path.write_text(json.dumps(_make_result(n=20)))
    analyzer = ImprovementAnalyzer()
    report = analyzer.compare_runs(b_path, c_path)
    assert any("size changed" in note for note in report.notes)


def test_compare_per_category_deltas(baseline_file, candidate_file):
    analyzer = ImprovementAnalyzer()
    report = analyzer.compare_runs(baseline_file, candidate_file)
    assert len(report.category_deltas) > 0
    cat_names = {cd.category for cd in report.category_deltas}
    assert "enforcement_violation" in cat_names


def test_compare_routing_diffs_detected(tmp_path):
    """Queries where planned_agents differ between runs should appear in routing_diffs."""
    b_path = tmp_path / "b.json"
    c_path = tmp_path / "c.json"

    # tool_exact=0 prevents the fixture from overriding planned_agents with expected_agents,
    # ensuring all entries keep the specified agents_override value.
    b_data = _make_result(n=5, tool_exact=0, agents_override=["osha_regs", "osha_enf"])
    c_data = _make_result(n=5, tool_exact=0, agents_override=["osha_enf"])
    b_path.write_text(json.dumps(b_data))
    c_path.write_text(json.dumps(c_data))

    analyzer = ImprovementAnalyzer()
    report = analyzer.compare_runs(b_path, c_path)
    # All 5 queries changed routing
    assert len(report.routing_diffs) == 5
    assert report.routing_diffs[0].baseline_agents == ["osha_enf", "osha_regs"]
    assert report.routing_diffs[0].candidate_agents == ["osha_enf"]


# ---------------------------------------------------------------------------
# generate_report tests
# ---------------------------------------------------------------------------


def test_generate_report_contains_metric_table(baseline_file, candidate_file):
    analyzer = ImprovementAnalyzer()
    report = analyzer.compare_runs(baseline_file, candidate_file)
    md_report = analyzer.generate_report(report)
    assert "## Metric Deltas" in md_report
    assert "Tool Exact Match" in md_report
    assert "Hit@5" in md_report


def test_generate_report_regression_section_shown(baseline_file, regression_file):
    analyzer = ImprovementAnalyzer()
    report = analyzer.compare_runs(baseline_file, regression_file)
    md_report = analyzer.generate_report(report)
    assert "Regressions" in md_report


def test_generate_report_no_regression_section_when_none(baseline_file, candidate_file):
    analyzer = ImprovementAnalyzer()
    report = analyzer.compare_runs(baseline_file, candidate_file)
    md_report = analyzer.generate_report(report)
    assert "Regressions" not in md_report


def test_generate_report_category_section(baseline_file, candidate_file):
    analyzer = ImprovementAnalyzer()
    report = analyzer.compare_runs(baseline_file, candidate_file)
    md_report = analyzer.generate_report(report)
    assert "Per-Category Tool Exact Match" in md_report
    assert "enforcement_violation" in md_report


def test_generate_report_routing_diffs_section(tmp_path):
    b_path = tmp_path / "b.json"
    c_path = tmp_path / "c.json"
    b_data = _make_result(n=3, tool_exact=0, agents_override=["osha_regs", "osha_enf"])
    c_data = _make_result(n=3, tool_exact=0, agents_override=["osha_enf"])
    b_path.write_text(json.dumps(b_data))
    c_path.write_text(json.dumps(c_data))
    analyzer = ImprovementAnalyzer()
    report = analyzer.compare_runs(b_path, c_path)
    md_report = analyzer.generate_report(report)
    assert "Agent Routing Changes" in md_report
    assert "osha_regs" in md_report
