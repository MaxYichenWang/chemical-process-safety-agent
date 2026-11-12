"""Tests for :mod:`src.evaluation.applicability`.

The behaviour that matters most here is restraint: quarantining a bad reference
answer must shrink the denominator of reference-dependent metrics *only*, and
an entry with no flag at all must never be treated as excluded.
"""

import pytest

from src.evaluation.applicability import (
    CONTEXT_DEPENDENT_METRICS,
    REFERENCE_DEPENDENT_METRICS,
    UNKNOWN_DOMAIN,
    dataset_applicability,
    defect_codes,
    domain_bucket,
    domain_breakdown,
    exclude_from_reference_metrics,
    exclude_from_retrieval_metrics,
    recompute_reference_metrics,
)


# --------------------------------------------------------------------------- #
# Predicates
# --------------------------------------------------------------------------- #
def test_only_explicit_false_excludes_from_reference_metrics():
    assert exclude_from_reference_metrics({"answer_gradable": False}) is True
    assert exclude_from_reference_metrics({"answer_gradable": True}) is False
    # Absent means "not assessed", never "excluded" -- otherwise every dataset
    # predating the flag would silently lose its whole denominator.
    assert exclude_from_reference_metrics({}) is False


def test_only_explicit_false_excludes_from_retrieval_metrics():
    assert exclude_from_retrieval_metrics({"l1_applicable": False}) is True
    assert exclude_from_retrieval_metrics({}) is False


def test_domain_bucket_defaults_when_unlabelled():
    assert domain_bucket({}) == UNKNOWN_DOMAIN
    assert domain_bucket({"domain_relevance": "construction"}) == "construction"


def test_defect_codes_is_sorted_and_deduplicated():
    entry = {"defects": [{"code": "b"}, {"code": "a"}, {"code": "a"}, "not-a-dict"]}
    assert defect_codes(entry) == ["a", "b"]
    assert defect_codes({}) == []


def test_reference_and_context_metric_families_are_disjoint():
    """Faithfulness must never be treated as reference-dependent — it is graded
    against retrieved context and stays valid when the reference is quarantined."""
    assert not (REFERENCE_DEPENDENT_METRICS & CONTEXT_DEPENDENT_METRICS)
    assert "faithfulness" in CONTEXT_DEPENDENT_METRICS
    assert "correctness" in REFERENCE_DEPENDENT_METRICS


# --------------------------------------------------------------------------- #
# Dataset summary
# --------------------------------------------------------------------------- #
def test_dataset_applicability_counts_each_denominator_separately():
    dataset = [
        {"domain_relevance": "process_safety"},
        {"l1_applicable": False, "domain_relevance": "process_safety"},
        {"answer_gradable": False, "domain_relevance": "construction", "defects": [{"code": "x"}]},
    ]
    summary = dataset_applicability(dataset)
    assert summary["n_entries"] == 3
    assert summary["retrieval_denominator"] == 2
    assert summary["reference_denominator"] == 2
    assert summary["domain_relevance"] == {"construction": 1, "process_safety": 2}
    assert summary["defect_codes"] == {"x": 1}
    assert summary["entries_with_defects"] == 1


# --------------------------------------------------------------------------- #
# Recomputation
# --------------------------------------------------------------------------- #
def test_recompute_returns_none_when_nothing_is_quarantined():
    """No adjustment must be reported as *no adjustment*, not as a zero delta."""
    dataset = [{}, {}]
    per_entry = {"0": {"correctness": 1.0}, "1": {"correctness": 0.0}}
    assert recompute_reference_metrics(per_entry, {"correctness": {"mean": 0.5}}, dataset) is None


def test_recompute_excludes_only_quarantined_entries():
    dataset = [{}, {"answer_gradable": False}, {}]
    per_entry = {
        "0": {"correctness": 1.0, "faithfulness": 0.5},
        "1": {"correctness": 0.0, "faithfulness": 0.5},
        "2": {"correctness": 1.0, "faithfulness": 0.5},
    }
    aggregates = {"correctness": {"mean": 2 / 3}, "faithfulness": {"mean": 0.5}}
    adjusted = recompute_reference_metrics(per_entry, aggregates, dataset)

    assert adjusted["metrics"]["correctness"]["mean"] == pytest.approx(1.0)
    assert adjusted["metrics"]["correctness"]["n_scored"] == 2
    assert adjusted["metrics"]["correctness"]["n_excluded"] == 1
    assert adjusted["metrics"]["correctness"]["delta"] == pytest.approx(1 / 3)
    # Faithfulness is context-graded and must be left completely alone.
    assert "faithfulness" not in adjusted["metrics"]
    assert adjusted["context_dependent_metrics_unchanged"] == ["faithfulness"]


def test_recompute_refuses_to_emit_a_mean_from_nothing():
    """§3.3 honest reporting: absent is absent, never 0.0."""
    dataset = [{"answer_gradable": False}]
    per_entry = {"0": {"correctness": 0.9}}
    adjusted = recompute_reference_metrics(per_entry, {"correctness": {"mean": 0.9}}, dataset)
    assert adjusted["metrics"]["correctness"]["mean"] is None
    assert adjusted["metrics"]["correctness"]["n_scored"] == 0


def test_recompute_ignores_unparseable_and_null_entries():
    dataset = [{}, {"answer_gradable": False}]
    per_entry = {"0": {"correctness": 1.0}, "1": {"correctness": 0.0}, "bad": {"correctness": 5.0}, "9": None}
    adjusted = recompute_reference_metrics(per_entry, {"correctness": {"mean": 0.5}}, dataset)
    assert adjusted["metrics"]["correctness"]["n_scored"] == 1


# --------------------------------------------------------------------------- #
# Domain breakdown
# --------------------------------------------------------------------------- #
def test_domain_breakdown_groups_by_bucket():
    dataset = [
        {"domain_relevance": "process_safety"},
        {"domain_relevance": "process_safety"},
        {"domain_relevance": "construction"},
    ]
    per_entry = {
        "0": {"correctness": 1.0},
        "1": {"correctness": 0.0},
        "2": {"correctness": 1.0},
    }
    breakdown = domain_breakdown(per_entry, dataset, ["correctness"])
    assert breakdown["process_safety"]["correctness"]["mean"] == pytest.approx(0.5)
    assert breakdown["construction"]["correctness"]["n_scored"] == 1


def test_domain_breakdown_applies_the_same_quarantine_as_the_headline():
    """Breakdown and headline must be computed over the same population, or the
    parts will not reconcile with the whole."""
    dataset = [
        {"domain_relevance": "process_safety"},
        {"domain_relevance": "process_safety", "answer_gradable": False},
    ]
    per_entry = {"0": {"correctness": 1.0, "faithfulness": 0.2}, "1": {"correctness": 0.0, "faithfulness": 0.4}}
    breakdown = domain_breakdown(per_entry, dataset, ["correctness", "faithfulness"])
    assert breakdown["process_safety"]["correctness"]["n_scored"] == 1
    # ...but the context-graded metric keeps both entries.
    assert breakdown["process_safety"]["faithfulness"]["n_scored"] == 2


def test_domain_breakdown_skips_out_of_range_indices():
    breakdown = domain_breakdown({"7": {"correctness": 1.0}}, [{}], ["correctness"])
    assert breakdown == {}
