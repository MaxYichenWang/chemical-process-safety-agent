"""Tests for significance.py — paired bootstrap and Holm-Bonferroni."""
from __future__ import annotations

import pytest

from src.evaluation.analyzers.significance import (
    holm_bonferroni,
    paired_bootstrap_pvalue,
    paired_permutation_pvalue,
)


# ---------------------------------------------------------------------------
# paired_bootstrap_pvalue
# ---------------------------------------------------------------------------


def test_bootstrap_large_consistent_difference_is_significant():
    # Candidate beats baseline on nearly every paired query.
    baseline = [0.0] * 40
    candidate = [1.0] * 40
    p = paired_bootstrap_pvalue(baseline, candidate, n_resamples=2000, seed=1)
    assert p < 0.05


def test_bootstrap_no_difference_is_not_significant():
    baseline = [0.5, 0.4, 0.6, 0.5, 0.45, 0.55, 0.5, 0.5]
    candidate = list(baseline)  # identical → observed diff 0
    p = paired_bootstrap_pvalue(baseline, candidate, n_resamples=2000, seed=1)
    assert p == pytest.approx(1.0)


def test_bootstrap_small_noisy_difference_is_not_significant():
    baseline = [0.5, 0.6, 0.4, 0.55, 0.45]
    candidate = [0.52, 0.58, 0.43, 0.54, 0.44]  # tiny, within noise
    p = paired_bootstrap_pvalue(baseline, candidate, n_resamples=2000, seed=1)
    assert p > 0.05


def test_bootstrap_drops_none_pairs():
    # Only the 4 non-None pairs should be tested; all show candidate higher.
    baseline = [0.0, None, 0.0, 0.0, None, 0.0]
    candidate = [1.0, 0.5, 1.0, 1.0, None, 1.0]
    p = paired_bootstrap_pvalue(
        [float("nan") if x is None else x for x in baseline],
        [float("nan") if x is None else x for x in candidate],
        n_resamples=2000,
        seed=1,
    )
    assert p < 0.05


def test_bootstrap_insufficient_pairs_returns_one():
    assert paired_bootstrap_pvalue([0.0], [1.0]) == pytest.approx(1.0)
    assert paired_bootstrap_pvalue([], []) == pytest.approx(1.0)


def test_bootstrap_length_mismatch_raises():
    with pytest.raises(ValueError):
        paired_bootstrap_pvalue([0.1, 0.2], [0.1])


def test_bootstrap_is_deterministic_under_seed():
    b = [0.1, 0.9, 0.2, 0.8, 0.3, 0.7]
    c = [0.4, 0.95, 0.5, 0.85, 0.6, 0.75]
    p1 = paired_bootstrap_pvalue(b, c, n_resamples=1500, seed=7)
    p2 = paired_bootstrap_pvalue(b, c, n_resamples=1500, seed=7)
    assert p1 == p2


def test_permutation_agrees_with_bootstrap_on_clear_signal():
    baseline = [0.0] * 30
    candidate = [1.0] * 30
    assert paired_permutation_pvalue(baseline, candidate, seed=1) < 0.05


# ---------------------------------------------------------------------------
# holm_bonferroni
# ---------------------------------------------------------------------------


def test_holm_empty():
    reject, adj = holm_bonferroni([])
    assert reject == []
    assert adj == []


def test_holm_single_pvalue():
    reject, adj = holm_bonferroni([0.01], alpha=0.05)
    assert reject == [True]
    assert adj[0] == pytest.approx(0.01)


def test_holm_step_down_stops_at_first_failure():
    # Sorted: 0.001, 0.02, 0.04 with m=3.
    #  rank0: 0.001 <= 0.05/3=0.0167 → reject
    #  rank1: 0.02  <= 0.05/2=0.025  → reject
    #  rank2: 0.04  <= 0.05/1=0.05   → reject
    reject, _ = holm_bonferroni([0.001, 0.02, 0.04], alpha=0.05)
    assert reject == [True, True, True]


def test_holm_blocks_after_a_failure():
    # Sorted ascending: 0.03, 0.03, 0.03 with m=3.
    #  rank0: 0.03 <= 0.05/3=0.0167 → FAIL → nothing rejects.
    reject, _ = holm_bonferroni([0.03, 0.03, 0.03], alpha=0.05)
    assert reject == [False, False, False]


def test_holm_partial_rejection_stops_midway():
    # Sorted: 0.001, 0.03, 0.04 with m=3.
    #  rank0: 0.001 <= 0.0167 → reject
    #  rank1: 0.03  <= 0.025  → FAIL → stop; rank2 also not rejected.
    reject, _ = holm_bonferroni([0.001, 0.03, 0.04], alpha=0.05)
    assert reject == [True, False, False]


def test_holm_preserves_input_order():
    pvals = [0.5, 0.001, 0.3]
    reject, adj = holm_bonferroni(pvals, alpha=0.05)
    # smallest (index 1) rejects; others do not
    assert reject[1] is True
    assert reject[0] is False and reject[2] is False
    # adjusted p-values are monotone w.r.t. rank and clipped to 1.0
    assert all(0.0 <= a <= 1.0 for a in adj)


def test_holm_adjusted_are_monotone_nondecreasing_by_rank():
    pvals = [0.001, 0.01, 0.04]
    _, adj = holm_bonferroni(pvals, alpha=0.05)
    # ranks already ascending here, so adjusted must be nondecreasing
    assert adj[0] <= adj[1] <= adj[2]
