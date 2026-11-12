"""significance.py — Real hypothesis tests for cross-run metric comparisons.

Replaces the informal "CIs don't overlap" proxy
(and its silent ``return True`` when CIs are missing) with an actual paired test
on per-query scores, corrected for multiple comparisons across the ~16 reported
metrics.

Two pieces:

- :func:`paired_bootstrap_pvalue` — a two-sided paired bootstrap test of the mean
  per-query difference between two runs on the *same* dataset (paired by index).
  A permutation (sign-flip) variant is also provided for the same null.
- :func:`holm_bonferroni` — Holm–Bonferroni step-down correction that converts a
  family of raw p-values into family-wise-error-controlled reject/keep decisions
  and adjusted p-values.

No I/O and no project dependencies beyond NumPy, so it is trivially unit-testable
and reusable from the reporting/aggregation layers.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

__all__ = [
    "paired_bootstrap_pvalue",
    "paired_permutation_pvalue",
    "holm_bonferroni",
]


def _clean_pairs(
    baseline: Sequence[float], candidate: Sequence[float]
) -> Tuple[np.ndarray, np.ndarray]:
    """Align two per-query arrays into a clean paired float matrix.

    Pairs are matched by position (the analyzer already assumes both runs share
    dataset order). Pairs where either side is ``None``/``NaN`` are dropped so a
    metric that only scored a subset (e.g. citation accuracy) still yields a
    valid test over the entries both runs scored.

    Raises:
        ValueError: If the two sequences differ in length.
    """
    if len(baseline) != len(candidate):
        raise ValueError(
            f"paired test needs equal-length sequences, got "
            f"{len(baseline)} vs {len(candidate)}"
        )
    b = np.asarray(baseline, dtype=np.float64)
    c = np.asarray(candidate, dtype=np.float64)
    mask = ~(np.isnan(b) | np.isnan(c))
    return b[mask], c[mask]


def paired_bootstrap_pvalue(
    baseline: Sequence[float],
    candidate: Sequence[float],
    n_resamples: int = 10_000,
    seed: int = 42,
) -> float:
    """Two-sided paired bootstrap p-value for a difference in means.

    Tests H0: the mean per-query difference ``candidate - baseline`` is zero.
    The bootstrap resamples the *paired differences* with replacement and centers
    them at zero to build the null distribution of the mean difference; the
    two-sided p-value is the fraction of null means at least as extreme as the
    observed one. This matches the paired structure of two runs scored on the
    same queries.

    Args:
        baseline: Per-query metric values for the baseline run.
        candidate: Per-query metric values for the candidate run (same order).
        n_resamples: Number of bootstrap resamples.
        seed: RNG seed for reproducibility.

    Returns:
        A p-value in ``[0, 1]``. Returns ``1.0`` when fewer than two valid pairs
        remain (nothing to test) or the observed difference is exactly zero.
    """
    b, c = _clean_pairs(baseline, candidate)
    if b.size < 2:
        return 1.0

    diffs = c - b
    observed = float(diffs.mean())
    if observed == 0.0:
        return 1.0

    rng = np.random.default_rng(seed)
    n = diffs.size
    # Center the differences at zero to sample under H0, then resample.
    centered = diffs - diffs.mean()
    idx = rng.integers(0, n, size=(n_resamples, n))
    null_means = centered[idx].mean(axis=1)
    # Two-sided: how often is the null mean at least as far from 0 as observed.
    extreme = np.abs(null_means) >= abs(observed)
    # +1 smoothing keeps the p-value strictly positive (never a spurious 0.0).
    return float((extreme.sum() + 1) / (n_resamples + 1))


def paired_permutation_pvalue(
    baseline: Sequence[float],
    candidate: Sequence[float],
    n_resamples: int = 10_000,
    seed: int = 42,
) -> float:
    """Two-sided paired permutation (sign-flip) p-value for a difference in means.

    Under H0 the sign of each paired difference is exchangeable, so the null
    distribution is built by randomly flipping the sign of each per-query
    difference. Provided as an alternative to :func:`paired_bootstrap_pvalue`;
    both target the same null.

    Args:
        baseline: Per-query metric values for the baseline run.
        candidate: Per-query metric values for the candidate run (same order).
        n_resamples: Number of sign-flip permutations.
        seed: RNG seed for reproducibility.

    Returns:
        A p-value in ``[0, 1]``.
    """
    b, c = _clean_pairs(baseline, candidate)
    if b.size < 2:
        return 1.0

    diffs = c - b
    observed = float(diffs.mean())
    if observed == 0.0:
        return 1.0

    rng = np.random.default_rng(seed)
    n = diffs.size
    signs = rng.choice(np.array([-1.0, 1.0]), size=(n_resamples, n))
    null_means = (signs * diffs).mean(axis=1)
    extreme = np.abs(null_means) >= abs(observed)
    return float((extreme.sum() + 1) / (n_resamples + 1))


def holm_bonferroni(
    pvalues: Sequence[float], alpha: float = 0.05
) -> Tuple[List[bool], List[float]]:
    """Holm–Bonferroni step-down correction over a family of p-values.

    Controls the family-wise error rate at ``alpha`` while being uniformly more
    powerful than plain Bonferroni. Sort p-values ascending; the i-th smallest is
    compared against ``alpha / (m - i)``; the first failure stops all rejections
    at or beyond it. Adjusted p-values are the running maximum of
    ``(m - i) * p`` (clipped to 1.0) so they are monotone and directly reportable.

    Args:
        pvalues: Raw per-metric p-values (order preserved in the output).
        alpha: Family-wise significance level.

    Returns:
        ``(reject, adjusted)`` where ``reject[i]`` is True when metric ``i`` is
        significant after correction and ``adjusted[i]`` is its adjusted p-value,
        both in the input order.
    """
    m = len(pvalues)
    if m == 0:
        return [], []

    order = sorted(range(m), key=lambda i: pvalues[i])
    reject = [False] * m
    adjusted = [1.0] * m

    running_max = 0.0
    still_rejecting = True
    for rank, i in enumerate(order):
        factor = m - rank
        adj = min(1.0, factor * pvalues[i])
        running_max = max(running_max, adj)
        adjusted[i] = running_max
        if still_rejecting and pvalues[i] <= alpha / factor:
            reject[i] = True
        else:
            still_rejecting = False

    return reject, adjusted
