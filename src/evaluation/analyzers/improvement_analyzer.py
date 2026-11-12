"""improvement_analyzer.py — Cross-run delta analysis for eval result JSONs.

Reads two result JSONs (baseline and candidate) and produces a structured
ComparisonReport with per-metric deltas, CI overlap tests, regression flags,
per-category breakdowns, and agent routing diffs.

No DB dependencies — operates entirely on result JSON files.

Usage:
    analyzer = ImprovementAnalyzer()
    report = analyzer.compare_runs(
        baseline_path=Path("runs/baseline/results.json"),
        candidate_path=Path("runs/candidate/results.json"),
    )
    print(analyzer.generate_report(report))
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.evaluation.analyzers.significance import (
    holm_bonferroni,
    paired_bootstrap_pvalue,
)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class MetricDelta:
    name: str
    baseline: float
    candidate: float
    delta: float
    baseline_ci: Optional[Tuple[float, float]] = None
    candidate_ci: Optional[Tuple[float, float]] = None
    # True when the difference is significant. Preference order:
    #   1. paired bootstrap p-value, Holm-Bonferroni corrected across metrics
    #   2. non-overlapping bootstrap CIs (weaker fallback, no per-query data)
    # Never True without one of those — see significance_method.
    significant: bool = False
    # Raw and family-wise-corrected paired-test p-values (None when no
    # per-query scores were available for this metric).
    p_value: Optional[float] = None
    p_value_adjusted: Optional[float] = None
    # How `significant` was decided: "holm_bootstrap" | "ci_overlap" | "none"
    significance_method: str = "none"
    # True when candidate is worse by > REGRESSION_THRESHOLD
    regression: bool = False


@dataclass
class CategoryDelta:
    category: str
    n: int
    baseline_tool_exact: float
    candidate_tool_exact: float
    delta: float


@dataclass
class RoutingDiff:
    query: str
    category: str
    baseline_agents: List[str]
    candidate_agents: List[str]


@dataclass
class ComparisonReport:
    baseline_path: str
    candidate_path: str
    n_baseline: int
    n_candidate: int
    metric_deltas: List[MetricDelta] = field(default_factory=list)
    category_deltas: List[CategoryDelta] = field(default_factory=list)
    routing_diffs: List[RoutingDiff] = field(default_factory=list)
    regressions: List[MetricDelta] = field(default_factory=list)
    improvements: List[MetricDelta] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Flag as regression if metric drops by more than this amount (absolute)
REGRESSION_THRESHOLD = 0.02

# Family-wise significance level for the Holm-Bonferroni-corrected paired tests.
SIGNIFICANCE_ALPHA = 0.05

# Friendly display names for metric paths
_METRIC_LABELS: Dict[str, str] = {
    "retrieval.custom_metrics.hit_rate_at_5": "Hit@5",
    "retrieval.custom_metrics.mrr": "MRR",
    "retrieval.custom_metrics.ndcg_at_5": "NDCG@5",
    "retrieval.deepeval_metrics.contextual_precision": "Contextual Precision",
    "retrieval.deepeval_metrics.contextual_recall": "Contextual Recall",
    "faithfulness.citation_accuracy": "Citation Accuracy (Faithfulness)",
    "orchestration.tool_call_accuracy.exact_match_rate": "Tool Exact Match",
    "orchestration.tool_call_accuracy.jaccard": "Tool Jaccard",
    "orchestration.pipeline_error_rate.error_rate": "Pipeline Error Rate",
    "domain.citation_accuracy": "Citation Accuracy (Domain)",
    "domain.incident_fidelity": "Incident Fidelity",
    "domain.hedging_detection.adversarial_hedge_rate": "Adversarial Hedge Rate",
    "domain.hedging_detection.standard_hedge_rate": "Hedging FP Rate",
    "analytical.sql_execution_success": "SQL Execution Success",
    "analytical.sql_correctness": "SQL Correctness",
    "analytical.numerical_fidelity": "Numerical Fidelity",
}

# Higher = better for all except error rate and hedging FP rate
_LOWER_IS_BETTER = {
    "orchestration.pipeline_error_rate.error_rate",
    "domain.hedging_detection.standard_hedge_rate",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _deep_get(d: dict, dotted_path: str, default=None) -> Any:
    """Navigate a nested dict with a dot-separated path."""
    keys = dotted_path.split(".")
    cur = d
    for k in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k, default)
        if cur is default:
            return default
    return cur


def _extract_mean_ci(
    blob: Any,
) -> Tuple[Optional[float], Optional[Tuple[float, float]]]:
    """Extract (mean, (ci_lower, ci_upper)) from a metric blob.

    Handles three shapes:
    - {"mean": float, "ci_lower": float, "ci_upper": float}
    - {"error_rate": float}   (pipeline_error_rate)
    - scalar float
    """
    if blob is None:
        return None, None
    if isinstance(blob, (int, float)):
        return float(blob), None
    if isinstance(blob, dict):
        # pipeline_error_rate shape
        if "error_rate" in blob:
            return float(blob["error_rate"]), None
        mean = blob.get("mean")
        if mean is None:
            return None, None
        ci_lower = blob.get("ci_lower")
        ci_upper = blob.get("ci_upper")
        ci = (
            (float(ci_lower), float(ci_upper))
            if ci_lower is not None and ci_upper is not None
            else None
        )
        return float(mean), ci
    return None, None


def _extract_perquery_list(metrics: dict, metric_path: str) -> Optional[List[float]]:
    """Return a per-query score list for *metric_path*, or None if absent.

    Recognises the ``<metric>_list`` sibling convention the aggregation layer
    uses (e.g. ``faithfulness_list`` next to ``faithfulness``), as well as a
    ``per_query`` / ``scores`` / ``list`` key nested inside the metric blob.
    Values are returned verbatim (including ``None``), so downstream pairing can
    drop entries only one run scored.
    """
    # 1. sibling "<leaf>_list" path, e.g. retrieval.custom_metrics.hit_rate_at_5_list
    sibling = _deep_get(metrics, metric_path + "_list")
    if isinstance(sibling, list):
        return sibling

    # 2. list nested inside the metric blob under a conventional key
    blob = _deep_get(metrics, metric_path)
    if isinstance(blob, dict):
        for key in ("per_query", "scores", "list", "values"):
            nested = blob.get(key)
            if isinstance(nested, list):
                return nested
    return None


def _to_float_or_nan(x: Any) -> float:
    """Coerce a per-query score to float, mapping None/non-numeric to NaN."""
    if x is None:
        return float("nan")
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def _cis_overlap(
    ci_a: Optional[Tuple[float, float]], ci_b: Optional[Tuple[float, float]]
) -> bool:
    """Return True if the two CIs overlap (or if either CI is unavailable).

    Used only as the weaker fallback significance signal when a metric has no
    per-query scores for the paired bootstrap test. When CIs are missing
    this returns True so the caller treats the metric as *not* distinguishable
    rather than silently flagging significance without evidence.
    """
    if ci_a is None or ci_b is None:
        return True  # no CIs and no per-query data → not distinguishable
    return ci_a[0] <= ci_b[1] and ci_b[0] <= ci_a[1]


def _load_result(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class ImprovementAnalyzer:
    """Compare two eval result JSONs and produce a structured delta report."""

    def compare_runs(
        self,
        baseline_path: Path,
        candidate_path: Path,
    ) -> ComparisonReport:
        """Compare baseline and candidate runs.

        Parameters
        ----------
        baseline_path : Path
            Path to the baseline result JSON (e.g. T2_v1).
        candidate_path : Path
            Path to the candidate result JSON (e.g. T2_v2 post-fix).

        Returns
        -------
        ComparisonReport
        """
        b_path = Path(baseline_path)
        c_path = Path(candidate_path)

        b_data = _load_result(b_path)
        c_data = _load_result(c_path)

        b_metrics = b_data.get("metrics", {})
        c_metrics = c_data.get("metrics", {})
        b_pipeline = b_data.get("pipeline_results", [])
        c_pipeline = c_data.get("pipeline_results", [])
        b_dataset = b_data.get("dataset", [])
        c_dataset = c_data.get("dataset", [])

        report = ComparisonReport(
            baseline_path=str(b_path),
            candidate_path=str(c_path),
            n_baseline=len(b_pipeline),
            n_candidate=len(c_pipeline),
        )

        # --- Per-metric deltas ---
        # First pass: build deltas with a CI-overlap fallback and collect any
        # per-query paired scores. Significance is finalised after the loop so
        # the paired p-values can be Holm-Bonferroni corrected as one family.
        perquery_pairs: Dict[int, Tuple[List[float], List[float]]] = {}
        for path, label in _METRIC_LABELS.items():
            b_blob = _deep_get(b_metrics, path)
            c_blob = _deep_get(c_metrics, path)
            b_mean, b_ci = _extract_mean_ci(b_blob)
            c_mean, c_ci = _extract_mean_ci(c_blob)

            if b_mean is None or c_mean is None:
                continue

            delta = c_mean - b_mean
            lower_is_better = path in _LOWER_IS_BETTER
            improvement_delta = -delta if lower_is_better else delta

            regression = improvement_delta < -REGRESSION_THRESHOLD

            md = MetricDelta(
                name=label,
                baseline=b_mean,
                candidate=c_mean,
                delta=delta,
                baseline_ci=b_ci,
                candidate_ci=c_ci,
                # Provisional: CI non-overlap. Overridden below by the paired
                # test whenever per-query scores exist for this metric.
                significant=not _cis_overlap(b_ci, c_ci),
                significance_method=(
                    "ci_overlap" if b_ci is not None and c_ci is not None else "none"
                ),
                regression=regression,
            )
            idx = len(report.metric_deltas)
            report.metric_deltas.append(md)

            b_list = _extract_perquery_list(b_metrics, path)
            c_list = _extract_perquery_list(c_metrics, path)
            if b_list is not None and c_list is not None and len(b_list) == len(c_list):
                perquery_pairs[idx] = (
                    [_to_float_or_nan(x) for x in b_list],
                    [_to_float_or_nan(x) for x in c_list],
                )

            if regression:
                report.regressions.append(md)
            elif improvement_delta > REGRESSION_THRESHOLD:
                report.improvements.append(md)

        # --- Real significance test: paired bootstrap + Holm-Bonferroni ---
        # Every metric with per-query scores gets an actual paired p-value; the
        # family of p-values is Holm-corrected across metrics, and `significant`
        # for those metrics is set strictly from the corrected result (never the
        # weaker CI-overlap proxy). Metrics without per-query data keep the
        # CI-overlap fallback and are labelled as such.
        if perquery_pairs:
            indices = list(perquery_pairs.keys())
            raw_pvalues = [paired_bootstrap_pvalue(*perquery_pairs[i]) for i in indices]
            reject, adjusted = holm_bonferroni(raw_pvalues, alpha=SIGNIFICANCE_ALPHA)
            for i, p_raw, p_adj, is_sig in zip(indices, raw_pvalues, adjusted, reject):
                md = report.metric_deltas[i]
                md.p_value = p_raw
                md.p_value_adjusted = p_adj
                md.significant = bool(is_sig)
                md.significance_method = "holm_bootstrap"
            if len(indices) < len(report.metric_deltas):
                report.notes.append(
                    f"Significance: {len(indices)}/{len(report.metric_deltas)} "
                    "metrics tested via paired bootstrap + Holm-Bonferroni; the "
                    "rest fall back to CI non-overlap (no per-query scores)."
                )

        # --- Per-category tool exact match ---
        b_cat = b_metrics.get("per_category", {})
        c_cat = c_metrics.get("per_category", {})
        all_cats = sorted(set(b_cat) | set(c_cat))
        for cat in all_cats:
            b_c = b_cat.get(cat, {})
            c_c = c_cat.get(cat, {})
            b_exact = _deep_get(
                b_c, "orchestration.tool_call_accuracy.exact_match_rate"
            )
            c_exact = _deep_get(
                c_c, "orchestration.tool_call_accuracy.exact_match_rate"
            )
            n = b_c.get("n_total") or c_c.get("n_total") or 0
            if b_exact is None and c_exact is None:
                continue
            report.category_deltas.append(
                CategoryDelta(
                    category=cat,
                    n=int(n),
                    baseline_tool_exact=float(b_exact or 0.0),
                    candidate_tool_exact=float(c_exact or 0.0),
                    delta=float((c_exact or 0.0) - (b_exact or 0.0)),
                )
            )

        # --- Agent routing diffs ---
        # Match queries by position (assumes same dataset order)
        paired = list(zip(b_dataset, b_pipeline, c_dataset, c_pipeline))
        for b_ds, b_pr, c_ds, c_pr in paired:
            b_agents = sorted(b_pr.get("planned_agents") or [])
            c_agents = sorted(c_pr.get("planned_agents") or [])
            if b_agents != c_agents:
                report.routing_diffs.append(
                    RoutingDiff(
                        query=b_ds.get("query", "")[:120],
                        category=b_ds.get("category", ""),
                        baseline_agents=b_agents,
                        candidate_agents=c_agents,
                    )
                )

        # --- Notes ---
        if report.n_baseline != report.n_candidate:
            report.notes.append(
                f"Dataset size changed: baseline={report.n_baseline}, candidate={report.n_candidate}. "
                "Per-metric comparisons may be affected."
            )
        if report.regressions:
            report.notes.append(
                f"{len(report.regressions)} regression(s) detected (dropped >{REGRESSION_THRESHOLD*100:.0f}pp): "
                + ", ".join(r.name for r in report.regressions)
            )

        return report

    # ------------------------------------------------------------------
    # Report rendering
    # ------------------------------------------------------------------

    def generate_report(self, report: ComparisonReport) -> str:
        """Render a markdown report suitable for committing alongside results."""
        lines: List[str] = []

        lines.append("# Improvement Analyzer Report")
        lines.append("")
        lines.append(
            f"- **Baseline:** `{report.baseline_path}` (N={report.n_baseline})"
        )
        lines.append(
            f"- **Candidate:** `{report.candidate_path}` (N={report.n_candidate})"
        )
        lines.append("")

        # --- Notes / warnings ---
        if report.notes:
            lines.append("## ⚠️ Notes")
            for note in report.notes:
                lines.append(f"- {note}")
            lines.append("")

        # --- Regressions ---
        if report.regressions:
            lines.append("## 🔴 Regressions (dropped >2pp)")
            lines.append("")
            lines.append("| Metric | Baseline | Candidate | Delta |")
            lines.append("|--------|----------|-----------|-------|")
            for r in report.regressions:
                lines.append(
                    f"| {r.name} | {r.baseline:.3f} | {r.candidate:.3f} "
                    f"| **{r.delta:+.3f}** |"
                )
            lines.append("")

        # --- All metric deltas ---
        lines.append("## Metric Deltas")
        lines.append("")
        lines.append(
            "| Metric | Baseline | Candidate | Delta | p (adj) | Significant | Test |"
        )
        lines.append(
            "|--------|----------|-----------|-------|---------|-------------|------|"
        )
        for md in report.metric_deltas:
            sig = "✓" if md.significant else ""
            reg = (
                " 🔴"
                if md.regression
                else (
                    " 🟢"
                    if (
                        md.delta > REGRESSION_THRESHOLD
                        and md.name not in _LOWER_IS_BETTER
                    )
                    else ""
                )
            )
            if md.p_value_adjusted is not None:
                p_str = f"{md.p_value_adjusted:.3f}"
            else:
                p_str = "—"
            method = {
                "holm_bootstrap": "boot+Holm",
                "ci_overlap": "CI",
                "none": "—",
            }.get(md.significance_method, md.significance_method)
            lines.append(
                f"| {md.name} | {md.baseline:.3f} | {md.candidate:.3f} "
                f"| {md.delta:+.3f}{reg} | {p_str} | {sig} | {method} |"
            )
        lines.append("")
        lines.append(
            "_Significance: paired bootstrap on per-query scores, Holm–Bonferroni "
            "corrected across metrics (`boot+Holm`); metrics without per-query "
            "scores fall back to bootstrap-CI non-overlap (`CI`). No metric is "
            "flagged significant without one of these tests._"
        )
        lines.append("")

        # --- Per-category tool exact ---
        if report.category_deltas:
            lines.append("## Per-Category Tool Exact Match")
            lines.append("")
            lines.append("| Category | N | Baseline | Candidate | Delta |")
            lines.append("|----------|---|----------|-----------|-------|")
            for cd in sorted(report.category_deltas, key=lambda x: -abs(x.delta)):
                arrow = "🟢" if cd.delta > 0.02 else ("🔴" if cd.delta < -0.02 else "")
                lines.append(
                    f"| {cd.category} | {cd.n} "
                    f"| {cd.baseline_tool_exact:.1%} | {cd.candidate_tool_exact:.1%} "
                    f"| {cd.delta:+.1%} {arrow} |"
                )
            lines.append("")

        # --- Routing diffs summary ---
        if report.routing_diffs:
            lines.append(
                f"## Agent Routing Changes ({len(report.routing_diffs)} queries changed)"
            )
            lines.append("")
            lines.append("| Category | Baseline Agents | Candidate Agents | Query |")
            lines.append("|----------|----------------|-----------------|-------|")
            for rd in report.routing_diffs[:30]:  # cap at 30 to keep report readable
                b = ", ".join(rd.baseline_agents) or "(none)"
                c = ", ".join(rd.candidate_agents) or "(none)"
                q = rd.query[:80].replace("|", "\\|")
                lines.append(f"| {rd.category} | {b} | {c} | {q} |")
            if len(report.routing_diffs) > 30:
                lines.append(f"| ... | | | *(+{len(report.routing_diffs)-30} more)* |")
            lines.append("")

        return "\n".join(lines)
