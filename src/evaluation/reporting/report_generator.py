import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)

try:
    from src.evaluation.config import EVAL_RESULTS_DIR
except ImportError:
    EVAL_RESULTS_DIR = Path("data/evaluation/results")


def format_metric(value: float, is_percentage: bool = True) -> str:
    if is_percentage:
        return f"{value * 100:.1f}%"
    return f"{value:.3f}"


def format_currency(value: float) -> str:
    return f"${value:.4f}"


def format_ci(ci_lower, ci_upper, is_percentage=True):
    if is_percentage:
        return f"[{ci_lower*100:.1f}% - {ci_upper*100:.1f}%]"
    return f"[{ci_lower:.3f} - {ci_upper:.3f}]"


def format_ci_safe(
    d: Dict[str, Any],
    lower_key: str = "ci_lower",
    upper_key: str = "ci_upper",
    *,
    is_percentage: bool = True,
) -> str:
    """'N/A' for an errored/skipped/incomplete section instead of the
    misleading ``[0.0% - 0.0%]`` a bare ``.get(..., 0.0)`` would print."""
    if not isinstance(d, dict) or "error" in d or "skipped" in d:
        return "N/A"
    if d.get(lower_key) is None or d.get(upper_key) is None:
        return "N/A"
    return format_ci(d[lower_key], d[upper_key], is_percentage=is_percentage)


def format_currency_safe(d: Dict[str, Any], key: str) -> str:
    """'N/A' instead of a silent ``$0.0000`` when the cost section
    errored/was skipped or never populated *key*."""
    if not isinstance(d, dict) or "error" in d or "skipped" in d or d.get(key) is None:
        return "N/A"
    return format_currency(d[key])


def safe_get(d, *keys, default=0.0):
    val = d
    for k in keys:
        if isinstance(val, dict) and k in val:
            val = val[k]
        else:
            return default
    return val


def safe_get_rate(d, *keys):
    """Like :func:`safe_get`, but returns ``None`` -- never a numeric default
    -- when the path doesn't resolve to a scalar. A path that dead-ends on an
    ``{"error": ...}``/``{"skipped": ...}`` dict (no ``"mean"``/whatever the
    last key is) is exactly that case: the caller should render "N/A", not
    silently reuse ``safe_get``'s ``0.0`` default."""
    val = d
    for k in keys:
        if isinstance(val, dict) and k in val:
            val = val[k]
        else:
            return None
    if isinstance(val, dict):
        return None
    return val


def format_rate_with_n(
    d: Dict[str, Any],
    value_key: str = "mean",
    n_key: str = "n",
    *,
    as_fraction: bool = True,
) -> str:
    """Render a rate with sample-size context, or ``"N/A"`` when the
    section failed/was skipped/has no value at all -- never a numeric 0 for a
    missing/failed section, which is indistinguishable from a real zero (a
    broken judge published as "0.0% faithful on every query", a crashed
    evaluator published as "0% routing accuracy").

    ``as_fraction`` controls how the ``n`` denominator is rendered, and MUST
    match the metric's actual per-item scoring:

    * ``True`` (default) -- ``"64.4% (132/205)"``. Only valid when every
      per-item contribution to *value_key* is strictly binary (0.0/1.0), so
      ``value * n`` really does recover an integer hit-count (e.g. Hit@5,
      tool-call exact match: both are literal ``count / total``).
    * ``False`` -- ``"35.6% (n=9)"``. Required whenever the underlying
      per-item scores are continuous or multi-valued (e.g. NDCG, MRR, SQL
      correctness's 0.4/0.6 hybrid weighting, hybrid-synthesis's 0/0.5/1) --
      for these, ``value * n`` is not a meaningful integer and would
      fabricate a numerator that looks precise but isn't real.
    """
    if (
        not isinstance(d, dict)
        or "error" in d
        or "skipped" in d
        or d.get("status") == "no_applicable_entries"
    ):
        return "N/A"
    value = d.get(value_key)
    if value is None:
        return "N/A"
    n = d.get(n_key)
    if n is None:
        return format_metric(value)
    if as_fraction:
        numerator = round(value * n)
        return f"{format_metric(value)} ({numerator}/{n})"
    return f"{format_metric(value)} (n={n})"


# Keys whose presence means "this number is missing", not "this number is zero".
_ISSUE_KEYS = ("error", "skipped")

# per_category repeats the same metric tree once per category; reporting its
# issues would bury the top-level ones under dozens of duplicates.
_ISSUE_SCAN_SKIP = ("per_category",)

# A metric section that scored fewer than this fraction of its rows renders
# a numeric mean fine, but silently hides how much of it is missing. 68/238
# scored (28.6%) must be flagged, not published as a clean 0.647.
_PARTIAL_SCORE_THRESHOLD = 0.95


def collect_metric_issues(results: Dict[str, Any]) -> list:
    """Walk the metrics tree and return ``(path, kind, message)`` for every gap.

    Metric sections that fail are rendered by the tables below as ``0.0%``,
    which is indistinguishable from a real zero -- exactly how a broken RAGAS
    judge was published as "0.0% faithful on every query" and how a crashed
    orchestration evaluator was published as 0% routing accuracy at $0.0000
    cost. Surfacing this list at the top of the report makes that impossible to
    miss again. Also flags sections that scored *some* but not enough rows
    ("partial") even though they render a clean numeric mean.
    """
    issues = []

    def _walk(node: Any, path: str) -> None:
        if not isinstance(node, dict):
            return
        for key in _ISSUE_KEYS:
            value = node.get(key)
            if isinstance(value, str):
                issues.append((path or "(root)", key, value))

        n_scored = node.get("n_scored")
        if isinstance(n_scored, (int, float)):
            n_failed = node.get("n_failed") or 0
            total = n_scored + n_failed
            if total > 0:
                ratio = n_scored / total
                if ratio < _PARTIAL_SCORE_THRESHOLD:
                    issues.append(
                        (
                            path or "(root)",
                            "partial",
                            f"{n_scored}/{total} scored ({ratio * 100:.1f}%) -- below "
                            f"{_PARTIAL_SCORE_THRESHOLD * 100:.0f}% threshold",
                        )
                    )

        for key, value in node.items():
            if key in _ISSUE_SCAN_SKIP or key in _ISSUE_KEYS:
                continue
            if isinstance(value, dict):
                _walk(value, f"{path}.{key}" if path else key)

    _walk(results, "")
    return issues


def _render_data_quality(results: Dict[str, Any]) -> list:
    issues = collect_metric_issues(results)
    if not issues:
        return ["## 0. Data Quality", "All metric sections computed successfully.", ""]

    errors = [i for i in issues if i[1] == "error"]
    skipped = [i for i in issues if i[1] == "skipped"]
    partial = [i for i in issues if i[1] == "partial"]

    summary_bits = []
    if errors:
        summary_bits.append(f"**{len(errors)} metric section(s) failed**")
    if partial:
        summary_bits.append(f"**{len(partial)} section(s) partially scored** (<95%)")
    if skipped:
        summary_bits.append(f"{len(skipped)} skipped by configuration")

    md = [
        "## 0. Data Quality",
        (
            (
                "⚠️ "
                + "; ".join(summary_bits)
                + ". Failed metrics render as N/A in the "
                "tables below, and partially-scored metrics still show their mean -- "
                "check the denominator before trusting it."
            )
            if (errors or partial)
            else f"{len(skipped)} metric section(s) skipped by configuration."
        ),
        "",
        "| Metric path | Status | Detail |",
        "|-------------|--------|--------|",
    ]
    _ICONS = {"error": "🔴 error", "skipped": "⚪ skipped", "partial": "🟡 partial"}
    for path, kind, message in errors + partial + skipped:
        icon = _ICONS.get(kind, kind)
        detail = message.replace("|", "\\|").replace("\n", " ")[:300]
        md.append(f"| `{path}` | {icon} | {detail} |")
    md.append("")
    return md


def generate_report(results: Dict[str, Any], run_id: str) -> str:
    """Generate Markdown report from metrics dict."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    md = [
        f"# Evaluation Run: {run_id}",
        f"**Date:** {timestamp}",
        "",
    ]
    md.extend(_render_data_quality(results))
    md.extend(
        [
            "## 1. Executive Summary",
            "Key evaluation metrics across all three measurement layers:",
            "",
        ]
    )

    # Layer 1 Metrics
    retrieval = results.get("retrieval", {})
    custom = retrieval.get("custom_metrics", {})
    r_ndcg = custom.get("ndcg_at_5", {})
    r_hit = custom.get("hit_rate_at_5", {})

    md.extend(
        [
            "### Layer 1: Retrieval Quality",
            "| Metric | Mean (n/N) | 95% CI |",
            "|--------|------------|--------|",
            f"| Hit Rate @ 5 | {format_rate_with_n(r_hit)} | {format_ci_safe(r_hit)} |",
            # NDCG is a continuous per-query score, not a binary hit/miss -- an
            # (x/n) fraction here would fabricate a numerator (see
            # format_rate_with_n's as_fraction docstring).
            f"| NDCG @ 5 | {format_rate_with_n(r_ndcg, as_fraction=False)} | {format_ci_safe(r_ndcg)} |",
            "",
        ]
    )

    # Layer 2 Metrics
    faithfulness = results.get("faithfulness", {})
    deep_faith = faithfulness.get("deepeval_metrics", {})
    faith_c_acc = faithfulness.get("citation_accuracy", {})
    # R2: Use domain citation accuracy as primary (covers OSHA stds + enforcement IDs)
    domain = results.get("domain", {})
    domain_c_acc = domain.get("citation_accuracy", {})

    deep_faithfulness = (
        deep_faith.get("faithfulness", {}) if isinstance(deep_faith, dict) else {}
    )
    deep_hallucination = (
        deep_faith.get("hallucination", {}) if isinstance(deep_faith, dict) else {}
    )

    # RAGAS faithfulness/answer-correctness are stored as bare floats (+ an
    # ``*_n_scored`` sibling), not the {mean, ci_*, n_scored} dicts DeepEval uses.
    ragas_faith = faithfulness.get("ragas_metrics", {})
    ragas_rows = []
    if isinstance(ragas_faith, dict):
        rf = ragas_faith.get("faithfulness")
        if isinstance(rf, (int, float)):
            n = ragas_faith.get("faithfulness_n_scored")
            n_note = f" (n={n})" if n is not None else ""
            ragas_rows.append(
                f"| Faithfulness | RAGAS | `faithfulness` | {format_metric(rf)}{n_note} |"
            )
        ac = ragas_faith.get("answer_correctness")
        if isinstance(ac, (int, float)):
            n = ragas_faith.get("answer_correctness_n_scored")
            n_note = f" (n={n})" if n is not None else ""
            ragas_rows.append(
                f"| Answer Correctness | RAGAS | `answer_correctness` | {format_metric(ac)}{n_note} |"
            )

    md.extend(
        [
            "### Layer 2: Generation Faithfulness",
            "| Metric | Framework | Component | Mean |",
            "|--------|-----------|-----------|------|",
            # DeepEval scores and citation accuracy are continuous/ratio-of-ratios,
            # not a single binary hit/miss per item -- as_fraction=False avoids
            # fabricating a numerator (see format_rate_with_n's docstring).
            f"| Faithfulness | DeepEval | `faithfulness` | {format_rate_with_n(deep_faithfulness, as_fraction=False)} |",
            f"| Hallucination | DeepEval | `hallucination` | {format_rate_with_n(deep_hallucination, as_fraction=False)} |",
            *ragas_rows,
            f"| Citation Accuracy (domain) | Custom | OSHA std + enforcement ID | {format_rate_with_n(domain_c_acc, as_fraction=False)} |",
            f"| Citation Accuracy (regex-only) | Custom | OSHA std only | {format_rate_with_n(faith_c_acc, as_fraction=False)} |",
            "",
        ]
    )

    # Layer 3 Orchestration
    orch = results.get("orchestration", {})
    tca = orch.get("tool_call_accuracy", {})
    pla = orch.get("plan_accuracy", {})
    frc = orch.get("fast_route_coverage", {})
    cost = orch.get("cost", {})
    rr = orch.get("retry_rate", {})
    per = orch.get("pipeline_error_rate", {})

    tca_jaccard = tca.get("jaccard", {}) if isinstance(tca, dict) else {}
    pla_jaccard = pla.get("jaccard", {}) if isinstance(pla, dict) else {}
    if (
        not isinstance(frc, dict)
        or "error" in frc
        or "skipped" in frc
        or frc.get("fast_route_rate") is None
    ):
        fast_route_row = "N/A"
    else:
        fr_count = frc.get("fast_route_count", 0)
        llm_count = frc.get("llm_plan_count", 0)
        fast_route_row = f"{format_metric(frc['fast_route_rate'])} ({fr_count}/{fr_count + llm_count})"

    md.extend(
        [
            "### Layer 3: Orchestration & Logic",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Pipeline Error Rate | {format_rate_with_n(per, value_key='error_rate', n_key='total')} |",
            f"| Tool Call Exact Match | {format_rate_with_n(tca, value_key='exact_match_rate')} |",
            f"| Tool Call Jaccard | {format_rate_with_n(tca_jaccard, as_fraction=False)} |",
            f"| Plan Exact Match | {format_rate_with_n(pla, value_key='exact_match_rate')} |",
            f"| Plan Jaccard | {format_rate_with_n(pla_jaccard, as_fraction=False)} |",
            f"| Fast-Route Rate | {fast_route_row} |",
            f"| Graph Retry Rate | {format_rate_with_n(rr, value_key='retry_rate')} |",
            f"| Mean Cost / Query | {format_currency_safe(cost, 'mean_per_query_usd')} |",
            f"| P95 Cost / Query | {format_currency_safe(cost, 'p95_per_query_usd')} |",
            f"| Total LLM Cost | {format_currency_safe(cost, 'total_usd')} |",
            "",
        ]
    )

    # Domain specific
    domain = results.get("domain", {})
    inc_fid = domain.get("incident_fidelity", {})
    hedg = domain.get("hedging_detection", {})

    md.extend(
        [
            "### Domain Safety Behaviors",
            "| Metric | Value | 95% CI |",
            "|--------|-------|--------|",
            f"| Incident Details Fidelity | {format_rate_with_n(inc_fid, as_fraction=False)} | {format_ci_safe(inc_fid)} |",
            f"| Adversarial Hedging Rate (Binary) | {format_rate_with_n(hedg, value_key='adversarial_hedge_rate', n_key='n_adversarial')} | {format_ci_safe(hedg, 'adversarial_hedge_ci_lower', 'adversarial_hedge_ci_upper')} |",
            f"| Adversarial Hedging (Graded) | {format_rate_with_n(hedg, value_key='adversarial_hedge_graded_mean', n_key='n_adversarial', as_fraction=False)} | {format_ci_safe(hedg, 'adversarial_hedge_graded_ci_lower', 'adversarial_hedge_graded_ci_upper')} |",
            f"| Standard Question Hedging (Binary FP) | {format_rate_with_n(hedg, value_key='standard_hedge_rate', n_key='n_standard')} | {format_ci_safe(hedg, 'standard_hedge_ci_lower', 'standard_hedge_ci_upper')} |",
            f"| Standard Question Hedging (Graded) | {format_rate_with_n(hedg, value_key='standard_hedge_graded_mean', n_key='n_standard', as_fraction=False)} | {format_ci_safe(hedg, 'standard_hedge_graded_ci_lower', 'standard_hedge_graded_ci_upper')} |",
            "",
        ]
    )

    # Layer 4: Statistical Aggregation
    analytical = results.get("analytical", {})
    sql_exec = analytical.get("sql_execution_success", {})
    sql_sc = analytical.get("sql_self_correction", {})
    sql_corr = analytical.get("sql_correctness", {})
    num_fid = analytical.get("numerical_fidelity", {})
    hybrid = analytical.get("hybrid_synthesis", {})

    if (
        isinstance(sql_sc, dict)
        and "error" not in sql_sc
        and "skipped" not in sql_sc
        and sql_sc.get("status") != "no_instrumented_entries"
        and sql_sc.get("n", 0) > 0
    ):
        first_att_row = f"| SQL First-Attempt Success | {format_rate_with_n(sql_sc, value_key='first_attempt_success_rate')} | {format_ci_safe(sql_sc, 'first_attempt_ci_lower', 'first_attempt_ci_upper')} |"
        post_corr_row = f"| SQL Post-Correction Success | {format_rate_with_n(sql_sc, value_key='post_correction_success_rate')} | {format_ci_safe(sql_sc, 'post_correction_ci_lower', 'post_correction_ci_upper')} |"
        n_first_errs = sql_sc.get("n_first_attempt_errors", 0)
        n_rec = sql_sc.get("n_recovered", 0)
        rec_rate = sql_sc.get("recovery_rate", 0.0)
        if n_first_errs > 0:
            rec_str = f"{format_metric(rec_rate)} ({n_rec}/{n_first_errs})"
        else:
            rec_str = f"{format_metric(rec_rate)} (0 first-pass errors)"
        rec_row = f"| SQL Self-Correction Recovery | {rec_str} | N/A |"
    else:
        first_att_row = "| SQL First-Attempt Success | N/A | N/A |"
        post_corr_row = "| SQL Post-Correction Success | N/A | N/A |"
        rec_row = "| SQL Self-Correction Recovery | N/A | N/A |"

    md.extend(
        [
            "### Layer 4: Statistical Aggregation",
            "| Metric | Mean (n/N) | 95% CI |",
            "|--------|------------|--------|",
            # SQL execution success is a real binary per-query success/fail count.
            # SQL correctness (0.4 row-count + 0.6 value-match hybrid weighting),
            # numerical fidelity (per-query matched/total-numbers ratio), and
            # hybrid synthesis (0/0.5/1 three-valued) are all continuous or
            # multi-valued per item, not a single hit/miss -- as_fraction=False
            # avoids fabricating a numerator (see format_rate_with_n's docstring).
            f"| SQL Execution Success | {format_rate_with_n(sql_exec)} | {format_ci_safe(sql_exec)} |",
            first_att_row,
            post_corr_row,
            rec_row,
            f"| SQL Correctness | {format_rate_with_n(sql_corr, as_fraction=False)} | {format_ci_safe(sql_corr)} |",
            f"| Numerical Fidelity | {format_rate_with_n(num_fid, as_fraction=False)} | {format_ci_safe(num_fid)} |",
            f"| Hybrid Synthesis | {format_rate_with_n(hybrid, as_fraction=False)} | {format_ci_safe(hybrid)} |",
            "",
        ]
    )

    # Promote metrics that were already computed somewhere in the
    # results tree but never rendered into report.md.
    promoted_md = []

    goal_acc = orch.get("deepeval_goal_accuracy")
    if (
        isinstance(goal_acc, dict)
        and "error" not in goal_acc
        and "skipped" not in goal_acc
        and goal_acc.get("mean") is not None
    ):
        ci_note = (
            f" (95% CI {format_ci(goal_acc['ci_lower'], goal_acc['ci_upper'])})"
            if "ci_lower" in goal_acc
            else ""
        )
        promoted_md.append(
            f"- **DeepEval Goal Accuracy**: {format_metric(goal_acc['mean'])}{ci_note}"
        )

    topic_adh = orch.get("ragas_topic_adherence")
    if (
        isinstance(topic_adh, dict)
        and "error" not in topic_adh
        and "skipped" not in topic_adh
        and topic_adh.get("topic_adherence") is not None
    ):
        promoted_md.append(
            f"- **RAGAS Topic Adherence**: {format_metric(topic_adh['topic_adherence'])}"
        )

    cross = faithfulness.get("cross_framework")
    if (
        isinstance(cross, dict)
        and "error" not in cross
        and ("pearson" in cross or "spearman" in cross)
    ):
        n_note = f", n={cross['n']}" if cross.get("n") is not None else ""
        pearson = cross.get("pearson")
        spearman = cross.get("spearman")
        p_str = f"{pearson:.3f}" if isinstance(pearson, (int, float)) else "N/A"
        s_str = f"{spearman:.3f}" if isinstance(spearman, (int, float)) else "N/A"
        promoted_md.append(
            f"- **Cross-Framework Agreement** (RAGAS vs DeepEval faithfulness): "
            f"Pearson {p_str}, Spearman {s_str}{n_note}"
        )

    if promoted_md:
        md.extend(["### Additional Computed Metrics", ""] + promoted_md + [""])

    judge_comparison = results.get("judge_comparison")
    if isinstance(judge_comparison, list) and judge_comparison:
        md.extend(
            [
                "### Self-Preference Bias (Judge Comparison)",
                "Per-metric mean delta between a cross-family judge (primary) and the "
                "self-family judge (secondary) over the same generation. Positive delta "
                "= primary judge scores higher.",
                "",
                "| Framework | Metric | Primary | Secondary | Δ (primary − secondary) | n |",
                "|-----------|--------|---------|-----------|-------------------------|---|",
            ]
        )
        for comp in judge_comparison:
            layer_name = comp.get("layer", "?")
            primary_j = comp.get("primary_judge", "?")
            secondary_j = comp.get("secondary_judge", "?")
            for metric, d in (comp.get("metrics") or {}).items():
                md.append(
                    f"| {layer_name} (`{primary_j}` vs `{secondary_j}`) | `{metric}` | "
                    f"{format_metric(d['primary_mean'])} | {format_metric(d['secondary_mean'])} | "
                    f"{d['delta']:+.3f} | {d['n']} |"
                )
        md.append("")

    per_worker = retrieval.get("per_worker_metrics")
    if isinstance(per_worker, dict) and per_worker and "error" not in per_worker:
        md.extend(
            [
                "### Per-Worker Retrieval Breakdown",
                "| Worker | Hit@5 | MRR | NDCG@5 |",
                "|--------|-------|-----|--------|",
            ]
        )
        for worker in sorted(per_worker.keys()):
            wm = (
                per_worker.get(worker, {})
                if isinstance(per_worker.get(worker), dict)
                else {}
            )
            # MRR/NDCG are continuous per-query scores, not binary hit/miss --
            # as_fraction=False avoids fabricating a numerator (only Hit@5 is
            # a true proportion; see format_rate_with_n's docstring).
            md.append(
                f"| {worker} | {format_rate_with_n(wm.get('hit_rate_at_5', {}))} "
                f"| {format_rate_with_n(wm.get('mrr', {}), as_fraction=False)} "
                f"| {format_rate_with_n(wm.get('ndcg_at_5', {}), as_fraction=False)} |"
            )
        md.append("")

    # Per-category deterministic rollups
    per_category = results.get("per_category", {})
    if isinstance(per_category, dict) and per_category and "error" not in per_category:
        md.extend(
            [
                "### Per-Category Summary",
                "| Category | N | Hit@5 | Citation Acc. | Tool Exact | Mean Cost/Query |",
                "|----------|---|-------|---------------|------------|-----------------|",
            ]
        )

        for category in sorted(per_category.keys()):
            cat = per_category.get(category, {})
            n_total = cat.get("n_total", 0)
            hit = safe_get_rate(
                cat, "retrieval", "custom_metrics", "hit_rate_at_5", "mean"
            )
            # R2: Use domain citation accuracy (covers enforcement IDs, not just OSHA stds)
            c_acc = safe_get_rate(cat, "domain", "citation_accuracy", "mean")
            tool_exact = safe_get_rate(
                cat, "orchestration", "tool_call_accuracy", "exact_match_rate"
            )
            mean_cost = safe_get_rate(
                cat, "orchestration", "cost", "mean_per_query_usd"
            )
            hit_str = format_metric(hit) if hit is not None else "N/A"
            c_acc_str = format_metric(c_acc) if c_acc is not None else "N/A"
            tool_str = format_metric(tool_exact) if tool_exact is not None else "N/A"
            cost_str = format_currency(mean_cost) if mean_cost is not None else "N/A"
            md.append(
                f"| {category} | {n_total} | {hit_str} | {c_acc_str} | {tool_str} | {cost_str} |"
            )

        md.append("")

    return "\n".join(md)


def generate_json_results(data: Dict[str, Any], run_id: str) -> Path:
    """Save full results containing dataset input, pipeline outputs, and metrics."""
    out_path = EVAL_RESULTS_DIR / f"{run_id}_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Helper to serialize AgentState or Langchain elements
    def default_serializer(obj):
        if hasattr(obj, "dict"):
            return obj.dict()
        if hasattr(obj, "to_json"):
            return obj.to_json()
        try:
            return list(obj)
        except Exception:
            return str(obj)

    with open(out_path, "w") as f:
        json.dump(data, f, indent=2, default=default_serializer)

    return out_path
