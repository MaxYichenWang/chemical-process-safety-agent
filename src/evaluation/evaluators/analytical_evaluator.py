"""
Analytical evaluator for Layer 4: Statistical Aggregation metrics.

Evaluates SQL execution success, SQL correctness, numerical fidelity,
hybrid synthesis quality, and SQL error taxonomy.
"""

import logging
import re
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


# Comma-grouped form FIRST, then an uncapped digit run.
#
# The previous pattern led with ``\d{1,3}(?:,\d{3})*``. Python ``re`` alternation
# takes the first alternative that matches at a position, not the longest, so any
# unformatted number with 4+ integer digits was split into a 3-digit prefix plus
# a leftover: "3110.27" -> [311.0, 0.27], "81796" -> [817.0, 96.0]. SQL result
# tables in ``context_gathered`` are rendered without thousands separators while
# good prose answers use them, so the better-formatted the answer, the worse it
# scored -- reported numerical fidelity was measuring the regex, not the model
# (see ARCHITECTURE.md).
#
# Ordering the comma-grouped alternative first keeps "81,796" parsing as one
# number, and requiring at least one comma group (``+`` not ``*``) stops it from
# claiming the leading 3 digits of an unformatted run.
_NUMBER_RE = re.compile(
    r"(?<![a-zA-Z0-9_])"
    r"(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)"
    r"(?:\s*(million|billion|thousand|k|m|b))?"
    r"(?![a-zA-Z0-9_])",
    re.IGNORECASE,
)

_LIST_MARKER_RE = re.compile(
    r"(?:^|\n)\s*(?:#{1,6}\s*)?(?:\*{1,2})?(\d+)[.)](?:\*{1,2})?\s+",
    re.MULTILINE,
)


def _extract_numbers(text: str, strip_list_markers: bool = False) -> List[float]:
    """Extract all numeric values from text, supporting scientific notation and units."""
    if not text:
        return []
    if strip_list_markers:
        text = _LIST_MARKER_RE.sub(r"\n", text)
    results = []
    for m in _NUMBER_RE.finditer(text):
        raw_num = m.group(1).replace(",", "")
        unit = (m.group(2) or "").lower()
        try:
            val = float(raw_num)
            if unit in ("million", "m"):
                val *= 1_000_000
            elif unit in ("billion", "b"):
                val *= 1_000_000_000
            elif unit in ("thousand", "k"):
                val *= 1_000
            results.append(val)
        except ValueError:
            continue
    return results


def _number_in_context(
    resp_num: float, ctx_numbers: Set[float], rel_tol: float = 0.015
) -> bool:
    """Check if resp_num matches any context number directly or within rounding tolerance."""
    if resp_num in ctx_numbers:
        return True
    for c in ctx_numbers:
        if c == 0:
            if abs(resp_num) < 1e-6:
                return True
        elif abs(resp_num - c) / abs(c) <= rel_tol:
            return True
    return False


def _is_analytical_entry(entry: Dict[str, Any]) -> bool:
    """Check if an evaluation entry is a statistical/analytical query."""
    agents = entry.get("expected_agents", [])
    category = entry.get("category", "")
    return (
        "data_analyst" in agents
        or category.startswith("statistical_")
        or category == "hybrid_analytical"
    )


def _get_analytical_retrieval_detail(result: Dict[str, Any]) -> Optional[Dict]:
    """Extract the analytical retrieval detail from pipeline results."""
    for detail in result.get("retrieval_details", []):
        if isinstance(detail, dict) and detail.get("collection") == "analytical":
            return detail
    return None


def _match_row_cells(gt_row: Dict, act_row: Dict) -> int:
    """Match cells between two rows, supporting exact keys, positional aliases, and value alignment."""
    if not gt_row or not act_row:
        return 0
    gt_items = list(gt_row.items())
    act_items = list(act_row.items())

    # 1. Exact key match (fast path)
    if all(k in act_row for k, _ in gt_items):
        return sum(1 for k, v in gt_items if _values_match(v, act_row[k]))

    # 2. Positional alignment (common for alias variations like AS count vs AS frequency)
    matched_pos = sum(
        1
        for (_, gt_val), (_, act_val) in zip(gt_items, act_items)
        if _values_match(gt_val, act_val)
    )
    if matched_pos == len(gt_items):
        return matched_pos

    # 3. Maximum bipartite match across columns
    used_act = set()
    matched_bipartite = 0
    for _, gt_val in gt_items:
        for idx, (_, act_val) in enumerate(act_items):
            if idx not in used_act and _values_match(gt_val, act_val):
                used_act.add(idx)
                matched_bipartite += 1
                break

    # 4. Partial key match
    matched_keys = sum(
        1 for k, v in gt_items if k in act_row and _values_match(v, act_row[k])
    )
    return max(matched_pos, matched_bipartite, matched_keys)


def _compare_sample_values(
    gt_values: List[Dict], actual_values: Optional[List[Dict]]
) -> float:
    """Score how well actual result rows match ground-truth sample rows.

    Compares up to ``min(len(gt), len(actual))`` rows with tolerance for:
    - Float rounding (1e-6 relative tolerance).
    - String case differences.
    - Column alias variations (e.g. ``total_violations`` vs ``violation_count``).
    - Row ordering variations (e.g. chronological ASC vs reverse-chronological DESC).

    Returns a float in [0, 1] representing the fraction of cell matches.
    """
    if not actual_values or not gt_values:
        return 0.0

    n_rows = min(len(gt_values), len(actual_values))
    if n_rows == 0:
        return 0.0

    total_cells = sum(len(r) for r in gt_values[:n_rows])
    if total_cells == 0:
        return 0.0

    # 1. Forward positional row match
    matched_fwd = sum(
        _match_row_cells(gt_row, act_row)
        for gt_row, act_row in zip(gt_values[:n_rows], actual_values[:n_rows])
    )

    # 2. Reverse row match (e.g. ASC vs DESC for trends)
    matched_rev = sum(
        _match_row_cells(gt_row, act_row)
        for gt_row, act_row in zip(
            gt_values[:n_rows], reversed(actual_values[-n_rows:])
        )
    )

    # 3. Optimal row-by-row assignment (order-invariant)
    matched_opt = 0
    used_act_rows = set()
    for gt_row in gt_values[:n_rows]:
        best_r = 0
        best_idx = -1
        for idx, act_row in enumerate(actual_values):
            if idx not in used_act_rows:
                score_r = _match_row_cells(gt_row, act_row)
                if score_r > best_r:
                    best_r = score_r
                    best_idx = idx
        if best_idx >= 0:
            used_act_rows.add(best_idx)
            matched_opt += best_r

    best_matched = max(matched_fwd, matched_rev, matched_opt)
    return best_matched / total_cells if total_cells > 0 else 0.0


def _values_match(gt_val: Any, act_val: Any) -> bool:
    """Type-aware comparison: numeric with relative tolerance, else string."""
    # Try numeric comparison first
    try:
        gt_num = float(gt_val) if not isinstance(gt_val, (int, float)) else gt_val
        act_num = float(act_val) if not isinstance(act_val, (int, float)) else act_val
        if gt_num == 0:
            return abs(act_num) < 1e-6
        return abs(gt_num - act_num) / abs(gt_num) < 1e-6
    except (TypeError, ValueError):
        pass
    # Fall back to case-insensitive string comparison
    return str(gt_val).strip().lower() == str(act_val).strip().lower()


class AnalyticalEvaluator:
    """Evaluator for Layer 4 Statistical Aggregation metrics."""

    def evaluate_sql_execution_success(
        self, dataset: List[Dict], results: List[Dict]
    ) -> Dict[str, Any]:
        """Fraction of analytical queries where SQL executed without error."""
        successes = []

        for entry, result in zip(dataset, results):
            if not _is_analytical_entry(entry):
                continue

            detail = _get_analytical_retrieval_detail(result)
            if detail is None:
                successes.append(0.0)
                continue

            # A successful execution has row_count >= 0 and no error key
            has_error = "error" in str(detail.get("sql_executed", "")).lower()
            successes.append(0.0 if has_error else 1.0)

        if not successes:
            return {"mean": 0.0, "ci_lower": 0.0, "ci_upper": 0.0, "n": 0}

        from src.evaluation.utils import bootstrap_ci

        mean_val = sum(successes) / len(successes)
        if len(successes) > 1:
            ci_lower, ci_upper = bootstrap_ci(successes)
        else:
            ci_lower, ci_upper = mean_val, mean_val

        return {
            "mean": mean_val,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "n": len(successes),
        }

    def evaluate_sql_self_correction(
        self, dataset: List[Dict], results: List[Dict]
    ) -> Dict[str, Any]:
        """split SQL success into first-attempt vs post-self-correction.

        Reporting a single "SQL success" number hides the agentic story: a first
        try that raises a DuckDB error and is then repaired counts identically to
        a query that never failed. This metric separates the two so a low
        first-pass rate paired with a high recovered rate reads as *effective
        self-correction* rather than an unqualified failure.

        Returns, over analytical entries that carry self-correction
        instrumentation (``first_attempt_error`` present in the retrieval detail):
        - ``first_attempt_success_rate``: fraction whose *first* SQL executed
          cleanly.
        - ``post_correction_success_rate``: fraction whose *final* SQL executed
          cleanly (after any retry).
        - ``recovery_rate``: of the first-attempt failures, the fraction that a
          retry recovered.
        - counts: ``n``, ``n_first_attempt_errors``, ``n_recovered``,
          ``n_unrecovered``, ``n_retried``.
        """
        first_ok: List[float] = []
        final_ok: List[float] = []
        n_first_errors = 0
        n_recovered = 0
        n_unrecovered = 0
        n_retried = 0

        for entry, result in zip(dataset, results):
            if not _is_analytical_entry(entry):
                continue
            detail = _get_analytical_retrieval_detail(result)
            # Only score entries produced by an instrumented worker; legacy runs
            # without the fields are skipped rather than counted as passing.
            if detail is None or "first_attempt_error" not in detail:
                continue

            first_error = bool(detail.get("first_attempt_error"))
            final_error = bool(
                detail.get(
                    "final_error",
                    "error" in str(detail.get("sql_executed", "")).lower(),
                )
            )
            if detail.get("retry_triggered"):
                n_retried += 1

            first_ok.append(0.0 if first_error else 1.0)
            final_ok.append(0.0 if final_error else 1.0)

            if first_error:
                n_first_errors += 1
                if final_error:
                    n_unrecovered += 1
                else:
                    n_recovered += 1

        n = len(first_ok)
        if n == 0:
            return {
                "first_attempt_success_rate": 0.0,
                "post_correction_success_rate": 0.0,
                "recovery_rate": 0.0,
                "n": 0,
                "n_first_attempt_errors": 0,
                "n_recovered": 0,
                "n_unrecovered": 0,
                "n_retried": 0,
                "status": "no_instrumented_entries",
            }

        from src.evaluation.utils import bootstrap_ci

        first_mean = sum(first_ok) / n
        final_mean = sum(final_ok) / n
        first_l, first_u = bootstrap_ci(first_ok) if n > 1 else (first_mean, first_mean)
        final_l, final_u = bootstrap_ci(final_ok) if n > 1 else (final_mean, final_mean)
        recovery_rate = (n_recovered / n_first_errors) if n_first_errors > 0 else 1.0

        return {
            "first_attempt_success_rate": first_mean,
            "first_attempt_ci_lower": first_l,
            "first_attempt_ci_upper": first_u,
            "post_correction_success_rate": final_mean,
            "post_correction_ci_lower": final_l,
            "post_correction_ci_upper": final_u,
            "recovery_rate": recovery_rate,
            "n": n,
            "n_first_attempt_errors": n_first_errors,
            "n_recovered": n_recovered,
            "n_unrecovered": n_unrecovered,
            "n_retried": n_retried,
        }

    def evaluate_sql_correctness(
        self, dataset: List[Dict], results: List[Dict]
    ) -> Dict[str, Any]:
        """Compare generated SQL output against ground truth results.

        Hybrid scoring (Q3 Option D):
        - If ground truth has ``sample_values`` (list of row-dicts for the first
          N rows), score is a weighted combination: 40% row_count match + 60%
          value match.  This catches both gross shape errors *and* subtle
          wrong-WHERE-clause bugs.
        - If ground truth only has ``row_count``, falls back to row_count-only
          comparison (backward-compatible).
        - If ground truth is empty / missing, the entry is skipped.
        """
        scores: List[float] = []

        for entry, result in zip(dataset, results):
            if not _is_analytical_entry(entry):
                continue
            gt = entry.get("ground_truth_result")
            if not gt:
                continue

            detail = _get_analytical_retrieval_detail(result)

            if detail is None:
                scores.append(0.0)
                continue

            # --- Row-count sub-score -----------------------------------
            gt_row_count = gt.get("row_count", -1)
            actual_row_count = detail.get("row_count", -2)
            row_count_ok = float(gt_row_count == actual_row_count)

            # --- Value sub-score (when ground truth provides samples) ---
            gt_values = gt.get("sample_values")  # list[dict] | None
            if gt_values:
                actual_values = detail.get("sample_values")  # list[dict] | None
                value_score = _compare_sample_values(gt_values, actual_values)
                # Weighted hybrid: 40% shape, 60% values
                scores.append(0.4 * row_count_ok + 0.6 * value_score)
            else:
                # Fallback: row_count-only (legacy ground truth)
                scores.append(row_count_ok)

        if not scores:
            return {"mean": 0.0, "ci_lower": 0.0, "ci_upper": 0.0, "n": 0}

        from src.evaluation.utils import bootstrap_ci

        mean_val = sum(scores) / len(scores)
        if len(scores) > 1:
            ci_lower, ci_upper = bootstrap_ci(scores)
        else:
            ci_lower, ci_upper = mean_val, mean_val

        return {
            "mean": mean_val,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "n": len(scores),
        }

    def evaluate_numerical_fidelity(
        self, dataset: List[Dict], results: List[Dict]
    ) -> Dict[str, Any]:
        """Check numbers in final_response match gathered context evidence."""
        fidelity_scores = []

        for entry, result in zip(dataset, results):
            if not _is_analytical_entry(entry):
                continue

            response = result.get("final_response", "")
            context = result.get("context_gathered", {})

            # Get context text from all gathered documents. Supports both shapes:
            # - dict[tool_name -> list[str]]: raw AgentState shape (legacy tests)
            # - list[doc-dict]: normalized E2E shape (production runs).
            context_text = ""
            if isinstance(context, dict):
                for k, items in context.items():
                    if isinstance(items, list):
                        for item in items:
                            if isinstance(item, str):
                                context_text += " " + item
            elif isinstance(context, list):
                for doc in context:
                    if isinstance(doc, dict):
                        context_text += " " + doc.get("text", "")

            response_numbers = _extract_numbers(response, strip_list_markers=True)
            context_numbers = set(
                _extract_numbers(context_text, strip_list_markers=False)
            )

            if not response_numbers:
                continue

            matched = sum(
                1
                for n in response_numbers
                if _number_in_context(n, context_numbers, rel_tol=0.015)
            )
            fidelity_scores.append(matched / len(response_numbers))

        if not fidelity_scores:
            return {"mean": 0.0, "ci_lower": 0.0, "ci_upper": 0.0, "n": 0}

        from src.evaluation.utils import bootstrap_ci

        mean_val = sum(fidelity_scores) / len(fidelity_scores)
        if len(fidelity_scores) > 1:
            ci_lower, ci_upper = bootstrap_ci(fidelity_scores)
        else:
            ci_lower, ci_upper = mean_val, mean_val

        return {
            "mean": mean_val,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "n": len(fidelity_scores),
        }

    def evaluate_hybrid_synthesis(
        self, dataset: List[Dict], results: List[Dict]
    ) -> Dict[str, Any]:
        """For hybrid queries: assess integration of statistical + qualitative evidence."""
        scores = []

        for entry, result in zip(dataset, results):
            if entry.get("category") != "hybrid_analytical":
                continue

            response = result.get("final_response", "")
            if not response:
                scores.append(0.0)
                continue

            # Check for statistical content
            has_stats = bool(_extract_numbers(response))

            # Check for qualitative content (non-numeric text)
            has_qualitative = (
                len(response) > 50
                and not response.strip().replace(",", "").replace(".", "").isdigit()
            )

            if has_stats and has_qualitative:
                scores.append(1.0)
            elif has_stats or has_qualitative:
                scores.append(0.5)
            else:
                scores.append(0.0)

        if not scores:
            return {"mean": 0.0, "ci_lower": 0.0, "ci_upper": 0.0, "n": 0}

        from src.evaluation.utils import bootstrap_ci

        mean_val = sum(scores) / len(scores)
        if len(scores) > 1:
            ci_lower, ci_upper = bootstrap_ci(scores)
        else:
            ci_lower, ci_upper = mean_val, mean_val

        return {
            "mean": mean_val,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "n": len(scores),
        }

    def evaluate_sql_error_taxonomy(self, results: List[Dict]) -> Dict[str, int]:
        """Classify SQL generation failures by error type."""
        taxonomy: Dict[str, int] = {
            "syntax_error": 0,
            "wrong_column": 0,
            "wrong_table": 0,
            "wrong_aggregation": 0,
            "wrong_filter": 0,
            "timeout": 0,
            "join_error": 0,
            "other": 0,
        }

        for result in results:
            detail = _get_analytical_retrieval_detail(result)
            if detail is None:
                continue

            sql_executed = str(detail.get("sql_executed", ""))
            if "error" not in sql_executed.lower():
                continue

            error_lower = sql_executed.lower()
            if "timeout" in error_lower:
                taxonomy["timeout"] += 1
            elif "column" in error_lower or "not found" in error_lower:
                taxonomy["wrong_column"] += 1
            elif "table" in error_lower or "relation" in error_lower:
                taxonomy["wrong_table"] += 1
            elif "syntax" in error_lower or "parse" in error_lower:
                taxonomy["syntax_error"] += 1
            elif "join" in error_lower:
                taxonomy["join_error"] += 1
            else:
                taxonomy["other"] += 1

        return taxonomy

    def evaluate(self, dataset: List[Dict], results: List[Dict]) -> Dict[str, Any]:
        """Run all Layer 4 metrics, each isolated from its siblings' failures."""
        from src.evaluation.utils import guarded

        return {
            "sql_execution_success": guarded(
                "analytical.sql_execution_success",
                self.evaluate_sql_execution_success,
                dataset,
                results,
            ),
            "sql_self_correction": guarded(
                "analytical.sql_self_correction",
                self.evaluate_sql_self_correction,
                dataset,
                results,
            ),
            "sql_correctness": guarded(
                "analytical.sql_correctness",
                self.evaluate_sql_correctness,
                dataset,
                results,
            ),
            "numerical_fidelity": guarded(
                "analytical.numerical_fidelity",
                self.evaluate_numerical_fidelity,
                dataset,
                results,
            ),
            "hybrid_synthesis": guarded(
                "analytical.hybrid_synthesis",
                self.evaluate_hybrid_synthesis,
                dataset,
                results,
            ),
            "sql_error_taxonomy": guarded(
                "analytical.sql_error_taxonomy",
                self.evaluate_sql_error_taxonomy,
                results,
            ),
        }
