"""
DuckDB analytical backend for the Statistical Aggregation Layer.

Provides schema registration, introspection, SQL validation, and guarded
execution over structured OSHA/CSB datasets stored as Parquet/CSV files.
"""

import logging
import re
import signal
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import duckdb
import sqlparse

from src.data_processing.config import (
    DATA_ROOT,
    SQL_MAX_RESULT_ROWS,
    SQL_QUERY_TIMEOUT_S,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dataset paths & prohibited column definition
# ---------------------------------------------------------------------------
_OSHA_ENF_DIR = DATA_ROOT / "processed" / "osha" / "enforcement"
_OSHA_REG_DIR = DATA_ROOT / "processed" / "osha" / "regulations"
_CSB_DIR = DATA_ROOT / "processed" / "csb_olmocr" / "csv"

_VIEW_DEFINITIONS: Dict[str, Path] = {
    "violations": _OSHA_ENF_DIR / "violations_struct.parquet",
    "inspections": _OSHA_ENF_DIR / "inspection.parquet",
    "accidents": _OSHA_ENF_DIR / "accidents_struct.parquet",
    "violation_events": _OSHA_ENF_DIR / "violation_event.parquet",
    "genduty_violations": _OSHA_ENF_DIR / "violation_genduty.parquet",
    "related_activities": _OSHA_ENF_DIR / "related_activity.parquet",
    "strategic_codes": _OSHA_ENF_DIR / "strategic_codes.parquet",
    "optional_info": _OSHA_ENF_DIR / "optional_info.parquet",
    "regulations": _OSHA_REG_DIR / "regulations_sections.parquet",
    "csb_incidents": _CSB_DIR / "final_reports_summary.csv",
    "csb_recommendations": _CSB_DIR / "recommendations_summary.csv",
    "csb_incident_meta": _CSB_DIR / "csb_incident_meta.csv",
    "csb_contributing_factors": _CSB_DIR / "csb_contributing_factors.csv",
}

_PROHIBITED_RAW_COLUMNS = frozenset(
    {
        # Text Blobs & Narratives (Must use vector retrieval workers instead)
        "text",
        "abstract_text",
        "event_desc",
        # Storage Paths & External URLs
        "pdf_path",
        "url",
        # Internal ETL & Scraping Metadata
        "load_dt",
        "scraped_at",
        "hash",
        "gpo_source",
        "extraction_date",
        "processing_date",
        "delete_flag",
    }
)

_FORBIDDEN_KEYWORDS = frozenset(
    {
        "INSERT",
        "UPDATE",
        "DELETE",
        "DROP",
        "CREATE",
        "ALTER",
        "TRUNCATE",
        "GRANT",
        "REVOKE",
    }
)

_JOIN_RELATIONSHIPS = """KEY JOIN RELATIONSHIPS:
- inspections.activity_nr = violations.activity_nr (inspection -> violations)
- inspections.activity_nr = accidents.activity_nr (inspection -> accidents)
- inspections.activity_nr = violation_events.activity_nr (inspection -> violation events)
- inspections.activity_nr = genduty_violations.activity_nr (inspection -> general duty violations)
- inspections.activity_nr = related_activities.activity_nr (inspection -> related activities)
- inspections.activity_nr = strategic_codes.activity_nr (inspection -> strategic codes)
- inspections.activity_nr = optional_info.activity_nr (inspection -> optional info)
- csb_incidents.incident_name = csb_incident_meta.incident_name (csb_incidents -> csb_incident_meta)
- csb_contributing_factors.incident_name = csb_incident_meta.incident_name (contributing factors -> incident meta)

CRITICAL TABLE ROLES & DATA GUIDANCE:
1. `csb_contributing_factors`:
   Columns: `incident_name`, `company_name`, `incident_year`, `naics_code`, `naics_325_flag`, `severity_group`, `operating_mode`, `factor_name`, `factor_rank_in_incident`, `is_multi_factor`.
   Use this table for:
   - Ranking chemical process safety contributing factors (Design, Preventive Maintenance, Safeguards/Controls, Operating Procedures, Hazard Awareness, Safety Culture, etc.).
   - Analyzing operational phases / operating modes (Normal Operations, Startup, Maintenance, Other).
   - Multi-factor breakdown counts by severity group.
2. `v_violations_parsed` (or `violations`):
   Pre-parsed view of violations with clean standard numbers and PSM paragraphs:
   - `standard_number`: Normalized standard identifier (e.g. '1910.119', '1910.1200', '1910.134', '1910.147').
   - `psm_paragraph`: Lowercase paragraph letter for 1910.119 citations ('j', 'd', 'e', 'f', 'l', etc.).
   Join with `inspections` on `activity_nr` and filter on `(i.naics_code LIKE '325%' OR i.sic_code LIKE '28%')`.
3. `accidents`:
   NOTE: `accidents.event_keyword` contains occupational injury codes (e.g. 'STRUCK BY', 'PPE', 'PELVIS'), NOT chemical process safety failure modes or plant operating phases.
   Join `accidents` with `inspections` using `inspections.activity_nr = accidents.activity_nr`."""

# ---------------------------------------------------------------------------
# Module-level lazy singleton
# ---------------------------------------------------------------------------
_DB_CONN: Optional[duckdb.DuckDBPyConnection] = None
_SCHEMA_CACHE: Optional[str] = None
_REGISTERED_VIEWS: set = set()


def get_analytical_db(
    view_definitions: Optional[Dict[str, Path]] = None,
) -> duckdb.DuckDBPyConnection:
    """Create an in-memory DuckDB connection with views over all structured data.

    Gracefully skips files that do not exist, registering only available views.
    Filters out prohibited raw narrative, blob, and metadata columns.

    Args:
        view_definitions: Override the default view definitions (useful for testing).

    Returns:
        A DuckDB connection with available views registered.
    """
    global _REGISTERED_VIEWS
    views = view_definitions or _VIEW_DEFINITIONS
    conn = duckdb.connect(":memory:")
    registered = set()

    for view_name, file_path in views.items():
        path_str = str(file_path)
        if not file_path.exists():
            logger.warning(
                "Data file not found for view '%s': %s — skipping",
                view_name,
                path_str,
            )
            continue

        read_expr = (
            f"read_csv_auto('{path_str}')"
            if file_path.suffix == ".csv"
            else f"read_parquet('{path_str}')"
        )
        try:
            cols_desc = conn.execute(f"DESCRIBE SELECT * FROM {read_expr}").fetchall()
            valid_cols = [
                c[0] for c in cols_desc if c[0] not in _PROHIBITED_RAW_COLUMNS
            ]
            if valid_cols:
                select_cols = ", ".join(f'"{c}"' for c in valid_cols)
                conn.execute(
                    f"CREATE VIEW {view_name} AS SELECT {select_cols} FROM {read_expr}"
                )
            else:
                conn.execute(f"CREATE VIEW {view_name} AS SELECT * FROM {read_expr}")
        except Exception:
            conn.execute(f"CREATE VIEW {view_name} AS SELECT * FROM {read_expr}")

        registered.add(view_name)

    if "violations" in registered:
        try:
            conn.execute(
                """
                CREATE VIEW v_violations_parsed AS
                SELECT *,
                       CASE
                         WHEN REGEXP_MATCHES(standard, '^[0-9]{8}') THEN
                           SUBSTR(standard, 1, 4) || '.' || CAST(CAST(SUBSTR(standard, 5, 4) AS INTEGER) AS VARCHAR)
                         WHEN REGEXP_MATCHES(standard, '^[0-9]{4}[.][0-9]+') THEN
                           REGEXP_EXTRACT(standard, '^([0-9]{4}[.][0-9]+)', 1)
                         ELSE standard
                       END AS standard_number,
                       CASE
                         WHEN standard ILIKE '19100119%' THEN
                           LOWER(REGEXP_EXTRACT(standard, '^19100119\\s*([A-Za-z])', 1))
                         ELSE NULL
                       END AS psm_paragraph
                FROM violations
            """
            )
            registered.add("v_violations_parsed")
        except Exception as e:
            logger.warning("Failed to create v_violations_parsed: %s", e)

    _REGISTERED_VIEWS = registered
    return conn


def get_registered_views() -> set:
    """Return the set of view names successfully registered in the last get_analytical_db() call."""
    return set(_REGISTERED_VIEWS)


def get_db() -> duckdb.DuckDBPyConnection:
    """Return the module-level DuckDB connection, creating it on first access."""
    global _DB_CONN
    if _DB_CONN is None:
        _DB_CONN = get_analytical_db()
    return _DB_CONN


def reset_db() -> None:
    """Reset the module-level connection (useful for testing)."""
    global _DB_CONN, _SCHEMA_CACHE, _REGISTERED_VIEWS
    _DB_CONN = None
    _SCHEMA_CACHE = None
    _REGISTERED_VIEWS = set()


# ---------------------------------------------------------------------------
# Schema introspection
# ---------------------------------------------------------------------------


def get_schema_description(
    conn: duckdb.DuckDBPyConnection,
    view_names: Optional[list] = None,
) -> str:
    """Generate a natural-language schema description for LLM prompt injection.

    Args:
        conn: Active DuckDB connection.
        view_names: Specific views to describe. Defaults to all registered views.

    Returns:
        Formatted string with column names, types, row counts, and sample rows.
    """
    global _SCHEMA_CACHE
    if _SCHEMA_CACHE is not None and view_names is None:
        return _SCHEMA_CACHE

    names = view_names or sorted(_REGISTERED_VIEWS) or list(_VIEW_DEFINITIONS.keys())
    descriptions = []

    # Categorical columns to enumerate distinct values for (improves SQL generation)
    _CATEGORICAL_COLS: Dict[str, list] = {
        "violations": ["viol_type", "gravity"],
        "inspections": ["insp_type", "safety_hlth", "site_state"],
        "accidents": ["fatality", "event_keyword"],
        "csb_incident_meta": ["accident_type"],
    }

    for view_name in names:
        try:
            cols = conn.execute(f"DESCRIBE {view_name}").fetchall()
            col_desc = ", ".join(f"{name} ({dtype})" for name, dtype, *_ in cols)
            row_count = conn.execute(f"SELECT COUNT(*) FROM {view_name}").fetchone()[0]
            sample = (
                conn.execute(f"SELECT * FROM {view_name} LIMIT 3").fetchdf().to_string()
            )
            section_parts = [
                f"## {view_name} ({row_count:,} rows)",
                f"Columns: {col_desc}",
                f"Sample rows:\n{sample}",
            ]
            # Append distinct values for key categorical columns
            if view_name in _CATEGORICAL_COLS:
                cat_lines = []
                for col in _CATEGORICAL_COLS[view_name]:
                    try:
                        distinct = conn.execute(
                            f"SELECT DISTINCT {col} FROM {view_name} WHERE {col} IS NOT NULL LIMIT 20"
                        ).fetchall()
                        values = [str(row[0]) for row in distinct]
                        if values:
                            cat_lines.append(
                                f"  {col} distinct values: {', '.join(values)}"
                            )
                    except Exception:
                        pass
                if cat_lines:
                    section_parts.append("Categorical values:\n" + "\n".join(cat_lines))
            descriptions.append("\n".join(section_parts))
        except Exception as e:
            descriptions.append(f"## {view_name}\nError reading schema: {e}")

    result = "\n\n".join(descriptions)
    if view_names is None:
        _SCHEMA_CACHE = result
    return result


def get_join_relationships() -> str:
    """Return a static string documenting the key join relationships."""
    return _JOIN_RELATIONSHIPS


# ---------------------------------------------------------------------------
# SQL validation
# ---------------------------------------------------------------------------


def validate_sql(sql: str) -> Tuple[bool, str]:
    """Validate that a SQL query is read-only and safe to execute.

    Args:
        sql: SQL query string to validate.

    Returns:
        Tuple of (is_valid, reason). Returns (True, "OK") for valid queries.
    """
    if not sql or not sql.strip():
        return False, "Empty SQL query"

    parsed = sqlparse.parse(sql)
    for statement in parsed:
        for token in statement.flatten():
            if token.ttype in (
                sqlparse.tokens.Keyword,
                sqlparse.tokens.Keyword.DDL,
                sqlparse.tokens.Keyword.DML,
            ):
                if token.normalized.upper() in _FORBIDDEN_KEYWORDS:
                    return False, f"Forbidden SQL keyword: {token.normalized}"

    # Check for forbidden raw/metadata column references
    sql_lower = sql.lower()
    for col in _PROHIBITED_RAW_COLUMNS:
        if re.search(rf"\b{col}\b", sql_lower):
            return False, (
                f"Forbidden raw/metadata column '{col}'. "
                "Analytical SQL queries must select structured columns (e.g. standard_number, title, viol_type, penalty, fatality) or aggregate functions."
            )

    return True, "OK"


# ---------------------------------------------------------------------------
# Guarded execution
# ---------------------------------------------------------------------------


def _execute_with_timeout(
    conn: duckdb.DuckDBPyConnection,
    sql: str,
    timeout_s: int = SQL_QUERY_TIMEOUT_S,
):
    """Execute SQL with timeout protection.

    Uses SIGALRM on Unix; falls back to threading on other platforms.
    """
    try:
        # Unix: use SIGALRM for reliable timeout
        def handler(signum, frame):
            raise TimeoutError(f"Query exceeded {timeout_s}s timeout")

        old_handler = signal.signal(signal.SIGALRM, handler)
        signal.alarm(timeout_s)
        try:
            result = conn.execute(sql).fetchdf()
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
        return result
    except (AttributeError, ValueError):
        # Non-Unix fallback: use threading
        result_holder = [None]
        error_holder = [None]

        def target():
            try:
                result_holder[0] = conn.execute(sql).fetchdf()
            except Exception as e:
                error_holder[0] = e

        thread = threading.Thread(target=target)
        thread.start()
        thread.join(timeout=timeout_s)

        if thread.is_alive():
            raise TimeoutError(f"Query exceeded {timeout_s}s timeout")
        if error_holder[0]:
            raise error_holder[0]
        return result_holder[0]


MAX_ANALYTICAL_OUTPUT_CHARS = 50_000

#: Rows captured into ``metadata["sample_values"]`` for evaluation. Ground-truth
#: results store only the first few rows, so capturing more buys nothing and
#: would bloat every stored generation.
MAX_SAMPLE_VALUE_ROWS = 10


def _sample_values(result_df: Any) -> List[Dict[str, Any]]:
    """First rows of a query result as JSON-safe row dicts.

    Mirrors the shape of ``ground_truth_result.sample_values`` so the evaluator
    can compare them cell by cell. Values are coerced through ``.item()`` where
    numpy scalars appear, because ``json.dump`` cannot serialise those and a
    generation that fails to serialise is worse than one that loses precision.
    """
    rows: List[Dict[str, Any]] = []
    for record in result_df.head(MAX_SAMPLE_VALUE_ROWS).to_dict("records"):
        row: Dict[str, Any] = {}
        for key, value in record.items():
            item = getattr(value, "item", None)
            try:
                row[str(key)] = item() if callable(item) else value
            except Exception:  # noqa: BLE001 - never fail a query over formatting
                row[str(key)] = str(value)
        rows.append(row)
    return rows


def execute_analytical_query(
    conn: duckdb.DuckDBPyConnection,
    sql: str,
    max_rows: int = SQL_MAX_RESULT_ROWS,
    timeout_s: int = SQL_QUERY_TIMEOUT_S,
) -> Tuple[str, dict]:
    """Execute a validated SQL query and return formatted results with metadata.

    Args:
        conn: Active DuckDB connection.
        sql: SQL query to execute.
        max_rows: Maximum rows to return.
        timeout_s: Query timeout in seconds.

    Returns:
        Tuple of (formatted_markdown_table, metadata_dict).
    """
    is_valid, reason = validate_sql(sql)
    if not is_valid:
        return f"Query rejected: {reason}", {"error": reason, "row_count": 0}

    # Inject LIMIT if not present
    sql_upper = sql.upper().strip().rstrip(";")
    if "LIMIT" not in sql_upper:
        sql = f"{sql.rstrip(';')} LIMIT {max_rows}"

    try:
        result_df = _execute_with_timeout(conn, sql, timeout_s)
        total_rows_retrieved = len(result_df)
        if total_rows_retrieved > max_rows:
            result_df = result_df.head(max_rows)
        row_count = len(result_df)

        # Truncate individual text cells (e.g. raw regulation text bodies) to keep analytical tables concise
        display_df = result_df.copy()
        for col in display_df.columns:
            if display_df[col].dtype == "object" or str(
                display_df[col].dtype
            ).startswith("string"):
                display_df[col] = display_df[col].apply(
                    lambda val: (
                        (str(val)[:200] + "...")
                        if isinstance(val, str) and len(val) > 200
                        else val
                    )
                )

        markdown_table = display_df.to_markdown(index=False)
        was_char_truncated = False
        if len(markdown_table) > MAX_ANALYTICAL_OUTPUT_CHARS:
            markdown_table = (
                markdown_table[:MAX_ANALYTICAL_OUTPUT_CHARS]
                + f"\n\n[Table output truncated to {MAX_ANALYTICAL_OUTPUT_CHARS} characters]"
            )
            was_char_truncated = True

        metadata = {
            "row_count": row_count,
            "columns": list(result_df.columns),
            "sql_executed": sql,
            "truncated": total_rows_retrieved >= max_rows or was_char_truncated,
            # Structured result rows, mirroring the shape of a dataset entry's
            # ``ground_truth_result.sample_values``.
            #
            # Without this the evaluator's SQL-correctness metric is capped at
            # 0.40: it scores 40% row-count match + 60% cell-value match, and
            # ``_compare_sample_values`` returns 0.0 when the actual side is
            # absent. No run ever emitted this key, so the value component was
            # structurally unearnable and the headline "35.6% SQL correctness"
            # was 0.4 x (8/9 row-counts correct) -- a measurement artifact
            # rather than a statement about the generated SQL.
            #
            # Taken from ``result_df`` (untruncated cells), not ``display_df``,
            # so a 200-char display truncation cannot cause a spurious mismatch.
            "sample_values": _sample_values(result_df),
        }
        return markdown_table, metadata
    except TimeoutError as e:
        return f"Query timeout: {e}", {"error": str(e), "row_count": 0}
    except Exception as e:
        return f"Query execution error: {e}", {"error": str(e), "row_count": 0}


# ---------------------------------------------------------------------------
# Data availability validation
# ---------------------------------------------------------------------------


def validate_data_availability(
    view_definitions: Optional[Dict[str, Path]] = None,
) -> dict:
    """Check which analytical data files are present on disk.

    Args:
        view_definitions: Override the default view definitions (useful for testing).

    Returns:
        Dict with keys: available (list), missing (list), status ("full"|"partial"|"none").
    """
    views = view_definitions or _VIEW_DEFINITIONS
    available = []
    missing = []
    for name, path in views.items():
        if path.exists():
            available.append(name)
        else:
            missing.append(name)

    if not available:
        status = "none"
    elif not missing:
        status = "full"
    else:
        status = "partial"

    return {"available": available, "missing": missing, "status": status}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analytical tools utilities")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check data file availability for all view definitions",
    )
    args = parser.parse_args()

    if args.check:
        report = validate_data_availability()
        print(f"Status: {report['status'].upper()}")
        print(f"\nAvailable ({len(report['available'])}):")
        for name in report["available"]:
            print(f"  ✓ {name}")
        if report["missing"]:
            print(f"\nMissing ({len(report['missing'])}):")
            for name in report["missing"]:
                print(f"  ✗ {name}")
    else:
        parser.print_help()
