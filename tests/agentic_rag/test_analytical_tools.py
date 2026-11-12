"""Tests for src.agentic_rag.analytical_tools.

Uses real DuckDB in-memory connections with small synthetic Parquet/CSV fixtures.
"""


import json
import pandas as pd
import pytest

from src.agentic_rag.analytical_tools import (
    execute_analytical_query,
    get_analytical_db,
    get_join_relationships,
    get_registered_views,
    get_schema_description,
    reset_db,
    validate_data_availability,
    validate_sql,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_data_dir(tmp_path):
    """Create small Parquet and CSV files for testing."""
    # Create a small Parquet file
    df_violations = pd.DataFrame({
        "activity_nr": [1, 2, 3, 4, 5],
        "standard_number": ["1910.119", "1910.147", "1910.119", "1910.146", "1910.119"],
        "violation_type": ["Serious", "Willful", "Serious", "Other", "Repeat"],
        "penalty_amount": [10000.0, 50000.0, 15000.0, 5000.0, 70000.0],
    })
    violations_path = tmp_path / "violations.parquet"
    df_violations.to_parquet(violations_path)

    # Create a small CSV file
    df_incidents = pd.DataFrame({
        "incident_name": ["Incident A", "Incident B"],
        "report_name": ["Report A", "Report B"],
        "num_sections": [10, 15],
        "num_pages": [50, 75],
    })
    incidents_path = tmp_path / "incidents.csv"
    df_incidents.to_csv(incidents_path, index=False)

    # Create a small csb_incident_meta CSV file
    df_meta = pd.DataFrame({
        "incident_name": ["Incident A", "Incident B"],
        "accident_occurred_on": ["2015-06-01", "2018-09-15"],
        "location": ["Houston, TX", "Geismar, LA"],
        "accident_type": ["Explosion", "Fire / Toxic Release"],
        "final_report_released_on": ["2017-01-01", "2020-03-01"],
        "incident_year": [2015, 2018],
        "num_pages": [50, 75],
        "num_sections": [10, 15],
    })
    meta_path = tmp_path / "csb_incident_meta.csv"
    df_meta.to_csv(meta_path, index=False)

    return tmp_path, violations_path, incidents_path, meta_path


@pytest.fixture()
def test_views(tmp_data_dir):
    """Return view definitions dict pointing to synthetic test files."""
    tmp_path, violations_path, incidents_path, meta_path = tmp_data_dir
    return {
        "violations": violations_path,
        "csb_incidents": incidents_path,
        "csb_incident_meta": meta_path,
    }


@pytest.fixture()
def test_conn(test_views):
    """Return a DuckDB connection with test views registered."""
    conn = get_analytical_db(view_definitions=test_views)
    yield conn
    conn.close()


# ---------------------------------------------------------------------------
# get_analytical_db
# ---------------------------------------------------------------------------


class TestGetAnalyticalDb:

    def test_creates_views(self, test_conn, test_views):
        """Views are created for all definitions."""
        for view_name in test_views:
            result = test_conn.execute(f"SELECT COUNT(*) FROM {view_name}").fetchone()
            assert result[0] > 0

    def test_handles_missing_files(self, tmp_path):
        """Gracefully skips missing data files instead of raising."""
        bad_views = {"missing_table": tmp_path / "nonexistent.parquet"}
        conn = get_analytical_db(view_definitions=bad_views)
        # View should not be created
        result = conn.execute("SELECT * FROM information_schema.tables WHERE table_name = 'missing_table'").fetchall()
        assert len(result) == 0
        conn.close()

    def test_parquet_view_queryable(self, test_conn):
        """Parquet views return correct row count."""
        count = test_conn.execute("SELECT COUNT(*) FROM violations").fetchone()[0]
        assert count == 5

    def test_csv_view_queryable(self, test_conn):
        """CSV views return correct row count."""
        count = test_conn.execute("SELECT COUNT(*) FROM csb_incidents").fetchone()[0]
        assert count == 2

    def test_csb_incident_meta_view_queryable(self, test_conn):
        """csb_incident_meta view returns correct row count and fields."""
        res = test_conn.execute(
            "SELECT incident_name, accident_occurred_on, location, accident_type, incident_year FROM csb_incident_meta"
        ).fetchall()
        assert len(res) == 2
        assert res[0][0] == "Incident A"
        assert res[0][4] == 2015

    def test_csb_incidents_meta_join(self, test_conn):
        """csb_incidents can join with csb_incident_meta on incident_name."""
        res = test_conn.execute("""
            SELECT i.incident_name, i.report_name, m.accident_occurred_on, m.incident_year
            FROM csb_incidents i
            JOIN csb_incident_meta m ON i.incident_name = m.incident_name
            WHERE m.incident_year >= 2018
        """).fetchall()
        assert len(res) == 1
        assert res[0][0] == "Incident B"
        assert res[0][3] == 2018


# ---------------------------------------------------------------------------
# get_schema_description
# ---------------------------------------------------------------------------


class TestGetSchemaDescription:

    def test_format(self, test_conn, test_views):
        """Schema description is a non-empty string with expected content."""
        desc = get_schema_description(test_conn, view_names=list(test_views.keys()))
        assert isinstance(desc, str)
        assert len(desc) > 0

    def test_includes_all_tables(self, test_conn, test_views):
        """Schema description mentions every registered view."""
        desc = get_schema_description(test_conn, view_names=list(test_views.keys()))
        for view_name in test_views:
            assert view_name in desc

    def test_includes_row_counts(self, test_conn, test_views):
        """Schema description includes row counts."""
        desc = get_schema_description(test_conn, view_names=list(test_views.keys()))
        assert "5" in desc  # violations has 5 rows
        assert "2" in desc  # csb_incidents has 2 rows

    def test_includes_column_names(self, test_conn, test_views):
        """Schema description includes column names."""
        desc = get_schema_description(test_conn, view_names=list(test_views.keys()))
        assert "activity_nr" in desc
        assert "penalty_amount" in desc


# ---------------------------------------------------------------------------
# get_join_relationships
# ---------------------------------------------------------------------------


class TestGetJoinRelationships:

    def test_content(self):
        """Join relationships string includes key relationships."""
        rels = get_join_relationships()
        assert "activity_nr" in rels
        assert "inspections" in rels
        assert "violations" in rels

    def test_is_string(self):
        assert isinstance(get_join_relationships(), str)


# ---------------------------------------------------------------------------
# validate_sql
# ---------------------------------------------------------------------------


class TestValidateSql:

    def test_allows_select(self):
        assert validate_sql("SELECT COUNT(*) FROM violations") == (True, "OK")

    def test_allows_complex_select(self):
        sql = """
            SELECT v.standard_number, COUNT(*) as cnt
            FROM violations v
            JOIN inspections i ON v.activity_nr = i.activity_nr
            WHERE i.site_state = 'TX'
            GROUP BY v.standard_number
            ORDER BY cnt DESC
            LIMIT 10
        """
        assert validate_sql(sql) == (True, "OK")

    def test_rejects_insert(self):
        valid, reason = validate_sql("INSERT INTO violations VALUES (1, '1910.119', 'Serious', 1000)")
        assert not valid
        assert "INSERT" in reason

    def test_rejects_update(self):
        valid, reason = validate_sql("UPDATE violations SET penalty_amount = 0")
        assert not valid
        assert "UPDATE" in reason

    def test_rejects_delete(self):
        valid, reason = validate_sql("DELETE FROM violations WHERE activity_nr = 1")
        assert not valid
        assert "DELETE" in reason

    def test_rejects_drop(self):
        valid, reason = validate_sql("DROP TABLE violations")
        assert not valid
        assert "DROP" in reason

    def test_rejects_create(self):
        valid, reason = validate_sql("CREATE TABLE test (id INT)")
        assert not valid
        assert "CREATE" in reason

    def test_rejects_alter(self):
        valid, reason = validate_sql("ALTER TABLE violations ADD COLUMN x INT")
        assert not valid
        assert "ALTER" in reason

    def test_rejects_truncate(self):
        valid, reason = validate_sql("TRUNCATE TABLE violations")
        assert not valid
        assert "TRUNCATE" in reason

    def test_handles_empty_string(self):
        valid, reason = validate_sql("")
        assert not valid
        assert "Empty" in reason

    def test_handles_whitespace(self):
        valid, reason = validate_sql("   ")
        assert not valid
        assert "Empty" in reason


# ---------------------------------------------------------------------------
# execute_analytical_query
# ---------------------------------------------------------------------------


class TestExecuteAnalyticalQuery:

    def test_valid_query(self, test_conn):
        """Valid SELECT returns markdown and metadata."""
        result, meta = execute_analytical_query(test_conn, "SELECT COUNT(*) as cnt FROM violations")
        assert "5" in result
        assert meta["row_count"] == 1
        assert "cnt" in meta["columns"]

    def test_rejects_forbidden(self, test_conn):
        """Forbidden SQL is rejected without execution."""
        result, meta = execute_analytical_query(test_conn, "DROP TABLE violations")
        assert "rejected" in result.lower()
        assert "error" in meta

    def test_injects_limit(self, test_conn):
        """LIMIT is injected when not present."""
        result, meta = execute_analytical_query(
            test_conn, "SELECT * FROM violations", max_rows=3
        )
        assert meta["row_count"] <= 3
        assert "LIMIT" in meta.get("sql_executed", "")

    def test_preserves_existing_limit(self, test_conn):
        """Existing LIMIT is not overridden."""
        result, meta = execute_analytical_query(
            test_conn, "SELECT * FROM violations LIMIT 2", max_rows=100
        )
        assert meta["row_count"] == 2

    def test_handles_execution_error(self, test_conn):
        """SQL execution error returns error message."""
        result, meta = execute_analytical_query(
            test_conn, "SELECT * FROM nonexistent_table"
        )
        assert "error" in result.lower() or "error" in meta

    def test_returns_metadata(self, test_conn):
        """Metadata includes expected fields."""
        result, meta = execute_analytical_query(
            test_conn, "SELECT activity_nr, penalty_amount FROM violations LIMIT 2"
        )
        assert "row_count" in meta
        assert "columns" in meta
        assert "sql_executed" in meta
        assert "truncated" in meta
        assert meta["row_count"] == 2
        assert set(meta["columns"]) == {"activity_nr", "penalty_amount"}

    def test_truncation_flag(self, test_conn):
        """Truncation flag is set when result reaches max_rows."""
        result, meta = execute_analytical_query(
            test_conn, "SELECT * FROM violations", max_rows=3
        )
        assert meta["truncated"] is True

    def test_no_truncation(self, test_conn):
        """Truncation flag is False when result is under max_rows."""
        result, meta = execute_analytical_query(
            test_conn, "SELECT * FROM violations", max_rows=100
        )
        assert meta["truncated"] is False

    def test_aggregation_query(self, test_conn):
        """Aggregation queries work correctly."""
        result, meta = execute_analytical_query(
            test_conn,
            "SELECT violation_type, COUNT(*) as cnt FROM violations GROUP BY violation_type ORDER BY cnt DESC",
        )
        assert meta["row_count"] > 0
        assert "Serious" in result

    def test_empty_string_rejected(self, test_conn):
        """Empty SQL is rejected."""
        result, meta = execute_analytical_query(test_conn, "")
        assert "rejected" in result.lower()


# ---------------------------------------------------------------------------
# get_registered_views
# ---------------------------------------------------------------------------


class TestGetRegisteredViews:

    def test_tracks_registered_views(self, test_conn, test_views):
        """Registered views match the successfully created views."""
        registered = get_registered_views()
        assert registered == set(test_views.keys())

    def test_empty_after_reset(self):
        """Registered views are empty after reset_db()."""
        reset_db()
        assert get_registered_views() == set()

    def test_skips_missing_in_registered(self, tmp_path):
        """Missing files are not included in registered views."""
        df = pd.DataFrame({"x": [1]})
        existing_path = tmp_path / "exists.parquet"
        df.to_parquet(existing_path)

        views = {
            "exists": existing_path,
            "missing": tmp_path / "nope.parquet",
        }
        conn = get_analytical_db(view_definitions=views)
        registered = get_registered_views()
        assert "exists" in registered
        assert "missing" not in registered
        conn.close()


# ---------------------------------------------------------------------------
# validate_data_availability
# ---------------------------------------------------------------------------


class TestValidateDataAvailability:

    def test_full_availability(self, tmp_data_dir):
        """All files present returns 'full' status."""
        tmp_path, violations_path, incidents_path, meta_path = tmp_data_dir
        views = {
            "violations": violations_path,
            "csb_incidents": incidents_path,
            "csb_incident_meta": meta_path,
        }
        report = validate_data_availability(view_definitions=views)
        assert report["status"] == "full"
        assert len(report["available"]) == 3
        assert len(report["missing"]) == 0

    def test_partial_availability(self, tmp_data_dir):
        """Some files missing returns 'partial' status."""
        tmp_path, violations_path, *_ = tmp_data_dir
        views = {
            "violations": violations_path,
            "missing": tmp_path / "nope.parquet",
        }
        report = validate_data_availability(view_definitions=views)
        assert report["status"] == "partial"
        assert "violations" in report["available"]
        assert "missing" in report["missing"]

    def test_no_availability(self, tmp_path):
        """No files present returns 'none' status."""
        views = {
            "a": tmp_path / "nope1.parquet",
            "b": tmp_path / "nope2.csv",
        }
        report = validate_data_availability(view_definitions=views)
        assert report["status"] == "none"
        assert len(report["available"]) == 0
        assert len(report["missing"]) == 2


# ---------------------------------------------------------------------------
# P3.3 — get_schema_description categorical column values
# ---------------------------------------------------------------------------


class TestSchemaDescriptionCategoricalValues:

    def test_categorical_values_included_for_known_views(self, tmp_path):
        """get_schema_description includes distinct values for categorical columns."""
        import pandas as pd

        df = pd.DataFrame({
            "activity_nr": [1, 2, 3],
            "viol_type": ["W", "S", "R"],
            "gravity": ["1", "5", "10"],
        })
        violations_path = tmp_path / "violations.parquet"
        df.to_parquet(violations_path)
        views = {"violations": violations_path}

        conn = get_analytical_db(view_definitions=views)
        desc = get_schema_description(conn, view_names=["violations"])
        conn.close()
        reset_db()

        assert "viol_type" in desc
        assert "Categorical values" in desc
        # All three viol_type values should appear
        assert "W" in desc
        assert "S" in desc
        assert "R" in desc

    def test_non_categorical_view_no_extra_section(self, tmp_path):
        """Views not in the categorical map don't get an extra section."""
        import pandas as pd

        df = pd.DataFrame({"col_a": [1, 2], "col_b": ["x", "y"]})
        path = tmp_path / "custom.parquet"
        df.to_parquet(path)
        views = {"custom_view": path}

        conn = get_analytical_db(view_definitions=views)
        desc = get_schema_description(conn, view_names=["custom_view"])
        conn.close()
        reset_db()

        # No categorical section should be present for this unknown view
        assert "## custom_view" in desc
        assert "Categorical values" not in desc


# ---------------------------------------------------------------------------
# P3.3 — get_join_relationships note on accidents join key
# ---------------------------------------------------------------------------


def test_join_relationships_accidents_note():
    """Join docs should clarify the accidents join key is activity_nr."""
    rels = get_join_relationships()
    assert "accidents" in rels
    assert "activity_nr" in rels
    # Confirm the clarifying NOTE is present
    assert "NOTE" in rels and "accidents" in rels


class TestExecuteAnalyticalQueryCapping:

    def test_max_rows_capping(self, test_conn):
        """Result dataframe is sliced to max_rows even if SQL has a higher LIMIT."""
        sql = "SELECT * FROM violations LIMIT 100"
        markdown_table, metadata = execute_analytical_query(test_conn, sql, max_rows=2)
        assert metadata["row_count"] == 2
        assert metadata["truncated"] is True

    def test_character_truncation(self, test_conn, monkeypatch):
        """Markdown output is truncated if it exceeds character threshold."""
        import src.agentic_rag.analytical_tools as tools
        monkeypatch.setattr(tools, "MAX_ANALYTICAL_OUTPUT_CHARS", 50)
        sql = "SELECT * FROM violations"
        markdown_table, metadata = execute_analytical_query(test_conn, sql)
        assert "[Table output truncated to 50 characters]" in markdown_table
        assert metadata["truncated"] is True


class TestProhibitedRawColumns:

    def test_validate_sql_rejects_prohibited_raw_columns(self):
        """validate_sql rejects queries selecting raw text/blob/metadata columns."""
        for col in ["text", "abstract_text", "event_desc", "pdf_path", "url", "load_dt"]:
            sql = f"SELECT {col} FROM dataset"
            is_valid, reason = validate_sql(sql)
            assert is_valid is False
            assert f"Forbidden raw/metadata column '{col}'" in reason

    def test_get_analytical_db_filters_prohibited_raw_columns(self, tmp_path):
        """get_analytical_db excludes prohibited raw columns from registered views."""
        df = pd.DataFrame({
            "standard_number": ["1910.119"],
            "title": ["Process Safety Management"],
            "text": ["Very long raw regulation text body..."],
            "load_dt": ["2026-01-01"],
            "pdf_path": ["/path/to/report.pdf"],
        })
        parquet_path = tmp_path / "raw_test.parquet"
        df.to_parquet(parquet_path)

        conn = get_analytical_db(view_definitions={"test_regulations": parquet_path})
        cols = [c[0] for c in conn.execute("DESCRIBE test_regulations").fetchall()]
        conn.close()

        assert "standard_number" in cols
        assert "title" in cols
        assert "text" not in cols
        assert "load_dt" not in cols
        assert "pdf_path" not in cols




# ---------------------------------------------------------------------------
# sample_values emission
#
# `sql_correctness` scores 40% row-count match + 60% cell-value match, and the
# value half compares metadata["sample_values"] against the dataset's
# ground_truth_result.sample_values. That key was absent from every generation
# ever produced (0 of 46 archived runs), so the value component always scored
# 0.0 and the metric was capped at 0.40 by construction -- the published "35.6%
# SQL correctness" was 0.4 x (8/9 row-counts correct), not a statement about the
# SQL. These tests exist so the key cannot silently disappear again.
# ---------------------------------------------------------------------------


class TestSampleValuesEmission:
    def test_metadata_carries_sample_values_matching_ground_truth_shape(self, test_conn):
        _, meta = execute_analytical_query(
            test_conn, "SELECT COUNT(*) AS total_violations FROM violations"
        )
        assert "sample_values" in meta, "sql_correctness cannot score without this key"
        assert isinstance(meta["sample_values"], list)
        assert meta["sample_values"], "a successful query must report at least one row"
        row = meta["sample_values"][0]
        assert isinstance(row, dict)
        # Same shape as ground_truth_result.sample_values: column name -> value.
        assert "total_violations" in row

    def test_sample_values_are_json_serialisable(self, test_conn):
        """A generation that cannot be serialised is worse than one losing precision."""
        _, meta = execute_analytical_query(
            test_conn, "SELECT COUNT(*) AS cnt FROM violations"
        )
        json.dumps(meta["sample_values"])  # must not raise on numpy scalars

    def test_sample_values_score_against_matching_ground_truth(self, test_conn):
        from src.evaluation.evaluators.analytical_evaluator import _compare_sample_values

        _, meta = execute_analytical_query(
            test_conn, "SELECT COUNT(*) AS cnt FROM violations"
        )
        actual = meta["sample_values"]
        assert _compare_sample_values(actual, actual) == 1.0
        # The regression this guards: a missing actual side silently scores 0.
        assert _compare_sample_values(actual, None) == 0.0

    def test_sample_values_untruncated_by_display_formatting(self, test_conn):
        """Display truncates long text to 200 chars; scoring must not see that."""
        long_text = "x" * 400
        test_conn.execute(f"CREATE TABLE _lt AS SELECT '{long_text}' AS body")
        _, meta = execute_analytical_query(test_conn, "SELECT body FROM _lt")
        assert meta["sample_values"][0]["body"] == long_text
