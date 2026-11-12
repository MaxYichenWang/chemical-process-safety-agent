"""Tests for data_analyst_agent in src.agentic_rag.workers.

Mocks LLM calls and DB connections to test the text-to-SQL pipeline.
"""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.agentic_rag.workers import _extract_sql_from_response, data_analyst_agent


@pytest.fixture(autouse=True)
def _force_supported_worker_params(monkeypatch):
    monkeypatch.setattr("src.agentic_rag.workers.WORKER_TEMPERATURE", 1.0)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_state(**overrides) -> dict:
    """Return a minimal AgentState dict."""
    base = {
        "query": "How many OSHA violations were issued in Texas?",
        "sub_tasks": ["Count violations in TX"],
        "active_agent": "data_analyst",
        "pending_agents": [],
        "planned_agents": [],
        "context_gathered": {},
        "draft_response": "",
        "final_response": "",
        "citations": [],
        "validation_score": 0.0,
        "critic_feedback": "",
        "retry_count": 0,
        "llm_calls": [],
        "retrieval_details": [],
        "worker_errors": [],
    }
    base.update(overrides)
    return base


@pytest.fixture()
def mock_test_conn(tmp_path):
    """Create a small DuckDB connection with test data."""
    import duckdb

    df = pd.DataFrame({
        "activity_nr": [1, 2, 3],
        "standard_number": ["1910.119", "1910.147", "1910.119"],
        "violation_type": ["Serious", "Willful", "Serious"],
        "penalty_amount": [10000.0, 50000.0, 15000.0],
    })
    parquet_path = tmp_path / "violations.parquet"
    df.to_parquet(parquet_path)

    conn = duckdb.connect(":memory:")
    conn.execute(
        f"CREATE VIEW violations AS SELECT * FROM read_parquet('{parquet_path}')"
    )
    return conn


def _mock_llm_response(content):
    """Create a mock LLM response object."""
    mock = MagicMock()
    mock.content = content
    mock.response_metadata = {
        "token_usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
        "model_name": "test-model",
        "finish_reason": "stop",
    }
    return mock


# ---------------------------------------------------------------------------
# _extract_sql_from_response
# ---------------------------------------------------------------------------


class TestExtractSqlFromResponse:

    def test_plain_sql(self):
        assert _extract_sql_from_response("SELECT 1") == "SELECT 1"

    def test_strips_code_fences(self):
        text = "Here is the SQL:\n```sql\nSELECT COUNT(*) FROM violations\n```"
        assert _extract_sql_from_response(text) == "SELECT COUNT(*) FROM violations"

    def test_strips_code_fences_no_language(self):
        text = "```\nSELECT 1\n```"
        assert _extract_sql_from_response(text) == "SELECT 1"

    def test_strips_whitespace(self):
        assert _extract_sql_from_response("  SELECT 1  ") == "SELECT 1"


# ---------------------------------------------------------------------------
# data_analyst_agent
# ---------------------------------------------------------------------------


class TestDataAnalystAgent:

    @patch("src.agentic_rag.workers.make_chat_model")
    @patch("src.agentic_rag.analytical_tools.get_db")
    @patch("src.agentic_rag.analytical_tools.get_registered_views")
    @patch("src.agentic_rag.analytical_tools.get_schema_description")
    @patch("src.agentic_rag.analytical_tools.get_join_relationships")
    @patch("src.agentic_rag.analytical_tools.execute_analytical_query")
    def test_returns_correct_keys(
        self, mock_exec, mock_join, mock_schema, mock_reg_views, mock_db, mock_llm_cls
    ):
        """Agent returns dict with required keys."""
        mock_db.return_value = MagicMock()
        mock_reg_views.return_value = {"violations"}
        mock_schema.return_value = "schema description"
        mock_join.return_value = "join relationships"
        mock_exec.return_value = ("| col |\n|---|\n| 1 |", {"row_count": 1, "columns": ["col"], "sql_executed": "SELECT 1", "truncated": False})

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = _mock_llm_response("SELECT 1")
        mock_llm_cls.return_value = mock_llm

        result = data_analyst_agent(_make_state())
        assert "context_gathered" in result
        assert "citations" in result
        assert "llm_calls" in result
        assert "retrieval_details" in result

    @patch("src.agentic_rag.workers.make_chat_model")
    @patch("src.agentic_rag.analytical_tools.get_db")
    @patch("src.agentic_rag.analytical_tools.get_registered_views")
    @patch("src.agentic_rag.analytical_tools.get_schema_description")
    @patch("src.agentic_rag.analytical_tools.get_join_relationships")
    @patch("src.agentic_rag.analytical_tools.execute_analytical_query")
    def test_populates_analytical_results(
        self, mock_exec, mock_join, mock_schema, mock_reg_views, mock_db, mock_llm_cls
    ):
        """Analytical results are added to context_gathered."""
        mock_db.return_value = MagicMock()
        mock_reg_views.return_value = {"violations"}
        mock_schema.return_value = "schema"
        mock_join.return_value = "joins"
        mock_exec.return_value = ("result table", {"row_count": 5, "columns": ["c"], "sql_executed": "SQL", "truncated": False})

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = _mock_llm_response("SELECT 1")
        mock_llm_cls.return_value = mock_llm

        result = data_analyst_agent(_make_state())
        assert "analytical_results" in result["context_gathered"]
        assert "result table" in result["context_gathered"]["analytical_results"][0]

    @patch("src.agentic_rag.workers.make_chat_model")
    @patch("src.agentic_rag.analytical_tools.get_db")
    @patch("src.agentic_rag.analytical_tools.get_registered_views")
    @patch("src.agentic_rag.analytical_tools.get_schema_description")
    @patch("src.agentic_rag.analytical_tools.get_join_relationships")
    @patch("src.agentic_rag.analytical_tools.execute_analytical_query")
    def test_adds_citations(
        self, mock_exec, mock_join, mock_schema, mock_reg_views, mock_db, mock_llm_cls
    ):
        """Citations include analytical source."""
        mock_db.return_value = MagicMock()
        mock_reg_views.return_value = {"violations"}
        mock_schema.return_value = "schema"
        mock_join.return_value = "joins"
        mock_exec.return_value = ("result", {"row_count": 1, "columns": ["c"], "sql_executed": "SELECT 1", "truncated": False})

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = _mock_llm_response("SELECT 1")
        mock_llm_cls.return_value = mock_llm

        result = data_analyst_agent(_make_state())
        citations = result["citations"]
        assert len(citations) == 1
        assert citations[0]["collection"] == "analytical"
        assert citations[0]["source"] == "DuckDB query"

    @patch("src.agentic_rag.workers.make_chat_model")
    @patch("src.agentic_rag.analytical_tools.get_db")
    @patch("src.agentic_rag.analytical_tools.get_registered_views")
    @patch("src.agentic_rag.analytical_tools.get_schema_description")
    @patch("src.agentic_rag.analytical_tools.get_join_relationships")
    @patch("src.agentic_rag.analytical_tools.execute_analytical_query")
    def test_logs_llm_calls(
        self, mock_exec, mock_join, mock_schema, mock_reg_views, mock_db, mock_llm_cls
    ):
        """LLM calls are logged."""
        mock_db.return_value = MagicMock()
        mock_reg_views.return_value = {"violations"}
        mock_schema.return_value = "schema"
        mock_join.return_value = "joins"
        mock_exec.return_value = ("result", {"row_count": 1, "columns": ["c"], "sql_executed": "SELECT 1", "truncated": False})

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = _mock_llm_response("SELECT 1")
        mock_llm_cls.return_value = mock_llm

        result = data_analyst_agent(_make_state())
        assert len(result["llm_calls"]) >= 1
        purposes = [c["call_purpose"] for c in result["llm_calls"]]
        assert "text_to_sql" in purposes

    @patch("src.agentic_rag.workers.make_chat_model")
    @patch("src.agentic_rag.analytical_tools.get_db")
    @patch("src.agentic_rag.analytical_tools.get_registered_views")
    @patch("src.agentic_rag.analytical_tools.get_schema_description")
    @patch("src.agentic_rag.analytical_tools.get_join_relationships")
    @patch("src.agentic_rag.analytical_tools.execute_analytical_query")
    def test_logs_retrieval_details(
        self, mock_exec, mock_join, mock_schema, mock_reg_views, mock_db, mock_llm_cls
    ):
        """Retrieval details capture SQL, row count, and columns."""
        mock_db.return_value = MagicMock()
        mock_reg_views.return_value = {"violations"}
        mock_schema.return_value = "schema"
        mock_join.return_value = "joins"
        mock_exec.return_value = ("result", {"row_count": 5, "columns": ["a", "b"], "sql_executed": "SELECT a, b FROM t", "truncated": False})

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = _mock_llm_response("SELECT a, b FROM t")
        mock_llm_cls.return_value = mock_llm

        result = data_analyst_agent(_make_state())
        details = result["retrieval_details"]
        assert len(details) == 1
        assert details[0]["collection"] == "analytical"
        assert details[0]["row_count"] == 5
        assert details[0]["columns"] == ["a", "b"]

    # ------------------------------------------------------------------
    # first-attempt-error self-correction + instrumentation
    # ------------------------------------------------------------------

    @patch("src.agentic_rag.workers.make_chat_model")
    @patch("src.agentic_rag.analytical_tools.get_db")
    @patch("src.agentic_rag.analytical_tools.get_registered_views")
    @patch("src.agentic_rag.analytical_tools.get_schema_description")
    @patch("src.agentic_rag.analytical_tools.get_join_relationships")
    @patch("src.agentic_rag.analytical_tools.execute_analytical_query")
    def test_clean_first_attempt_marks_no_error(
        self, mock_exec, mock_join, mock_schema, mock_reg_views, mock_db, mock_llm_cls
    ):
        """A clean first attempt records first_attempt_error=False, no retry."""
        mock_db.return_value = MagicMock()
        mock_reg_views.return_value = {"violations"}
        mock_schema.return_value = "schema"
        mock_join.return_value = "joins"
        mock_exec.return_value = (
            "result",
            {"row_count": 1, "columns": ["c"], "sql_executed": "SELECT 1", "truncated": False},
        )

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = _mock_llm_response("SELECT 1")
        mock_llm_cls.return_value = mock_llm

        detail = data_analyst_agent(_make_state())["retrieval_details"][0]
        assert detail["first_attempt_error"] is False
        assert detail["retry_triggered"] is False
        assert detail["recovered"] is False
        assert detail["final_error"] is False

    @patch("src.agentic_rag.workers.make_chat_model")
    @patch("src.agentic_rag.analytical_tools.get_db")
    @patch("src.agentic_rag.analytical_tools.get_registered_views")
    @patch("src.agentic_rag.analytical_tools.get_schema_description")
    @patch("src.agentic_rag.analytical_tools.get_join_relationships")
    @patch("src.agentic_rag.analytical_tools.execute_analytical_query")
    def test_first_attempt_error_triggers_recovery(
        self, mock_exec, mock_join, mock_schema, mock_reg_views, mock_db, mock_llm_cls
    ):
        """A first-attempt DuckDB error triggers one corrective retry that
        recovers, and the instrumentation records the recovery."""
        mock_db.return_value = MagicMock()
        mock_reg_views.return_value = {"violations"}
        mock_schema.return_value = "schema"
        mock_join.return_value = "joins"
        # First execution errors; the corrective retry succeeds.
        mock_exec.side_effect = [
            ("", {"error": "Binder Error: no such column foo", "row_count": 0}),
            ("| c |\n|---|\n| 1 |", {"row_count": 1, "columns": ["c"], "sql_executed": "SELECT 1", "truncated": False}),
        ]

        mock_llm = MagicMock()
        # gen -> bad SQL; error-retry -> good SQL; verification -> "OK" (no MISMATCH)
        mock_llm.invoke.side_effect = [
            _mock_llm_response("SELECT foo FROM violations"),
            _mock_llm_response("SELECT 1"),
            _mock_llm_response("OK - the result answers the question"),
        ]
        mock_llm_cls.return_value = mock_llm

        result = data_analyst_agent(_make_state())
        detail = result["retrieval_details"][0]
        assert detail["first_attempt_error"] is True
        assert "Binder Error" in detail["first_attempt_error_msg"]
        assert detail["retry_triggered"] is True
        assert "execution_error" in detail["retry_reason"]
        assert detail["recovered"] is True
        assert detail["final_error"] is False
        # The corrective LLM call is logged with its own purpose.
        purposes = [c["call_purpose"] for c in result["llm_calls"]]
        assert "text_to_sql_error_retry" in purposes

    @patch("src.agentic_rag.workers.make_chat_model")
    @patch("src.agentic_rag.analytical_tools.get_db")
    @patch("src.agentic_rag.analytical_tools.get_registered_views")
    @patch("src.agentic_rag.analytical_tools.get_schema_description")
    @patch("src.agentic_rag.analytical_tools.get_join_relationships")
    @patch("src.agentic_rag.analytical_tools.execute_analytical_query")
    def test_unrecovered_error_marks_final_error(
        self, mock_exec, mock_join, mock_schema, mock_reg_views, mock_db, mock_llm_cls
    ):
        """When the retry also errors, final_error stays True and recovered False."""
        mock_db.return_value = MagicMock()
        mock_reg_views.return_value = {"violations"}
        mock_schema.return_value = "schema"
        mock_join.return_value = "joins"
        mock_exec.side_effect = [
            ("", {"error": "Parser Error", "row_count": 0}),
            ("", {"error": "Parser Error again", "row_count": 0}),
        ]

        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = [
            _mock_llm_response("SELECT bad"),
            _mock_llm_response("SELECT still bad"),
        ]
        mock_llm_cls.return_value = mock_llm

        detail = data_analyst_agent(_make_state())["retrieval_details"][0]
        assert detail["first_attempt_error"] is True
        assert detail["retry_triggered"] is True
        assert detail["recovered"] is False
        assert detail["final_error"] is True

    @patch("src.agentic_rag.analytical_tools.get_db")
    def test_handles_missing_db(self, mock_db):
        """Gracefully handles database connection failure."""
        mock_db.side_effect = FileNotFoundError("Data not found")

        result = data_analyst_agent(_make_state())
        assert "analytical_results" in result["context_gathered"]
        assert "Error" in result["context_gathered"]["analytical_results"][0]
        assert len(result["worker_errors"]) == 1

    @patch("src.agentic_rag.analytical_tools.get_db")
    @patch("src.agentic_rag.analytical_tools.get_registered_views")
    def test_handles_zero_views(self, mock_reg_views, mock_db):
        """Returns error when no views are registered."""
        mock_db.return_value = MagicMock()
        mock_reg_views.return_value = set()

        result = data_analyst_agent(_make_state())
        assert "analytical_results" in result["context_gathered"]
        assert "No data files" in result["context_gathered"]["analytical_results"][0]
        assert len(result["worker_errors"]) == 1

    @patch("src.agentic_rag.workers.make_chat_model")
    @patch("src.agentic_rag.analytical_tools.get_db")
    @patch("src.agentic_rag.analytical_tools.get_registered_views")
    @patch("src.agentic_rag.analytical_tools.get_schema_description")
    @patch("src.agentic_rag.analytical_tools.get_join_relationships")
    @patch("src.agentic_rag.analytical_tools.execute_analytical_query")
    def test_handles_empty_results(
        self, mock_exec, mock_join, mock_schema, mock_reg_views, mock_db, mock_llm_cls
    ):
        """Empty query results produce a descriptive message."""
        mock_db.return_value = MagicMock()
        mock_reg_views.return_value = {"violations"}
        mock_schema.return_value = "schema"
        mock_join.return_value = "joins"
        mock_exec.return_value = ("", {"row_count": 0, "columns": [], "sql_executed": "SELECT 1 WHERE FALSE", "truncated": False})

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = _mock_llm_response("SELECT 1 WHERE FALSE")
        mock_llm_cls.return_value = mock_llm

        result = data_analyst_agent(_make_state())
        assert "No matching records" in result["context_gathered"]["analytical_results"][0]

    @patch("src.agentic_rag.workers.make_chat_model")
    @patch("src.agentic_rag.analytical_tools.get_db")
    @patch("src.agentic_rag.analytical_tools.get_registered_views")
    @patch("src.agentic_rag.analytical_tools.get_schema_description")
    @patch("src.agentic_rag.analytical_tools.get_join_relationships")
    @patch("src.agentic_rag.analytical_tools.execute_analytical_query")
    def test_handles_invalid_sql_generation(
        self, mock_exec, mock_join, mock_schema, mock_reg_views, mock_db, mock_llm_cls
    ):
        """Invalid SQL generated by LLM is handled gracefully."""
        mock_db.return_value = MagicMock()
        mock_reg_views.return_value = {"violations"}
        mock_schema.return_value = "schema"
        mock_join.return_value = "joins"
        mock_exec.return_value = ("Query rejected: Forbidden SQL keyword: DROP", {"error": "Forbidden SQL keyword: DROP", "row_count": 0})

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = _mock_llm_response("DROP TABLE violations")
        mock_llm_cls.return_value = mock_llm

        result = data_analyst_agent(_make_state())
        assert "analytical_results" in result["context_gathered"]

    @patch("src.agentic_rag.workers.make_chat_model")
    @patch("src.agentic_rag.analytical_tools.get_db")
    @patch("src.agentic_rag.analytical_tools.get_registered_views")
    @patch("src.agentic_rag.analytical_tools.get_schema_description")
    @patch("src.agentic_rag.analytical_tools.get_join_relationships")
    @patch("src.agentic_rag.analytical_tools.execute_analytical_query")
    def test_verification_retry_on_mismatch(
        self, mock_exec, mock_join, mock_schema, mock_reg_views, mock_db, mock_llm_cls
    ):
        """Verification mismatch triggers one retry."""
        mock_db.return_value = MagicMock()
        mock_reg_views.return_value = {"violations"}
        mock_schema.return_value = "schema"
        mock_join.return_value = "joins"
        mock_exec.return_value = ("result", {"row_count": 1, "columns": ["c"], "sql_executed": "SELECT 1", "truncated": False})

        mock_llm = MagicMock()
        # First call: SQL gen, Second call: verification (MISMATCH), Third call: retry SQL
        mock_llm.invoke.side_effect = [
            _mock_llm_response("SELECT wrong_col FROM t"),
            _mock_llm_response("MISMATCH: wrong column selected"),
            _mock_llm_response("SELECT correct_col FROM t"),
        ]
        mock_llm_cls.return_value = mock_llm

        result = data_analyst_agent(_make_state())
        # Should have 3 LLM calls: text_to_sql, verification, retry
        purposes = [c["call_purpose"] for c in result["llm_calls"]]
        assert "text_to_sql" in purposes
        assert "result_verification" in purposes
        assert "text_to_sql_retry" in purposes

    @patch("src.agentic_rag.workers.make_chat_model")
    @patch("src.agentic_rag.analytical_tools.get_db")
    @patch("src.agentic_rag.analytical_tools.get_registered_views")
    @patch("src.agentic_rag.analytical_tools.get_schema_description")
    @patch("src.agentic_rag.analytical_tools.get_join_relationships")
    @patch("src.agentic_rag.analytical_tools.execute_analytical_query")
    def test_returns_only_new_data(
        self, mock_exec, mock_join, mock_schema, mock_reg_views, mock_db, mock_llm_cls
    ):
        """Worker returns only its own new data, not accumulated state."""
        mock_db.return_value = MagicMock()
        mock_reg_views.return_value = {"violations"}
        mock_schema.return_value = "schema"
        mock_join.return_value = "joins"
        mock_exec.return_value = ("result", {"row_count": 1, "columns": ["c"], "sql_executed": "SELECT 1", "truncated": False})

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = _mock_llm_response("SELECT 1")
        mock_llm_cls.return_value = mock_llm

        # State with pre-existing data
        state = _make_state(
            citations=[{"source": "previous", "collection": "csb"}],
            llm_calls=[{"call_purpose": "previous"}],
            retrieval_details=[{"collection": "previous"}],
        )

        result = data_analyst_agent(state)
        # Should NOT include the pre-existing data
        assert len(result["citations"]) == 1
        assert result["citations"][0]["collection"] == "analytical"
        assert all(c["call_purpose"] != "previous" for c in result["llm_calls"])
