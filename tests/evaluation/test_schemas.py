"""
Tests for src.evaluation.schemas.

Covers validate_eval_entry and validate_eval_dataset with valid entries,
missing fields, invalid types, invalid enum values, and dataset-level
checks including duplicate detection.
"""

import pytest

from src.evaluation.schemas import (
    VALID_AGENTS,
    VALID_CATEGORIES,
    VALID_COMPLEXITIES,
    VALID_SOURCES,
    validate_eval_dataset,
    validate_eval_entry,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_entry(**overrides):
    """Return a minimal valid EvalEntry dict, with optional overrides."""
    base = {
        "query": "What does 1910.119 require?",
        "expected_answer": "PSM requirements for highly hazardous chemicals.",
        "ground_truth_chunks": ["chunk_001", "chunk_002"],
        "source": "auto_regulatory",
        "category": "regulatory_factual",
    }
    base.update(overrides)
    return base


def _make_full_entry(**overrides):
    """Return a fully populated EvalEntry dict."""
    base = _make_entry(
        key_claims=[
            {"claim": "PSM applies to HHC", "source_chunk": "chunk_001"},
            {"claim": "Employers must comply", "source_chunk": "chunk_002"},
        ],
        expected_agents=["osha_regs"],
        complexity="simple",
        metadata={"standard_number": "1910.119"},
    )
    base.update(overrides)
    return base


# ============================================================================
# validate_eval_entry – required fields
# ============================================================================


class TestValidateEvalEntryRequired:

    def test_valid_minimal_entry(self):
        assert validate_eval_entry(_make_entry()) == []

    def test_valid_full_entry(self):
        assert validate_eval_entry(_make_full_entry()) == []

    @pytest.mark.parametrize(
        "missing_key",
        ["query", "expected_answer", "ground_truth_chunks", "source", "category"],
    )
    def test_missing_required_key(self, missing_key):
        entry = _make_entry()
        del entry[missing_key]
        errors = validate_eval_entry(entry)
        assert len(errors) == 1
        assert "Missing required" in errors[0]
        assert missing_key in errors[0]

    def test_missing_multiple_keys(self):
        entry = {"query": "hello"}
        errors = validate_eval_entry(entry)
        assert len(errors) == 1
        assert "Missing required" in errors[0]

    def test_empty_query(self):
        entry = _make_entry(query="")
        errors = validate_eval_entry(entry)
        assert any("'query'" in e for e in errors)

    def test_whitespace_only_query(self):
        entry = _make_entry(query="   ")
        errors = validate_eval_entry(entry)
        assert any("'query'" in e for e in errors)

    def test_empty_expected_answer(self):
        entry = _make_entry(expected_answer="")
        errors = validate_eval_entry(entry)
        assert any("'expected_answer'" in e for e in errors)


# ============================================================================
# validate_eval_entry – type checks
# ============================================================================


class TestValidateEvalEntryTypes:

    def test_ground_truth_chunks_not_list(self):
        entry = _make_entry(ground_truth_chunks="chunk_001")
        errors = validate_eval_entry(entry)
        assert any("'ground_truth_chunks' must be a list" in e for e in errors)

    def test_ground_truth_chunks_non_string_items(self):
        entry = _make_entry(ground_truth_chunks=["chunk_001", 42])
        errors = validate_eval_entry(entry)
        assert any("string" in e for e in errors)

    def test_source_not_string(self):
        entry = _make_entry(source=123)
        errors = validate_eval_entry(entry)
        assert any("'source' must be a string" in e for e in errors)

    def test_category_not_string(self):
        entry = _make_entry(category=None)
        errors = validate_eval_entry(entry)
        assert any("'category' must be a string" in e for e in errors)

    def test_metadata_not_dict(self):
        entry = _make_entry(metadata="not a dict")
        errors = validate_eval_entry(entry)
        assert any("'metadata' must be a dict" in e for e in errors)


# ============================================================================
# validate_eval_entry – enum validation
# ============================================================================


class TestValidateEvalEntryEnums:

    def test_invalid_source(self):
        entry = _make_entry(source="unknown_source")
        errors = validate_eval_entry(entry)
        assert any("'source' must be one of" in e for e in errors)

    def test_invalid_category(self):
        entry = _make_entry(category="unknown_category")
        errors = validate_eval_entry(entry)
        assert any("'category' must be one of" in e for e in errors)

    def test_invalid_complexity(self):
        entry = _make_entry(complexity="extreme")
        errors = validate_eval_entry(entry)
        assert any("'complexity' must be one of" in e for e in errors)

    def test_invalid_agent(self):
        entry = _make_entry(expected_agents=["osha_regs", "fake_agent"])
        errors = validate_eval_entry(entry)
        assert any("fake_agent" in e for e in errors)

    @pytest.mark.parametrize("source", sorted(VALID_SOURCES))
    def test_all_valid_sources(self, source):
        entry = _make_entry(source=source)
        assert validate_eval_entry(entry) == []

    @pytest.mark.parametrize("category", sorted(VALID_CATEGORIES))
    def test_all_valid_categories(self, category):
        entry = _make_entry(category=category)
        assert validate_eval_entry(entry) == []

    @pytest.mark.parametrize("complexity", sorted(VALID_COMPLEXITIES))
    def test_all_valid_complexities(self, complexity):
        entry = _make_entry(complexity=complexity)
        assert validate_eval_entry(entry) == []

    @pytest.mark.parametrize("agent", sorted(VALID_AGENTS))
    def test_all_valid_agents(self, agent):
        entry = _make_entry(expected_agents=[agent])
        assert validate_eval_entry(entry) == []


# ============================================================================
# validate_eval_entry – key_claims
# ============================================================================


class TestValidateEvalEntryKeyClaims:

    def test_valid_key_claims(self):
        entry = _make_entry(
            key_claims=[
                {"claim": "A", "source_chunk": "c1"},
                {"claim": "B", "source_chunk": "c2"},
            ]
        )
        assert validate_eval_entry(entry) == []

    def test_key_claims_not_list(self):
        entry = _make_entry(key_claims="not a list")
        errors = validate_eval_entry(entry)
        assert any("'key_claims' must be a list" in e for e in errors)

    def test_key_claims_item_not_dict(self):
        entry = _make_entry(key_claims=["not a dict"])
        errors = validate_eval_entry(entry)
        assert any("must be a dict" in e for e in errors)

    def test_key_claims_missing_claim_key(self):
        entry = _make_entry(key_claims=[{"source_chunk": "c1"}])
        errors = validate_eval_entry(entry)
        assert any("missing 'claim'" in e for e in errors)

    def test_key_claims_claim_only(self):
        """'source_chunk' is optional; only 'claim' is required."""
        entry = _make_entry(key_claims=[{"claim": "A"}])
        assert validate_eval_entry(entry) == []


# ============================================================================
# validate_eval_entry – expected_agents
# ============================================================================


class TestValidateEvalEntryExpectedAgents:

    def test_expected_agents_not_list(self):
        entry = _make_entry(expected_agents="osha_regs")
        errors = validate_eval_entry(entry)
        assert any("'expected_agents' must be a list" in e for e in errors)

    def test_multiple_valid_agents(self):
        entry = _make_entry(expected_agents=["osha_regs", "csb_incident"])
        assert validate_eval_entry(entry) == []

    def test_empty_agents_list(self):
        entry = _make_entry(expected_agents=[])
        assert validate_eval_entry(entry) == []


# ============================================================================
# validate_eval_dataset
# ============================================================================


class TestValidateEvalDataset:

    def test_all_valid(self):
        entries = [
            _make_entry(query="Q1"),
            _make_entry(query="Q2"),
            _make_entry(query="Q3"),
        ]
        report = validate_eval_dataset(entries)
        assert report["total"] == 3
        assert report["valid"] == 3
        assert report["invalid"] == 0
        assert report["errors"] == []
        assert report["duplicate_queries"] == []

    def test_mixed_valid_invalid(self):
        entries = [
            _make_entry(query="Q1"),
            {"query": "Q2"},  # missing required keys
            _make_entry(query="Q3", source="bad_source"),
        ]
        report = validate_eval_dataset(entries)
        assert report["total"] == 3
        assert report["valid"] == 1
        assert report["invalid"] == 2
        assert len(report["errors"]) == 2

    def test_empty_dataset(self):
        report = validate_eval_dataset([])
        assert report["total"] == 0
        assert report["valid"] == 0
        assert report["duplicate_queries"] == []

    def test_duplicate_query_detection(self):
        entries = [
            _make_entry(query="What does 1910.119 require?"),
            _make_entry(query="What does 1910.119 require?"),
        ]
        report = validate_eval_dataset(entries)
        assert len(report["duplicate_queries"]) == 1

    def test_case_insensitive_duplicate_detection(self):
        entries = [
            _make_entry(query="What does 1910.119 require?"),
            _make_entry(query="what does 1910.119 require?"),
        ]
        report = validate_eval_dataset(entries)
        assert len(report["duplicate_queries"]) >= 1

    def test_source_counts(self):
        entries = [
            _make_entry(query="Q1", source="auto_regulatory"),
            _make_entry(query="Q2", source="auto_regulatory"),
            _make_entry(query="Q3", source="auto_incident"),
        ]
        report = validate_eval_dataset(entries)
        assert report["source_counts"] == {
            "auto_regulatory": 2,
            "auto_incident": 1,
        }

    def test_category_counts(self):
        entries = [
            _make_entry(query="Q1", category="regulatory_factual"),
            _make_entry(query="Q2", category="incident_factual"),
        ]
        report = validate_eval_dataset(entries)
        assert report["category_counts"] == {
            "regulatory_factual": 1,
            "incident_factual": 1,
        }

    def test_complexity_counts(self):
        entries = [
            _make_entry(query="Q1", complexity="simple"),
            _make_entry(query="Q2", complexity="reasoning"),
            _make_entry(query="Q3"),  # no complexity
        ]
        report = validate_eval_dataset(entries)
        assert report["complexity_counts"] == {
            "simple": 1,
            "reasoning": 1,
        }

    def test_error_includes_index(self):
        entries = [
            _make_entry(query="Q1"),
            {"query": "bad entry"},  # invalid
        ]
        report = validate_eval_dataset(entries)
        assert report["errors"][0]["index"] == 1


# ============================================================================
# Statistical Aggregation extensions
# ============================================================================


class TestStatisticalSources:

    def test_auto_statistical_is_valid_source(self):
        assert "auto_statistical" in VALID_SOURCES

    def test_auto_statistical_entry(self):
        entry = _make_entry(source="auto_statistical")
        assert validate_eval_entry(entry) == []


class TestStatisticalCategories:

    @pytest.mark.parametrize("category", [
        "statistical_factual",
        "statistical_trend",
        "statistical_comparative",
        "hybrid_analytical",
    ])
    def test_statistical_category_is_valid(self, category):
        assert category in VALID_CATEGORIES
        entry = _make_entry(category=category)
        assert validate_eval_entry(entry) == []


class TestDataAnalystAgent:

    def test_data_analyst_is_valid_agent(self):
        assert "data_analyst" in VALID_AGENTS

    def test_data_analyst_entry(self):
        entry = _make_entry(expected_agents=["data_analyst"])
        assert validate_eval_entry(entry) == []


class TestOptionalSqlFields:

    def test_expected_sql_accepted(self):
        entry = _make_entry(expected_sql="SELECT COUNT(*) FROM violations")
        assert validate_eval_entry(entry) == []

    def test_ground_truth_result_accepted(self):
        entry = _make_entry(ground_truth_result={"row_count": 5, "columns": ["count"]})
        assert validate_eval_entry(entry) == []

    def test_expected_sql_wrong_type(self):
        entry = _make_entry(expected_sql=123)
        errors = validate_eval_entry(entry)
        assert any("expected_sql" in e for e in errors)

    def test_ground_truth_result_wrong_type(self):
        entry = _make_entry(ground_truth_result="not a dict")
        errors = validate_eval_entry(entry)
        assert any("ground_truth_result" in e for e in errors)
