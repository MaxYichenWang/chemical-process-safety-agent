"""
Tests for src.evaluation.generators.violation_qa_generator.

Covers ViolationQAGenerator: data loading, selection strategies,
question template application, LLM refinement, cross-corpus mapping,
and end-to-end generation with mocked dependencies.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.evaluation.generators.violation_qa_generator import (
    ViolationQAGenerator,
    _TEMPLATES,
    _VIOL_TYPE_LABELS,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_violations_df(n: int = 10) -> pd.DataFrame:
    """Create a small synthetic violations DataFrame."""
    rows = []
    for i in range(n):
        rows.append({
            "chunk_id": f"viol_chunk_{i:03d}",
            "corpus": "osha_enf",
            "doc_type": "enforcement_violation",
            "text": f"Violation {i}: The employer failed to comply with standard "
                    f"1910.119 regarding process safety management. "
                    f"Employees were exposed to hazardous chemicals.",
            "token_count": 100 + i * 10,
            "activity_nr": f"act_{i}",
            "citation_id": f"CIT_{i:04d}",
            "standard_number": "1910.119" if i < 5 else "1910.147",
            "viol_type": "W" if i < 3 else ("R" if i < 5 else "S"),
            "initial_penalty": 10000.0 * (n - i),
            "current_penalty": 8000.0 * (n - i),
            "site_state": "TX" if i % 2 == 0 else "LA",
            "site_city": "Houston" if i % 2 == 0 else "Baton Rouge",
            "naics_code": "325110",
            "establishment_name": f"Chemical Corp {i}",
            "issuance_date": f"2024-0{(i % 9) + 1}-15",
        })
    return pd.DataFrame(rows)


def _make_regulations_df(n: int = 5) -> pd.DataFrame:
    """Create a small synthetic regulations DataFrame."""
    rows = []
    for i in range(n):
        rows.append({
            "chunk_id": f"reg_chunk_{i:03d}",
            "standard_number": "1910.119" if i < 3 else "1910.147",
            "section_id": f"1910.119({chr(97 + i)})" if i < 3 else f"1910.147({chr(97 + i)})",
            "title": f"Section {i} Title",
        })
    return pd.DataFrame(rows)


def _mock_llm_response(question: str = "Refined question?",
                        answer: str = "Expected answer.",
                        claims: list | None = None):
    """Create a mock LLM response."""
    if claims is None:
        claims = [{"claim": "Test claim", "source_chunk": "chunk_001"}]
    response = MagicMock()
    response.content = json.dumps({
        "question": question,
        "expected_answer": answer,
        "key_claims": claims,
    })
    return response


# ============================================================================
# Data loading
# ============================================================================


class TestDataLoading:

    def test_missing_violations_file(self, tmp_path):
        gen = ViolationQAGenerator(
            violations_path=tmp_path / "nonexistent.parquet"
        )
        with pytest.raises(FileNotFoundError, match="Violations parquet"):
            gen._load_violations()

    def test_missing_regulations_file(self, tmp_path):
        gen = ViolationQAGenerator(
            regulations_path=tmp_path / "nonexistent.parquet"
        )
        with pytest.raises(FileNotFoundError, match="Regulations parquet"):
            gen._load_regulations_chunks()


# ============================================================================
# Selection strategies
# ============================================================================


class TestSelectionStrategies:

    def test_select_factual_violations(self):
        df = _make_violations_df(20)
        gen = ViolationQAGenerator()
        selected = gen._select_factual_violations(df)
        assert len(selected) > 0
        assert len(selected) <= 18  # _TARGET_FACTUAL

    def test_select_factual_prioritizes_willful_repeat(self):
        df = _make_violations_df(20)
        gen = ViolationQAGenerator()
        selected = gen._select_factual_violations(df)
        priority_types = selected[selected["viol_type"].isin(["W", "R"])]
        assert len(priority_types) > 0

    def test_select_pattern_violations(self):
        df = _make_violations_df(20)
        # Ensure there are enough W violations under 1910.119
        gen = ViolationQAGenerator()
        groups = gen._select_pattern_violations(df)
        assert isinstance(groups, list)
        for group in groups:
            assert "standard_number" in group
            assert "viol_type" in group
            assert "chunk_ids" in group
            assert len(group["chunk_ids"]) >= 2

    def test_select_high_penalty_violations(self):
        df = _make_violations_df(10)
        gen = ViolationQAGenerator()
        selected = gen._select_high_penalty_violations(df)
        assert len(selected) > 0
        # Should be sorted by penalty descending
        penalties = selected["initial_penalty"].tolist()
        assert penalties == sorted(penalties, reverse=True)

    def test_find_regulation_chunks(self):
        regs_df = _make_regulations_df()
        gen = ViolationQAGenerator()
        chunks = gen._find_regulation_chunks("1910.119", regs_df)
        assert len(chunks) > 0
        assert all(c.startswith("reg_chunk_") for c in chunks)

    def test_find_regulation_chunks_no_match(self):
        regs_df = _make_regulations_df()
        gen = ViolationQAGenerator()
        chunks = gen._find_regulation_chunks("1910.999", regs_df)
        assert chunks == []


# ============================================================================
# Question templates
# ============================================================================


class TestTemplates:

    def test_all_templates_have_placeholders(self):
        for name, tpl in _TEMPLATES.items():
            assert "{" in tpl, f"Template '{name}' has no placeholders"

    def test_violation_factual_template_format(self):
        result = _TEMPLATES["violation_factual"].format(
            citation_id="CIT_001",
            standard_number="1910.119",
            establishment_name="Test Corp",
        )
        assert "CIT_001" in result
        assert "1910.119" in result
        assert "Test Corp" in result

    def test_viol_type_labels_coverage(self):
        assert "W" in _VIOL_TYPE_LABELS
        assert "R" in _VIOL_TYPE_LABELS
        assert "S" in _VIOL_TYPE_LABELS


# ============================================================================
# LLM refinement
# ============================================================================


class TestLLMRefinement:

    @patch("src.evaluation.generators.violation_qa_generator.ChatOpenAI")
    def test_refine_with_llm_success(self, mock_chat):
        mock_instance = MagicMock()
        mock_instance.invoke.return_value = _mock_llm_response()
        mock_chat.return_value = mock_instance

        gen = ViolationQAGenerator()
        gen._llm = mock_instance
        row = _make_violations_df(1).iloc[0].to_dict()
        result = gen._refine_with_llm("Draft question?", row)

        assert result is not None
        assert "question" in result
        assert "expected_answer" in result
        assert "key_claims" in result

    @patch("src.evaluation.generators.violation_qa_generator.ChatOpenAI")
    def test_refine_with_llm_parse_failure(self, mock_chat):
        mock_instance = MagicMock()
        bad_response = MagicMock()
        bad_response.content = "not valid json"
        mock_instance.invoke.return_value = bad_response
        mock_chat.return_value = mock_instance

        gen = ViolationQAGenerator()
        gen._llm = mock_instance
        row = _make_violations_df(1).iloc[0].to_dict()
        result = gen._refine_with_llm("Draft question?", row)

        assert result is None


# ============================================================================
# End-to-end generation (with mocks)
# ============================================================================


class TestGenerateEndToEnd:

    @patch("src.evaluation.generators.violation_qa_generator.ChatOpenAI")
    def test_generate_produces_valid_entries(self, mock_chat, tmp_path):
        # Create small parquet files
        violations_df = _make_violations_df(10)
        regs_df = _make_regulations_df(5)

        violations_path = tmp_path / "osha_violations_sample_chunks.parquet"
        regulations_path = tmp_path / "osha_regulations_chunks.parquet"
        violations_df.to_parquet(violations_path)
        regs_df.to_parquet(regulations_path)

        # Mock LLM
        mock_instance = MagicMock()
        mock_instance.invoke.return_value = _mock_llm_response(
            question="What violation occurred?",
            answer="A willful violation was cited.",
        )
        mock_chat.return_value = mock_instance

        gen = ViolationQAGenerator(
            violations_path=violations_path,
            regulations_path=regulations_path,
        )
        gen._llm = mock_instance

        entries = gen.generate()

        assert isinstance(entries, list)
        assert len(entries) > 0

        for entry in entries:
            assert entry["source"] in ("auto_violation",)
            assert entry["category"] in (
                "enforcement_violation", "cross_corpus",
            )
            assert len(entry["ground_truth_chunks"]) > 0
            assert entry["query"]
            assert entry["expected_answer"]

    @patch("src.evaluation.generators.violation_qa_generator.ChatOpenAI")
    def test_generate_no_duplicate_queries(self, mock_chat, tmp_path):
        violations_df = _make_violations_df(5)
        regs_df = _make_regulations_df(3)

        violations_path = tmp_path / "violations.parquet"
        regulations_path = tmp_path / "regulations.parquet"
        violations_df.to_parquet(violations_path)
        regs_df.to_parquet(regulations_path)

        mock_instance = MagicMock()
        # Return the same question text to test deduplication
        mock_instance.invoke.return_value = _mock_llm_response(
            question="Same question",
            answer="Same answer.",
        )
        mock_chat.return_value = mock_instance

        gen = ViolationQAGenerator(
            violations_path=violations_path,
            regulations_path=regulations_path,
        )
        gen._llm = mock_instance

        entries = gen.generate()
        queries = [e["query"] for e in entries]
        # All queries should be unique (deduplication)
        assert len(queries) == len(set(q.strip().lower() for q in queries))

    def test_generate_missing_file_raises(self, tmp_path):
        gen = ViolationQAGenerator(
            violations_path=tmp_path / "nonexistent.parquet",
        )
        with pytest.raises(FileNotFoundError):
            gen.generate()
