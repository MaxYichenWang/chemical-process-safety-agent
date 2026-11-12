"""
Tests for src.evaluation.generators.genduty_qa_generator.

Covers GendutyQAGenerator: data loading, hazard classification,
selection strategies, question templates, LLM refinement, cross-corpus
mapping, and end-to-end generation with mocked dependencies.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.evaluation.generators.genduty_qa_generator import (
    GendutyQAGenerator,
    _HAZARD_KEYWORDS,
    _TEMPLATES,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_genduty_df(n: int = 10) -> pd.DataFrame:
    """Create a small synthetic general duty violations DataFrame."""
    hazard_texts = [
        "fire and explosion hazard. Employees exposed to flammable materials.",
        "chemical exposure to toxic substances in the workplace.",
        "confined space entry without proper atmospheric testing.",
        "process safety management failure leading to chemical release.",
        "mechanical hazard from unguarded rotating equipment.",
        "fall hazard from elevated platform without guardrail.",
        "fire and explosion from improper storage of combustible materials.",
        "chemical exposure to hydrogen sulfide in confined space.",
        "process hazard analysis not conducted for modified equipment.",
        "ignition source near flammable vapor accumulation.",
    ]
    rows = []
    for i in range(n):
        text = hazard_texts[i % len(hazard_texts)]
        rows.append({
            "chunk_id": f"genduty_chunk_{i:03d}",
            "corpus": "osha_enf",
            "subcorpus": "osha_enf",
            "doc_type": "enforcement_general_duty",
            "text": f"Section 5(a)(1): {text} "
                    f"Feasible abatement methods include implementing proper "
                    f"safety controls per 1910.119 requirements.",
            "token_count": 100 + i * 15,
            "hash": f"hash_{i}",
            "activity_nr": f"act_gd_{i}",
            "citation_id": f"GDCIT_{i:04d}",
            "standard_number": "5(a)(1)",
            "site_state": "TX" if i % 3 == 0 else ("CA" if i % 3 == 1 else "LA"),
            "site_city": "Houston" if i % 3 == 0 else "Los Angeles",
            "naics_code": "325110" if i % 2 == 0 else "324110",
            "establishment_name": f"Safety Corp {i}",
            "chunk_index": 0,
            "total_chunks": 1,
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
            "title": f"PSM Section {i} Title",
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

    def test_missing_genduty_file(self, tmp_path):
        gen = GendutyQAGenerator(
            genduty_path=tmp_path / "nonexistent.parquet"
        )
        with pytest.raises(FileNotFoundError, match="General duty parquet"):
            gen._load_genduty()

    def test_missing_regulations_file(self, tmp_path):
        gen = GendutyQAGenerator(
            regulations_path=tmp_path / "nonexistent.parquet"
        )
        with pytest.raises(FileNotFoundError, match="Regulations parquet"):
            gen._load_regulations_chunks()


# ============================================================================
# Hazard classification
# ============================================================================


class TestHazardClassification:

    def test_fire_explosion_detection(self):
        gen = GendutyQAGenerator()
        result = gen._classify_hazard("fire and explosion hazard at plant")
        assert result == "fire_explosion"

    def test_chemical_exposure_detection(self):
        gen = GendutyQAGenerator()
        result = gen._classify_hazard("toxic chemical exposure to employees")
        assert result == "chemical_exposure"

    def test_confined_space_detection(self):
        gen = GendutyQAGenerator()
        result = gen._classify_hazard("confined space entry hazard")
        assert result == "confined_space"

    def test_process_safety_detection(self):
        gen = GendutyQAGenerator()
        result = gen._classify_hazard("process safety management failure")
        assert result == "process_safety"

    def test_no_match_returns_none(self):
        gen = GendutyQAGenerator()
        result = gen._classify_hazard("completely unrelated text")
        assert result is None

    def test_hazard_keywords_completeness(self):
        """All expected hazard categories are defined."""
        expected = {
            "fire_explosion",
            "chemical_exposure",
            "confined_space",
            "fall_hazard",
            "mechanical",
            "process_safety",
        }
        assert set(_HAZARD_KEYWORDS.keys()) == expected


# ============================================================================
# Selection strategies
# ============================================================================


class TestSelectionStrategies:

    def test_select_factual_chunks(self):
        df = _make_genduty_df(20)
        gen = GendutyQAGenerator()
        selected = gen._select_factual_chunks(df)
        assert len(selected) > 0
        assert len(selected) <= 13  # _TARGET_FACTUAL

    def test_select_factual_prioritizes_psm(self):
        df = _make_genduty_df(20)
        gen = GendutyQAGenerator()
        selected = gen._select_factual_chunks(df)
        # Some selected chunks should be PSM-related
        psm_texts = selected["text"].str.lower().str.contains("process safety")
        assert psm_texts.any()

    def test_select_pattern_groups(self):
        df = _make_genduty_df(20)
        gen = GendutyQAGenerator()
        groups = gen._select_pattern_groups(df)
        assert isinstance(groups, list)
        for group in groups:
            assert "hazard_category" in group
            assert "chunk_ids" in group
            assert len(group["chunk_ids"]) >= 2

    def test_find_regulation_chunks(self):
        regs_df = _make_regulations_df()
        gen = GendutyQAGenerator()
        chunks = gen._find_regulation_chunks("1910.119", regs_df)
        assert len(chunks) > 0


# ============================================================================
# Question templates
# ============================================================================


class TestTemplates:

    def test_all_templates_exist(self):
        expected_keys = {
            "genduty_factual",
            "genduty_abatement",
            "hazard_pattern",
            "cross_corpus",
        }
        assert set(_TEMPLATES.keys()) == expected_keys

    def test_genduty_factual_template_format(self):
        result = _TEMPLATES["genduty_factual"].format(
            establishment_name="Test Corp",
        )
        assert "Test Corp" in result
        assert "General Duty" in result


# ============================================================================
# LLM refinement
# ============================================================================


class TestLLMRefinement:

    @patch("src.evaluation.generators.genduty_qa_generator.ChatOpenAI")
    def test_refine_with_llm_success(self, mock_chat):
        mock_instance = MagicMock()
        mock_instance.invoke.return_value = _mock_llm_response()
        mock_chat.return_value = mock_instance

        gen = GendutyQAGenerator()
        gen._llm = mock_instance
        row = _make_genduty_df(1).iloc[0].to_dict()
        result = gen._refine_with_llm("Draft question?", row)

        assert result is not None
        assert "question" in result
        assert "expected_answer" in result

    @patch("src.evaluation.generators.genduty_qa_generator.ChatOpenAI")
    def test_refine_with_llm_parse_failure(self, mock_chat):
        mock_instance = MagicMock()
        bad_response = MagicMock()
        bad_response.content = "not valid json at all"
        mock_instance.invoke.return_value = bad_response
        mock_chat.return_value = mock_instance

        gen = GendutyQAGenerator()
        gen._llm = mock_instance
        row = _make_genduty_df(1).iloc[0].to_dict()
        result = gen._refine_with_llm("Draft question?", row)

        assert result is None


# ============================================================================
# End-to-end generation (with mocks)
# ============================================================================


class TestGenerateEndToEnd:

    @patch("src.evaluation.generators.genduty_qa_generator.ChatOpenAI")
    def test_generate_produces_valid_entries(self, mock_chat, tmp_path):
        genduty_df = _make_genduty_df(15)
        regs_df = _make_regulations_df(5)

        genduty_path = tmp_path / "osha_genduty_sample_chunks.parquet"
        regulations_path = tmp_path / "osha_regulations_chunks.parquet"
        genduty_df.to_parquet(genduty_path)
        regs_df.to_parquet(regulations_path)

        mock_instance = MagicMock()
        mock_instance.invoke.return_value = _mock_llm_response(
            question="What hazard was cited?",
            answer="A fire and explosion hazard was cited.",
        )
        mock_chat.return_value = mock_instance

        gen = GendutyQAGenerator(
            genduty_path=genduty_path,
            regulations_path=regulations_path,
        )
        gen._llm = mock_instance

        entries = gen.generate()

        assert isinstance(entries, list)
        assert len(entries) > 0

        for entry in entries:
            assert entry["source"] in ("auto_genduty",)
            assert entry["category"] in (
                "enforcement_genduty", "cross_corpus",
            )
            assert len(entry["ground_truth_chunks"]) > 0
            assert entry["query"]
            assert entry["expected_answer"]

    @patch("src.evaluation.generators.genduty_qa_generator.ChatOpenAI")
    def test_generate_no_duplicate_queries(self, mock_chat, tmp_path):
        genduty_df = _make_genduty_df(5)
        regs_df = _make_regulations_df(3)

        genduty_path = tmp_path / "genduty.parquet"
        regulations_path = tmp_path / "regulations.parquet"
        genduty_df.to_parquet(genduty_path)
        regs_df.to_parquet(regulations_path)

        mock_instance = MagicMock()
        mock_instance.invoke.return_value = _mock_llm_response(
            question="Same question",
            answer="Same answer.",
        )
        mock_chat.return_value = mock_instance

        gen = GendutyQAGenerator(
            genduty_path=genduty_path,
            regulations_path=regulations_path,
        )
        gen._llm = mock_instance

        entries = gen.generate()
        queries = [e["query"] for e in entries]
        assert len(queries) == len(set(q.strip().lower() for q in queries))

    def test_generate_missing_file_raises(self, tmp_path):
        gen = GendutyQAGenerator(
            genduty_path=tmp_path / "nonexistent.parquet",
        )
        with pytest.raises(FileNotFoundError):
            gen.generate()
