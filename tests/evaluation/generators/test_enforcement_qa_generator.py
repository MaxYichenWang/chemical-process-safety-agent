"""
Tests for src.evaluation.generators.enforcement_qa_generator.

Covers EnforcementQAGenerator: data loading, selection strategies,
question template application, LLM refinement, cross-corpus mapping,
and end-to-end generation with mocked dependencies.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.evaluation.generators.enforcement_qa_generator import (
    EnforcementQAGenerator,
    _ACCIDENT_KEYWORDS,
    _TEMPLATES,
    _classify_accident,
    _extract_year,
    _is_fatality_text,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_accidents_df(n: int = 20) -> pd.DataFrame:
    """Create a small synthetic accidents DataFrame.

    Mirrors the real schema: ``fatality`` bool column is always False, but
    the text prefix contains ``fatality=X`` for ~half the rows to match the
    actual parquet behaviour.
    """
    rows = []
    keywords_pool = [
        "explosion at the facility",
        "chemical exposure to fumes",
        "fall from scaffold",
        "struck by falling object",
        "confined space entry incident",
        "electrocution from power line",
        "routine workplace inspection",
        "employee injured in maintenance",
    ]
    for i in range(n):
        is_fatal = i % 2 == 0
        fatality_flag = "X" if is_fatal else ""
        keyword_text = keywords_pool[i % len(keywords_pool)]
        year = 2015 + (i % 10)
        rows.append({
            "chunk_id": f"osha_acc_{10000 + i}",
            "corpus": "osha_enf",
            "subcorpus": "osha_enf",
            "doc_type": "enforcement_accident",
            "text": (
                f"Accident {10000 + i} on {year}-06-15 00:00:00, "
                f"fatality={fatality_flag}, state_flag=. "
                f"Description: {keyword_text.upper()}. "
                f"Abstract: Employee was involved in a {keyword_text}. "
                f"The incident occurred at the processing plant. "
                f"Standard 1910.119 was referenced."
            ),
            "token_count": 100 + i * 10,
            "hash": f"sha256:test{i}",
            "schema_version": 1,
            "embeddings_version": "test",
            "build_manifest_id": "test",
            "source_path": "/test",
            "source_url": None,
            "created_at": "2025-01-01T00:00:00Z",
            "summary_nr": str(10000 + i),
            "event_date": f"{year}-06-15 00:00:00",
            "state": None,
            "fatality": False,
            "hospitalization": 0,
            "amputation": 0,
            "event_description": None,
            "event_keyword": None,
        })
    return pd.DataFrame(rows)


def _make_regulations_df(n: int = 5) -> pd.DataFrame:
    """Create a small synthetic regulations DataFrame."""
    rows = []
    for i in range(n):
        rows.append({
            "chunk_id": f"reg_chunk_{i:03d}",
            "standard_number": "1910.119" if i < 3 else "1910.147",
            "section_id": (
                f"1910.119({chr(97 + i)})" if i < 3
                else f"1910.147({chr(97 + i)})"
            ),
            "title": f"Section {i} Title",
        })
    return pd.DataFrame(rows)


def _mock_llm_response(
    question: str = "Refined question?",
    answer: str = "Expected answer.",
    claims: list | None = None,
):
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
# Helper functions
# ============================================================================


class TestHelperFunctions:

    def test_is_fatality_text_positive(self):
        assert _is_fatality_text("Accident 123, fatality=X, state_flag=.")

    def test_is_fatality_text_negative(self):
        assert not _is_fatality_text("Accident 123, fatality=, state_flag=.")

    def test_is_fatality_text_empty(self):
        assert not _is_fatality_text("")

    def test_extract_year_valid(self):
        assert _extract_year("2023-06-15 00:00:00") == 2023

    def test_extract_year_none(self):
        assert _extract_year(None) is None

    def test_extract_year_empty(self):
        assert _extract_year("") is None

    def test_extract_year_no_match(self):
        assert _extract_year("no year here") is None

    def test_classify_accident_explosion(self):
        assert _classify_accident("There was a large explosion at the plant") == "explosion_fire"

    def test_classify_accident_chemical(self):
        assert _classify_accident("Toxic chemical exposure occurred") == "chemical_exposure"

    def test_classify_accident_fall(self):
        assert _classify_accident("Employee fell from the scaffold") == "fall"

    def test_classify_accident_none(self):
        assert _classify_accident("routine paperwork") is None


# ============================================================================
# Data loading
# ============================================================================


class TestDataLoading:

    def test_missing_accidents_file(self, tmp_path):
        gen = EnforcementQAGenerator(
            accidents_path=tmp_path / "nonexistent.parquet"
        )
        with pytest.raises(FileNotFoundError, match="Accidents parquet"):
            gen._load_accidents()

    def test_missing_regulations_file(self, tmp_path):
        gen = EnforcementQAGenerator(
            regulations_path=tmp_path / "nonexistent.parquet"
        )
        with pytest.raises(FileNotFoundError, match="Regulations parquet"):
            gen._load_regulations_chunks()


# ============================================================================
# Selection strategies
# ============================================================================


class TestSelectionStrategies:

    def test_select_factual_accidents(self):
        df = _make_accidents_df(20)
        gen = EnforcementQAGenerator()
        selected = gen._select_factual_accidents(df)
        assert len(selected) > 0
        assert len(selected) <= 18  # _TARGET_FACTUAL

    def test_select_factual_prioritizes_fatalities(self):
        df = _make_accidents_df(20)
        gen = EnforcementQAGenerator()
        selected = gen._select_factual_accidents(df)
        fatality_count = selected["_is_fatality"].sum()
        assert fatality_count > 0

    def test_select_pattern_groups(self):
        df = _make_accidents_df(20)
        gen = EnforcementQAGenerator()
        groups = gen._select_pattern_groups(df)
        assert isinstance(groups, list)
        assert len(groups) > 0
        for group in groups:
            assert "pattern_type" in group
            assert "chunk_ids" in group
            assert len(group["chunk_ids"]) >= 2

    def test_select_pattern_groups_year_type(self):
        df = _make_accidents_df(20)
        gen = EnforcementQAGenerator()
        groups = gen._select_pattern_groups(df)
        year_groups = [g for g in groups if g["pattern_type"] == "year"]
        for g in year_groups:
            assert g["year"] is not None
            assert isinstance(g["year"], int)

    def test_select_pattern_groups_keyword_type(self):
        df = _make_accidents_df(20)
        gen = EnforcementQAGenerator()
        groups = gen._select_pattern_groups(df)
        keyword_groups = [g for g in groups if g["pattern_type"] == "keyword"]
        for g in keyword_groups:
            assert g["keyword"] is not None

    def test_find_regulation_chunks(self):
        regs_df = _make_regulations_df()
        gen = EnforcementQAGenerator()
        chunks = gen._find_regulation_chunks("1910.119", regs_df)
        assert len(chunks) > 0
        assert all(c.startswith("reg_chunk_") for c in chunks)

    def test_find_regulation_chunks_no_match(self):
        regs_df = _make_regulations_df()
        gen = EnforcementQAGenerator()
        chunks = gen._find_regulation_chunks("1910.999", regs_df)
        assert chunks == []


# ============================================================================
# Question templates
# ============================================================================


class TestTemplates:

    def test_all_templates_have_placeholders(self):
        for name, tpl in _TEMPLATES.items():
            assert "{" in tpl, f"Template '{name}' has no placeholders"

    def test_accident_factual_template_format(self):
        result = _TEMPLATES["accident_factual"].format(
            summary_nr="12345",
            event_date="2023-06-15",
        )
        assert "12345" in result
        assert "2023-06-15" in result

    def test_accident_pattern_template_format(self):
        result = _TEMPLATES["accident_pattern"].format(year=2023)
        assert "2023" in result

    def test_accident_pattern_keyword_template_format(self):
        result = _TEMPLATES["accident_pattern_keyword"].format(
            keyword="chemical exposure"
        )
        assert "chemical exposure" in result

    def test_accident_keywords_coverage(self):
        assert "chemical_exposure" in _ACCIDENT_KEYWORDS
        assert "explosion_fire" in _ACCIDENT_KEYWORDS
        assert "confined_space" in _ACCIDENT_KEYWORDS
        assert "fall" in _ACCIDENT_KEYWORDS


# ============================================================================
# LLM refinement
# ============================================================================


class TestLLMRefinement:

    @patch("src.evaluation.generators.enforcement_qa_generator.ChatOpenAI")
    def test_refine_with_llm_success(self, mock_chat):
        mock_instance = MagicMock()
        mock_instance.invoke.return_value = _mock_llm_response()
        mock_chat.return_value = mock_instance

        gen = EnforcementQAGenerator()
        gen._llm = mock_instance
        row = _make_accidents_df(1).iloc[0].to_dict()
        result = gen._refine_with_llm("Draft question?", row)

        assert result is not None
        assert "question" in result
        assert "expected_answer" in result
        assert "key_claims" in result

    @patch("src.evaluation.generators.enforcement_qa_generator.ChatOpenAI")
    def test_refine_with_llm_parse_failure(self, mock_chat):
        mock_instance = MagicMock()
        bad_response = MagicMock()
        bad_response.content = "not valid json"
        mock_instance.invoke.return_value = bad_response
        mock_chat.return_value = mock_instance

        gen = EnforcementQAGenerator()
        gen._llm = mock_instance
        row = _make_accidents_df(1).iloc[0].to_dict()
        result = gen._refine_with_llm("Draft question?", row)

        assert result is None


# ============================================================================
# Checkpointing
# ============================================================================


class TestCheckpointing:

    def test_load_existing_entries_no_file(self, tmp_path):
        gen = EnforcementQAGenerator(
            checkpoint_path=tmp_path / "missing.json",
        )
        assert gen._load_existing_entries() == []

    def test_load_existing_entries_valid(self, tmp_path):
        cp = tmp_path / "checkpoint.json"
        cp.write_text(json.dumps([{"query": "test?"}]))
        gen = EnforcementQAGenerator(checkpoint_path=cp)
        entries = gen._load_existing_entries()
        assert len(entries) == 1
        assert entries[0]["query"] == "test?"

    def test_load_existing_entries_invalid_json(self, tmp_path):
        cp = tmp_path / "bad.json"
        cp.write_text("not json")
        gen = EnforcementQAGenerator(checkpoint_path=cp)
        assert gen._load_existing_entries() == []

    def test_load_existing_entries_not_list(self, tmp_path):
        cp = tmp_path / "dict.json"
        cp.write_text(json.dumps({"key": "value"}))
        gen = EnforcementQAGenerator(checkpoint_path=cp)
        assert gen._load_existing_entries() == []

    def test_write_checkpoint(self, tmp_path):
        cp = tmp_path / "out.json"
        gen = EnforcementQAGenerator(checkpoint_path=cp)
        gen._write_checkpoint([{"query": "written"}])
        data = json.loads(cp.read_text())
        assert len(data) == 1
        assert data[0]["query"] == "written"

    def test_write_checkpoint_none_path(self):
        gen = EnforcementQAGenerator(checkpoint_path=None)
        gen._write_checkpoint([{"query": "ignored"}])
        # Should not raise


# ============================================================================
# End-to-end generation (with mocks)
# ============================================================================


class TestGenerateEndToEnd:

    @patch("src.evaluation.generators.enforcement_qa_generator.ChatOpenAI")
    def test_generate_produces_valid_entries(self, mock_chat, tmp_path):
        accidents_df = _make_accidents_df(20)
        regs_df = _make_regulations_df(5)

        accidents_path = tmp_path / "osha_accidents_chunks.parquet"
        regulations_path = tmp_path / "osha_regulations_chunks.parquet"
        accidents_df.to_parquet(accidents_path)
        regs_df.to_parquet(regulations_path)

        mock_instance = MagicMock()
        mock_instance.invoke.return_value = _mock_llm_response(
            question="What accident occurred?",
            answer="A workplace accident was documented.",
        )
        mock_chat.return_value = mock_instance

        gen = EnforcementQAGenerator(
            accidents_path=accidents_path,
            regulations_path=regulations_path,
        )
        gen._llm = mock_instance

        entries = gen.generate()

        assert isinstance(entries, list)
        assert len(entries) > 0

        for entry in entries:
            assert entry["source"] == "auto_enforcement"
            assert entry["category"] in (
                "enforcement_factual",
                "enforcement_analytical",
                "cross_corpus",
            )
            assert len(entry["ground_truth_chunks"]) > 0
            assert entry["query"]
            assert entry["expected_answer"]

    @patch("src.evaluation.generators.enforcement_qa_generator.ChatOpenAI")
    def test_generate_no_duplicate_queries(self, mock_chat, tmp_path):
        accidents_df = _make_accidents_df(10)
        regs_df = _make_regulations_df(3)

        accidents_path = tmp_path / "accidents.parquet"
        regulations_path = tmp_path / "regulations.parquet"
        accidents_df.to_parquet(accidents_path)
        regs_df.to_parquet(regulations_path)

        mock_instance = MagicMock()
        mock_instance.invoke.return_value = _mock_llm_response(
            question="Same question",
            answer="Same answer.",
        )
        mock_chat.return_value = mock_instance

        gen = EnforcementQAGenerator(
            accidents_path=accidents_path,
            regulations_path=regulations_path,
        )
        gen._llm = mock_instance

        entries = gen.generate()
        queries = [e["query"] for e in entries]
        assert len(queries) == len(set(q.strip().lower() for q in queries))

    @patch("src.evaluation.generators.enforcement_qa_generator.ChatOpenAI")
    def test_generate_with_checkpoint(self, mock_chat, tmp_path):
        accidents_df = _make_accidents_df(10)
        regs_df = _make_regulations_df(3)

        accidents_path = tmp_path / "accidents.parquet"
        regulations_path = tmp_path / "regulations.parquet"
        cp_path = tmp_path / "checkpoint.json"
        accidents_df.to_parquet(accidents_path)
        regs_df.to_parquet(regulations_path)

        mock_instance = MagicMock()
        mock_instance.invoke.return_value = _mock_llm_response(
            question="What accident occurred?",
            answer="A workplace accident was documented.",
        )
        mock_chat.return_value = mock_instance

        gen = EnforcementQAGenerator(
            accidents_path=accidents_path,
            regulations_path=regulations_path,
            checkpoint_path=cp_path,
            checkpoint_every=5,
        )
        gen._llm = mock_instance

        entries = gen.generate()
        assert isinstance(entries, list)
        assert cp_path.exists()

    @patch("src.evaluation.generators.enforcement_qa_generator.ChatOpenAI")
    def test_generate_respects_target(self, mock_chat, tmp_path):
        accidents_df = _make_accidents_df(30)
        regs_df = _make_regulations_df(5)

        accidents_path = tmp_path / "accidents.parquet"
        regulations_path = tmp_path / "regulations.parquet"
        accidents_df.to_parquet(accidents_path)
        regs_df.to_parquet(regulations_path)

        call_count = 0

        def unique_response(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return _mock_llm_response(
                question=f"Unique question {call_count}?",
                answer=f"Unique answer {call_count}.",
            )

        mock_instance = MagicMock()
        mock_instance.invoke.side_effect = unique_response
        mock_chat.return_value = mock_instance

        target = 10
        gen = EnforcementQAGenerator(
            accidents_path=accidents_path,
            regulations_path=regulations_path,
            target_entries=target,
        )
        gen._llm = mock_instance

        entries = gen.generate()
        assert len(entries) <= target

    def test_generate_missing_file_raises(self, tmp_path):
        gen = EnforcementQAGenerator(
            accidents_path=tmp_path / "nonexistent.parquet",
        )
        with pytest.raises(FileNotFoundError):
            gen.generate()

    @patch("src.evaluation.generators.enforcement_qa_generator.ChatOpenAI")
    def test_generate_cross_corpus_entries(self, mock_chat, tmp_path):
        accidents_df = _make_accidents_df(20)
        regs_df = _make_regulations_df(5)

        accidents_path = tmp_path / "accidents.parquet"
        regulations_path = tmp_path / "regulations.parquet"
        accidents_df.to_parquet(accidents_path)
        regs_df.to_parquet(regulations_path)

        call_count = 0

        def unique_response(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return _mock_llm_response(
                question=f"Cross-corpus question {call_count}?",
                answer=f"Cross-corpus answer {call_count}.",
            )

        mock_instance = MagicMock()
        mock_instance.invoke.side_effect = unique_response
        mock_chat.return_value = mock_instance

        gen = EnforcementQAGenerator(
            accidents_path=accidents_path,
            regulations_path=regulations_path,
            target_entries=30,
        )
        gen._llm = mock_instance

        entries = gen.generate()
        cross_corpus = [
            e for e in entries if e["category"] == "cross_corpus"
        ]
        # Our synthetic data includes "1910.119" in text, so cross-corpus
        # entries should be generated
        if cross_corpus:
            for cc in cross_corpus:
                assert "osha_enf" in cc["expected_agents"]
                assert "osha_regs" in cc["expected_agents"]
                assert cc["complexity"] == "multi_context"
                assert len(cc["ground_truth_chunks"]) >= 2
