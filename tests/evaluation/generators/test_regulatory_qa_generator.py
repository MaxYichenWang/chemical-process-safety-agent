"""
Tests for src.evaluation.generators.regulatory_qa_generator.

Uses mocked DuckDB/parquet reads and LLM calls for deterministic,
cost-free testing.
"""

import json
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.evaluation.generators.regulatory_qa_generator import (
    RegulatoryQAGenerator,
    _DEFAULT_CHECKPOINT_EVERY,
    _DEFAULT_TARGET_ENTRIES,
)
from src.evaluation.schemas import validate_eval_entry


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _sample_regulations_df():
    """Small DataFrame mimicking regulations_sections.parquet."""
    return pd.DataFrame(
        [
            {
                "standard_number": "1910.119",
                "section_id": "1910.119(a)",
                "title": "Process Safety Management",
                "text": "This section contains requirements for preventing ...",
                "part_number": "1910",
                "subpart": "H",
                "url": "https://osha.gov/1910.119",
                "gpo_source": "e-CFR",
                "scraped_at": "2025-01-01",
                "hash": "abc123",
            },
            {
                "standard_number": "1910.119",
                "section_id": "1910.119(j)",
                "title": "Mechanical Integrity",
                "text": "Employers must establish procedures for mechanical ...",
                "part_number": "1910",
                "subpart": "H",
                "url": "https://osha.gov/1910.119",
                "gpo_source": "e-CFR",
                "scraped_at": "2025-01-01",
                "hash": "def456",
            },
            {
                "standard_number": "1910.147",
                "section_id": "1910.147(c)(1)",
                "title": "Lockout/Tagout",
                "text": "The employer shall establish a program for ...",
                "part_number": "1910",
                "subpart": "J",
                "url": "https://osha.gov/1910.147",
                "gpo_source": "e-CFR",
                "scraped_at": "2025-01-01",
                "hash": "ghi789",
            },
            {
                "standard_number": "1926.501",
                "section_id": "1926.501(b)(1)",
                "title": "Fall Protection",
                "text": "Each employee on a walking/working surface ...",
                "part_number": "1926",
                "subpart": "M",
                "url": "https://osha.gov/1926.501",
                "gpo_source": "e-CFR",
                "scraped_at": "2025-01-01",
                "hash": "jkl012",
            },
            {
                "standard_number": "1926.502",
                "section_id": "1926.502(a)",
                "title": "Fall Protection Systems",
                "text": "Fall protection systems required by this part ...",
                "part_number": "1926",
                "subpart": "M",
                "url": "https://osha.gov/1926.502",
                "gpo_source": "e-CFR",
                "scraped_at": "2025-01-01",
                "hash": "mno345",
            },
        ]
    )


def _sample_chunks_df():
    """Small DataFrame mimicking osha_regulations_chunks.parquet."""
    return pd.DataFrame(
        [
            {
                "chunk_id": "osha_reg_chunk_001",
                "section_id": "1910.119(a)",
                "standard_number": "1910.119",
            },
            {
                "chunk_id": "osha_reg_chunk_002",
                "section_id": "1910.119(j)",
                "standard_number": "1910.119",
            },
            {
                "chunk_id": "osha_reg_chunk_003",
                "section_id": "1910.147(c)(1)",
                "standard_number": "1910.147",
            },
            {
                "chunk_id": "osha_reg_chunk_004",
                "section_id": "1926.501(b)(1)",
                "standard_number": "1926.501",
            },
        ]
    )


def _mock_llm_response(
    question: str = "Refined question?", answer: str = "Expected answer."
):
    """Create a mock LLM response returning valid JSON."""
    payload = json.dumps(
        {
            "question": question,
            "expected_answer": answer,
            "key_claims": [
                {"claim": "Claim 1", "source_chunk": ""},
                {"claim": "Claim 2", "source_chunk": ""},
            ],
        }
    )
    mock_resp = MagicMock()
    mock_resp.content = payload
    return mock_resp


# ============================================================================
# _select_sections
# ============================================================================


class TestSelectSections:

    def test_filters_key_standards(self):
        gen = RegulatoryQAGenerator()
        df = _sample_regulations_df()
        selected = gen._select_sections(df)

        key_standards = selected[
            selected["standard_number"].str.startswith("1910.119")
            | selected["standard_number"].str.startswith("1910.147")
        ]
        assert len(key_standards) >= 3

    def test_includes_random_sample(self):
        gen = RegulatoryQAGenerator()
        df = _sample_regulations_df()
        selected = gen._select_sections(df)

        non_key = selected[
            ~(
                selected["standard_number"].str.startswith("1910.119")
                | selected["standard_number"].str.startswith("1910.147")
                | selected["standard_number"].str.startswith("1910.146")
                | selected["standard_number"].str.startswith("1910.1200")
                | selected["standard_number"].str.startswith("1910.134")
                | selected["standard_number"].str.startswith("1910.120")
                | selected["standard_number"].str.startswith("1910.106")
                | selected["standard_number"].str.startswith("1910.38")
            )
        ]
        # With only 2 non-key sections and min sample of 10, we get all of them
        assert len(non_key) <= 2

    def test_excludes_empty_text(self):
        gen = RegulatoryQAGenerator()
        df = _sample_regulations_df()
        df.loc[0, "text"] = ""
        selected = gen._select_sections(df)
        assert "1910.119(a)" not in selected["section_id"].values


# ============================================================================
# _apply_question_templates
# ============================================================================


class TestApplyQuestionTemplates:

    def test_generates_three_templates(self):
        gen = RegulatoryQAGenerator()
        section = {
            "standard_number": "1910.119",
            "section_id": "1910.119(a)",
            "title": "Process Safety Management",
            "text": "Requirements ...",
        }
        questions = gen._apply_question_templates(section)
        assert len(questions) == 3

    def test_template_names(self):
        gen = RegulatoryQAGenerator()
        section = {
            "standard_number": "1910.119",
            "section_id": "1910.119(a)",
            "title": "PSM",
            "text": "...",
        }
        questions = gen._apply_question_templates(section)
        template_names = {q["template"] for q in questions}
        assert template_names == {
            "requirement_lookup",
            "obligation_identification",
            "definition",
        }

    def test_standard_number_in_draft(self):
        gen = RegulatoryQAGenerator()
        section = {
            "standard_number": "1910.147",
            "section_id": "1910.147(c)(1)",
            "title": "LOTO",
            "text": "...",
        }
        questions = gen._apply_question_templates(section)
        for q in questions:
            assert "1910.147" in q["draft_question"]

    def test_section_with_empty_title(self):
        gen = RegulatoryQAGenerator()
        section = {
            "standard_number": "1910.119",
            "section_id": "1910.119(a)",
            "title": "",
            "text": "...",
        }
        questions = gen._apply_question_templates(section)
        assert len(questions) == 3
        # Definition template should fall back to section_id
        definition_q = [q for q in questions if q["template"] == "definition"][0]
        assert "1910.119(a)" in definition_q["draft_question"]


# ============================================================================
# _map_to_chunks
# ============================================================================


class TestMapToChunks:

    def test_finds_matching_chunks(self):
        gen = RegulatoryQAGenerator()
        chunks_df = _sample_chunks_df()
        result = gen._map_to_chunks("1910.119(a)", chunks_df)
        assert result == ["osha_reg_chunk_001"]

    def test_no_match_returns_empty(self):
        gen = RegulatoryQAGenerator()
        chunks_df = _sample_chunks_df()
        result = gen._map_to_chunks("1910.999(z)", chunks_df)
        assert result == []

    def test_multiple_chunks_for_section(self):
        gen = RegulatoryQAGenerator()
        chunks_df = _sample_chunks_df()
        extra_row = pd.DataFrame(
            [
                {
                    "chunk_id": "osha_reg_chunk_005",
                    "section_id": "1910.119(a)",
                    "standard_number": "1910.119",
                }
            ]
        )
        chunks_df = pd.concat([chunks_df, extra_row], ignore_index=True)
        result = gen._map_to_chunks("1910.119(a)", chunks_df)
        assert len(result) == 2


# ============================================================================
# _refine_with_llm
# ============================================================================


class TestRefineWithLlm:

    @patch("src.evaluation.generators.regulatory_qa_generator.ChatOpenAI")
    def test_returns_parsed_response(self, mock_chat_cls):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = _mock_llm_response(
            "What does PSM require?", "Employers must ..."
        )
        mock_chat_cls.return_value = mock_llm

        gen = RegulatoryQAGenerator()
        raw_q = {
            "template": "requirement_lookup",
            "draft_question": "Draft?",
            "section": {
                "standard_number": "1910.119",
                "section_id": "1910.119(a)",
                "title": "PSM",
                "text": "Requirements ...",
            },
        }
        result = gen._refine_with_llm(raw_q, ["chunk_001"])
        assert result is not None
        assert result["question"] == "What does PSM require?"
        assert result["expected_answer"] == "Employers must ..."
        assert len(result["key_claims"]) == 2

    @patch("src.evaluation.generators.regulatory_qa_generator.ChatOpenAI")
    def test_returns_none_on_bad_json(self, mock_chat_cls):
        mock_llm = MagicMock()
        bad_resp = MagicMock()
        bad_resp.content = "not valid json {"
        mock_llm.invoke.return_value = bad_resp
        mock_chat_cls.return_value = mock_llm

        gen = RegulatoryQAGenerator()
        raw_q = {
            "template": "requirement_lookup",
            "draft_question": "Draft?",
            "section": {
                "standard_number": "1910.119",
                "section_id": "1910.119(a)",
                "title": "PSM",
                "text": "Requirements ...",
            },
        }
        result = gen._refine_with_llm(raw_q, ["chunk_001"])
        assert result is None


# ============================================================================
# generate – end-to-end with mocks
# ============================================================================


class TestGenerate:

    @patch("src.evaluation.generators.regulatory_qa_generator.ChatOpenAI")
    def test_end_to_end_produces_valid_entries(self, mock_chat_cls):
        mock_llm = MagicMock()
        call_count = 0

        def side_effect(messages):
            nonlocal call_count
            call_count += 1
            return _mock_llm_response(
                f"Refined question {call_count}?",
                f"Expected answer {call_count}.",
            )

        mock_llm.invoke.side_effect = side_effect
        mock_chat_cls.return_value = mock_llm

        gen = RegulatoryQAGenerator()

        with (
            patch.object(
                gen, "_load_regulations", return_value=_sample_regulations_df()
            ),
            patch.object(gen, "_load_chunk_mapping", return_value=_sample_chunks_df()),
        ):
            entries = gen.generate()

        assert len(entries) > 0
        for entry in entries:
            errors = validate_eval_entry(entry)
            assert errors == [], f"Invalid entry: {errors}"
            assert entry["source"] == "auto_regulatory"
            assert entry["category"] == "regulatory_factual"
            assert entry["expected_agents"] == ["osha_regs"]

    @patch("src.evaluation.generators.regulatory_qa_generator.ChatOpenAI")
    def test_no_duplicate_queries(self, mock_chat_cls):
        mock_llm = MagicMock()
        # Return the same question every time to test dedup
        mock_llm.invoke.return_value = _mock_llm_response(
            "Same question?", "Same answer."
        )
        mock_chat_cls.return_value = mock_llm

        gen = RegulatoryQAGenerator()

        with (
            patch.object(
                gen, "_load_regulations", return_value=_sample_regulations_df()
            ),
            patch.object(gen, "_load_chunk_mapping", return_value=_sample_chunks_df()),
        ):
            entries = gen.generate()

        queries = [e["query"] for e in entries]
        assert len(queries) == len(set(q.strip().lower() for q in queries))

    @patch("src.evaluation.generators.regulatory_qa_generator.ChatOpenAI")
    def test_entries_have_metadata(self, mock_chat_cls):
        mock_llm = MagicMock()
        call_count = 0

        def side_effect(messages):
            nonlocal call_count
            call_count += 1
            return _mock_llm_response(
                f"Question {call_count}?", f"Answer {call_count}."
            )

        mock_llm.invoke.side_effect = side_effect
        mock_chat_cls.return_value = mock_llm

        gen = RegulatoryQAGenerator()

        with (
            patch.object(
                gen, "_load_regulations", return_value=_sample_regulations_df()
            ),
            patch.object(gen, "_load_chunk_mapping", return_value=_sample_chunks_df()),
        ):
            entries = gen.generate()

        for entry in entries:
            meta = entry.get("metadata", {})
            assert "standard_number" in meta
            assert "section_id" in meta
            assert "template" in meta

    def test_missing_regulations_parquet_raises(self, tmp_path):
        gen = RegulatoryQAGenerator(
            regulations_path=tmp_path / "nonexistent.parquet",
            chunks_path=tmp_path / "also_nonexistent.parquet",
        )
        with pytest.raises(FileNotFoundError, match="Regulations parquet"):
            gen.generate()

    def test_missing_chunks_parquet_raises(self, tmp_path):
        regs_path = tmp_path / "regs.parquet"
        _sample_regulations_df().to_parquet(regs_path)

        gen = RegulatoryQAGenerator(
            regulations_path=regs_path,
            chunks_path=tmp_path / "nonexistent_chunks.parquet",
        )
        with pytest.raises(FileNotFoundError, match="Chunks parquet"):
            gen.generate()

    @patch("src.evaluation.generators.regulatory_qa_generator.ChatOpenAI")
    def test_handles_llm_failure_gracefully(self, mock_chat_cls):
        """When all LLM calls fail, generate() returns an empty list."""
        mock_llm = MagicMock()
        bad_resp = MagicMock()
        bad_resp.content = "invalid json"
        mock_llm.invoke.return_value = bad_resp
        mock_chat_cls.return_value = mock_llm

        gen = RegulatoryQAGenerator()

        with (
            patch.object(
                gen, "_load_regulations", return_value=_sample_regulations_df()
            ),
            patch.object(gen, "_load_chunk_mapping", return_value=_sample_chunks_df()),
        ):
            entries = gen.generate()

        assert entries == []


# ============================================================================
# _load_regulations / _load_chunk_mapping with real parquet files
# ============================================================================


class TestDataLoading:

    def test_load_regulations_from_parquet(self, tmp_path):
        df = _sample_regulations_df()
        parquet_path = tmp_path / "regulations_sections.parquet"
        df.to_parquet(parquet_path)

        gen = RegulatoryQAGenerator(regulations_path=parquet_path)
        loaded = gen._load_regulations()
        assert len(loaded) == len(df)
        assert "standard_number" in loaded.columns

    def test_load_chunk_mapping_from_parquet(self, tmp_path):
        df = _sample_chunks_df()
        parquet_path = tmp_path / "osha_regulations_chunks.parquet"
        df.to_parquet(parquet_path)

        gen = RegulatoryQAGenerator(chunks_path=parquet_path)
        loaded = gen._load_chunk_mapping()
        assert len(loaded) == len(df)
        assert set(loaded.columns) == {"chunk_id", "section_id", "standard_number"}


# ============================================================================
# Helpers for large-DataFrame tests
# ============================================================================


def _large_regulations_df(n: int = 100) -> pd.DataFrame:
    """Build a DataFrame with *n* key-standard sections (all 1910.119)."""
    rows = []
    for i in range(n):
        rows.append(
            {
                "standard_number": "1910.119",
                "section_id": f"1910.119(sec_{i})",
                "title": f"PSM Section {i}",
                "text": f"Regulation text for section {i} ...",
                "part_number": "1910",
                "subpart": "H",
                "url": "https://osha.gov/1910.119",
                "gpo_source": "e-CFR",
                "scraped_at": "2025-01-01",
                "hash": f"hash_{i}",
            }
        )
    return pd.DataFrame(rows)


def _large_chunks_df(n: int = 100) -> pd.DataFrame:
    """Matching chunk_ids for :func:`_large_regulations_df`."""
    rows = []
    for i in range(n):
        rows.append(
            {
                "chunk_id": f"chunk_{i:04d}",
                "section_id": f"1910.119(sec_{i})",
                "standard_number": "1910.119",
            }
        )
    return pd.DataFrame(rows)


# ============================================================================
# Early stopping
# ============================================================================


class TestEarlyStopping:

    @patch("src.evaluation.generators.regulatory_qa_generator.ChatOpenAI")
    def test_stops_at_target_entries(self, mock_chat_cls):
        """Generation must stop once *target_entries* unique entries exist."""
        call_count = 0
        mock_llm = MagicMock()

        def side_effect(messages):
            nonlocal call_count
            call_count += 1
            return _mock_llm_response(
                f"Unique question {call_count}?",
                f"Answer {call_count}.",
            )

        mock_llm.invoke.side_effect = side_effect
        mock_chat_cls.return_value = mock_llm

        target = 5
        gen = RegulatoryQAGenerator(target_entries=target)

        with (
            patch.object(
                gen, "_load_regulations", return_value=_large_regulations_df(50)
            ),
            patch.object(gen, "_load_chunk_mapping", return_value=_large_chunks_df(50)),
        ):
            entries = gen.generate()

        assert len(entries) == target
        # With 50 sections * 3 templates = 150 possible LLM calls,
        # early stopping should have called far fewer.
        assert call_count <= target + 10

    @patch("src.evaluation.generators.regulatory_qa_generator.ChatOpenAI")
    def test_returns_fewer_if_data_exhausted(self, mock_chat_cls):
        """When data is smaller than target, return all valid entries."""
        call_count = 0
        mock_llm = MagicMock()

        def side_effect(messages):
            nonlocal call_count
            call_count += 1
            return _mock_llm_response(f"Q {call_count}?", f"A {call_count}.")

        mock_llm.invoke.side_effect = side_effect
        mock_chat_cls.return_value = mock_llm

        gen = RegulatoryQAGenerator(target_entries=100)

        with (
            patch.object(
                gen, "_load_regulations", return_value=_sample_regulations_df()
            ),
            patch.object(gen, "_load_chunk_mapping", return_value=_sample_chunks_df()),
        ):
            entries = gen.generate()

        # 5 sections * 3 templates = max 15 entries (with dedup)
        assert 0 < len(entries) <= 15
        assert len(entries) < 100

    def test_default_target_entries_constant(self):
        assert _DEFAULT_TARGET_ENTRIES == 55

    def test_default_checkpoint_every_constant(self):
        assert _DEFAULT_CHECKPOINT_EVERY == 10


# ============================================================================
# Checkpointing
# ============================================================================


class TestCheckpointing:

    @patch("src.evaluation.generators.regulatory_qa_generator.ChatOpenAI")
    def test_checkpoint_file_created(self, mock_chat_cls, tmp_path):
        """A checkpoint file is written when checkpoint_path is set."""
        call_count = 0
        mock_llm = MagicMock()

        def side_effect(messages):
            nonlocal call_count
            call_count += 1
            return _mock_llm_response(
                f"CKP question {call_count}?", f"CKP answer {call_count}."
            )

        mock_llm.invoke.side_effect = side_effect
        mock_chat_cls.return_value = mock_llm

        ckpt_path = tmp_path / "subdir" / "checkpoint.json"

        gen = RegulatoryQAGenerator(
            target_entries=5,
            checkpoint_path=ckpt_path,
            checkpoint_every=3,
        )

        with (
            patch.object(
                gen, "_load_regulations", return_value=_large_regulations_df(20)
            ),
            patch.object(gen, "_load_chunk_mapping", return_value=_large_chunks_df(20)),
        ):
            entries = gen.generate()

        assert ckpt_path.exists()
        with open(ckpt_path, "r", encoding="utf-8") as fh:
            saved = json.load(fh)
        assert len(saved) == len(entries)
        assert len(saved) == 5

    @patch("src.evaluation.generators.regulatory_qa_generator.ChatOpenAI")
    def test_checkpoint_written_before_completion(self, mock_chat_cls, tmp_path):
        """Checkpoint is flushed before the target is reached."""
        call_count = 0
        ckpt_sizes: list[int] = []
        mock_llm = MagicMock()

        def side_effect(messages):
            nonlocal call_count
            call_count += 1
            return _mock_llm_response(
                f"CKP2 question {call_count}?", f"CKP2 answer {call_count}."
            )

        mock_llm.invoke.side_effect = side_effect
        mock_chat_cls.return_value = mock_llm

        ckpt_path = tmp_path / "incremental.json"

        gen = RegulatoryQAGenerator(
            target_entries=10,
            checkpoint_path=ckpt_path,
            checkpoint_every=3,
        )

        original_write = gen._write_checkpoint

        def spy_write(entries):
            ckpt_sizes.append(len(entries))
            original_write(entries)

        with (
            patch.object(
                gen, "_load_regulations", return_value=_large_regulations_df(20)
            ),
            patch.object(gen, "_load_chunk_mapping", return_value=_large_chunks_df(20)),
            patch.object(gen, "_write_checkpoint", side_effect=spy_write),
        ):
            gen.generate()

        # There should be at least 2 checkpoint writes (intermediate + final)
        assert len(ckpt_sizes) >= 2
        # Sizes should be monotonically non-decreasing
        assert ckpt_sizes == sorted(ckpt_sizes)

    @patch("src.evaluation.generators.regulatory_qa_generator.ChatOpenAI")
    def test_no_checkpoint_without_path(self, mock_chat_cls, tmp_path):
        """No file is written when checkpoint_path is None."""
        call_count = 0
        mock_llm = MagicMock()

        def side_effect(messages):
            nonlocal call_count
            call_count += 1
            return _mock_llm_response(f"NoCKP {call_count}?", f"NoCKP {call_count}.")

        mock_llm.invoke.side_effect = side_effect
        mock_chat_cls.return_value = mock_llm

        gen = RegulatoryQAGenerator(
            target_entries=3,
            checkpoint_path=None,
        )

        with (
            patch.object(
                gen, "_load_regulations", return_value=_sample_regulations_df()
            ),
            patch.object(gen, "_load_chunk_mapping", return_value=_sample_chunks_df()),
        ):
            gen.generate()

        # No JSON file should appear in tmp_path
        assert list(tmp_path.glob("*.json")) == []
