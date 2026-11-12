"""
Tests for src.evaluation.generators.incident_qa_generator.

Uses mocked file system, DuckDB/parquet reads, and LLM calls for
deterministic, cost-free testing.
"""

import json
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.evaluation.generators.incident_qa_generator import IncidentQAGenerator
from src.evaluation.schemas import validate_eval_entry


# ---------------------------------------------------------------------------
# Fixtures – sample CSB data
# ---------------------------------------------------------------------------


def _make_final_report_json(incident_name="Test_Explosion"):
    """Minimal CSB final report structured JSON."""
    return {
        "metadata": {
            "incident_name": incident_name,
            "report_name": "final_report",
            "document_type": "final_report",
            "num_pages": 3,
        },
        "pages": [
            {"page_number": 1, "text": "Executive Summary: An explosion occurred ..."},
            {"page_number": 2, "text": "The incident involved ammonium nitrate ..."},
            {"page_number": 3, "text": "Key Findings: Multiple safety failures ..."},
        ],
        "sections": [
            {
                "title": "Executive Summary",
                "level": 1,
                "content": "An explosion occurred at the facility on Jan 1.",
            },
            {
                "title": "Key Findings",
                "level": 1,
                "content": "Multiple safety failures were identified including lack of PSM.",
            },
            {
                "title": "Root Cause",
                "level": 1,
                "content": "The root cause was inadequate process hazard analysis.",
            },
        ],
    }


def _make_recommendation_json(incident_name="Test_Explosion"):
    """Minimal CSB recommendation structured JSON."""
    return {
        "metadata": {
            "incident_name": incident_name,
            "report_name": "recommendation_1",
            "document_type": "recommendation",
            "num_pages": 1,
        },
        "pages": [
            {"page_number": 1, "text": "Recommendation: Implement PSM program ..."},
        ],
        "sections": [
            {
                "title": "Recommendation Text",
                "level": 1,
                "content": "The CSB recommends that the facility implement a comprehensive PSM program.",
            },
            {
                "title": "Recipient",
                "level": 1,
                "content": "Facility Owner Corp.",
            },
            {
                "title": "New Status",
                "level": 1,
                "content": "Open - Acceptable Response",
            },
        ],
    }


def _sample_reports_chunks_df():
    """Mimics csb_final_reports_chunks.parquet."""
    return pd.DataFrame(
        [
            {
                "chunk_id": "csb_chunk_001",
                "incident_name": "Test_Explosion",
                "report_name": "final_report",
            },
            {
                "chunk_id": "csb_chunk_002",
                "incident_name": "Test_Explosion",
                "report_name": "final_report",
            },
            {
                "chunk_id": "csb_chunk_003",
                "incident_name": "Another_Incident",
                "report_name": "final_report",
            },
        ]
    )


def _sample_recs_chunks_df():
    """Mimics csb_recommendations_chunks.parquet."""
    return pd.DataFrame(
        [
            {
                "chunk_id": "csb_rec_chunk_001",
                "incident_name": "Test_Explosion",
                "report_name": "recommendation_1",
            },
        ]
    )


def _mock_llm_response(question="Refined?", answer="Expected answer."):
    payload = json.dumps(
        {
            "question": question,
            "expected_answer": answer,
            "key_claims": [
                {"claim": "Claim A", "source_chunk": ""},
            ],
        }
    )
    mock_resp = MagicMock()
    mock_resp.content = payload
    return mock_resp


def _mock_nli_entailed():
    mock_resp = MagicMock()
    mock_resp.content = "entailed"
    return mock_resp


def _create_test_json_dir(tmp_path, incidents=None):
    """Create a temporary CSB JSON directory structure."""
    if incidents is None:
        incidents = [
            ("Test_Explosion", "final_report", _make_final_report_json("Test_Explosion")),
            ("Test_Explosion", "recommendation_1", _make_recommendation_json("Test_Explosion")),
            ("Another_Incident", "final_report", _make_final_report_json("Another_Incident")),
        ]

    json_dir = tmp_path / "json"
    for incident_name, report_name, data in incidents:
        incident_dir = json_dir / incident_name
        incident_dir.mkdir(parents=True, exist_ok=True)
        json_path = incident_dir / f"{report_name}_structured.json"
        json_path.write_text(json.dumps(data), encoding="utf-8")

    return json_dir


# ============================================================================
# _classify_document
# ============================================================================


class TestClassifyDocument:

    def test_final_report_from_metadata(self, tmp_path):
        gen = IncidentQAGenerator(csb_json_dir=tmp_path, csb_toc_dir=tmp_path)
        data = _make_final_report_json()
        assert gen._classify_document(data, "Test", "report") == "final_report"

    def test_recommendation_from_metadata(self, tmp_path):
        gen = IncidentQAGenerator(csb_json_dir=tmp_path, csb_toc_dir=tmp_path)
        data = _make_recommendation_json()
        assert gen._classify_document(data, "Test", "rec") == "recommendation"

    def test_fallback_to_toc_exists(self, tmp_path):
        toc_dir = tmp_path / "toc" / "Test"
        toc_dir.mkdir(parents=True)
        (toc_dir / "report_toc.json").write_text("{}", encoding="utf-8")

        gen = IncidentQAGenerator(
            csb_json_dir=tmp_path,
            csb_toc_dir=tmp_path / "toc",
        )
        data = {"metadata": {}}  # no document_type
        assert gen._classify_document(data, "Test", "report") == "final_report"

    def test_fallback_no_toc(self, tmp_path):
        gen = IncidentQAGenerator(
            csb_json_dir=tmp_path,
            csb_toc_dir=tmp_path / "toc",
        )
        data = {"metadata": {}}
        assert gen._classify_document(data, "Test", "report") == "recommendation"


# ============================================================================
# _extract_report_data
# ============================================================================


class TestExtractReportData:

    def test_extracts_matching_sections(self):
        gen = IncidentQAGenerator()
        data = _make_final_report_json()
        result = gen._extract_report_data(data)
        titles = [r["section_title"] for r in result]
        assert "Key Findings" in titles
        assert "Root Cause" in titles
        assert "Executive Summary" in titles

    def test_skips_empty_sections(self):
        gen = IncidentQAGenerator()
        data = _make_final_report_json()
        data["sections"].append(
            {"title": "Key Findings", "level": 1, "content": ""}
        )
        result = gen._extract_report_data(data)
        # empty content section should not be included again
        for r in result:
            assert r["text"].strip() != ""

    def test_fallback_to_full_text(self):
        gen = IncidentQAGenerator()
        data = {
            "metadata": {},
            "pages": [
                {"page_number": 1, "text": "Some page content here."},
            ],
        }
        result = gen._extract_report_data(data)
        assert len(result) == 1
        assert result[0]["section_title"] == "Full Report"

    def test_no_content_returns_empty(self):
        gen = IncidentQAGenerator()
        data = {"metadata": {}, "pages": [], "sections": []}
        result = gen._extract_report_data(data)
        assert result == []


# ============================================================================
# _extract_recommendation_data
# ============================================================================


class TestExtractRecommendationData:

    def test_extracts_sections(self):
        gen = IncidentQAGenerator()
        data = _make_recommendation_json()
        result = gen._extract_recommendation_data(data)
        assert len(result) >= 3
        titles = [r["section_title"] for r in result]
        assert "Recommendation Text" in titles
        assert "Recipient" in titles
        assert "New Status" in titles

    def test_recommendation_number_field(self):
        gen = IncidentQAGenerator()
        data = _make_recommendation_json()
        result = gen._extract_recommendation_data(data)
        rec_text_items = [
            r for r in result if "recommendation_number" in r
        ]
        assert len(rec_text_items) >= 1

    def test_fallback_to_pages(self):
        gen = IncidentQAGenerator()
        data = {
            "metadata": {},
            "pages": [{"page_number": 1, "text": "Recommendation content."}],
        }
        result = gen._extract_recommendation_data(data)
        assert len(result) == 1
        assert result[0]["section_title"] == "Full Recommendation"


# ============================================================================
# _generate_questions
# ============================================================================


class TestGenerateQuestions:

    def test_report_templates(self):
        gen = IncidentQAGenerator()
        extracted = [
            {"section_title": "Key Findings", "text": "Findings ..."},
        ]
        questions = gen._generate_questions(extracted, "final_report", "Test_Fire")
        assert len(questions) == 3
        for q in questions:
            assert "Test Fire" in q["draft_question"]  # underscore replaced

    def test_recommendation_templates_with_number(self):
        gen = IncidentQAGenerator()
        extracted = [
            {
                "section_title": "Recommendation Text",
                "text": "Implement PSM ...",
                "recommendation_number": "2024-01-I-TX-R1",
            },
        ]
        questions = gen._generate_questions(
            extracted, "recommendation", "Test_Fire"
        )
        # All 3 templates should produce questions since we have a rec number
        assert len(questions) == 3

    def test_recommendation_templates_without_number(self):
        gen = IncidentQAGenerator()
        extracted = [
            {"section_title": "Some Section", "text": "Content ..."},
        ]
        questions = gen._generate_questions(
            extracted, "recommendation", "Test_Fire"
        )
        # Templates with {recommendation_number} should be skipped (no rec number)
        for q in questions:
            assert "N/A" not in q["draft_question"]


# ============================================================================
# _map_to_chunks
# ============================================================================


class TestMapToChunks:

    def test_finds_matching_chunks(self):
        gen = IncidentQAGenerator()
        chunks_df = _sample_reports_chunks_df()
        result = gen._map_to_chunks("Test_Explosion", "final_report", chunks_df)
        assert "csb_chunk_001" in result
        assert "csb_chunk_002" in result
        assert "csb_chunk_003" not in result

    def test_no_match_returns_empty(self):
        gen = IncidentQAGenerator()
        chunks_df = _sample_reports_chunks_df()
        result = gen._map_to_chunks("Nonexistent", "report", chunks_df)
        assert result == []



# ============================================================================
# _scan_structured_jsons
# ============================================================================


class TestScanStructuredJsons:

    def test_finds_json_files(self, tmp_path):
        json_dir = _create_test_json_dir(tmp_path)
        gen = IncidentQAGenerator(
            csb_json_dir=json_dir, csb_toc_dir=tmp_path / "toc"
        )
        results = gen._scan_structured_jsons()
        assert len(results) == 3
        incident_names = {r["incident_name"] for r in results}
        assert "Test_Explosion" in incident_names
        assert "Another_Incident" in incident_names

    def test_missing_dir_raises(self, tmp_path):
        gen = IncidentQAGenerator(csb_json_dir=tmp_path / "nonexistent")
        with pytest.raises(FileNotFoundError, match="CSB JSON directory"):
            gen._scan_structured_jsons()

    def test_empty_dir(self, tmp_path):
        json_dir = tmp_path / "json"
        json_dir.mkdir()
        gen = IncidentQAGenerator(csb_json_dir=json_dir, csb_toc_dir=tmp_path)
        results = gen._scan_structured_jsons()
        assert results == []


# ============================================================================
# generate – end-to-end with mocks
# ============================================================================


class TestGenerate:

    @patch("src.evaluation.generators.incident_qa_generator.ChatOpenAI")
    def test_end_to_end_produces_valid_entries(self, mock_chat_cls, tmp_path):
        call_count = 0
        mock_llm = MagicMock()

        def side_effect(messages):
            nonlocal call_count
            call_count += 1
            # Alternate between refine responses and NLI entailed
            content = messages[-1].content if messages else ""
            if "entailed" in content.lower() or "contradicted" in content.lower():
                return _mock_nli_entailed()
            return _mock_llm_response(
                f"Refined question {call_count}?",
                f"Answer {call_count}.",
            )

        mock_llm.invoke.side_effect = side_effect
        mock_chat_cls.return_value = mock_llm

        json_dir = _create_test_json_dir(tmp_path)

        gen = IncidentQAGenerator(
            csb_json_dir=json_dir,
            csb_toc_dir=tmp_path / "toc",
        )

        reports_chunks = _sample_reports_chunks_df()
        recs_chunks = _sample_recs_chunks_df()

        with (
            patch.object(gen, "_load_chunks_df") as mock_load_chunks,
        ):
            def load_chunks_side_effect(doc_type):
                if doc_type == "final_report":
                    return reports_chunks
                return recs_chunks

            mock_load_chunks.side_effect = load_chunks_side_effect
            entries = gen.generate()

        assert len(entries) > 0
        for entry in entries:
            errors = validate_eval_entry(entry)
            assert errors == [], f"Invalid entry: {errors}"
            assert entry["source"] == "auto_incident"
            assert entry["category"] == "incident_factual"

    @patch("src.evaluation.generators.incident_qa_generator.ChatOpenAI")
    def test_no_duplicate_queries(self, mock_chat_cls, tmp_path):
        mock_llm = MagicMock()

        def side_effect(messages):
            content = messages[-1].content if messages else ""
            if "entailed" in content.lower() or "contradicted" in content.lower():
                return _mock_nli_entailed()
            return _mock_llm_response("Same question?", "Same answer.")

        mock_llm.invoke.side_effect = side_effect
        mock_chat_cls.return_value = mock_llm

        json_dir = _create_test_json_dir(tmp_path)
        gen = IncidentQAGenerator(
            csb_json_dir=json_dir,
            csb_toc_dir=tmp_path / "toc",
        )

        with patch.object(gen, "_load_chunks_df") as mock_load_chunks:
            mock_load_chunks.return_value = _sample_reports_chunks_df()
            entries = gen.generate()

        queries = [e["query"] for e in entries]
        normalized = [q.strip().lower() for q in queries]
        assert len(normalized) == len(set(normalized))

    @patch("src.evaluation.generators.incident_qa_generator.ChatOpenAI")
    def test_entries_have_incident_metadata(self, mock_chat_cls, tmp_path):
        call_count = 0
        mock_llm = MagicMock()

        def side_effect(messages):
            nonlocal call_count
            call_count += 1
            content = messages[-1].content if messages else ""
            if "entailed" in content.lower() or "contradicted" in content.lower():
                return _mock_nli_entailed()
            return _mock_llm_response(
                f"Question {call_count}?", f"Answer {call_count}."
            )

        mock_llm.invoke.side_effect = side_effect
        mock_chat_cls.return_value = mock_llm

        json_dir = _create_test_json_dir(tmp_path)
        gen = IncidentQAGenerator(
            csb_json_dir=json_dir,
            csb_toc_dir=tmp_path / "toc",
        )

        with patch.object(gen, "_load_chunks_df") as mock_load_chunks:
            mock_load_chunks.return_value = _sample_reports_chunks_df()
            entries = gen.generate()

        for entry in entries:
            meta = entry.get("metadata", {})
            assert "incident_name" in meta
            assert "report_name" in meta
            assert "document_type" in meta

    @patch("src.evaluation.generators.incident_qa_generator.ChatOpenAI")
    def test_covers_multiple_incidents(self, mock_chat_cls, tmp_path):
        call_count = 0
        mock_llm = MagicMock()

        def side_effect(messages):
            nonlocal call_count
            call_count += 1
            content = messages[-1].content if messages else ""
            if "entailed" in content.lower() or "contradicted" in content.lower():
                return _mock_nli_entailed()
            return _mock_llm_response(
                f"Unique question {call_count}?", f"Answer {call_count}."
            )

        mock_llm.invoke.side_effect = side_effect
        mock_chat_cls.return_value = mock_llm

        json_dir = _create_test_json_dir(tmp_path)
        gen = IncidentQAGenerator(
            csb_json_dir=json_dir,
            csb_toc_dir=tmp_path / "toc",
        )

        with patch.object(gen, "_load_chunks_df") as mock_load_chunks:
            mock_load_chunks.return_value = _sample_reports_chunks_df()
            entries = gen.generate()

        incidents = {e["metadata"]["incident_name"] for e in entries}
        assert len(incidents) >= 2

    def test_missing_json_dir_raises(self, tmp_path):
        gen = IncidentQAGenerator(csb_json_dir=tmp_path / "nonexistent")
        with pytest.raises(FileNotFoundError, match="CSB JSON directory"):
            gen.generate()

    @patch("src.evaluation.generators.incident_qa_generator.ChatOpenAI")
    def test_handles_missing_chunks_gracefully(self, mock_chat_cls, tmp_path):
        """When chunk parquets are missing, generator still works with empty mapping."""
        call_count = 0
        mock_llm = MagicMock()

        def side_effect(messages):
            nonlocal call_count
            call_count += 1
            content = messages[-1].content if messages else ""
            if "entailed" in content.lower() or "contradicted" in content.lower():
                return _mock_nli_entailed()
            return _mock_llm_response(
                f"Question {call_count}?", f"Answer {call_count}."
            )

        mock_llm.invoke.side_effect = side_effect
        mock_chat_cls.return_value = mock_llm

        json_dir = _create_test_json_dir(tmp_path)
        gen = IncidentQAGenerator(
            csb_json_dir=json_dir,
            csb_toc_dir=tmp_path / "toc",
            reports_chunks_path=tmp_path / "missing.parquet",
            recs_chunks_path=tmp_path / "also_missing.parquet",
        )

        entries = gen.generate()
        # Should still produce entries, but with empty ground_truth_chunks
        assert len(entries) > 0
        for entry in entries:
            assert entry["ground_truth_chunks"] == []
