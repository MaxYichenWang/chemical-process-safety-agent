"""
Tests for src.evaluation.generators.external_source_curator.
"""

import json
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.evaluation.generators.external_source_curator import ExternalSourceCurator


@pytest.fixture
def curator():
    with patch("src.evaluation.generators.external_source_curator.ChatOpenAI"):
        c = ExternalSourceCurator(llm_model="dummy")
        yield c


def test_curator_initialization(curator):
    assert curator.psm_test_file is not None


@patch("src.evaluation.generators.external_source_curator.ExternalSourceCurator._generate_psm_training_qs")
@patch("src.evaluation.generators.external_source_curator.ExternalSourceCurator._generate_csb_cross_corpus_qs")
@patch("src.evaluation.generators.external_source_curator.ExternalSourceCurator._generate_enforcement_analytical_qs")
def test_generate_combines_results(mock_enf, mock_csb, mock_psm, curator):
    mock_psm.return_value = [{"source": "external_osha"}]
    mock_csb.return_value = [{"source": "external_csb"}]
    mock_enf.return_value = [{"source": "external_osha", "category": "enforcement_analytical"}]

    entries = curator.generate()
    assert len(entries) == 3


@patch("src.evaluation.generators.external_source_curator.duckdb.connect")
def test_generate_enforcement_analytical_empty_db(mock_db, curator):
    mock_conn = MagicMock()
    mock_db.return_value = mock_conn
    mock_conn.execute.return_value.fetchall.return_value = []
    mock_conn.execute.return_value.fetchone.return_value = None

    entries = curator._generate_enforcement_analytical_qs()
    assert len(entries) == 0


def test_psm_training_file_links_local_evidence(curator, tmp_path, monkeypatch):
    regulations_path = tmp_path / "regulations.parquet"
    pd.DataFrame(
        [
            {
                "section_id": "example_restart_section",
                "chunk_id": "example_chunk",
                "text": "Review the example operating procedure before restart.",
            }
        ]
    ).to_parquet(regulations_path, index=False)
    curator.psm_test_file = tmp_path / "psm_training_test.json"
    curator.psm_test_file.write_text(
        json.dumps(
            [
                {
                    "question": "Which procedure does the supplied example require?",
                    "correct_answer": "Review the example operating procedure before restart.",
                    "reference": "example_restart_section",
                    "type": "short_answer",
                }
            ]
        )
    )
    monkeypatch.setattr(
        "src.evaluation.generators.external_source_curator.REGULATIONS_CHUNKS_PARQUET",
        regulations_path,
    )
    curator._llm.invoke.return_value.content = "entailed"

    entries = curator._generate_psm_training_qs()

    assert len(entries) == 1
    assert entries[0]["ground_truth_chunks"] == ["example_chunk"]
    assert entries[0]["metadata"]["reference"] == "example_restart_section"
    curator._llm.invoke.assert_called_once()


def test_optional_psm_training_file_can_be_absent(curator, tmp_path):
    curator.psm_test_file = tmp_path / "missing.json"

    assert curator._generate_psm_training_qs() == []
    curator._llm.invoke.assert_not_called()
