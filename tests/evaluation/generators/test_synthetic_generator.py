"""
Tests for SyntheticGenerator behavior.
"""
import json
from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage

from src.evaluation.generators.synthetic_generator import SyntheticGenerator


@patch("src.evaluation.generators.synthetic_generator.ChatOpenAI")
@patch("src.evaluation.generators.synthetic_generator.duckdb")
def test_synthetic_initialization_and_generation(mock_duckdb, mock_llm_class):
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    # Mock DuckDB finding a chunk
    mock_cursor.fetchall.return_value = [("chunk_123", "Some dense regulation text")]
    mock_conn.execute.return_value = mock_cursor
    mock_duckdb.connect.return_value = mock_conn

    # Mock LLM returning valid JSON
    mock_llm_instance = MagicMock()
    mock_llm_instance.invoke.return_value = AIMessage(
        content=json.dumps({"query": "Test?", "expected_answer": "Yes."})
    )
    mock_llm_class.return_value = mock_llm_instance

    gen = SyntheticGenerator(target_size=1)
    results = gen.generate()

    assert len(results) == 1
    assert results[0]["query"] == "Test?"
    assert results[0]["ground_truth_chunks"] == ["chunk_123"]
    assert results[0]["category"] == "synthetic_reasoning"
