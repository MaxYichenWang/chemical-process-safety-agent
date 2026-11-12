"""
Advanced Unit tests for BatchEmbeddingPipeline covering orchestration and API interaction.
"""

import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from src.data_processing.embeddings.batch_embedding_pipeline import BatchEmbeddingPipeline


@pytest.fixture
def sample_chunks():
    """Create sample chunks for testing."""
    return [
        {
            "chunk_id": "test_001",
            "text": "This is test chunk one.",
            "corpus": "test",
            "hash": "hash001",
            "token_count": 100
        },
        {
            "chunk_id": "test_002",
            "text": "This is test chunk two.",
            "corpus": "test",
            "hash": "hash002",
            "token_count": 120
        }
    ]


@pytest.fixture
def temp_cache_db():
    """Create a temporary cache database."""
    with tempfile.NamedTemporaryFile(suffix=".duckdb", delete=False) as f:
        db_path = f.name

    # Ensure file doesn't exist before yielding (DuckDB creates it)
    if os.path.exists(db_path):
        os.remove(db_path)

    yield db_path

    # Cleanup
    if os.path.exists(db_path):
        os.remove(db_path)


@pytest.fixture
def mock_pipeline(temp_cache_db):
    """Create a pipeline instance with mocked OpenAI client."""
    with patch("src.data_processing.embeddings.batch_embedding_pipeline.OpenAI") as mock_openai:
        os.environ["OPENAI_API_KEY"] = "test_key"
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        pipeline = BatchEmbeddingPipeline(cache_db=temp_cache_db)

        yield pipeline, mock_client


def test_monitor_batch_job_completion(mock_pipeline):
    """Test monitoring a batch job that completes successfully."""
    pipeline, client = mock_pipeline

    # Mock batch retrieval
    mock_batch = MagicMock()
    mock_batch.status = "completed"
    mock_batch.output_file_id = "file-output-123"
    mock_batch.output_file_id = "file-output-123"

    # Configure request_counts to be a MagicMock but with specific attributes
    # The error was because MagicMock() > 0 is not supported
    counts_mock = MagicMock()
    counts_mock.total = 10
    counts_mock.completed = 10
    counts_mock.failed = 0
    mock_batch.request_counts = counts_mock

    client.batches.retrieve.return_value = mock_batch

    # Test
    status = pipeline.monitor_batch_job("job-123", poll_interval=0.1)

    # Verify
    client.batches.retrieve.assert_called_with("job-123")
    assert status["status"] == "completed"
    assert status["output_file_id"] == "file-output-123"


def test_monitor_batch_job_failure(mock_pipeline):
    """Test monitoring a batch job that fails."""
    pipeline, client = mock_pipeline

    # Mock batch retrieval
    mock_batch = MagicMock()
    mock_batch.status = "failed"
    mock_batch.error_file_id = "file-error-123"

    # Even on failure, code might check counts
    counts_mock = MagicMock()
    counts_mock.total = 10
    counts_mock.completed = 0
    counts_mock.failed = 10
    mock_batch.request_counts = counts_mock

    client.batches.retrieve.return_value = mock_batch

    # Test
    status = pipeline.monitor_batch_job("job-123", poll_interval=0.1)

    # Verify
    assert status["status"] == "failed"
    assert status["error_file_id"] == "file-error-123"


def test_monitor_batch_job_wait_loop(mock_pipeline):
    """Test monitoring a batch job that waits before completing."""
    pipeline, client = mock_pipeline

    # Mock batch retrieval (in_progress then completed)
    mock_batch_progress = MagicMock()
    mock_batch_progress.status = "in_progress"

    counts_mock_progress = MagicMock()
    counts_mock_progress.total = 10
    counts_mock_progress.completed = 5
    counts_mock_progress.failed = 0
    mock_batch_progress.request_counts = counts_mock_progress

    mock_batch_done = MagicMock()
    mock_batch_done.status = "completed"
    mock_batch_done.status = "completed"
    mock_batch_done.output_file_id = "out-123"

    counts_mock_done = MagicMock()
    counts_mock_done.total = 10
    counts_mock_done.completed = 10
    counts_mock_done.failed = 0
    mock_batch_done.request_counts = counts_mock_done

    client.batches.retrieve.side_effect = [mock_batch_progress, mock_batch_done]

    # Test
    status = pipeline.monitor_batch_job("job-123", poll_interval=0.01)

    # Verify called twice
    assert client.batches.retrieve.call_count == 2
    assert status["status"] == "completed"


def test_download_batch_results(mock_pipeline, sample_chunks):
    """Test downloading and parsing batch results."""
    pipeline, client = mock_pipeline

    # Map chunk id to chunk
    chunks_by_id = {c["chunk_id"]: c for c in sample_chunks}

    # Mock file content
    results_jsonl = ""
    for chunk in sample_chunks:
        result = {
            "custom_id": chunk["chunk_id"],
            "response": {
                "status_code": 200,
                "body": {
                    "data": [{"embedding": [0.1] * 1024}]
                }
            }
        }
        results_jsonl += json.dumps(result) + "\n"

    mock_content = MagicMock()
    mock_content.read.return_value = results_jsonl.encode("utf-8")
    client.files.content.return_value = mock_content

    # Test
    results = pipeline.download_batch_results("job-123", "file-123", chunks_by_id)

    # Verify
    client.files.content.assert_called_with("file-123")
    assert len(results) == 2
    assert results[0][0]["chunk_id"] == "test_001"
    assert len(results[0][1]) == 1024

    # Verify results were cached
    cached = pipeline.get_cached_embeddings(sample_chunks)
    assert len(cached) == 2


def test_generate_embeddings_full_flow(mock_pipeline, sample_chunks):
    """Test full orchestration flow with mocks."""
    pipeline, client = mock_pipeline

    # 1. Mock Upload
    client.files.create.return_value.id = "file-upload-123"

    # 2. Mock Create Batch
    client.batches.create.return_value.id = "job-123"

    # 3. Mock Monitor
    mock_batch = MagicMock()
    mock_batch.status = "completed"
    mock_batch.status = "completed"
    mock_batch.output_file_id = "file-output-123"

    counts_mock = MagicMock()
    counts_mock.total = 2
    counts_mock.completed = 2
    counts_mock.failed = 0
    mock_batch.request_counts = counts_mock
    client.batches.retrieve.return_value = mock_batch

    # 4. Mock Download
    results_jsonl = ""
    # Only return result for first chunk to test mixing with cache/failures
    result = {
        "custom_id": sample_chunks[0]["chunk_id"],
        "response": {
            "status_code": 200,
            "body": {
                "data": [{"embedding": [0.1] * 1024}]
            }
        }
    }
    results_jsonl += json.dumps(result) + "\n"

    mock_content = MagicMock()
    mock_content.read.return_value = results_jsonl.encode("utf-8")
    client.files.content.return_value = mock_content

    # Run orchestration
    with tempfile.TemporaryDirectory() as tmpdir:
        result = pipeline.generate_embeddings_for_corpus(
            corpus="test",
            chunks=sample_chunks,
            output_dir=tmpdir,
            resume=False,
            wait_for_completion=True
        )

    # Verify
    assert result["status"] == "completed"
    assert result["processed_chunks"] == 1 # Defined by download mock
    assert result["total_chunks"] == 2
    assert len(result["results"]) == 1

    # Check DB job record
    job_status = pipeline.get_batch_job_status("job-123")
    assert job_status is not None
    assert job_status["status"] == "completed"


def test_generate_embeddings_all_cached(mock_pipeline, sample_chunks):
    """Test orchestration when everything is already cached."""
    pipeline, client = mock_pipeline

    # Pre-cache everything
    results = [(c, [0.1]*1024) for c in sample_chunks]
    pipeline._cache_embeddings(results, "pre-cache")

    # Run orchestration
    with tempfile.TemporaryDirectory() as tmpdir:
        result = pipeline.generate_embeddings_for_corpus(
            corpus="test",
            chunks=sample_chunks,
            output_dir=tmpdir,
            resume=True, # Important to use cache
            wait_for_completion=True
        )

    # Verify no API calls
    client.files.create.assert_not_called()
    client.batches.create.assert_not_called()

    assert result["status"] == "cached"
    assert result["processed_chunks"] == 2
    assert len(result["results"]) == 2


def test_generate_embeddings_no_wait(mock_pipeline, sample_chunks):
    """Test orchestration when wait_for_completion=False."""
    pipeline, client = mock_pipeline

    client.files.create.return_value.id = "file-123"
    client.batches.create.return_value.id = "job-123"

    with tempfile.TemporaryDirectory() as tmpdir:
        result = pipeline.generate_embeddings_for_corpus(
            corpus="test",
            chunks=sample_chunks,
            output_dir=tmpdir,
            resume=False,
            wait_for_completion=False
        )

    assert result["status"] == "submitted"
    assert "job-123" in result["job_ids"]

    # Monitor should not be called
    client.batches.retrieve.assert_not_called()
