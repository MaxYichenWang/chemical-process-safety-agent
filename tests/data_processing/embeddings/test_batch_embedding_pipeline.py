"""
Unit tests for BatchEmbeddingPipeline
"""

import json
import os
import tempfile
from pathlib import Path
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

    # Ensure file doesn't exist before yielding
    if os.path.exists(db_path):
        os.remove(db_path)

    yield db_path

    # Cleanup
    if os.path.exists(db_path):
        os.remove(db_path)


@pytest.fixture
def mock_openai_client():
    """Create a mock OpenAI client."""
    with patch("src.data_processing.embeddings.batch_embedding_pipeline.OpenAI") as mock:
        client = MagicMock()
        mock.return_value = client
        yield client


def test_pipeline_init(temp_cache_db):
    """Test BatchEmbeddingPipeline initialization."""
    os.environ["OPENAI_API_KEY"] = "test_key"

    pipeline = BatchEmbeddingPipeline(cache_db=temp_cache_db)

    assert pipeline.model == "text-embedding-3-large"
    assert pipeline.dimensions == 1024
    assert pipeline.cache_db == temp_cache_db


def test_pipeline_init_no_api_key(temp_cache_db):
    """Test initialization without API key."""
    if "OPENAI_API_KEY" in os.environ:
        del os.environ["OPENAI_API_KEY"]

    with pytest.raises(ValueError, match="API key"):
        BatchEmbeddingPipeline(api_key=None, cache_db=temp_cache_db)


def test_create_batch_file(temp_cache_db, sample_chunks, mock_openai_client):
    """Test batch file creation."""
    os.environ["OPENAI_API_KEY"] = "test_key"
    pipeline = BatchEmbeddingPipeline(cache_db=temp_cache_db)

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "batch.jsonl"

        result_path, chunks_to_process = pipeline.create_batch_file(
            chunks=sample_chunks,
            output_path=str(output_path),
            use_cache=False
        )

        assert os.path.exists(result_path)
        assert len(chunks_to_process) == 2

        # Verify JSONL format
        with open(result_path, 'r') as f:
            lines = f.readlines()
            assert len(lines) == 2

            request = json.loads(lines[0])
            assert "custom_id" in request
            assert "body" in request
            assert request["body"]["model"] == "text-embedding-3-large"


def test_estimate_cost(temp_cache_db, sample_chunks, mock_openai_client):
    """Test cost estimation."""
    os.environ["OPENAI_API_KEY"] = "test_key"
    pipeline = BatchEmbeddingPipeline(cache_db=temp_cache_db)

    cost = pipeline.estimate_cost(sample_chunks)

    assert cost > 0
    assert isinstance(cost, float)


def test_get_cached_embeddings(temp_cache_db, mock_openai_client):
    """Test getting cached embeddings."""
    os.environ["OPENAI_API_KEY"] = "test_key"
    pipeline = BatchEmbeddingPipeline(cache_db=temp_cache_db)

    # Initially, no cached embeddings
    chunks = [
        {"chunk_id": "test_001", "hash": "hash001"}
    ]

    cached = pipeline.get_cached_embeddings(chunks)

    assert len(cached) == 0


def test_cache_embeddings(temp_cache_db, sample_chunks, mock_openai_client):
    """Test caching embeddings."""
    os.environ["OPENAI_API_KEY"] = "test_key"
    pipeline = BatchEmbeddingPipeline(cache_db=temp_cache_db)

    # Create sample embedding results
    results = [
        (sample_chunks[0], [0.1] * 1024),
        (sample_chunks[1], [0.2] * 1024)
    ]

    pipeline._cache_embeddings(results, "test_job_id")

    # Verify embeddings were cached
    cached = pipeline.get_cached_embeddings(sample_chunks)

    assert len(cached) == 2
    assert "test_001" in cached
    assert "test_002" in cached


def test_get_cache_stats(temp_cache_db, mock_openai_client):
    """Test getting cache statistics."""
    os.environ["OPENAI_API_KEY"] = "test_key"
    pipeline = BatchEmbeddingPipeline(cache_db=temp_cache_db)

    stats = pipeline.get_cache_stats()

    assert "cached_embeddings" in stats
    assert "jobs_by_status" in stats
    assert "total_cost_completed" in stats


def test_store_batch_job(temp_cache_db, mock_openai_client):
    """Test storing batch job info."""
    os.environ["OPENAI_API_KEY"] = "test_key"
    pipeline = BatchEmbeddingPipeline(cache_db=temp_cache_db)

    pipeline._store_batch_job(
        job_id="test_job",
        corpus="test_corpus",
        file_id="test_file",
        chunks_count=100,
        cost_estimate=1.5
    )

    # Verify job was stored
    job = pipeline.get_batch_job_status("test_job")

    assert job is not None
    assert job["corpus"] == "test_corpus"
    assert job["chunks_count"] == 100


def test_update_batch_job_status(temp_cache_db, mock_openai_client):
    """Test updating batch job status."""
    os.environ["OPENAI_API_KEY"] = "test_key"
    pipeline = BatchEmbeddingPipeline(cache_db=temp_cache_db)

    # Store a job first
    pipeline._store_batch_job(
        job_id="test_job",
        corpus="test_corpus",
        file_id="test_file",
        chunks_count=100,
        cost_estimate=1.5
    )

    # Update status
    pipeline._update_batch_job_status("test_job", "completed")

    # Verify update
    job = pipeline.get_batch_job_status("test_job")

    assert job["status"] == "completed"
    assert job["completed_at"] is not None


def test_list_batch_jobs(temp_cache_db, mock_openai_client):
    """Test listing batch jobs."""
    os.environ["OPENAI_API_KEY"] = "test_key"
    pipeline = BatchEmbeddingPipeline(cache_db=temp_cache_db)

    # Store multiple jobs
    pipeline._store_batch_job(
        job_id="job1",
        corpus="corpus1",
        file_id="file1",
        chunks_count=100,
        cost_estimate=1.0
    )
    pipeline._store_batch_job(
        job_id="job2",
        corpus="corpus2",
        file_id="file2",
        chunks_count=200,
        cost_estimate=2.0
    )

    # List all jobs
    jobs = pipeline.list_batch_jobs()

    assert len(jobs) >= 2

    # List jobs for specific corpus
    jobs_corpus1 = pipeline.list_batch_jobs(corpus="corpus1")

    assert len(jobs_corpus1) >= 1
    assert all(job["corpus"] == "corpus1" for job in jobs_corpus1)
