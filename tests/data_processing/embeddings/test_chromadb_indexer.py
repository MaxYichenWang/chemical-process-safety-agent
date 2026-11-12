"""
Unit tests for ChromaDBIndexer
"""

import tempfile
from pathlib import Path

import pytest

from src.data_processing.embeddings.chromadb_indexer import ChromaDBIndexer


@pytest.fixture
def temp_chromadb_dir():
    """Create a temporary ChromaDB directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def sample_chunks_with_embeddings():
    """Create sample chunks with embeddings."""
    chunks = [
        {
            "chunk_id": "test_001",
            "text": "This is test chunk one.",
            "corpus": "osha_reg",
            "subcorpus": "regulations",
            "hash": "hash001"
        },
        {
            "chunk_id": "test_002",
            "text": "This is test chunk two.",
            "corpus": "osha_reg",
            "subcorpus": "regulations",
            "hash": "hash002"
        }
    ]

    embeddings = [
        [0.1] * 1024,
        [0.2] * 1024
    ]

    return [(chunk, embedding) for chunk, embedding in zip(chunks, embeddings)]


def test_indexer_init(temp_chromadb_dir):
    """Test ChromaDBIndexer initialization."""
    indexer = ChromaDBIndexer(persist_directory=temp_chromadb_dir)

    assert indexer.persist_directory == temp_chromadb_dir
    assert "osha_reg" in indexer.collection_mapping


def test_get_or_create_collection(temp_chromadb_dir):
    """Test getting or creating a collection."""
    indexer = ChromaDBIndexer(persist_directory=temp_chromadb_dir)

    collection = indexer.get_or_create_collection("test_collection")

    assert collection is not None
    assert "test_collection" in indexer._collections


def test_index_chunks(temp_chromadb_dir, sample_chunks_with_embeddings):
    """Test indexing chunks into a collection."""
    indexer = ChromaDBIndexer(persist_directory=temp_chromadb_dir)

    result = indexer.index_chunks(
        corpus="osha_reg",
        chunks_with_embeddings=sample_chunks_with_embeddings,
        show_progress=False
    )

    assert result["status"] == "completed"
    assert result["indexed_count"] == 2
    assert result["collection"] == "osha_regulations"


def test_index_chunks_empty(temp_chromadb_dir):
    """Test indexing with empty chunks list."""
    indexer = ChromaDBIndexer(persist_directory=temp_chromadb_dir)

    result = indexer.index_chunks(
        corpus="osha_reg",
        chunks_with_embeddings=[],
        show_progress=False
    )

    assert result["status"] == "no_data"
    assert result["indexed_count"] == 0


def test_index_all_chunks(temp_chromadb_dir, sample_chunks_with_embeddings):
    """Test indexing all chunks across multiple corpora."""
    indexer = ChromaDBIndexer(persist_directory=temp_chromadb_dir)

    chunks_by_corpus = {
        "osha_reg": sample_chunks_with_embeddings[:1],
        "csb_reports": sample_chunks_with_embeddings[1:]
    }

    results = indexer.index_all_chunks(
        chunks_by_corpus=chunks_by_corpus,
        show_progress=False
    )

    assert results["total_indexed"] == 2
    assert len(results["collections"]) == 2


def test_get_collection_stats(temp_chromadb_dir, sample_chunks_with_embeddings):
    """Test getting collection statistics."""
    indexer = ChromaDBIndexer(persist_directory=temp_chromadb_dir)

    # Index some chunks first
    indexer.index_chunks(
        corpus="osha_reg",
        chunks_with_embeddings=sample_chunks_with_embeddings,
        show_progress=False
    )

    # Get stats
    stats = indexer.get_collection_stats("osha_regulations")

    assert stats["exists"] is True
    assert stats["count"] == 2
    assert stats["name"] == "osha_regulations"


def test_get_collection_stats_nonexistent(temp_chromadb_dir):
    """Test getting stats for non-existent collection."""
    indexer = ChromaDBIndexer(persist_directory=temp_chromadb_dir)

    stats = indexer.get_collection_stats("nonexistent_collection")

    assert stats["exists"] is False


def test_get_all_collection_stats(temp_chromadb_dir, sample_chunks_with_embeddings):
    """Test getting statistics for all collections."""
    indexer = ChromaDBIndexer(persist_directory=temp_chromadb_dir)

    # Index some chunks
    indexer.index_chunks(
        corpus="osha_reg",
        chunks_with_embeddings=sample_chunks_with_embeddings,
        show_progress=False
    )

    # Get all stats
    stats = indexer.get_all_collection_stats()

    assert "collections" in stats
    assert "total_chunks" in stats
    assert stats["total_chunks"] > 0


def test_verify_indexing(temp_chromadb_dir, sample_chunks_with_embeddings):
    """Test verifying indexing results."""
    indexer = ChromaDBIndexer(persist_directory=temp_chromadb_dir)

    # Index chunks
    indexer.index_chunks(
        corpus="osha_reg",
        chunks_with_embeddings=sample_chunks_with_embeddings,
        show_progress=False
    )

    # Verify
    expected_chunks = {"osha_reg": 2}
    verification = indexer.verify_indexing(expected_chunks)

    assert verification["verified"] is True
    assert verification["total_expected"] == 2
    assert verification["total_actual"] == 2


def test_verify_indexing_mismatch(temp_chromadb_dir, sample_chunks_with_embeddings):
    """Test verification with count mismatch."""
    indexer = ChromaDBIndexer(persist_directory=temp_chromadb_dir)

    # Index chunks
    indexer.index_chunks(
        corpus="osha_reg",
        chunks_with_embeddings=sample_chunks_with_embeddings,
        show_progress=False
    )

    # Verify with wrong expected count
    expected_chunks = {"osha_reg": 5}  # Wrong count
    verification = indexer.verify_indexing(expected_chunks)

    assert verification["verified"] is False


def test_upsert_chunks(temp_chromadb_dir, sample_chunks_with_embeddings):
    """Test upserting chunks."""
    indexer = ChromaDBIndexer(persist_directory=temp_chromadb_dir)

    result = indexer.upsert_chunks(
        corpus="osha_reg",
        chunks_with_embeddings=sample_chunks_with_embeddings,
        show_progress=False
    )

    assert result["status"] == "completed"
    assert result["upserted_count"] == 2


def test_delete_collection(temp_chromadb_dir, sample_chunks_with_embeddings):
    """Test deleting a collection."""
    indexer = ChromaDBIndexer(persist_directory=temp_chromadb_dir)

    # Create a collection first
    indexer.index_chunks(
        corpus="osha_reg",
        chunks_with_embeddings=sample_chunks_with_embeddings,
        show_progress=False
    )

    # Delete it
    success = indexer.delete_collection("osha_regulations")

    assert success is True


def test_prepare_metadata(temp_chromadb_dir):
    """Test metadata preparation for ChromaDB."""
    indexer = ChromaDBIndexer(persist_directory=temp_chromadb_dir)

    chunk = {
        "chunk_id": "test_001",
        "text": "This should not be in metadata",
        "corpus": "test",
        "token_count": 100,
        "complex_field": {"nested": "value"}
    }

    metadata = indexer._prepare_metadata(chunk)

    # text should not be in metadata
    assert "text" not in metadata
    # Other fields should be present
    assert metadata["chunk_id"] == "test_001"
    assert metadata["corpus"] == "test"
    assert metadata["token_count"] == 100
    # Complex fields should be converted to string
    assert isinstance(metadata["complex_field"], str)


def test_export_collection_manifest(temp_chromadb_dir, sample_chunks_with_embeddings):
    """Test exporting collection manifest."""
    indexer = ChromaDBIndexer(persist_directory=temp_chromadb_dir)

    # Index some chunks
    indexer.index_chunks(
        corpus="osha_reg",
        chunks_with_embeddings=sample_chunks_with_embeddings,
        show_progress=False
    )

    # Export manifest
    manifest_path = Path(temp_chromadb_dir) / "manifest.json"
    indexer.export_collection_manifest(str(manifest_path))

    assert manifest_path.exists()

    # Verify manifest content
    import json
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)

    assert "created_at" in manifest
    assert "statistics" in manifest
    assert "collection_mapping" in manifest


def test_query_collection(temp_chromadb_dir, sample_chunks_with_embeddings):
    """Test querying a collection."""
    indexer = ChromaDBIndexer(persist_directory=temp_chromadb_dir)

    # Index chunks
    indexer.index_chunks(
        corpus="osha_reg",
        chunks_with_embeddings=sample_chunks_with_embeddings,
        show_progress=False
    )

    # Query
    query_embedding = [[0.15] * 1024]
    results = indexer.query_collection(
        collection_name="osha_regulations",
        query_embeddings=query_embedding,
        n_results=1
    )

    assert results is not None
    assert "ids" in results or len(results) > 0
