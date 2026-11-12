"""
Tests for embeddings utilities.
"""

import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest
from chromadb import EmbeddingFunction

from src.data_processing.utilities.embeddings_utils import (
    ChromaDBManager,
    EmbeddingsGenerator,
    create_embeddings_pipeline,
)


class LocalEmbeddingFunction(EmbeddingFunction):
    """Small deterministic vectors for local storage and query tests."""

    def __init__(self):
        pass

    def __call__(self, input):
        return [[float(len(text)), float(text.count("safety") + 1), 1.0] for text in input]

    @staticmethod
    def name():
        return "test_local"

    def get_config(self):
        return {}

    @staticmethod
    def build_from_config(config):
        return LocalEmbeddingFunction()


@pytest.fixture
def mock_openai_client():
    """Create a mock OpenAI client."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.data = [
        MagicMock(embedding=[0.1] * 1024),
        MagicMock(embedding=[0.2] * 1024),
    ]
    mock_client.embeddings.create.return_value = mock_response
    return mock_client


@pytest.fixture
def temp_chromadb_dir():
    """Create a temporary directory for ChromaDB."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


class TestEmbeddingsGenerator:
    """Tests for EmbeddingsGenerator class."""

    def test_init_with_api_key(self, mock_openai_client):
        """Test initialization with API key."""
        with patch(
            "src.data_processing.utilities.embeddings_utils.OpenAI",
            return_value=mock_openai_client,
        ):
            generator = EmbeddingsGenerator(api_key="test-key")
            assert generator.api_key == "test-key"
            assert generator.model == "text-embedding-3-large"
            assert generator.dimensions == 1024
            assert generator.batch_size == 500

    def test_init_with_env_var(self, mock_openai_client):
        """Test initialization with environment variable."""
        with patch(
            "src.data_processing.utilities.embeddings_utils.OpenAI",
            return_value=mock_openai_client,
        ):
            with patch.dict(os.environ, {"OPENAI_API_KEY": "env-key"}):
                generator = EmbeddingsGenerator()
                assert generator.api_key == "env-key"

    def test_init_without_api_key(self):
        """Test initialization fails without API key."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="OpenAI API key"):
                EmbeddingsGenerator()

    def test_embed_texts_single_batch(self, mock_openai_client):
        """Test embedding texts in a single batch."""
        with patch(
            "src.data_processing.utilities.embeddings_utils.OpenAI",
            return_value=mock_openai_client,
        ):
            generator = EmbeddingsGenerator(api_key="test-key")
            texts = ["Text 1", "Text 2"]

            embeddings = generator.embed_texts(texts, show_progress=False)

            assert len(embeddings) == 2
            assert len(embeddings[0]) == 1024
            mock_openai_client.embeddings.create.assert_called_once()

    def test_embed_texts_multiple_batches(self, mock_openai_client):
        """Test embedding texts across multiple batches."""
        with patch(
            "src.data_processing.utilities.embeddings_utils.OpenAI",
            return_value=mock_openai_client,
        ):
            generator = EmbeddingsGenerator(api_key="test-key", batch_size=2)
            texts = ["Text 1", "Text 2", "Text 3", "Text 4", "Text 5"]

            # Mock responses: batch 1 (2 items), batch 2 (2 items), batch 3 (1 item)
            def create_mock_response(batch_size):
                mock_response = MagicMock()
                mock_response.data = [
                    MagicMock(embedding=[0.1] * 1024) for _ in range(batch_size)
                ]
                return mock_response

            mock_openai_client.embeddings.create.side_effect = [
                create_mock_response(2),  # First batch: 2 items
                create_mock_response(2),  # Second batch: 2 items
                create_mock_response(1),  # Third batch: 1 item
            ]

            embeddings = generator.embed_texts(texts, show_progress=False)

            assert len(embeddings) == 5
            assert mock_openai_client.embeddings.create.call_count == 3

    def test_embed_single(self, mock_openai_client):
        """Test embedding a single text."""
        with patch(
            "src.data_processing.utilities.embeddings_utils.OpenAI",
            return_value=mock_openai_client,
        ):
            generator = EmbeddingsGenerator(api_key="test-key")

            embedding = generator.embed_single("Test text")

            assert len(embedding) == 1024
            mock_openai_client.embeddings.create.assert_called_once()


class TestChromaDBManager:
    """Tests for ChromaDBManager class."""

    def test_init(self, temp_chromadb_dir):
        """Test ChromaDB manager initialization."""
        manager = ChromaDBManager(
            persist_directory=temp_chromadb_dir, collection_name="test_collection"
        )

        assert manager.persist_directory == temp_chromadb_dir
        assert manager.collection_name == "test_collection"
        assert manager.collection is not None

    def test_add_chunks_with_embeddings(self, temp_chromadb_dir):
        """Test adding chunks with pre-computed embeddings."""
        manager = ChromaDBManager(
            persist_directory=temp_chromadb_dir, collection_name="test_collection"
        )

        chunks = [
            {"chunk_id": "chunk1", "text": "Text 1", "metadata_field": "value1"},
            {"chunk_id": "chunk2", "text": "Text 2", "metadata_field": "value2"},
        ]

        embeddings = [[0.1] * 1024, [0.2] * 1024]

        manager.add_chunks(chunks, embeddings, show_progress=False)

        # Verify chunks were added
        stats = manager.get_collection_stats()
        assert stats["count"] == 2

    def test_add_chunks_without_embeddings(self, temp_chromadb_dir):
        """Test adding chunks without embeddings."""
        manager = ChromaDBManager(
            persist_directory=temp_chromadb_dir, collection_name="test_collection",
            embedding_function=LocalEmbeddingFunction(),
        )

        chunks = [{"chunk_id": "chunk1", "text": "Text 1", "metadata_field": "value1"}]

        manager.add_chunks(chunks, embeddings=None, show_progress=False)

        # Verify chunk was added
        stats = manager.get_collection_stats()
        assert stats["count"] == 1

    def test_query(self, temp_chromadb_dir):
        """Test querying the collection."""
        manager = ChromaDBManager(
            persist_directory=temp_chromadb_dir, collection_name="test_collection",
            embedding_function=LocalEmbeddingFunction(),
        )

        chunks = [
            {
                "chunk_id": "chunk1",
                "text": "Text about safety",
                "doc_type": "regulation",
            }
        ]

        # Add chunks without embeddings - ChromaDB will use default embedding function
        manager.add_chunks(chunks, embeddings=None, show_progress=False)

        results = manager.query("safety", n_results=1)

        assert "ids" in results
        assert "documents" in results
        assert "metadatas" in results

    def test_prepare_metadata(self, temp_chromadb_dir):
        """Test metadata preparation."""
        manager = ChromaDBManager(
            persist_directory=temp_chromadb_dir, collection_name="test_collection"
        )

        chunk = {
            "chunk_id": "chunk1",
            "text": "Text",
            "string_field": "value",
            "int_field": 42,
            "float_field": 3.14,
            "bool_field": True,
            "none_field": None,
            "list_field": [1, 2, 3],  # Should be converted to string
        }

        metadata = manager._prepare_metadata(chunk)

        assert "text" not in metadata
        assert metadata["string_field"] == "value"
        assert metadata["int_field"] == 42
        assert metadata["float_field"] == 3.14
        assert metadata["bool_field"] is True
        assert metadata["none_field"] == ""
        assert metadata["list_field"] == "[1, 2, 3]"

    def test_get_collection_stats(self, temp_chromadb_dir):
        """Test getting collection statistics."""
        manager = ChromaDBManager(
            persist_directory=temp_chromadb_dir, collection_name="test_collection"
        )

        stats = manager.get_collection_stats()

        assert stats["name"] == "test_collection"
        assert stats["count"] == 0
        assert stats["persist_directory"] == temp_chromadb_dir


class TestCreateEmbeddingsPipeline:
    """Tests for create_embeddings_pipeline function."""

    def test_create_embeddings_pipeline(self, mock_openai_client, temp_chromadb_dir):
        """Test the complete embeddings pipeline."""
        with patch(
            "src.data_processing.utilities.embeddings_utils.OpenAI",
            return_value=mock_openai_client,
        ):
            chunks = [
                {"chunk_id": "chunk1", "text": "Text 1", "doc_type": "regulation"},
                {"chunk_id": "chunk2", "text": "Text 2", "doc_type": "regulation"},
            ]

            manager = create_embeddings_pipeline(
                chunks=chunks,
                api_key="test-key",
                chromadb_dir=temp_chromadb_dir,
                collection_name="test_collection",
                show_progress=False,
            )

            assert isinstance(manager, ChromaDBManager)
            stats = manager.get_collection_stats()
            assert stats["count"] == 2
            assert stats["name"] == "test_collection"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
