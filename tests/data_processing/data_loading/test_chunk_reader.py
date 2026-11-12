"""
Unit tests for ChunkReader
"""

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from src.data_processing.data_loading.chunk_reader import ChunkReader


@pytest.fixture
def sample_chunks_dir():
    """Create a temporary directory with sample parquet files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create sample regulation chunks
        reg_chunks = [
            {
                "chunk_id": "osha_reg_001",
                "text": "OSHA regulation text",
                "corpus": "osha_reg",
                "subcorpus": "regulations",
                "hash": "abc123",
                "token_count": 100
            },
            {
                "chunk_id": "osha_reg_002",
                "text": "Another OSHA regulation",
                "corpus": "osha_reg",
                "subcorpus": "regulations",
                "hash": "def456",
                "token_count": 120
            }
        ]
        df_reg = pd.DataFrame(reg_chunks)
        df_reg.to_parquet(Path(tmpdir) / "osha_regulations_chunks.parquet", index=False)

        # Create sample violations sample chunks
        viol_chunks = [
            {
                "chunk_id": "osha_viol_sample_001",
                "text": "PSM violation sample text",
                "corpus": "osha_enf",
                "subcorpus": "osha_enf",
                "hash": "viol123",
                "token_count": 80
            },
        ]
        df_viol = pd.DataFrame(viol_chunks)
        df_viol.to_parquet(
            Path(tmpdir) / "osha_violations_sample_chunks.parquet", index=False
        )

        # Create sample general duty sample chunks
        genduty_chunks = [
            {
                "chunk_id": "osha_genduty_sample_001",
                "text": "Chemical industry general duty violation text",
                "corpus": "osha_enf",
                "subcorpus": "osha_enf",
                "hash": "gd123",
                "token_count": 90
            },
        ]
        df_gd = pd.DataFrame(genduty_chunks)
        df_gd.to_parquet(
            Path(tmpdir) / "osha_genduty_sample_chunks.parquet", index=False
        )

        yield tmpdir


def test_chunk_reader_init(sample_chunks_dir):
    """Test ChunkReader initialization."""
    reader = ChunkReader(sample_chunks_dir)

    assert reader.chunks_dir == Path(sample_chunks_dir)
    assert "osha_reg" in reader.corpus_files


def test_chunk_reader_init_invalid_dir():
    """Test ChunkReader initialization with invalid directory."""
    with pytest.raises(ValueError, match="does not exist"):
        ChunkReader("/nonexistent/path")


def test_read_by_file(sample_chunks_dir):
    """Test reading chunks from a specific file."""
    reader = ChunkReader(sample_chunks_dir)

    chunks = reader.read_by_file("osha_regulations_chunks.parquet")

    assert len(chunks) == 2
    assert chunks[0]["chunk_id"] == "osha_reg_001"
    assert chunks[1]["chunk_id"] == "osha_reg_002"


def test_read_by_file_not_found(sample_chunks_dir):
    """Test reading from non-existent file."""
    reader = ChunkReader(sample_chunks_dir)

    with pytest.raises(FileNotFoundError):
        reader.read_by_file("nonexistent.parquet")


def test_read_by_corpus(sample_chunks_dir):
    """Test reading chunks by corpus."""
    reader = ChunkReader(sample_chunks_dir)

    chunks = reader.read_by_corpus("osha_reg")

    assert len(chunks) == 2
    assert all(chunk["corpus"] == "osha_reg" for chunk in chunks)


def test_read_by_corpus_invalid(sample_chunks_dir):
    """Test reading with invalid corpus."""
    reader = ChunkReader(sample_chunks_dir)

    with pytest.raises(ValueError, match="Unknown corpus"):
        reader.read_by_corpus("invalid_corpus")


def test_read_all_chunks(sample_chunks_dir):
    """Test reading all chunks."""
    reader = ChunkReader(sample_chunks_dir)

    chunks = reader.read_all_chunks()

    assert len(chunks) == 4  # 2 reg + 1 viol sample + 1 genduty sample


def test_get_chunk_stats(sample_chunks_dir):
    """Test getting chunk statistics."""
    reader = ChunkReader(sample_chunks_dir)

    stats = reader.get_chunk_stats()

    assert "by_corpus" in stats
    assert "total_chunks" in stats
    assert stats["total_chunks"] == 4
    assert stats["by_corpus"]["osha_reg"]["chunk_count"] == 2
    assert stats["by_corpus"]["osha_reg"]["exists"] is True
    assert stats["by_corpus"]["osha_enf_violations"]["chunk_count"] == 1
    assert stats["by_corpus"]["osha_enf_violations"]["exists"] is True
    assert stats["by_corpus"]["osha_enf_genduty"]["chunk_count"] == 1
    assert stats["by_corpus"]["osha_enf_genduty"]["exists"] is True


def test_list_available_corpora(sample_chunks_dir):
    """Test listing available corpora."""
    reader = ChunkReader(sample_chunks_dir)

    available = reader.list_available_corpora()

    assert "osha_reg" in available
    assert "osha_enf_violations" in available
    assert "osha_enf_genduty" in available
    assert len(available) >= 3


def test_sample_chunks(sample_chunks_dir):
    """Test sampling chunks."""
    reader = ChunkReader(sample_chunks_dir)

    samples = reader.sample_chunks(corpus="osha_reg", n_samples=1)

    assert len(samples) <= 1
    assert samples[0]["corpus"] == "osha_reg"


def test_read_by_corpus_violations_sample(sample_chunks_dir):
    """Test reading violations sample chunks by corpus key."""
    reader = ChunkReader(sample_chunks_dir)

    chunks = reader.read_by_corpus("osha_enf_violations")

    assert len(chunks) == 1
    assert chunks[0]["chunk_id"] == "osha_viol_sample_001"
    assert chunks[0]["corpus"] == "osha_enf"


def test_read_by_corpus_genduty_sample(sample_chunks_dir):
    """Test reading general duty sample chunks by corpus key."""
    reader = ChunkReader(sample_chunks_dir)

    chunks = reader.read_by_corpus("osha_enf_genduty")

    assert len(chunks) == 1
    assert chunks[0]["chunk_id"] == "osha_genduty_sample_001"
    assert chunks[0]["corpus"] == "osha_enf"


def test_corpus_files_has_new_entries(sample_chunks_dir):
    """Test that corpus_files dict contains the new enforcement sample entries."""
    reader = ChunkReader(sample_chunks_dir)

    assert "osha_enf_violations" in reader.corpus_files
    assert reader.corpus_files["osha_enf_violations"] == "osha_violations_sample_chunks.parquet"
    assert "osha_enf_genduty" in reader.corpus_files
    assert reader.corpus_files["osha_enf_genduty"] == "osha_genduty_sample_chunks.parquet"
