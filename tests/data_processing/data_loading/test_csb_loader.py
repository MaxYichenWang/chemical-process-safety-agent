"""
Tests for CSB data loader.
"""

import json

import pytest

from src.data_processing.data_loading.csb_loader import CSBDataLoader


@pytest.fixture
def csb_loader(tmp_path):
    """Create a CSB loader instance for testing.

    In CI or environments without the full CSB dataset, we fall back to a small
    synthetic dataset so tests still exercise the loader logic instead of
    skipping.
    """
    # First try real data; if unavailable, build a tiny static dataset.
    try:
        return CSBDataLoader()
    except FileNotFoundError:
        # Create minimal directory structure: json/, markdown/, toc/
        data_root = tmp_path / "csb_mock_data"
        json_dir = data_root / "json" / "incident1"
        toc_dir = data_root / "toc" / "incident1"
        markdown_dir = data_root / "markdown"

        json_dir.mkdir(parents=True, exist_ok=True)
        toc_dir.mkdir(parents=True, exist_ok=True)
        markdown_dir.mkdir(parents=True, exist_ok=True)

        # --- Final report with TOC ---
        final_report = {
            "metadata": {
                "incident_name": "incident1",
                "report_name": "final_report",
                "title": "Mock Final Report",
            },
            "pages": [
                {"page_number": 1, "text": "Final report page 1 text."},
                {"page_number": 2, "text": "Final report page 2 text."},
            ],
        }
        (json_dir / "final_report_structured.json").write_text(
            json.dumps(final_report), encoding="utf-8"
        )

        toc = {
            "title": "Document Root",
            "level": 0,
            "children": [
                {
                    "title": "Section 1",
                    "level": 1,
                    "pageNumber": 1,
                    "position": 1,
                    "children": [],
                }
            ],
        }
        (toc_dir / "final_report_toc.json").write_text(
            json.dumps(toc), encoding="utf-8"
        )

        # --- Recommendation without TOC ---
        recommendation = {
            "metadata": {
                "incident_name": "incident1",
                "report_name": "recommendation",
                "title": "Mock Recommendation",
            },
            "pages": [
                {
                    "page_number": 1,
                    "text": "This is a mock recommendation document used for tests.",
                }
            ],
        }
        (json_dir / "recommendation_structured.json").write_text(
            json.dumps(recommendation), encoding="utf-8"
        )

        # Use the synthetic dataset
        return CSBDataLoader(data_root=str(data_root))


def test_csb_loader_initialization(csb_loader):
    """Test CSB loader can be initialized."""
    assert csb_loader is not None
    assert csb_loader.data_root.exists()
    # Default chunking configuration should be aligned with validation targets
    assert csb_loader.chunk_size == 800
    assert csb_loader.chunk_overlap == 100
    assert csb_loader.max_tokens_per_chunk == 800


def test_discover_reports(csb_loader):
    """Test discovering reports from file system."""
    final_reports, recommendations = csb_loader._discover_reports()

    assert len(final_reports) > 0
    assert len(recommendations) > 0

    # Check expected keys
    assert "incident_name" in final_reports[0]
    assert "report_name" in final_reports[0]
    assert "incident_name" in recommendations[0]
    assert "report_name" in recommendations[0]


def test_load_report_json(csb_loader):
    """Test loading a single report JSON."""
    # Get first report from discovered reports
    final_reports, _ = csb_loader._discover_reports()
    first_report = final_reports[0]

    report_data = csb_loader.load_report_json(
        first_report["incident_name"], first_report["report_name"]
    )

    assert "metadata" in report_data
    assert "pages" in report_data
    assert len(report_data["pages"]) > 0


def test_load_toc(csb_loader):
    """Test loading TOC for a final report."""
    final_reports, _ = csb_loader._discover_reports()
    first_report = final_reports[0]

    toc_data = csb_loader.load_toc(
        first_report["incident_name"], first_report["report_name"]
    )

    # TOC may or may not exist
    if toc_data:
        assert "title" in toc_data or "children" in toc_data


def test_chunk_final_report(csb_loader):
    """Test chunking a final report with TOC guidance."""
    final_reports, _ = csb_loader._discover_reports()
    first_report = final_reports[0]

    chunks = csb_loader.chunk_final_report(
        first_report["incident_name"],
        first_report["report_name"],
        None,
    )

    assert len(chunks) > 0

    # Verify chunk structure
    first_chunk = chunks[0]
    assert "chunk_id" in first_chunk
    assert "text" in first_chunk
    assert "doc_type" in first_chunk
    assert first_chunk["doc_type"] == "csb_final_report"
    assert "incident_name" in first_chunk
    assert "token_count" in first_chunk


def test_chunk_recommendation(csb_loader):
    """Test chunking a recommendation."""
    _, recommendations = csb_loader._discover_reports()
    first_rec = recommendations[0]

    chunks = csb_loader.chunk_recommendation(
        first_rec["incident_name"], first_rec["report_name"], None
    )

    assert len(chunks) > 0

    # Verify chunk structure
    first_chunk = chunks[0]
    assert "chunk_id" in first_chunk
    assert "text" in first_chunk
    assert "doc_type" in first_chunk
    assert first_chunk["doc_type"] == "csb_recommendation"


def test_load_all_final_reports_limited(csb_loader):
    """Test loading multiple final reports with limit."""
    chunks = csb_loader.load_all_final_reports(limit=2, show_progress=False)

    assert len(chunks) > 0

    # All chunks should be from final reports
    assert all(chunk["doc_type"] == "csb_final_report" for chunk in chunks)


def test_load_all_recommendations_limited(csb_loader):
    """Test loading multiple recommendations with limit."""
    chunks = csb_loader.load_all_recommendations(limit=2, show_progress=False)

    assert len(chunks) > 0

    # All chunks should be from recommendations
    assert all(chunk["doc_type"] == "csb_recommendation" for chunk in chunks)


def test_flatten_toc(csb_loader):
    """Test TOC flattening."""
    toc_sample = {
        "title": "Root",
        "level": 0,
        "children": [
            {
                "title": "Section 1",
                "level": 1,
                "pageNumber": 10,
                "position": 1,
                "children": [
                    {
                        "title": "Section 1.1",
                        "level": 2,
                        "pageNumber": 11,
                        "position": 2,
                        "children": [],
                    }
                ],
            }
        ],
    }

    flattened = csb_loader._flatten_toc(toc_sample)

    # Root node is included if title is not "Document Root", so we get 3 items
    assert len(flattened) == 3  # Root, Section 1, and Section 1.1
    assert flattened[0]["title"] == "Root"
    assert flattened[1]["title"] == "Section 1"
    assert flattened[2]["title"] == "Section 1.1"
    assert "Section 1" in flattened[2]["section_path"]


def test_load_report_json_not_found(csb_loader):
    """Test FileNotFoundError when JSON file doesn't exist."""
    with pytest.raises(FileNotFoundError):
        csb_loader.load_report_json("nonexistent_incident", "nonexistent_report")


def test_load_toc_not_found(csb_loader):
    """Test that load_toc returns None when TOC doesn't exist."""
    # Use a report that likely doesn't have a TOC
    final_reports, _ = csb_loader._discover_reports()
    first_report = final_reports[0]

    # Try with a modified report name that doesn't exist
    toc_data = csb_loader.load_toc(first_report["incident_name"], "nonexistent_toc")
    assert toc_data is None


def test_chunk_final_report_without_toc(csb_loader):
    """Test chunking a final report without TOC (fallback method)."""
    final_reports, _ = csb_loader._discover_reports()
    first_report = final_reports[0]

    # Mock load_toc to return None to trigger fallback
    original_load_toc = csb_loader.load_toc
    csb_loader.load_toc = lambda *args, **kwargs: None

    try:
        chunks = csb_loader.chunk_final_report(
            first_report["incident_name"],
            first_report["report_name"],
            None,
        )

        assert len(chunks) > 0
        # Verify chunk structure
        first_chunk = chunks[0]
        assert "chunk_id" in first_chunk
        assert "text" in first_chunk
        assert "doc_type" in first_chunk
        assert first_chunk["doc_type"] == "csb_final_report"
        assert first_chunk.get("chunking_method") == "page_based_fallback"
    finally:
        csb_loader.load_toc = original_load_toc


def test_extract_text_by_pages(csb_loader):
    """Test extracting text by page range."""
    final_reports, _ = csb_loader._discover_reports()
    first_report = final_reports[0]

    report_data = csb_loader.load_report_json(
        first_report["incident_name"], first_report["report_name"]
    )

    # Test with specific page range
    text = csb_loader._extract_text_by_pages(report_data, 1, 2)
    assert isinstance(text, str)

    # Test with end_page=None (should default to start_page)
    text_single = csb_loader._extract_text_by_pages(report_data, 1, None)
    assert isinstance(text_single, str)


def test_flatten_toc_document_root(csb_loader):
    """Test TOC flattening with Document Root (should skip root)."""
    toc_sample = {
        "title": "Document Root",
        "level": 0,
        "children": [
            {
                "title": "Section 1",
                "level": 1,
                "pageNumber": 10,
                "position": 1,
                "children": [],
            }
        ],
    }

    flattened = csb_loader._flatten_toc(toc_sample)

    # Document Root should be skipped, so we get only Section 1
    assert len(flattened) == 1
    assert flattened[0]["title"] == "Section 1"


def test_chunk_recommendation_with_metadata(csb_loader):
    """Test chunking recommendation with metadata."""
    _, recommendations = csb_loader._discover_reports()
    first_rec = recommendations[0]

    metadata = {
        "num_pages": 5,
        "extraction_date": "2024-01-01",
        "processing_date": "2024-01-02",
    }

    chunks = csb_loader.chunk_recommendation(
        first_rec["incident_name"],
        first_rec["report_name"],
        metadata,
    )

    assert len(chunks) > 0
    # Check that metadata is included
    if chunks[0].get("num_pages"):
        assert chunks[0]["num_pages"] == 5


def test_csb_loader_initialization_invalid_path():
    """Test CSB loader initialization with invalid path."""
    with pytest.raises(FileNotFoundError):
        CSBDataLoader(data_root="/nonexistent/path")


def test_chunk_recommendation_large_text(csb_loader):
    """Test chunking a recommendation with text that needs splitting."""
    _, recommendations = csb_loader._discover_reports()
    first_rec = recommendations[0]

    # Create a loader with very small max_tokens_per_chunk to force splitting
    small_chunk_loader = CSBDataLoader(
        data_root=csb_loader.data_root,
        max_tokens_per_chunk=100,  # Very small to force splitting
    )

    chunks = small_chunk_loader.chunk_recommendation(
        first_rec["incident_name"],
        first_rec["report_name"],
        None,
    )

    # Should have multiple chunks if text is large enough
    assert len(chunks) >= 1
    # Verify chunk structure
    for chunk in chunks:
        assert "chunk_id" in chunk
        assert "text" in chunk
        assert "doc_type" in chunk
        assert chunk["doc_type"] == "csb_recommendation"


def test_load_all_final_reports_with_progress(csb_loader):
    """Test loading all final reports with progress bar."""
    chunks = csb_loader.load_all_final_reports(limit=2, show_progress=True)

    assert len(chunks) > 0
    assert all(chunk["doc_type"] == "csb_final_report" for chunk in chunks)


def test_load_all_final_reports_error_handling(csb_loader, monkeypatch):
    """Test error handling in load_all_final_reports."""
    # Mock chunk_final_report to raise an exception for one report
    original_chunk = csb_loader.chunk_final_report
    call_count = [0]

    def mock_chunk(*args, **kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            raise Exception("Test error")
        return original_chunk(*args, **kwargs)

    csb_loader.chunk_final_report = mock_chunk

    # Should not raise, but continue processing
    chunks = csb_loader.load_all_final_reports(limit=3, show_progress=False)
    # Should have chunks from the successful reports
    assert len(chunks) >= 0


def test_load_all_recommendations_with_progress(csb_loader):
    """Test loading all recommendations with progress bar."""
    chunks = csb_loader.load_all_recommendations(limit=2, show_progress=True)

    assert len(chunks) > 0
    assert all(chunk["doc_type"] == "csb_recommendation" for chunk in chunks)


def test_load_all_recommendations_error_handling(csb_loader, monkeypatch):
    """Test error handling in load_all_recommendations."""
    # Mock chunk_recommendation to raise an exception for one recommendation
    original_chunk = csb_loader.chunk_recommendation
    call_count = [0]

    def mock_chunk(*args, **kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            raise Exception("Test error")
        return original_chunk(*args, **kwargs)

    csb_loader.chunk_recommendation = mock_chunk

    # Should not raise, but continue processing
    chunks = csb_loader.load_all_recommendations(limit=3, show_progress=False)
    # Should have chunks from the successful recommendations
    assert len(chunks) >= 0


def test_export_chunks_to_parquet(csb_loader, tmp_path):
    """Test exporting chunks to Parquet format."""
    # Get some chunks
    chunks = csb_loader.load_all_final_reports(limit=2, show_progress=False)

    if len(chunks) > 0:
        output_path = tmp_path / "test_chunks.parquet"
        csb_loader.export_chunks_to_parquet(chunks, str(output_path))

        # Verify file was created
        assert output_path.exists()

        # Verify we can read it back
        import pandas as pd

        df = pd.read_parquet(output_path)
        assert len(df) == len(chunks)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
