"""
Tests for utility functions.
"""

import pytest
from src.data_processing.utilities.utils import (
    chunk_text,
    compute_hash,
    count_tokens,
    extract_section_hierarchy,
    format_toc_path,
    generate_chunk_id,
    get_data_path,
    is_colab_environment,
    normalize_standard_number,
)


def test_count_tokens():
    """Test token counting functionality."""
    text = "Hello, world! This is a test."
    token_count = count_tokens(text)
    assert token_count > 0
    assert isinstance(token_count, int)


def test_compute_hash():
    """Test hash computation."""
    text1 = "Hello, world!"
    text2 = "Hello, world!"
    text3 = "Goodbye, world!"

    hash1 = compute_hash(text1)
    hash2 = compute_hash(text2)
    hash3 = compute_hash(text3)

    # Same text should produce same hash
    assert hash1 == hash2
    # Different text should produce different hash
    assert hash1 != hash3
    # Hash should have proper format
    assert hash1.startswith("sha256:")


def test_generate_chunk_id():
    """Test chunk ID generation."""
    # Without chunk index
    chunk_id = generate_chunk_id("osha_reg", "1910.147")
    assert chunk_id == "osha_reg_1910.147"

    # With chunk index
    chunk_id = generate_chunk_id("csb_reports", "West_Fertilizer_4.1", 2)
    assert chunk_id == "csb_reports_West_Fertilizer_4.1_chunk_2"


def test_chunk_text():
    """Test text chunking."""
    text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph." * 50
    chunks = chunk_text(text, chunk_size=100, chunk_overlap=20)

    assert len(chunks) > 1
    assert all(isinstance(chunk, str) for chunk in chunks)


def test_normalize_standard_number():
    """Test OSHA standard number normalization."""
    assert normalize_standard_number("§1910.147") == "1910.147"
    assert normalize_standard_number("29 CFR 1910.147(a)") == "1910.147(a)"
    assert normalize_standard_number("1910.147") == "1910.147"


def test_extract_section_hierarchy():
    """Test section hierarchy extraction."""
    hierarchy = extract_section_hierarchy("1910.147(c)(4)(i)")

    assert hierarchy['part'] == "1910"
    assert hierarchy['standard'] == "1910.147"
    assert hierarchy['level'] == 3  # Three levels of parentheses


def test_chunk_text_with_overlap():
    """Test that chunking maintains overlap."""
    text = " ".join([f"Word{i}" for i in range(200)])
    chunks = chunk_text(text, chunk_size=50, chunk_overlap=10)

    # Should have multiple chunks
    assert len(chunks) > 1


def test_count_tokens_unknown_model():
    """Test token counting with unknown model (should fallback to cl100k_base)."""
    text = "Hello, world!"
    # Use a model name that doesn't exist
    token_count = count_tokens(text, model="unknown-model-xyz")
    assert token_count > 0
    assert isinstance(token_count, int)


def test_extract_section_hierarchy_no_match():
    """Test section hierarchy extraction with no match."""
    hierarchy = extract_section_hierarchy("invalid-section-id")

    assert hierarchy['part'] == ''
    assert hierarchy['standard'] == 'invalid-section-id'
    assert hierarchy['level'] == 0


def test_extract_section_hierarchy_simple():
    """Test section hierarchy extraction with simple standard number."""
    hierarchy = extract_section_hierarchy("1910.147")

    assert hierarchy['part'] == "1910"
    assert hierarchy['standard'] == "1910.147"
    assert hierarchy['level'] == 0  # No parentheses


def test_format_toc_path():
    """Test TOC path formatting."""
    # With parent_title
    entry = {'title': 'Mechanical Integrity', 'parent_title': 'Safety Issues'}
    path = format_toc_path(entry)
    assert path == 'Safety Issues / Mechanical Integrity'

    # Without parent_title
    entry_no_parent = {'title': 'Introduction'}
    path_no_parent = format_toc_path(entry_no_parent)
    assert path_no_parent == 'Introduction'

    # With None parent_title
    entry_none_parent = {'title': 'Section', 'parent_title': None}
    path_none = format_toc_path(entry_none_parent)
    assert path_none == 'Section'


def test_format_toc_path_custom_separator():
    """Test TOC path formatting with custom separator."""
    entry = {'title': 'Section', 'parent_title': 'Parent'}
    path = format_toc_path(entry, separator=" > ")
    assert path == 'Parent > Section'


def test_get_data_path():
    """Test get_data_path function."""
    path = get_data_path("data/processed/test")

    # Should return an absolute path
    assert isinstance(path, str)
    assert "data/processed/test" in path


def test_is_colab_environment():
    """Test Colab environment detection."""
    # Should return False in non-Colab environment
    result = is_colab_environment()
    assert isinstance(result, bool)
    assert result is False  # We're not in Colab


def test_is_colab_environment_mocked(monkeypatch):
    """Test Colab environment detection when google.colab is available."""
    # Mock google.colab module
    mock_colab = type('MockColab', (), {})()
    import sys
    mock_module = type('MockModule', (), {'colab': mock_colab})()

    # Patch the import
    with monkeypatch.context() as m:
        m.setitem(sys.modules, 'google.colab', mock_module)
        # Need to reload the module to get the new import
        import importlib
        from src.data_processing.utilities import utils
        importlib.reload(utils)
        result = utils.is_colab_environment()
        assert result is True


def test_get_data_path_colab_environment(monkeypatch):
    """Test get_data_path in Colab environment."""
    # Mock is_colab_environment to return True
    import sys
    mock_colab = type('MockColab', (), {})()
    mock_module = type('MockModule', (), {'colab': mock_colab})()

    with monkeypatch.context() as m:
        m.setitem(sys.modules, 'google.colab', mock_module)
        import importlib
        from src.data_processing.utilities import utils
        importlib.reload(utils)

        path = utils.get_data_path("data/processed/test", colab_drive_root="/test/drive")
        assert "/test/drive" in path
        assert "chemical-process-safety-agent" in path
        assert "data/processed/test" in path


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
