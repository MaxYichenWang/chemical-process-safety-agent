"""
Unit tests for ChunkValidator
"""

import pytest

from src.data_processing.data_loading.chunk_validator import ChunkValidator


@pytest.fixture
def validator():
    """Create a ChunkValidator instance."""
    return ChunkValidator()


@pytest.fixture
def valid_chunks():
    """Create sample valid chunks."""
    return [
        {
            "chunk_id": "test_001",
            "text": "This is a test chunk with sufficient content for validation.",
            "corpus": "test_corpus",
            "subcorpus": "test_subcorpus",
            "hash": "abc123",
            "token_count": 500,
            "doc_type": "test"
        },
        {
            "chunk_id": "test_002",
            "text": "Another test chunk with good content.",
            "corpus": "test_corpus",
            "subcorpus": "test_subcorpus",
            "hash": "def456",
            "token_count": 450,
            "doc_type": "test"
        }
    ]


@pytest.fixture
def invalid_chunks():
    """Create sample invalid chunks."""
    return [
        {
            # Missing chunk_id
            "text": "Test text",
            "corpus": "test_corpus",
            "hash": "xyz789"
        },
        {
            "chunk_id": "test_003",
            # Empty text
            "text": "",
            "corpus": "test_corpus",
            "hash": "empty123"
        },
        {
            "chunk_id": "test_004",
            "text": "Test text",
            # Missing corpus
            "hash": "nocorpus456"
        }
    ]


def test_validator_init(validator):
    """Test ChunkValidator initialization."""
    assert validator.TARGET_MIN_TOKENS == 300
    assert validator.TARGET_MAX_TOKENS == 800
    assert "chunk_id" in validator.CRITICAL_FIELDS


def test_validate_valid_chunks(validator, valid_chunks):
    """Test validation with valid chunks."""
    report = validator.validate(valid_chunks)

    assert report.total_chunks == 2
    assert report.valid_chunks == 2
    assert report.invalid_chunks == 0
    assert len(report.errors) == 0


def test_validate_invalid_chunks(validator, invalid_chunks):
    """Test validation with invalid chunks."""
    report = validator.validate(invalid_chunks)

    assert report.total_chunks == 3
    assert report.invalid_chunks > 0
    assert len(report.errors) > 0


def test_check_schema_valid(validator):
    """Test schema checking with valid chunk."""
    chunk = {
        "chunk_id": "test_001",
        "text": "Valid text",
        "corpus": "test",
        "hash": "abc123"
    }

    errors = validator.check_schema(chunk)

    assert len(errors) == 0


def test_check_schema_missing_critical_field(validator):
    """Test schema checking with missing critical field."""
    chunk = {
        "text": "Valid text",
        "corpus": "test",
        "hash": "abc123"
        # Missing chunk_id
    }

    errors = validator.check_schema(chunk)

    assert len(errors) > 0
    assert any("chunk_id" in error for error in errors)


def test_check_schema_empty_text(validator):
    """Test schema checking with empty text."""
    chunk = {
        "chunk_id": "test_001",
        "text": "",
        "corpus": "test",
        "hash": "abc123"
    }

    errors = validator.check_schema(chunk)

    assert len(errors) > 0
    assert any("empty" in error.lower() for error in errors)


def test_check_quality(validator, valid_chunks):
    """Test quality checking."""
    metrics = validator.check_quality(valid_chunks)

    # check_quality returns a QualityMetrics object
    assert hasattr(metrics, "avg_token_count")
    assert hasattr(metrics, "duplicate_count")
    assert metrics.avg_token_count > 0


def test_validate_hash(validator):
    """Test hash validation."""
    import hashlib

    text = "Test text for hashing"
    correct_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()

    chunk_valid = {
        "text": text,
        "hash": correct_hash
    }

    chunk_invalid = {
        "text": text,
        "hash": "wrong_hash"
    }

    assert validator.validate_hash(chunk_valid) is True
    assert validator.validate_hash(chunk_invalid) is False


def test_validate_hash_with_prefixed_hash(validator):
    """Test hash validation with a sha256: prefixed hash."""
    import hashlib

    text = "Test text for hashing with prefix"
    raw_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()

    chunk_prefixed = {
        "text": text,
        "hash": f"sha256:{raw_hash}",
    }

    assert validator.validate_hash(chunk_prefixed) is True


def test_count_tokens(validator):
    """Test token counting."""
    text = "This is a test sentence."

    token_count = validator.count_tokens(text)

    assert token_count > 0
    assert isinstance(token_count, int)


def test_validation_report_summary(validator, valid_chunks):
    """Test ValidationReport summary generation."""
    report = validator.validate(valid_chunks)

    summary = report.summary()

    assert isinstance(summary, str)
    assert "CHUNK VALIDATION REPORT" in summary
    assert "Total Chunks:" in summary


def test_duplicate_detection(validator):
    """Test duplicate detection by hash."""
    chunks = [
        {
            "chunk_id": "test_001",
            "text": "Test text",
            "corpus": "test",
            "hash": "same_hash",
            "token_count": 500
        },
        {
            "chunk_id": "test_002",
            "text": "Test text",
            "corpus": "test",
            "hash": "same_hash",  # Duplicate hash
            "token_count": 500
        }
    ]

    report = validator.validate(chunks)

    assert report.quality_metrics["duplicate_count"] > 0


def test_token_count_warnings(validator):
    """Test warnings for out-of-range token counts."""
    chunks = [
        {
            "chunk_id": "test_001",
            "text": "Short text",
            "corpus": "test",
            "hash": "abc123",
            "token_count": 50  # Below minimum
        },
        {
            "chunk_id": "test_002",
            "text": "Long text" * 100,
            "corpus": "test",
            "hash": "def456",
            "token_count": 1500  # Above maximum
        }
    ]

    report = validator.validate(chunks)

    assert len(report.warnings) > 0
    assert any("token_count" in w.get("type", "") for w in report.warnings)



def test_empty_validation_report_has_a_readable_summary(validator):
    report = validator.validate([])
    assert report.total_chunks == 0
    assert "(0.0%)" in report.summary()
