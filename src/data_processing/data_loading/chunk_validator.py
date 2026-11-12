"""
Chunk Validator for Schema and Quality Checks

This module provides utilities for validating chunks against schema
requirements and quality standards.
"""

import hashlib
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Set

import tiktoken

from .. import config


@dataclass
class ValidationReport:
    """Report containing validation results."""

    total_chunks: int
    valid_chunks: int
    invalid_chunks: int
    errors: List[Dict[str, Any]]
    warnings: List[Dict[str, Any]]
    quality_metrics: Dict[str, Any]

    def summary(self) -> str:
        """
        Generate a human-readable summary of the validation report.

        Returns:
            Summary string
        """
        denominator = self.total_chunks or 1
        lines = [
            "=" * 80,
            "CHUNK VALIDATION REPORT",
            "=" * 80,
            "",
            f"Total Chunks:   {self.total_chunks:,}",
            f"Valid Chunks:   {self.valid_chunks:,} ({self.valid_chunks/denominator*100:.1f}%)",
            f"Invalid Chunks: {self.invalid_chunks:,} ({self.invalid_chunks/denominator*100:.1f}%)",
            "",
            f"Errors:   {len(self.errors):,}",
            f"Warnings: {len(self.warnings):,}",
            "",
            "=" * 80,
            "QUALITY METRICS",
            "=" * 80,
            "",
        ]

        # Add quality metrics
        for key, value in self.quality_metrics.items():
            if isinstance(value, float):
                lines.append(f"{key}: {value:.2f}")
            elif isinstance(value, int):
                lines.append(f"{key}: {value:,}")
            else:
                lines.append(f"{key}: {value}")

        # Add error samples (first 10)
        if self.errors:
            lines.extend(
                [
                    "",
                    "=" * 80,
                    f"ERROR SAMPLES (showing first 10 of {len(self.errors)})",
                    "=" * 80,
                    "",
                ]
            )
            for error in self.errors[:10]:
                lines.append(
                    f"- {error.get('chunk_id', 'UNKNOWN')}: {error.get('message', '')}"
                )

        # Add warning samples (first 10)
        if self.warnings:
            lines.extend(
                [
                    "",
                    "=" * 80,
                    f"WARNING SAMPLES (showing first 10 of {len(self.warnings)})",
                    "=" * 80,
                    "",
                ]
            )
            for warning in self.warnings[:10]:
                lines.append(
                    f"- {warning.get('chunk_id', 'UNKNOWN')}: {warning.get('message', '')}"
                )

        return "\n".join(lines)


@dataclass
class QualityMetrics:
    """Quality metrics for chunks."""

    avg_token_count: float
    min_token_count: int
    max_token_count: int
    token_count_std: float
    duplicate_count: int
    empty_text_count: int
    missing_metadata_count: int
    corpus_distribution: Dict[str, int]


class ChunkValidator:
    """Validate chunks against schema and quality requirements."""

    # Required fields (STRICT - must exist)
    CRITICAL_FIELDS = {"chunk_id", "text", "corpus", "hash"}

    # Important fields (WARN if missing)
    IMPORTANT_FIELDS = {"token_count", "subcorpus", "doc_type"}

    # Target token range
    TARGET_MIN_TOKENS = config.CHUNK_TARGET_MIN_TOKENS
    TARGET_MAX_TOKENS = config.CHUNK_TARGET_MAX_TOKENS

    def __init__(self, encoding_name: str = "cl100k_base"):
        """
        Initialize validator.

        Args:
            encoding_name: Tokenizer encoding name for token counting
        """
        self.encoding = tiktoken.get_encoding(encoding_name)

    def validate(self, chunks: List[Dict[str, Any]]) -> ValidationReport:
        """
        Validate chunks and return comprehensive report.

        Args:
            chunks: List of chunk dictionaries to validate

        Returns:
            ValidationReport with errors, warnings, and quality metrics
        """
        errors = []
        warnings = []
        valid_count = 0

        # Track metrics
        token_counts = []
        seen_hashes: Set[str] = set()
        duplicate_count = 0
        empty_text_count = 0
        missing_metadata_count = 0
        corpus_distribution: Dict[str, int] = defaultdict(int)

        for i, chunk in enumerate(chunks):
            chunk_id = chunk.get("chunk_id", f"UNKNOWN_{i}")
            is_valid = True

            # Check schema compliance
            schema_errors = self.check_schema(chunk)
            if schema_errors:
                is_valid = False
                for error_msg in schema_errors:
                    errors.append(
                        {
                            "chunk_id": chunk_id,
                            "type": "schema_error",
                            "message": error_msg,
                        }
                    )

            # Check for empty text
            if chunk.get("text", "").strip() == "":
                empty_text_count += 1
                warnings.append(
                    {
                        "chunk_id": chunk_id,
                        "type": "empty_text",
                        "message": "Chunk has empty text",
                    }
                )

            # Check for duplicates by hash
            chunk_hash = chunk.get("hash", "")
            if chunk_hash:
                if chunk_hash in seen_hashes:
                    duplicate_count += 1
                    warnings.append(
                        {
                            "chunk_id": chunk_id,
                            "type": "duplicate",
                            "message": f"Duplicate hash detected: {chunk_hash}",
                        }
                    )
                else:
                    seen_hashes.add(chunk_hash)

            # Track token counts
            token_count = chunk.get("token_count")
            if token_count is not None:
                token_counts.append(token_count)

                # Warn if outside target range
                if token_count < self.TARGET_MIN_TOKENS:
                    warnings.append(
                        {
                            "chunk_id": chunk_id,
                            "type": "low_token_count",
                            "message": f"Token count {token_count} below target minimum {self.TARGET_MIN_TOKENS}",
                        }
                    )
                elif token_count > self.TARGET_MAX_TOKENS:
                    warnings.append(
                        {
                            "chunk_id": chunk_id,
                            "type": "high_token_count",
                            "message": f"Token count {token_count} above target maximum {self.TARGET_MAX_TOKENS}",
                        }
                    )
            else:
                warnings.append(
                    {
                        "chunk_id": chunk_id,
                        "type": "missing_token_count",
                        "message": "Missing token_count field",
                    }
                )

            # Check metadata completeness
            missing_important = []
            for field in self.IMPORTANT_FIELDS:
                if field not in chunk or chunk[field] is None:
                    missing_important.append(field)

            if missing_important:
                missing_metadata_count += 1
                warnings.append(
                    {
                        "chunk_id": chunk_id,
                        "type": "missing_metadata",
                        "message": f"Missing important fields: {', '.join(missing_important)}",
                    }
                )

            # Track corpus distribution
            corpus = chunk.get("corpus", "unknown")
            corpus_distribution[corpus] += 1

            # Update valid count
            if is_valid:
                valid_count += 1

        # Calculate quality metrics
        quality_metrics = self._calculate_quality_metrics(
            token_counts=token_counts,
            duplicate_count=duplicate_count,
            empty_text_count=empty_text_count,
            missing_metadata_count=missing_metadata_count,
            corpus_distribution=dict(corpus_distribution),
        )

        return ValidationReport(
            total_chunks=len(chunks),
            valid_chunks=valid_count,
            invalid_chunks=len(chunks) - valid_count,
            errors=errors,
            warnings=warnings,
            quality_metrics=quality_metrics,
        )

    def check_schema(self, chunk: Dict[str, Any]) -> List[str]:
        """
        Check schema compliance for a single chunk.

        Args:
            chunk: Chunk dictionary to validate

        Returns:
            List of error messages (empty if valid)
        """
        errors = []

        # Check critical fields
        for field in self.CRITICAL_FIELDS:
            if field not in chunk:
                errors.append(f"Missing critical field: {field}")
            elif chunk[field] is None:
                errors.append(f"Critical field is None: {field}")
            elif field == "text" and not isinstance(chunk[field], str):
                errors.append(f"Field 'text' must be string, got {type(chunk[field])}")
            elif field == "text" and chunk[field].strip() == "":
                errors.append("Field 'text' is empty")

        return errors

    def check_quality(self, chunks: List[Dict[str, Any]]) -> QualityMetrics:
        """
        Check quality metrics for chunks.

        Args:
            chunks: List of chunk dictionaries

        Returns:
            QualityMetrics object
        """
        token_counts = []
        seen_hashes: Set[str] = set()
        duplicate_count = 0
        empty_text_count = 0
        missing_metadata_count = 0
        corpus_distribution: Dict[str, int] = defaultdict(int)

        for chunk in chunks:
            # Token counts
            if "token_count" in chunk and chunk["token_count"] is not None:
                token_counts.append(chunk["token_count"])

            # Duplicates
            chunk_hash = chunk.get("hash", "")
            if chunk_hash:
                if chunk_hash in seen_hashes:
                    duplicate_count += 1
                else:
                    seen_hashes.add(chunk_hash)

            # Empty text
            if chunk.get("text", "").strip() == "":
                empty_text_count += 1

            # Missing metadata
            missing_important = [
                field
                for field in self.IMPORTANT_FIELDS
                if field not in chunk or chunk[field] is None
            ]
            if missing_important:
                missing_metadata_count += 1

            # Corpus distribution
            corpus = chunk.get("corpus", "unknown")
            corpus_distribution[corpus] += 1

        metrics_dict = self._calculate_quality_metrics(
            token_counts=token_counts,
            duplicate_count=duplicate_count,
            empty_text_count=empty_text_count,
            missing_metadata_count=missing_metadata_count,
            corpus_distribution=dict(corpus_distribution),
        )

        return QualityMetrics(
            avg_token_count=metrics_dict["avg_token_count"],
            min_token_count=metrics_dict["min_token_count"],
            max_token_count=metrics_dict["max_token_count"],
            token_count_std=metrics_dict["token_count_std"],
            duplicate_count=metrics_dict["duplicate_count"],
            empty_text_count=metrics_dict["empty_text_count"],
            missing_metadata_count=metrics_dict["missing_metadata_count"],
            corpus_distribution=metrics_dict["corpus_distribution"],
        )

    def _calculate_quality_metrics(
        self,
        token_counts: List[int],
        duplicate_count: int,
        empty_text_count: int,
        missing_metadata_count: int,
        corpus_distribution: Dict[str, int],
    ) -> Dict[str, Any]:
        """Calculate quality metrics from collected data."""
        if token_counts:
            avg_tokens = sum(token_counts) / len(token_counts)
            min_tokens = min(token_counts)
            max_tokens = max(token_counts)

            # Calculate standard deviation
            variance = sum((x - avg_tokens) ** 2 for x in token_counts) / len(
                token_counts
            )
            std_tokens = variance**0.5
        else:
            avg_tokens = 0
            min_tokens = 0
            max_tokens = 0
            std_tokens = 0

        return {
            "avg_token_count": round(avg_tokens, 2),
            "min_token_count": min_tokens,
            "max_token_count": max_tokens,
            "token_count_std": round(std_tokens, 2),
            "duplicate_count": duplicate_count,
            "empty_text_count": empty_text_count,
            "missing_metadata_count": missing_metadata_count,
            "corpus_distribution": corpus_distribution,
        }

    def validate_hash(self, chunk: Dict[str, Any]) -> bool:
        """
        Validate that chunk hash matches computed hash of text.

        Args:
            chunk: Chunk dictionary with 'text' and 'hash' fields

        Returns:
            True if hash is valid, False otherwise
        """
        if "hash" not in chunk or "text" not in chunk:
            return False

        # Compute hash of text (raw hex digest, no prefix)
        text = chunk["text"]
        computed_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()

        # Normalise stored hash to support both prefixed and raw formats
        stored_hash = str(chunk["hash"])
        if stored_hash.startswith("sha256:"):
            stored_hash = stored_hash.split(":", 1)[1]

        # Compare normalised stored hash with computed hash
        return stored_hash == computed_hash

    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text using tiktoken.

        Args:
            text: Text to count tokens for

        Returns:
            Token count
        """
        return len(self.encoding.encode(text))
