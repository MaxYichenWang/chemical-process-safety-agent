"""
Utility Functions for Data Processing

This module provides common utility functions for text processing, hashing,
token counting, and chunking operations.
"""

import hashlib
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .. import config


def count_tokens(text: str, model: str = "gpt-4") -> int:
    """
    Count the number of tokens in a text using tiktoken.

    Args:
        text: Input text to count tokens
        model: Model name for tokenizer (default: gpt-4)

    Returns:
        Number of tokens in the text

    Example:
        >>> count_tokens("Hello, world!")
        4
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        # Fallback to cl100k_base for unknown models
        encoding = tiktoken.get_encoding("cl100k_base")

    return len(encoding.encode(text))


def compute_hash(text: str) -> str:
    """
    Compute SHA256 hash of text for deduplication and change detection.

    Args:
        text: Input text to hash

    Returns:
        SHA256 hash as hexadecimal string with 'sha256:' prefix

    Example:
        >>> compute_hash("Hello, world!")
        'sha256:315f5bdb76d078c43b8ac0064e4a0164612b1fce77c869345bfc94c75894edd3'
    """
    hash_obj = hashlib.sha256(text.encode("utf-8"))
    return f"sha256:{hash_obj.hexdigest()}"


def generate_chunk_id(
    corpus: str, identifier: str, chunk_index: Optional[int] = None
) -> str:
    """
    Generate a unique chunk ID following the project naming convention.

    Args:
        corpus: Corpus identifier (e.g., 'osha_reg', 'csb_reports')
        identifier: Document/section identifier
        chunk_index: Optional chunk index for multi-chunk documents

    Returns:
        Unique chunk ID string

    Example:
        >>> generate_chunk_id("osha_reg", "1910.147(a)")
        'osha_reg_1910.147(a)'
        >>> generate_chunk_id("csb_reports", "West_Fertilizer_4.1", 2)
        'csb_reports_West_Fertilizer_4.1_chunk_2'
    """
    chunk_id = f"{corpus}_{identifier}"
    if chunk_index is not None:
        chunk_id += f"_chunk_{chunk_index}"
    return chunk_id


def chunk_text(
    text: str,
    chunk_size: int = config.CHUNK_SIZE,
    chunk_overlap: int = config.CHUNK_OVERLAP,
    separators: Optional[List[str]] = None,
) -> List[str]:
    """
    Split text into chunks using LangChain's RecursiveCharacterTextSplitter.

    Args:
        text: Input text to chunk
        chunk_size: Target size for each chunk in tokens
        chunk_overlap: Number of tokens to overlap between chunks
        separators: List of separators to use for splitting (default: paragraph/sentence)

    Returns:
        List of text chunks

    Example:
        >>> text = "First paragraph.\\n\\nSecond paragraph.\\n\\nThird paragraph."
        >>> chunks = chunk_text(text, chunk_size=20, chunk_overlap=5)
        >>> len(chunks)
        2
    """
    if separators is None:
        # Default separators: paragraph breaks, newlines, sentences, spaces
        separators = ["\n\n", "\n", ". ", " ", ""]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators,
        length_function=count_tokens,
        is_separator_regex=False,
    )

    return splitter.split_text(text)


def normalize_standard_number(standard: str) -> str:
    """
    Normalize OSHA standard numbers to consistent format.

    Args:
        standard: OSHA standard number (may have variations)

    Returns:
        Normalized standard number

    Example:
        >>> normalize_standard_number("§1910.147")
        '1910.147'
        >>> normalize_standard_number("29 CFR 1910.147(a)")
        '1910.147(a)'
    """
    # Remove common prefixes
    standard = re.sub(r"^(§|29\s*CFR\s*)", "", standard.strip())
    # Remove extra whitespace
    standard = re.sub(r"\s+", "", standard)
    return standard


def extract_section_hierarchy(section_id: str) -> Dict[str, Any]:
    """
    Extract hierarchical information from a section ID.

    Args:
        section_id: Section identifier (e.g., "1910.147(c)(4)(i)")

    Returns:
        Dictionary with hierarchy information

    Example:
        >>> extract_section_hierarchy("1910.147(c)(4)(i)")
        {'part': '1910', 'standard': '1910.147', 'level': 3}
    """
    match = re.match(r"(\d+)\.(\d+)(\([^)]+\))*", section_id)
    if not match:
        return {"part": "", "standard": section_id, "level": 0}

    part = match.group(1)
    standard = f"{match.group(1)}.{match.group(2)}"

    # Count levels by parentheses
    subsections = section_id[len(standard) :]
    level = subsections.count("(")

    return {"part": part, "standard": standard, "level": level}


def create_chunk_metadata(
    chunk_id: str,
    corpus: str,
    subcorpus: str,
    doc_type: str,
    text: str,
    source_path: str,
    source_url: Optional[str] = None,
    embeddings_version: str = config.EMBEDDINGS_VERSION,
    schema_version: int = config.EMBEDDINGS_SCHEMA_VERSION,
    build_manifest_id: Optional[str] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Create a standardized chunk metadata dictionary following the canonical schema.

    Args:
        chunk_id: Unique chunk identifier
        corpus: High-level corpus (osha_reg, osha_enf, csb)
        subcorpus: Sub-corpus identifier
        doc_type: Document type (regulation, enforcement_violation, csb_final_report, etc.)
        text: Chunk text content
        source_path: Path to source file
        source_url: Optional URL to source document
        embeddings_version: Version string for embeddings
        schema_version: Schema version number
        build_manifest_id: Optional build manifest ID
        **kwargs: Additional metadata fields

    Returns:
        Standardized metadata dictionary

    Example:
        >>> metadata = create_chunk_metadata(
        ...     chunk_id="osha_reg_1910.147",
        ...     corpus="osha_reg",
        ...     subcorpus="osha_reg",
        ...     doc_type="regulation",
        ...     text="Lock out tag out requirements...",
        ...     source_path="data/processed/osha/regulations/regulations_sections.parquet",
        ...     standard_number="1910.147"
        ... )
        >>> metadata['doc_type']
        'regulation'
    """
    token_count = count_tokens(text)
    text_hash = compute_hash(text)
    created_at = datetime.utcnow().isoformat() + "Z"

    if build_manifest_id is None:
        build_manifest_id = f"{corpus}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

    metadata = {
        "chunk_id": chunk_id,
        "corpus": corpus,
        "subcorpus": subcorpus,
        "doc_type": doc_type,
        "text": text,
        "token_count": token_count,
        "hash": text_hash,
        "schema_version": schema_version,
        "embeddings_version": embeddings_version,
        "build_manifest_id": build_manifest_id,
        "source_path": source_path,
        "source_url": source_url,
        "created_at": created_at,
    }

    # Add any additional metadata fields
    metadata.update(kwargs)

    return metadata


def format_toc_path(toc_entry: Dict[str, Any], separator: str = " / ") -> str:
    """
    Format a TOC entry into a hierarchical path string.

    Args:
        toc_entry: TOC entry dictionary with 'title' and optional parent info
        separator: String to separate path components

    Returns:
        Formatted path string

    Example:
        >>> entry = {'title': 'Mechanical Integrity', 'parent_title': 'Safety Issues'}
        >>> format_toc_path(entry)
        'Safety Issues / Mechanical Integrity'
    """
    path_parts = []

    if "parent_title" in toc_entry and toc_entry["parent_title"]:
        path_parts.append(toc_entry["parent_title"])

    path_parts.append(toc_entry["title"])

    return separator.join(path_parts)


def is_colab_environment() -> bool:
    """
    Detect if code is running in Google Colab environment.

    Returns:
        True if running in Colab, False otherwise
    """
    try:
        import google.colab  # noqa: F401

        return True
    except ImportError:
        return False


def get_data_path(
    relative_path: str, colab_drive_root: str = "/content/drive/MyDrive"
) -> str:
    """
    Get appropriate data path for local or Colab environment.

    Args:
        relative_path: Relative path from project root
        colab_drive_root: Root path for Google Drive in Colab

    Returns:
        Absolute path appropriate for the environment

    Example:
        >>> # In local environment
        >>> get_data_path("data/processed/osha/regulations")
        '/path/to/project/data/processed/osha/regulations'

        >>> # In Colab environment
        >>> get_data_path("data/processed/osha/regulations")
        '/content/drive/MyDrive/chemical-process-safety-agent/data/processed/osha/regulations'
    """
    import os

    if is_colab_environment():
        # Assume project is in chemical-process-safety-agent folder in Drive
        return os.path.join(
            colab_drive_root, "chemical-process-safety-agent", relative_path
        )
    else:
        # Local environment - use current working directory
        return os.path.abspath(relative_path)
