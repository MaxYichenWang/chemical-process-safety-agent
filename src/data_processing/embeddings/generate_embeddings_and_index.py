#!/usr/bin/env python3
"""
Complete Orchestration Script for Embeddings Generation and ChromaDB Indexing

This script orchestrates the entire pipeline:
1. Read chunks from parquet files (DuckDB)
2. Validate chunks (schema, quality)
3. Generate embeddings (OpenAI Batch API with caching)
4. Store in ChromaDB (multiple collections)
5. Generate final report

Usage:
    # Full pipeline
    python -m src.data_processing.embeddings.generate_embeddings_and_index \\
        --chunks-dir data/loaded_chunks \\
        --chromadb-dir ./chromadb \\
        --resume

    # Validate only
    python -m src.data_processing.embeddings.generate_embeddings_and_index \\
        --chunks-dir data/loaded_chunks \\
        --validate-only

    # Generate embeddings only (skip ChromaDB)
    python -m src.data_processing.embeddings.generate_embeddings_and_index \\
        --chunks-dir data/loaded_chunks \\
        --embeddings-only \\
        --cache-db data/embeddings_cache.duckdb
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .. import config
from ..data_loading.chunk_reader import ChunkReader
from ..data_loading.chunk_validator import ChunkValidator
from .batch_embedding_pipeline import BatchEmbeddingPipeline
from .chromadb_indexer import ChromaDBIndexer


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate embeddings and index chunks into ChromaDB",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Input/output directories
    parser.add_argument(
        "--chunks-dir",
        type=str,
        default=config.CHUNKS_DIR,
        help=f"Directory containing chunk parquet files (default: {config.CHUNKS_DIR})",
    )
    parser.add_argument(
        "--chromadb-dir",
        type=str,
        default=config.CHROMADB_DIR,
        help=f"ChromaDB persistence directory (default: {config.CHROMADB_DIR})",
    )
    parser.add_argument(
        "--cache-db",
        type=str,
        default=config.EMBEDDINGS_CACHE_DB,
        help=f"DuckDB cache database path (default: {config.EMBEDDINGS_CACHE_DB})",
    )
    parser.add_argument(
        "--batch-files-dir",
        type=str,
        default=config.BATCH_FILES_DIR,
        help=f"Directory for temporary batch files (default: {config.BATCH_FILES_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=config.EMBEDDINGS_OUTPUT_DIR,
        help=f"Output directory for reports (default: {config.EMBEDDINGS_OUTPUT_DIR})",
    )

    # API configuration
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="OpenAI API key (if not set, reads from OPENAI_API_KEY env var)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=config.EMBEDDING_MODEL,
        help=f"Embedding model name (default: {config.EMBEDDING_MODEL})",
    )
    parser.add_argument(
        "--dimensions",
        type=int,
        default=config.EMBEDDING_DIMENSIONS,
        help=f"Embedding dimensions (default: {config.EMBEDDING_DIMENSIONS})",
    )

    # Pipeline control
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from previous run (skip cached chunks)",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate chunks (skip embeddings and indexing)",
    )
    parser.add_argument(
        "--embeddings-only",
        action="store_true",
        help="Only generate embeddings (skip ChromaDB indexing)",
    )
    parser.add_argument(
        "--skip-validation", action="store_true", help="Skip validation step"
    )
    parser.add_argument(
        "--corpus",
        type=str,
        default=None,
        help=(
            "Process only specific corpus. Valid keys: "
            "'osha_reg', 'osha_enf_accidents', 'osha_enf_violations', "
            "'osha_enf_genduty', 'csb_reports', 'csb_recs'"
        ),
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Process only N sample chunks (for testing)",
    )
    parser.add_argument(
        "--no-wait",
        action="store_true",
        help="Submit batch jobs without waiting for completion",
    )
    parser.add_argument(
        "--upsert",
        action="store_true",
        help="Use upsert instead of add when indexing into ChromaDB (safe for existing collections)",
    )
    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=None,
        help=(
            "Maximum number of requests per OpenAI Batch API sub-batch. "
            "Use a small value (e.g. 10) for testing. Default: 5000"
        ),
    )

    return parser.parse_args()


def _read_chunks(reader: ChunkReader, corpus: str | None) -> List[Dict[str, Any]]:
    message = (
        f"No chunks found in {reader.chunks_dir} "
        f"(corpus filter: {corpus or 'all'}). Prepare chunk Parquets first."
    )
    try:
        chunks = reader.read_by_corpus(corpus) if corpus else reader.read_all_chunks()
    except FileNotFoundError as exc:
        sys.exit(f"{message} {exc}")
    if not chunks:
        sys.exit(message)
    return chunks


def validate_chunks_step(
    reader: ChunkReader, corpus: str = None
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Step 1: Read and validate chunks.

    Args:
        reader: ChunkReader instance
        corpus: Optional corpus filter

    Returns:
        Tuple of (chunks, validation_report)
    """
    print(f"\n{'='*80}")
    print("STEP 1: READING AND VALIDATING CHUNKS")
    print(f"{'='*80}\n")

    # Read chunks
    if corpus:
        print(f"Reading chunks for corpus: {corpus}")
    else:
        print("Reading all chunks...")
    t0 = time.time()
    chunks = _read_chunks(reader, corpus)
    elapsed = time.time() - t0
    print(f"Read {len(chunks):,} chunks in {elapsed:.1f}s")

    # Validate chunks
    print("\nValidating chunks...", flush=True)
    t0 = time.time()
    validator = ChunkValidator()
    report = validator.validate(chunks)
    elapsed = time.time() - t0
    print(f"Validation completed in {elapsed:.1f}s")

    # Print summary
    print("\n" + report.summary())

    return chunks, report


def generate_embeddings_step(
    chunks: List[Dict[str, Any]],
    pipeline: BatchEmbeddingPipeline,
    batch_files_dir: str,
    resume: bool = True,
    wait_for_completion: bool = True,
) -> Dict[str, List[Tuple[Dict[str, Any], List[float]]]]:
    """
    Step 2: Generate embeddings using batch API.

    Args:
        chunks: List of chunk dictionaries
        pipeline: BatchEmbeddingPipeline instance
        batch_files_dir: Directory for temporary batch files
        resume: Whether to skip cached chunks
        wait_for_completion: Whether to wait for batch jobs to complete

    Returns:
        Dictionary mapping corpus to list of (chunk, embedding) tuples
    """
    print(f"\n{'='*80}")
    print("STEP 2: GENERATING EMBEDDINGS")
    print(f"{'='*80}\n")

    # Group chunks by corpus
    chunks_by_corpus: Dict[str, List[Dict[str, Any]]] = {}

    for chunk in chunks:
        corpus = chunk.get("corpus", "unknown")
        if corpus not in chunks_by_corpus:
            chunks_by_corpus[corpus] = []
        chunks_by_corpus[corpus].append(chunk)

    print(f"Chunks grouped into {len(chunks_by_corpus)} corpora:")
    for corpus, corpus_chunks in chunks_by_corpus.items():
        print(f"  - {corpus}: {len(corpus_chunks):,} chunks")

    # Generate embeddings for each corpus
    results_by_corpus = {}

    for corpus, corpus_chunks in chunks_by_corpus.items():
        result = pipeline.generate_embeddings_for_corpus(
            corpus=corpus,
            chunks=corpus_chunks,
            output_dir=batch_files_dir,
            resume=resume,
            wait_for_completion=wait_for_completion,
        )

        if result["status"] in ["completed", "cached", "partial"]:
            results_by_corpus[corpus] = result["results"]
        else:
            print(f"\nWarning: Corpus {corpus} not completed: {result['status']}")
            if not wait_for_completion:
                job_ids = result.get("job_ids", [result.get("job_id", "N/A")])
                print(f"  Job IDs: {job_ids}")

    # Print summary
    total_embeddings = sum(len(results) for results in results_by_corpus.values())
    print(f"\n{'='*80}")
    print(f"EMBEDDINGS GENERATED: {total_embeddings:,} total")
    print(f"{'='*80}\n")

    return results_by_corpus


def index_chunks_step(
    chunks_by_corpus: Dict[str, List[Tuple[Dict[str, Any], List[float]]]],
    indexer: ChromaDBIndexer,
    upsert: bool = False,
) -> Dict[str, Any]:
    """
    Step 3: Index chunks into ChromaDB.

    Args:
        chunks_by_corpus: Dictionary mapping corpus to (chunk, embedding) tuples
        indexer: ChromaDBIndexer instance
        upsert: If True, use upsert instead of add (safe for existing collections)

    Returns:
        Indexing results dictionary
    """
    print(f"\n{'='*80}")
    print("STEP 3: INDEXING INTO CHROMADB")
    print(f"{'='*80}\n")

    results = indexer.index_all_chunks(
        chunks_by_corpus=chunks_by_corpus,
        batch_size=1000,
        show_progress=True,
        upsert=upsert,
    )

    return results


def verify_step(
    indexer: ChromaDBIndexer, chunks_by_corpus: Dict[str, List[Any]]
) -> Dict[str, Any]:
    """
    Step 4: Verify indexing.

    Args:
        indexer: ChromaDBIndexer instance
        chunks_by_corpus: Dictionary mapping corpus to chunks

    Returns:
        Verification results dictionary
    """
    print(f"\n{'='*80}")
    print("STEP 4: VERIFICATION")
    print(f"{'='*80}\n")

    # Build expected counts by doc_type or subcorpus
    expected_groups = {}
    for corpus, chunks in chunks_by_corpus.items():
        for chunk_tuple in chunks:
            chunk = chunk_tuple[0]  # chunk is the first item in the tuple
            # Use doc_type if available, otherwise subcorpus, otherwise corpus
            group_key = chunk.get("doc_type") or chunk.get("subcorpus") or corpus
            if group_key not in expected_groups:
                expected_groups[group_key] = {"corpus": corpus, "count": 0}
            expected_groups[group_key]["count"] += 1

    # Verify indexing
    verification = indexer.verify_indexing(expected_groups)

    return verification


def generate_final_report(
    output_dir: str,
    validation_report: Any,
    embedding_results: Dict[str, Any],
    indexing_results: Dict[str, Any],
    verification_results: Dict[str, Any],
    pipeline: BatchEmbeddingPipeline,
) -> None:
    """
    Generate final report.

    Args:
        output_dir: Output directory for report
        validation_report: Validation report
        embedding_results: Embedding generation results
        indexing_results: Indexing results
        verification_results: Verification results
        pipeline: BatchEmbeddingPipeline for cache stats
    """
    print(f"\n{'='*80}")
    print("GENERATING FINAL REPORT")
    print(f"{'='*80}\n")

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Get cache stats
    cache_stats = pipeline.get_cache_stats()

    # Build report
    report = {
        "generated_at": datetime.now().isoformat(),
        "validation": {
            "total_chunks": validation_report.total_chunks,
            "valid_chunks": validation_report.valid_chunks,
            "invalid_chunks": validation_report.invalid_chunks,
            "error_count": len(validation_report.errors),
            "warning_count": len(validation_report.warnings),
            "quality_metrics": validation_report.quality_metrics,
        },
        "embeddings": embedding_results,
        "indexing": indexing_results,
        "verification": verification_results,
        "cache": cache_stats,
    }

    # Save JSON report
    json_path = (
        Path(output_dir)
        / f"embedding_pipeline_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"JSON report saved to: {json_path}")

    # Generate markdown report
    md_path = (
        Path(output_dir)
        / f"embedding_pipeline_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    )

    md_lines = [
        "# Embedding Pipeline Report",
        "",
        f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Summary",
        "",
        f"- **Total Chunks Processed**: {validation_report.total_chunks:,}",
        f"- **Valid Chunks**: {validation_report.valid_chunks:,}",
        f"- **Total Indexed**: {indexing_results.get('total_indexed', 0):,}",
        f"- **Verification**: {'✓ PASSED' if verification_results.get('verified', False) else '✗ FAILED'}",
        f"- **Total Cost**: ${cache_stats.get('total_cost_completed', 0):.2f}",
        "",
        "## Validation Results",
        "",
        f"- Valid: {validation_report.valid_chunks:,}",
        f"- Invalid: {validation_report.invalid_chunks:,}",
        f"- Errors: {len(validation_report.errors):,}",
        f"- Warnings: {len(validation_report.warnings):,}",
        "",
        "### Quality Metrics",
        "",
    ]

    for key, value in validation_report.quality_metrics.items():
        if isinstance(value, dict):
            md_lines.append(f"- **{key}**:")
            for k, v in value.items():
                md_lines.append(f"  - {k}: {v:,}")
        else:
            md_lines.append(f"- **{key}**: {value}")

    md_lines.extend(
        [
            "",
            "## Indexing Results",
            "",
            f"- **Total Indexed**: {indexing_results.get('total_indexed', 0):,}",
            f"- **Collections**: {', '.join(indexing_results.get('collections', []))}",
            "",
            "### By Corpus",
            "",
        ]
    )

    for corpus, result in indexing_results.get("by_corpus", {}).items():
        md_lines.append(
            f"- **{corpus}**: {result.get('indexed_count', 0):,} chunks → `{result.get('collection', 'N/A')}`"
        )

    md_lines.extend(
        [
            "",
            "## Verification Results",
            "",
            f"- **Verified**: {'✓ YES' if verification_results.get('verified', False) else '✗ NO'}",
            f"- **Total Expected**: {verification_results.get('total_expected', 0):,}",
            f"- **Total Actual**: {verification_results.get('total_actual', 0):,}",
            "",
            "### By Corpus",
            "",
        ]
    )

    for corpus, result in verification_results.get("by_corpus", {}).items():
        status = "✓" if result.get("matches", False) else "✗"
        md_lines.append(
            f"- {status} **{corpus}**: {result.get('actual', 0):,} / {result.get('expected', 0):,} "
            f"(collection: `{result.get('collection', 'N/A')}`)"
        )

    md_lines.extend(
        [
            "",
            "## Cache Statistics",
            "",
            f"- **Cached Embeddings**: {cache_stats.get('cached_embeddings', 0):,}",
            f"- **Total Cost (Completed Jobs)**: ${cache_stats.get('total_cost_completed', 0):.2f}",
            "",
            "### Batch Jobs by Status",
            "",
        ]
    )

    for status, count in cache_stats.get("jobs_by_status", {}).items():
        md_lines.append(f"- {status}: {count}")

    md_lines.append("")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    print(f"Markdown report saved to: {md_path}")
    print(f"\n{'='*80}")
    print("PIPELINE COMPLETE!")
    print(f"{'='*80}\n")


def main():
    """Main orchestration function."""
    args = parse_args()

    # Print configuration
    print(f"\n{'='*80}")
    print("EMBEDDINGS AND INDEXING PIPELINE")
    print(f"{'='*80}\n")
    print("Configuration:")
    print(f"  Chunks Directory: {args.chunks_dir}")
    print(f"  ChromaDB Directory: {args.chromadb_dir}")
    print(f"  Cache Database: {args.cache_db}")
    print(f"  Model: {args.model}")
    print(f"  Dimensions: {args.dimensions}")
    print(f"  Resume: {args.resume}")
    print(f"  Corpus Filter: {args.corpus or 'All'}")
    print(f"  Sample: {args.sample or 'All'}")

    # Initialize components
    print("\nInitializing components...")

    reader = ChunkReader(args.chunks_dir)

    # Step 1: Read and validate chunks
    if not args.skip_validation:
        chunks, validation_report = validate_chunks_step(reader, args.corpus)

        if args.validate_only:
            print("\nValidation complete. Exiting (--validate-only flag set).")
            sys.exit(0)

    else:
        print("\nSKIPPING VALIDATION (--skip-validation flag set)")
        chunks = _read_chunks(reader, args.corpus)
        print(f"Read {len(chunks):,} chunks")
        validation_report = None

    # Initialize pipeline and indexer only if needed (not validate-only)
    pipeline_kwargs = dict(
        api_key=args.api_key,
        cache_db=args.cache_db,
        model=args.model,
        dimensions=args.dimensions,
    )
    if args.max_batch_size is not None:
        pipeline_kwargs["max_batch_requests"] = args.max_batch_size
    pipeline = BatchEmbeddingPipeline(**pipeline_kwargs)
    indexer = ChromaDBIndexer(persist_directory=args.chromadb_dir)

    # Deduplicate by chunk_id (source JOINs can produce duplicate rows)
    print("\nDeduplicating chunks by chunk_id...", flush=True)
    seen_ids: set = set()
    unique_chunks = []
    for chunk in chunks:
        cid = chunk.get("chunk_id", "")
        if cid not in seen_ids:
            seen_ids.add(cid)
            unique_chunks.append(chunk)
    if len(unique_chunks) < len(chunks):
        print(
            f"Deduplicated: {len(chunks):,} -> {len(unique_chunks):,} "
            f"({len(chunks) - len(unique_chunks):,} duplicate chunk_ids removed)"
        )
    else:
        print(f"No duplicates found ({len(chunks):,} unique chunk_ids)")
    chunks = unique_chunks

    # Apply sample if specified
    if args.sample:
        print(f"\nSampling {args.sample} chunks for testing...")
        chunks = chunks[: args.sample]

    # Step 2: Generate embeddings
    chunks_by_corpus_with_embeddings = generate_embeddings_step(
        chunks=chunks,
        pipeline=pipeline,
        batch_files_dir=args.batch_files_dir,
        resume=args.resume,
        wait_for_completion=not args.no_wait,
    )

    if args.no_wait:
        print("\nBatch jobs submitted. Exiting (--no-wait flag set).")
        print("Run this script again later to check status and index results.")
        sys.exit(0)

    if args.embeddings_only:
        print("\nEmbeddings generated. Exiting (--embeddings-only flag set).")
        sys.exit(0)

    # Step 3: Index into ChromaDB
    indexing_results = index_chunks_step(
        chunks_by_corpus=chunks_by_corpus_with_embeddings,
        indexer=indexer,
        upsert=args.upsert,
    )

    # Step 4: Verify indexing
    verification_results = verify_step(
        indexer=indexer, chunks_by_corpus=chunks_by_corpus_with_embeddings
    )

    # Generate final report
    if validation_report:
        generate_final_report(
            output_dir=args.output_dir,
            validation_report=validation_report,
            embedding_results={
                "total_embeddings": sum(
                    len(r) for r in chunks_by_corpus_with_embeddings.values()
                ),
                "by_corpus": {
                    corpus: len(results)
                    for corpus, results in chunks_by_corpus_with_embeddings.items()
                },
            },
            indexing_results=indexing_results,
            verification_results=verification_results,
            pipeline=pipeline,
        )

    # Export collection manifest
    manifest_path = Path(args.chromadb_dir) / "collections_manifest.json"
    indexer.export_collection_manifest(str(manifest_path))


if __name__ == "__main__":
    main()
