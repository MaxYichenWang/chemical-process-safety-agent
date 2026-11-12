"""
Batch Embedding Pipeline using OpenAI Batch API

This module provides utilities for generating embeddings using the OpenAI Batch API
with 50% cost savings, caching, resume capability, and progress tracking.
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import duckdb
from openai import OpenAI

from .. import config

DEFAULT_MAX_BATCH_REQUESTS = config.BATCH_MAX_REQUESTS


class BatchEmbeddingPipeline:
    """Orchestrate embedding generation using OpenAI Batch API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        cache_db: str = config.EMBEDDINGS_CACHE_DB,
        model: str = config.EMBEDDING_MODEL,
        dimensions: int = config.EMBEDDING_DIMENSIONS,
        max_batch_requests: int = DEFAULT_MAX_BATCH_REQUESTS,
        max_concurrent_batches: int = config.BATCH_MAX_CONCURRENT,
    ):
        """
        Initialize batch embedding pipeline.

        Args:
            api_key: OpenAI API key (if None, reads from OPENAI_API_KEY env var)
            cache_db: Path to DuckDB cache database
            model: Embedding model name
            dimensions: Embedding dimensions
            max_batch_requests: Maximum requests per sub-batch (default 5000)
            max_concurrent_batches: Max batches in flight simultaneously (default 3).
                3 × 5000 chunks × ~78 tokens ≈ 1.2M tokens, well under 3M limit.
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key must be provided or set in OPENAI_API_KEY environment variable"
            )

        self.client = OpenAI(api_key=self.api_key)
        self.model = model
        self.dimensions = dimensions
        self.cache_db = cache_db
        self.max_batch_requests = max_batch_requests
        self.max_concurrent_batches = max_concurrent_batches

        # Initialize cache database
        self._init_cache_db()

    def _init_cache_db(self) -> None:
        """Initialize cache database with required tables."""
        # Create parent directory if needed
        cache_path = Path(self.cache_db)
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        conn = duckdb.connect(self.cache_db)

        try:
            # Create embedding_cache table
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS embedding_cache (
                    text_hash TEXT PRIMARY KEY,
                    embedding BLOB,
                    model TEXT,
                    dimensions INT,
                    created_at TIMESTAMP,
                    batch_job_id TEXT
                )
            """
            )

            # Create batch_jobs table
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS batch_jobs (
                    job_id TEXT PRIMARY KEY,
                    corpus TEXT,
                    file_id TEXT,
                    status TEXT,
                    created_at TIMESTAMP,
                    completed_at TIMESTAMP,
                    chunks_count INT,
                    cost_estimate REAL,
                    error_message TEXT
                )
            """
            )

            conn.commit()

        finally:
            conn.close()

    def create_batch_file(
        self, chunks: List[Dict[str, Any]], output_path: str, use_cache: bool = True
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Create JSONL batch file for OpenAI Batch API.

        Args:
            chunks: List of chunk dictionaries with 'chunk_id', 'text', 'hash'
            output_path: Path to save JSONL file
            use_cache: Whether to check cache and skip cached chunks

        Returns:
            Tuple of (output_path, chunks_to_process)
        """
        chunks_to_process = []

        if use_cache:
            if len(chunks) > 1000:
                print(f"Checking cache for {len(chunks):,} chunks...", flush=True)
            conn = duckdb.connect(self.cache_db)
            try:
                all_hashes = [c.get("hash", "") for c in chunks]
                non_empty = [h for h in all_hashes if h]
                cached_hashes: set = set()
                if non_empty:
                    batch_sz = 10_000
                    for i in range(0, len(non_empty), batch_sz):
                        batch = non_empty[i : i + batch_sz]
                        placeholders = ", ".join(["?"] * len(batch))
                        rows = conn.execute(
                            f"SELECT text_hash FROM embedding_cache WHERE text_hash IN ({placeholders})",
                            batch,
                        ).fetchall()
                        cached_hashes.update(r[0] for r in rows)

                for chunk in chunks:
                    text_hash = chunk.get("hash", "")
                    if not text_hash or text_hash not in cached_hashes:
                        chunks_to_process.append(chunk)
            finally:
                conn.close()
        else:
            chunks_to_process = chunks

        if not chunks_to_process:
            return output_path, []

        # Create output directory if needed
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # Write JSONL file
        with open(output_path, "w", encoding="utf-8") as f:
            for chunk in chunks_to_process:
                request = {
                    "custom_id": chunk["chunk_id"],
                    "method": "POST",
                    "url": "/v1/embeddings",
                    "body": {
                        "model": self.model,
                        "input": chunk["text"],
                        "encoding_format": "float",
                        "dimensions": self.dimensions,
                    },
                }
                f.write(json.dumps(request) + "\n")

        return output_path, chunks_to_process

    def upload_batch_file(self, file_path: str) -> str:
        """
        Upload batch file to OpenAI.

        Args:
            file_path: Path to JSONL batch file

        Returns:
            File ID from OpenAI
        """
        with open(file_path, "rb") as f:
            response = self.client.files.create(file=f, purpose="batch")

        return response.id

    def create_batch_job(
        self, file_id: str, metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create batch job on OpenAI.

        Args:
            file_id: File ID from upload_batch_file
            metadata: Optional metadata to attach to batch job

        Returns:
            Batch job ID
        """
        response = self.client.batches.create(
            input_file_id=file_id,
            endpoint="/v1/embeddings",
            completion_window=config.BATCH_COMPLETION_WINDOW,
            metadata=metadata or {},
        )

        return response.id

    def monitor_batch_job(
        self,
        job_id: str,
        poll_interval: int = config.BATCH_POLL_INTERVAL_S,
        timeout_minutes: int = config.BATCH_TIMEOUT_MINUTES,
    ) -> Dict[str, Any]:
        """
        Monitor batch job progress until completion.

        Args:
            job_id: Batch job ID
            poll_interval: Seconds between status checks
            timeout_minutes: Maximum minutes to wait before raising TimeoutError

        Returns:
            Final batch job status dictionary

        Raises:
            TimeoutError: If job exceeds timeout_minutes without completing
        """
        print(f"Monitoring batch job: {job_id} (timeout: {timeout_minutes}min)")
        start_time = time.time()
        deadline = start_time + timeout_minutes * 60

        while True:
            batch = self.client.batches.retrieve(job_id)
            status = batch.status
            elapsed = (time.time() - start_time) / 60

            print(f"[{elapsed:.1f}min] Status: {status}", end="")

            if hasattr(batch, "request_counts"):
                counts = batch.request_counts
                total = counts.total if hasattr(counts, "total") else 0
                completed = counts.completed if hasattr(counts, "completed") else 0
                failed = counts.failed if hasattr(counts, "failed") else 0

                if total > 0:
                    print(
                        f" | Progress: {completed}/{total} ({completed/total*100:.1f}%) | Failed: {failed}"
                    )
                else:
                    print()
            else:
                print()

            # Check if complete
            if status in ["completed", "failed", "expired", "cancelled"]:
                return {
                    "job_id": job_id,
                    "status": status,
                    "output_file_id": (
                        batch.output_file_id
                        if hasattr(batch, "output_file_id")
                        else None
                    ),
                    "error_file_id": (
                        batch.error_file_id if hasattr(batch, "error_file_id") else None
                    ),
                    "request_counts": (
                        {
                            "total": counts.total if hasattr(counts, "total") else 0,
                            "completed": (
                                counts.completed if hasattr(counts, "completed") else 0
                            ),
                            "failed": counts.failed if hasattr(counts, "failed") else 0,
                        }
                        if hasattr(batch, "request_counts")
                        else {}
                    ),
                }

            # Check timeout
            if time.time() >= deadline:
                raise TimeoutError(
                    f"Batch job {job_id} timed out after {timeout_minutes} minutes. "
                    f"Last status: {status}. "
                    f"You can check status later via client.batches.retrieve('{job_id}')"
                )

            # Wait before next check
            time.sleep(poll_interval)

    def download_batch_results(
        self, job_id: str, output_file_id: str, chunks_by_id: Dict[str, Dict[str, Any]]
    ) -> List[Tuple[Dict[str, Any], List[float]]]:
        """
        Download completed embeddings from batch job.

        Args:
            job_id: Batch job ID
            output_file_id: Output file ID from completed batch
            chunks_by_id: Mapping of chunk_id to chunk dict for metadata

        Returns:
            List of (chunk, embedding) tuples
        """
        # Download file content
        file_response = self.client.files.content(output_file_id)
        file_content = file_response.read()

        # Parse JSONL results
        results = []
        lines = file_content.decode("utf-8").strip().split("\n")

        for line in lines:
            if not line.strip():
                continue

            result = json.loads(line)
            custom_id = result.get("custom_id", "")

            if custom_id not in chunks_by_id:
                print(f"Warning: Unknown chunk_id in results: {custom_id}")
                continue

            # Extract embedding
            if result.get("response", {}).get("status_code") == 200:
                embedding_data = result["response"]["body"]["data"][0]
                embedding = embedding_data["embedding"]

                chunk = chunks_by_id[custom_id]
                results.append((chunk, embedding))
            else:
                print(
                    f"Error processing chunk {custom_id}: {result.get('error', 'Unknown error')}"
                )

        # Store in cache
        self._cache_embeddings(results, job_id)

        return results

    def _cache_embeddings(
        self, results: List[Tuple[Dict[str, Any], List[float]]], job_id: str
    ) -> None:
        """Cache embeddings in DuckDB using bulk insert."""
        if not results:
            return

        conn = duckdb.connect(self.cache_db)
        now = datetime.now()

        try:
            # Build parameter list for bulk insert
            rows = []
            for chunk, embedding in results:
                text_hash = chunk.get("hash", "")
                if not text_hash:
                    continue
                embedding_bytes = json.dumps(embedding).encode("utf-8")
                rows.append(
                    (
                        text_hash,
                        embedding_bytes,
                        self.model,
                        self.dimensions,
                        now,
                        job_id,
                    )
                )

            if rows:
                conn.executemany(
                    """
                    INSERT OR REPLACE INTO embedding_cache
                    (text_hash, embedding, model, dimensions, created_at, batch_job_id)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    rows,
                )
                conn.commit()
                print(f"  Cached {len(rows):,} embeddings in bulk")

        finally:
            conn.close()

    def get_cached_embeddings(
        self, chunks: List[Dict[str, Any]]
    ) -> Dict[str, List[float]]:
        """
        Get cached embeddings for chunks.

        Args:
            chunks: List of chunk dictionaries with 'hash' field

        Returns:
            Dictionary mapping chunk_id to embedding (only cached chunks)
        """
        cached: Dict[str, List[float]] = {}
        hash_to_ids: Dict[str, List[str]] = {}
        for chunk in chunks:
            text_hash = chunk.get("hash", "")
            chunk_id = chunk.get("chunk_id", "")
            if text_hash:
                hash_to_ids.setdefault(text_hash, []).append(chunk_id)

        if not hash_to_ids:
            return cached

        unique_hashes = list(hash_to_ids.keys())
        print(
            f"Retrieving cached embeddings for {len(unique_hashes):,} unique hashes "
            f"({len(chunks):,} chunks)..."
        )

        conn = duckdb.connect(self.cache_db)
        try:
            batch_sz = 5000
            for i in range(0, len(unique_hashes), batch_sz):
                batch = unique_hashes[i : i + batch_sz]
                placeholders = ", ".join(["?"] * len(batch))
                rows = conn.execute(
                    f"SELECT text_hash, embedding FROM embedding_cache "
                    f"WHERE text_hash IN ({placeholders})",
                    batch,
                ).fetchall()
                for text_hash, emb_blob in rows:
                    embedding = json.loads(emb_blob.decode("utf-8"))
                    for cid in hash_to_ids.get(text_hash, []):
                        cached[cid] = embedding
                done = min(i + batch_sz, len(unique_hashes))
                print(
                    f"  Retrieved: {done:,}/{len(unique_hashes):,} hashes "
                    f"({done * 100 // len(unique_hashes)}%) — "
                    f"{len(cached):,} embeddings loaded",
                    flush=True,
                )
        finally:
            conn.close()

        print(f"  Done: {len(cached):,} cached embeddings loaded")
        return cached

    def _wait_for_enqueue_capacity(self, label: str, poll_interval: int = 30) -> None:
        """Block until no in_progress/validating/cancelling embedding batches remain."""
        while True:
            batches = self.client.batches.list(limit=100)
            blocking = [
                b
                for b in batches.data
                if b.endpoint == "/v1/embeddings"
                and b.status in ("in_progress", "validating", "cancelling")
            ]
            if not blocking:
                return
            summaries = [f"...{b.id[-12:]}({b.status})" for b in blocking]
            print(
                f"  {label} Waiting for {len(blocking)} in-flight batch(es) to clear: "
                f"{', '.join(summaries)}",
                flush=True,
            )
            time.sleep(poll_interval)

    def _submit_sub_batch(
        self,
        corpus: str,
        sub_chunks: List[Dict[str, Any]],
        batch_file_path: str,
        batch_index: int,
        total_batches: int,
        max_retries: int = 10,
        retry_wait: int = 60,
    ) -> Dict[str, Any]:
        """Submit a single sub-batch to OpenAI without waiting for completion.

        Includes retry with exponential backoff for token_limit_exceeded
        and rate_limit_exceeded errors from the Batch API.

        Returns dict with job_id, file_id, sub_chunks, and status.
        """
        label = f"[batch {batch_index + 1}/{total_batches}]"

        # Write JSONL for this sub-batch
        print(f"\n{label} Writing {len(sub_chunks):,} requests to JSONL...", flush=True)
        Path(batch_file_path).parent.mkdir(parents=True, exist_ok=True)
        with open(batch_file_path, "w", encoding="utf-8") as f:
            for idx_w, chunk in enumerate(sub_chunks):
                request = {
                    "custom_id": chunk["chunk_id"],
                    "method": "POST",
                    "url": "/v1/embeddings",
                    "body": {
                        "model": self.model,
                        "input": chunk["text"],
                        "encoding_format": "float",
                        "dimensions": self.dimensions,
                    },
                }
                f.write(json.dumps(request) + "\n")
                if (idx_w + 1) % 50_000 == 0:
                    print(
                        f"  {label} Wrote {idx_w + 1:,}/{len(sub_chunks):,} lines",
                        flush=True,
                    )
        print(f"  {label} JSONL write complete: {len(sub_chunks):,} lines")

        print(f"{label} Uploading {len(sub_chunks):,} requests...")
        file_id = self.upload_batch_file(batch_file_path)
        print(f"{label} File uploaded: {file_id}")

        # Retry loop for batch creation (handles token limit errors)
        job_id = None
        for attempt in range(max_retries):
            try:
                print(f"{label} Creating batch job...")
                job_id = self.create_batch_job(
                    file_id=file_id,
                    metadata={"corpus": corpus, "batch_index": str(batch_index)},
                )
                print(f"{label} Batch job created: {job_id}")
                break
            except Exception as e:
                err_str = str(e).lower()
                if "token_limit" in err_str or "rate_limit" in err_str:
                    wait_secs = retry_wait * (attempt + 1)
                    print(
                        f"  {label} ⚠ Token/rate limit hit. "
                        f"Waiting {wait_secs}s (attempt {attempt + 1}/{max_retries})...",
                        flush=True,
                    )
                    time.sleep(wait_secs)
                else:
                    raise

        if job_id is None:
            return {
                "job_id": "",
                "file_id": file_id,
                "batch_index": batch_index,
                "sub_chunks": sub_chunks,
                "status": "failed",
                "error": "Max retries exceeded for batch creation",
            }

        self._store_batch_job(
            job_id=job_id,
            corpus=corpus,
            file_id=file_id,
            chunks_count=len(sub_chunks),
            cost_estimate=self.estimate_cost(sub_chunks),
        )

        return {
            "job_id": job_id,
            "file_id": file_id,
            "batch_index": batch_index,
            "sub_chunks": sub_chunks,
            "status": "submitted",
        }

    def generate_embeddings_for_corpus(
        self,
        corpus: str,
        chunks: List[Dict[str, Any]],
        output_dir: str = config.BATCH_FILES_DIR,
        resume: bool = True,
        wait_for_completion: bool = True,
    ) -> Dict[str, Any]:
        """
        Generate embeddings for a corpus using batch API.

        Automatically splits into sub-batches and submits them in parallel.
        All sub-batches are submitted upfront, then monitored concurrently
        in a single polling loop.

        Args:
            corpus: Corpus identifier
            chunks: List of chunk dictionaries
            output_dir: Directory for temporary batch files
            resume: Whether to skip cached chunks
            wait_for_completion: Whether to wait for batch to complete

        Returns:
            Dictionary with job info and results (if wait_for_completion=True)
        """
        print(f"\n{'='*80}")
        print(f"Processing corpus: {corpus}")
        print(f"Total chunks: {len(chunks):,}")
        print(f"{'='*80}\n")

        # Filter out cached chunks using batch lookup for performance
        chunks_to_process = []
        if resume:
            print("Checking embedding cache for already-processed chunks...")
            conn = duckdb.connect(self.cache_db)
            try:
                all_hashes = [c.get("hash", "") for c in chunks]
                non_empty_hashes = [h for h in all_hashes if h]
                print(f"  Collected {len(non_empty_hashes):,} hashes to check")

                cached_hashes: set = set()
                if non_empty_hashes:
                    batch_sz = 10_000
                    total_batches = (len(non_empty_hashes) + batch_sz - 1) // batch_sz
                    for batch_idx in range(0, len(non_empty_hashes), batch_sz):
                        batch = non_empty_hashes[batch_idx : batch_idx + batch_sz]
                        placeholders = ", ".join(["?"] * len(batch))
                        rows = conn.execute(
                            f"SELECT text_hash FROM embedding_cache WHERE text_hash IN ({placeholders})",
                            batch,
                        ).fetchall()
                        cached_hashes.update(r[0] for r in rows)
                        done = min(batch_idx + batch_sz, len(non_empty_hashes))
                        print(
                            f"  Cache check progress: {done:,}/{len(non_empty_hashes):,} "
                            f"({done * 100 // len(non_empty_hashes)}%) — "
                            f"{len(cached_hashes):,} hits so far",
                            flush=True,
                        )

                for chunk in chunks:
                    text_hash = chunk.get("hash", "")
                    if not text_hash or text_hash not in cached_hashes:
                        chunks_to_process.append(chunk)
                print(
                    f"  Cache check complete: {len(cached_hashes):,} cached, "
                    f"{len(chunks_to_process):,} to process"
                )
            finally:
                conn.close()
        else:
            chunks_to_process = list(chunks)

        if not chunks_to_process:
            print(f"All chunks already cached for {corpus}. Skipping.")
            cached = self.get_cached_embeddings(chunks)
            results = [
                (chunk, cached[chunk["chunk_id"]])
                for chunk in chunks
                if chunk["chunk_id"] in cached
            ]
            return {
                "corpus": corpus,
                "status": "cached",
                "total_chunks": len(chunks),
                "processed_chunks": len(results),
                "results": results,
            }

        cached_count = len(chunks) - len(chunks_to_process)
        print(f"Chunks to process (after cache): {len(chunks_to_process):,}")
        if cached_count > 0:
            print(f"Chunks already cached: {cached_count:,}")

        # Split into sub-batches of max_batch_requests
        sub_batches = [
            chunks_to_process[i : i + self.max_batch_requests]
            for i in range(0, len(chunks_to_process), self.max_batch_requests)
        ]
        total_batches = len(sub_batches)

        if total_batches > 1:
            print(
                f"\nSplitting into {total_batches} sub-batches "
                f"(max {self.max_batch_requests:,} per batch, "
                f"max {self.max_concurrent_batches} concurrent)"
            )

        # Wait for any blocking batches before starting
        self._wait_for_enqueue_capacity("[init]")

        # --- Sliding window: keep max_concurrent_batches in flight ---
        all_new_results: List[Tuple[Dict[str, Any], List[float]]] = []
        all_job_ids: List[str] = []
        any_failed = False
        next_to_submit = 0  # index into sub_batches

        # Active jobs currently in flight on OpenAI
        pending: Dict[str, Dict[str, Any]] = {}  # job_id -> job_info

        start_time = time.time()
        timeout_deadline = start_time + config.BATCH_GLOBAL_TIMEOUT_MINUTES * 60

        def _submit_next() -> None:
            """Submit the next batch from the queue."""
            nonlocal next_to_submit
            idx = next_to_submit
            batch_file = Path(output_dir) / f"{corpus}_batch_{idx}.jsonl"
            job_info = self._submit_sub_batch(
                corpus=corpus,
                sub_chunks=sub_batches[idx],
                batch_file_path=str(batch_file),
                batch_index=idx,
                total_batches=total_batches,
            )
            if job_info["status"] == "failed":
                all_job_ids.append(job_info.get("job_id", ""))
            else:
                pending[job_info["job_id"]] = job_info
                all_job_ids.append(job_info["job_id"])
            next_to_submit += 1

        # Fill the initial window
        while (
            next_to_submit < total_batches
            and len(pending) < self.max_concurrent_batches
        ):
            _submit_next()

        if not wait_for_completion:
            # Submit remaining without waiting
            while next_to_submit < total_batches:
                _submit_next()
            return {
                "corpus": corpus,
                "status": "submitted",
                "job_ids": all_job_ids,
                "chunks_count": len(chunks_to_process),
            }

        # Monitor loop: poll pending, download completed, backfill slots
        print(
            f"\nMonitoring {self.max_concurrent_batches} concurrent batch(es) "
            f"({total_batches} total, timeout: {config.BATCH_GLOBAL_TIMEOUT_MINUTES}min)...",
            flush=True,
        )

        while pending or next_to_submit < total_batches:
            elapsed = (time.time() - start_time) / 60
            completed_this_cycle = []

            for job_id, job_info in list(pending.items()):
                batch = self.client.batches.retrieve(job_id)
                status = batch.status
                label = f"[batch {job_info['batch_index'] + 1}/{total_batches}]"

                progress_str = ""
                if hasattr(batch, "request_counts"):
                    counts = batch.request_counts
                    t = counts.total if hasattr(counts, "total") else 0
                    c = counts.completed if hasattr(counts, "completed") else 0
                    f_count = counts.failed if hasattr(counts, "failed") else 0
                    if t > 0:
                        progress_str = f" {c}/{t} ({c/t*100:.0f}%)"
                    if f_count:
                        progress_str += f" failed:{f_count}"

                done_count = (
                    len(all_job_ids) - len(pending) - (total_batches - next_to_submit)
                    if next_to_submit < total_batches
                    else len(all_job_ids) - len(pending)
                )
                print(
                    f"  [{elapsed:.1f}min] {label} {status}{progress_str} "
                    f"(inflight:{len(pending)} done:{done_count}/{total_batches})",
                    flush=True,
                )

                if status == "completed":
                    output_file_id = (
                        batch.output_file_id
                        if hasattr(batch, "output_file_id")
                        else None
                    )
                    if output_file_id:
                        sub_chunks = job_info["sub_chunks"]
                        chunks_by_id = {c["chunk_id"]: c for c in sub_chunks}
                        results = self.download_batch_results(
                            job_id=job_id,
                            output_file_id=output_file_id,
                            chunks_by_id=chunks_by_id,
                        )
                        all_new_results.extend(results)
                        print(f"  {label} ✓ Downloaded {len(results):,} embeddings")
                    self._update_batch_job_status(job_id, "completed")
                    completed_this_cycle.append(job_id)

                elif status in ("failed", "expired", "cancelled"):
                    self._update_batch_job_status(
                        job_id,
                        status,
                        error_message=f"Job ended with status: {status}",
                    )
                    completed_this_cycle.append(job_id)
                    any_failed = True
                    print(f"  {label} ⚠ Job {status}")

            # Remove completed/failed from pending
            for jid in completed_this_cycle:
                del pending[jid]

            # Backfill: submit next batches to keep the window full
            while (
                next_to_submit < total_batches
                and len(pending) < self.max_concurrent_batches
            ):
                _submit_next()

            if not pending and next_to_submit >= total_batches:
                break

            # Check timeout
            if time.time() >= timeout_deadline:
                pending_ids = list(pending.keys())
                raise TimeoutError(
                    f"{len(pending)} batch job(s) timed out after {config.BATCH_GLOBAL_TIMEOUT_MINUTES} minutes. "
                    f"Pending IDs: {pending_ids}"
                )

            time.sleep(config.BATCH_POLL_INTERVAL_S)

        total_done = len(all_job_ids)
        print(
            f"\nAll {total_done}/{total_batches} batch jobs processed. "
            f"Total new embeddings: {len(all_new_results):,}"
        )

        # Merge new results with cached embeddings
        cached = self.get_cached_embeddings(chunks)
        new_ids = {c["chunk_id"] for c, _ in all_new_results}
        all_results = list(all_new_results)
        for chunk in chunks:
            cid = chunk["chunk_id"]
            if cid in cached and cid not in new_ids:
                all_results.append((chunk, cached[cid]))

        status = "completed" if not any_failed else "partial"
        return {
            "corpus": corpus,
            "status": status,
            "job_ids": all_job_ids,
            "total_chunks": len(chunks),
            "processed_chunks": len(all_new_results),
            "cached_chunks": cached_count,
            "results": all_results,
        }

    def _store_batch_job(
        self,
        job_id: str,
        corpus: str,
        file_id: str,
        chunks_count: int,
        cost_estimate: float,
    ) -> None:
        """Store batch job info in database."""
        conn = duckdb.connect(self.cache_db)

        try:
            conn.execute(
                """
                INSERT INTO batch_jobs
                (job_id, corpus, file_id, status, created_at, chunks_count, cost_estimate)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
                [
                    job_id,
                    corpus,
                    file_id,
                    "pending",
                    datetime.now(),
                    chunks_count,
                    cost_estimate,
                ],
            )

            conn.commit()

        finally:
            conn.close()

    def _update_batch_job_status(
        self, job_id: str, status: str, error_message: Optional[str] = None
    ) -> None:
        """Update batch job status in database."""
        conn = duckdb.connect(self.cache_db)

        try:
            conn.execute(
                """
                UPDATE batch_jobs
                SET status = ?, completed_at = ?, error_message = ?
                WHERE job_id = ?
            """,
                [status, datetime.now(), error_message, job_id],
            )

            conn.commit()

        finally:
            conn.close()

    def estimate_cost(self, chunks: List[Dict[str, Any]]) -> float:
        """
        Estimate API cost using batch API pricing.

        Args:
            chunks: List of chunk dictionaries

        Returns:
            Estimated cost in USD
        """
        price_per_1k_tokens = config.EMBEDDING_PRICE_PER_1K_TOKENS

        # Estimate tokens (use token_count if available, otherwise estimate)
        total_tokens = 0
        for chunk in chunks:
            if "token_count" in chunk and chunk["token_count"]:
                total_tokens += chunk["token_count"]
            else:
                # Rough estimate: ~200 tokens per chunk
                total_tokens += 200

        cost = (total_tokens / 1000) * price_per_1k_tokens

        return cost

    def get_batch_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Get batch job status from database.

        Args:
            job_id: Batch job ID

        Returns:
            Job status dictionary or None if not found
        """
        conn = duckdb.connect(self.cache_db)

        try:
            result = conn.execute(
                "SELECT * FROM batch_jobs WHERE job_id = ?", [job_id]
            ).fetchone()

            if result:
                columns = [desc[0] for desc in conn.description]
                return dict(zip(columns, result))

            return None

        finally:
            conn.close()

    def list_batch_jobs(self, corpus: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List batch jobs.

        Args:
            corpus: Optional corpus filter

        Returns:
            List of job status dictionaries
        """
        conn = duckdb.connect(self.cache_db)

        try:
            if corpus:
                result = conn.execute(
                    "SELECT * FROM batch_jobs WHERE corpus = ? ORDER BY created_at DESC",
                    [corpus],
                ).fetchall()
            else:
                result = conn.execute(
                    "SELECT * FROM batch_jobs ORDER BY created_at DESC"
                ).fetchall()

            columns = [desc[0] for desc in conn.description]

            return [dict(zip(columns, row)) for row in result]

        finally:
            conn.close()

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache stats
        """
        conn = duckdb.connect(self.cache_db)

        try:
            # Count cached embeddings
            cache_count = conn.execute(
                "SELECT COUNT(*) FROM embedding_cache"
            ).fetchone()[0]

            # Count batch jobs by status
            jobs = conn.execute(
                """
                SELECT status, COUNT(*) as count
                FROM batch_jobs
                GROUP BY status
            """
            ).fetchall()

            jobs_by_status = {row[0]: row[1] for row in jobs}

            # Total cost estimate
            total_cost = (
                conn.execute(
                    """
                SELECT SUM(cost_estimate)
                FROM batch_jobs
                WHERE status = 'completed'
            """
                ).fetchone()[0]
                or 0
            )

            return {
                "cached_embeddings": cache_count,
                "jobs_by_status": jobs_by_status,
                "total_cost_completed": round(total_cost, 2),
            }

        finally:
            conn.close()
