"""
Chunk Reader for Parquet Files using DuckDB

This module provides utilities for reading chunks from parquet files
using DuckDB for efficient zero-copy reads.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import duckdb


class ChunkReader:
    """Read chunks from parquet files using DuckDB."""

    def __init__(self, chunks_dir: str):
        """
        Initialize with directory containing chunk parquet files.

        Args:
            chunks_dir: Directory containing chunk parquet files
        """
        self.chunks_dir = Path(chunks_dir)
        if not self.chunks_dir.exists():
            raise ValueError(f"Chunks directory does not exist: {chunks_dir}")

        # Map corpus names to parquet files
        self.corpus_files = {
            "osha_reg": "osha_regulations_chunks.parquet",
            "osha_enf_accidents": "osha_accidents_chunks.parquet",
            "osha_enf_violations": "osha_violations_sample_chunks.parquet",
            "osha_enf_genduty": "osha_genduty_sample_chunks.parquet",
            "csb_reports": "csb_final_reports_chunks.parquet",
            "csb_recs": "csb_recommendations_chunks.parquet",
        }

    def read_all_chunks(self) -> List[Dict[str, Any]]:
        """
        Read all chunks from all parquet files.

        Returns:
            List of chunk dictionaries
        """
        all_chunks = []

        for corpus, filename in self.corpus_files.items():
            file_path = self.chunks_dir / filename
            if file_path.exists():
                chunks = self.read_by_file(filename)
                all_chunks.extend(chunks)

        return all_chunks

    def read_by_corpus(self, corpus: str) -> List[Dict[str, Any]]:
        """
        Read chunks for a specific corpus.

        Args:
            corpus: Corpus identifier (e.g., 'osha_reg', 'csb_reports')

        Returns:
            List of chunk dictionaries

        Raises:
            ValueError: If corpus is not recognized
        """
        if corpus not in self.corpus_files:
            raise ValueError(
                f"Unknown corpus: {corpus}. "
                f"Valid options: {list(self.corpus_files.keys())}"
            )

        filename = self.corpus_files[corpus]
        return self.read_by_file(filename)

    def read_by_file(self, filename: str) -> List[Dict[str, Any]]:
        """
        Read chunks from a specific parquet file.

        Args:
            filename: Name of parquet file (e.g., 'osha_regulations_chunks.parquet')

        Returns:
            List of chunk dictionaries

        Raises:
            FileNotFoundError: If file does not exist
        """
        file_path = self.chunks_dir / filename

        if not file_path.exists():
            raise FileNotFoundError(f"Chunk file not found: {file_path}")

        # Use DuckDB for efficient reading
        conn = duckdb.connect(":memory:")

        try:
            # Read parquet file
            query = f"SELECT * FROM read_parquet('{file_path}')"
            result = conn.execute(query).fetchall()
            columns = [desc[0] for desc in conn.description]

            # Convert to list of dictionaries
            chunks = [dict(zip(columns, row)) for row in result]

            return chunks

        finally:
            conn.close()

    def get_chunk_stats(self) -> Dict[str, Any]:
        """
        Get statistics about available chunks.

        Returns:
            Dictionary with statistics per corpus and totals
        """
        stats = {"by_corpus": {}, "total_chunks": 0, "total_files": 0}

        conn = duckdb.connect(":memory:")

        try:
            for corpus, filename in self.corpus_files.items():
                file_path = self.chunks_dir / filename

                if file_path.exists():
                    # Get row count efficiently
                    query = f"SELECT COUNT(*) FROM read_parquet('{file_path}')"
                    count = conn.execute(query).fetchone()[0]

                    # Get file size
                    file_size = file_path.stat().st_size

                    stats["by_corpus"][corpus] = {
                        "filename": filename,
                        "chunk_count": count,
                        "file_size_mb": round(file_size / (1024 * 1024), 2),
                        "exists": True,
                    }

                    stats["total_chunks"] += count
                    stats["total_files"] += 1
                else:
                    stats["by_corpus"][corpus] = {"filename": filename, "exists": False}

            return stats

        finally:
            conn.close()

    def sample_chunks(
        self, corpus: Optional[str] = None, n_samples: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Sample random chunks for validation or testing.

        Args:
            corpus: Optional corpus to sample from (if None, samples from all)
            n_samples: Number of samples to return

        Returns:
            List of sampled chunk dictionaries
        """
        conn = duckdb.connect(":memory:")

        try:
            if corpus:
                # Sample from specific corpus
                if corpus not in self.corpus_files:
                    raise ValueError(f"Unknown corpus: {corpus}")

                file_path = self.chunks_dir / self.corpus_files[corpus]
                if not file_path.exists():
                    raise FileNotFoundError(f"Chunk file not found: {file_path}")

                query = f"""
                    SELECT * FROM read_parquet('{file_path}')
                    ORDER BY random()
                    LIMIT {n_samples}
                """

            else:
                # Sample from all corpora
                # Build UNION query for all existing files
                queries = []
                for corpus_name, filename in self.corpus_files.items():
                    file_path = self.chunks_dir / filename
                    if file_path.exists():
                        queries.append(f"SELECT * FROM read_parquet('{file_path}')")

                if not queries:
                    return []

                union_query = " UNION ALL ".join(queries)
                query = f"""
                    SELECT * FROM ({union_query})
                    ORDER BY random()
                    LIMIT {n_samples}
                """

            result = conn.execute(query).fetchall()
            columns = [desc[0] for desc in conn.description]

            # Convert to list of dictionaries
            chunks = [dict(zip(columns, row)) for row in result]

            return chunks

        finally:
            conn.close()

    def list_available_corpora(self) -> List[str]:
        """
        List available corpora (files that exist).

        Returns:
            List of corpus identifiers that have existing parquet files
        """
        available = []

        for corpus, filename in self.corpus_files.items():
            file_path = self.chunks_dir / filename
            if file_path.exists():
                available.append(corpus)

        return available
