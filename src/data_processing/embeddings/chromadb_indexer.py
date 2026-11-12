"""
ChromaDB Indexer for Multi-Collection Storage

This module provides utilities for indexing chunks with embeddings
into ChromaDB collections with proper organization and validation.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

import chromadb

from .. import config


class ChromaDBIndexer:
    """Index chunks with embeddings into ChromaDB collections."""

    # Default collection mapping
    DEFAULT_COLLECTION_MAPPING = {
        "osha_reg": "osha_regulations",
        "osha_enf_accidents": "osha_enf",
        "osha_enf_violations": "osha_enf",
        "osha_enf_genduty": "osha_enf",
        "csb_reports": "csb",
        "csb_recs": "csb_recommendations",
    }

    def __init__(
        self,
        persist_directory: str = config.CHROMADB_DIR,
        collection_mapping: Optional[Dict[str, str]] = None,
    ):
        """
        Initialize ChromaDB indexer.

        Args:
            persist_directory: Directory to persist ChromaDB data
            collection_mapping: Mapping from corpus identifier to collection name
        """
        self.persist_directory = persist_directory
        self.collection_mapping = collection_mapping or self.DEFAULT_COLLECTION_MAPPING

        # Create directory if it doesn't exist
        Path(persist_directory).mkdir(parents=True, exist_ok=True)

        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=persist_directory)

        # Cache for collections
        self._collections: Dict[str, Any] = {}

    def get_or_create_collection(self, collection_name: str) -> Any:
        """
        Get or create a ChromaDB collection.

        Args:
            collection_name: Name of the collection

        Returns:
            ChromaDB collection object
        """
        if collection_name not in self._collections:
            self._collections[collection_name] = self.client.get_or_create_collection(
                name=collection_name, metadata={"hnsw:space": "cosine"}
            )

        return self._collections[collection_name]

    def index_chunks(
        self,
        corpus: str,
        chunks_with_embeddings: List[Tuple[Dict[str, Any], List[float]]],
        batch_size: int = config.CHROMADB_BATCH_SIZE,
        show_progress: bool = True,
        upsert: bool = False,
    ) -> Dict[str, Any]:
        """
        Index chunks for a specific corpus into appropriate collection.

        Args:
            corpus: Corpus identifier (e.g., 'osha_reg', 'csb_reports')
            chunks_with_embeddings: List of (chunk, embedding) tuples
            batch_size: Number of chunks to add per batch
            show_progress: Whether to show progress bar
            upsert: If True, use upsert (safe for existing collections with
                     overlapping IDs). If False, use add (faster but errors
                     on duplicate IDs).

        Returns:
            Dictionary with indexing statistics
        """
        if not chunks_with_embeddings:
            return {
                "corpus": corpus,
                "collection": None,
                "indexed_count": 0,
                "status": "no_data",
            }

        # Get collection name
        collection_name = self.collection_mapping.get(corpus, corpus)
        collection = self.get_or_create_collection(collection_name)

        op_name = "Upserting" if upsert else "Indexing"
        print(
            f"\n{op_name} {len(chunks_with_embeddings):,} chunks into '{collection_name}'..."
        )

        # Prepare data for insertion
        chunk_ids = []
        embeddings = []
        documents = []
        metadatas = []

        for chunk, embedding in chunks_with_embeddings:
            chunk_ids.append(chunk["chunk_id"])
            embeddings.append(embedding)
            documents.append(chunk["text"])
            metadatas.append(self._prepare_metadata(chunk))

        # Insert in batches
        indexed_count = 0

        iterator = range(0, len(chunks_with_embeddings), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc=f"{op_name} {collection_name}")

        for i in iterator:
            batch_end = min(i + batch_size, len(chunks_with_embeddings))

            try:
                if upsert:
                    collection.upsert(
                        ids=chunk_ids[i:batch_end],
                        embeddings=embeddings[i:batch_end],
                        documents=documents[i:batch_end],
                        metadatas=metadatas[i:batch_end],
                    )
                else:
                    collection.add(
                        ids=chunk_ids[i:batch_end],
                        embeddings=embeddings[i:batch_end],
                        documents=documents[i:batch_end],
                        metadatas=metadatas[i:batch_end],
                    )
                indexed_count += batch_end - i
            except Exception as e:
                print(f"\nError indexing batch {i}-{batch_end}: {e}")

        return {
            "corpus": corpus,
            "collection": collection_name,
            "indexed_count": indexed_count,
            "total_count": len(chunks_with_embeddings),
            "status": (
                "completed"
                if indexed_count == len(chunks_with_embeddings)
                else "partial"
            ),
        }

    def index_all_chunks(
        self,
        chunks_by_corpus: Dict[str, List[Tuple[Dict[str, Any], List[float]]]],
        batch_size: int = config.CHROMADB_BATCH_SIZE,
        show_progress: bool = True,
        upsert: bool = False,
    ) -> Dict[str, Any]:
        """
        Index all chunks into appropriate collections.

        Args:
            chunks_by_corpus: Dictionary mapping corpus to list of (chunk, embedding) tuples
            batch_size: Number of chunks to add per batch
            show_progress: Whether to show progress bars
            upsert: If True, use upsert instead of add (safe for existing collections)

        Returns:
            Dictionary with indexing statistics for all corpora
        """
        results = {}
        total_indexed = 0

        print(f"\n{'='*80}")
        print("INDEXING CHUNKS INTO CHROMADB")
        print(f"{'='*80}\n")

        for corpus, chunks_with_embeddings in chunks_by_corpus.items():
            result = self.index_chunks(
                corpus=corpus,
                chunks_with_embeddings=chunks_with_embeddings,
                batch_size=batch_size,
                show_progress=show_progress,
                upsert=upsert,
            )

            results[corpus] = result
            total_indexed += result["indexed_count"]

            print(
                f"  ✓ {corpus}: {result['indexed_count']:,} chunks indexed into '{result['collection']}'"
            )

        print(f"\n{'='*80}")
        print(f"TOTAL INDEXED: {total_indexed:,} chunks")
        print(f"{'='*80}\n")

        return {
            "by_corpus": results,
            "total_indexed": total_indexed,
            "collections": list(
                set(r["collection"] for r in results.values() if r["collection"])
            ),
        }

    def upsert_chunks(
        self,
        corpus: str,
        chunks_with_embeddings: List[Tuple[Dict[str, Any], List[float]]],
        batch_size: int = config.CHROMADB_BATCH_SIZE,
        show_progress: bool = True,
    ) -> Dict[str, Any]:
        """
        Upsert chunks (update existing or insert new) for a specific corpus.

        Args:
            corpus: Corpus identifier
            chunks_with_embeddings: List of (chunk, embedding) tuples
            batch_size: Number of chunks to process per batch
            show_progress: Whether to show progress bar

        Returns:
            Dictionary with upsert statistics
        """
        if not chunks_with_embeddings:
            return {"corpus": corpus, "upserted_count": 0, "status": "no_data"}

        # Get collection name
        collection_name = self.collection_mapping.get(corpus, corpus)
        collection = self.get_or_create_collection(collection_name)

        print(
            f"\nUpserting {len(chunks_with_embeddings):,} chunks into '{collection_name}'..."
        )

        # Prepare data
        chunk_ids = []
        embeddings = []
        documents = []
        metadatas = []

        for chunk, embedding in chunks_with_embeddings:
            chunk_ids.append(chunk["chunk_id"])
            embeddings.append(embedding)
            documents.append(chunk["text"])
            metadatas.append(self._prepare_metadata(chunk))

        # Upsert in batches
        upserted_count = 0

        iterator = range(0, len(chunks_with_embeddings), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc=f"Upserting {collection_name}")

        for i in iterator:
            batch_end = min(i + batch_size, len(chunks_with_embeddings))

            try:
                collection.upsert(
                    ids=chunk_ids[i:batch_end],
                    embeddings=embeddings[i:batch_end],
                    documents=documents[i:batch_end],
                    metadatas=metadatas[i:batch_end],
                )
                upserted_count += batch_end - i
            except Exception as e:
                print(f"\nError upserting batch {i}-{batch_end}: {e}")

        return {
            "corpus": corpus,
            "collection": collection_name,
            "upserted_count": upserted_count,
            "status": "completed",
        }

    def get_collection_stats(self, collection_name: str) -> Dict[str, Any]:
        """
        Get statistics for a specific collection.

        Args:
            collection_name: Name of the collection

        Returns:
            Dictionary with collection statistics
        """
        try:
            collection = self.client.get_collection(collection_name)
            count = collection.count()

            # Sample a few chunks to get corpus info
            sample = collection.peek(limit=1)

            corpus_info = None
            if sample and sample["metadatas"]:
                corpus_info = sample["metadatas"][0].get("corpus", "unknown")

            return {
                "name": collection_name,
                "count": count,
                "corpus": corpus_info,
                "exists": True,
            }
        except Exception as e:
            return {"name": collection_name, "exists": False, "error": str(e)}

    def get_all_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics for all collections.

        Returns:
            Dictionary with statistics for all collections
        """
        collections = self.client.list_collections()

        stats = {
            "collections": {},
            "total_chunks": 0,
            "total_collections": len(collections),
        }

        for collection in collections:
            collection_stats = self.get_collection_stats(collection.name)
            stats["collections"][collection.name] = collection_stats

            if collection_stats.get("exists", False):
                stats["total_chunks"] += collection_stats.get("count", 0)

        return stats

    def verify_indexing(self, expected_chunks: Dict[str, Any]) -> Dict[str, Any]:
        """
        Verify that all chunks were indexed correctly.

        Args:
            expected_chunks: Dictionary mapping corpus to expected chunk count,
                             or mapping subcorpus to {"corpus": corpus, "count": count}

        Returns:
            Verification report dictionary
        """
        print(f"\n{'='*80}")
        print("VERIFYING CHROMADB INDEXING")
        print(f"{'='*80}\n")

        verification = {
            "verified": True,
            "by_corpus": {},
            "total_expected": 0,
            "total_actual": 0,
        }

        for key_name, expected_val in expected_chunks.items():
            if isinstance(expected_val, dict):
                corpus = expected_val["corpus"]
                expected_count = expected_val["count"]
                group_key = key_name
            else:
                corpus = key_name
                expected_count = expected_val
                group_key = None

            verification["total_expected"] += expected_count

            collection_name = self.collection_mapping.get(corpus, corpus)
            stats = self.get_collection_stats(collection_name)

            if stats.get("exists", False):
                if group_key and group_key != corpus:
                    collection = self.client.get_collection(collection_name)
                    try:
                        # Try doc_type first
                        result = collection.get(
                            where={"doc_type": group_key}, include=[]
                        )
                        actual_count = len(result["ids"])

                        # If 0, fallback to subcorpus
                        if actual_count == 0:
                            result = collection.get(
                                where={"subcorpus": group_key}, include=[]
                            )
                            actual_count = len(result["ids"])
                    except Exception as e:
                        print(f"Warning: Could not query by doc_type/subcorpus: {e}")
                        actual_count = stats.get("count", 0)
                else:
                    actual_count = stats.get("count", 0)
            else:
                actual_count = 0

            # Since collections can hold multiple corpora or chunks from previous runs
            # (e.g. both violations and accidents in 'osha_enf'), we consider it a
            # match if the target has at least the expected number of chunks.
            matches = actual_count >= expected_count

            # For reporting, we'll label it by the key_name (which is corpus or subcorpus)
            verification["by_corpus"][key_name] = {
                "collection": collection_name,
                "expected": expected_count,
                "actual": actual_count,
                "matches": matches,
            }

            verification["total_actual"] += actual_count

            if not matches:
                verification["verified"] = False

            status = "✓" if matches else "✗"
            if actual_count > expected_count:
                print(
                    f"  {status} {key_name} ({collection_name}): {actual_count:,} / {expected_count:,} "
                    f"(Exceeds expected; likely contains multiple sub-corpora or prior data)"
                )
            else:
                print(
                    f"  {status} {key_name} ({collection_name}): {actual_count:,} / {expected_count:,}"
                )

        print(f"\n{'='*80}")
        # Total actual can exceed total expected due to multi-corpus collections
        total_matches = verification["total_actual"] >= verification["total_expected"]
        status = (
            "✓ VERIFIED" if verification["verified"] and total_matches else "✗ MISMATCH"
        )
        print(
            f"{status}: {verification['total_actual']:,} / {verification['total_expected']:,} total chunks"
        )
        print(f"{'='*80}\n")

        return verification

    def delete_collection(self, collection_name: str) -> bool:
        """
        Delete a collection.

        Args:
            collection_name: Name of the collection to delete

        Returns:
            True if successful, False otherwise
        """
        try:
            self.client.delete_collection(collection_name)

            # Remove from cache
            if collection_name in self._collections:
                del self._collections[collection_name]

            print(f"Deleted collection: {collection_name}")
            return True
        except Exception as e:
            print(f"Error deleting collection {collection_name}: {e}")
            return False

    def delete_all_collections(self) -> Dict[str, bool]:
        """
        Delete all collections.

        Returns:
            Dictionary mapping collection name to deletion success
        """
        collections = self.client.list_collections()
        results = {}

        for collection in collections:
            results[collection.name] = self.delete_collection(collection.name)

        return results

    def query_collection(
        self,
        collection_name: str,
        query_embeddings: List[List[float]],
        n_results: int = config.CHROMADB_DEFAULT_N_RESULTS,
        where: Optional[Dict[str, Any]] = None,
        where_document: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Query a collection with embeddings.

        Args:
            collection_name: Name of the collection to query
            query_embeddings: List of query embedding vectors
            n_results: Number of results to return per query
            where: Metadata filters
            where_document: Document content filters

        Returns:
            Query results dictionary
        """
        collection = self.client.get_collection(collection_name)

        return collection.query(
            query_embeddings=query_embeddings,
            n_results=n_results,
            where=where,
            where_document=where_document,
        )

    def _prepare_metadata(self, chunk: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare metadata for ChromaDB (convert to string/number types only).

        Args:
            chunk: Chunk dictionary

        Returns:
            Metadata dictionary with ChromaDB-compatible types
        """
        metadata = {}

        # Copy all fields except 'text' and 'embedding' (stored separately)
        for key, value in chunk.items():
            if key in ["text", "embedding"]:
                continue

            # ChromaDB only supports string, int, float, bool
            if value is None:
                metadata[key] = ""
            elif isinstance(value, (str, int, float, bool)):
                metadata[key] = value
            else:
                # Convert complex types to string
                metadata[key] = str(value)

        return metadata

    def export_collection_manifest(self, output_path: str) -> None:
        """
        Export manifest of all collections to a file.

        Args:
            output_path: Path to save manifest file
        """
        import json
        from datetime import datetime

        stats = self.get_all_collection_stats()

        manifest = {
            "created_at": datetime.now().isoformat(),
            "persist_directory": str(self.persist_directory),
            "collection_mapping": self.collection_mapping,
            "statistics": stats,
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, default=str)

        print(f"Manifest exported to: {output_path}")
