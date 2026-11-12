"""
Embedding Generation and Vector Storage Utilities

This module provides utilities for generating embeddings and storing them
in vector databases (ChromaDB).
"""

import os
from typing import Any, Dict, List, Optional

from openai import OpenAI
from tqdm import tqdm

import chromadb

from .. import config


class EmbeddingsGenerator:
    """
    Generate embeddings using OpenAI's embedding models.

    Supports batch processing and rate limiting for efficient embedding generation.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = config.EMBEDDING_MODEL,
        dimensions: int = config.EMBEDDING_DIMENSIONS,
        batch_size: int = config.OPENAI_BATCH_SIZE,
    ):
        """
        Initialize the embeddings generator.

        Args:
            api_key: OpenAI API key (if None, reads from OPENAI_API_KEY env var)
            model: Embedding model name
            dimensions: Embedding dimensions (1024 for text-embedding-3-large)
            batch_size: Number of texts to embed in each batch
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key must be provided or set in OPENAI_API_KEY environment variable"
            )

        self.client = OpenAI(api_key=self.api_key)
        self.model = model
        self.dimensions = dimensions
        self.batch_size = batch_size

    def embed_texts(
        self, texts: List[str], show_progress: bool = True
    ) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.

        Args:
            texts: List of texts to embed
            show_progress: Whether to show progress bar

        Returns:
            List of embedding vectors
        """
        embeddings = []

        # Process in batches
        iterator = range(0, len(texts), self.batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Generating embeddings")

        for i in iterator:
            batch = texts[i : i + self.batch_size]

            # Call OpenAI API
            response = self.client.embeddings.create(
                input=batch, model=self.model, dimensions=self.dimensions
            )

            # Extract embeddings from response
            batch_embeddings = [item.embedding for item in response.data]
            embeddings.extend(batch_embeddings)

        return embeddings

    def embed_single(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        response = self.client.embeddings.create(
            input=[text], model=self.model, dimensions=self.dimensions
        )

        return response.data[0].embedding


class ChromaDBManager:
    """
    Manage ChromaDB collections for vector storage and retrieval.

    Supports both local and Google Drive persistence for Colab compatibility.
    """

    def __init__(
        self,
        persist_directory: str = config.CHROMADB_DIR,
        collection_name: str = "default_collection",
        *,
        embedding_function: Optional[chromadb.EmbeddingFunction] = None,
    ):
        """
        Initialize ChromaDB manager.

        Args:
            persist_directory: Directory to persist ChromaDB data
            collection_name: Name of the collection to use
            embedding_function: Optional embedding function for this collection.
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name

        # Create directory if it doesn't exist
        os.makedirs(persist_directory, exist_ok=True)

        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=persist_directory)

        # Get or create collection
        collection_kwargs = {
            "name": collection_name,
            "metadata": {"hnsw:space": "cosine"},
        }
        if embedding_function is not None:
            collection_kwargs["embedding_function"] = embedding_function
        self.collection = self.client.get_or_create_collection(**collection_kwargs)

    def add_chunks(
        self,
        chunks: List[Dict[str, Any]],
        embeddings: Optional[List[List[float]]] = None,
        show_progress: bool = True,
    ) -> None:
        """
        Add chunks to the ChromaDB collection.

        Args:
            chunks: List of chunk dictionaries with 'chunk_id', 'text', and metadata
            embeddings: Precomputed vectors. If omitted, Chroma embeds documents
                with the collection's embedding function.
            show_progress: Whether to show progress bar
        """
        chunk_ids = [chunk["chunk_id"] for chunk in chunks]
        documents = [chunk["text"] for chunk in chunks]
        metadatas = [self._prepare_metadata(chunk) for chunk in chunks]

        if embeddings is not None:
            # Add with embeddings
            if show_progress:
                iterator = tqdm(range(0, len(chunks), 1000), desc="Adding to ChromaDB")
            else:
                iterator = range(0, len(chunks), 1000)

            for i in iterator:
                batch_end = min(i + 1000, len(chunks))
                self.collection.add(
                    ids=chunk_ids[i:batch_end],
                    embeddings=embeddings[i:batch_end],
                    documents=documents[i:batch_end],
                    metadatas=metadatas[i:batch_end],
                )
        else:
            # Add without embeddings (ChromaDB will use default embedding function)
            if show_progress:
                iterator = tqdm(range(0, len(chunks), 1000), desc="Adding to ChromaDB")
            else:
                iterator = range(0, len(chunks), 1000)

            for i in iterator:
                batch_end = min(i + 1000, len(chunks))
                self.collection.add(
                    ids=chunk_ids[i:batch_end],
                    documents=documents[i:batch_end],
                    metadatas=metadatas[i:batch_end],
                )

    def query(
        self,
        query_text: str,
        n_results: int = config.CHROMADB_DEFAULT_N_RESULTS,
        where: Optional[Dict[str, Any]] = None,
        where_document: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Query the ChromaDB collection.

        Args:
            query_text: Query text
            n_results: Number of results to return
            where: Metadata filters
            where_document: Document content filters

        Returns:
            Query results dictionary
        """
        return self.collection.query(
            query_texts=[query_text],
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

        # Copy all fields except 'text' (stored separately as document)
        for key, value in chunk.items():
            if key == "text":
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

    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the collection.

        Returns:
            Dictionary with collection statistics
        """
        count = self.collection.count()

        return {
            "name": self.collection_name,
            "count": count,
            "persist_directory": self.persist_directory,
        }


def create_embeddings_pipeline(
    chunks: List[Dict[str, Any]],
    api_key: Optional[str] = None,
    model: str = config.EMBEDDING_MODEL,
    dimensions: int = config.EMBEDDING_DIMENSIONS,
    chromadb_dir: str = config.CHROMADB_DIR,
    collection_name: str = "default_collection",
    show_progress: bool = True,
) -> ChromaDBManager:
    """
    Complete pipeline: generate embeddings and store in ChromaDB.

    Args:
        chunks: List of chunk dictionaries
        api_key: OpenAI API key
        model: Embedding model name
        dimensions: Embedding dimensions
        chromadb_dir: ChromaDB persistence directory
        collection_name: ChromaDB collection name
        show_progress: Whether to show progress bars

    Returns:
        ChromaDBManager instance with chunks loaded

    Example:
        >>> chunks = [
        ...     {"chunk_id": "1", "text": "Hello world", "doc_type": "test"},
        ...     {"chunk_id": "2", "text": "Goodbye world", "doc_type": "test"}
        ... ]
        >>> manager = create_embeddings_pipeline(
        ...     chunks=chunks,
        ...     api_key="sk-...",
        ...     collection_name="test_collection"
        ... )
        >>> print(manager.get_collection_stats())
    """
    # Generate embeddings
    generator = EmbeddingsGenerator(api_key=api_key, model=model, dimensions=dimensions)

    texts = [chunk["text"] for chunk in chunks]
    embeddings = generator.embed_texts(texts, show_progress=show_progress)

    # Store in ChromaDB
    manager = ChromaDBManager(
        persist_directory=chromadb_dir, collection_name=collection_name
    )

    manager.add_chunks(chunks, embeddings, show_progress=show_progress)

    return manager
