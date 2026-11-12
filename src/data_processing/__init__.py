"""
Data Processing Module for Chemical Process Safety LLM

This module provides data loading and chunking functionality for OSHA and CSB datasets
to be used in RAG (Retrieval-Augmented Generation) systems.
"""

__version__ = "1.0.0"

# Data loading components
from .data_loading import (
    ChunkReader,
    ChunkValidator,
    CSBDataLoader,
    OSHAEnforcementLoader,
    OSHARegulationsLoader,
    QualityMetrics,
    ValidationReport,
)

# Utility functions
from .utilities import (
    ChromaDBManager,
    EmbeddingsGenerator,
    chunk_text,
    compute_hash,
    count_tokens,
    create_embeddings_pipeline,
    generate_chunk_id,
)

__all__ = [
    # Data loaders
    "CSBDataLoader",
    "OSHARegulationsLoader",
    "OSHAEnforcementLoader",
    # Chunking utilities
    "chunk_text",
    "count_tokens",
    "generate_chunk_id",
    "compute_hash",
    # Embeddings and ChromaDB utilities
    "EmbeddingsGenerator",
    "ChromaDBManager",
    "create_embeddings_pipeline",
    # Chunk reading/validation
    "ChunkReader",
    "ChunkValidator",
    "ValidationReport",
    "QualityMetrics",
]

# Embeddings and ChromaDB components (optional)
try:
    from .embeddings import (
        BatchEmbeddingPipeline,
        ChromaDBIndexer,
    )
except ImportError:
    # Embeddings module is optional; base data loading utilities should still work
    BatchEmbeddingPipeline = None  # type: ignore[assignment]
    ChromaDBIndexer = None  # type: ignore[assignment]
else:
    __all__.extend(
        [
            "BatchEmbeddingPipeline",
            "ChromaDBIndexer",
        ]
    )
