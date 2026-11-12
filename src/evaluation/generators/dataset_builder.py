"""
Dataset Builder for overarching evaluation dataset aggregation.

Assembles parsed, synthetic, factual, and algorithmic QA JSONs
into the final validated evaluation set, pruning duplicates via vector similarity.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from langchain_openai import OpenAIEmbeddings

from src.evaluation.config import (
    EVAL_DATASET_DIR,
    EVAL_DEDUP_EMBEDDING_MODEL,
    EVAL_DEDUP_SIMILARITY_THRESHOLD,
)
from src.evaluation.schemas import validate_eval_dataset

logger = logging.getLogger(__name__)


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two 1D vectors."""
    n_a = np.linalg.norm(a)
    n_b = np.linalg.norm(b)
    if n_a == 0 or n_b == 0:
        return 0.0
    return float(np.dot(a, b) / (n_a * n_b))


class DatasetBuilder:
    """Consolidates QA generation sources into standard distribution."""

    def __init__(self, output_file: Path | None = None):
        self.embeddings = OpenAIEmbeddings(model=EVAL_DEDUP_EMBEDDING_MODEL)
        self.output_file = output_file or EVAL_DATASET_DIR / "eval_dataset_v1.json"

    def build_dataset(self) -> list[dict]:
        """Load, deduplicate, validate, and persist the unified Phase 1 Dataset."""
        entries = []

        # Load all seed chunks
        for file_path in sorted(EVAL_DATASET_DIR.glob("*.json")):
            if file_path.name in (
                "eval_dataset_v1.json",
                "eval_dataset_adversarial.json",
            ):
                continue

            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        entries.extend(data)
                        logger.info(f"Loaded {len(data)} entries from {file_path.name}")
            except Exception as e:
                logger.warning(f"Failed to load {file_path.name}: {e}")

        if not entries:
            logger.error("No raw evaluation entries discovered. Aborting merge.")
            return []

        logger.info(f"Gross total entries retrieved: {len(entries)}")

        # Extract queries for bulk embedding
        queries = [str(e.get("query", "")) for e in entries]

        try:
            logger.info("Computing dense embeddings for duplicate detection...")
            query_embeddings = self.embeddings.embed_documents(queries)
        except Exception as e:
            logger.error(f"Embedding resolution failed natively: {e}")
            return []

        unique_entries = []
        kept_indices = []

        # Greedy O(N^2) selection (perfectly fine for N < 1000)
        for i, emb in enumerate(query_embeddings):
            is_duplicate = False
            for j in kept_indices:
                if (
                    cosine_similarity(emb, query_embeddings[j])
                    > EVAL_DEDUP_SIMILARITY_THRESHOLD
                ):
                    is_duplicate = True
                    logger.debug(
                        f"Pruned duplicate query: '{queries[i][:50]}...' matching '{queries[j][:50]}...'"
                    )
                    break

            if not is_duplicate:
                unique_entries.append(entries[i])
                kept_indices.append(i)

        dupes_dropped = len(entries) - len(unique_entries)
        logger.info(
            f"Cosine similarity (>{EVAL_DEDUP_SIMILARITY_THRESHOLD}) deduplication removed {dupes_dropped} redundant queries."
        )

        # Unified validation
        stats = validate_eval_dataset(unique_entries)
        if stats["invalid"] > 0:
            logger.warning(
                f"Dataset merge yielded {stats['invalid']} schema-invalid entries."
            )

        logger.info(f"Final valid composition: {stats['valid']} entries.")

        # Save output
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_file, "w", encoding="utf-8") as f:
            json.dump(unique_entries, f, indent=2)

        logger.info(f"Saved primary evaluation dataset to {self.output_file.name}")
        logger.info(f"Distribution Statistics:\\n{json.dumps(stats, indent=2)}")

        return unique_entries


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    builder = DatasetBuilder()
    builder.build_dataset()
