"""
Automated Synthetic Expansion Q&A Generator.

Bypasses massive async embedding loads and rate limits natively
by deterministically creating highly stable synthetic reasoning queries.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import duckdb
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from src.evaluation.config import (
    EVAL_DATASET_DIR,
    EVAL_LLM_MODEL,
    EVAL_LLM_TEMPERATURE,
    REGULATIONS_CHUNKS_PARQUET,
)
from src.evaluation.schemas import validate_eval_entry

logger = logging.getLogger(__name__)


class SyntheticGenerator:
    """Generates synthetic evaluations using native Langchain routines."""

    def __init__(
        self,
        output_file: Path | None = None,
        target_size: int = 12,
    ) -> None:
        self.output_file = (
            output_file or EVAL_DATASET_DIR / "eval_dataset_synthetic.json"
        )
        self.target_size = target_size
        self.llm = ChatOpenAI(
            model=EVAL_LLM_MODEL,
            temperature=EVAL_LLM_TEMPERATURE,
            model_kwargs={"response_format": {"type": "json_object"}},
            max_retries=5,
        )

    def generate(self) -> list[dict[str, Any]]:
        conn = duckdb.connect(":memory:")
        try:
            # Extract robust target chunks dynamically
            rows = conn.execute(
                f"SELECT chunk_id, text FROM '{REGULATIONS_CHUNKS_PARQUET}' "
                f"WHERE token_count > 100 LIMIT {self.target_size}"
            ).fetchall()
        finally:
            conn.close()

        entries: list[dict[str, Any]] = []
        for chunk_id, text in rows:
            prompt = (
                f"Based EXCLUSIVELY on this text, generate ONE complex scenario-based reasoning "
                f"evaluation question and its exact factual answer.\n\nText:\n{text[:2000]}\n\n"
                f"Return valid JSON strictly with keys 'query' and 'expected_answer'."
            )
            try:
                res = self.llm.invoke(
                    [
                        SystemMessage(
                            content="You generate strict JSON chemical regulation evaluation questions."
                        ),
                        HumanMessage(content=prompt),
                    ]
                )

                content = res.content.strip()
                if content.startswith("```json"):
                    content = content[7:]
                if content.startswith("```"):
                    content = content[3:]
                if content.endswith("```"):
                    content = content[:-3]

                parsed = json.loads(content.strip())
                entry = {
                    "query": parsed.get("query", ""),
                    "expected_answer": parsed.get("expected_answer", ""),
                    "ground_truth_chunks": [chunk_id],
                    "source": "synthetic",
                    "category": "synthetic_reasoning",
                    "complexity": "reasoning",
                    "expected_agents": ["osha_regs"],
                    "key_claims": [],
                }

                errs = validate_eval_entry(entry)
                if not errs:
                    entries.append(entry)
                else:
                    logger.debug(f"Validation failed: {errs}")
            except Exception as e:
                logger.debug(f"Bypassing failed parse: {e}")

        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_file, "w", encoding="utf-8") as f:
            json.dump(entries, f, indent=2)

        logger.info(f"Generated {len(entries)} robust synthetic reasoning entries.")
        return entries


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    gen = SyntheticGenerator()
    gen.generate()
