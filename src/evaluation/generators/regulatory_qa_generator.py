"""
Automated Regulatory Q&A Generator.

Generates 40-60 evaluation Q&A pairs from OSHA regulation sections stored
in ``regulations_sections.parquet``.  Each generated entry follows the
unified :class:`~src.evaluation.schemas.EvalEntry` schema.
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from src.evaluation.config import (
    EVAL_LLM_MODEL,
    EVAL_LLM_TEMPERATURE,
    GENERATOR_CHECKPOINT_EVERY,
    GENERATOR_SEED,
    KEY_OSHA_STANDARDS,
    LOADED_CHUNKS_DIR,
    PROCESSED_OSHA_DIR,
)
from src.evaluation.schemas import validate_eval_entry

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_REGULATIONS_PARQUET = (
    PROCESSED_OSHA_DIR / "regulations" / "regulations_sections.parquet"
)
_CHUNKS_PARQUET = LOADED_CHUNKS_DIR / "osha_regulations_chunks.parquet"

# ---------------------------------------------------------------------------
# Question templates
# ---------------------------------------------------------------------------

_TEMPLATES = {
    "requirement_lookup": (
        "What does OSHA standard {standard_number} require regarding {title}?"
    ),
    "obligation_identification": (
        "What are the employer obligations under {standard_number} "
        "Section {section_id}?"
    ),
    "definition": (
        "How does OSHA {standard_number} define the requirements for "
        "{extracted_topic}?"
    ),
}

# ---------------------------------------------------------------------------
# LLM prompts
# ---------------------------------------------------------------------------

_REFINE_SYSTEM = (
    "You are a chemical process safety expert creating evaluation questions "
    "for an OSHA regulatory Q&A system. You produce precise, natural-language "
    "questions and answers grounded strictly in the provided regulation text."
)

_REFINE_TEMPLATE = """\
Given the following OSHA regulation section, refine the draft question into \
a clear, natural-language question and generate an expected answer.

**Standard**: {standard_number}
**Section**: {section_id}
**Title**: {title}

**Regulation text**:
{text}

**Draft question**: {draft_question}

Respond in valid JSON with exactly these keys:
- "question": the refined natural-language question (string)
- "expected_answer": the expected answer based ONLY on the regulation text above (string)
- "key_claims": a JSON array of 2-4 objects, each with "claim" (string) and \
"source_chunk" (string – the chunk_id provided below, or empty string if unknown)

chunk_id to use for source_chunk: {chunk_id}

Return ONLY the JSON object, no markdown fences."""

# Number of additional random sections to sample from non-key standards
_RANDOM_SAMPLE_MIN = 10
_RANDOM_SAMPLE_MAX = 15


_DEFAULT_TARGET_ENTRIES = 55
_DEFAULT_CHECKPOINT_EVERY = GENERATOR_CHECKPOINT_EVERY


class RegulatoryQAGenerator:
    """Generate regulatory Q&A evaluation entries from OSHA parquet data."""

    def __init__(
        self,
        llm_model: str = EVAL_LLM_MODEL,
        temperature: float = EVAL_LLM_TEMPERATURE,
        regulations_path: Path = _REGULATIONS_PARQUET,
        chunks_path: Path = _CHUNKS_PARQUET,
        seed: int = GENERATOR_SEED,
        target_entries: int = _DEFAULT_TARGET_ENTRIES,
        checkpoint_path: Path | None = None,
        checkpoint_every: int = _DEFAULT_CHECKPOINT_EVERY,
    ) -> None:
        self._llm_model = llm_model
        self._temperature = temperature
        self._regulations_path = regulations_path
        self._chunks_path = chunks_path
        self._seed = seed
        self._target_entries = target_entries
        self._checkpoint_path = checkpoint_path
        self._checkpoint_every = checkpoint_every
        self._llm: ChatOpenAI | None = None

    # -- lazy LLM init -------------------------------------------------------

    def _get_llm(self) -> ChatOpenAI:
        if self._llm is None:
            self._llm = ChatOpenAI(model=self._llm_model, temperature=self._temperature)
        return self._llm

    # -- data loading ---------------------------------------------------------

    def _load_regulations(self) -> pd.DataFrame:
        """Load ``regulations_sections.parquet`` via DuckDB."""
        path = self._regulations_path
        if not path.exists():
            raise FileNotFoundError(
                f"Regulations parquet not found: {path}. "
                "Ensure OSHA data has been processed first."
            )
        conn = duckdb.connect(":memory:")
        try:
            df = conn.execute(f"SELECT * FROM read_parquet('{path}')").df()
        finally:
            conn.close()
        return df

    def _load_chunk_mapping(self) -> pd.DataFrame:
        """Load ``osha_regulations_chunks.parquet`` for chunk_id mapping."""
        path = self._chunks_path
        if not path.exists():
            raise FileNotFoundError(
                f"Chunks parquet not found: {path}. "
                "Ensure data loading has been completed first."
            )
        conn = duckdb.connect(":memory:")
        try:
            df = conn.execute(
                f"SELECT chunk_id, section_id, standard_number "
                f"FROM read_parquet('{path}')"
            ).df()
        finally:
            conn.close()
        return df

    # -- section selection ----------------------------------------------------

    def _select_sections(self, regulations_df: pd.DataFrame) -> pd.DataFrame:
        """Select key-standard sections plus a random sample of others.

        Filters to sections whose ``standard_number`` starts with any
        value in :data:`KEY_OSHA_STANDARDS`, then appends 10-15 randomly
        sampled sections from other standards.  Only sections with
        non-empty text are included.
        """
        rng = random.Random(self._seed)

        df = regulations_df[regulations_df["text"].str.strip().str.len() > 0].copy()

        mask = pd.Series(False, index=df.index)
        for std in KEY_OSHA_STANDARDS:
            mask |= df["standard_number"].str.startswith(std)

        key_sections = df[mask]
        other_sections = df[~mask]

        n_random = rng.randint(_RANDOM_SAMPLE_MIN, _RANDOM_SAMPLE_MAX)
        n_random = min(n_random, len(other_sections))
        if n_random > 0:
            random_sample = other_sections.sample(n=n_random, random_state=self._seed)
        else:
            random_sample = other_sections.iloc[:0]

        selected = pd.concat([key_sections, random_sample], ignore_index=True)
        logger.info(
            "Selected %d sections (%d key-standard, %d random)",
            len(selected),
            len(key_sections),
            len(random_sample),
        )
        return selected

    # -- question templates ---------------------------------------------------

    def _apply_question_templates(
        self, section: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Apply the three question templates to a single regulation section.

        Returns a list of raw question dicts (one per template).
        """
        title = section.get("title", "").strip()
        extracted_topic = title if title else section.get("section_id", "")

        questions: list[dict[str, Any]] = []
        for tpl_name, tpl in _TEMPLATES.items():
            draft = tpl.format(
                standard_number=section.get("standard_number", ""),
                section_id=section.get("section_id", ""),
                title=title,
                extracted_topic=extracted_topic,
            )
            questions.append(
                {
                    "template": tpl_name,
                    "draft_question": draft,
                    "section": section,
                }
            )
        return questions

    # -- chunk mapping --------------------------------------------------------

    def _map_to_chunks(self, section_id: str, chunks_df: pd.DataFrame) -> list[str]:
        """Find chunk_ids that match a given ``section_id``."""
        matches = chunks_df[chunks_df["section_id"] == section_id]
        return matches["chunk_id"].tolist()

    # -- LLM refinement -------------------------------------------------------

    def _refine_with_llm(
        self, raw_question: dict[str, Any], chunk_ids: list[str]
    ) -> dict[str, Any] | None:
        """Use the LLM to refine a draft question into a full eval entry.

        Returns a dict with ``question``, ``expected_answer``, and
        ``key_claims``, or ``None`` if the LLM response cannot be parsed.
        """
        section = raw_question["section"]
        chunk_id = chunk_ids[0] if chunk_ids else ""

        prompt = _REFINE_TEMPLATE.format(
            standard_number=section.get("standard_number", ""),
            section_id=section.get("section_id", ""),
            title=section.get("title", ""),
            text=section.get("text", ""),
            draft_question=raw_question["draft_question"],
            chunk_id=chunk_id,
        )

        llm = self._get_llm()
        response = llm.invoke(
            [
                SystemMessage(content=_REFINE_SYSTEM),
                HumanMessage(content=prompt),
            ]
        )

        try:
            parsed = json.loads(response.content)
        except (json.JSONDecodeError, TypeError):
            logger.warning(
                "Failed to parse LLM response for section %s, template %s",
                section.get("section_id"),
                raw_question["template"],
            )
            return None

        return {
            "question": parsed.get("question", raw_question["draft_question"]),
            "expected_answer": parsed.get("expected_answer", ""),
            "key_claims": parsed.get("key_claims", []),
        }

    # -- checkpointing --------------------------------------------------------

    def _load_existing_entries(self) -> list[dict[str, Any]]:
        """Load existing entries from checkpoint_path, if present."""
        if self._checkpoint_path is None or not self._checkpoint_path.exists():
            return []
        try:
            with open(self._checkpoint_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning(
                "Could not read existing checkpoint %s (%s); starting fresh.",
                self._checkpoint_path,
                exc,
            )
            return []
        if not isinstance(data, list):
            logger.warning(
                "Checkpoint %s is not a JSON list; starting fresh.",
                self._checkpoint_path,
            )
            return []
        return [entry for entry in data if isinstance(entry, dict)]

    def _write_checkpoint(self, entries: list[dict[str, Any]]) -> None:
        """Write *entries* to the checkpoint file, if configured."""
        if self._checkpoint_path is None:
            return
        self._checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._checkpoint_path, "w", encoding="utf-8") as fh:
            json.dump(entries, fh, indent=2)
        logger.info(
            "Checkpoint: wrote %d entries to %s",
            len(entries),
            self._checkpoint_path,
        )

    # -- main generation pipeline ---------------------------------------------

    def generate(self) -> list[dict[str, Any]]:
        """Run the full regulatory Q&A generation pipeline.

        Generation stops as soon as ``target_entries`` valid entries have
        been collected (default 55).  If ``checkpoint_path`` was provided
        at construction time, intermediate results are flushed to disk
        every ``checkpoint_every`` new entries.

        Returns:
            List of evaluation entry dicts conforming to the unified schema.

        Raises:
            FileNotFoundError: If required parquet files are absent.
        """
        regulations_df = self._load_regulations()
        chunks_df = self._load_chunk_mapping()
        selected = self._select_sections(regulations_df)

        logger.info(
            "Regulatory QA generation: %d selected sections, "
            "up to %d templates each -> max ~%d LLM calls "
            "(target: %d entries)",
            len(selected),
            len(_TEMPLATES),
            len(selected) * len(_TEMPLATES),
            self._target_entries,
        )

        entries = self._load_existing_entries()
        seen_queries: set[str] = {
            str(entry.get("query", "")).strip().lower()
            for entry in entries
            if entry.get("query")
        }
        last_checkpoint_size = len(entries)
        if entries:
            logger.info(
                "Loaded %d existing entries from %s; new entries will be appended.",
                len(entries),
                self._checkpoint_path,
            )

        for _, row in selected.iterrows():
            if len(entries) >= self._target_entries:
                logger.info(
                    "Reached target of %d entries; stopping early.",
                    self._target_entries,
                )
                break

            section = row.to_dict()
            section_id = section.get("section_id", "")
            chunk_ids = self._map_to_chunks(section_id, chunks_df)

            raw_questions = self._apply_question_templates(section)

            for raw_q in raw_questions:
                if len(entries) >= self._target_entries:
                    break

                refined = self._refine_with_llm(raw_q, chunk_ids)
                if refined is None:
                    continue

                query = refined["question"]
                normalized = query.strip().lower()
                if normalized in seen_queries:
                    continue
                seen_queries.add(normalized)

                key_claims = refined.get("key_claims", [])
                if isinstance(key_claims, list):
                    for claim in key_claims:
                        if isinstance(claim, dict) and not claim.get("source_chunk"):
                            claim["source_chunk"] = chunk_ids[0] if chunk_ids else ""

                entry: dict[str, Any] = {
                    "query": query,
                    "expected_answer": refined["expected_answer"],
                    "ground_truth_chunks": chunk_ids,
                    "key_claims": key_claims,
                    "source": "auto_regulatory",
                    "category": "regulatory_factual",
                    "expected_agents": ["osha_regs"],
                    "complexity": "simple",
                    "metadata": {
                        "standard_number": section.get("standard_number", ""),
                        "section_id": section_id,
                        "title": section.get("title", ""),
                        "template": raw_q["template"],
                    },
                }

                errors = validate_eval_entry(entry)
                if errors:
                    logger.warning(
                        "Skipping invalid entry (section %s): %s",
                        section_id,
                        errors,
                    )
                    continue

                entries.append(entry)

                if len(entries) - last_checkpoint_size >= self._checkpoint_every:
                    self._write_checkpoint(entries)
                    last_checkpoint_size = len(entries)

        # Final checkpoint with all entries
        self._write_checkpoint(entries)

        logger.info("Generated %d regulatory Q&A entries", len(entries))
        return entries
