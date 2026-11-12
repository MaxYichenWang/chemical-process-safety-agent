"""
Automated Violation Q&A Generator.

Generates 30-45 evaluation Q&A pairs from OSHA violation sample chunks.
Each generated entry follows the unified :class:`~src.evaluation.schemas.EvalEntry` schema.

Data source: ``data/loaded_chunks/osha_violations_sample_chunks.parquet``
(362K chunks with citation text, standard_number, viol_type, penalties, etc.)
"""

from __future__ import annotations

import json
import logging
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
    GENERATOR_MIN_TOKEN_COUNT,
    GENERATOR_SEED,
    KEY_OSHA_STANDARDS,
    REGULATIONS_CHUNKS_PARQUET,
    VIOLATIONS_SAMPLE_CHUNKS_PARQUET,
)
from src.evaluation.schemas import validate_eval_entry

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Question templates
# ---------------------------------------------------------------------------

_TEMPLATES = {
    "violation_factual": (
        "What OSHA violation was cited in citation {citation_id} under "
        "standard {standard_number} at {establishment_name}?"
    ),
    "violation_description": (
        "Describe the OSHA violation issued to {establishment_name} "
        "for non-compliance with standard {standard_number}."
    ),
    "penalty_inquiry": (
        "What penalties were assessed for OSHA violations at "
        "{establishment_name} involving standard {standard_number}?"
    ),
    "violation_pattern": (
        "What types of {viol_type_label} violations has OSHA cited "
        "under standard {standard_number}?"
    ),
}

_VIOL_TYPE_LABELS = {
    "W": "willful",
    "R": "repeat",
    "S": "serious",
    "O": "other-than-serious",
}

# ---------------------------------------------------------------------------
# LLM prompts
# ---------------------------------------------------------------------------

_REFINE_SYSTEM = (
    "You are a chemical process safety expert creating evaluation questions "
    "for an OSHA enforcement Q&A system. You produce precise, natural-language "
    "questions and answers grounded strictly in the provided violation text."
)

_REFINE_TEMPLATE = """\
Given the following OSHA violation citation, refine the draft question into \
a clear, natural-language question and generate an expected answer.

**Standard**: {standard_number}
**Violation type**: {viol_type}
**Establishment**: {establishment_name}
**Location**: {site_city}, {site_state}
**Initial penalty**: ${initial_penalty:,.2f}

**Violation text**:
{text}

**Draft question**: {draft_question}

Respond in valid JSON with exactly these keys:
- "question": the refined natural-language question (string)
- "expected_answer": the expected answer based ONLY on the violation text above (string)
- "key_claims": a JSON array of 2-4 objects, each with "claim" (string) and \
"source_chunk" (string – use the chunk_id provided below, or empty string)

chunk_id to use for source_chunk: {chunk_id}

Return ONLY the JSON object, no markdown fences."""

# Selection parameters
_MIN_TOKEN_COUNT = GENERATOR_MIN_TOKEN_COUNT
_TARGET_FACTUAL = 18
_TARGET_PATTERN = 6
_TARGET_CROSS_CORPUS = 6
_TARGET_PENALTY = 6

_DEFAULT_TARGET_ENTRIES = 36
_DEFAULT_CHECKPOINT_EVERY = GENERATOR_CHECKPOINT_EVERY


class ViolationQAGenerator:
    """Generate violation Q&A evaluation entries from OSHA violations parquet."""

    def __init__(
        self,
        llm_model: str = EVAL_LLM_MODEL,
        temperature: float = EVAL_LLM_TEMPERATURE,
        violations_path: Path = VIOLATIONS_SAMPLE_CHUNKS_PARQUET,
        regulations_path: Path = REGULATIONS_CHUNKS_PARQUET,
        seed: int = GENERATOR_SEED,
        target_entries: int = _DEFAULT_TARGET_ENTRIES,
        checkpoint_path: Path | None = None,
        checkpoint_every: int = _DEFAULT_CHECKPOINT_EVERY,
    ) -> None:
        self._llm_model = llm_model
        self._temperature = temperature
        self._violations_path = violations_path
        self._regulations_path = regulations_path
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

    def _load_violations(self) -> pd.DataFrame:
        """Load violations sample chunks via DuckDB."""
        path = self._violations_path
        if not path.exists():
            raise FileNotFoundError(
                f"Violations parquet not found: {path}. "
                "Ensure OSHA violation data has been loaded first."
            )
        conn = duckdb.connect(":memory:")
        try:
            df = conn.execute(
                f"SELECT * FROM read_parquet('{path}') "
                f"WHERE token_count > {_MIN_TOKEN_COUNT}"
            ).df()
        finally:
            conn.close()
        return df

    def _load_regulations_chunks(self) -> pd.DataFrame:
        """Load regulation chunks for cross-corpus mapping."""
        path = self._regulations_path
        if not path.exists():
            raise FileNotFoundError(
                f"Regulations parquet not found: {path}. "
                "Ensure regulation data has been loaded first."
            )
        conn = duckdb.connect(":memory:")
        try:
            df = conn.execute(
                f"SELECT chunk_id, standard_number, section_id, title "
                f"FROM read_parquet('{path}')"
            ).df()
        finally:
            conn.close()
        return df

    # -- selection strategies -------------------------------------------------

    def _select_factual_violations(self, df: pd.DataFrame) -> pd.DataFrame:
        # Prioritize willful/repeat violations
        priority = df[df["viol_type"].isin(["W", "R"])]
        other = df[~df["viol_type"].isin(["W", "R"])]

        n_priority = min(len(priority), _TARGET_FACTUAL // 2)
        n_other = min(len(other), _TARGET_FACTUAL - n_priority)

        selected_priority = (
            priority.sample(n=n_priority, random_state=self._seed)
            if n_priority > 0
            else priority.iloc[:0]
        )

        selected_other = (
            other.sample(n=n_other, random_state=self._seed + 1)
            if n_other > 0
            else other.iloc[:0]
        )

        selected = pd.concat([selected_priority, selected_other], ignore_index=True)
        logger.info(
            "Selected %d violations for factual questions " "(%d priority, %d other)",
            len(selected),
            n_priority,
            n_other,
        )
        return selected

    def _select_pattern_violations(self, df: pd.DataFrame) -> list[dict[str, Any]]:
        """Group violations for pattern-based questions."""
        groups: list[dict[str, Any]] = []

        # Group by viol_type + standard_number for PSM standards
        for std in KEY_OSHA_STANDARDS[:4]:  # Top 4 standards
            std_df = df[df["standard_number"].str.startswith(std, na=False)]
            for vt in ["W", "R", "S"]:
                vt_df = std_df[std_df["viol_type"] == vt]
                if len(vt_df) >= 2:
                    sample = vt_df.sample(
                        n=min(5, len(vt_df)),
                        random_state=self._seed,
                    )
                    groups.append(
                        {
                            "standard_number": std,
                            "viol_type": vt,
                            "viol_type_label": _VIOL_TYPE_LABELS.get(vt, vt),
                            "chunk_ids": sample["chunk_id"].tolist(),
                            "sample_texts": sample["text"].tolist()[:3],
                        }
                    )
                    if len(groups) >= _TARGET_PATTERN:
                        break
            if len(groups) >= _TARGET_PATTERN:
                break

        logger.info("Created %d violation pattern groups", len(groups))
        return groups

    def _select_high_penalty_violations(self, df: pd.DataFrame) -> pd.DataFrame:
        """Select top-penalty violations."""
        if "initial_penalty" not in df.columns:
            return df.iloc[:0]
        sorted_df = df.sort_values("initial_penalty", ascending=False)
        n = min(len(sorted_df), _TARGET_PENALTY)
        return sorted_df.head(n)

    def _find_regulation_chunks(
        self, standard_number: str, regs_df: pd.DataFrame
    ) -> list[str]:
        """Find regulation chunk_ids matching a standard number."""
        mask = regs_df["standard_number"].str.startswith(standard_number, na=False)
        return regs_df[mask]["chunk_id"].tolist()[:5]

    # -- LLM refinement -------------------------------------------------------

    def _refine_with_llm(
        self, draft_question: str, row: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Refine a draft question via LLM."""
        chunk_id = row.get("chunk_id", "")

        prompt = _REFINE_TEMPLATE.format(
            standard_number=row.get("standard_number", ""),
            viol_type=row.get("viol_type", ""),
            establishment_name=row.get("establishment_name", ""),
            site_city=row.get("site_city", ""),
            site_state=row.get("site_state", ""),
            initial_penalty=float(row.get("initial_penalty", 0) or 0),
            text=str(row.get("text", ""))[:4000],
            draft_question=draft_question,
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
                "Failed to parse LLM response for violation %s",
                row.get("citation_id", "unknown"),
            )
            return None

        return {
            "question": parsed.get("question", draft_question),
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
        """Run the full violation Q&A generation pipeline.

        Returns:
            List of evaluation entry dicts conforming to the unified schema.

        Raises:
            FileNotFoundError: If required parquet files are absent.
        """
        violations_df = self._load_violations()
        regs_df = self._load_regulations_chunks()

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

        logger.info(
            "Violation QA generation started (target: %d entries)",
            self._target_entries,
        )

        def _add_entry(entry: dict[str, Any]) -> bool:
            """Validate and add an entry; return True if added."""
            nonlocal last_checkpoint_size
            query = entry["query"]
            normalized = query.strip().lower()
            if normalized in seen_queries:
                return False
            errors = validate_eval_entry(entry)
            if errors:
                logger.warning("Skipping invalid entry: %s", errors)
                return False
            seen_queries.add(normalized)
            entries.append(entry)
            if len(entries) - last_checkpoint_size >= self._checkpoint_every:
                self._write_checkpoint(entries)
                last_checkpoint_size = len(entries)
            return True

        # --- 1. Factual questions ---
        factual_df = self._select_factual_violations(violations_df)
        for _, row in factual_df.iterrows():
            if len(entries) >= self._target_entries:
                break
            row_dict = row.to_dict()
            draft = _TEMPLATES["violation_factual"].format(
                citation_id=row_dict.get("citation_id", ""),
                standard_number=row_dict.get("standard_number", ""),
                establishment_name=row_dict.get("establishment_name", ""),
            )
            refined = self._refine_with_llm(draft, row_dict)
            if refined is None:
                continue

            key_claims = refined.get("key_claims", [])
            if isinstance(key_claims, list):
                for claim in key_claims:
                    if isinstance(claim, dict) and not claim.get("source_chunk"):
                        claim["source_chunk"] = row_dict.get("chunk_id", "")

            _add_entry(
                {
                    "query": refined["question"],
                    "expected_answer": refined["expected_answer"],
                    "ground_truth_chunks": [row_dict.get("chunk_id", "")],
                    "key_claims": key_claims,
                    "source": "auto_violation",
                    "category": "enforcement_violation",
                    "expected_agents": ["osha_enf"],
                    "complexity": "simple",
                    "metadata": {
                        "citation_id": row_dict.get("citation_id", ""),
                        "standard_number": row_dict.get("standard_number", ""),
                        "viol_type": row_dict.get("viol_type", ""),
                        "establishment_name": row_dict.get("establishment_name", ""),
                        "initial_penalty": float(
                            row_dict.get("initial_penalty", 0) or 0
                        ),
                        "template": "violation_factual",
                    },
                }
            )

        # --- 2. Pattern questions ---
        pattern_groups = self._select_pattern_violations(violations_df)
        for group in pattern_groups:
            if len(entries) >= self._target_entries:
                break
            draft = _TEMPLATES["violation_pattern"].format(
                viol_type_label=group["viol_type_label"],
                standard_number=group["standard_number"],
            )
            # Use first chunk for LLM context
            first_row = violations_df[
                violations_df["chunk_id"] == group["chunk_ids"][0]
            ]
            if first_row.empty:
                continue
            row_dict = first_row.iloc[0].to_dict()

            refined = self._refine_with_llm(draft, row_dict)
            if refined is None:
                continue

            _add_entry(
                {
                    "query": refined["question"],
                    "expected_answer": refined["expected_answer"],
                    "ground_truth_chunks": group["chunk_ids"],
                    "key_claims": refined.get("key_claims", []),
                    "source": "auto_violation",
                    "category": "enforcement_violation",
                    "expected_agents": ["osha_enf"],
                    "complexity": "reasoning",
                    "metadata": {
                        "standard_number": group["standard_number"],
                        "viol_type": group["viol_type"],
                        "template": "violation_pattern",
                    },
                }
            )

        # --- 3. Cross-corpus questions ---
        if len(entries) < self._target_entries:
            psm_violations = violations_df[
                violations_df["standard_number"].str.startswith(
                    tuple(KEY_OSHA_STANDARDS[:4]), na=False
                )
            ]
        else:
            psm_violations = violations_df.iloc[:0]
        if len(psm_violations) > 0:
            sample = psm_violations.sample(
                n=min(_TARGET_CROSS_CORPUS, len(psm_violations)),
                random_state=self._seed + 2,
            )
            for _, row in sample.iterrows():
                if len(entries) >= self._target_entries:
                    break
                row_dict = row.to_dict()
                std = row_dict.get("standard_number", "")
                reg_chunks = self._find_regulation_chunks(std, regs_df)
                if not reg_chunks:
                    continue

                draft = (
                    f"What does OSHA standard {std} require, and what "
                    f"violation was cited at "
                    f"{row_dict.get('establishment_name', 'the facility')}?"
                )
                refined = self._refine_with_llm(draft, row_dict)
                if refined is None:
                    continue

                ground_truth = [row_dict.get("chunk_id", "")] + reg_chunks[:3]
                _add_entry(
                    {
                        "query": refined["question"],
                        "expected_answer": refined["expected_answer"],
                        "ground_truth_chunks": ground_truth,
                        "key_claims": refined.get("key_claims", []),
                        "source": "auto_violation",
                        "category": "cross_corpus",
                        "expected_agents": ["osha_enf", "osha_regs"],
                        "complexity": "multi_context",
                        "metadata": {
                            "citation_id": row_dict.get("citation_id", ""),
                            "standard_number": std,
                            "establishment_name": row_dict.get(
                                "establishment_name", ""
                            ),
                            "template": "cross_corpus_violation_regulation",
                        },
                    }
                )

        # --- 4. High-penalty questions ---
        penalty_df = self._select_high_penalty_violations(violations_df)
        for _, row in penalty_df.iterrows():
            if len(entries) >= self._target_entries:
                break
            row_dict = row.to_dict()
            draft = _TEMPLATES["penalty_inquiry"].format(
                establishment_name=row_dict.get("establishment_name", ""),
                standard_number=row_dict.get("standard_number", ""),
            )
            refined = self._refine_with_llm(draft, row_dict)
            if refined is None:
                continue

            _add_entry(
                {
                    "query": refined["question"],
                    "expected_answer": refined["expected_answer"],
                    "ground_truth_chunks": [row_dict.get("chunk_id", "")],
                    "key_claims": refined.get("key_claims", []),
                    "source": "auto_violation",
                    "category": "enforcement_violation",
                    "expected_agents": ["osha_enf"],
                    "complexity": "simple",
                    "metadata": {
                        "citation_id": row_dict.get("citation_id", ""),
                        "standard_number": row_dict.get("standard_number", ""),
                        "establishment_name": row_dict.get("establishment_name", ""),
                        "initial_penalty": float(
                            row_dict.get("initial_penalty", 0) or 0
                        ),
                        "template": "penalty_inquiry",
                    },
                }
            )

        self._write_checkpoint(entries)
        logger.info("Generated %d violation Q&A entries", len(entries))
        return entries
