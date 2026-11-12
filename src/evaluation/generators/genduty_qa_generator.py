"""
Automated General Duty Q&A Generator.

Generates 20-30 evaluation Q&A pairs from OSHA General Duty Clause
(Section 5(a)(1)) violation chunks.  Each generated entry follows the
unified :class:`~src.evaluation.schemas.EvalEntry` schema.

Data source: ``data/loaded_chunks/osha_genduty_sample_chunks.parquet``
(138K chunks with detailed hazard descriptions and abatement methods.)
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
    GENDUTY_SAMPLE_CHUNKS_PARQUET,
    GENERATOR_CHECKPOINT_EVERY,
    GENERATOR_MIN_TOKEN_COUNT,
    GENERATOR_SEED,
    REGULATIONS_CHUNKS_PARQUET,
)
from src.evaluation.schemas import validate_eval_entry
from src.evaluation.utils import extract_osha_standards

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hazard keyword groups for pattern-based selection
# ---------------------------------------------------------------------------

_HAZARD_KEYWORDS = {
    "fire_explosion": [
        "fire",
        "explosion",
        "flammable",
        "combustible",
        "ignition",
        "deflagration",
        "detonation",
    ],
    "chemical_exposure": [
        "chemical exposure",
        "toxic",
        "inhalation",
        "exposure to",
        "hazardous substance",
        "chemical release",
    ],
    "confined_space": [
        "confined space",
        "permit-required",
        "atmospheric hazard",
    ],
    "fall_hazard": [
        "fall",
        "elevated",
        "unguarded",
        "guardrail",
    ],
    "mechanical": [
        "machine guarding",
        "caught-in",
        "struck-by",
        "amputation",
        "rotating",
        "pinch point",
    ],
    "process_safety": [
        "process safety",
        "process hazard",
        "psm",
        "management of change",
        "operating procedure",
        "mechanical integrity",
    ],
}

# ---------------------------------------------------------------------------
# Question templates
# ---------------------------------------------------------------------------

_TEMPLATES = {
    "genduty_factual": (
        "What recognized hazard was cited in the General Duty Clause "
        "violation at {establishment_name}?"
    ),
    "genduty_abatement": (
        "What abatement methods were recommended for the General Duty "
        "Clause violation at {establishment_name}?"
    ),
    "hazard_pattern": (
        "What {hazard_category} hazards have been cited under OSHA's "
        "General Duty Clause?"
    ),
    "cross_corpus": (
        "For the General Duty violation at {establishment_name}, "
        "what PSM requirements under {standard_number} are relevant?"
    ),
}

# ---------------------------------------------------------------------------
# LLM prompts
# ---------------------------------------------------------------------------

_REFINE_SYSTEM = (
    "You are a chemical process safety expert creating evaluation questions "
    "for an OSHA enforcement Q&A system. You produce precise, natural-language "
    "questions and answers grounded strictly in the provided General Duty "
    "Clause violation text."
)

_REFINE_TEMPLATE = """\
Given the following OSHA General Duty Clause (Section 5(a)(1)) violation, \
refine the draft question into a clear, natural-language question and \
generate an expected answer.

**Establishment**: {establishment_name}
**Location**: {site_city}, {site_state}
**Industry (NAICS)**: {naics_code}

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
_TARGET_FACTUAL = 13
_TARGET_PATTERN = 7
_TARGET_CROSS_CORPUS = 6

_DEFAULT_TARGET_ENTRIES = 26
_DEFAULT_CHECKPOINT_EVERY = GENERATOR_CHECKPOINT_EVERY


class GendutyQAGenerator:
    """Generate General Duty Q&A evaluation entries from OSHA genduty parquet."""

    def __init__(
        self,
        llm_model: str = EVAL_LLM_MODEL,
        temperature: float = EVAL_LLM_TEMPERATURE,
        genduty_path: Path = GENDUTY_SAMPLE_CHUNKS_PARQUET,
        regulations_path: Path = REGULATIONS_CHUNKS_PARQUET,
        seed: int = GENERATOR_SEED,
        target_entries: int = _DEFAULT_TARGET_ENTRIES,
        checkpoint_path: Path | None = None,
        checkpoint_every: int = _DEFAULT_CHECKPOINT_EVERY,
    ) -> None:
        self._llm_model = llm_model
        self._temperature = temperature
        self._genduty_path = genduty_path
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

    def _load_genduty(self) -> pd.DataFrame:
        """Load general duty sample chunks via DuckDB."""
        path = self._genduty_path
        if not path.exists():
            raise FileNotFoundError(
                f"General duty parquet not found: {path}. "
                "Ensure OSHA general duty data has been loaded first."
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

    def _classify_hazard(self, text: str) -> str | None:
        """Classify a violation text into a hazard category."""
        text_lower = text.lower()
        for category, keywords in _HAZARD_KEYWORDS.items():
            if any(kw in text_lower for kw in keywords):
                return category
        return None

    def _select_factual_chunks(self, df: pd.DataFrame) -> pd.DataFrame:
        """Select diverse genduty chunks for factual questions."""
        # Prioritize process-safety related violations
        psm_mask = (
            df["text"]
            .str.lower()
            .str.contains("|".join(_HAZARD_KEYWORDS["process_safety"]), na=False)
        )
        psm_chunks = df[psm_mask]
        other_chunks = df[~psm_mask]

        n_psm = min(len(psm_chunks), _TARGET_FACTUAL // 2)
        n_other = min(len(other_chunks), _TARGET_FACTUAL - n_psm)

        selected_psm = (
            psm_chunks.sample(n=n_psm, random_state=self._seed)
            if n_psm > 0
            else psm_chunks.iloc[:0]
        )

        selected_other = (
            other_chunks.sample(n=n_other, random_state=self._seed + 1)
            if n_other > 0
            else other_chunks.iloc[:0]
        )

        selected = pd.concat([selected_psm, selected_other], ignore_index=True)
        logger.info(
            "Selected %d genduty chunks for factual questions "
            "(%d PSM-related, %d other)",
            len(selected),
            n_psm,
            n_other,
        )
        return selected

    def _select_pattern_groups(self, df: pd.DataFrame) -> list[dict[str, Any]]:
        """Group violations by hazard category for pattern questions."""
        groups: list[dict[str, Any]] = []
        df = df.copy()
        df["hazard_category"] = df["text"].apply(self._classify_hazard)
        categorized = df.dropna(subset=["hazard_category"])

        for category in _HAZARD_KEYWORDS:
            cat_df = categorized[categorized["hazard_category"] == category]
            if len(cat_df) >= 2:
                sample = cat_df.sample(
                    n=min(5, len(cat_df)),
                    random_state=self._seed,
                )
                display_category = category.replace("_", " ")
                groups.append(
                    {
                        "hazard_category": display_category,
                        "chunk_ids": sample["chunk_id"].tolist(),
                        "sample_texts": sample["text"].tolist()[:3],
                    }
                )
                if len(groups) >= _TARGET_PATTERN:
                    break

        logger.info("Created %d hazard pattern groups", len(groups))
        return groups

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
            establishment_name=row.get("establishment_name", ""),
            site_city=row.get("site_city", ""),
            site_state=row.get("site_state", ""),
            naics_code=row.get("naics_code", ""),
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
                "Failed to parse LLM response for genduty chunk %s",
                row.get("chunk_id", "unknown"),
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
        """Run the full general duty Q&A generation pipeline.

        Returns:
            List of evaluation entry dicts conforming to the unified schema.

        Raises:
            FileNotFoundError: If required parquet files are absent.
        """
        genduty_df = self._load_genduty()
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
            "General Duty QA generation started (target: %d entries)",
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
        factual_df = self._select_factual_chunks(genduty_df)
        for _, row in factual_df.iterrows():
            if len(entries) >= self._target_entries:
                break
            row_dict = row.to_dict()

            # Alternate between factual and abatement templates
            template_key = (
                "genduty_factual" if len(entries) % 2 == 0 else "genduty_abatement"
            )
            draft = _TEMPLATES[template_key].format(
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
                    "source": "auto_genduty",
                    "category": "enforcement_genduty",
                    "expected_agents": ["osha_enf"],
                    "complexity": "simple",
                    "metadata": {
                        "establishment_name": row_dict.get("establishment_name", ""),
                        "site_state": row_dict.get("site_state", ""),
                        "naics_code": row_dict.get("naics_code", ""),
                        "hazard_category": self._classify_hazard(
                            str(row_dict.get("text", ""))
                        ),
                        "template": template_key,
                    },
                }
            )

        # --- 2. Hazard pattern questions ---
        pattern_groups = self._select_pattern_groups(genduty_df)
        for group in pattern_groups:
            if len(entries) >= self._target_entries:
                break
            draft = _TEMPLATES["hazard_pattern"].format(
                hazard_category=group["hazard_category"],
            )
            # Use first chunk for LLM context
            first_row = genduty_df[genduty_df["chunk_id"] == group["chunk_ids"][0]]
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
                    "source": "auto_genduty",
                    "category": "enforcement_genduty",
                    "expected_agents": ["osha_enf"],
                    "complexity": "reasoning",
                    "metadata": {
                        "hazard_category": group["hazard_category"],
                        "template": "hazard_pattern",
                    },
                }
            )

        # --- 3. Cross-corpus questions ---
        # Find genduty chunks that mention OSHA standards
        for _, row in genduty_df.sample(
            n=min(50, len(genduty_df)),
            random_state=self._seed + 3,
        ).iterrows():
            if len(entries) >= self._target_entries:
                break
            if (
                len([e for e in entries if e.get("complexity") == "multi_context"])
                >= _TARGET_CROSS_CORPUS
            ):
                break

            row_dict = row.to_dict()
            text = str(row_dict.get("text", ""))
            standards = extract_osha_standards(text)

            if not standards:
                continue

            # Use the first referenced standard
            std = standards[0].split("(")[0]  # base standard without subsection
            reg_chunks = self._find_regulation_chunks(std, regs_df)
            if not reg_chunks:
                continue

            draft = _TEMPLATES["cross_corpus"].format(
                establishment_name=row_dict.get("establishment_name", ""),
                standard_number=std,
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
                    "source": "auto_genduty",
                    "category": "cross_corpus",
                    "expected_agents": ["osha_enf", "osha_regs"],
                    "complexity": "multi_context",
                    "metadata": {
                        "establishment_name": row_dict.get("establishment_name", ""),
                        "referenced_standard": std,
                        "template": "cross_corpus_genduty_regulation",
                    },
                }
            )

        self._write_checkpoint(entries)
        logger.info("Generated %d general duty Q&A entries", len(entries))
        return entries
