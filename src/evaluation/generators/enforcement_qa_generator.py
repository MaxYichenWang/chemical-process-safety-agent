"""
Automated Enforcement (Accident) Q&A Generator.

Generates 25-35 evaluation Q&A pairs from OSHA accident investigation
chunks.  Each generated entry follows the unified
:class:`~src.evaluation.schemas.EvalEntry` schema.

Data source: ``data/loaded_chunks/osha_accidents_chunks.parquet``
(165K chunks with accident descriptions, event dates, and severity flags
encoded in the text).  Cross-corpus questions also pull from
``data/loaded_chunks/osha_regulations_chunks.parquet``.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from src.evaluation.config import (
    ACCIDENTS_CHUNKS_PARQUET,
    EVAL_LLM_MODEL,
    EVAL_LLM_TEMPERATURE,
    GENERATOR_CHECKPOINT_EVERY,
    GENERATOR_MIN_TOKEN_COUNT,
    GENERATOR_SEED,
    REGULATIONS_CHUNKS_PARQUET,
)
from src.evaluation.schemas import validate_eval_entry
from src.evaluation.utils import extract_osha_standards

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Question templates
# ---------------------------------------------------------------------------

_TEMPLATES = {
    "accident_factual": (
        "What happened in OSHA accident report {summary_nr} on {event_date}?"
    ),
    "accident_description": (
        "Describe the workplace incident documented in OSHA accident " "{summary_nr}."
    ),
    "accident_severity": (
        "What injuries or fatalities resulted from OSHA accident {summary_nr}?"
    ),
    "accident_pattern": (
        "What types of fatal workplace accidents were reported to OSHA " "in {year}?"
    ),
    "accident_pattern_keyword": (
        "Describe workplace accidents involving {keyword} found in "
        "OSHA enforcement data."
    ),
}

# Keywords used to group accidents into thematic patterns
_ACCIDENT_KEYWORDS = {
    "chemical_exposure": [
        "chemical",
        "toxic",
        "exposure",
        "inhalation",
        "hazardous substance",
        "chemical release",
        "fumes",
    ],
    "explosion_fire": [
        "explosion",
        "fire",
        "ignition",
        "flammable",
        "combustible",
        "flash fire",
        "deflagration",
    ],
    "confined_space": [
        "confined space",
        "atmospheric",
        "oxygen deficient",
        "engulfment",
    ],
    "fall": [
        "fall from",
        "fell from",
        "fall to",
        "fell to",
        "fall through",
        "fell through",
    ],
    "struck_by": [
        "struck by",
        "struck-by",
        "crushed by",
        "caught in",
        "caught-in",
    ],
    "electrocution": [
        "electrocution",
        "electrocuted",
        "electric shock",
        "electrical contact",
    ],
}

# ---------------------------------------------------------------------------
# LLM prompts
# ---------------------------------------------------------------------------

_REFINE_SYSTEM = (
    "You are a chemical process safety expert creating evaluation questions "
    "for an OSHA enforcement Q&A system. You produce precise, natural-language "
    "questions and answers grounded strictly in the provided accident text."
)

_REFINE_TEMPLATE = """\
Given the following OSHA accident report, refine the draft question into \
a clear, natural-language question and generate an expected answer.

**Accident ID (summary_nr)**: {summary_nr}
**Event date**: {event_date}
**Fatality**: {is_fatality}

**Accident text**:
{text}

**Draft question**: {draft_question}

Respond in valid JSON with exactly these keys:
- "question": the refined natural-language question (string)
- "expected_answer": the expected answer based ONLY on the accident text above (string)
- "key_claims": a JSON array of 2-4 objects, each with "claim" (string) and \
"source_chunk" (string – use the chunk_id provided below, or empty string)

chunk_id to use for source_chunk: {chunk_id}

Return ONLY the JSON object, no markdown fences."""

# Selection parameters
_MIN_TOKEN_COUNT = GENERATOR_MIN_TOKEN_COUNT
_TARGET_FACTUAL = 18
_TARGET_PATTERN = 7
_TARGET_CROSS_CORPUS = 7

_DEFAULT_TARGET_ENTRIES = 30
_DEFAULT_CHECKPOINT_EVERY = GENERATOR_CHECKPOINT_EVERY

# Regex to detect "fatality=X" in the structured text prefix
_FATALITY_RE = re.compile(r"fatality=X", re.IGNORECASE)


def _is_fatality_text(text: str) -> bool:
    """Return True if the accident text indicates a fatality."""
    return bool(_FATALITY_RE.search(text))


def _extract_year(event_date: str | None) -> int | None:
    """Parse a four-digit year from an *event_date* string."""
    if not event_date:
        return None
    match = re.search(r"\b(\d{4})\b", str(event_date))
    return int(match.group(1)) if match else None


def _classify_accident(text: str) -> str | None:
    """Classify accident text into a keyword category."""
    text_lower = text.lower()
    for category, keywords in _ACCIDENT_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            return category
    return None


class EnforcementQAGenerator:
    """Generate enforcement (accident) Q&A entries from OSHA accidents parquet."""

    def __init__(
        self,
        llm_model: str = EVAL_LLM_MODEL,
        temperature: float = EVAL_LLM_TEMPERATURE,
        accidents_path: Path = ACCIDENTS_CHUNKS_PARQUET,
        regulations_path: Path = REGULATIONS_CHUNKS_PARQUET,
        seed: int = GENERATOR_SEED,
        target_entries: int = _DEFAULT_TARGET_ENTRIES,
        checkpoint_path: Path | None = None,
        checkpoint_every: int = _DEFAULT_CHECKPOINT_EVERY,
    ) -> None:
        self._llm_model = llm_model
        self._temperature = temperature
        self._accidents_path = accidents_path
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

    def _load_accidents(self) -> pd.DataFrame:
        """Load accident chunks via DuckDB, filtering to substantial text."""
        path = self._accidents_path
        if not path.exists():
            raise FileNotFoundError(
                f"Accidents parquet not found: {path}. "
                "Ensure OSHA accident data has been loaded first."
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

    def _select_factual_accidents(self, df: pd.DataFrame) -> pd.DataFrame:
        """Select diverse accidents for factual questions.

        Prioritises fatality-flagged accidents (``fatality=X`` in text) and
        ensures year diversity by sampling across decades.
        """
        df = df.copy()
        df["_is_fatality"] = df["text"].apply(_is_fatality_text)
        df["_year"] = df["event_date"].apply(_extract_year)

        fatality_df = df[df["_is_fatality"]]
        non_fatality_df = df[~df["_is_fatality"]]

        n_fatality = min(len(fatality_df), _TARGET_FACTUAL * 2 // 3)
        n_other = min(len(non_fatality_df), _TARGET_FACTUAL - n_fatality)

        selected_fatal = (
            fatality_df.sample(n=n_fatality, random_state=self._seed)
            if n_fatality > 0
            else fatality_df.iloc[:0]
        )
        selected_other = (
            non_fatality_df.sample(n=n_other, random_state=self._seed + 1)
            if n_other > 0
            else non_fatality_df.iloc[:0]
        )

        selected = pd.concat([selected_fatal, selected_other], ignore_index=True)
        logger.info(
            "Selected %d accidents for factual questions " "(%d fatality, %d other)",
            len(selected),
            n_fatality,
            n_other,
        )
        return selected

    def _select_pattern_groups(self, df: pd.DataFrame) -> list[dict[str, Any]]:
        """Group accidents for pattern-based questions.

        Groups by keyword category and year range to create analytical
        questions about accident patterns.
        """
        groups: list[dict[str, Any]] = []
        df = df.copy()
        df["_is_fatality"] = df["text"].apply(_is_fatality_text)
        df["_year"] = df["event_date"].apply(_extract_year)

        # Year-based pattern: fatal accidents in a given year
        fatal_df = df[df["_is_fatality"]]
        if not fatal_df.empty:
            year_counts = fatal_df["_year"].dropna().value_counts()
            top_years = year_counts.head(3).index.tolist()
            for year in top_years:
                year_df = fatal_df[fatal_df["_year"] == year]
                if len(year_df) >= 2:
                    sample = year_df.sample(
                        n=min(5, len(year_df)),
                        random_state=self._seed,
                    )
                    groups.append(
                        {
                            "pattern_type": "year",
                            "year": int(year),
                            "keyword": None,
                            "chunk_ids": sample["chunk_id"].tolist(),
                            "sample_texts": sample["text"].tolist()[:3],
                        }
                    )
                    if len(groups) >= _TARGET_PATTERN:
                        break

        # Keyword-based pattern
        for category, keywords in _ACCIDENT_KEYWORDS.items():
            if len(groups) >= _TARGET_PATTERN:
                break
            pattern = "|".join(re.escape(kw) for kw in keywords)
            cat_df = df[df["text"].str.lower().str.contains(pattern, na=False)]
            if len(cat_df) >= 2:
                sample = cat_df.sample(
                    n=min(5, len(cat_df)),
                    random_state=self._seed + hash(category) % 100,
                )
                display = category.replace("_", " ")
                groups.append(
                    {
                        "pattern_type": "keyword",
                        "year": None,
                        "keyword": display,
                        "chunk_ids": sample["chunk_id"].tolist(),
                        "sample_texts": sample["text"].tolist()[:3],
                    }
                )

        logger.info("Created %d accident pattern groups", len(groups))
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
        text = str(row.get("text", ""))

        prompt = _REFINE_TEMPLATE.format(
            summary_nr=row.get("summary_nr", ""),
            event_date=row.get("event_date", ""),
            is_fatality="Yes" if _is_fatality_text(text) else "No",
            text=text[:4000],
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
                "Failed to parse LLM response for accident %s",
                row.get("summary_nr", "unknown"),
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
        """Run the full enforcement (accident) Q&A generation pipeline.

        Returns:
            List of evaluation entry dicts conforming to the unified schema.

        Raises:
            FileNotFoundError: If required parquet files are absent.
        """
        accidents_df = self._load_accidents()
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
            "Enforcement QA generation started (target: %d entries)",
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

        # --- 1. Accident factual questions ---
        factual_df = self._select_factual_accidents(accidents_df)
        template_keys = [
            "accident_factual",
            "accident_description",
            "accident_severity",
        ]
        for idx, (_, row) in enumerate(factual_df.iterrows()):
            if len(entries) >= self._target_entries:
                break
            row_dict = row.to_dict()
            tpl_key = template_keys[idx % len(template_keys)]
            event_date = str(row_dict.get("event_date", ""))
            draft = _TEMPLATES[tpl_key].format(
                summary_nr=row_dict.get("summary_nr", ""),
                event_date=event_date.split(" ")[0] if event_date else "",
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
                    "source": "auto_enforcement",
                    "category": "enforcement_factual",
                    "expected_agents": ["osha_enf"],
                    "complexity": "simple",
                    "metadata": {
                        "summary_nr": row_dict.get("summary_nr", ""),
                        "event_date": event_date,
                        "is_fatality": _is_fatality_text(str(row_dict.get("text", ""))),
                        "accident_category": _classify_accident(
                            str(row_dict.get("text", ""))
                        ),
                        "template": tpl_key,
                    },
                }
            )

        # --- 2. Accident pattern questions ---
        pattern_groups = self._select_pattern_groups(accidents_df)
        for group in pattern_groups:
            if len(entries) >= self._target_entries:
                break

            if group["pattern_type"] == "year":
                draft = _TEMPLATES["accident_pattern"].format(
                    year=group["year"],
                )
            else:
                draft = _TEMPLATES["accident_pattern_keyword"].format(
                    keyword=group["keyword"],
                )

            first_row = accidents_df[accidents_df["chunk_id"] == group["chunk_ids"][0]]
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
                    "source": "auto_enforcement",
                    "category": "enforcement_analytical",
                    "expected_agents": ["osha_enf"],
                    "complexity": "reasoning",
                    "metadata": {
                        "pattern_type": group["pattern_type"],
                        "year": group.get("year"),
                        "keyword": group.get("keyword"),
                        "template": (
                            "accident_pattern"
                            if group["pattern_type"] == "year"
                            else "accident_pattern_keyword"
                        ),
                    },
                }
            )

        # --- 3. Cross-corpus questions (accident + regulation) ---
        candidates = accidents_df.sample(
            n=min(100, len(accidents_df)),
            random_state=self._seed + 3,
        )
        cross_corpus_count = sum(
            1 for e in entries if e.get("complexity") == "multi_context"
        )
        for _, row in candidates.iterrows():
            if len(entries) >= self._target_entries:
                break
            if cross_corpus_count >= _TARGET_CROSS_CORPUS:
                break

            row_dict = row.to_dict()
            text = str(row_dict.get("text", ""))
            standards = extract_osha_standards(text)

            if not standards:
                continue

            std = standards[0].split("(")[0]
            reg_chunks = self._find_regulation_chunks(std, regs_df)
            if not reg_chunks:
                continue

            draft = (
                f"What OSHA regulations are relevant to the type of "
                f"incident described in accident {row_dict.get('summary_nr', '')},"
                f" and what does standard {std} require?"
            )
            refined = self._refine_with_llm(draft, row_dict)
            if refined is None:
                continue

            ground_truth = [row_dict.get("chunk_id", "")] + reg_chunks[:3]
            added = _add_entry(
                {
                    "query": refined["question"],
                    "expected_answer": refined["expected_answer"],
                    "ground_truth_chunks": ground_truth,
                    "key_claims": refined.get("key_claims", []),
                    "source": "auto_enforcement",
                    "category": "cross_corpus",
                    "expected_agents": ["osha_enf", "osha_regs"],
                    "complexity": "multi_context",
                    "metadata": {
                        "summary_nr": row_dict.get("summary_nr", ""),
                        "referenced_standard": std,
                        "template": "cross_corpus_accident_regulation",
                    },
                }
            )
            if added:
                cross_corpus_count += 1

        self._write_checkpoint(entries)
        logger.info("Generated %d enforcement Q&A entries", len(entries))
        return entries
