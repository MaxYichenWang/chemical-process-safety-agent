"""
Automated Incident Q&A Generator.

Generates 30-50 evaluation Q&A pairs from CSB structured JSON reports
(final reports and recommendations).  Each generated entry follows the
unified :class:`~src.evaluation.schemas.EvalEntry` schema.
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
    LOADED_CHUNKS_DIR,
    PROCESSED_CSB_DIR,
)
from src.evaluation.schemas import validate_eval_entry

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_CSB_JSON_DIR = PROCESSED_CSB_DIR / "json"
_CSB_TOC_DIR = PROCESSED_CSB_DIR / "toc"
_CSB_REPORTS_CHUNKS = LOADED_CHUNKS_DIR / "csb_final_reports_chunks.parquet"
_CSB_RECS_CHUNKS = LOADED_CHUNKS_DIR / "csb_recommendations_chunks.parquet"

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

_DEFAULT_TARGET_ENTRIES = 45
_DEFAULT_CHECKPOINT_EVERY = GENERATOR_CHECKPOINT_EVERY

# ---------------------------------------------------------------------------
# Question templates
# ---------------------------------------------------------------------------

_REPORT_TEMPLATES = [
    "What were the key findings of the CSB investigation into the {incident_name}?",
    "What chemical(s) were involved in the {incident_name}?",
    "What was the immediate cause of the {incident_name}?",
]

_RECOMMENDATION_TEMPLATES = [
    "What safety recommendations did the CSB issue following the {incident_name}?",
    "What was the status of CSB recommendation {recommendation_number}?",
    "Who was the target recipient of CSB recommendation {recommendation_number}?",
]

# Sections of interest in final reports
_REPORT_SECTION_TITLES = {
    "key findings",
    "root cause",
    "incident description",
    "executive summary",
    "key issues",
    "root causes",
    "causal analysis",
    "incident summary",
}

# ---------------------------------------------------------------------------
# LLM prompts
# ---------------------------------------------------------------------------

_REFINE_SYSTEM = (
    "You are a chemical process safety expert creating evaluation questions "
    "for a CSB incident investigation Q&A system. You produce precise, "
    "natural-language questions and answers grounded strictly in the "
    "provided source text."
)

_REFINE_TEMPLATE = """\
Given the following CSB incident data, refine the draft question into a \
clear, natural-language question and generate an expected answer.

**Incident**: {incident_name}
**Document type**: {document_type}

**Source text**:
{source_text}

**Draft question**: {draft_question}

Respond in valid JSON with exactly these keys:
- "question": the refined natural-language question (string)
- "expected_answer": the expected answer based ONLY on the source text (string)
- "key_claims": a JSON array of 2-4 objects, each with "claim" (string) and \
"source_chunk" (string – use the chunk_id provided below, or empty string)

chunk_id to use for source_chunk: {chunk_id}

Return ONLY the JSON object, no markdown fences."""

_NLI_SYSTEM = (
    "You are a strict textual entailment judge. Determine whether the "
    "hypothesis is fully entailed by the premise."
)

_NLI_TEMPLATE = """\
**Premise** (source text):
{source_text}

**Hypothesis** (expected answer):
{expected_answer}

Is the hypothesis fully entailed by the premise? Respond with ONLY one word: \
"entailed" or "contradicted"."""


class IncidentQAGenerator:
    """Generate incident Q&A evaluation entries from CSB structured JSONs."""

    def __init__(
        self,
        llm_model: str = EVAL_LLM_MODEL,
        temperature: float = EVAL_LLM_TEMPERATURE,
        csb_json_dir: Path = _CSB_JSON_DIR,
        csb_toc_dir: Path = _CSB_TOC_DIR,
        reports_chunks_path: Path = _CSB_REPORTS_CHUNKS,
        recs_chunks_path: Path = _CSB_RECS_CHUNKS,
        target_entries: int = _DEFAULT_TARGET_ENTRIES,
        checkpoint_path: Path | None = None,
        checkpoint_every: int = _DEFAULT_CHECKPOINT_EVERY,
    ) -> None:
        self._llm_model = llm_model
        self._temperature = temperature
        self._csb_json_dir = csb_json_dir
        self._csb_toc_dir = csb_toc_dir
        self._reports_chunks_path = reports_chunks_path
        self._recs_chunks_path = recs_chunks_path
        self._target_entries = target_entries
        self._checkpoint_path = checkpoint_path
        self._checkpoint_every = checkpoint_every
        self._llm: ChatOpenAI | None = None

    # -- lazy LLM init -------------------------------------------------------

    def _get_llm(self) -> ChatOpenAI:
        if self._llm is None:
            self._llm = ChatOpenAI(model=self._llm_model, temperature=self._temperature)
        return self._llm

    # -- scanning structured JSONs -------------------------------------------

    def _scan_structured_jsons(self) -> list[dict[str, Any]]:
        """Scan the CSB JSON directory for ``*_structured.json`` files.

        Returns a list of dicts with ``incident_name``, ``report_name``,
        ``json_path``, and ``document_type``.

        Raises:
            FileNotFoundError: If the CSB JSON directory does not exist.
        """
        json_dir = self._csb_json_dir
        if not json_dir.exists():
            raise FileNotFoundError(
                f"CSB JSON directory not found: {json_dir}. "
                "Ensure CSB data has been processed first."
            )

        results: list[dict[str, Any]] = []
        for incident_dir in sorted(json_dir.iterdir()):
            if not incident_dir.is_dir():
                continue
            incident_name = incident_dir.name
            for json_file in sorted(incident_dir.glob("*_structured.json")):
                report_name = json_file.stem.replace("_structured", "")
                report_data = self._load_json(json_file)
                doc_type = self._classify_document(
                    report_data, incident_name, report_name
                )
                results.append(
                    {
                        "incident_name": incident_name,
                        "report_name": report_name,
                        "json_path": json_file,
                        "document_type": doc_type,
                    }
                )

        logger.info("Found %d structured JSON files", len(results))
        return results

    @staticmethod
    def _load_json(path: Path) -> dict[str, Any]:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    # -- document classification ---------------------------------------------

    def _classify_document(
        self,
        report_data: dict[str, Any],
        incident_name: str,
        report_name: str,
    ) -> str:
        """Determine whether the document is a final_report or recommendation.

        Uses ``metadata.document_type`` when available, otherwise falls back
        to checking for a TOC file (matching the csb_loader.py pattern).
        """
        metadata = report_data.get("metadata", {})
        doc_type = metadata.get("document_type", "")
        if doc_type in ("final_report", "recommendation"):
            return doc_type

        toc_path = self._csb_toc_dir / incident_name / f"{report_name}_toc.json"
        return "final_report" if toc_path.exists() else "recommendation"

    # -- data extraction from structured JSON ---------------------------------

    def _extract_report_data(self, report_data: dict[str, Any]) -> list[dict[str, Any]]:
        """Extract relevant sections from a CSB final report.

        Returns a list of dicts with ``section_title`` and ``text``.
        """
        sections: list[dict[str, Any]] = []

        # Try pages-based extraction: scan page text for section titles
        pages = report_data.get("pages", [])
        full_text = "\n\n".join(p.get("text", "") for p in pages)

        # If the JSON has a sections key, use it
        json_sections = report_data.get("sections", [])
        if json_sections:
            for sec in json_sections:
                title = sec.get("title", "").strip()
                if title.lower() in _REPORT_SECTION_TITLES:
                    content = sec.get("content", "").strip()
                    if content:
                        sections.append({"section_title": title, "text": content})

        # If no sections matched, use the full text (truncated)
        if not sections and full_text.strip():
            sections.append(
                {
                    "section_title": "Full Report",
                    "text": full_text[:5000],
                }
            )

        return sections

    def _extract_recommendation_data(
        self, report_data: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Extract recommendation details from a CSB recommendation document.

        Returns a list of dicts with ``section_title``, ``text``, and
        optionally ``recommendation_number``, ``recipient``, ``status``.
        """
        recommendations: list[dict[str, Any]] = []

        json_sections = report_data.get("sections", [])
        for sec in json_sections:
            title = sec.get("title", "").strip()
            content = sec.get("content", "").strip()
            if not content:
                continue

            rec: dict[str, Any] = {
                "section_title": title,
                "text": content,
            }

            title_lower = title.lower()
            if "recommendation" in title_lower and "text" in title_lower:
                rec["recommendation_number"] = title
            if "recipient" in title_lower:
                rec["recipient"] = content
            if "status" in title_lower:
                rec["status"] = content

            recommendations.append(rec)

        # Fallback: use pages
        if not recommendations:
            pages = report_data.get("pages", [])
            full_text = "\n\n".join(p.get("text", "") for p in pages)
            if full_text.strip():
                recommendations.append(
                    {
                        "section_title": "Full Recommendation",
                        "text": full_text[:5000],
                    }
                )

        return recommendations

    # -- question generation --------------------------------------------------

    def _generate_questions(
        self,
        extracted_data: list[dict[str, Any]],
        document_type: str,
        incident_name: str,
    ) -> list[dict[str, Any]]:
        """Apply question templates to extracted data.

        Returns a list of raw question dicts with ``draft_question``,
        ``source_text``, and ``incident_name``.
        """
        templates = (
            _REPORT_TEMPLATES
            if document_type == "final_report"
            else _RECOMMENDATION_TEMPLATES
        )
        display_name = incident_name.replace("_", " ")

        questions: list[dict[str, Any]] = []

        for section_data in extracted_data:
            rec_number = section_data.get("recommendation_number", "N/A")

            for tpl in templates:
                # Skip recommendation-specific templates when no number
                if "{recommendation_number}" in tpl and rec_number == "N/A":
                    continue

                draft = tpl.format(
                    incident_name=display_name,
                    recommendation_number=rec_number,
                )
                questions.append(
                    {
                        "draft_question": draft,
                        "source_text": section_data["text"],
                        "incident_name": incident_name,
                        "document_type": document_type,
                        "section_title": section_data.get("section_title", ""),
                    }
                )

        return questions

    # -- chunk mapping --------------------------------------------------------

    def _load_chunks_df(self, document_type: str) -> pd.DataFrame:
        path = (
            self._reports_chunks_path
            if document_type == "final_report"
            else self._recs_chunks_path
        )
        if not path.exists():
            raise FileNotFoundError(
                f"Chunks parquet not found: {path}. "
                "Ensure data loading has been completed first."
            )
        conn = duckdb.connect(":memory:")
        try:
            df = conn.execute(
                f"SELECT chunk_id, incident_name, report_name "
                f"FROM read_parquet('{path}')"
            ).df()
        finally:
            conn.close()
        return df

    def _map_to_chunks(
        self,
        incident_name: str,
        report_name: str,
        chunks_df: pd.DataFrame,
    ) -> list[str]:
        """Find chunk_ids for a given incident and report."""
        mask = chunks_df["incident_name"] == incident_name
        if "report_name" in chunks_df.columns:
            mask &= chunks_df["report_name"] == report_name
        matches = chunks_df[mask]
        return matches["chunk_id"].tolist()

    # -- LLM refinement -------------------------------------------------------

    def _refine_with_llm(
        self, raw_question: dict[str, Any], chunk_ids: list[str]
    ) -> dict[str, Any] | None:
        """Refine a draft question via LLM into a full evaluation entry."""
        chunk_id = chunk_ids[0] if chunk_ids else ""

        prompt = _REFINE_TEMPLATE.format(
            incident_name=raw_question["incident_name"].replace("_", " "),
            document_type=raw_question["document_type"],
            source_text=raw_question["source_text"][:4000],
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
                "Failed to parse LLM response for %s: %s",
                raw_question["incident_name"],
                raw_question["draft_question"][:80],
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

    # -- NLI filtering --------------------------------------------------------

    def _is_entailed(self, entry: dict[str, Any]) -> bool:
        """Return True if expected answer is entailed by source_text.

        Unparseable verdicts are treated as entailed (kept), matching the
        previous batch-filter behavior.
        """
        source_text = entry.get("metadata", {}).get("source_text", "")
        if not source_text:
            return True

        prompt = _NLI_TEMPLATE.format(
            source_text=source_text[:4000],
            expected_answer=entry["expected_answer"],
        )

        llm = self._get_llm()
        response = llm.invoke(
            [
                SystemMessage(content=_NLI_SYSTEM),
                HumanMessage(content=prompt),
            ]
        )

        verdict = response.content.strip().lower()
        if verdict == "contradicted":
            logger.info("NLI rejected entry: %s", entry["query"][:80])
            return False
        return True

    # -- main generation pipeline ---------------------------------------------

    def generate(self) -> list[dict[str, Any]]:
        """Run the full incident Q&A generation pipeline.

        Returns:
            List of evaluation entry dicts conforming to the unified schema.

        Raises:
            FileNotFoundError: If CSB JSON directory or chunk parquets are absent.
        """
        file_infos = self._scan_structured_jsons()

        # Pre-load chunk dataframes
        try:
            reports_chunks_df = self._load_chunks_df("final_report")
        except FileNotFoundError:
            reports_chunks_df = pd.DataFrame(
                columns=["chunk_id", "incident_name", "report_name"]
            )
            logger.warning(
                "Final reports chunks parquet not found; chunk mapping will be empty"
            )

        try:
            recs_chunks_df = self._load_chunks_df("recommendation")
        except FileNotFoundError:
            recs_chunks_df = pd.DataFrame(
                columns=["chunk_id", "incident_name", "report_name"]
            )
            logger.warning(
                "Recommendations chunks parquet not found; chunk mapping will be empty"
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

        logger.info(
            "Incident QA generation: %d structured JSON files discovered "
            "(target: %d entries)",
            len(file_infos),
            self._target_entries,
        )

        for file_info in file_infos:
            if len(entries) >= self._target_entries:
                logger.info(
                    "Reached target of %d entries; stopping early.",
                    self._target_entries,
                )
                break

            incident_name = file_info["incident_name"]
            report_name = file_info["report_name"]
            doc_type = file_info["document_type"]
            report_data = self._load_json(file_info["json_path"])

            # Extract relevant sections
            if doc_type == "final_report":
                extracted = self._extract_report_data(report_data)
                chunks_df = reports_chunks_df
                expected_agents = ["csb_incident"]
            else:
                extracted = self._extract_recommendation_data(report_data)
                chunks_df = recs_chunks_df
                expected_agents = ["csb_recs"]

            if not extracted:
                continue

            chunk_ids = self._map_to_chunks(incident_name, report_name, chunks_df)

            raw_questions = self._generate_questions(extracted, doc_type, incident_name)

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
                    "source": "auto_incident",
                    "category": "incident_factual",
                    "expected_agents": expected_agents,
                    "complexity": "simple",
                    "metadata": {
                        "incident_name": incident_name,
                        "report_name": report_name,
                        "document_type": doc_type,
                        "section_title": raw_q.get("section_title", ""),
                        "source_text": raw_q["source_text"][:2000],
                    },
                }

                errors = validate_eval_entry(entry)
                if errors:
                    logger.warning(
                        "Skipping invalid entry (incident %s): %s",
                        incident_name,
                        errors,
                    )
                    continue

                if not self._is_entailed(entry):
                    continue

                entries.append(entry)

                if len(entries) - last_checkpoint_size >= self._checkpoint_every:
                    self._write_checkpoint(entries)
                    last_checkpoint_size = len(entries)

        # Final checkpoint with all entries
        self._write_checkpoint(entries)

        logger.info(
            "Generated %d incident Q&A entries covering %d incidents",
            len(entries),
            len({e["metadata"]["incident_name"] for e in entries}),
        )
        return entries
