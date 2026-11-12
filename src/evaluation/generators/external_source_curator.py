"""
External Source Curator.

Curates Q&A from external sources, specifically OSHA PSM training tests,
CSB cross-corpus incident deep-dives, and enforcement analytical questions.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import duckdb
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from src.evaluation.config import (
    ACCIDENTS_CHUNKS_PARQUET,
    CSB_REPORTS_CHUNKS_PARQUET,
    EVAL_DATASET_DIR,
    EVAL_LLM_MODEL,
    EVAL_LLM_TEMPERATURE,
    REGULATIONS_CHUNKS_PARQUET,
)
from src.evaluation.schemas import validate_eval_entry

logger = logging.getLogger(__name__)

_NLI_SYSTEM = (
    "You are a strict textual entailment judge. Determine whether the "
    "hypothesis is fully entailed by the premise."
)

_NLI_TEMPLATE = """\
**Premise** (source text):
{source_text}

**Hypothesis** (expected answer):
{expected_answer}

Is the hypothesis fully entailed by the premise? Respond with ONLY one word: "entailed" or "contradicted"."""

_CSB_CROSS_CORPUS_INCIDENTS = [
    "West_Fertilizer",
    "BP_Texas_City",
    "Arkema",
    "Chevron_Richmond",
    "Williams_Olefins",
    "Husky_Energy",
    "Kuraray",
    "AB_Specialty_Silicones",
]

_CSB_GEN_SYSTEM = (
    "You are a chemical safety expert tasked with generating evaluation "
    "QA pairs that cross-reference CSB incident findings with OSHA regulations."
)

_CSB_GEN_TEMPLATE = """\
Based on the following excerpts from a CSB report and OSHA regulations, \
generate ONE complex evaluation question that requires synthesizing both sources.

**CSB Incident Text**:
{csb_text}

**OSHA Regulation Text**:
{osha_text}

Respond in valid JSON with exactly these keys:
- "question": the complex cross-corpus question (string)
- "expected_answer": the expected answer synthesizing both sources (string)
- "key_claims": a JSON array of 2-4 objects, each with "claim" (string) and "source_chunk" (string)

Available CSB chunk_id: {csb_chunk_id}
Available OSHA chunk_id: {osha_chunk_id}

Return ONLY the JSON object, no markdown fences.
"""


class ExternalSourceCurator:
    """Curate evaluation sets from authoritative external sources and analytics."""

    def __init__(
        self,
        llm_model: str = EVAL_LLM_MODEL,
        temperature: float = EVAL_LLM_TEMPERATURE,
    ) -> None:
        self._llm = ChatOpenAI(model=llm_model, temperature=temperature)
        self.psm_test_file = (
            EVAL_DATASET_DIR / "external_sources" / "psm_training_test.json"
        )

    def _is_entailed(self, source_text: str, expected_answer: str) -> bool:
        """Use LLM strictly for NLI checking to ensure no hallucinations."""
        if not source_text:
            return True
        prompt = _NLI_TEMPLATE.format(
            source_text=source_text[:4000],
            expected_answer=expected_answer,
        )
        response = self._llm.invoke(
            [
                SystemMessage(content=_NLI_SYSTEM),
                HumanMessage(content=prompt),
            ]
        )
        verdict = response.content.strip().lower()
        return "contradicted" not in verdict

    def _generate_psm_training_qs(self) -> list[dict[str, Any]]:
        """Generate eval entries tightly grounded to the manually curated PSM JSON."""
        if not self.psm_test_file.exists():
            logger.warning(f"PSM test file not found: {self.psm_test_file}")
            return []

        with open(self.psm_test_file, "r") as f:
            raw_qs = json.load(f)

        entries: list[dict[str, Any]] = []
        conn = duckdb.connect(":memory:")
        try:
            for q in raw_qs:
                ref = q.get("reference", "")
                if not ref:
                    continue
                # Mapping exactly to our OSHA regulations parquet via section_id
                res = conn.execute(
                    f"SELECT chunk_id, text FROM '{REGULATIONS_CHUNKS_PARQUET}' "
                    f"WHERE section_id = ?",
                    [ref],
                ).fetchone()

                if not res:
                    logger.warning(
                        f"Could not find regulation chunk for section: {ref}"
                    )
                    continue

                chunk_id, text = res
                expected_ans = str(q.get("correct_answer", ""))

                # Check entailment to confirm dataset ground truth validity
                if not self._is_entailed(text, expected_ans):
                    logger.warning(
                        f"NLI rejected PSM test question for reference {ref} - chunk might lack full context."
                    )
                    continue

                entry: dict[str, Any] = {
                    "query": q["question"],
                    "expected_answer": expected_ans,
                    "ground_truth_chunks": [chunk_id],
                    "source": "external_osha",
                    "category": "regulatory_factual",
                    "expected_agents": ["osha_regs"],
                    "complexity": "simple",
                    "metadata": {"reference": ref, "question_type": q.get("type", "")},
                }

                # Strictly validate before appending
                errors = validate_eval_entry(entry)
                if not errors:
                    entries.append(entry)
                else:
                    logger.warning(f"Invalid generated PSM entry: {errors}")
        finally:
            conn.close()

        logger.info(f"Generated {len(entries)} PSM training Qs")
        return entries

    def _generate_csb_cross_corpus_qs(self) -> list[dict[str, Any]]:
        """Synthesize multi-hop queries pulling from both CSB and OSHA domains."""
        entries: list[dict[str, Any]] = []
        conn = duckdb.connect(":memory:")

        # We cycle over different PSM subsections for cross-linking
        psm_subsections = ["c", "d", "e", "f", "h", "i", "j", "l"]

        try:
            for idx, incident in enumerate(_CSB_CROSS_CORPUS_INCIDENTS):
                subsection = psm_subsections[idx % len(psm_subsections)]

                res_csb = conn.execute(
                    f"SELECT chunk_id, text, incident_name FROM '{CSB_REPORTS_CHUNKS_PARQUET}' "
                    f"WHERE incident_name LIKE ? AND token_count > 100 LIMIT 1",
                    [f"%{incident}%"],
                ).fetchone()

                if not res_csb:
                    continue

                csb_chunk_id, csb_text, full_incident_name = res_csb

                # Pull an OSHA regulating chunk (e.g. 1910.119(j) mechanical integrity)
                res_osha = conn.execute(
                    f"SELECT chunk_id, text FROM '{REGULATIONS_CHUNKS_PARQUET}' "
                    f"WHERE standard_number = '1910.119' AND section_id LIKE ? LIMIT 1",
                    [f"1910.119({subsection})%"],
                ).fetchone()

                if not res_osha:
                    continue

                osha_chunk_id, osha_text = res_osha

                prompt = _CSB_GEN_TEMPLATE.format(
                    csb_text=csb_text[:2000],
                    osha_text=osha_text[:2000],
                    csb_chunk_id=csb_chunk_id,
                    osha_chunk_id=osha_chunk_id,
                )

                response = self._llm.invoke(
                    [
                        SystemMessage(content=_CSB_GEN_SYSTEM),
                        HumanMessage(content=prompt),
                    ]
                )

                try:
                    parsed = json.loads(response.content)
                except Exception:
                    logger.warning(
                        f"Failed to parse LLM response for {incident} cross-corpus question."
                    )
                    continue

                entry: dict[str, Any] = {
                    "query": parsed.get("question", ""),
                    "expected_answer": parsed.get("expected_answer", ""),
                    "ground_truth_chunks": [csb_chunk_id, osha_chunk_id],
                    "source": "external_csb",
                    "category": "cross_corpus",
                    "expected_agents": ["csb_incident", "osha_regs"],
                    "complexity": "multi_context",
                    "metadata": {"incident_name": full_incident_name},
                }

                key_claims = parsed.get("key_claims", [])
                if isinstance(key_claims, list) and all(
                    isinstance(c, dict) for c in key_claims
                ):
                    entry["key_claims"] = key_claims

                if not validate_eval_entry(entry):
                    entries.append(entry)
        finally:
            conn.close()

        logger.info(f"Generated {len(entries)} CSB cross corpus Qs")
        return entries

    def _generate_enforcement_analytical_qs(self) -> list[dict[str, Any]]:
        """Run analytical queries on chunk parquets directly via duckdb."""
        entries: list[dict[str, Any]] = []
        conn = duckdb.connect(":memory:")
        try:
            # Question 1: Year with most fatality accidents (from chunks)
            res_fatality = conn.execute(
                f"SELECT EXTRACT(year FROM strptime(regexp_extract(event_date, '[0-9]+/[0-9]+/[0-9]+'), '%m/%d/%Y')) as yr, "
                f"COUNT(*) as c, list(chunk_id)[:10] as chunks "
                f"FROM '{ACCIDENTS_CHUNKS_PARQUET}' "
                f"WHERE fatality = true AND event_date IS NOT NULL "
                f"GROUP BY yr ORDER BY c DESC LIMIT 3"
            ).fetchall()

            if res_fatality:
                top_year, top_count, chunk_ids = res_fatality[0]
                entry: dict[str, Any] = {
                    "query": "Based on OSHA enforcement data, which recent year had the highest number of fatal workplace accidents?",
                    "expected_answer": f"The year {int(top_year)} had the highest number, with approximately {top_count} fatal accidents recorded inside the loaded corpus.",
                    "ground_truth_chunks": list(chunk_ids),
                    "source": "external_osha",
                    "category": "enforcement_analytical",
                    "expected_agents": ["osha_enf"],
                    "complexity": "reasoning",
                }
                if not validate_eval_entry(entry):
                    entries.append(entry)

            # Question 2: Proportion of accidents involving amputation vs fatality
            res_events = conn.execute(
                f"SELECT SUM(CASE WHEN fatality THEN 1 ELSE 0 END) as fatal_c, "
                f"SUM(CASE WHEN amputation THEN 1 ELSE 0 END) as amp_c, "
                f"list(chunk_id)[:10] as chunks "
                f"FROM '{ACCIDENTS_CHUNKS_PARQUET}'"
            ).fetchone()

            if res_events:
                fatal_c, amp_c, chunk_ids = res_events
                entry2: dict[str, Any] = {
                    "query": "Are there more fatal accidents or accidents involving amputations recorded in the OSHA accidents enforcement dataset?",
                    "expected_answer": f"In the loaded dataset, there are {amp_c} accidents involving amputations compared to {fatal_c} fatal accidents.",
                    "ground_truth_chunks": list(chunk_ids),
                    "source": "external_osha",
                    "category": "enforcement_analytical",
                    "expected_agents": ["osha_enf"],
                    "complexity": "reasoning",
                }
                if not validate_eval_entry(entry2):
                    entries.append(entry2)

            # Question 3: Dynamic LLM analytical grouping based on random fatals
            res_random = conn.execute(
                f"SELECT list(chunk_id), list(text) "
                f"FROM (SELECT chunk_id, text FROM '{ACCIDENTS_CHUNKS_PARQUET}' WHERE fatality = true AND token_count > 50 LIMIT 5)"
            ).fetchone()

            if res_random and res_random[0]:
                c_ids, texts = res_random
                combined_text = "\\n---\\n".join(texts)
                prompt = (
                    "Based on the following 5 fatal workplace accidents, generate ONE analytical question about "
                    "accident patterns (e.g., common causes, industries, or outcomes) and provide the expected answer grounded in the text.\\n\\n"
                    f"{combined_text}\\n\\n"
                    "Return valid JSON with keys: 'question', 'expected_answer'."
                )
                response = self._llm.invoke(
                    [
                        SystemMessage(
                            content="You generate analytical questions from accident data."
                        ),
                        HumanMessage(content=prompt),
                    ]
                )
                try:
                    parsed = json.loads(response.content)
                    entry_dyn: dict[str, Any] = {
                        "query": parsed.get("question", ""),
                        "expected_answer": parsed.get("expected_answer", ""),
                        "ground_truth_chunks": list(c_ids),
                        "source": "external_osha",
                        "category": "enforcement_analytical",
                        "expected_agents": ["osha_enf"],
                        "complexity": "reasoning",
                    }
                    if not validate_eval_entry(entry_dyn):
                        entries.append(entry_dyn)
                except Exception:
                    logger.warning(
                        "Failed to map dynamic enforcement analytical question."
                    )

        finally:
            conn.close()

        logger.info(f"Generated {len(entries)} analytical enforcement Qs")
        return entries

    def generate(self) -> list[dict[str, Any]]:
        """Run all generation strategies to output a combined list."""
        entries: list[dict[str, Any]] = []
        entries.extend(self._generate_psm_training_qs())
        entries.extend(self._generate_csb_cross_corpus_qs())
        entries.extend(self._generate_enforcement_analytical_qs())

        logger.info(f"ExternalSourceCurator generated {len(entries)} total entries.")
        return entries


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    curator = ExternalSourceCurator()
    res = curator.generate()
    from pprint import pprint

    pprint(res[:2])

    # Save output to disk
    out_path = EVAL_DATASET_DIR / "external_qa_entries.json"
    with open(out_path, "w") as f:
        json.dump(res, f, indent=2)
    logger.info(f"Saved entries to {out_path}")
