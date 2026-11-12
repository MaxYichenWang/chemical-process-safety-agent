"""
Adversarial Q&A Generator.

Generates 24 well-crafted adversarial evaluation pairs spanning four categories:
1. Non-existent incidents
2. Entity-swapped questions
3. Out-of-scope queries
4. Reasoning-beyond-context

Outputs follow the unified EvalEntry schema with 'source' and 'category' set to 'adversarial'.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from src.evaluation.config import ADVERSARIAL_DATASET_FILE
from src.evaluation.schemas import validate_eval_entry

logger = logging.getLogger(__name__)

# Pre-defined adversarial questions covering the 4 categories
_ADVERSARIAL_CASES = [
    # Category 1: Non-existent incidents
    {
        "query": "What were the key findings from the CSB investigation of the 2023 Springfield Chemical Plant ammonium nitrate explosion?",
        "expected_answer": "Insufficient evidence. The 2023 Springfield Chemical Plant incident does not exist in the CSB or OSHA database.",
        "complexity": "simple",
    },
    {
        "query": "Describe the OSHA PSM violations cited after the 2018 Metropolis Refinery gas leak.",
        "expected_answer": "This incident does not exist in the available OSHA enforcement corpus.",
        "complexity": "simple",
    },
    {
        "query": "According to the CSB report, what caused the 2020 Oceanview Plastics Facility fire?",
        "expected_answer": "The Oceanview Plastics Facility fire is a fictitious or undocumented incident. The CSB corpus does not contain a report for this event.",
        "complexity": "simple",
    },
    {
        "query": "How many fatalities were recorded in the 2019 Dunder Mifflin Paper Mill boiler explosion based on OSHA accident investigations?",
        "expected_answer": "There is no record of a 2019 Dunder Mifflin Paper Mill boiler explosion in the OSHA accident data.",
        "complexity": "simple",
    },
    {
        "query": "What were the safety recommendations issued following the 2024 Gotham Chemical Corporation incident?",
        "expected_answer": "The database contains no CSB recommendations for a 2024 Gotham Chemical Corporation incident.",
        "complexity": "simple",
    },
    {
        "query": "Detail the Process Safety Management failures identified in the 2021 Star City Fertilizer plant disaster.",
        "expected_answer": "The 2021 Star City Fertilizer plant disaster is not a documented incident in the available data.",
        "complexity": "simple",
    },
    # Category 2: Entity-swapped questions
    {
        "query": "What caused the chlorine gas release at the West Fertilizer plant in West, Texas?",
        "expected_answer": "The premise of the question is incorrect. The West Fertilizer plant incident involved an ammonium nitrate explosion, not a chlorine gas release.",
        "complexity": "reasoning",
    },
    {
        "query": "According to the BP Texas City investigation, what were the failures in managing the liquid nitrogen storage?",
        "expected_answer": "The premise is incorrect. The BP Texas City incident involved an isomerization unit and a raffinate splitter tower overflow resulting in a hydrocarbon explosion, not liquid nitrogen storage.",
        "complexity": "reasoning",
    },
    {
        "query": "How did the 29 CFR 1910.119 PSM mechanical integrity failures directly cause the Arkema Crosby chemical plant flooding?",
        "expected_answer": "The premise is incorrect. The Arkema Crosby flooding was caused by extreme rainfall from Hurricane Harvey. The flooding led to power loss and warming of organic peroxides, rather than a mechanical integrity failure causing the flood itself.",
        "complexity": "reasoning",
    },
    {
        "query": "What were the main conclusions about the hydrogen sulfide release at the Chevron Richmond refinery?",
        "expected_answer": "The Chevron Richmond incident involved a ruptured pipe causing a massive hydrocarbon vapor cloud leak and fire, not a hydrogen sulfide release.",
        "complexity": "reasoning",
    },
    {
        "query": "Explain the OSHA violations related to the Williams Olefins plant combustible dust explosion.",
        "expected_answer": "The premise is incorrect. The Williams Olefins incident was caused by the overpressure and rupture of a propylene fractionator reboiler, not a combustible dust explosion.",
        "complexity": "reasoning",
    },
    {
        "query": "What did the CSB recommend for combustible dust hazards following the Husky Energy refinery incident?",
        "expected_answer": "The Husky Energy refinery incident involved a fluid catalytic cracking unit (FCCU) and an explosion leading to an asphalt fire, not combustible dust hazards.",
        "complexity": "reasoning",
    },
    # Category 3: Out-of-scope queries
    {
        "query": "What are the EPA Clean Air Act Title V permitting requirements for coal-fired power plants?",
        "expected_answer": "Out of scope. The database covers OSHA Process Safety Management (PSM) and CSB incident investigations, not EPA clean air permitting.",
        "complexity": "simple",
    },
    {
        "query": "How do I comply with DOT regulations for transporting hazardous materials across state lines?",
        "expected_answer": "Out of scope. The system contains OSHA workplace safety data and CSB reports, not Department of Transportation (DOT) shipping regulations.",
        "complexity": "simple",
    },
    {
        "query": "What are the FDA guidelines for manufacturing pharmaceutical-grade chemicals?",
        "expected_answer": "Out of scope. Food and Drug Administration (FDA) guidelines are not covered in the OSHA/CSB chemical process safety dataset.",
        "complexity": "simple",
    },
    {
        "query": "Describe the OSHA fall protection standards for residential construction sites.",
        "expected_answer": "Out of scope. While fall protection is an OSHA standard, this system is focused on chemical process safety (1910.119) and industrial accidents rather than residential construction.",
        "complexity": "simple",
    },
    {
        "query": "What are the environmental remediation procedures for a superfund site under CERCLA?",
        "expected_answer": "Out of scope. The dataset covers industrial safety enforcement and process incidents, not environmental remediation or superfund protocols.",
        "complexity": "simple",
    },
    {
        "query": "How does the IRS calculate tax credits for renewable energy production at chemical facilities?",
        "expected_answer": "Out of scope. The system is designed for answering queries related to process safety management and incidents, not tax compliance.",
        "complexity": "simple",
    },
    # Category 4: Reasoning-beyond-context
    {
        "query": "Based on historical CSB data, predict which chemical facilities are most likely to have incidents in 2027.",
        "expected_answer": "The system cannot predict future incidents. Finding such predictions requires extrapolating beyond the historical factual data stored in the corpus.",
        "complexity": "reasoning",
    },
    {
        "query": "What emerging chemical safety regulations will OSHA likely promulgate in the next decade?",
        "expected_answer": "Speculating on future regulatory actions goes beyond the documented historical OSHA standards and enforcement data included in the database.",
        "complexity": "reasoning",
    },
    {
        "query": "Given the trends in the OSHA enforcement data, will mechanical integrity violations increase or decrease next year?",
        "expected_answer": "The system cannot forecast compliance trends. The database provides historical data, not predictive statistics.",
        "complexity": "reasoning",
    },
    {
        "query": "If the West Fertilizer facility had used an AI-based process control system, would the accident have been prevented?",
        "expected_answer": "This requires speculation beyond the scope of the CSB investigation report, which only details the actual causes and recommendations surrounding the incident.",
        "complexity": "reasoning",
    },
    {
        "query": "Which current CEO of a major chemical company has the best track record for process safety?",
        "expected_answer": "The system lacks subjective ratings of modern corporate executives' track records; it contains specific incident and enforcement findings.",
        "complexity": "reasoning",
    },
    {
        "query": "What will be the exact financial impact of climate change on the frequency of CSB investigations?",
        "expected_answer": "The provided corpus contains historical process safety incidents and regulations. It does not contain financial projections regarding climate change impact.",
        "complexity": "reasoning",
    },
]


class AdversarialQAGenerator:
    """Provides a programmatic pipeline to output adversarial Q&A evaluation entries."""

    def __init__(self, output_file: Path | None = None):
        self.output_file = output_file or ADVERSARIAL_DATASET_FILE

    def generate(self) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []
        for case in _ADVERSARIAL_CASES:
            entry = {
                "query": case["query"],
                "expected_answer": case["expected_answer"],
                "ground_truth_chunks": [],  # Adversarial cases by nature often omit grounding
                "source": "adversarial",
                "category": "adversarial",
                "complexity": case["complexity"],
                # adversarial queries expect a refusal, not a retrieval
                # agent. No worker should be routed; correct behaviour is to
                # decline / flag the unverifiable premise.
                "expected_agents": [],
                "expected_behavior": "refuse",
                # no vector ground truth exists => exclude from Layer-1
                # retrieval metrics so denominators stay honest.
                "l1_applicable": False,
            }

            errs = validate_eval_entry(entry)
            if not errs:
                entries.append(entry)
            else:
                logger.warning(
                    f"Adversarial schema discrepancy on '{case['query'][:30]}...': {errs}"
                )

        logger.info(f"Generated {len(entries)} adversarial entries.")

        # Write to JSON
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_file, "w") as f:
            json.dump(entries, f, indent=2)

        logger.info(f"Saved adversarial dataset to {self.output_file}")
        return entries


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    gen = AdversarialQAGenerator()
    res = gen.generate()
