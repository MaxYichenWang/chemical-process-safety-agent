"""
Statistical Q&A generator for evaluation dataset Track F.

Produces evaluation entries for statistical queries against structured
OSHA/CSB data, spanning four sub-categories: factual, comparative, trend,
and hybrid analytical.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.evaluation.config import GENERATOR_CHECKPOINT_EVERY

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Question templates by sub-category
# ---------------------------------------------------------------------------

_STATISTICAL_FACTUAL_TEMPLATES = [
    {
        "query": "How many OSHA violations are recorded in the database?",
        "expected_sql": "SELECT COUNT(*) as total_violations FROM violations LIMIT 1000",
        "category": "statistical_factual",
    },
    {
        "query": "What is the average penalty amount for OSHA violations?",
        "expected_sql": "SELECT AVG(penalty_amount) as avg_penalty FROM violations LIMIT 1000",
        "category": "statistical_factual",
    },
    {
        "query": "How many OSHA inspections are recorded in the database?",
        "expected_sql": "SELECT COUNT(*) as total_inspections FROM inspections LIMIT 1000",
        "category": "statistical_factual",
    },
    {
        "query": "How many CSB-investigated incidents are recorded in the database?",
        "expected_sql": "SELECT COUNT(*) as total_incidents FROM csb_incidents LIMIT 1000",
        "category": "statistical_factual",
    },
    {
        "query": "How many CSB safety recommendations are in the database?",
        "expected_sql": "SELECT COUNT(*) as total_recommendations FROM csb_recommendations LIMIT 1000",
        "category": "statistical_factual",
    },
    {
        "query": "How many OSHA accident records exist in the database?",
        "expected_sql": "SELECT COUNT(*) as total_accidents FROM accidents LIMIT 1000",
        "category": "statistical_factual",
    },
    {
        "query": "What is the total penalty amount for all OSHA violations?",
        "expected_sql": "SELECT SUM(penalty_amount) as total_penalties FROM violations LIMIT 1000",
        "category": "statistical_factual",
    },
    {
        "query": "How many unique OSHA standard numbers appear in violation records?",
        "expected_sql": "SELECT COUNT(DISTINCT standard_number) as unique_standards FROM violations LIMIT 1000",
        "category": "statistical_factual",
    },
    {
        "query": "How many general duty violations are recorded?",
        "expected_sql": "SELECT COUNT(*) as genduty_count FROM genduty_violations LIMIT 1000",
        "category": "statistical_factual",
    },
    {
        "query": "How many regulation sections are in the database?",
        "expected_sql": "SELECT COUNT(*) as regulation_sections FROM regulations LIMIT 1000",
        "category": "statistical_factual",
    },
]

_STATISTICAL_COMPARATIVE_TEMPLATES = [
    {
        "query": "Which violation types have the highest count in the OSHA violations database?",
        "expected_sql": "SELECT violation_type, COUNT(*) as cnt FROM violations GROUP BY violation_type ORDER BY cnt DESC LIMIT 10",
        "category": "statistical_comparative",
    },
    {
        "query": "What are the top 5 most common OSHA standard numbers in violation records?",
        "expected_sql": "SELECT standard_number, COUNT(*) as cnt FROM violations GROUP BY standard_number ORDER BY cnt DESC LIMIT 5",
        "category": "statistical_comparative",
    },
    {
        "query": "Compare the count of violations by type (Serious vs Willful vs Repeat).",
        "expected_sql": "SELECT violation_type, COUNT(*) as cnt FROM violations WHERE violation_type IN ('Serious', 'Willful', 'Repeat') GROUP BY violation_type ORDER BY cnt DESC LIMIT 1000",
        "category": "statistical_comparative",
    },
    {
        "query": "What are the top 5 OSHA standards with the highest average penalty?",
        "expected_sql": "SELECT standard_number, AVG(penalty_amount) as avg_penalty FROM violations GROUP BY standard_number ORDER BY avg_penalty DESC LIMIT 5",
        "category": "statistical_comparative",
    },
    {
        "query": "Which violation types have the highest average penalty amount?",
        "expected_sql": "SELECT violation_type, AVG(penalty_amount) as avg_penalty FROM violations GROUP BY violation_type ORDER BY avg_penalty DESC LIMIT 10",
        "category": "statistical_comparative",
    },
    {
        "query": "Compare the total number of violations for PSM standard 1910.119 versus lockout/tagout standard 1910.147.",
        "expected_sql": "SELECT standard_number, COUNT(*) as cnt FROM violations WHERE standard_number IN ('1910.119', '1910.147') GROUP BY standard_number ORDER BY cnt DESC LIMIT 1000",
        "category": "statistical_comparative",
    },
    {
        "query": "What are the top 10 most frequently cited OSHA standards?",
        "expected_sql": "SELECT standard_number, COUNT(*) as citation_count FROM violations GROUP BY standard_number ORDER BY citation_count DESC LIMIT 10",
        "category": "statistical_comparative",
    },
    {
        "query": "Which OSHA standards have the highest total penalty amounts?",
        "expected_sql": "SELECT standard_number, SUM(penalty_amount) as total_penalty FROM violations GROUP BY standard_number ORDER BY total_penalty DESC LIMIT 10",
        "category": "statistical_comparative",
    },
]

_STATISTICAL_TREND_TEMPLATES = [
    {
        "query": "How has the number of OSHA violations changed over the years?",
        "expected_sql": "SELECT EXTRACT(YEAR FROM issuance_date) as year, COUNT(*) as cnt FROM violations WHERE issuance_date IS NOT NULL GROUP BY year ORDER BY year LIMIT 1000",
        "category": "statistical_trend",
    },
    {
        "query": "What is the trend of average OSHA penalties over time?",
        "expected_sql": "SELECT EXTRACT(YEAR FROM issuance_date) as year, AVG(penalty_amount) as avg_penalty FROM violations WHERE issuance_date IS NOT NULL GROUP BY year ORDER BY year LIMIT 1000",
        "category": "statistical_trend",
    },
    {
        "query": "How has the distribution of violation types changed year over year?",
        "expected_sql": "SELECT EXTRACT(YEAR FROM issuance_date) as year, violation_type, COUNT(*) as cnt FROM violations WHERE issuance_date IS NOT NULL GROUP BY year, violation_type ORDER BY year, cnt DESC LIMIT 1000",
        "category": "statistical_trend",
    },
    {
        "query": "What is the year-over-year trend of total OSHA penalties?",
        "expected_sql": "SELECT EXTRACT(YEAR FROM issuance_date) as year, SUM(penalty_amount) as total_penalty FROM violations WHERE issuance_date IS NOT NULL GROUP BY year ORDER BY year LIMIT 1000",
        "category": "statistical_trend",
    },
    {
        "query": "How has the count of Willful violations trended over the years?",
        "expected_sql": "SELECT EXTRACT(YEAR FROM issuance_date) as year, COUNT(*) as cnt FROM violations WHERE violation_type = 'Willful' AND issuance_date IS NOT NULL GROUP BY year ORDER BY year LIMIT 1000",
        "category": "statistical_trend",
    },
]

_HYBRID_ANALYTICAL_TEMPLATES = [
    {
        "query": "What are the most frequently violated OSHA standards, and what do CSB investigations reveal about these violations?",
        "expected_sql": "SELECT standard_number, COUNT(*) as violation_count FROM violations GROUP BY standard_number ORDER BY violation_count DESC LIMIT 10",
        "category": "hybrid_analytical",
        "expected_agents": ["data_analyst", "csb_incident"],
    },
    {
        "query": "Which OSHA standards have the highest penalties, and what are the regulatory requirements for those standards?",
        "expected_sql": "SELECT standard_number, AVG(penalty_amount) as avg_penalty, COUNT(*) as cnt FROM violations GROUP BY standard_number ORDER BY avg_penalty DESC LIMIT 10",
        "category": "hybrid_analytical",
        "expected_agents": ["data_analyst", "osha_regs"],
    },
    {
        "query": "How many Willful OSHA violations exist, and what root causes are commonly associated with willful violations?",
        "expected_sql": "SELECT COUNT(*) as willful_count FROM violations WHERE violation_type = 'Willful' LIMIT 1000",
        "category": "hybrid_analytical",
        "expected_agents": ["data_analyst", "root_cause"],
    },
    {
        "query": "What is the total count of OSHA violations for PSM standard 1910.119, and what are the key PSM regulatory requirements?",
        "expected_sql": "SELECT COUNT(*) as psm_violations FROM violations WHERE standard_number ILIKE '1910.119%' LIMIT 1000",
        "category": "hybrid_analytical",
        "expected_agents": ["data_analyst", "osha_regs"],
    },
    {
        "query": "How many CSB recommendations exist, and what are the most common recommendation themes?",
        "expected_sql": "SELECT COUNT(*) as total_recs FROM csb_recommendations LIMIT 1000",
        "category": "hybrid_analytical",
        "expected_agents": ["data_analyst", "csb_incident"],
    },
    {
        "query": "What are the total OSHA penalties by violation type, and what enforcement actions does OSHA typically take for each type?",
        "expected_sql": "SELECT violation_type, SUM(penalty_amount) as total_penalty, COUNT(*) as cnt FROM violations GROUP BY violation_type ORDER BY total_penalty DESC LIMIT 1000",
        "category": "hybrid_analytical",
        "expected_agents": ["data_analyst", "osha_enf"],
    },
    {
        "query": "What are the top violated OSHA standards in terms of penalty amounts, and what safety recommendations has CSB issued related to those standards?",
        "expected_sql": "SELECT standard_number, SUM(penalty_amount) as total_penalty FROM violations GROUP BY standard_number ORDER BY total_penalty DESC LIMIT 10",
        "category": "hybrid_analytical",
        "expected_agents": ["data_analyst", "csb_incident"],
    },
]


class StatisticalQAGenerator:
    """Generates evaluation entries for statistical queries.

    Produces entries conforming to the extended EvalEntry schema with
    ``expected_sql`` and ``ground_truth_result`` fields.
    """

    def __init__(
        self,
        target_entries: int = 40,
        checkpoint_path: Optional[Path] = None,
        checkpoint_every: int = GENERATOR_CHECKPOINT_EVERY,
    ):
        self.target_entries = target_entries
        self.checkpoint_path = checkpoint_path
        self.checkpoint_every = checkpoint_every
        self._all_templates = (
            _STATISTICAL_FACTUAL_TEMPLATES
            + _STATISTICAL_COMPARATIVE_TEMPLATES
            + _STATISTICAL_TREND_TEMPLATES
            + _HYBRID_ANALYTICAL_TEMPLATES
        )

    def _execute_ground_truth(self, sql: str, conn) -> Optional[Dict[str, Any]]:
        """Execute SQL and return the result as a serializable dict."""
        try:
            df = conn.execute(sql).fetchdf()
            return {
                "columns": list(df.columns),
                "row_count": len(df),
                "data": df.head(20).to_dict(orient="records"),
            }
        except Exception as e:
            logger.warning(f"Failed to execute ground truth SQL: {e}")
            return None

    def generate(self, conn=None) -> List[Dict[str, Any]]:
        """Generate evaluation entries for statistical queries.

        Args:
            conn: Optional DuckDB connection. If provided, executes SQL
                to populate ground_truth_result fields.

        Returns:
            List of validated EvalEntry dicts.
        """
        entries: List[Dict[str, Any]] = []

        for template in self._all_templates:
            if len(entries) >= self.target_entries:
                break

            entry: Dict[str, Any] = {
                "query": template["query"],
                "expected_answer": f"Statistical result from: {template['expected_sql']}",
                "ground_truth_chunks": [],
                "source": "auto_statistical",
                "category": template["category"],
                "expected_sql": template["expected_sql"],
                "expected_agents": template.get("expected_agents", ["data_analyst"]),
            }

            if conn is not None:
                gt = self._execute_ground_truth(template["expected_sql"], conn)
                if gt is not None:
                    entry["ground_truth_result"] = gt

            entries.append(entry)

            # Checkpoint
            if self.checkpoint_path and len(entries) % self.checkpoint_every == 0:
                self._save_checkpoint(entries)

        if self.checkpoint_path:
            self._save_checkpoint(entries)

        return entries

    def _save_checkpoint(self, entries: List[Dict[str, Any]]) -> None:
        """Save current entries to checkpoint file."""
        if self.checkpoint_path is None:
            return
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.checkpoint_path, "w", encoding="utf-8") as f:
            json.dump(entries, f, indent=2, default=str)
        logger.info(
            f"Checkpoint saved: {len(entries)} entries to {self.checkpoint_path}"
        )
