"""
Evaluation Framework for Chemical Process Safety LLM

Provides dataset construction, metric computation, and reporting utilities
for evaluating the agentic RAG pipeline across retrieval, faithfulness,
orchestration, and domain-specific quality dimensions.
"""

__version__ = "0.3.0"

from .config import (
    ACCIDENTS_CHUNKS_PARQUET,
    ADVERSARIAL_DATASET_FILE,
    BOOTSTRAP_ITERATIONS,
    CHROMADB_DIR,
    COLLECTION_TO_AGENT,
    CONFIDENCE_LEVEL,
    CSB_RECS_CHUNKS_PARQUET,
    CSB_REPORTS_CHUNKS_PARQUET,
    EVAL_DATASET_DIR,
    EVAL_DATASET_FILE,
    EVAL_HISTORY_DIR,
    EVAL_LLM_MODEL,
    EVAL_LLM_TEMPERATURE,
    EVAL_LOGS_DIR,
    EVAL_RESULTS_DIR,
    GENDUTY_SAMPLE_CHUNKS_PARQUET,
    INSPECTION_PARQUET,
    KEY_OSHA_STANDARDS,
    LOADED_CHUNKS_DIR,
    PROCESSED_CSB_DIR,
    PROCESSED_OSHA_DIR,
    REGULATIONS_CHUNKS_PARQUET,
    THRESHOLDS,
    VIOLATIONS_SAMPLE_CHUNKS_PARQUET,
)
from .generators.adversarial_qa_generator import AdversarialQAGenerator
from .generators.dataset_builder import DatasetBuilder
from .generators.enforcement_qa_generator import EnforcementQAGenerator
from .generators.external_source_curator import ExternalSourceCurator
from .generators.genduty_qa_generator import GendutyQAGenerator
from .generators.incident_qa_generator import IncidentQAGenerator
from .generators.regulatory_qa_generator import RegulatoryQAGenerator
from .generators.synthetic_generator import SyntheticGenerator
from .generators.violation_qa_generator import ViolationQAGenerator
from .schemas import (
    VALID_AGENTS,
    VALID_CATEGORIES,
    VALID_COMPLEXITIES,
    VALID_SOURCES,
    validate_eval_dataset,
    validate_eval_entry,
)
from .utils import (
    bootstrap_ci,
    configure_evaluation_file_logging,
    default_eval_log_path,
    extract_entity_names,
    extract_osha_standards,
    generate_run_id,
    load_eval_dataset,
    normalize_graph_state_for_eval,
    save_eval_results,
)

__all__ = [
    # Config constants
    "EVAL_DATASET_DIR",
    "EVAL_RESULTS_DIR",
    "EVAL_LOGS_DIR",
    "EVAL_HISTORY_DIR",
    "CHROMADB_DIR",
    "COLLECTION_TO_AGENT",
    "LOADED_CHUNKS_DIR",
    "PROCESSED_OSHA_DIR",
    "PROCESSED_CSB_DIR",
    "EVAL_DATASET_FILE",
    "ADVERSARIAL_DATASET_FILE",
    "EVAL_LLM_MODEL",
    "EVAL_LLM_TEMPERATURE",
    "THRESHOLDS",
    "BOOTSTRAP_ITERATIONS",
    "CONFIDENCE_LEVEL",
    "KEY_OSHA_STANDARDS",
    "REGULATIONS_CHUNKS_PARQUET",
    "ACCIDENTS_CHUNKS_PARQUET",
    "VIOLATIONS_SAMPLE_CHUNKS_PARQUET",
    "GENDUTY_SAMPLE_CHUNKS_PARQUET",
    "CSB_REPORTS_CHUNKS_PARQUET",
    "CSB_RECS_CHUNKS_PARQUET",
    "INSPECTION_PARQUET",
    # Schema validation
    "validate_eval_entry",
    "validate_eval_dataset",
    "VALID_SOURCES",
    "VALID_CATEGORIES",
    "VALID_COMPLEXITIES",
    "VALID_AGENTS",
    # Utility functions
    "load_eval_dataset",
    "normalize_graph_state_for_eval",
    "configure_evaluation_file_logging",
    "default_eval_log_path",
    "save_eval_results",
    "bootstrap_ci",
    "extract_osha_standards",
    "extract_entity_names",
    "generate_run_id",
    # Generators
    "RegulatoryQAGenerator",
    "IncidentQAGenerator",
    "EnforcementQAGenerator",
    "ViolationQAGenerator",
    "GendutyQAGenerator",
    "ExternalSourceCurator",
    "AdversarialQAGenerator",
    "SyntheticGenerator",
    "DatasetBuilder",
]
