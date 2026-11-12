"""
Tests for src.evaluation.config.

Verifies that all configuration constants are importable, have the expected
types, and contain their documented default values.
"""

from pathlib import Path

import importlib
import json
import os
import subprocess
import sys

from src.evaluation import config

from src.evaluation.config import (
    ADVERSARIAL_DATASET_FILE,
    BOOTSTRAP_ITERATIONS,
    CHROMADB_DIR,
    CONFIDENCE_LEVEL,
    EVAL_DATASET_DIR,
    EVAL_DATASET_FILE,
    EVAL_HISTORY_DIR,
    EVAL_LOGS_DIR,
    EVAL_LLM_MODEL,
    EVAL_LLM_TEMPERATURE,
    EVAL_RESULTS_DIR,
    KEY_OSHA_STANDARDS,
    LOADED_CHUNKS_DIR,
    PROCESSED_CSB_DIR,
    PROCESSED_OSHA_DIR,
    THRESHOLDS,
)


# -- Path constants ----------------------------------------------------------


def test_path_constants_are_path_instances():
    """All directory/file constants should be pathlib.Path objects."""
    path_vars = [
        EVAL_DATASET_DIR,
        EVAL_RESULTS_DIR,
        EVAL_LOGS_DIR,
        EVAL_HISTORY_DIR,
        CHROMADB_DIR,
        LOADED_CHUNKS_DIR,
        PROCESSED_OSHA_DIR,
        PROCESSED_CSB_DIR,
        EVAL_DATASET_FILE,
        ADVERSARIAL_DATASET_FILE,
    ]
    for p in path_vars:
        assert isinstance(p, Path), f"{p!r} is not a Path"


def test_results_dir_is_child_of_dataset_dir():
    """EVAL_RESULTS_DIR should live under EVAL_DATASET_DIR."""
    assert EVAL_RESULTS_DIR.parts[:-1] == EVAL_DATASET_DIR.parts


def test_logs_dir_is_child_of_results_dir():
    """EVAL_LOGS_DIR should live under EVAL_RESULTS_DIR."""
    assert EVAL_LOGS_DIR.parts[:-1] == EVAL_RESULTS_DIR.parts


def test_history_dir_is_child_of_dataset_dir():
    """EVAL_HISTORY_DIR should live under EVAL_DATASET_DIR."""
    assert EVAL_HISTORY_DIR.parts[:-1] == EVAL_DATASET_DIR.parts


def test_dataset_file_extension():
    """Dataset files should end in .json."""
    assert EVAL_DATASET_FILE.suffix == ".json"
    assert ADVERSARIAL_DATASET_FILE.suffix == ".json"


# -- LLM settings -----------------------------------------------------------


def test_eval_llm_model_is_string():
    """EVAL_LLM_MODEL must be a non-empty string."""
    assert isinstance(EVAL_LLM_MODEL, str)
    assert len(EVAL_LLM_MODEL) > 0


def test_eval_llm_temperature():
    """EVAL_LLM_TEMPERATURE must be a valid float."""
    assert isinstance(EVAL_LLM_TEMPERATURE, float)


# -- Thresholds --------------------------------------------------------------


_EXPECTED_THRESHOLD_KEYS = {
    "context_precision",
    "context_recall",
    "hit_rate_at_5",
    "mrr",
    "ndcg_at_5",
    "faithfulness",
    "hallucination_rate",
    "answer_correctness",
    "factual_correctness",
    "citation_accuracy",
    "goal_accuracy",
    "tool_call_accuracy",
    "topic_adherence",
    "critic_score_mean",
    "retry_rate",
    "sql_execution_success",
    "sql_correctness",
    "sql_correctness_relaxed",
    "numerical_fidelity",
    "hybrid_synthesis_quality",
    "adversarial_hedge_rate",
    "standard_hedge_rate",
}


def test_thresholds_has_all_expected_keys():
    """THRESHOLDS dict must contain every metric listed in the plan."""
    assert set(THRESHOLDS.keys()) == _EXPECTED_THRESHOLD_KEYS


def test_thresholds_values_are_floats():
    """Every threshold value should be a float in [0, 1]."""
    for key, value in THRESHOLDS.items():
        assert isinstance(value, float), f"THRESHOLDS['{key}'] is not float"
        assert 0.0 <= value <= 1.0, f"THRESHOLDS['{key}']={value} not in [0,1]"


# -- Bootstrap settings ------------------------------------------------------


def test_bootstrap_iterations():
    """BOOTSTRAP_ITERATIONS should be a positive integer."""
    assert isinstance(BOOTSTRAP_ITERATIONS, int)
    assert BOOTSTRAP_ITERATIONS > 0


def test_confidence_level():
    """CONFIDENCE_LEVEL should be a float in (0, 1)."""
    assert isinstance(CONFIDENCE_LEVEL, float)
    assert 0.0 < CONFIDENCE_LEVEL < 1.0


# -- KEY_OSHA_STANDARDS ------------------------------------------------------


def test_key_osha_standards_is_list_of_strings():
    """KEY_OSHA_STANDARDS should be a non-empty list of strings."""
    assert isinstance(KEY_OSHA_STANDARDS, list)
    assert len(KEY_OSHA_STANDARDS) > 0
    for std in KEY_OSHA_STANDARDS:
        assert isinstance(std, str)


def test_key_osha_standards_format():
    """Each entry should look like an OSHA standard number (1910.xxx)."""
    for std in KEY_OSHA_STANDARDS:
        assert std.startswith("1910."), f"'{std}' does not start with '1910.'"


def test_dataset_directory_override(monkeypatch, tmp_path):
    """A downloaded benchmark can live outside the source checkout."""
    with monkeypatch.context() as env:
        env.setenv("EVAL_DATASET_DIR", str(tmp_path))
        env.delenv("EVAL_DATASET_FILE", raising=False)
        importlib.reload(config)
        try:
            assert config.EVAL_DATASET_FILE == tmp_path / "eval_dataset_v1.json"
            assert config.ADVERSARIAL_DATASET_FILE == tmp_path / "eval_dataset_adversarial.json"
            assert config.EVAL_RESULTS_DIR == tmp_path / "results"
        finally:
            env.undo()
            importlib.reload(config)


def test_dataset_file_override_takes_precedence(monkeypatch, tmp_path):
    """The main split can be replaced without changing the adversarial path."""
    fixture = Path(__file__).resolve().parents[1] / "fixtures" / "sample_eval_dataset.json"
    with monkeypatch.context() as env:
        env.setenv("EVAL_DATASET_DIR", str(tmp_path))
        env.setenv("EVAL_DATASET_FILE", str(fixture))
        importlib.reload(config)
        try:
            assert config.EVAL_DATASET_FILE == fixture
            assert config.ADVERSARIAL_DATASET_FILE.parent == tmp_path
            from src.evaluation.utils import load_eval_dataset

            assert len(load_eval_dataset(config.EVAL_DATASET_FILE)) == 2
        finally:
            env.undo()
            importlib.reload(config)


def test_shared_corpus_paths_and_cli_audit_from_another_directory(tmp_path):
    """Use a new process so cached imports cannot hide ignored path overrides."""
    root = Path(__file__).resolve().parents[2]
    index = tmp_path / "custom-index"
    index.mkdir()
    chunks = tmp_path / "custom-chunks"
    dataset = tmp_path / "dataset.json"
    dataset.write_text(json.dumps([{
        "query": "Synthetic", "expected_answer": "Synthetic", "ground_truth_chunks": []
    }]))
    source = '''
from pathlib import Path
import os
from src.data_processing import config as processing
from src.evaluation import config as evaluation
from src.evaluation import run
assert evaluation.CHROMADB_DIR == processing.CHROMADB_DIR == Path(os.environ["CHROMADB_DIR"])
assert evaluation.LOADED_CHUNKS_DIR == processing.CHUNKS_DIR == Path(os.environ["CHUNKS_DIR"])
assert evaluation.PROCESSED_CSB_DIR == processing.CSB_DATA_DIR
assert evaluation.PROCESSED_OSHA_DIR == processing.OSHA_DATA_DIR
assert evaluation.REGULATIONS_CHUNKS_PARQUET == processing.CHUNKS_DIR / "osha_regulations_chunks.parquet"
assert run.main(["generate", "--dataset", os.environ["TEST_DATASET"], "--dry-run"]) == 0
'''
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join([str(root), os.environ.get("PYTHONPATH", "")]),
        "CHROMADB_DIR": str(index), "CHUNKS_DIR": str(chunks),
        "CSB_DATA_DIR": str(tmp_path / "csb"), "OSHA_DATA_DIR": str(tmp_path / "osha"),
        "TEST_DATASET": str(dataset),
    }
    result = subprocess.run([sys.executable, "-c", source], cwd=tmp_path, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "gt-validation: would run" in result.stdout
    assert not (tmp_path / "chromadb").exists()
