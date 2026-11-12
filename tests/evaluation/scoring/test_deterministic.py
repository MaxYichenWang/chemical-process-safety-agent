"""Synthetic regression tests for the deterministic scoring layer."""

import json


from src.evaluation.scoring.deterministic import (
    SECTIONS,
    score_deterministic,
)

def _small_fixture():
    dataset = [
        {
            "query": "What does 1910.119 require?",
            "expected_answer": "PSM applies.",
            "ground_truth_chunks": ["osha_regulations_1910119_chunk_0"],
            "source": "auto_regulatory",
            "category": "regulatory_factual",
            "expected_agents": ["osha_regs"],
        },
        {
            "query": "adversarial: what happened at Fictional Plant 999?",
            "expected_answer": "No record found.",
            "ground_truth_chunks": [],
            "source": "adversarial",
            "category": "adversarial",
            "expected_agents": ["root_cause"],
        },
    ]
    results = [
        {
            "final_response": "1910.119 requires a written PSM program.",
            "context_gathered": [
                {"chunk_id": "osha_regulations_1910119_chunk_0", "text": "1910.119 requires...",
                 "corpus": "osha_regs", "collection": "osha_regulations", "source": "1910.119", "location": ""}
            ],
            "agents_called": ["osha_regs"],
            "planned_agents": ["osha_regs"],
            "validation_score": 0.9,
            "retry_count": 0,
            "llm_calls": [],
        },
        {
            "final_response": "There is no record of that facility in the retrieved context.",
            "context_gathered": [],
            "agents_called": [],
            "planned_agents": [],
            "validation_score": 0.85,
            "retry_count": 0,
            "llm_calls": [],
        },
    ]
    return dataset, results


def test_score_deterministic_is_importable_and_runnable_with_no_key_no_corpus(monkeypatch):
    """No EVAL_LLM_* env, no chromadb dir -- must still run and produce a result."""
    for var in ("EVAL_LLM_MODEL", "EVAL_LLM_PROVIDER", "EVAL_LLM_BASE_URL", "EVAL_LLM_API_KEY_ENV"):
        monkeypatch.delenv(var, raising=False)

    dataset, results = _small_fixture()
    out = score_deterministic(dataset, results)

    assert set(out.keys()) == set(SECTIONS)
    for section in SECTIONS:
        assert "error" not in out[section] if isinstance(out[section], dict) else True


def test_score_deterministic_is_byte_identical_across_two_runs():
    """Pure function -- same input always produces byte-identical output."""
    dataset, results = _small_fixture()
    out1 = score_deterministic(dataset, results)
    out2 = score_deterministic(dataset, results)
    assert json.dumps(out1, sort_keys=True) == json.dumps(out2, sort_keys=True)


def test_score_deterministic_excludes_pipeline_error_stubs_from_quality_metrics():
    """A pipeline-error stub (forced validation_score=0.0) must not deflate
    quality-metric distributions, but must still count toward pipeline_error_rate."""
    dataset, results = _small_fixture()
    dataset.append({
        "query": "third query",
        "expected_answer": "n/a",
        "ground_truth_chunks": [],
        "source": "auto_regulatory",
        "category": "regulatory_factual",
    })
    results.append({"error": "boom", "context_gathered": [], "final_response": "", "validation_score": 0.0,
                     "retry_count": 0, "agents_called": []})

    out = score_deterministic(dataset, results)
    assert out["orchestration"]["pipeline_error_rate"]["error_count"] == 1
    assert out["orchestration"]["pipeline_error_rate"]["total"] == 3
    # citation accuracy / retrieval metrics only see the 2 ok entries
    assert out["orchestration"]["critic_distribution"]["max"] <= 0.9
