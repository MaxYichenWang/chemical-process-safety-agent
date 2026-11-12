import json

import pytest

from src.evaluation.scoring.contract import (
    GenerationMismatchError,
    build_score_file,
    sanitize_judge_id,
    sha256_bytes,
    sha256_file,
    sha256_json,
    verify_generation_sha,
)


def test_sha256_file_matches_manual_hash(tmp_path):
    p = tmp_path / "generation.json"
    p.write_text('{"a": 1}', encoding="utf-8")
    assert sha256_file(p) == sha256_bytes(b'{"a": 1}')


def test_sha256_json_is_order_independent_over_dict_keys():
    a = {"b": 2, "a": 1}
    b = {"a": 1, "b": 2}
    assert sha256_json(a) == sha256_json(b)


def test_sha256_json_is_order_dependent_over_list_order():
    assert sha256_json([1, 2]) != sha256_json([2, 1])


def test_sanitize_judge_id_replaces_dots_and_slashes():
    assert sanitize_judge_id("qwen3.7-flash") == "qwen3-7-flash"
    assert sanitize_judge_id("meta/llama-3.3") == "meta-llama-3-3"
    assert sanitize_judge_id(None) == "none"


def test_build_score_file_has_all_contract_fields():
    sf = build_score_file(
        layer="deepeval",
        generation_sha256="abc123",
        dataset_sha256="def456",
        per_entry={"0": {"faithfulness": 0.9}},
        aggregates={"faithfulness": {"mean": 0.9}},
        started_utc="2026-08-01T00:00:00+00:00",
        wall_seconds=12.3,
        n_scored=1,
        n_failed=0,
        judge={"provider": "qwen", "model": "qwen3.7-flash", "temperature": 0.0,
               "seed": 42, "same_family_as_pipeline": False},
        judge_calls=3,
        judge_cost_usd=0.01,
    )
    for key in (
        "layer", "judge", "generation_sha256", "dataset_sha256", "evaluator_git_sha",
        "started_utc", "wall_seconds", "judge_calls", "judge_cost_usd",
        "per_entry", "aggregates", "n_scored", "n_failed", "failures",
    ):
        assert key in sf, key
    assert sf["generation_sha256"] == "abc123"
    assert sf["failures"] == []
    # JSON-serializable end to end
    json.dumps(sf)


def test_verify_generation_sha_passes_on_match():
    sf = {"generation_sha256": "abc"}
    verify_generation_sha(sf, "abc")  # must not raise


def test_verify_generation_sha_hard_fails_on_mismatch():
    sf = {"generation_sha256": "abc"}
    with pytest.raises(GenerationMismatchError):
        verify_generation_sha(sf, "different")
