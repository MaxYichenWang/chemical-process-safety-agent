"""Schema test for the ``run_metadata`` block persisted in every
``generation.json``. See ``src.evaluation.provenance`` for the full
rationale -- this test enforces the acceptance criterion verbatim: "Every new
generation has a complete block; schema test enforces required keys."
"""

from src.evaluation.provenance import (
    REQUIRED_KEYS,
    build_run_metadata,
    validate_run_metadata,
)

_FULL_KWARGS = dict(
    run_id="test_run",
    tier="T0",
    dataset_path="data/evaluation/eval_dataset.json",
    dataset_sha256="a" * 64,
    include_adversarial=False,
    sample_size_per_source=2,
    sample_seed=42,
    max_concurrent=3,
    roles={"supervisor": {"provider": "deepseek", "model": "deepseek-v4-flash", "temperature": 0.0}},
    ablation_flags={"CRITIC_ENABLED": True, "WORKER_QUERY_REFORMULATION_ENABLED": True},
    wall_start_utc="2026-08-01T00:00:00Z",
    wall_end_utc="2026-08-01T00:01:00Z",
    wall_seconds=60.0,
    pipeline_cost_usd=0.05,
    n_entries=13,
    pipeline_error_count=0,
)


def test_build_run_metadata_produces_every_required_key():
    metadata = build_run_metadata(**_FULL_KWARGS)
    missing = validate_run_metadata(metadata)
    assert missing == [], f"missing required run_metadata keys: {missing}"
    for key in REQUIRED_KEYS:
        assert key in metadata


def test_build_run_metadata_fills_derived_fields_not_passed_explicitly():
    """pipeline_git_sha/dep_versions_sha256/python_version/platform are
    computed internally, not accepted as kwargs -- confirm they land in the
    output rather than being silently absent."""
    metadata = build_run_metadata(**_FULL_KWARGS)
    for key in ("pipeline_git_sha", "dep_versions_sha256", "python_version", "platform"):
        assert metadata[key] is not None
        assert metadata[key] != ""


def test_build_run_metadata_dep_versions_sha256_override_is_honored():
    """cmd_generate can pass a precomputed dep hash (e.g. cached across a
    batch) instead of paying pip freeze's cost on every call."""
    metadata = build_run_metadata(**_FULL_KWARGS, dep_versions_sha256="deadbeef")
    assert metadata["dep_versions_sha256"] == "deadbeef"


def test_validate_run_metadata_reports_each_missing_key():
    incomplete = {"run_id": "x", "tier": "T0"}
    missing = validate_run_metadata(incomplete)
    assert "run_id" not in missing
    assert "tier" not in missing
    assert "dataset_sha256" in missing
    assert "pipeline_git_sha" in missing
    assert set(missing) == set(REQUIRED_KEYS) - {"run_id", "tier"}


def test_validate_run_metadata_empty_dict_reports_all_keys():
    assert set(validate_run_metadata({})) == set(REQUIRED_KEYS)


def test_required_keys_covers_d15_acceptance_criterion_fields():
    """Spot-check the fields the acceptance criterion calls out by name:
    per-role provider/model/temperature, pipeline git SHA, dataset sha256,
    tier, concurrency, dep versions hash, UTC start/end, wall-clock, cost."""
    expected_subset = {
        "roles",
        "pipeline_git_sha",
        "dataset_sha256",
        "tier",
        "max_concurrent",
        "dep_versions_sha256",
        "wall_start_utc",
        "wall_end_utc",
        "wall_seconds",
        "pipeline_cost_usd",
    }
    assert expected_subset.issubset(set(REQUIRED_KEYS))
