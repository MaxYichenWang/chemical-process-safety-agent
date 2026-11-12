"""Run provenance: the ``run_metadata`` block persisted in every ``generation.json``.

``end_to_end_evaluator.py`` already computes ``wall_start``/``wall_time`` and
only logs it (see its ``__main__`` block) -- this module is what actually
persists a complete, schema-checked provenance record alongside the
generation, so every published number can be traced back to exactly which
pipeline git SHA, dependency set, provider/model/temperature per role, and
dataset it came from.
"""

from __future__ import annotations

import hashlib
import platform
import subprocess
import sys
from typing import Any, Dict, List, Optional

from src.evaluation.scoring.contract import evaluator_git_sha

# Every key the schema test enforces as required.
REQUIRED_KEYS = (
    "run_id",
    "tier",
    "dataset_path",
    "dataset_sha256",
    "include_adversarial",
    "sample_size_per_source",
    "sample_seed",
    "max_concurrent",
    "roles",
    "ablation_flags",
    "pipeline_git_sha",
    "dep_versions_sha256",
    "python_version",
    "platform",
    "wall_start_utc",
    "wall_end_utc",
    "wall_seconds",
    "pipeline_cost_usd",
    "n_entries",
    "pipeline_error_count",
)


def pip_freeze_hash() -> str:
    """Best-effort sha256 of ``pip freeze`` output -- a dependency-drift fingerprint.

    Returns ``"unknown"`` rather than raising if pip is unavailable or slow;
    provenance is best-effort, not a hard dependency of generation.
    """
    try:
        out = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        if out.returncode != 0 or not out.stdout:
            return "unknown"
        return hashlib.sha256(out.stdout.encode("utf-8")).hexdigest()
    except Exception:
        return "unknown"


def collect_role_configs(roles: List[str] = None) -> Dict[str, Dict[str, Any]]:
    """Resolve provider/model/temperature for each pipeline role.

    Lazily imports ``src.agentic_rag.llm_factory`` (langchain-dependent) so
    modules that only need the schema/shape (tests, ``--dry-run``) never pay
    for that import.
    """
    from src.agentic_rag.llm_factory import resolve_role_config

    roles = roles or ["supervisor", "worker", "reasoning", "critic", "data_analyst"]
    out: Dict[str, Dict[str, Any]] = {}
    for role in roles:
        cfg = resolve_role_config(role)
        out[role] = {
            "provider": cfg.provider,
            "model": cfg.model,
            "temperature": cfg.temperature,
            "same_family_as_pipeline": None,  # filled in by the judge's own config, not here
        }
    return out


def build_run_metadata(
    *,
    run_id: str,
    tier: str,
    dataset_path: str,
    dataset_sha256: str,
    include_adversarial: bool,
    sample_size_per_source: Optional[int],
    sample_seed: Optional[int],
    max_concurrent: int,
    roles: Dict[str, Dict[str, Any]],
    ablation_flags: Dict[str, bool],
    wall_start_utc: str,
    wall_end_utc: str,
    wall_seconds: float,
    pipeline_cost_usd: float,
    n_entries: int,
    pipeline_error_count: int,
    dep_versions_sha256: Optional[str] = None,
    system: str = "full",
) -> Dict[str, Any]:
    """Assemble the ``run_metadata`` block. See :data:`REQUIRED_KEYS` for the schema.

    ``system`` records which system produced the generation ("full" for the
    multi-agent graph, or a Stage-E comparator baseline). It is additive and not
    part of :data:`REQUIRED_KEYS`, so archived generations without it still
    validate.
    """
    metadata = {
        "run_id": run_id,
        "tier": tier,
        "system": system,
        "dataset_path": dataset_path,
        "dataset_sha256": dataset_sha256,
        "include_adversarial": include_adversarial,
        "sample_size_per_source": sample_size_per_source,
        "sample_seed": sample_seed,
        "max_concurrent": max_concurrent,
        "roles": roles,
        "ablation_flags": ablation_flags,
        "pipeline_git_sha": evaluator_git_sha(),
        "dep_versions_sha256": (
            dep_versions_sha256
            if dep_versions_sha256 is not None
            else pip_freeze_hash()
        ),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "wall_start_utc": wall_start_utc,
        "wall_end_utc": wall_end_utc,
        "wall_seconds": wall_seconds,
        "pipeline_cost_usd": pipeline_cost_usd,
        "n_entries": n_entries,
        "pipeline_error_count": pipeline_error_count,
    }
    missing = [k for k in REQUIRED_KEYS if k not in metadata]
    assert (
        not missing
    ), f"build_run_metadata is missing required keys: {missing}"  # pragma: no cover
    return metadata


def validate_run_metadata(metadata: Dict[str, Any]) -> List[str]:
    """Return a list of missing required keys (empty = valid). Used by the schema test."""
    return [k for k in REQUIRED_KEYS if k not in metadata]
