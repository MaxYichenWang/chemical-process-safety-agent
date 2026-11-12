"""Score-file integrity contract shared by every scoring stage.

Every score file (``scores/deterministic.json``, ``scores/deepeval.<judge>.json``,
``scores/ragas.<judge>.json``) carries a ``generation_sha256`` computed by
``sha256_json`` from the parsed generation, and a digest of the effective
dataset in row order. Whitespace and gzip compression do not change these
canonical JSON digests. Aggregation rejects scores from another generation
or dataset revision before writing a report.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


class GenerationMismatchError(Exception):
    """Raised when a score file's ``generation_sha256`` disagrees with the
    generation it is being merged into."""


class DatasetMismatchError(ValueError):
    """Raised when a dataset differs from the generation or score artifacts."""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    """Hash of a file's exact on-disk bytes."""
    return sha256_bytes(Path(path).read_bytes())


def sha256_json(obj: Any) -> str:
    """Hash of a canonical (sorted-keys) JSON encoding of *obj*.

    Used for the resolved dataset (a Python list assembled in memory, e.g.
    after appending the adversarial split), where there is no single
    on-disk file to hash directly.
    """
    canonical = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
    return sha256_bytes(canonical.encode("utf-8"))


def evaluator_git_sha() -> str:
    """Best-effort ``git rev-parse HEAD`` of the current checkout.

    Returns ``"unknown"`` rather than raising if git is unavailable or this
    isn't a git checkout -- provenance is best-effort, not a hard dependency.
    """
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        sha = out.stdout.strip()
        return sha if out.returncode == 0 and sha else "unknown"
    except Exception:
        return "unknown"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sanitize_judge_id(model: Optional[str]) -> str:
    """Filesystem-safe judge id for score-file naming (``qwen3.7-flash`` -> ``qwen3-7-flash``)."""
    if not model:
        return "none"
    return model.replace(".", "-").replace("/", "-").replace(":", "-")


def sanitize_tag(tag: Optional[str]) -> str:
    """Filesystem-safe variant tag for non-clobbering re-scores (``temp=0.0`` -> ``temp-0-0``)."""
    if not tag:
        return ""
    safe = "".join(c if (c.isalnum() or c in "-_") else "-" for c in str(tag))
    return safe.strip("-") or "tag"


def build_score_file(
    *,
    layer: str,
    generation_sha256: str,
    dataset_sha256: str,
    per_entry: Dict[str, Any],
    aggregates: Dict[str, Any],
    started_utc: str,
    wall_seconds: float,
    n_scored: int,
    n_failed: int,
    failures: Optional[List[Dict[str, Any]]] = None,
    judge: Optional[Dict[str, Any]] = None,
    judge_calls: int = 0,
    judge_cost_usd: float = 0.0,
    evaluator_git_sha_override: Optional[str] = None,
    tag: Optional[str] = None,
) -> Dict[str, Any]:
    """Assemble a score file matching the integrity contract exactly.

    ``tag`` is an optional variant label (e.g. ``temp0``) that lets the same
    ``(layer, judge)`` be re-scored under different settings without either
    clobbering the previous score file on disk or overwriting its slot in
    ``results.json`` -- ``aggregate`` folds ``tag`` into its per-layer key.
    """
    return {
        "layer": layer,
        "tag": tag,
        "judge": judge,
        "generation_sha256": generation_sha256,
        "dataset_sha256": dataset_sha256,
        "evaluator_git_sha": evaluator_git_sha_override or evaluator_git_sha(),
        "started_utc": started_utc,
        "wall_seconds": wall_seconds,
        "judge_calls": judge_calls,
        "judge_cost_usd": judge_cost_usd,
        "per_entry": per_entry,
        "aggregates": aggregates,
        "n_scored": n_scored,
        "n_failed": n_failed,
        "failures": failures or [],
    }


def verify_generation_sha(
    score_file: Dict[str, Any], generation_sha256: str, *, source: str = ""
) -> None:
    """Raise :class:`GenerationMismatchError` if *score_file* doesn't match *generation_sha256*.

    This is the hard-fail that guarantees a score file computed against a
    different (or since-regenerated) ``generation.json`` must never be
    silently merged into ``results.json``.
    """
    got = score_file.get("generation_sha256")
    if got != generation_sha256:
        where = f" ({source})" if source else ""
        raise GenerationMismatchError(
            f"generation_sha256 mismatch{where}: score file was computed against "
            f"{got!r}, current generation is {generation_sha256!r}. Re-run scoring "
            "against the current generation before aggregating."
        )
