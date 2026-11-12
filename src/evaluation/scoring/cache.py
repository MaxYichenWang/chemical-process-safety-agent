"""Per-entry resumable scoring cache.

Judge-dependent scoring (Stage 3 DeepEval, Stage 4 RAGAS) is the expensive,
slow part of an eval run -- a crash partway through a multi-hour scoring pass
should cost re-scoring the unfinished entries, not the whole dataset (or
worse, the whole 11.5-hour pipeline run). This cache persists one JSON blob
per ``(generation_sha, entry_id, metric, judge_id)`` so ``run.py score`` can
resume after interruption by only (re-)scoring what's missing.

Not used by the deterministic layer (:mod:`src.evaluation.scoring.deterministic`):
that layer is free and instantaneous, so there is nothing to resume.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


class ScoringCache:
    """Flat JSON-backed cache keyed by ``(generation_sha, entry_id, metric, judge_id)``."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._lock = threading.Lock()
        self._data: Dict[str, Any] = {}
        if self.path.exists():
            try:
                self._data = json.loads(self.path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                # Corrupt/partial cache file (e.g. killed mid-write before the
                # atomic replace below existed) -- treat as empty rather than
                # blocking a resume.
                self._data = {}

    @staticmethod
    def _key(generation_sha: str, entry_id: str, metric: str, judge_id: str) -> str:
        return f"{generation_sha}\x1f{entry_id}\x1f{metric}\x1f{judge_id}"

    def get(
        self, generation_sha: str, entry_id: str, metric: str, judge_id: str
    ) -> Optional[Any]:
        with self._lock:
            return self._data.get(self._key(generation_sha, entry_id, metric, judge_id))

    def has(
        self, generation_sha: str, entry_id: str, metric: str, judge_id: str
    ) -> bool:
        with self._lock:
            return self._key(generation_sha, entry_id, metric, judge_id) in self._data

    def put(
        self, generation_sha: str, entry_id: str, metric: str, judge_id: str, value: Any
    ) -> None:
        with self._lock:
            self._data[self._key(generation_sha, entry_id, metric, judge_id)] = value
            self._flush()

    def has_all(
        self, generation_sha: str, entry_id: str, metrics: Iterable[str], judge_id: str
    ) -> bool:
        return all(self.has(generation_sha, entry_id, m, judge_id) for m in metrics)

    def pending_entry_ids(
        self,
        entry_ids: Iterable[str],
        generation_sha: str,
        metrics: Iterable[str],
        judge_id: str,
    ) -> list:
        """Entry ids missing at least one of *metrics* in the cache -- what a
        resumed ``score`` run still needs to (re-)compute."""
        metrics = list(metrics)
        return [
            eid
            for eid in entry_ids
            if not self.has_all(generation_sha, eid, metrics, judge_id)
        ]

    def _flush(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(self._data), encoding="utf-8")
        tmp.replace(self.path)  # atomic on POSIX + Windows (same filesystem)
