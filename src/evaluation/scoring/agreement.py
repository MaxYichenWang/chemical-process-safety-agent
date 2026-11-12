"""Cross-framework and cross-judge agreement statistics.

Pure functions of score files -- no judge, no key, no corpus -- so
``aggregate`` can compute them offline. Correlations are computed with a dependency-light Pearson/Spearman so
this module imports on a machine without scipy.
"""

from __future__ import annotations

from math import isnan
from typing import Any, Dict, List, Optional, Tuple


def _pearson(a: List[float], b: List[float]) -> float:
    n = len(a)
    if n < 2:
        return float("nan")
    ma = sum(a) / n
    mb = sum(b) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    den = (sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b)) ** 0.5
    return num / den if den else float("nan")


def _rank(values: List[float]) -> List[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def _spearman(a: List[float], b: List[float]) -> float:
    if len(a) < 2:
        return float("nan")
    return _pearson(_rank(a), _rank(b))


def _metric_by_entry(score_file: Dict[str, Any], metric: str) -> Dict[str, float]:
    """Map ``entry_id -> metric value`` from a score file's ``per_entry`` block,
    skipping entries where the metric is missing or ``None``."""
    out: Dict[str, float] = {}
    for entry_id, entry in (score_file.get("per_entry") or {}).items():
        if not isinstance(entry, dict):
            continue
        val = entry.get(metric)
        if isinstance(val, (int, float)) and not isnan(float(val)):
            out[str(entry_id)] = float(val)
    return out


def _paired(
    a: Dict[str, float], b: Dict[str, float]
) -> Tuple[List[float], List[float]]:
    keys = sorted(set(a) & set(b), key=lambda k: (len(k), k))
    return [a[k] for k in keys], [b[k] for k in keys]


def cross_framework_agreement(
    deepeval_score_file: Dict[str, Any],
    ragas_score_file: Dict[str, Any],
    metric: str = "faithfulness",
) -> Optional[Dict[str, Any]]:
    """Pearson/Spearman between DeepEval and RAGAS per-entry ``metric`` scores.

    Returns ``None`` when either layer lacks per-entry data or fewer than two
    entries overlap (so the caller renders nothing rather than a fake 0.0)."""
    de = _metric_by_entry(deepeval_score_file, metric)
    ra = _metric_by_entry(ragas_score_file, metric)
    xs, ys = _paired(de, ra)
    if len(xs) < 2:
        return None
    pearson = _pearson(xs, ys)
    spearman = _spearman(xs, ys)
    return {
        "metric": metric,
        "pearson": None if isnan(pearson) else round(pearson, 4),
        "spearman": None if isnan(spearman) else round(spearman, 4),
        "n": len(xs),
        "deepeval_mean": round(sum(xs) / len(xs), 4),
        "ragas_mean": round(sum(ys) / len(ys), 4),
        "deepeval_judge": (deepeval_score_file.get("judge") or {}).get("model"),
        "ragas_judge": (ragas_score_file.get("judge") or {}).get("model"),
    }


def judge_bias(
    primary_score_file: Dict[str, Any],
    secondary_score_file: Dict[str, Any],
    metrics: List[str],
) -> Optional[Dict[str, Any]]:
    """Per-metric mean delta (primary - secondary) between two judges scoring the
    same generation with the same framework -- the self-preference-bias
    measurement. Deltas are over the entries both judges scored."""
    layer = primary_score_file.get("layer")
    primary_judge = (primary_score_file.get("judge") or {}).get("model")
    secondary_judge = (secondary_score_file.get("judge") or {}).get("model")
    deltas: Dict[str, Any] = {}
    for metric in metrics:
        pa = _metric_by_entry(primary_score_file, metric)
        sb = _metric_by_entry(secondary_score_file, metric)
        xs, ys = _paired(pa, sb)
        if len(xs) < 1:
            continue
        pm = sum(xs) / len(xs)
        sm = sum(ys) / len(ys)
        deltas[metric] = {
            "primary_mean": round(pm, 4),
            "secondary_mean": round(sm, 4),
            "delta": round(pm - sm, 4),
            "n": len(xs),
        }
    if not deltas:
        return None
    return {
        "layer": layer,
        "primary_judge": primary_judge,
        "secondary_judge": secondary_judge,
        "metrics": deltas,
    }
