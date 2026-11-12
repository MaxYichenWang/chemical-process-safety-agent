"""Tests for cross-framework and cross-judge agreement statistics."""

from src.evaluation.scoring.agreement import (
    cross_framework_agreement,
    judge_bias,
)


def _sf(layer, judge_model, per_entry, same_family=False):
    return {
        "layer": layer,
        "judge": {"model": judge_model, "same_family_as_pipeline": same_family},
        "per_entry": per_entry,
    }


def test_cross_framework_agreement_pairs_by_entry_and_skips_none():
    de = _sf(
        "deepeval",
        "qwen",
        {"0": {"faithfulness": 1.0}, "1": {"faithfulness": 0.5}, "2": {"faithfulness": 0.0}},
    )
    ra = _sf(
        "ragas",
        "qwen",
        # entry "2" is None in ragas -> dropped from the paired set
        {"0": {"faithfulness": 0.9}, "1": {"faithfulness": 0.4}, "2": {"faithfulness": None}},
    )
    out = cross_framework_agreement(de, ra)
    assert out["n"] == 2
    assert out["pearson"] == 1.0  # (1.0,0.9) & (0.5,0.4) are perfectly rank/linear-correlated
    assert out["deepeval_judge"] == "qwen" and out["ragas_judge"] == "qwen"


def test_cross_framework_agreement_none_when_insufficient_overlap():
    de = _sf("deepeval", "qwen", {"0": {"faithfulness": 1.0}})
    ra = _sf("ragas", "qwen", {"9": {"faithfulness": 0.9}})
    assert cross_framework_agreement(de, ra) is None


def test_judge_bias_primary_minus_secondary():
    primary = _sf(
        "ragas", "qwen", {"0": {"faithfulness": 1.0}, "1": {"faithfulness": 0.8}}
    )
    secondary = _sf(
        "ragas", "deepseek", {"0": {"faithfulness": 0.8}, "1": {"faithfulness": 0.6}}
    )
    out = judge_bias(primary, secondary, ["faithfulness"])
    assert out["primary_judge"] == "qwen" and out["secondary_judge"] == "deepseek"
    fm = out["metrics"]["faithfulness"]
    assert fm["primary_mean"] == 0.9 and fm["secondary_mean"] == 0.7
    assert fm["delta"] == 0.2 and fm["n"] == 2
