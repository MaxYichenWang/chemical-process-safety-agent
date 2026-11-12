"""dataset lint: the shipped evaluation dataset files must validate
against the live schema / agent registry, and the honesty
conventions must hold.

This is the CI guard the plan calls for: every ``expected_agents`` value must
be a real agent name (or empty for refusal entries), and every adversarial
entry must follow the refusal / no-retrieval convention with the L1-exclusion
flag set.
"""

import json
from pathlib import Path

import pytest

from src.evaluation.schemas import validate_eval_entry

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "tests" / "fixtures"

MAIN_DATASET = DATA_DIR / "sample_eval_dataset.json"
ADVERSARIAL_DATASET = DATA_DIR / "sample_adversarial.json"

DATASET_FILES = [MAIN_DATASET, ADVERSARIAL_DATASET]


def _load(path: Path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


@pytest.mark.parametrize("path", DATASET_FILES, ids=lambda p: p.name)
def test_shipped_dataset_entries_validate(path):
    """Every entry in every shipped dataset file passes schema validation
    (this includes the expected_agents-against-registry check)."""
    entries = _load(path)
    problems = []
    for idx, entry in enumerate(entries):
        errs = validate_eval_entry(entry)
        if errs:
            problems.append((idx, errs))
    assert not problems, f"{path.name} has invalid entries: {problems[:5]}"


@pytest.mark.parametrize("path", DATASET_FILES, ids=lambda p: p.name)
def test_no_unknown_expected_agents(path):
    """No entry names an agent outside the registry (guards the D1/ defect
    where adversarial entries were labelled with a bogus routing target)."""
    entries = _load(path)
    offenders = [
        (i, e.get("expected_agents"))
        for i, e in enumerate(entries)
        if any("Unknown agent" in err for err in validate_eval_entry(e))
    ]
    assert not offenders, f"{path.name} names unknown agents: {offenders[:5]}"


def test_adversarial_entries_follow_refusal_convention():
    """adversarial entries expect a refusal, not a retrieval agent.

    They must carry no ``expected_agents`` (empty list) and be flagged
    ``l1_applicable: false`` so they never route or count toward
    retrieval denominators.
    """
    entries = _load(ADVERSARIAL_DATASET)
    assert entries, "adversarial dataset is empty"
    for idx, e in enumerate(entries):
        assert e.get("expected_agents", []) == [], (
            f"adversarial[{idx}] must expect no agents (refusal), "
            f"got {e.get('expected_agents')!r}"
        )
        assert e.get("l1_applicable") is False, (
            f"adversarial[{idx}] must set l1_applicable=false"
        )


def test_zero_gt_entries_are_flagged_l1_not_applicable():
    """any entry with no ground-truth chunks must be explicitly excluded
    from Layer-1 retrieval metrics via ``l1_applicable: false`` — otherwise it
    silently scores 0% and depresses the denominator."""
    for path in DATASET_FILES:
        entries = _load(path)
        leaked = [
            i
            for i, e in enumerate(entries)
            if not e.get("ground_truth_chunks")
            and e.get("l1_applicable") is not False
        ]
        assert not leaked, (
            f"{path.name}: zero-GT entries missing l1_applicable=false: {leaked[:10]}"
        )
