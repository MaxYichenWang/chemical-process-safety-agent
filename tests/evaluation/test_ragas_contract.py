"""Contract tests against the *real* installed ragas package.

The unit tests for the evaluators use stubs, which is what let a whole
migration ship with wrong ``ascore`` signatures and a wrong execution entry
point: the stubs were written to match the code rather than the library. These
tests introspect the actually-installed ragas classes, so a signature change on
upgrade fails here instead of mid-run at T2.

Skipped entirely when ragas is not installed (e.g. the dev machine).
"""

from __future__ import annotations

import inspect
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version

import pytest

try:
    version("ragas")
except PackageNotFoundError:
    pytest.skip("ragas not installed", allow_module_level=True)

# An installed package with a broken dependency must fail collection.
ragas_collections = import_module("ragas.metrics.collections")


def _params(cls, method: str) -> set:
    return {
        name
        for name in inspect.signature(getattr(cls, method)).parameters
        if name not in ("self", "args", "kwargs")
    }


# (class name, method, kwargs the evaluators pass)
CALL_SITES = [
    ("Faithfulness", "ascore", {"user_input", "response", "retrieved_contexts"}),
    ("AnswerCorrectness", "ascore", {"user_input", "response", "reference"}),
    ("FactualCorrectness", "ascore", {"response", "reference"}),
    (
        "ContextPrecisionWithReference",
        "ascore",
        {"user_input", "reference", "retrieved_contexts"},
    ),
    ("ContextRecall", "ascore", {"user_input", "retrieved_contexts", "reference"}),
    ("AnswerRelevancy", "ascore", {"user_input", "response"}),
]

CONSTRUCTORS = [
    ("Faithfulness", {"llm"}),
    ("AnswerCorrectness", {"llm", "embeddings"}),
    ("FactualCorrectness", {"llm"}),
    ("ContextPrecisionWithReference", {"llm"}),
    ("ContextRecall", {"llm"}),
    ("AnswerRelevancy", {"llm", "embeddings"}),
]


@pytest.mark.parametrize("class_name,method,passed", CALL_SITES)
def test_ascore_signature_matches_call_site(class_name, method, passed):
    cls = getattr(ragas_collections, class_name, None)
    if cls is None:
        pytest.skip(f"{class_name} not in this ragas build")
    accepted = _params(cls, method)
    unexpected = passed - accepted
    assert not unexpected, (
        f"{class_name}.{method}() does not accept {sorted(unexpected)}; "
        f"it accepts {sorted(accepted)}"
    )
    required = {
        name
        for name, param in inspect.signature(getattr(cls, method)).parameters.items()
        if name != "self" and param.default is inspect.Parameter.empty
        and param.kind
        in (param.POSITIONAL_OR_KEYWORD, param.KEYWORD_ONLY)
    }
    missing = required - passed
    assert not missing, f"{class_name}.{method}() requires {sorted(missing)}"


@pytest.mark.parametrize("class_name,passed", CONSTRUCTORS)
def test_constructor_required_args_are_supplied(class_name, passed):
    cls = getattr(ragas_collections, class_name, None)
    if cls is None:
        pytest.skip(f"{class_name} not in this ragas build")
    required = {
        name
        for name, param in inspect.signature(cls.__init__).parameters.items()
        if name != "self" and param.default is inspect.Parameter.empty
        and param.kind
        in (param.POSITIONAL_OR_KEYWORD, param.KEYWORD_ONLY)
    }
    missing = required - passed
    assert not missing, (
        f"{class_name}.__init__ requires {sorted(missing)} but the evaluator "
        f"passes only {sorted(passed)}"
    )


def test_evaluators_do_not_use_the_experiment_api():
    """``@experiment``/``arun`` needs a Dataset+backend; direct ascore does not.

    Guards against reintroducing the wrapper whose ``__call__(rows)`` misuse
    broke every real run while the mocked tests passed.
    """
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[2] / "src" / "evaluation"
    offenders = [
        path.name
        for path in root.rglob("*.py")
        if "from ragas import experiment" in path.read_text()
    ]
    assert not offenders, f"ragas experiment API reintroduced in: {offenders}"
