"""Shared citation extraction/matching helpers for the evaluation metrics.

Two evaluators score "did the model cite an identifier that the retrieved
context actually supports?":

* ``FaithfulnessEvaluator.evaluate_citation_accuracy`` -- OSHA standard numbers
  only (the "regex-only" variant in reports).
* ``DomainEvaluator.evaluate_citation_accuracy`` -- standards *plus*
  enforcement IDs (the "domain-aware" variant, the primary reported metric).

Both used to carry their own copy of the extraction logic, which is how the two
bugs below survived several eval runs unnoticed. Everything lives here now so
the two variants can only differ in *which identifier families they count*, not
in how matching works.

Citation normalization handles the following failure modes
--------------------------------------------------------------------------

1. **Context metadata was never scanned.** The old implementations joined only
   each context doc's ``text``. For the ``osha_regulations`` collection a
   chunk's own section number very often appears *only* in its ``source`` /
   ``chunk_id`` metadata (e.g. ``source="1910.1001 §1910.1001(j)"``) and is not
   restated in the chunk prose. That metadata string is also exactly the source
   label the pipeline shows the model, so it is exactly what a well-grounded
   answer cites back -- and the metric scored every one of those as a miss.

2. **The ``osha_enf`` corpus stores standard numbers squashed.**
   ``"19101001 J02 III"`` is that corpus's encoding of ``1910.1001(j)(2)(iii)``
   and ``"5A0001"`` its encoding of ``5(a)(1)``. A dotted-format regex can never
   match either, in ``text`` or metadata, so any answer citing a normal-looking
   standard number against enforcement evidence was structurally guaranteed to
   score a miss regardless of whether it was faithful.

Deliberate asymmetries (not bugs -- read before "simplifying" them)
------------------------------------------------------------------

* **Ancestor credit, context side only.** If the context establishes
  ``1910.1001(j)(2)(iii)``, an answer citing the parent ``1910.1001(j)`` is
  grounded, so context citations are expanded to include their ancestors. The
  reverse is *not* true (context ``1910.1001`` does not support an answer's
  ``1910.1001(j)(2)``), so response citations are never expanded. Set
  ``EVAL_CITATION_CREDIT_ANCESTORS=0`` to score strict exact matches instead.
* **Bare enforcement IDs, metadata side only.** In an answer, a bare 9-digit
  run is usually not a citation, so the response side requires a keyword
  ("accident 221198146"). In ``source``/``chunk_id`` metadata a long digit run
  *is* the identifier, so no keyword is required there.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Set

# 1910.119, 1910.119(j), 1910.119(j)(4)(i), ...
OSHA_STD_RE = re.compile(r"19[0-9]{2}\.[0-9]+(?:\([a-z0-9]+\))*")

# OSH Act General Duty Clause, cited in prose and stored as "5(a)(1)" in osha_enf.
GENERAL_DUTY_RE = re.compile(r"\b5\(a\)\(1\)")

# Accident/inspection IDs in prose: 8-12 digit runs introduced by a keyword.
ENF_ID_RE = re.compile(
    r"(?:(?:accident|inspection|citation|report|case)\s*(?:#|number|no\.?)?\s*)"
    r"(\d{8,12})",
    re.IGNORECASE,
)

# Bare 8-12 digit runs. Only ever applied to source/chunk_id metadata.
BARE_ID_RE = re.compile(r"\d{8,12}")

# osha_enf squashed standard number: 8 digits then optional paragraph tokens,
# e.g. "19101001 J02 III", "19100132 A", "19100147 D06".
_SQUASHED_STD_RE = re.compile(r"\b(\d{8})((?:\s+[A-Z]+\d*)+)?(?![\d.])")

# osha_enf squashed General Duty Clause, e.g. "5A0001".
_SQUASHED_GENDUTY_RE = re.compile(r"\b5([A-Z])(\d{4})\b")

# A paragraph token is either a roman numeral or a letter with an optional index.
_ROMAN_RE = re.compile(r"^[IVXL]+$")
_LETTER_INDEX_RE = re.compile(r"^([A-Z])(\d*)$")

# Only decode squashed numbers whose leading 4 digits are a real 29 CFR part.
# Without this, an unseparated date such as "19850325" would decode to the
# plausible-looking but nonexistent standard "1985.325".
_CFR_PARTS = frozenset(
    {"1903", "1904", "1908", "1910", "1915", "1917", "1918", "1926", "1928"}
)

_METADATA_FIELDS = ("source", "chunk_id")


def _paragraph_suffix(tokens: str) -> Optional[str]:
    """Turn ``" J02 III"`` into ``"(j)(2)(iii)"``.

    Returns ``None`` if any token is unrecognised, so a partially-parsed (and
    therefore wrong) citation is never emitted.
    """
    suffix: List[str] = []
    for token in tokens.split():
        if _ROMAN_RE.match(token):
            suffix.append(f"({token.lower()})")
            continue
        match = _LETTER_INDEX_RE.match(token)
        if not match:
            return None
        letter, index = match.groups()
        suffix.append(f"({letter.lower()})")
        if index:
            suffix.append(f"({int(index)})")
    return "".join(suffix)


def normalize_squashed_standard(number: str, tokens: Optional[str]) -> Optional[str]:
    """Convert an ``osha_enf`` squashed standard number to dotted form.

    ``("19101001", " J02 III")`` -> ``"1910.1001(j)(2)(iii)"``.
    Returns ``None`` when the input does not decode cleanly.
    """
    if len(number) != 8:
        return None
    part, section = number[:4], number[4:]
    if part not in _CFR_PARTS:
        return None
    try:
        section_number = int(section)
    except ValueError:
        return None
    if section_number == 0:
        return None
    dotted = f"{part}.{section_number}"
    if not tokens:
        return dotted
    suffix = _paragraph_suffix(tokens)
    if suffix is None:
        return None
    return dotted + suffix


def _iter_squashed_standards(text: str) -> Iterator[str]:
    for number, tokens in _SQUASHED_STD_RE.findall(text):
        normalized = normalize_squashed_standard(number, tokens or None)
        if normalized:
            yield normalized
    for letter, index in _SQUASHED_GENDUTY_RE.findall(text):
        try:
            yield f"5({letter.lower()})({int(index)})"
        except ValueError:  # pragma: no cover - regex guarantees digits
            continue


def citation_ancestors(citation: str) -> Set[str]:
    """Return the paragraph-level ancestors of a citation, excluding itself.

    ``"1910.1001(j)(2)"`` -> ``{"1910.1001", "1910.1001(j)"}``.
    """
    head, _, rest = citation.partition("(")
    if not rest:
        return set()
    groups = re.findall(r"\([a-z0-9]+\)", "(" + rest)
    ancestors = {head}
    current = head
    for group in groups[:-1]:
        current += group
        ancestors.add(current)
    return ancestors


def _doc_text(doc: Any) -> str:
    return doc.get("text", "") or "" if isinstance(doc, dict) else ""


def _doc_metadata(doc: Any) -> str:
    if not isinstance(doc, dict):
        return ""
    return " ".join(str(doc.get(field, "") or "") for field in _METADATA_FIELDS)


def extract_response_standards(response: str) -> Set[str]:
    """Standard numbers an answer claims. Never expanded or normalized."""
    response = response or ""
    citations = set(OSHA_STD_RE.findall(response))
    if GENERAL_DUTY_RE.search(response):
        citations.add("5(a)(1)")
    return citations


def extract_context_standards(
    docs: Sequence[Any], *, credit_ancestors: bool = True
) -> Set[str]:
    """Standard numbers the retrieved context supports.

    Scans doc ``text`` *and* ``source``/``chunk_id`` metadata (bug 1), decodes
    the ``osha_enf`` squashed encoding (bug 2), and -- unless disabled -- credits
    ancestor paragraphs of every citation found.
    """
    citations: Set[str] = set()
    for doc in docs or []:
        blob = f"{_doc_text(doc)} {_doc_metadata(doc)}"
        citations.update(OSHA_STD_RE.findall(blob))
        if GENERAL_DUTY_RE.search(blob):
            citations.add("5(a)(1)")
        citations.update(_iter_squashed_standards(blob))
    if credit_ancestors:
        for citation in list(citations):
            citations.update(citation_ancestors(citation))
    return citations


def extract_response_enf_ids(response: str) -> Set[str]:
    """Enforcement IDs an answer claims. Requires a keyword introducer."""
    return set(ENF_ID_RE.findall(response or ""))


def extract_context_enf_ids(docs: Sequence[Any]) -> Set[str]:
    """Enforcement IDs the retrieved context supports.

    Keyword-introduced in prose; bare digit runs are accepted in metadata, where
    a long digit run *is* the identifier (e.g. ``osha_genduty_315506691_01001``).
    """
    ids: Set[str] = set()
    for doc in docs or []:
        ids.update(ENF_ID_RE.findall(_doc_text(doc)))
        ids.update(BARE_ID_RE.findall(_doc_metadata(doc)))
    return ids


def score_citation_accuracy(
    results: Iterable[Dict[str, Any]],
    *,
    include_enforcement_ids: bool,
    credit_ancestors: Optional[bool] = None,
) -> Dict[str, Any]:
    """Fraction of an answer's cited identifiers that the context supports.

    Answers that cite nothing are skipped (not scored 0.0) -- an answer with no
    citations has no citation accuracy to measure. Returns the usual
    ``{"mean", "ci_lower", "ci_upper"}`` shape plus ``"n"`` so a reader can tell
    a real 0.0 from "nothing was scored".
    """
    if credit_ancestors is None:
        try:
            from src.evaluation.config import EVAL_CITATION_CREDIT_ANCESTORS

            credit_ancestors = EVAL_CITATION_CREDIT_ANCESTORS
        except ImportError:  # pragma: no cover - config always importable in-tree
            credit_ancestors = True

    accuracies: List[float] = []
    for result in results or []:
        response = result.get("final_response", "")
        docs = result.get("context_gathered", []) or []

        claimed = extract_response_standards(response)
        supported = extract_context_standards(docs, credit_ancestors=credit_ancestors)
        if include_enforcement_ids:
            claimed = claimed | extract_response_enf_ids(response)
            supported = supported | extract_context_enf_ids(docs)

        if not claimed:
            continue
        accuracies.append(len(claimed & supported) / len(claimed))

    if not accuracies:
        return {"mean": 0.0, "ci_lower": 0.0, "ci_upper": 0.0, "n": 0}

    from src.evaluation.utils import bootstrap_ci

    mean_val = sum(accuracies) / len(accuracies)
    lower, upper = (
        bootstrap_ci(accuracies) if len(accuracies) > 1 else (mean_val, mean_val)
    )
    return {
        "mean": mean_val,
        "ci_lower": lower,
        "ci_upper": upper,
        "n": len(accuracies),
    }
