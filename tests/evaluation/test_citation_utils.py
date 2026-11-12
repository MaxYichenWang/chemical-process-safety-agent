"""Tests for the shared citation extraction/matching helpers.

Cases cover citation metadata, enforcement encodings, paragraph ancestry,
and source matching.
"""

import pytest

from src.evaluation.citation_utils import (
    citation_ancestors,
    extract_context_enf_ids,
    extract_context_standards,
    extract_response_standards,
    normalize_squashed_standard,
    score_citation_accuracy,
)


class TestSquashedStandardNormalization:
    @pytest.mark.parametrize(
        "number,tokens,expected",
        [
            ("19101001", " J02 III", "1910.1001(j)(2)(iii)"),
            ("19100132", " A", "1910.132(a)"),
            ("19100023", " C01", "1910.23(c)(1)"),
            ("19100212", " A03 II", "1910.212(a)(3)(ii)"),
            ("19100147", " D06", "1910.147(d)(6)"),
            ("19101025", None, "1910.1025"),
        ],
    )
    def test_decodes_osha_enf_encoding(self, number, tokens, expected):
        assert normalize_squashed_standard(number, tokens) == expected

    def test_rejects_unparseable_tokens(self):
        assert normalize_squashed_standard("19101001", " ZZ99X") is None

    def test_rejects_unknown_cfr_part_and_wrong_length(self):
        assert normalize_squashed_standard("29101001", None) is None
        assert normalize_squashed_standard("1910100", None) is None

    def test_does_not_decode_an_unseparated_date(self):
        """"19850325" (1985-03-25) must not become the fake standard 1985.325."""
        assert normalize_squashed_standard("19850325", None) is None
        assert "1985.325" not in extract_context_standards(
            [{"text": "accident 14217376 on 19850325"}]
        )

    def test_general_duty_squashed_form_is_recognized(self):
        assert "5(a)(1)" in extract_context_standards([{"source": "5A0001"}])


class TestAncestorCredit:
    def test_ancestors_of_paragraph_citation(self):
        assert citation_ancestors("1910.1001(j)(2)(iii)") == {
            "1910.1001",
            "1910.1001(j)",
            "1910.1001(j)(2)",
        }

    def test_section_only_citation_has_no_ancestors(self):
        assert citation_ancestors("1910.119") == set()

    def test_context_child_supports_response_parent(self):
        results = [
            {
                "final_response": "See 1910.1001(j).",
                "context_gathered": [{"text": "per 1910.1001(j)(2)(iii) ..."}],
            }
        ]
        assert score_citation_accuracy(
            results, include_enforcement_ids=False
        )["mean"] == 1.0

    def test_response_child_is_not_supported_by_context_parent(self):
        """Ancestor credit is context-side only; over-precision still misses."""
        results = [
            {
                "final_response": "See 1910.1001(j)(2).",
                "context_gathered": [{"text": "per 1910.1001 ..."}],
            }
        ]
        assert score_citation_accuracy(
            results, include_enforcement_ids=False
        )["mean"] == 0.0

    def test_can_be_disabled_for_strict_scoring(self):
        results = [
            {
                "final_response": "See 1910.1001(j).",
                "context_gathered": [{"text": "per 1910.1001(j)(2)(iii) ..."}],
            }
        ]
        strict = score_citation_accuracy(
            results, include_enforcement_ids=False, credit_ancestors=False
        )
        assert strict["mean"] == 0.0


class TestContextMetadata:
    def test_section_number_only_in_source_metadata_counts(self):
        docs = [
            {
                "text": "Communication of hazards to employees.",
                "source": "1910.1053 §1910.1053(g)(2)",
                "chunk_id": "osha_reg_1910.1053(g)(2)",
            }
        ]
        assert "1910.1053(g)(2)" in extract_context_standards(docs)

    def test_squashed_enf_standard_in_text_counts(self):
        docs = [{"text": "Citation 19101001 J02 was issued.", "source": ""}]
        assert "1910.1001(j)(2)" in extract_context_standards(docs)

    def test_bare_enforcement_id_in_chunk_id_counts(self):
        docs = [{"text": "", "chunk_id": "osha_genduty_315506691_01001"}]
        assert "315506691" in extract_context_enf_ids(docs)

    def test_bare_digits_in_a_response_are_not_treated_as_citations(self):
        assert extract_response_standards("The plant processed 315506691 units") == set()


class TestScoring:
    def test_answers_without_citations_are_skipped_not_zeroed(self):
        out = score_citation_accuracy(
            [{"final_response": "Safety matters.", "context_gathered": [{"text": "1910.119"}]}],
            include_enforcement_ids=False,
        )
        assert out["n"] == 0
        assert out["mean"] == 0.0

    def test_partial_credit(self):
        out = score_citation_accuracy(
            [
                {
                    "final_response": "See 1910.119 and 1910.120.",
                    "context_gathered": [{"text": "1910.119 details PSM."}],
                }
            ],
            include_enforcement_ids=False,
        )
        assert out["mean"] == 0.5
        assert out["n"] == 1

    def test_enforcement_ids_only_counted_in_domain_variant(self):
        results = [
            {
                "final_response": "In accident 221198146 the employee fell.",
                "context_gathered": [{"text": "", "chunk_id": "osha_acc_221198146_1"}],
            }
        ]
        assert score_citation_accuracy(results, include_enforcement_ids=True)["mean"] == 1.0
        # The regex-only variant sees no OSHA standard, so nothing to score.
        assert score_citation_accuracy(results, include_enforcement_ids=False)["n"] == 0
