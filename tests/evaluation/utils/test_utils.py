"""
Tests for src.evaluation.utils.

Covers all six utility functions with normal, edge-case, and error scenarios.
No live API calls or external service dependencies.
"""

import json
import logging
import re

import pytest

from src.evaluation.utils import (
    bootstrap_ci,
    configure_evaluation_file_logging,
    default_eval_log_path,
    extract_entity_names,
    extract_osha_standards,
    generate_run_id,
    load_eval_dataset,
    normalize_graph_state_for_eval,
    save_eval_results,
)


# ============================================================================
# load_eval_dataset
# ============================================================================


class TestLoadEvalDataset:
    """Tests for load_eval_dataset."""

    def _write_json(self, tmp_path, data):
        path = tmp_path / "dataset.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        return path

    def test_valid_dataset(self, tmp_path):
        """A well-formed dataset should load without errors."""
        entries = [
            {
                "query": "What does 1910.119 require?",
                "expected_answer": "PSM requirements.",
                "ground_truth_chunks": ["chunk_1"],
            },
            {
                "query": "Describe the West Fertilizer incident.",
                "expected_answer": "Ammonium nitrate explosion.",
                "ground_truth_chunks": ["chunk_2", "chunk_3"],
                "extra_field": "allowed",
            },
        ]
        path = self._write_json(tmp_path, entries)
        result = load_eval_dataset(path)
        assert len(result) == 2
        assert result[0]["query"] == "What does 1910.119 require?"

    def test_missing_required_key_raises(self, tmp_path):
        """Entries missing a required key should raise ValueError."""
        entries = [{"query": "hello", "expected_answer": "world"}]
        path = self._write_json(tmp_path, entries)
        with pytest.raises(ValueError, match="ground_truth_chunks"):
            load_eval_dataset(path)

    def test_not_a_list_raises(self, tmp_path):
        """Top-level JSON that is not a list should raise ValueError."""
        path = self._write_json(tmp_path, {"query": "oops"})
        with pytest.raises(ValueError, match="JSON array"):
            load_eval_dataset(path)

    def test_malformed_json_raises(self, tmp_path):
        """Broken JSON should raise json.JSONDecodeError."""
        path = tmp_path / "bad.json"
        path.write_text("{not valid json", encoding="utf-8")
        with pytest.raises(json.JSONDecodeError):
            load_eval_dataset(path)

    def test_file_not_found_raises(self, tmp_path):
        """A non-existent file should raise FileNotFoundError."""
        path = tmp_path / "missing.json"
        with pytest.raises(FileNotFoundError):
            load_eval_dataset(path)

    def test_empty_list(self, tmp_path):
        """An empty list is technically valid (no entries to validate)."""
        path = self._write_json(tmp_path, [])
        result = load_eval_dataset(path)
        assert result == []


# ============================================================================
# save_eval_results
# ============================================================================


class TestSaveEvalResults:
    """Tests for save_eval_results."""

    def test_round_trip(self, tmp_path, monkeypatch):
        """Saved results should be loadable and match the original dict."""
        monkeypatch.setattr(
            "src.evaluation.utils.EVAL_RESULTS_DIR", tmp_path / "results"
        )
        data = {"metric_a": 0.95, "metric_b": 0.80}
        output_path = save_eval_results(data, "run_001")

        assert output_path.exists()
        assert output_path.name == "run_001_results.json"

        loaded = json.loads(output_path.read_text(encoding="utf-8"))
        assert loaded == data

    def test_creates_directory(self, tmp_path, monkeypatch):
        """The target directory should be created if it does not exist."""
        target = tmp_path / "deep" / "nested" / "results"
        monkeypatch.setattr("src.evaluation.utils.EVAL_RESULTS_DIR", target)
        output_path = save_eval_results({"ok": True}, "run_002")
        assert output_path.exists()
        assert target.is_dir()

    def test_overwrites_existing_file(self, tmp_path, monkeypatch):
        """Writing to the same run_id should overwrite the previous file."""
        monkeypatch.setattr("src.evaluation.utils.EVAL_RESULTS_DIR", tmp_path)
        save_eval_results({"version": 1}, "run_003")
        save_eval_results({"version": 2}, "run_003")
        loaded = json.loads(
            (tmp_path / "run_003_results.json").read_text(encoding="utf-8")
        )
        assert loaded["version"] == 2


# ============================================================================
# bootstrap_ci
# ============================================================================


class TestBootstrapCI:
    """Tests for bootstrap_ci."""

    def test_known_distribution(self):
        """CI for a uniform-ish list should bracket the true mean."""
        scores = [0.8, 0.85, 0.9, 0.95, 1.0]
        lower, upper = bootstrap_ci(scores, n_iterations=2000, confidence=0.95)
        true_mean = sum(scores) / len(scores)
        assert lower <= true_mean <= upper

    def test_identical_values(self):
        """If all scores are equal, CI should collapse to that value."""
        scores = [0.5, 0.5, 0.5, 0.5]
        lower, upper = bootstrap_ci(scores, n_iterations=500)
        assert lower == pytest.approx(0.5)
        assert upper == pytest.approx(0.5)

    def test_single_value(self):
        """A single score should still produce a valid (degenerate) CI."""
        lower, upper = bootstrap_ci([0.7], n_iterations=500)
        assert lower == pytest.approx(0.7)
        assert upper == pytest.approx(0.7)

    def test_empty_raises(self):
        """An empty list should raise ValueError."""
        with pytest.raises(ValueError, match="empty"):
            bootstrap_ci([])

    def test_ci_bounds_order(self):
        """Lower bound should always be <= upper bound."""
        scores = [0.1, 0.5, 0.9, 0.3, 0.7, 0.6]
        lower, upper = bootstrap_ci(scores, n_iterations=1000)
        assert lower <= upper

    def test_wider_ci_with_lower_confidence(self):
        """A 99 % CI should be at least as wide as a 90 % CI."""
        scores = [0.2, 0.4, 0.6, 0.8, 1.0, 0.3, 0.5, 0.7, 0.9]
        l90, u90 = bootstrap_ci(scores, n_iterations=2000, confidence=0.90)
        l99, u99 = bootstrap_ci(scores, n_iterations=2000, confidence=0.99)
        width_90 = u90 - l90
        width_99 = u99 - l99
        assert width_99 >= width_90 - 0.01  # small tolerance for randomness


# ============================================================================
# extract_osha_standards
# ============================================================================


class TestExtractOshaStandards:
    """Tests for extract_osha_standards."""

    def test_simple_standard(self):
        """Should match a bare standard number like 1910.119."""
        result = extract_osha_standards("See OSHA 1910.119 for PSM.")
        assert "1910.119" in result

    def test_standard_with_subsections(self):
        """Should match standards with subsection references."""
        text = "Per 1910.119(j)(4)(i), employers must ..."
        result = extract_osha_standards(text)
        assert "1910.119(j)(4)(i)" in result

    def test_cfr_prefix(self):
        """Should match 29 CFR prefixed standards."""
        text = "Under 29 CFR 1910.147, lockout/tagout ..."
        result = extract_osha_standards(text)
        assert any("1910.147" in r for r in result)

    def test_multiple_standards(self):
        """Should extract all distinct standards from a text."""
        text = (
            "Standards 1910.119 and 1910.146 apply. "
            "Also see 1910.1200(h)."
        )
        result = extract_osha_standards(text)
        nums = [r.split("CFR ")[-1] if "CFR" in r else r for r in result]
        assert len(nums) >= 3

    def test_no_match(self):
        """Text without OSHA standards should return an empty list."""
        assert extract_osha_standards("No standards here.") == []

    def test_deduplication(self):
        """Repeated standard references should appear only once."""
        text = "1910.119 requires ... per 1910.119 the employer must ..."
        result = extract_osha_standards(text)
        assert result.count("1910.119") == 1

    def test_does_not_match_non_osha_numbers(self):
        """Numbers that don't look like OSHA standards should be ignored."""
        text = "The population is 1910119 and the year is 2026."
        result = extract_osha_standards(text)
        assert len(result) == 0


# ============================================================================
# extract_entity_names
# ============================================================================


class TestExtractEntityNames:
    """Tests for extract_entity_names."""

    def test_chemical_extraction(self):
        """Known chemical keywords should be found in text."""
        text = "The release involved ammonia and chlorine gas."
        result = extract_entity_names(text, "chemical")
        assert "ammonia" in result
        assert "chlorine" in result

    def test_facility_extraction(self):
        """Distinctive facility keywords should be detected."""
        text = "The storage warehouse and the nearby refinery were affected."
        result = extract_entity_names(text, "facility")
        assert "warehouse" in result
        assert "refinery" in result

    def test_no_match_returns_empty(self):
        """If no keywords match, an empty list should be returned."""
        result = extract_entity_names("Nothing relevant here.", "chemical")
        assert result == []

    def test_case_insensitive(self):
        """Matching should be case-insensitive."""
        result = extract_entity_names("AMMONIA leak reported.", "chemical")
        assert "ammonia" in result

    def test_invalid_entity_type_raises(self):
        """An unsupported entity_type should raise ValueError."""
        with pytest.raises(ValueError, match="entity_type"):
            extract_entity_names("some text", "person")

    def test_deduplication(self):
        """Duplicate mentions should appear only once."""
        text = "The refinery had a refinery-wide shutdown."
        result = extract_entity_names(text, "facility")
        assert result.count("refinery") == 1

    def test_generic_facility_nouns_excluded(self):
        """generic common-nouns must not count as facility entities.

        These boilerplate words ("plant", "facility", "lab", "station", "works",
        "complex", "mill", "terminal") flipped incident_fidelity with context size
        rather than grounding, so they were removed from the keyword list.
        """
        text = "The facility plant had a lab, a station, and a terminal complex."
        result = extract_entity_names(text, "facility")
        for noun in ("plant", "facility", "lab", "station", "works", "complex", "mill", "terminal"):
            assert noun not in result

    def test_word_boundary_matching(self):
        """keywords must match on word boundaries, not as substrings."""
        # "lab" must not match inside "labor"; "laboratory" is a distinct keyword.
        assert extract_entity_names("The labor union met.", "facility") == []
        # "methane" must not be matched inside an unrelated longer token.
        assert "methane" not in extract_entity_names("amethanectomy", "chemical")
        # but a standalone token still matches.
        assert "methane" in extract_entity_names("A methane release occurred.", "chemical")


# ============================================================================
# configure_evaluation_file_logging / default_eval_log_path
# ============================================================================


class TestConfigureEvaluationFileLogging:
    """Tests for optional eval run file logging."""

    def test_writes_log_records(self, tmp_path):
        """Configured file should receive root logger output."""
        log_path = tmp_path / "run.log"
        h = configure_evaluation_file_logging(log_path)
        root = logging.getLogger()
        try:
            logging.getLogger("some.module").info("hello file")
            text = log_path.read_text(encoding="utf-8")
            assert "hello file" in text
        finally:
            root.removeHandler(h)
            h.close()

    def test_idempotent_same_path(self, tmp_path):
        """Calling twice with the same path should not duplicate handlers."""
        log_path = tmp_path / "same.log"
        h1 = configure_evaluation_file_logging(log_path)
        h2 = configure_evaluation_file_logging(log_path)
        root = logging.getLogger()
        try:
            assert h1 is h2
        finally:
            root.removeHandler(h1)
            h1.close()

    def test_default_eval_log_path(self):
        """Default log path should be under EVAL_LOGS_DIR with run_id suffix."""
        p = default_eval_log_path("eval_2099-01-01_v1")
        assert p.name == "eval_2099-01-01_v1.log"
        assert p.parent.name == "logs"


# ============================================================================
# generate_run_id
# ============================================================================


class TestGenerateRunId:
    """Tests for generate_run_id."""

    def test_format(self):
        """Run ID should follow the eval_YYYY-MM-DD_v1 pattern."""
        run_id = generate_run_id()
        assert re.match(r"^eval_\d{4}-\d{2}-\d{2}_v1$", run_id)

    def test_starts_with_eval(self):
        """Run ID should start with 'eval_'."""
        assert generate_run_id().startswith("eval_")

    def test_contains_date(self):
        """Run ID should contain a date substring."""
        run_id = generate_run_id()
        date_part = run_id.split("_")[1]
        parts = date_part.split("-")
        assert len(parts) == 3
        assert all(p.isdigit() for p in parts)

    def test_deterministic_within_same_day(self):
        """Two calls on the same UTC day should produce the same ID."""
        id1 = generate_run_id()
        id2 = generate_run_id()
        assert id1 == id2


# ============================================================================
# normalize_graph_state_for_eval
# ============================================================================


class TestNormalizeGraphStateForEval:
    """Tests for LangGraph state -> evaluator flat context conversion."""

    def test_pass_through_list_legacy(self):
        """List-shaped context_gathered (tests) is unchanged."""
        raw = [
            {"chunk_id": "a", "text": "t", "corpus": "osha_regs"},
        ]
        docs, agents = normalize_graph_state_for_eval({"context_gathered": raw})
        assert docs == raw
        assert agents == ["osha_regs"]

    def test_dict_with_citations_and_formatted_blocks(self):
        """Dict-shaped worker output is flattened using citations + parsed blocks."""
        fmt = (
            "[Source 1] Collection: osha_regulations | Document: doc | Location: loc\n"
            "body one"
        )
        state = {
            "context_gathered": {"osha_regulations": [fmt]},
            "citations": [
                {
                    "chunk_id": "osha_reg_1",
                    "collection": "osha_regulations",
                    "source": "doc",
                    "location": "loc",
                }
            ],
        }
        docs, agents = normalize_graph_state_for_eval(state)
        assert len(docs) == 1
        assert docs[0]["chunk_id"] == "osha_reg_1"
        assert docs[0]["text"] == "body one"
        assert docs[0]["corpus"] == "osha_regs"
        assert agents == ["osha_regs"]

    def test_root_cause_narrative_appended(self):
        """Root cause LLM text is appended as a separate doc with corpus root_cause."""
        state = {
            "context_gathered": {
                "csb": [],
                "root_cause_analysis": ["analysis text"],
            },
            "citations": [],
        }
        docs, agents = normalize_graph_state_for_eval(state)
        assert len(docs) == 1
        assert docs[0]["corpus"] == "root_cause"
        assert docs[0]["text"] == "analysis text"
        assert agents == ["root_cause"]
