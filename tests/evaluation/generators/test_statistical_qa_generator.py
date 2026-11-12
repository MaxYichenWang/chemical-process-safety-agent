"""Tests for StatisticalQAGenerator."""


from src.evaluation.generators.statistical_qa_generator import StatisticalQAGenerator


class TestStatisticalQAGenerator:
    """Tests for the StatisticalQAGenerator class."""

    def test_init_default(self):
        gen = StatisticalQAGenerator()
        assert gen.target_entries == 40

    def test_init_custom(self):
        gen = StatisticalQAGenerator(target_entries=5)
        assert gen.target_entries == 5

    def test_generate_returns_list(self):
        gen = StatisticalQAGenerator(target_entries=5)
        entries = gen.generate()
        assert isinstance(entries, list)
        assert len(entries) == 5

    def test_entry_has_required_fields(self):
        gen = StatisticalQAGenerator(target_entries=3)
        entries = gen.generate()
        for entry in entries:
            assert "query" in entry
            assert "expected_answer" in entry
            assert "source" in entry
            assert "category" in entry
            assert "expected_agents" in entry
            assert "expected_sql" in entry

    def test_source_is_auto_statistical(self):
        gen = StatisticalQAGenerator(target_entries=3)
        entries = gen.generate()
        for entry in entries:
            assert entry["source"] == "auto_statistical"

    def test_agents_include_data_analyst(self):
        gen = StatisticalQAGenerator(target_entries=3)
        entries = gen.generate()
        for entry in entries:
            assert "data_analyst" in entry["expected_agents"]

    def test_categories_covered(self):
        gen = StatisticalQAGenerator(target_entries=40)
        entries = gen.generate()
        categories = {e["category"] for e in entries}
        valid = {
            "statistical_factual",
            "statistical_trend",
            "statistical_comparative",
            "hybrid_analytical",
        }
        assert categories.issubset(valid)
        # With 40 entries all 4 categories should appear
        assert len(categories) == 4

    def test_respects_target_entries(self):
        gen = StatisticalQAGenerator(target_entries=10)
        entries = gen.generate()
        assert len(entries) == 10

    def test_target_larger_than_templates(self):
        gen = StatisticalQAGenerator(target_entries=1000)
        entries = gen.generate()
        # Should return all templates (capped by template count)
        assert len(entries) <= 1000
        assert len(entries) > 0
