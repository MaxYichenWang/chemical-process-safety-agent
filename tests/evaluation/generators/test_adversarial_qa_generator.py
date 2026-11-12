"""
Tests for adversarial QA generator
"""

from src.evaluation.generators.adversarial_qa_generator import AdversarialQAGenerator

def test_adversarial_qa_generation():
    gen = AdversarialQAGenerator(output_file=None)
    # Patch the output file so it doesn't actually write during basic test passing
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as td:
        tf = Path(td) / "temp.json"
        gen.output_file = tf
        entries = gen.generate()

        assert len(entries) == 24
        assert all(e["source"] == "adversarial" for e in entries)
        assert all(e["category"] == "adversarial" for e in entries)
        assert tf.exists()
