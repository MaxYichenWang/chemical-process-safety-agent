from src.evaluation.scoring.cache import ScoringCache


def test_put_then_get_roundtrips(tmp_path):
    cache = ScoringCache(tmp_path / "cache.json")
    assert cache.get("gensha", "0", "faithfulness", "qwen") is None
    cache.put("gensha", "0", "faithfulness", "qwen", 0.9)
    assert cache.get("gensha", "0", "faithfulness", "qwen") == 0.9


def test_cache_persists_across_instances_resume_after_kill(tmp_path):
    path = tmp_path / "cache.json"
    cache1 = ScoringCache(path)
    cache1.put("gensha", "0", "faithfulness", "qwen", 0.9)
    cache1.put("gensha", "1", "faithfulness", "qwen", 0.5)

    # Simulate the process being killed and a fresh ``score`` invocation
    # picking the cache back up.
    cache2 = ScoringCache(path)
    assert cache2.get("gensha", "0", "faithfulness", "qwen") == 0.9
    assert cache2.has("gensha", "1", "faithfulness", "qwen")
    assert not cache2.has("gensha", "2", "faithfulness", "qwen")


def test_pending_entry_ids_only_lists_incomplete_entries(tmp_path):
    cache = ScoringCache(tmp_path / "cache.json")
    cache.put("gensha", "0", "faithfulness", "qwen", 0.9)
    cache.put("gensha", "0", "hallucination", "qwen", 0.1)
    cache.put("gensha", "1", "faithfulness", "qwen", 0.5)
    # entry "1" is missing "hallucination" -> still pending
    pending = cache.pending_entry_ids(
        ["0", "1", "2"], "gensha", ["faithfulness", "hallucination"], "qwen"
    )
    assert pending == ["1", "2"]


def test_cache_scoped_by_generation_sha_and_judge_id(tmp_path):
    cache = ScoringCache(tmp_path / "cache.json")
    cache.put("gensha-A", "0", "faithfulness", "qwen", 0.9)
    # Different generation -> not cached even for the same entry id/metric.
    assert cache.get("gensha-B", "0", "faithfulness", "qwen") is None
    # Different judge -> not cached (avoids cross-judge contamination).
    assert cache.get("gensha-A", "0", "faithfulness", "deepseek") is None


def test_corrupt_cache_file_is_treated_as_empty(tmp_path):
    path = tmp_path / "cache.json"
    path.write_text("{not valid json", encoding="utf-8")
    cache = ScoringCache(path)
    assert cache.get("gensha", "0", "faithfulness", "qwen") is None
