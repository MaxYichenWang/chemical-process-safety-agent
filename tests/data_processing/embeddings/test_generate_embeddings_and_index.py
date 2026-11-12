"""Embedding entry-point checks that stop before clients and model calls."""

import sys
from unittest.mock import Mock

import pytest

from src.data_processing.embeddings import generate_embeddings_and_index as runner


@pytest.mark.parametrize("skip_validation", [False, True])
@pytest.mark.parametrize("corpus", [None, "csb_recs"])
def test_empty_input_exits_before_clients(
    tmp_path, monkeypatch, skip_validation, corpus
):
    arguments = ["generate_embeddings_and_index", "--chunks-dir", str(tmp_path)]
    if skip_validation:
        arguments.append("--skip-validation")
    if corpus:
        arguments.extend(["--corpus", corpus])
    monkeypatch.setattr(sys, "argv", arguments)
    validator = Mock()
    pipeline = Mock()
    indexer = Mock()
    monkeypatch.setattr(runner, "ChunkValidator", validator)
    monkeypatch.setattr(runner, "BatchEmbeddingPipeline", pipeline)
    monkeypatch.setattr(runner, "ChromaDBIndexer", indexer)

    with pytest.raises(SystemExit) as raised:
        runner.main()

    assert str(tmp_path) in str(raised.value)
    assert f"corpus filter: {corpus or 'all'}" in str(raised.value)
    validator.assert_not_called()
    pipeline.assert_not_called()
    indexer.assert_not_called()
