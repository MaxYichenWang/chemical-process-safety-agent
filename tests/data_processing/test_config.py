"""Path settings accept absolute overrides used by installed packages."""

import runpy
from pathlib import Path

import pytest

from src.data_processing import config


@pytest.mark.parametrize(
    "name",
    [
        "CHROMADB_DIR",
        "EMBEDDINGS_CACHE_DB",
        "CHUNKS_DIR",
        "BATCH_FILES_DIR",
        "EMBEDDINGS_OUTPUT_DIR",
        "CSB_DATA_DIR",
        "OSHA_DATA_DIR",
    ],
)
@pytest.mark.parametrize("relative", [False, True])
def test_processing_paths_are_resolved_paths(tmp_path, monkeypatch, name, relative):
    configured = "review-data" if relative else str(tmp_path / "review-data")
    monkeypatch.setenv(name, configured)

    loaded = runpy.run_path(config.__file__)

    expected = (config.PROJECT_ROOT / configured).resolve()
    assert isinstance(loaded[name], Path)
    assert loaded[name] == expected
