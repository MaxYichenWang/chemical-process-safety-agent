"""
Test configuration for the chemical-process-safety-agent project.

Ensures that the project root is on sys.path so that imports like
`from src.data_processing...` and `from src.data_collection...` work
both locally and in CI.
"""

import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("OPENAI_API_KEY", "sk-dummy-for-ci")
os.environ.setdefault("CHROMA_OPENAI_API_KEY", "sk-dummy-for-ci")
os.environ.setdefault("ANONYMIZED_TELEMETRY", "false")
os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")


@pytest.fixture(autouse=True)
def local_tokenizer(monkeypatch):
    """Exercise token-based code with a reversible byte vocabulary, without downloads."""
    import tiktoken
    import tiktoken.model

    encoding = tiktoken.Encoding(
        name="test_bytes",
        pat_str=r"(?s).",
        mergeable_ranks={bytes([value]): value for value in range(256)},
        special_tokens={},
    )
    monkeypatch.setattr(tiktoken, "get_encoding", lambda name: encoding)
    monkeypatch.setattr(tiktoken.model, "get_encoding", lambda name: encoding)


def _ensure_project_root_on_path() -> None:
    # tests/ -> project_root/
    project_root = Path(__file__).resolve().parent.parent

    # Use string path for compatibility with sys.path entries
    project_root_str = str(project_root)

    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)


_ensure_project_root_on_path()
