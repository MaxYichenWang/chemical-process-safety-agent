"""Web server configuration.

Env-var driven, mirroring the convention in ``src/agentic_rag/config.py``.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path.cwd() / ".env")

WEB_HOST = os.getenv("CPS_WEB_HOST", "127.0.0.1")
WEB_PORT = int(os.getenv("CPS_WEB_PORT", "8000"))

# Admission uses estimated costs recorded by completed runs.
WEB_MAX_CONCURRENCY = int(os.getenv("CPS_WEB_MAX_CONCURRENCY", "3"))
WEB_DAILY_SPEND_CEILING_USD = float(
    os.getenv("CPS_WEB_DAILY_SPEND_CEILING_USD", "20.0")
)

WEB_ACCESS_TOKEN = os.getenv("CPS_WEB_ACCESS_TOKEN", "")
WEB_REQUIRE_TOKEN = os.getenv("CPS_WEB_REQUIRE_TOKEN", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
