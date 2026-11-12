"""Daily spend tracking against `run_analytics.csv`.

Reuses the per-call cost data `_save_log()` / `_append_run_analytics_csv()`
already record (design doc §3.4) rather than inventing a second source of
truth. No new datastore.
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path


def get_daily_spend_usd(log_dir: Path, *, on_date: str | None = None) -> float:
    """Sum `total_cost_usd` for rows timestamped on `on_date` (UTC, default: today)."""
    csv_path = log_dir / "run_analytics.csv"
    if not csv_path.exists():
        return 0.0

    day = on_date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    total = 0.0
    try:
        with csv_path.open("r", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                ts = row.get("timestamp", "")
                if ts.startswith(day):
                    try:
                        total += float(row.get("total_cost_usd") or 0.0)
                    except ValueError:
                        continue
    except OSError:
        return 0.0
    return total


def check_spend_ceiling(
    log_dir: Path, ceiling_usd: float, *, on_date: str | None = None
) -> tuple[bool, float]:
    """Returns (ok_to_proceed, spent_so_far_today)."""
    spent = get_daily_spend_usd(log_dir, on_date=on_date)
    return spent < ceiling_usd, spent
