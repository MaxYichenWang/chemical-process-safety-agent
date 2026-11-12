"""Tests for src/web/spend.py."""

from src.web.spend import check_spend_ceiling, get_daily_spend_usd


def _write_analytics(tmp_path, rows):
    log_dir = tmp_path / "logs" / "runs"
    log_dir.mkdir(parents=True)
    csv_path = log_dir / "run_analytics.csv"
    header = "run_id,timestamp,query,total_elapsed_s,total_tokens,total_cost_usd,retry_count,validation_score,worker_count,worker_errors_count,planning_latency_s,synthesis_latency_s,critic_latency_s\n"
    lines = [header]
    for r in rows:
        lines.append(
            f"{r['run_id']},{r['timestamp']},q,1.0,100,{r['total_cost_usd']},0,0.9,1,0,0.1,0.1,0.1\n"
        )
    csv_path.write_text("".join(lines))
    return log_dir


def test_get_daily_spend_sums_only_matching_day(tmp_path):
    log_dir = _write_analytics(
        tmp_path,
        [
            {"run_id": "a", "timestamp": "2026-01-01T00:00:00", "total_cost_usd": "1.5"},
            {"run_id": "b", "timestamp": "2026-01-01T12:00:00", "total_cost_usd": "2.5"},
            {"run_id": "c", "timestamp": "2026-01-02T00:00:00", "total_cost_usd": "9.0"},
        ],
    )
    assert get_daily_spend_usd(log_dir, on_date="2026-01-01") == 4.0
    assert get_daily_spend_usd(log_dir, on_date="2026-01-02") == 9.0


def test_get_daily_spend_returns_zero_when_no_csv(tmp_path):
    assert get_daily_spend_usd(tmp_path / "logs" / "runs") == 0.0


def test_check_spend_ceiling_ok_below_ceiling(tmp_path):
    log_dir = _write_analytics(
        tmp_path, [{"run_id": "a", "timestamp": "2026-01-01T00:00:00", "total_cost_usd": "1.0"}]
    )
    ok, spent = check_spend_ceiling(log_dir, 20.0, on_date="2026-01-01")
    assert ok is True
    assert spent == 1.0


def test_check_spend_ceiling_blocks_at_or_above_ceiling(tmp_path):
    log_dir = _write_analytics(
        tmp_path, [{"run_id": "a", "timestamp": "2026-01-01T00:00:00", "total_cost_usd": "20.0"}]
    )
    ok, spent = check_spend_ceiling(log_dir, 20.0, on_date="2026-01-01")
    assert ok is False
    assert spent == 20.0
