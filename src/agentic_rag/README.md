# Agentic RAG framework

This module contains the LangGraph-based multi-agent retrieval system for chemical process safety.

See [ARCHITECTURE.md](../../ARCHITECTURE.md) for graph flow. Parallel workers return only new contributions for state reducers. Worker reformulation limits and critic retry thresholds are configurable in [config.py](config.py).

## Analytical data

The `DataAnalystWorker` uses DuckDB views over prepared Parquet and CSV files. The `_VIEW_DEFINITIONS` mapping in [analytical_tools.py](analytical_tools.py) lists all 13 input paths. See the [processing guide](../data_processing/README.md#rebuild-inputs-from-public-sources) for extraction and normalization.

```bash
make check-data
```

This checks file presence and exits successfully even when all inputs are missing. Query execution uses the available views and records structured errors in `worker_errors` when required evidence is unavailable. Generated SQL and numerical conclusions need verification against the source data.

## Safety

The critic can revise answers within a retry limit and disclose low confidence. Verify claims against primary sources and obtain qualified engineering review before using outputs in process safety decisions. See the [safety and license notice](../../README.md#safety-and-license).
