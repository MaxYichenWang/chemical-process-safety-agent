# Chemical Process Safety Agent

A supervisor multi-agent retrieval framework for exploring chemical process safety evidence from U.S. Chemical Safety Board (CSB) investigations and OSHA regulations and enforcement records. It combines source-attributed retrieval, DuckDB analytics, an automated critic, and an evaluation harness that separates generation from scoring.

The benchmark is maintained on [Hugging Face](https://huggingface.co/datasets/maxyichenwang/chemical-process-safety-bench). Frozen research artifacts will be available in the Zenodo paper bundle; its DOI and the paper's arXiv identifier are pending publication.

## Quick start from a source checkout

Python 3.11 and 3.12 are supported. CI is configured to verify both versions. The development version is recorded in [.tool-versions](.tool-versions).

```bash
git clone https://github.com/MaxYichenWang/chemical-process-safety-agent.git
cd chemical-process-safety-agent
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
cps-eval --help
```

From the checkout root, run this synthetic smoke test. It reads the fixture and prints the planned run without writing results or calling a provider.

```bash
cps-eval generate --dataset tests/fixtures/sample_eval_dataset.json --dry-run
```

To run the agent after rebuilding the corpus and setting provider credentials:

```bash
python -m scripts.demo_agentic_rag \
  --query 'What evidence explains pressure vessel failures?'
```

After configuring the corpus and provider credentials, run `make web` from the checkout root and open `http://127.0.0.1:8000`. The launcher reads host, port and access settings from the environment and `.env`. For example, `CPS_WEB_PORT=8001 make web` serves port 8001. The server streams progress, citations, and answers and supports follow-up questions and local session history.

For access from another machine on a trusted network, set `CPS_WEB_HOST` and a random `CPS_WEB_ACCESS_TOKEN` in your local `.env`. Start with `make web`, then open `http://<server-address>:<port>/?token=<your-token>` to set the browser's access cookie. Treat that URL as a credential. The supported launcher disables HTTP access logs and keeps the token out of startup output. The shared token grants access to all local sessions, and traffic uses plain HTTP. See [.env.example](.env.example) for server settings.

## Configuration

Copy [.env.example](.env.example) to `.env` with `cp .env.example .env` and configure the provider, model, API-key variable, and temperature for each role: supervisor, worker, reasoning, critic, and data analyst. Retrieval embeddings require `OPENAI_API_KEY` even when chat roles use another provider. Judge settings use the `EVAL_LLM_*` variables. Keys stay in your local environment.

`EVAL_DATASET_DIR` selects the benchmark directory and evaluation output location. `EVAL_DATASET_FILE` overrides the main dataset file independently. Relative paths resolve from the working directory. Example model settings are framework defaults; the paper's exact model configuration belongs to the frozen paper bundle.

For a wheel install, pass your own dataset path to `cps-eval` and use absolute paths for local data and indexes, including `DATA_ROOT`, `CHUNKS_DIR` and `CHROMADB_DIR`. Start the web interface with `python -m src.web`. Web and terminal history are written under `logs/runs/` and `logs/cli/` in the working directory, or under `CPS_PROJECT_ROOT` when configured. Run from a user-writable directory. The fixture examples and Make targets require a source checkout.

History records queries, answers, retrieved context and model-call details locally. Review history before sharing it, and use suitable data when sending prompts to external providers. `CPS_WEB_DAILY_SPEND_CEILING_USD` gates new queries using recorded cost estimates. Work already in flight and untracked provider charges can exceed that estimate.

## Rebuild the data

Follow the [processing guide](src/data_processing/README.md#rebuild-inputs-from-public-sources) for collection, external extraction, input contracts, chunk outputs, and indexing. Raw corpora, embeddings, and databases are built locally. Embedding and model calls can incur charges.

## Evaluate your agent

See [EVALUATION.md](EVALUATION.md) for benchmark download instructions, custom dataset requirements, deterministic scoring, judge configuration, and the generation-to-score hash chain. See [ARCHITECTURE.md](ARCHITECTURE.md) for flowcharts, the code map, and extension points.

## Development

For test and contribution instructions, see [CONTRIBUTING.md](CONTRIBUTING.md).

## Citing

Use [CITATION.cff](CITATION.cff) to cite the software at tag `v0.1.0`. When using the benchmark, also cite [Chemical Process Safety Bench](https://huggingface.co/datasets/maxyichenwang/chemical-process-safety-bench), revision `v1.0` once published. For research results, cite *A Supervisor Multi-Agent Retrieval System for Chemical Process Safety: Reproducing and Extending a Decade of Incident Meta-Analysis with Auditable Evaluation* and its Zenodo paper bundle. The software DOI, bundle DOI, Software Heritage identifier, and arXiv identifier remain pending.

## Safety and license

This is research software. Generated answers and automated critique can be wrong or incomplete. Verify claims against primary sources and obtain qualified engineering review before using outputs in process safety decisions. The framework does not certify compliance or replace a process hazard analysis.

Framework code is [MIT licensed](LICENSE). The separately released benchmark and author-owned bundle data/documentation use CC BY 4.0; third-party rights are identified separately in the bundle.
