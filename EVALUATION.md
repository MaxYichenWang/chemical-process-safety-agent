# Evaluate your own agent

The harness separates four processing stages: **generate → score → aggregate → report**. The CLI exposes `generate`, `score`, and `aggregate`; aggregation also writes the human-readable report. Frozen paper runs and replication instructions are distributed in the separate Zenodo paper bundle.

## Obtain a benchmark

Install the Hugging Face command-line client separately if needed, then download the paper revision after it is published:

```bash
hf download maxyichenwang/chemical-process-safety-bench \
  --repo-type dataset --revision v1.0 --local-dir data/evaluation
```

The repository may be empty until publication; the command requires uploaded files and the `v1.0` tag. Older clients use `huggingface-cli download` with the same repository arguments. The main file is `eval_dataset_v1.json`; the adversarial file is `eval_dataset_adversarial.json`. The analytical file is a subset of the main split, not an additional evaluation population.

Set paths before starting a new CLI process:

```bash
export EVAL_DATASET_DIR=/absolute/path/to/benchmark
# Optional: use a different main dataset while retaining the directory above.
export EVAL_DATASET_FILE=/absolute/path/to/my_benchmark.json
```

The default directory is `data/evaluation`. The directory override also controls the adversarial split and output subdirectories. An explicit `--dataset` overrides the main file for generation. No HF client is required at runtime once the JSON files are available.

## Bring your own dataset

Supply a JSON array. Each entry requires `query`, `expected_answer`, `ground_truth_chunks` (a list of string chunk identifiers), `source`, and `category`. Accepted source/category values and optional fields are defined in [schemas.py](src/evaluation/schemas.py). Worker routing targets must be registered agent names.

A synthetic example is in [sample_eval_dataset.json](tests/fixtures/sample_eval_dataset.json); refusal cases are in [sample_adversarial.json](tests/fixtures/sample_adversarial.json). For entries without retrieval ground truth, set `l1_applicable: false`. For reference answers that cannot be graded, set `answer_gradable: false`. SQL entries may provide `expected_sql` and `ground_truth_result`. These flags control applicability independently of schema validation.

Validate your file locally:

```python
import json
from pathlib import Path
from src.evaluation.schemas import validate_eval_dataset

result = validate_eval_dataset(json.loads(Path('my_benchmark.json').read_text()))
assert result['invalid'] == 0, result['errors']
assert not result['duplicate_queries'], result['duplicate_queries']
```

Chunk identifiers must refer to your own indexed corpus for live retrieval evaluation. Downloading the benchmark does not download that corpus.

## 1. Generate

From the checkout root, use the synthetic fixture to check dataset loading and planned run settings without API calls or a corpus. This dry run writes no generation artifact.

```bash
cps-eval generate --dataset tests/fixtures/sample_eval_dataset.json --dry-run
```

For live evaluation, use the separate benchmark or your own validated JSON and an indexed corpus with matching chunk IDs. After configuring your corpus and provider credentials, generate responses.

```bash
cps-eval generate --dataset my_benchmark.json --run-id my_agent --save-logs
```

Generation uses the full agent by default. `--system vanilla`, `single_agent`, and `no_retrieval` select included baselines. To evaluate an external agent, adapt its responses to the generation artifact's `run_metadata` and `pipeline_results` contract documented in [run.py](src/evaluation/run.py) and the [scoring contract](src/evaluation/scoring/contract.py); preserve the dataset-to-response order and relevant retrieval/tool metadata.

Generation exits unsuccessfully for empty inputs, interrupted batches, or pipeline errors. Its `run_metadata.completion` records the status and expected, processed, successful, failed and unprocessed entry counts. Interrupted batches retain a result slot for every dataset row; unprocessed slots carry `not_processed: true` and an error. Preserve these diagnostic artifacts. Scoring and aggregation reject generations marked `incomplete` or `failed`. A fully processed run with some failed rows is marked `complete_with_errors`; it can still be scored with the failed rows excluded from quality metrics and counted in pipeline error rates.

Corpus-backed evaluation shares `CHROMADB_DIR`, `CHUNKS_DIR`, `CSB_DATA_DIR` and `OSHA_DATA_DIR` with processing and retrieval. Set absolute paths before launching a new process when the corpus is outside the checkout. The ground-truth audit uses the configured ChromaDB directory.

## 2. Score

Scoring requires an existing generation artifact. Deterministic scoring requires no judge calls.

```bash
cps-eval score --run-id my_agent --layer deterministic
```

Judge layers require credentials and can incur charges. Configure `EVAL_LLM_PROVIDER`, `EVAL_LLM_MODEL`, `EVAL_LLM_BASE_URL`, `EVAL_LLM_API_KEY_ENV`, and `EVAL_LLM_TEMPERATURE` in the environment as appropriate. For example:

```bash
cps-eval score --run-id my_agent --layer deepeval \
  --judge YOUR_JUDGE_MODEL --judge-provider YOUR_PROVIDER \
  --judge-api-key-env YOUR_API_KEY_VARIABLE --max-concurrency 4
```

OpenAI-compatible judge endpoints also require `--judge-base-url` or `EVAL_LLM_BASE_URL`.

Use `--layer ragas` for the alternative judge framework. Metrics that use embeddings configure them separately with `EVAL_EMBEDDING_MODEL`, `EVAL_EMBEDDING_API_KEY_ENV` (default `OPENAI_API_KEY`), and `EVAL_EMBEDDING_BASE_URL`. These embedding calls can incur charges. Keep the judge model and settings fixed across compared agents. Scores from different frameworks or judges measure different conditions; a model judging its own outputs can introduce self-preference bias. Use a new `--tag` when changing judge temperature so the result and resume cache stay distinct. Framework defaults are not the paper's frozen replication configuration.

## 3. Aggregate

```bash
cps-eval aggregate --run-id my_agent
```

This merges available deterministic and judge layers and records missing layers explicitly. The previous results are archived unless `--no-archive` is supplied. Applicability flags determine each metric's denominator.

## 4. Read the report

Aggregation writes `results.json` and `report.md` alongside the generation artifact:

```text
<EVAL_DATASET_DIR>/runs/my_agent/
  generation.json                 # gzip also supported
  scores/
    deterministic.json
    deepeval.<judge>.json
    ragas.<judge>.json
  results.json
  report.md
```

Inspect both metrics and denominators. A report with missing judge layers is not a complete quality assessment.

## Integrity and dataset revisions

Every score artifact records `generation_sha256` and the effective `dataset_sha256`. Aggregation verifies that all layers belong to the generation and the same dataset revision, and rejects mismatches before writing results. The generation metadata also records the dataset hash and model/run settings. Preserve the artifacts together when sharing results.

Keep the exact dataset file and row order used for generation. Use absolute paths or run each stage from the same working directory. Scoring and aggregation reject a dataset hash mismatch. Existing score files can still be aggregated after the external dataset is archived when their dataset hashes match the hash recorded by the generation. Reconstructing applicability details requires the dataset.

Historical imported generations may embed a frozen dataset. To apply later annotations to exactly the same question set, use `--rebind-dataset` for **both** `score` and `aggregate`; the harness rejects changed question sets, aligns current entries to the snapshot's question order, and records the effective dataset hash. Resume caches include the dataset revision. Re-score every layer after a rebind. Keep the original generations immutable and document the dataset revision used. For the paper's archived runs and expected results, use the separate bundle's `docs/EVALUATION_paper.md` and `docs/REPLICATION.md`.
