# Evaluation package

For benchmark downloads, custom dataset validation, and the staged CLI, start with [EVALUATION.md](../../EVALUATION.md). The synthetic fixtures under `tests/fixtures/` support smoke tests. Corpus-backed dataset generation requires separately prepared inputs and may make paid model calls.

## Generator inputs

Run from the checkout root after the [source setup](../../README.md#quick-start-from-a-source-checkout) and [corpus preparation](../data_processing/README.md#rebuild-inputs-from-public-sources). Paths below use the defaults in [config.py](config.py) and the generator constructors.

| Generator | Inputs and behavior |
|---|---|
| [RegulatoryQAGenerator](generators/regulatory_qa_generator.py) | `data/processed/osha/regulations/regulations_sections.parquet` and `data/loaded_chunks/osha_regulations_chunks.parquet`. Uses an OpenAI chat model and supports checkpoints. |
| [ViolationQAGenerator](generators/violation_qa_generator.py) | `osha_violations_sample_chunks.parquet` and `osha_regulations_chunks.parquet` under `data/loaded_chunks/`. Uses an OpenAI chat model and supports checkpoints. |
| [GendutyQAGenerator](generators/genduty_qa_generator.py) | `osha_genduty_sample_chunks.parquet` and `osha_regulations_chunks.parquet` under `data/loaded_chunks/`. Uses an OpenAI chat model and supports checkpoints. |
| [EnforcementQAGenerator](generators/enforcement_qa_generator.py) | `osha_accidents_chunks.parquet` and `osha_regulations_chunks.parquet` under `data/loaded_chunks/`. Uses an OpenAI chat model and supports checkpoints. |
| [IncidentQAGenerator](generators/incident_qa_generator.py) | CSB structured JSON and TOCs under `data/processed/csb_olmocr/`, plus final-report and recommendation chunk Parquets. Uses an OpenAI chat model and supports checkpoints. |
| [ExternalSourceCurator](generators/external_source_curator.py) | `<EVAL_DATASET_DIR>/external_sources/psm_training_test.json`, plus regulation, CSB, and accident chunks. The training JSON must be supplied separately. Uses an OpenAI chat model. |
| [AdversarialQAGenerator](generators/adversarial_qa_generator.py) | Built-in refusal cases. Writes `eval_dataset_adversarial.json` under `EVAL_DATASET_DIR` without provider calls. |
| [StatisticalQAGenerator](generators/statistical_qa_generator.py) | Built-in SQL templates. Passing a DuckDB connection to `generate(conn=...)` populates ground-truth results for successful queries. Check SQL against your table schemas. |
| [SyntheticGenerator](generators/synthetic_generator.py) | Regulation chunks and an OpenAI chat model. Writes `eval_dataset_synthetic.json` under `EVAL_DATASET_DIR`. |

Chat-based generators use `OPENAI_API_KEY` and the model and temperature passed to their constructors or read from evaluation config. Review provider settings before live generation. Preserve source chunk IDs and provenance when curating entries.

## Optional PSM training input

Place a user-supplied JSON array at
`<EVAL_DATASET_DIR>/external_sources/psm_training_test.json`. Each row supplies
`question`, `correct_answer`, and `reference`, with an optional `type` label.
The `reference` must match a `section_id` in the local regulation chunk Parquet.
The curator links the chunk and checks the answer with its entailment judge.
If the JSON is absent, it logs a warning and skips the PSM strategy. Its CSB and
enforcement strategies still run against their own prepared inputs.

For example, a synthetic training file can contain the following row when the
local regulation chunk supports its answer.

```json
[
  {
    "question": "Which procedure does the supplied example require?",
    "correct_answer": "Review the example operating procedure before restart.",
    "reference": "example_restart_section",
    "type": "short_answer"
  }
]
```

## Generate a dataset

After preparing the regulatory inputs and credentials, this example writes partial and final entries to a checkpoint file.

```python
from src.evaluation.config import EVAL_DATASET_DIR
from src.evaluation.generators.regulatory_qa_generator import RegulatoryQAGenerator

output = EVAL_DATASET_DIR / "regulatory_qa_entries.json"
generator = RegulatoryQAGenerator(
    target_entries=55,
    checkpoint_path=output,
    checkpoint_every=10,
)
entries = generator.generate()
print(f"Generated {len(entries)} entries at {output}")
```

The target is an upper bound. Inspect entries and use the [dataset validation instructions](../../EVALUATION.md#bring-your-own-dataset) before evaluation. Keep your generated datasets in a separate directory from a downloaded benchmark.

[DatasetBuilder](generators/dataset_builder.py) merges top-level JSON arrays under `EVAL_DATASET_DIR`, excluding the default main and adversarial filenames, and calls OpenAI embeddings for similarity-based deduplication. It writes `eval_dataset_v1.json` by default and logs schema problems. Review the output before use and record its dataset revision.

## Evaluators

| Evaluator | Purpose |
|---|---|
| [RetrievalEvaluator](evaluators/retrieval_evaluator.py) | Retrieval ranking and context precision or recall |
| [FaithfulnessEvaluator](evaluators/faithfulness_evaluator.py) | Answer grounding, hallucination, and citation checks |
| [OrchestrationEvaluator](evaluators/orchestration_evaluator.py) | Worker routing, critic behavior, and tool use |
| [DomainEvaluator](evaluators/domain_evaluator.py) | Safety-domain boundaries and source identifiers |
| [AnalyticalEvaluator](evaluators/analytical_evaluator.py) | SQL execution, numerical fidelity, and analytical synthesis |

Use the [staged evaluation instructions](../../EVALUATION.md) to generate responses, score layers, aggregate results, and inspect reports. That guide also owns the hash contract, applicability flags, judge settings, and resume-cache tags. [schemas.py](schemas.py) defines entry fields and accepted worker names.

## Model sweeps

The [sweep runner](../../scripts/reusable/run_model_sweep.py) uses the
end-to-end evaluator to execute named model presets. It assigns a unique
`--run-id` to each evaluation and compares that run's exact result file with
the configured baseline. Set `EVAL_DATASET_DIR` in the environment or project
`.env` to choose the output directory. Failed evaluations make the sweep return
a nonzero status. Inspect the plan before live calls.

```bash
python scripts/reusable/run_model_sweep.py \
  --config configs/model_sweep.example.yaml --dry-run
```

Direct users of `python -m src.evaluation.evaluators.end_to_end_evaluator` can
also set `--run-id my_comparison` to choose the identifier shared by result,
report, checkpoint, and log files.

The evaluator and sweep return nonzero for operational pipeline failures,
incomplete batches, or failed metrics, including failures nested within a
metric section. Diagnostic JSON and reports retain completion counts and
failure paths. Explicitly disabled metrics and metrics with no applicable
entries do not fail a run. A low measured score is a valid result, not an
operational failure. Corpus paths use the same environment settings as the
processing pipeline; use absolute overrides outside a source checkout.

## Testing

```bash
python -m pytest tests/evaluation/
```

See [CONTRIBUTING.md](../../CONTRIBUTING.md) for the complete checks and [LICENSE](../../LICENSE) for the framework license.
