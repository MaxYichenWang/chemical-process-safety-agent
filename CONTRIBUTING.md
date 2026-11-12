# Contributing

Contributions should keep the agent framework understandable and its evaluation results traceable to evidence.

## Ground rules

1. Report measured metrics with their denominators and source artifacts. Keep exploratory results separate from frozen published results.
2. Preserve the integrity contract: score artifacts must record the `generation_sha256` of the generation they scored. Do not combine mismatched artifacts.
3. Hold judge settings fixed across agent comparisons. Record model, provider, temperature, dataset revision, and corpus provenance.
4. Use synthetic data and mocked provider calls in tests. Keep credentials, downloaded corpora, and local outputs out of commits.

## Development setup

Follow the [source checkout setup](README.md#quick-start-from-a-source-checkout), then run the checks. `make install` installs the same package and development dependencies. CI uses this environment on Python 3.11 and 3.12.

```bash
make verify
```

Tests supply dummy keys, mock provider calls, and use a reversible byte tokenizer and deterministic local vectors for chunking and storage checks. The suite runs with empty model caches after dependency installation. Tokenization fixtures exercise chunking and validation behavior. Exact production tokenizer counts and live model quality require separate evaluation against your corpus.

`make verify` runs the test suite and source lint checks. For a focused run, use `python -m pytest tests/<package>/`. Add regression coverage for behavior changes. Update [ARCHITECTURE.md](ARCHITECTURE.md) or [EVALUATION.md](EVALUATION.md) when changing public interfaces.

## Release verification

The Ubuntu CI matrix is configured to check Python 3.11 and 3.12, the installed console command,
source tests, distribution contents, and a fresh wheel installation outside the
checkout. The wheel smoke check uses synthetic data and performs no provider
calls. Run it with the wheel environment's Python and an absolute fixture path:

```bash
/path/to/venv/bin/python scripts/reusable/verify_installed_wheel.py \
  --fixture "$PWD/tests/fixtures/sample_eval_dataset.json"
```

Before publishing, require a passing CI run for the exact source commit. After
publication, verify signed-out access to the `v0.1.0` source archive and compare
its files with that commit. Confirm the benchmark's `v1.0` revision contains
`eval_dataset_v1.json`, `eval_dataset_adversarial.json`, and its license. Verify
the separate paper archive's DOI, frozen inputs/results, dependency freeze,
interpreter instructions, replay checks, provenance and third-party notices.
Insert assigned citation identifiers only after they resolve publicly. These
publication checks cannot pass while the repository or artifact hosts are empty.

## Reporting issues

Include the run identifier, affected artifact, metric and denominator, configuration, and a minimal reproduction. Remove credentials and unrelated sensitive content first.

## License

Contributions to this framework use the [MIT License](LICENSE). The separately released benchmark and paper bundle have their own notices and licenses.
