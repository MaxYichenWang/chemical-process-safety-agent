# Embeddings and ChromaDB pipeline

The runner reads chunk Parquets, reports schema and quality checks, generates OpenAI Batch embeddings, and indexes them into ChromaDB. Follow the [processing guide](../README.md#rebuild-inputs-from-public-sources) to prepare nonempty chunk files and the [root setup](../../../README.md#quick-start-from-a-source-checkout) to install dependencies.

## Validate chunks

Run from the checkout root.

```bash
python -m src.data_processing.embeddings.generate_embeddings_and_index \
  --chunks-dir data/loaded_chunks --validate-only
```

This prints a validation report and exits before constructing the embedding client or ChromaDB indexer. An empty input exits with an error identifying the directory and corpus filter. Tokenizer initialization may download tiktoken files on first use. Inspect the printed errors and warnings before generating embeddings.

The [validator](../data_loading/chunk_validator.py) requires `chunk_id`, `text`, `corpus`, and `hash`. It warns about missing `token_count`, `subcorpus`, or `doc_type`, duplicate hashes, and token counts outside the configured range. Validation reports issues without removing invalid chunks.

## Generate embeddings and index

Set `OPENAI_API_KEY` in the environment or local `.env`. Embedding calls can incur charges.

```bash
python -m src.data_processing.embeddings.generate_embeddings_and_index \
  --chunks-dir data/loaded_chunks --chromadb-dir ./chromadb --resume
```

Use `--embeddings-only` to skip indexing. Use `--corpus osha_reg --resume` to process one input. `--sample N` limits embedding work after reading and validation. See the runner's help for path overrides, batch limits, and submission options.

```bash
python -m src.data_processing.embeddings.generate_embeddings_and_index --help
```

## Inputs and collection mapping

The [reader](../data_loading/chunk_reader.py) recognizes these files under `data/loaded_chunks/`. The [indexer](chromadb_indexer.py) maps their corpus keys to collections.

| Corpus key | Chunk Parquet | ChromaDB collection |
|---|---|---|
| `osha_reg` | `osha_regulations_chunks.parquet` | `osha_regulations` |
| `osha_enf_accidents` | `osha_accidents_chunks.parquet` | `osha_enf` |
| `osha_enf_violations` | `osha_violations_sample_chunks.parquet` | `osha_enf` |
| `osha_enf_genduty` | `osha_genduty_sample_chunks.parquet` | `osha_enf` |
| `csb_reports` | `csb_final_reports_chunks.parquet` | `csb` |
| `csb_recs` | `csb_recommendations_chunks.parquet` | `csb_recommendations` |

The runner deduplicates by `chunk_id` before embedding and indexing.

## Outputs and resuming

| Default path | Contents |
|---|---|
| `data/embeddings_cache.duckdb` | Embeddings keyed by content hash and batch job records |
| `data/batch_files/` | Batch request and result files |
| `data/embeddings_output/` | Timestamped JSON and Markdown pipeline reports |
| `chromadb/` | Persistent vector collections |
| `chromadb/collections_manifest.json` | Exported collection manifest |

`--validate-only` prints its report. `--embeddings-only` and `--no-wait` exit before the final indexing report and collection manifest.

`--resume` skips content hashes already in the embedding cache. Keep the cache with its matching embedding model and dimensions. Batch job records support monitoring and downloading results. Review job status before rerunning an interrupted submission.

Use `--corpus csb_recs --resume --upsert` to update matching records in an existing collection. Upsert retains IDs absent from the new input.

## Python interfaces

- [ChunkReader](../data_loading/chunk_reader.py) reads named corpus files and exposes chunk statistics.
- [ChunkValidator](../data_loading/chunk_validator.py) returns errors, warnings, and quality metrics.
- [BatchEmbeddingPipeline](batch_embedding_pipeline.py) handles caching and batch submissions.
- [ChromaDBIndexer](chromadb_indexer.py) supports insertion, upsert, verification, and manifest export.

## Testing

```bash
python -m pytest tests/data_processing/embeddings/
```

See [CONTRIBUTING.md](../../../CONTRIBUTING.md) for test prerequisites and [LICENSE](../../../LICENSE) for the framework license.
