# Data Processing Module for Chemical Process Safety Agent

This module provides high-quality data loading and chunking functionality for OSHA regulations, OSHA enforcement data, and CSB investigation reports. It implements methodology-compliant chunking strategies optimized for Retrieval-Augmented Generation (RAG) systems.

## Table of Contents

1. [Features](#features)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Detailed Usage](#detailed-usage)
5. [API Reference](#api-reference)
6. [Testing](#testing)
7. [Architecture](#architecture)

## Features

### Core Capabilities

- **CSB Data Loading**: TOC-guided hierarchical chunking for final reports, minimal chunking for recommendations
- **OSHA Regulations**: Section-preserving chunking that respects regulatory boundaries
- **OSHA Enforcement**: Record-level chunking for violations and accidents with rich metadata
- **Embedding Generation**: OpenAI Batch API integration with 50% cost savings and caching
- **Vector Storage**: Multi-collection ChromaDB indexing with verification
- **Environment Flexibility**: Works in both local Python venv and Google Colab

### Embeddings and ChromaDB pipeline

The module includes a pipeline for generating embeddings and indexing them into ChromaDB:

- **ChunkReader**: Efficient parquet reading with DuckDB
- **ChunkValidator**: Schema and quality validation
- **BatchEmbeddingPipeline**: OpenAI Batch API integration with caching and resume capability
- **ChromaDBIndexer**: Multi-collection indexing with verification
- **Orchestration Script**: End-to-end automation with progress tracking

See [embeddings/README.md](./embeddings/README.md) for detailed documentation.

### Methodology Compliance

See the [architecture](../../ARCHITECTURE.md) for the complete pipeline:

- ✅ Semantic coherence (300-800 token chunks)
- ✅ Boundary respect (sections, paragraphs, sentences)
- ✅ Hierarchical metadata preservation
- ✅ Content-based hashing for deduplication
- ✅ Standardized chunk schema

## Installation

Follow the [source checkout setup](../../README.md#quick-start-from-a-source-checkout).

## Rebuild inputs from public sources

Run commands from the checkout root. See the [collection guide](../data_collection/README.md) for CSB downloads, manual OSHA enforcement acquisition, and the separate Part 1910 regulation scrape.

### Prepare processed inputs

The loaders consume processed files. Rebuilding requires an external OLMoCR installation and local normalization code. The historical extraction notebooks are private, and the corpus must be prepared locally.

- For CSB reports, prepare `json/<incident>/<report>_structured.json`, `markdown/`, and `toc/<incident>/<report>_toc.json` under `data/processed/csb_olmocr/`. The loader classifies structured JSON with a matching TOC as a final report and JSON without one as a recommendation. Preserve that distinction. See [csb_loader.py](data_loading/csb_loader.py) and its [synthetic input examples](../../tests/data_processing/data_loading/test_csb_loader.py) for page, text, and TOC fields.
- For OSHA regulations, normalize JSON to `data/processed/osha/regulations/regulations_sections.parquet`.
- For OSHA enforcement chunking, prepare `violations_text.parquet`, `accidents_text.parquet`, `violation_genduty_text.parquet`, and `inspection.parquet` under `data/processed/osha/enforcement/`. Preserve record identifiers, dates, text, normalized standard numbers, and NAICS codes. The default loader builds filtered violation and general-duty samples. Its violation sample can also enrich records from `data/raw/osha/enforcement/osha_violation.csv` when present, so copy the collected CSV there if using that enrichment. See [osha_loader.py](data_loading/osha_loader.py) for input columns and joins.
- For analytical queries, prepare the separate files in `_VIEW_DEFINITIONS` in [analytical_tools.py](../agentic_rag/analytical_tools.py). These include `violations_struct.parquet`, `accidents_struct.parquet`, and CSB CSV summaries under `data/processed/csb_olmocr/csv/`.

Record download dates, source URLs, extraction software and settings, model versions, and third-party rights. Source updates and extraction choices affect the rebuilt corpus and its chunk identifiers.

### Load chunks

```bash
python -m src.data_processing.data_loading.load_all_data
```

By default the loader writes six chunk Parquets under `data/loaded_chunks/`, listed in the [embedding input table](embeddings/README.md#inputs-and-collection-mapping). It also writes `DATA_LOADING_VALIDATION_REPORT.md` and `csb_osha_data_loading_results.json` there. Existing chunk Parquets are reused without checking whether their inputs changed. Use a fresh `CHUNKS_DIR` when rebuilding changed inputs.

### Validate and index

Follow the [embedding guide](embeddings/README.md) for chunk validation, OpenAI Batch credentials, caching, output paths, and ChromaDB indexing. Embedding calls can incur charges.

### Check analytical files

```bash
make check-data
```

This reports presence of the 13 analytical input files. It exits successfully even when every file is missing. Query execution still requires the necessary tables and columns.

Relative processing paths such as `CHUNKS_DIR` and `OSHA_DATA_DIR` resolve against the package project root, while `DATA_ROOT` resolves against the working directory. For an installed wheel, set absolute data and index paths. See [config.py](config.py).

## Quick Start

### Load CSB Final Reports

```python
from src.data_processing import CSBDataLoader

csb_loader = CSBDataLoader()

# Load first 5 final reports with TOC-guided chunking
chunks = csb_loader.load_all_final_reports(limit=5, show_progress=True)

print(f"Generated {len(chunks)} chunks")
print(f"First chunk: {chunks[0]['text'][:200]}...")
```

### Load OSHA Regulations

```python
from src.data_processing import OSHARegulationsLoader

osha_reg_loader = OSHARegulationsLoader()

# Load Part 1910 regulations
chunks = osha_reg_loader.load_all_regulations(
    part_number='1910',
    limit=50,
    show_progress=True
)

print(f"Generated {len(chunks)} regulation chunks")
```

### Load OSHA Violations

```python
from src.data_processing import OSHAEnforcementLoader

osha_enf_loader = OSHAEnforcementLoader()

# Load California violations for specific standard
chunks = osha_enf_loader.load_all_violations(
    state='CA',
    standard_number='1910.147',
    limit=100,
    show_progress=True
)

print(f"Generated {len(chunks)} violation chunks")
```

## Detailed Usage

### 1. CSB Data Loading with TOC Guidance

```python
from src.data_processing import CSBDataLoader

# Initialize with custom parameters
csb_loader = CSBDataLoader(
    chunk_size=800,         # Target tokens per chunk
    chunk_overlap=100,      # Overlap for context preservation
    max_tokens_per_chunk=1000  # Hard limit before splitting
)

# Discover reports from file system
final_reports, recommendations = csb_loader._discover_reports()
print(f"Available: {len(final_reports)} final reports, {len(recommendations)} recommendations")

# Load single report
sample = final_reports[0]
chunks = csb_loader.chunk_final_report(
    incident_name=sample['incident_name'],
    report_name=sample['report_name'],
    report_metadata=None
)

# Each chunk contains rich metadata
for chunk in chunks[:3]:
    print(f"\nSection: {chunk['section_title']}")
    print(f"Hierarchy: {chunk['section_path']}")
    print(f"Pages: {chunk['page_start']}-{chunk['page_end']}")
    print(f"Tokens: {chunk['token_count']}")
```

### 2. OSHA Regulations with Section Preservation

```python
from src.data_processing import OSHARegulationsLoader

osha_reg_loader = OSHARegulationsLoader()

regulations_df = osha_reg_loader.load_regulations(
    part_number='1910',
    limit=100
)

# Chunk with section preservation
chunks = []
for _, row in regulations_df.iterrows():
    section_chunks = osha_reg_loader.chunk_regulation_section(row.to_dict())
    chunks.extend(section_chunks)

# Analyze chunk sizes
import numpy as np
token_counts = [c['token_count'] for c in chunks]
print(f"Mean: {np.mean(token_counts):.1f}, Median: {np.median(token_counts):.1f}")
```

### 3. OSHA Enforcement with Filtering

```python
from src.data_processing import OSHAEnforcementLoader

osha_enf_loader = OSHAEnforcementLoader()

# Load violations with complex filtering
violations_df = osha_enf_loader.load_violations(
    state='TX',
    standard_number='1910.147',
    violation_type='Serious',
    limit=500
)

# Load fatality accidents
accidents_df = osha_enf_loader.load_accidents(
    fatality_only=True,
    limit=100
)

# Chunk violations
violation_chunks = osha_enf_loader.load_all_violations(
    state='TX',
    limit=1000,
    show_progress=True
)

# Chunk accidents
accident_chunks = osha_enf_loader.load_all_accidents(
    fatality_only=True,
    limit=500,
    show_progress=True
)
```

### 4. Export to Parquet

```python
# Export chunks for later use
csb_loader.export_chunks_to_parquet(
    chunks=csb_chunks,
    output_path="data/processed/rag/chunks_csb_reports.parquet"
)

osha_reg_loader.export_chunks_to_parquet(
    chunks=osha_chunks,
    output_path="data/processed/rag/chunks_osha_reg.parquet"
)

# Load back later
import pandas as pd
loaded_chunks = pd.read_parquet("data/processed/rag/chunks_csb_reports.parquet")
```

## API Reference

### CSBDataLoader

```python
CSBDataLoader(
    data_root: Optional[str] = None,  # Root dir for CSB data
    chunk_size: int = 800,             # Target tokens per chunk
    chunk_overlap: int = 100,          # Overlap between chunks
    max_tokens_per_chunk: int = 1000  # Max before forced split
)
```

**Methods:**

- `_discover_reports()`: Discover reports and recommendations from file system
- `load_report_json(incident_name, report_name)`: Load single report
- `load_toc(incident_name, report_name)`: Load table of contents
- `chunk_final_report(incident_name, report_name, metadata)`: Chunk with TOC
- `chunk_recommendation(incident_name, report_name, metadata)`: Minimal chunking
- `load_all_final_reports(limit, show_progress)`: Bulk load reports
- `load_all_recommendations(limit, show_progress)`: Bulk load recommendations
- `export_chunks_to_parquet(chunks, output_path)`: Export to Parquet

### OSHARegulationsLoader

```python
OSHARegulationsLoader(
    data_root: Optional[str] = None,
    chunk_size: int = 800,
    chunk_overlap: int = 100
)
```

**Methods:**

- `load_regulations(part_number, standard_number, limit)`: Query regulations
- `chunk_regulation_section(section_data)`: Chunk single section
- `load_all_regulations(part_number, limit, show_progress)`: Bulk load
- `export_chunks_to_parquet(chunks, output_path)`: Export

### OSHAEnforcementLoader

```python
OSHAEnforcementLoader(
    data_root: Optional[str] = None
)
```

**Methods:**

- `load_violations(state, standard_number, violation_type, limit)`: Query violations
- `load_accidents(state, fatality_only, limit)`: Query accidents
- `chunk_violation(violation_data)`: Chunk single violation
- `chunk_accident(accident_data)`: Chunk single accident
- `load_all_violations(state, standard_number, limit, show_progress)`: Bulk load
- `load_all_accidents(state, fatality_only, limit, show_progress)`: Bulk load
- `export_chunks_to_parquet(chunks, output_path)`: Export

### Utility Functions

```python
# Token counting
count_tokens(text: str, model: str = "gpt-4") -> int

# Hashing for deduplication
compute_hash(text: str) -> str

# Chunk ID generation
generate_chunk_id(corpus: str, identifier: str, chunk_index: Optional[int] = None) -> str

# Text chunking
chunk_text(text: str, chunk_size: int = 800, chunk_overlap: int = 100) -> List[str]

# Metadata creation
create_chunk_metadata(chunk_id, corpus, subcorpus, doc_type, text, source_path, **kwargs) -> Dict
```

## Testing

```bash
python -m pytest tests/data_processing/
```

See [CONTRIBUTING.md](../../CONTRIBUTING.md) for the complete checks and first-run downloads.

## Architecture

### Module Structure

```
src/data_processing/
├── __init__.py              # Main exports
├── data_loading/            # Data loading and chunking
│   ├── __init__.py
│   ├── csb_loader.py        # CSB data loader
│   ├── osha_loader.py      # OSHA data loaders
│   ├── load_all_data.py    # Orchestration script
│   ├── chunk_reader.py     # Chunk reading utilities
│   └── chunk_validator.py  # Chunk validation
├── utilities/              # Common utilities
│   ├── __init__.py
│   ├── utils.py            # Core utilities (tokens, hashing, chunking)
│   └── embeddings_utils.py # Embeddings & ChromaDB utilities
├── embeddings/             # Embedding generation and indexing
│   ├── __init__.py
│   ├── batch_embedding_pipeline.py  # Batch embedding pipeline
│   ├── chromadb_indexer.py          # ChromaDB indexing
│   └── generate_embeddings_and_index.py  # Orchestration script
└── README.md               # This file
```

### Data Flow

```
┌─────────────────────────────────────────────────────────────┐
│                   DATA LOADING PIPELINE                      │
└─────────────────────────────────────────────────────────────┘

1. SOURCE DATA (Parquet/JSON/CSV)
   │
   ├─► CSB:  data/processed/csb_olmocr/
   ├─► OSHA Reg: data/processed/osha/regulations/
   └─► OSHA Enf: data/processed/osha/enforcement/

2. DUCKDB (LOAD & QUERY)
   │
   └─► Zero-copy Parquet reads, SQL filtering

3. CHUNKING (LangChain TextSplitter)
   │
   ├─► CSB: TOC-guided hierarchical
   ├─► OSHA Reg: Section-preserving
   └─► OSHA Enf: Record-level

4. METADATA ENRICHMENT
   │
   └─► Standardized schema with provenance

5. EMBEDDING GENERATION (Optional)
   │
   └─► OpenAI API, batch processing

6. CHROMADB STORAGE (Optional)
   │
   └─► Persistent vector store with metadata filtering

7. EXPORT (Parquet)
   │
   └─► Reusable chunks for downstream systems
```

### Chunk Schema

All chunks follow a standardized schema:

```python
{
    # Required fields
    "chunk_id": "csb_reports_AirGas_57",
    "corpus": "csb",
    "subcorpus": "csb_reports",
    "doc_type": "csb_final_report",
    "text": "...",
    "token_count": 756,
    "hash": "sha256:...",
    "schema_version": 1,
    "embeddings_version": "openai.t3l.1024@2025-11-12",
    "build_manifest_id": "csb_20251112_143022",
    "source_path": "data/processed/csb_olmocr/json/...",
    "source_url": None,
    "created_at": "2025-11-12T14:30:22Z",

    # Corpus-specific metadata
    "incident_name": "AirGas_Facility_Fatal_Explosion",
    "report_name": "Airgas_Investigation_Report_Final",
    "section_title": "Mechanical Integrity",
    "section_level": 2,
    "section_path": "4 SAFETY ISSUES / 4.1 Mechanical Integrity",
    "page_start": 42,
    "page_end": 48,
    ...
}
```

## Performance Considerations

### Memory Usage

- **CSB Loader**: ~100-200 MB for 10 reports
- **OSHA Loader**: ~50-100 MB for 1000 regulations
- **ChromaDB**: ~500 MB for 10K chunks with embeddings

### Optimization Tips

1. **Use limits during development** to avoid long processing times
2. **Process in batches** for large datasets
3. **Export to Parquet** and reuse instead of reprocessing
4. **Use DuckDB filtering** before chunking to reduce data volume
5. **Cache embeddings** using content hashes

## Troubleshooting

### Common Issues

**Issue**: `FileNotFoundError: Required directory not found`

- **Solution**: Verify data is in `data/processed/` directory
- Check paths match your environment (local vs Colab)

**Issue**: `ModuleNotFoundError: No module named 'src'`

- **Solution**: Ensure project root is in Python path

```python
import sys
sys.path.insert(0, '/path/to/chemical-process-safety-agent')
```

**Issue**: `OpenAI API key not found`

- **Solution**: Set environment variable

```python
os.environ['OPENAI_API_KEY'] = 'sk-...'
```

**Issue**: ChromaDB persistence errors in Colab

- **Solution**: Use Google Drive path

```python
persist_dir = '/content/drive/MyDrive/chromadb'
```

## Examples

The examples above use locally prepared data. The original extraction notebooks are retained privately; they are not part of this source release.

## Contributing

See [CONTRIBUTING.md](../../CONTRIBUTING.md). Preserve loader input contracts, chunk metadata, provenance, and content hashes when changing preprocessing.

## License

See [LICENSE](../../LICENSE) file in project root.

## Support

For issues, questions, or contributions:

- Open an issue on GitHub
- Refer to [ARCHITECTURE.md](../../ARCHITECTURE.md)
- Check test files for usage examples
