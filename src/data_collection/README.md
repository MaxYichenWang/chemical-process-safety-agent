# Data collection

The collection package downloads CSB investigation documents and OSHA regulation pages. Follow the [source checkout setup](../../README.md#quick-start-from-a-source-checkout) and run commands from the checkout root.

## Collect CSB investigations

```bash
python -m src.data_collection.main --mode csb
```

Use `--limit 5` to process five investigations. Listing investigation URLs still scans the completed-investigation pages. Use an absolute output path for a custom location.

```bash
python -m src.data_collection.main --mode csb --limit 5 --data-dir "$PWD/my_data"
```

The default output is `data/raw/csb/`.

```text
data/raw/csb/
  csb_investigation_links.csv
  csb_investigation_data.csv
  csb_investigation_index.json
  <investigation>/
    investigation_data.json
    documents/
      final_reports/
      recommendations/
      related_documents/
```

Existing document files and investigation JSON are reused. Collection summaries record source URLs. Preserve download dates with your corpus.

## Obtain OSHA enforcement data

The shipped enforcement scraper targets `https://enforcedata.dol.gov/views/data_summary.php`. That endpoint redirects to the DOL portal and returned no download links in the release review. Download bulk records through the [DOL data portal](https://data.dol.gov/) using the [OSHA data page](https://www.osha.gov/data), then prepare the inputs in the [processing guide](../data_processing/README.md#rebuild-inputs-from-public-sources).

The legacy command remains available.

```bash
python -m src.data_collection.main --mode osha --limit 1
```

When links are available, it writes `osha_*.csv` files and `osha_enforcement_data_summary.csv` under `data/raw/osha/`. The command exits unsuccessfully when it finds no links or downloads no files. Inspect the output summaries before proceeding.

## Collect OSHA regulations

Install Chrome and a compatible ChromeDriver. The scraper uses Selenium Manager, with a system ChromeDriver fallback. Driver setup may require internet access.

```bash
python -m src.data_collection.main --mode osha_regulations
```

This collects Part 1910 pages into `data/raw/osha/regulations/<standard>.json` and writes `data/raw/osha/osha_regulations_summary.json`. Use `--limit` for a smaller run. Source-site access restrictions may prevent collection.

The `full` mode runs CSB and OSHA enforcement collection. Run `osha_regulations` separately.

## Python interfaces

`CSBInvestigationScraper(data_dir=...)` exposes `get_investigation_links(url)` and `run_scraper(links, csv_file, limit=...)`. `OSHAEnforcementDataScraper(data_dir=...)` and `OSHARegulationsScraper(data_dir=...)` expose `run_scraper(limit=...)`. Relative `data_dir` values resolve against the package project root. Use an absolute path when calling an installed package.

See [config.py](config.py) and [.env.example](../../.env.example) for source URLs, request timeouts, delays, and output filenames. Scrapers log individual failures and may return partial results. Inspect the summaries and preserve source provenance before normalization.

## Development

```bash
python -m pytest tests/data_collection/
```

See [CONTRIBUTING.md](../../CONTRIBUTING.md) for the complete checks and [LICENSE](../../LICENSE) for the framework license.
