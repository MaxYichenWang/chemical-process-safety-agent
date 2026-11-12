"""Collect CSB investigations, OSHA enforcement data, and OSHA regulations."""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Import scrapers via the src.data_collection package so tests and package usage
# both resolve these modules consistently.
from src.data_collection.config import (
    CSB_BASE_URL,
    CSB_DATA_FILENAME,
    CSB_LINKS_FILENAME,
)
from src.data_collection.csb_investigation_scraper import CSBInvestigationScraper
from src.data_collection.osha_enforcement_data_scraper import OSHAEnforcementDataScraper
from src.data_collection.osha_regulations_scraper import OSHARegulationsScraper

# Get project root directory (two levels up from this file)
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
DEFAULT_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw")


def scrape_csb_data(data_dir: str = DEFAULT_DATA_DIR, limit: Optional[int] = None):
    """
    Scrape CSB investigation data.

    Args:
        data_dir: Directory to store scraped data
        limit: Optional limit on number of investigations to process
    """
    print("🔍 Starting CSB data scraping...")

    scraper = CSBInvestigationScraper(data_dir=data_dir)

    url_base = CSB_BASE_URL + "/investigations/completed-investigations/"
    investigation_links_csv_file_path = os.path.join(data_dir, CSB_LINKS_FILENAME)

    if os.path.exists(investigation_links_csv_file_path):
        try:
            import pandas as pd

            links_df = pd.read_csv(investigation_links_csv_file_path)
            investigation_links = links_df["url"].tolist()
            print(
                f"Loaded {len(investigation_links)} investigation links from CSV file."
            )
        except ImportError:
            print("pandas not available, falling back to manual CSV reading")
            investigation_links = []
            with open(investigation_links_csv_file_path) as f:
                lines = f.readlines()
                for line in lines[1:]:  # Skip header
                    investigation_links.append(
                        line.strip().split(",")[1]
                    )  # URL is second column
            print(
                f"Loaded {len(investigation_links)} investigation links from CSV file."
            )
    else:
        investigation_links = scraper.get_investigation_links(url_base)
        print(
            f"Downloaded {len(investigation_links)} investigation links from website."
        )

    csv_file_name = CSB_DATA_FILENAME
    results = scraper.run_scraper(investigation_links, csv_file_name, limit=limit)

    print(f"✅ CSB scraping completed. Processed {len(results)} investigations.")
    return results


def scrape_osha_data(data_dir: str = DEFAULT_DATA_DIR, limit: Optional[int] = None):
    """
    Scrape OSHA enforcement data.

    Args:
        data_dir: Directory to store scraped data
        limit: Optional limit on number of files to download (for testing)
    """
    print("🔍 Starting OSHA data scraping...")

    scraper = OSHAEnforcementDataScraper(data_dir=data_dir)
    results = scraper.run_scraper(limit=limit)

    if not results:
        raise RuntimeError(
            "No OSHA enforcement download links were found. "
            "Use the manual acquisition instructions in src/data_collection/README.md."
        )

    successful_downloads = sum(
        1 for result in results.values() if result.get("file_path")
    )
    total_files = len(results)
    if successful_downloads == 0:
        raise RuntimeError(
            "No OSHA enforcement files were downloaded; inspect the collection logs."
        )

    print(
        f"✅ OSHA scraping completed. Successfully downloaded {successful_downloads}/{total_files} files."
    )
    return results


def scrape_osha_regulations(
    data_dir: str = DEFAULT_DATA_DIR, limit: Optional[int] = None
):
    """
    Scrape OSHA regulations data.

    Args:
        data_dir: Directory to store scraped data
        limit: Optional limit on number of regulations to scrape (for testing)
    """
    print("🔍 Starting OSHA regulations scraping...")

    # Use Selenium version for better success rate
    scraper = OSHARegulationsScraper(data_dir=data_dir)
    results = scraper.run_scraper(limit=limit)

    successful = sum(1 for result in results if result.get("filepath"))
    total_regulations = len(results)

    print(
        f"✅ OSHA regulations scraping completed. Successfully scraped {successful}/{total_regulations} regulations."
    )
    return results


def full_pipeline(data_dir: str = DEFAULT_DATA_DIR, scrape_limit: Optional[int] = None):
    """
    Run the complete data processing pipeline.

    Args:
        data_dir: Directory to store all data
        scrape_limit: Optional limit on number of investigations/files to scrape
    """
    print("🚀 Starting data processing pipeline...")

    scrape_results = scrape_csb_data(data_dir, limit=scrape_limit)

    osha_results = scrape_osha_data(data_dir, limit=scrape_limit)

    print("🎉 Pipeline completed!")

    return {"scrape_results": scrape_results, "osha_results": osha_results}


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(
        description="Chemical Process Safety Data Processing"
    )

    parser.add_argument(
        "--mode",
        choices=["csb", "osha", "osha_regulations", "full"],
        default="full",
        help="Processing mode: csb (CSB data only), osha (OSHA data only), osha_regulations (OSHA regulations), or full (both CSB and OSHA)",
    )

    parser.add_argument(
        "--data-dir",
        default=DEFAULT_DATA_DIR,
        help=f"Directory to store/read data (default: {DEFAULT_DATA_DIR})",
    )

    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of investigations/files to scrape (for testing)",
    )

    args = parser.parse_args()

    os.makedirs(args.data_dir, exist_ok=True)

    try:
        if args.mode == "csb":
            scrape_csb_data(args.data_dir, args.limit)
        elif args.mode == "osha":
            scrape_osha_data(args.data_dir, args.limit)
        elif args.mode == "osha_regulations":
            scrape_osha_regulations(args.data_dir, args.limit)
        elif args.mode == "full":
            full_pipeline(args.data_dir, args.limit)
    except RuntimeError as exc:
        parser.exit(1, f"Collection failed: {exc}\n")


if __name__ == "__main__":
    main()
