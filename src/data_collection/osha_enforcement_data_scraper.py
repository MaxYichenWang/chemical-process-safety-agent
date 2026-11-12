"""
OSHA Enforcement Data Scraper

This module provides functionality to scrape enforcement data from the
OSHA Enforcement Data Catalog (https://enforcedata.dol.gov/views/data_summary.php),
including various datasets related to workplace safety inspections and violations.
"""

import csv
import logging
import os
import re
import time
from datetime import datetime
from typing import Dict, Optional

import pandas as pd
import requests
from bs4 import BeautifulSoup

from src.data_collection.config import (
    CSV_CHUNK_ROWS,
    DOWNLOAD_CHUNK_SIZE,
    LARGE_FILE_THRESHOLD,
    OSHA_ENFORCEMENT_BASE_URL,
    OSHA_ENFORCEMENT_SUMMARY_FILENAME,
    SCRAPER_REQUEST_TIMEOUT,
    SCRAPER_SLEEP_DELAY,
    SCRAPER_USER_AGENT,
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OSHAEnforcementDataScraper:
    """Scraper for OSHA enforcement data."""

    def __init__(
        self, base_url: str = OSHA_ENFORCEMENT_BASE_URL, data_dir: str = "data/raw"
    ):
        """
        Initialize the OSHA scraper.

        Args:
            base_url: Base URL for the OSHA enforcement data website
            data_dir: Directory to store scraped data
        """
        self.base_url = base_url
        # Convert to absolute path if relative
        if not os.path.isabs(data_dir):
            # Get the directory containing this file (src/data_collection)
            current_dir = os.path.dirname(os.path.abspath(__file__))
            # Go up two levels to project root
            project_root = os.path.dirname(os.path.dirname(current_dir))
            self.data_dir = os.path.join(project_root, data_dir)
        else:
            self.data_dir = data_dir
        self.osha_dir = os.path.join(self.data_dir, "osha")
        self.headers = {"User-Agent": SCRAPER_USER_AGENT}

        # Create data directories if they don't exist
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir, exist_ok=True)
        if not os.path.exists(self.osha_dir):
            os.makedirs(self.osha_dir, exist_ok=True)

        # Define the files to download with their metadata
        self.files_to_download = {
            "osha_accident": {
                "refresh_frequency": "Periodically (currently reflects data through Present)",
                "description": "Accident data",
            },
            "osha_accident_abstract": {
                "refresh_frequency": "Periodically (currently reflects data through Present)",
                "description": "Accident abstract data",
            },
            "osha_accident_injury": {
                "refresh_frequency": "Periodically (currently reflects data through Present)",
                "description": "Accident injury data",
            },
            "osha_accident_lookup": {
                "refresh_frequency": "As required",
                "description": "Accident lookup data",
            },
            "osha_data_dictionary": {
                "refresh_frequency": "As required",
                "description": "Data dictionary",
            },
            "osha_inspection": {
                "refresh_frequency": "Daily",
                "description": "Inspection data",
            },
            "osha_metadata": {"refresh_frequency": "Daily", "description": "Metadata"},
            "osha_optional_info": {
                "refresh_frequency": "Daily",
                "description": "Optional information",
            },
            "osha_related_activity": {
                "refresh_frequency": "Daily",
                "description": "Related activity data",
            },
            "osha_strategic_codes": {
                "refresh_frequency": "Daily",
                "description": "Strategic codes",
            },
            "osha_violation": {
                "refresh_frequency": "Daily",
                "description": "Violation data",
            },
            "osha_violation_event": {
                "refresh_frequency": "Daily",
                "description": "Violation event data",
            },
            "osha_violation_gen_duty_std": {
                "refresh_frequency": "Daily",
                "description": "Violation general duty standard data",
            },
        }

    def get_download_urls(self) -> Dict[str, str]:
        """
        Get download URLs for all OSHA files by submitting form to data_summary.php.

        Returns:
            Dictionary mapping filename to download URL
        """
        logger.info(
            "Fetching download URLs from OSHA data summary page via form submission"
        )

        url = f"{self.base_url}/views/data_summary.php"
        form_data = {"agency": "osha"}

        try:
            response = requests.post(url, data=form_data, headers=self.headers)

            if response.status_code != 200:
                logger.error(
                    f"Failed to fetch {url}, status code: {response.status_code}"
                )
                return {}

            soup = BeautifulSoup(response.text, "html.parser")
            download_urls = {}

            # Look for CSV download links
            links = soup.find_all("a", href=True)

            for link in links:
                href = link.get("href", "")

                # Check if this is a CSV download link
                if "csv.zip" in href.lower() and "enfxfr.dol.gov" in href:
                    # Extract filename from URL - handle patterns like:
                    # osha_inspection_20251025.csv.zip -> osha_inspection
                    # osha_related_activity_20251025.csv.zip -> osha_related_activity
                    # osha_accident_lookup2_20251025.csv.zip -> osha_accident_lookup

                    # Match pattern: osha_<name>_<date>.csv.zip
                    # This handles: osha_inspection_20251025.csv.zip -> inspection
                    # And: osha_related_activity_20251025.csv.zip -> related_activity
                    # And: osha_accident_lookup2_20251025.csv.zip -> accident_lookup2
                    # Match everything up to the date pattern (8 digits) before .csv.zip
                    filename_match = re.search(
                        r"osha_([^_\s]+(?:_[^_\s]+)*?)(?:_\d{8})?\.csv\.zip",
                        href.lower(),
                    )
                    if filename_match:
                        base_name = filename_match.group(1)
                        filename = f"osha_{base_name}"

                        # Map to our expected filenames
                        filename_mapping = {
                            "osha_accident_lookup2": "osha_accident_lookup",
                        }

                        mapped_filename = filename_mapping.get(filename, filename)

                        if mapped_filename in self.files_to_download:
                            download_urls[mapped_filename] = href
                            logger.info(
                                f"Found download URL for {mapped_filename}: {href}"
                            )

            return download_urls

        except Exception as e:
            logger.error(f"Error fetching download URLs: {e}")
            return {}

    def download_file(self, url: str, filename: str) -> Optional[str]:
        """
        Download a file from the given URL and extract if it's a ZIP file.

        Args:
            url: URL to download from
            filename: Name of the file to save

        Returns:
            Path to downloaded file if successful, None otherwise
        """
        try:
            logger.info(f"Downloading {filename} from {url}")

            response = requests.get(
                url, headers=self.headers, stream=True, timeout=SCRAPER_REQUEST_TIMEOUT
            )

            if response.status_code == 200:
                # Determine file extension based on URL
                if url.endswith(".zip"):
                    filepath = os.path.join(self.osha_dir, f"{filename}.zip")
                    csv_filepath = os.path.join(self.osha_dir, f"{filename}.csv")
                else:
                    filepath = os.path.join(self.osha_dir, f"{filename}.csv")
                    csv_filepath = filepath

                # Skip if CSV file already exists
                if os.path.exists(csv_filepath):
                    logger.info(
                        f"CSV file already exists, skipping download: {filename}.csv"
                    )
                    return csv_filepath

                # Download the file
                with open(filepath, "wb") as f:
                    for chunk in response.iter_content(chunk_size=DOWNLOAD_CHUNK_SIZE):
                        f.write(chunk)

                logger.info(f"Successfully downloaded: {os.path.basename(filepath)}")

                # Extract ZIP file if needed
                if url.endswith(".zip"):
                    try:
                        import zipfile

                        with zipfile.ZipFile(filepath, "r") as zip_ref:
                            # List all files in the ZIP
                            file_list = zip_ref.infolist()
                            logger.info(
                                f"Files in ZIP: {[f.filename for f in file_list]}"
                            )

                            # Extract the CSV file(s)
                            csv_files = [
                                f for f in file_list if f.filename.endswith(".csv")
                            ]

                            if not csv_files:
                                logger.warning(f"No CSV file found in ZIP: {filepath}")
                                return None

                            # Extract all files from the ZIP to handle nested structures
                            zip_ref.extractall(self.osha_dir)

                            # Collect all extracted CSV file paths
                            extracted_csv_paths = []
                            zip_csv_basenames = {
                                os.path.basename(csv_f.filename) for csv_f in csv_files
                            }

                            for root, _dirs, files in os.walk(self.osha_dir):
                                for f in files:
                                    file_path = os.path.join(root, f)
                                    # Find recently extracted CSV files from the ZIP
                                    if (
                                        file_path.endswith(".csv")
                                        and os.path.basename(file_path)
                                        in zip_csv_basenames
                                    ):
                                        extracted_csv_paths.append(file_path)

                            if not extracted_csv_paths:
                                logger.error("Could not find extracted CSV files")
                                return None

                            # Sort CSV files by name for consistent concatenation
                            extracted_csv_paths = sorted(extracted_csv_paths)

                            # Separate files that match target name from those that don't
                            matching_files = [
                                p
                                for p in extracted_csv_paths
                                if os.path.basename(p) == f"{filename}.csv"
                            ]
                            non_matching_files = [
                                p
                                for p in extracted_csv_paths
                                if os.path.basename(p) != f"{filename}.csv"
                            ]

                            logger.info(
                                f"Found {len(extracted_csv_paths)} CSV files: {[os.path.basename(p) for p in extracted_csv_paths]}"
                            )

                            final_path = os.path.join(self.osha_dir, f"{filename}.csv")

                            # If only one file and it already has the correct name, just return it
                            if (
                                len(extracted_csv_paths) == 1
                                and len(matching_files) == 1
                            ):
                                logger.info(
                                    f"Single file with correct name: {filename}.csv"
                                )
                            else:
                                # Concatenate all CSV files (both matching and non-matching)
                                all_files = matching_files + non_matching_files
                                dfs = []
                                for csv_path in all_files:
                                    logger.info(
                                        f"Reading {os.path.basename(csv_path)}..."
                                    )
                                    df = pd.read_csv(csv_path, low_memory=False)
                                    logger.info(
                                        f"  Loaded {len(df):,} rows from {os.path.basename(csv_path)}"
                                    )
                                    dfs.append(df)

                                combined_df = pd.concat(dfs, ignore_index=True)
                                logger.info(
                                    f"Total concatenated rows: {len(combined_df):,}"
                                )

                                # Save the concatenated file
                                combined_df.to_csv(final_path, index=False)
                                logger.info(f"Saved concatenated file: {filename}.csv")

                                # Clean up ALL original CSV files since we've saved them to final_path
                                for csv_path in all_files:
                                    try:
                                        os.remove(csv_path)
                                        logger.info(
                                            f"Removed source file: {os.path.basename(csv_path)}"
                                        )
                                    except Exception as e:
                                        logger.warning(
                                            f"Could not remove {csv_path}: {e}"
                                        )

                            # Clean up any empty directories left behind
                            for root, _dirs, _files in os.walk(
                                self.osha_dir, topdown=False
                            ):
                                for d in _dirs:
                                    try:
                                        dir_path = os.path.join(root, d)
                                        if not os.listdir(dir_path):
                                            os.rmdir(dir_path)
                                    except Exception:
                                        pass

                            logger.info(
                                f"Extracted and concatenated CSV file: {filename}.csv"
                            )

                            # Clean up ZIP file
                            os.remove(filepath)

                            return final_path
                    except Exception as e:
                        logger.error(f"Error extracting ZIP file {filepath}: {e}")
                        import traceback

                        logger.error(traceback.format_exc())
                        return None

                return filepath
            else:
                logger.error(
                    f"Failed to download {filename}, status code: {response.status_code}"
                )
                return None

        except Exception as e:
            logger.error(f"Error downloading {filename}: {e}")
            return None

    def get_file_info(self, filepath: str) -> Dict:
        """
        Get information about a downloaded file.

        Args:
            filepath: Path to the file

        Returns:
            Dictionary with file information
        """
        info = {
            "file_path": filepath,
            "file_size": 0,
            "record_count": 0,
            "last_modified": None,
            "download_timestamp": datetime.now().isoformat(),
        }

        if os.path.exists(filepath):
            stat = os.stat(filepath)
            info["file_size"] = stat.st_size
            info["last_modified"] = datetime.fromtimestamp(stat.st_mtime).isoformat()

            # Try to count records in CSV
            try:
                # Read headers first
                df = pd.read_csv(filepath, nrows=0)
                info["columns"] = list(df.columns)

                # Count total rows efficiently for large files
                try:
                    # For large files, read in chunks instead of loading entirely into memory
                    file_size = os.path.getsize(filepath)

                    # If file is larger than threshold, use iterative counting
                    if file_size > LARGE_FILE_THRESHOLD:
                        # Count rows by reading file in chunks (more memory efficient)
                        chunk_count = 0
                        chunk_size = CSV_CHUNK_ROWS
                        for chunk in pd.read_csv(
                            filepath, chunksize=chunk_size, low_memory=False
                        ):
                            chunk_count += len(chunk)
                        info["record_count"] = chunk_count
                    else:
                        # For smaller files, read normally
                        df_full = pd.read_csv(filepath)
                        info["record_count"] = len(df_full)

                except Exception:
                    # If we can't read the file, estimate from file size
                    file_size = os.path.getsize(filepath)
                    # Rough estimate: average CSV row is ~500 bytes
                    estimated_rows = file_size // 500
                    info["record_count"] = f"Unknown (~{estimated_rows:,})"

            except Exception as e:
                logger.warning(f"Could not read CSV file {filepath}: {e}")
                info["record_count"] = "Error reading file"

        return info

    def create_overview_csv(self, download_results: Dict[str, Dict]) -> str:
        """
        Create an overview CSV file with information about all downloaded files.

        Args:
            download_results: Results from the download process

        Returns:
            Path to the created overview CSV file
        """
        overview_data = []

        for filename, file_info in download_results.items():
            metadata = self.files_to_download.get(filename, {})

            overview_row = {
                "file_name": filename,
                "description": metadata.get("description", ""),
                "refresh_frequency": metadata.get("refresh_frequency", ""),
                "file_path": file_info.get("file_path", ""),
                "file_size_bytes": file_info.get("file_size", 0),
                "record_count": file_info.get("record_count", 0),
                "columns": ", ".join(file_info.get("columns", [])),
                "last_modified": file_info.get("last_modified", ""),
                "download_timestamp": file_info.get("download_timestamp", ""),
                "download_status": (
                    "Success" if file_info.get("file_path") else "Failed"
                ),
            }

            overview_data.append(overview_row)

        # Save overview CSV
        overview_path = os.path.join(self.osha_dir, OSHA_ENFORCEMENT_SUMMARY_FILENAME)

        with open(overview_path, "w", newline="", encoding="utf-8") as csvfile:
            if overview_data:
                fieldnames = overview_data[0].keys()
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(overview_data)

        logger.info(f"Overview data saved to {overview_path}")
        return overview_path

    def run_scraper(self, limit: Optional[int] = None) -> Dict[str, Dict]:
        """
        Run the OSHA scraper to download all specified files.

        Args:
            limit: Optional limit on number of files to download (for testing)

        Returns:
            Dictionary with download results for each file
        """
        logger.info("Starting OSHA data scraping...")

        # Get download URLs
        download_urls = self.get_download_urls()

        if not download_urls:
            logger.error("No download URLs found")
            return {}

        # Limit files if specified
        files_to_process = list(download_urls.keys())
        if limit:
            files_to_process = files_to_process[:limit]
            logger.info(f"Limited to {limit} files for testing")

        download_results = {}

        for i, filename in enumerate(files_to_process):
            logger.info(f"Processing file {i+1}/{len(files_to_process)}: {filename}")

            url = download_urls.get(filename)
            if not url:
                logger.warning(f"No URL found for {filename}")
                download_results[filename] = {
                    "file_path": None,
                    "error": "No URL found",
                }
                continue

            # Download file
            filepath = self.download_file(url, filename)

            # Get file information
            file_info = self.get_file_info(filepath) if filepath else {}
            file_info["url"] = url

            download_results[filename] = file_info

            # Add delay between downloads to be respectful
            if i < len(files_to_process) - 1:
                time.sleep(SCRAPER_SLEEP_DELAY)

        # Create overview CSV
        overview_path = self.create_overview_csv(download_results)

        successful_downloads = sum(
            1 for result in download_results.values() if result.get("file_path")
        )

        logger.info(
            f"OSHA scraping completed. Successfully downloaded {successful_downloads}/{len(files_to_process)} files"
        )
        logger.info(f"Overview saved to: {overview_path}")

        return download_results


def main():
    """Main function to run the OSHA scraper."""
    scraper = OSHAEnforcementDataScraper()
    results = scraper.run_scraper()
    return results


if __name__ == "__main__":
    main()
