"""
OSHA Regulations Scraper using Selenium

This module provides functionality to scrape regulations data from the
OSHA website using Selenium WebDriver to bypass bot protection.

REQUIREMENTS:
- Chrome browser installed
- ChromeDriver installed (or undetected-chromedriver)
- Selenium package: pip install selenium

USAGE:
    scraper = OSHARegulationsScraper()
    results = scraper.run_scraper(limit=10)
"""

import json
import logging
import os
import re
import time
from datetime import datetime
from typing import Dict, List, Optional
from urllib.parse import urljoin

from src.data_collection.config import (
    OSHA_REGULATIONS_BASE_URL,
    OSHA_REGULATIONS_SUMMARY_FILENAME,
    SCRAPER_REGULATION_SLEEP_DELAY,
    WEBDRIVER_WAIT_TIMEOUT,
)

# Default Selenium-related symbols so attribute access works even if Selenium
# is not installed. These will be overwritten if the real Selenium imports
# succeed below.


class _WebDriverStub:
    """
    Placeholder webdriver object used when Selenium is unavailable.

    Defining a Chrome attribute ensures that test-time patches like
    `patch(\"src.data_collection.osha_regulations_scraper.webdriver.Chrome\")`
    succeed even if Selenium isn't installed and this module was imported
    before the tests apply their sys.modules mocks.
    """

    Chrome = object()


webdriver = _WebDriverStub()  # type: ignore[assignment]
TimeoutException = Exception  # type: ignore[assignment]
Options = object  # type: ignore[assignment]
By = object  # type: ignore[assignment]
EC = object  # type: ignore[assignment]
WebDriverWait = object  # type: ignore[assignment]

# Selenium imports
try:
    from selenium import webdriver as _webdriver
    from selenium.common.exceptions import TimeoutException as _TimeoutException
    from selenium.webdriver.chrome.options import Options as _Options
    from selenium.webdriver.common.by import By as _By
    from selenium.webdriver.support import expected_conditions as _EC
    from selenium.webdriver.support.ui import WebDriverWait as _WebDriverWait

    webdriver = _webdriver
    TimeoutException = _TimeoutException
    Options = _Options
    By = _By
    EC = _EC
    WebDriverWait = _WebDriverWait

    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False
    logging.warning("Selenium not installed. Install with: pip install selenium")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OSHARegulationsScraper:
    """Scraper for OSHA regulations data using Selenium WebDriver."""

    def __init__(
        self, base_url: str = OSHA_REGULATIONS_BASE_URL, data_dir: str = "data/raw"
    ):
        """
        Initialize the OSHA regulations scraper with Selenium.

        Args:
            base_url: Base URL for the OSHA website
            data_dir: Directory to store scraped data
        """
        if not SELENIUM_AVAILABLE:
            raise ImportError(
                "Selenium is not installed. Install with: pip install selenium\n"
                "Also ensure Chrome browser and ChromeDriver are installed."
            )

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
        self.osha_dir = os.path.join(self.data_dir, "osha", "regulations")

        # Create data directories if they don't exist
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir, exist_ok=True)
        if not os.path.exists(self.osha_dir):
            os.makedirs(self.osha_dir, exist_ok=True)

        self.driver = None

    def setup_driver(self):
        """Set up the Chrome WebDriver with optimal settings."""
        logger.info("Setting up Chrome WebDriver...")

        # Selenium discovers ChromeDriver, with a system driver fallback.
        self._setup_standard_driver()

    def _setup_standard_driver(self):
        """Set up standard ChromeDriver."""
        logger.info("Setting up standard ChromeDriver")

        options = Options()
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option("useAutomationExtension", False)
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-infobars")
        options.add_argument("--disable-extensions")

        # Try to create Chrome driver - let selenium find it automatically
        try:
            logger.info("Attempting to start Chrome...")
            # Try with options only first
            self.driver = webdriver.Chrome(options=options)
        except Exception as e1:
            logger.error(f"Failed with default ChromeDriver: {e1}")
            logger.info(
                "ChromeDriver version mismatch. Attempting to continue with available driver..."
            )
            # Try without service for compatibility
            try:
                import shutil

                from selenium.webdriver.chrome.service import Service

                chromedriver_path = shutil.which("chromedriver")
                if chromedriver_path:
                    logger.info(f"Trying system ChromeDriver: {chromedriver_path}")
                    service = Service(chromedriver_path)
                    self.driver = webdriver.Chrome(service=service, options=options)
                else:
                    raise FileNotFoundError
            except Exception as e2:
                logger.error(f"Failed to start Chrome: {e2}")
                logger.error(
                    "\nTROUBLESHOOTING:\n1. Update ChromeDriver: brew upgrade chromedriver"
                )
                logger.error(
                    "2. Or use: chromedriver --version to check current version"
                )
                logger.error("3. Chrome version: Check Chrome -> About Chrome")
                raise

    def get_regulation_links(
        self,
        url: str = "https://www.osha.gov/laws-regs/regulations/standardnumber/1910",
    ) -> List[Dict[str, str]]:
        """
        Get all regulation links from the OSHA regulations page.

        Args:
            url: URL of the regulations listing page

        Returns:
            List of dictionaries containing regulation information
        """
        logger.info(f"Fetching regulation links from {url}")

        if not self.driver:
            self.setup_driver()

        try:
            self.driver.get(url)

            # Wait for page to load
            WebDriverWait(self.driver, WEBDRIVER_WAIT_TIMEOUT).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )

            logger.info(f"Page loaded. Title: {self.driver.title}")

            # Add a small delay to ensure page is fully loaded
            time.sleep(SCRAPER_REGULATION_SLEEP_DELAY)

            # Parse the page with BeautifulSoup
            from bs4 import BeautifulSoup

            soup = BeautifulSoup(self.driver.page_source, "html.parser")

            regulations = []

            # Find all links that point to individual regulations
            all_links = soup.find_all("a", href=True)

            for link in all_links:
                href = link.get("href", "")
                text = link.get_text(strip=True)

                # Filter for regulation links
                if "/laws-regs/regulations/standardnumber/1910/" in href:
                    # Get regulation number from the URL
                    parts = href.rstrip("/").split("/")
                    if len(parts) >= 1:
                        reg_number = parts[-1]

                        # Only include if it's a valid regulation number
                        if reg_number.startswith("1910.") or reg_number.isdigit():
                            full_url = (
                                urljoin(self.base_url, href)
                                if not href.startswith("http")
                                else href
                            )

                            regulations.append(
                                {"number": reg_number, "title": text, "url": full_url}
                            )

            # Remove duplicates based on URL
            seen_urls = set()
            unique_regulations = []
            for reg in regulations:
                if reg["url"] not in seen_urls:
                    seen_urls.add(reg["url"])
                    unique_regulations.append(reg)

            logger.info(f"Found {len(unique_regulations)} unique regulations")
            return unique_regulations

        except TimeoutException:
            logger.error("Timeout waiting for page to load")
            return []
        except Exception as e:
            logger.error(f"Error fetching regulation links: {e}")
            return []

    def extract_regulation_content(self, url: str) -> Optional[Dict[str, str]]:
        """
        Extract content from a single regulation page with structured format.

        Args:
            url: URL of the regulation page

        Returns:
            Dictionary containing regulation content, or None if extraction fails
        """
        try:
            logger.info(f"Extracting content from {url}")

            self.driver.get(url)

            # Wait for page to load
            WebDriverWait(self.driver, WEBDRIVER_WAIT_TIMEOUT).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )

            # Small delay to ensure dynamic content loads
            time.sleep(SCRAPER_REGULATION_SLEEP_DELAY)

            # Parse with BeautifulSoup
            from bs4 import BeautifulSoup

            soup = BeautifulSoup(self.driver.page_source, "html.parser")

            # Extract metadata from the regulation page
            metadata = {}

            # Extract header metadata from bulleted list items
            # Find the ul with the right class, then get all li elements
            header_list = soup.find("ul", class_="bulleted-list-header")
            header_items = []
            if header_list:
                header_items = header_list.find_all("li")

            logger.debug(f"Found {len(header_items)} header items")

            for item in header_items:
                # Get all text in the list item
                full_text = item.get_text(separator=" ", strip=True)
                logger.debug(f"Processing header item: {full_text[:100]}")

                # Try to parse "Label: Value" format
                if ":" in full_text:
                    parts = full_text.split(":", 1)
                    if len(parts) == 2:
                        label = parts[0].strip()
                        value = parts[1].strip()

                        # Store the metadata
                        if label and value:
                            metadata[label] = value
                            logger.debug(f"Extracted metadata: {label} = {value[:50]}")

            # Find the main content area with regulations
            article = soup.find("article")
            if article:
                # Extract regulation paragraphs
                paragraphs = []

                # Find all regulation paragraphs
                reg_paragraphs = article.find_all(
                    "div", class_="field--name-field-standard-paragraph-body-p"
                )

                for para in reg_paragraphs:
                    # Get the section ID (like 1910.3(a))
                    parent_para = para.find_parent(
                        "div", class_="paragraph--type--regulations-standard-number"
                    )
                    if parent_para:
                        section_id = parent_para.find("span", id=re.compile(r"^1910\."))
                        if section_id:
                            section_number = section_id.get_text(strip=True)
                            para_text = para.get_text(strip=True)
                            paragraphs.append(
                                {"section": section_number, "text": para_text}
                            )

                content = {"metadata": metadata, "paragraphs": paragraphs}
            else:
                # Fallback if structure is different
                content_div = (
                    soup.find("div", {"role": "main"})
                    or soup.find("div", class_="content")
                    or soup.find("main")
                )

                if content_div:
                    # Extract text content
                    content = {
                        "metadata": metadata,
                        "text": content_div.get_text(separator="\n", strip=True),
                    }
                else:
                    content = {
                        "metadata": metadata,
                        "text": soup.get_text(separator="\n", strip=True),
                    }

            return content

        except Exception as e:
            logger.error(f"Error extracting content from {url}: {e}")
            return None

    def sanitize_filename(self, filename: str) -> str:
        """Sanitize filename to be valid on most file systems."""
        for char in [":", "/", "\\", "?", "*", '"', "<", ">", "|", "."]:
            if char == "." and filename.count(".") <= 1:
                continue
            filename = filename.replace(char, "_")
        filename = re.sub(r"[\s_]+", "_", filename)
        filename = filename.strip("_ ")
        return filename

    def save_regulation(self, regulation_data: Dict, reg_number: str) -> Optional[str]:
        """
        Save regulation data to a JSON file.

        Args:
            regulation_data: Dictionary containing regulation data
            reg_number: Regulation number (used for filename)

        Returns:
            Path to saved file if successful, None otherwise
        """
        try:
            filename = f"{reg_number}.json"
            filename = self.sanitize_filename(filename)

            filepath = os.path.join(self.osha_dir, filename)

            # Skip if file already exists
            if os.path.exists(filepath):
                logger.info(f"File already exists, skipping: {filename}")
                return filepath

            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(regulation_data, f, indent=2, ensure_ascii=False)

            logger.info(f"Saved regulation to {filename}")
            return filepath

        except Exception as e:
            logger.error(f"Error saving regulation {reg_number}: {e}")
            return None

    def process_regulation(
        self, reg_url: str, reg_number: str, reg_title: str
    ) -> Optional[Dict]:
        """Process a single regulation and return its data."""
        logger.info(f"Processing regulation {reg_number}: {reg_title}")

        content = self.extract_regulation_content(reg_url)

        if not content:
            logger.warning(f"Could not extract content for {reg_number}")
            return None

        # Structure the data as requested - flatten metadata to top level
        metadata = content.get("metadata", {})

        regulation_data = {
            # Add metadata fields at top level
            "Part Number": metadata.get("Part Number", ""),
            "Part Number Title": metadata.get("Part Number Title", ""),
            "Subpart": metadata.get("Subpart", ""),
            "Subpart Title": metadata.get("Subpart Title", ""),
            "Standard Number": metadata.get("Standard Number", reg_number),
            "Title": reg_title,
            "GPO Source": metadata.get("GPO Source", ""),
            "url": reg_url,
            "scraped_at": datetime.now().isoformat(),
        }

        # Add paragraphs if available
        if "paragraphs" in content:
            regulation_data["paragraphs"] = content["paragraphs"]
        elif "text" in content:
            regulation_data["content"] = content["text"]

        filepath = self.save_regulation(regulation_data, reg_number)

        return {
            "number": reg_number,
            "title": reg_title,
            "url": reg_url,
            "filepath": filepath,
            "content_length": len(str(content)),
        }

    def create_summary_json(self, results: List[Dict]) -> str:
        """Create a summary JSON file with metadata about all scraped regulations."""
        summary = {
            "scraped_at": datetime.now().isoformat(),
            "total_regulations": len(results),
            "successful": sum(1 for r in results if r.get("filepath")),
            "failed": sum(1 for r in results if not r.get("filepath")),
            "regulations": [
                {
                    "number": r["number"],
                    "title": r["title"],
                    "url": r["url"],
                    "filepath": r.get("filepath"),
                    "status": "success" if r.get("filepath") else "failed",
                }
                for r in results
            ],
        }

        # Save summary in parent directory (osha/ not osha/regulations/)
        summary_dir = os.path.dirname(self.osha_dir)
        summary_path = os.path.join(summary_dir, OSHA_REGULATIONS_SUMMARY_FILENAME)

        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        logger.info(f"Summary saved to {summary_path}")
        return summary_path

    def run_scraper(self, limit: Optional[int] = None) -> List[Dict]:
        """
        Run the OSHA regulations scraper.

        Args:
            limit: Optional limit on number of regulations to scrape (for testing)

        Returns:
            List of processing results for each regulation
        """
        logger.info("Starting OSHA regulations scraping with Selenium...")

        if not self.driver:
            self.setup_driver()

        try:
            # Get all regulation links
            regulations = self.get_regulation_links()

            if not regulations:
                logger.error("No regulations found")
                return []

            # Limit if specified
            regulations_to_process = regulations[:limit] if limit else regulations
            logger.info(f"Processing {len(regulations_to_process)} regulations")

            results = []

            for i, reg in enumerate(regulations_to_process):
                logger.info(
                    f"Processing {i+1}/{len(regulations_to_process)}: {reg['number']}"
                )

                try:
                    result = self.process_regulation(
                        reg["url"], reg["number"], reg["title"]
                    )

                    if result:
                        results.append(result)

                    # Add delay between requests to be respectful
                    time.sleep(SCRAPER_REGULATION_SLEEP_DELAY)

                except Exception as e:
                    logger.error(f"Error processing {reg['number']}: {e}")
                    results.append(
                        {
                            "number": reg["number"],
                            "title": reg["title"],
                            "url": reg["url"],
                            "filepath": None,
                            "error": str(e),
                        }
                    )

            # Create summary
            summary_path = self.create_summary_json(results)

            successful = sum(1 for r in results if r.get("filepath"))
            logger.info(
                f"OSHA regulations scraping completed. "
                f"Successfully scraped {successful}/{len(regulations_to_process)} regulations"
            )
            logger.info(f"Summary saved to {summary_path}")

            return results

        finally:
            # Always close the driver
            if self.driver:
                self.driver.quit()
                logger.info("ChromeDriver closed")


def main():
    """Main function to run the OSHA regulations scraper with Selenium."""
    try:
        scraper = OSHARegulationsScraper()
        results = scraper.run_scraper()
        return results
    except Exception as e:
        logger.error(f"Failed to run scraper: {e}")
        logger.error(
            "\nINSTALLATION INSTRUCTIONS:\n"
            "1. pip install selenium\n"
            "2. Install Chrome browser from https://www.google.com/chrome/\n"
            "3. Selenium discovers ChromeDriver or uses chromedriver on PATH"
        )
        raise


if __name__ == "__main__":
    main()
