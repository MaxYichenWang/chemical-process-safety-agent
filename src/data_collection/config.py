"""
Configuration module for data collection scrapers.
"""

import os

from dotenv import load_dotenv

load_dotenv()

# Base URLs
CSB_BASE_URL = os.getenv("CSB_BASE_URL", "https://www.csb.gov")
OSHA_ENFORCEMENT_BASE_URL = os.getenv(
    "OSHA_ENFORCEMENT_BASE_URL", "https://enforcedata.dol.gov"
)
OSHA_REGULATIONS_BASE_URL = os.getenv(
    "OSHA_REGULATIONS_BASE_URL", "https://www.osha.gov"
)

# HTTP Settings
SCRAPER_USER_AGENT = os.getenv(
    "SCRAPER_USER_AGENT",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36",
)
SCRAPER_REQUEST_TIMEOUT = int(os.getenv("SCRAPER_REQUEST_TIMEOUT", "30"))
SCRAPER_SLEEP_DELAY = int(os.getenv("SCRAPER_SLEEP_DELAY", "1"))
SCRAPER_REGULATION_SLEEP_DELAY = int(os.getenv("SCRAPER_REGULATION_SLEEP_DELAY", "2"))
WEBDRIVER_WAIT_TIMEOUT = int(os.getenv("WEBDRIVER_WAIT_TIMEOUT", "30"))

# Download Settings
DOWNLOAD_CHUNK_SIZE = int(os.getenv("DOWNLOAD_CHUNK_SIZE", "8192"))
LARGE_FILE_THRESHOLD = int(os.getenv("LARGE_FILE_THRESHOLD", str(50 * 1024 * 1024)))
CSV_CHUNK_ROWS = int(os.getenv("CSV_CHUNK_ROWS", "10000"))

# Output Filenames
CSB_LINKS_FILENAME = os.getenv("CSB_LINKS_FILENAME", "csb_investigation_links.csv")
CSB_DATA_FILENAME = os.getenv("CSB_DATA_FILENAME", "csb_investigation_data.csv")
OSHA_ENFORCEMENT_SUMMARY_FILENAME = os.getenv(
    "OSHA_ENFORCEMENT_SUMMARY_FILENAME", "osha_enforcement_data_summary.csv"
)
OSHA_REGULATIONS_SUMMARY_FILENAME = os.getenv(
    "OSHA_REGULATIONS_SUMMARY_FILENAME", "osha_regulations_summary.json"
)
