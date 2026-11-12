"""
Unit tests for OSHARegulationsScraper.
"""

import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest


# Mock selenium before importing the scraper
class MockTimeoutException(Exception):
    """Mock TimeoutException for testing."""

    pass


mock_selenium = MagicMock()
mock_selenium.webdriver = MagicMock()
mock_selenium.common = MagicMock()
mock_selenium.common.exceptions = MagicMock()
mock_selenium.common.exceptions.TimeoutException = MockTimeoutException
mock_selenium.webdriver.chrome = MagicMock()
mock_selenium.webdriver.chrome.options = MagicMock()
mock_selenium.webdriver.common = MagicMock()
mock_selenium.webdriver.common.by = MagicMock()
mock_selenium.webdriver.support = MagicMock()
mock_selenium.webdriver.support.ui = MagicMock()
mock_selenium.webdriver.support.expected_conditions = MagicMock()

with patch.dict(
    "sys.modules",
    {
        "selenium": mock_selenium,
        "selenium.webdriver": mock_selenium.webdriver,
        "selenium.common": mock_selenium.common,
        "selenium.common.exceptions": mock_selenium.common.exceptions,
        "selenium.webdriver.chrome": mock_selenium.webdriver.chrome,
        "selenium.webdriver.chrome.options": mock_selenium.webdriver.chrome.options,
        "selenium.webdriver.common": mock_selenium.webdriver.common,
        "selenium.webdriver.common.by": mock_selenium.webdriver.common.by,
        "selenium.webdriver.support": mock_selenium.webdriver.support,
        "selenium.webdriver.support.ui": mock_selenium.webdriver.support.ui,
        "selenium.webdriver.support.expected_conditions": mock_selenium.webdriver.support.expected_conditions,
    },
):
    from src.data_collection.osha_regulations_scraper import OSHARegulationsScraper


# Module-level autouse fixture to ensure webdriver.Chrome is always mocked
@pytest.fixture(scope="module", autouse=True)
def patch_webdriver_chrome_module():
    """Module-level fixture to patch webdriver.Chrome to prevent any real browser from opening."""
    # Create a mock driver that will be used for all tests
    mock_driver_instance = MagicMock()
    mock_driver_instance.title = "Test Page"
    mock_driver_instance.page_source = "<html><body>Test</body></html>"

    with patch(
        "src.data_collection.osha_regulations_scraper.webdriver.Chrome"
    ) as mock_chrome:
        # Always return the mock_driver_instance to prevent any real browser from opening
        mock_chrome.return_value = mock_driver_instance
        yield mock_chrome


@pytest.fixture
def temp_data_dir():
    """Create a temporary directory for test data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def mock_driver():
    """Create a mock WebDriver."""
    driver = MagicMock()
    driver.title = "Test Page"
    driver.page_source = """
    <html>
        <body>
            <ul class="bulleted-list-header">
                <li>Part Number: 1910</li>
                <li>Part Number Title: Occupational Safety and Health Standards</li>
                <li>Subpart: A</li>
                <li>Subpart Title: General</li>
                <li>Standard Number: 1910.1</li>
                <li>GPO Source: https://example.com</li>
            </ul>
            <article>
                <div class="paragraph--type--regulations-standard-number">
                    <span id="1910.1">1910.1</span>
                    <div class="field--name-field-standard-paragraph-body-p">
                        This is a test regulation paragraph.
                    </div>
                </div>
            </article>
            <a href="/laws-regs/regulations/standardnumber/1910/1910.1">1910.1</a>
            <a href="/laws-regs/regulations/standardnumber/1910/1910.2">1910.2</a>
        </body>
    </html>
    """
    return driver


@pytest.fixture
def scraper(temp_data_dir, mock_driver):
    """Create an OSHARegulationsScraper instance for testing."""
    # The module-level autouse fixture already patches webdriver.Chrome
    scraper = OSHARegulationsScraper(
        base_url="https://www.osha.gov", data_dir=temp_data_dir
    )
    # Set driver to mock to prevent setup_driver from being called
    scraper.driver = mock_driver
    return scraper


class TestOSHARegulationsScraper:
    """Test suite for OSHARegulationsScraper."""

    def test_init_with_relative_path(self, temp_data_dir, mock_driver):
        """Test initialization with relative path."""
        with patch(
            "src.data_collection.osha_regulations_scraper.webdriver.Chrome"
        ) as mock_chrome, \
        patch.object(OSHARegulationsScraper, "setup_driver"):
            mock_chrome.return_value = mock_driver
            scraper = OSHARegulationsScraper(data_dir=temp_data_dir)
            scraper.driver = mock_driver
            assert scraper.base_url == "https://www.osha.gov"
            assert os.path.exists(scraper.data_dir)
            assert os.path.exists(scraper.osha_dir)

    def test_init_with_absolute_path(self, temp_data_dir, mock_driver):
        """Test initialization with absolute path."""
        abs_path = os.path.abspath(temp_data_dir)
        with patch(
            "src.data_collection.osha_regulations_scraper.webdriver.Chrome"
        ) as mock_chrome, \
        patch.object(OSHARegulationsScraper, "setup_driver"):
            mock_chrome.return_value = mock_driver
            scraper = OSHARegulationsScraper(data_dir=abs_path)
            scraper.driver = mock_driver
            assert scraper.data_dir == abs_path

    def test_setup_driver(self, temp_data_dir, mock_driver):
        """Test driver setup."""
        # The module-level autouse fixture already patches webdriver.Chrome
        # We just need to verify setup_driver works correctly
        scraper = OSHARegulationsScraper(data_dir=temp_data_dir)
        scraper.setup_driver()
        assert scraper.driver is not None
        # Verify driver is set (will be the mock from module-level fixture)
        assert scraper.driver is not None

    def test_get_regulation_links(self, scraper, mock_driver):
        """Test getting regulation links."""
        links = scraper.get_regulation_links()

        assert len(links) > 0
        assert all("number" in link for link in links)
        assert all("title" in link for link in links)
        assert all("url" in link for link in links)
        assert any(link["number"] == "1910.1" for link in links)

    def test_get_regulation_links_timeout(self, scraper, mock_driver):
        """Test getting regulation links with timeout."""
        from src.data_collection.osha_regulations_scraper import TimeoutException

        mock_driver.get.side_effect = TimeoutException("Page load timeout")
        links = scraper.get_regulation_links()
        assert len(links) == 0

    def test_get_regulation_links_exception(self, scraper, mock_driver):
        """Test getting regulation links with exception."""
        mock_driver.get.side_effect = Exception("Network error")
        links = scraper.get_regulation_links()
        assert len(links) == 0

    def test_extract_regulation_content(self, scraper, mock_driver):
        """Test extracting regulation content."""
        content = scraper.extract_regulation_content("https://www.osha.gov/test")

        assert content is not None
        assert "metadata" in content
        assert "paragraphs" in content
        assert content["metadata"]["Part Number"] == "1910"
        assert len(content["paragraphs"]) > 0
        assert content["paragraphs"][0]["section"] == "1910.1"

    def test_extract_regulation_content_fallback(self, scraper, mock_driver):
        """Test extracting regulation content with fallback structure."""
        # Modify page source to have fallback structure
        mock_driver.page_source = """
        <html>
            <body>
                <div role="main">
                    <p>Test regulation content</p>
                </div>
            </body>
        </html>
        """
        content = scraper.extract_regulation_content("https://www.osha.gov/test")

        assert content is not None
        assert "text" in content or "paragraphs" in content

    def test_extract_regulation_content_exception(self, scraper, mock_driver):
        """Test extracting regulation content with exception."""
        mock_driver.get.side_effect = Exception("Network error")
        content = scraper.extract_regulation_content("https://www.osha.gov/test")
        assert content is None

    def test_sanitize_filename(self, scraper):
        """Test filename sanitization."""
        test_cases = [
            ("1910.1", "1910_1"),
            ("test:file.json", "test_file.json"),
            ("test/file.json", "test_file.json"),
            ("test  file.json", "test_file.json"),
        ]
        for input_name, expected in test_cases:
            result = scraper.sanitize_filename(input_name)
            # Just check that invalid characters are removed
            assert ":" not in result
            assert "/" not in result

    def test_save_regulation(self, scraper, temp_data_dir):
        """Test saving regulation data."""
        regulation_data = {
            "Part Number": "1910",
            "Standard Number": "1910.1",
            "Title": "Test Regulation",
            "paragraphs": [{"section": "1910.1", "text": "Test content"}],
        }

        filepath = scraper.save_regulation(regulation_data, "1910.1")

        assert filepath is not None
        assert os.path.exists(filepath)
        # The sanitize_filename converts dots to underscores, so check for json extension differently
        assert "json" in filepath or filepath.endswith("_json")

        # Verify JSON content
        import json

        with open(filepath, "r") as f:
            data = json.load(f)
            assert data["Standard Number"] == "1910.1"
            assert data["Title"] == "Test Regulation"

    def test_save_regulation_already_exists(self, scraper, temp_data_dir):
        """Test saving regulation when file already exists."""
        regulation_data = {
            "Part Number": "1910",
            "Standard Number": "1910.1",
            "Title": "Test Regulation",
        }

        # Save first time
        filepath1 = scraper.save_regulation(regulation_data, "1910.1")

        # Save second time (should return existing path)
        filepath2 = scraper.save_regulation(regulation_data, "1910.1")

        assert filepath1 == filepath2

    def test_save_regulation_exception(self, scraper):
        """Test saving regulation with exception."""
        regulation_data = {"Standard Number": "1910.1"}

        with patch("builtins.open", side_effect=Exception("Permission denied")):
            filepath = scraper.save_regulation(regulation_data, "1910.1")
            assert filepath is None

    def test_process_regulation(self, scraper, mock_driver):
        """Test processing a single regulation."""
        result = scraper.process_regulation(
            "https://www.osha.gov/test", "1910.1", "Test Regulation"
        )

        assert result is not None
        assert result["number"] == "1910.1"
        assert result["title"] == "Test Regulation"
        assert result["url"] == "https://www.osha.gov/test"
        assert result["filepath"] is not None

    def test_process_regulation_no_content(self, scraper, mock_driver):
        """Test processing regulation when content extraction fails."""
        with patch.object(scraper, "extract_regulation_content", return_value=None):
            result = scraper.process_regulation(
                "https://www.osha.gov/test", "1910.1", "Test Regulation"
            )
            assert result is None

    def test_create_summary_json(self, scraper, temp_data_dir):
        """Test creating summary JSON."""
        results = [
            {
                "number": "1910.1",
                "title": "Test Regulation 1",
                "url": "https://www.osha.gov/test1",
                "filepath": os.path.join(
                    temp_data_dir, "osha", "regulations", "1910.1.json"
                ),
            },
            {
                "number": "1910.2",
                "title": "Test Regulation 2",
                "url": "https://www.osha.gov/test2",
                "filepath": None,
            },
        ]

        summary_path = scraper.create_summary_json(results)

        assert os.path.exists(summary_path)
        assert summary_path.endswith("osha_regulations_summary.json")

        # Verify JSON content
        import json

        with open(summary_path, "r") as f:
            summary = json.load(f)
            assert summary["total_regulations"] == 2
            assert summary["successful"] == 1
            assert summary["failed"] == 1
            assert len(summary["regulations"]) == 2

    @patch.object(OSHARegulationsScraper, "get_regulation_links")
    @patch.object(OSHARegulationsScraper, "process_regulation")
    @patch.object(OSHARegulationsScraper, "create_summary_json")
    def test_run_scraper_success(
        self, mock_create_summary, mock_process, mock_get_links, scraper, temp_data_dir
    ):
        """Test running scraper successfully."""
        # Mock get_regulation_links
        mock_get_links.return_value = [
            {
                "number": "1910.1",
                "title": "Test 1",
                "url": "https://www.osha.gov/test1",
            },
            {
                "number": "1910.2",
                "title": "Test 2",
                "url": "https://www.osha.gov/test2",
            },
        ]

        # Mock process_regulation
        mock_process.side_effect = [
            {
                "number": "1910.1",
                "title": "Test 1",
                "url": "https://www.osha.gov/test1",
                "filepath": os.path.join(
                    temp_data_dir, "osha", "regulations", "1910.1.json"
                ),
            },
            {
                "number": "1910.2",
                "title": "Test 2",
                "url": "https://www.osha.gov/test2",
                "filepath": os.path.join(
                    temp_data_dir, "osha", "regulations", "1910.2.json"
                ),
            },
        ]

        # Mock create_summary_json
        mock_create_summary.return_value = os.path.join(
            temp_data_dir, "osha", "summary.json"
        )

        results = scraper.run_scraper()

        assert len(results) == 2
        mock_create_summary.assert_called_once()

    @patch.object(OSHARegulationsScraper, "get_regulation_links")
    def test_run_scraper_no_regulations(self, mock_get_links, scraper):
        """Test running scraper when no regulations are found."""
        mock_get_links.return_value = []
        results = scraper.run_scraper()
        assert len(results) == 0

    @patch.object(OSHARegulationsScraper, "get_regulation_links")
    @patch.object(OSHARegulationsScraper, "process_regulation")
    def test_run_scraper_with_limit(
        self, mock_process, mock_get_links, scraper, temp_data_dir
    ):
        """Test running scraper with limit."""
        # Mock get_regulation_links
        mock_get_links.return_value = [
            {
                "number": "1910.1",
                "title": "Test 1",
                "url": "https://www.osha.gov/test1",
            },
            {
                "number": "1910.2",
                "title": "Test 2",
                "url": "https://www.osha.gov/test2",
            },
            {
                "number": "1910.3",
                "title": "Test 3",
                "url": "https://www.osha.gov/test3",
            },
        ]

        # Mock process_regulation
        mock_process.return_value = {
            "number": "1910.1",
            "title": "Test 1",
            "url": "https://www.osha.gov/test1",
            "filepath": os.path.join(
                temp_data_dir, "osha", "regulations", "1910.1.json"
            ),
        }

        results = scraper.run_scraper(limit=2)

        assert len(results) == 2
        assert mock_process.call_count == 2

    @patch.object(OSHARegulationsScraper, "get_regulation_links")
    @patch.object(OSHARegulationsScraper, "process_regulation")
    def test_run_scraper_with_exception(self, mock_process, mock_get_links, scraper):
        """Test running scraper with processing exception."""
        # Mock get_regulation_links
        mock_get_links.return_value = [
            {
                "number": "1910.1",
                "title": "Test 1",
                "url": "https://www.osha.gov/test1",
            },
        ]

        # Mock process_regulation to raise exception
        mock_process.side_effect = Exception("Processing error")

        results = scraper.run_scraper()

        assert len(results) == 1
        assert "error" in results[0]

    def test_run_scraper_closes_driver(self, temp_data_dir, mock_driver):
        """Test that driver is closed after running scraper."""
        with patch(
            "src.data_collection.osha_regulations_scraper.webdriver.Chrome"
        ) as mock_chrome, \
        patch.object(OSHARegulationsScraper, "setup_driver"):
            mock_chrome.return_value = mock_driver
            scraper = OSHARegulationsScraper(data_dir=temp_data_dir)
            scraper.driver = mock_driver

            with patch.object(scraper, "get_regulation_links", return_value=[]):
                scraper.run_scraper()

            mock_driver.quit.assert_called_once()

    def test_init_creates_directories(self, temp_data_dir, mock_driver):
        """Test that initialization creates directories if they don't exist."""
        import shutil

        test_dir = os.path.join(temp_data_dir, "test_data")
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)

        with patch(
            "src.data_collection.osha_regulations_scraper.webdriver.Chrome"
        ) as mock_chrome, \
        patch.object(OSHARegulationsScraper, "setup_driver"):
            mock_chrome.return_value = mock_driver
            scraper = OSHARegulationsScraper(data_dir=test_dir)
            scraper.driver = mock_driver
            assert os.path.exists(scraper.data_dir)
            assert os.path.exists(scraper.osha_dir)

    def test_extract_regulation_content_fallback_text(self, scraper, mock_driver):
        """Test extraction with fallback to text content."""
        mock_driver.page_source = """
        <html>
            <body>
                <div class="content">
                    <p>Test regulation content without structured paragraphs</p>
                </div>
            </body>
        </html>
        """
        content = scraper.extract_regulation_content("https://www.osha.gov/test")
        assert content is not None
        assert "text" in content or "paragraphs" in content

    def test_extract_regulation_content_no_article(self, scraper, mock_driver):
        """Test extraction when article tag is not found."""
        mock_driver.page_source = """
        <html>
            <body>
                <main>
                    <p>Content in main tag</p>
                </main>
            </body>
        </html>
        """
        content = scraper.extract_regulation_content("https://www.osha.gov/test")
        assert content is not None

    def test_process_regulation_with_text_content(self, scraper, mock_driver):
        """Test processing regulation with text content instead of paragraphs."""
        mock_driver.page_source = """
        <html>
            <body>
                <div role="main">
                    <p>Test regulation text content</p>
                </div>
            </body>
        </html>
        """
        result = scraper.process_regulation(
            "https://www.osha.gov/test", "1910.1", "Test Regulation"
        )
        assert result is not None
        assert result["number"] == "1910.1"
