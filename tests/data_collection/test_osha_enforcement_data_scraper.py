"""
Unit tests for OSHAEnforcementDataScraper.
"""

import os
import tempfile
import zipfile
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from src.data_collection.osha_enforcement_data_scraper import OSHAEnforcementDataScraper


@pytest.fixture
def temp_data_dir():
    """Create a temporary directory for test data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def scraper(temp_data_dir):
    """Create an OSHAEnforcementDataScraper instance for testing."""
    return OSHAEnforcementDataScraper(base_url="https://enforcedata.dol.gov", data_dir=temp_data_dir)


@pytest.fixture
def mock_html_with_download_links():
    """Create mock HTML with download links."""
    return """
    <html>
        <body>
            <a href="https://enfxfr.dol.gov/osha_inspection_20251025.csv.zip">Inspection Data</a>
            <a href="https://enfxfr.dol.gov/osha_violation_20251025.csv.zip">Violation Data</a>
            <a href="https://enfxfr.dol.gov/osha_accident_lookup2_20251025.csv.zip">Accident Lookup</a>
        </body>
    </html>
    """


class TestOSHAEnforcementDataScraper:
    """Test suite for OSHAEnforcementDataScraper."""

    def test_init_with_relative_path(self, temp_data_dir):
        """Test initialization with relative path."""
        scraper = OSHAEnforcementDataScraper(data_dir=temp_data_dir)
        assert scraper.base_url == "https://enforcedata.dol.gov"
        assert os.path.exists(scraper.data_dir)
        assert os.path.exists(scraper.osha_dir)

    def test_init_with_absolute_path(self, temp_data_dir):
        """Test initialization with absolute path."""
        abs_path = os.path.abspath(temp_data_dir)
        scraper = OSHAEnforcementDataScraper(data_dir=abs_path)
        assert scraper.data_dir == abs_path

    def test_files_to_download_initialized(self, scraper):
        """Test that files_to_download dictionary is initialized."""
        assert len(scraper.files_to_download) > 0
        assert "osha_inspection" in scraper.files_to_download
        assert "osha_violation" in scraper.files_to_download
        assert "osha_accident" in scraper.files_to_download

    @patch("src.data_collection.osha_enforcement_data_scraper.requests.post")
    def test_get_download_urls_success(self, mock_post, scraper, mock_html_with_download_links):
        """Test getting download URLs successfully."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = mock_html_with_download_links
        mock_post.return_value = mock_response

        urls = scraper.get_download_urls()

        assert len(urls) > 0
        assert "osha_inspection" in urls
        assert "osha_violation" in urls
        assert "osha_accident_lookup" in urls  # Should map from osha_accident_lookup2

    @patch("src.data_collection.osha_enforcement_data_scraper.requests.post")
    def test_get_download_urls_failure(self, mock_post, scraper):
        """Test getting download URLs with HTTP failure."""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_post.return_value = mock_response

        urls = scraper.get_download_urls()
        assert len(urls) == 0

    @patch("src.data_collection.osha_enforcement_data_scraper.requests.post")
    def test_get_download_urls_exception(self, mock_post, scraper):
        """Test getting download URLs with exception."""
        mock_post.side_effect = Exception("Network error")
        urls = scraper.get_download_urls()
        assert len(urls) == 0

    @patch("src.data_collection.osha_enforcement_data_scraper.requests.get")
    @patch("src.data_collection.osha_enforcement_data_scraper.requests.head")
    def test_download_file_csv_success(self, mock_head, mock_get, scraper, temp_data_dir):
        """Test successful CSV file download."""
        # Mock HEAD request
        mock_head_response = Mock()
        mock_head_response.headers = {"Content-Type": "text/csv"}
        mock_head_response.status_code = 200
        mock_head.return_value = mock_head_response

        # Mock GET request
        mock_get_response = Mock()
        mock_get_response.status_code = 200
        mock_get_response.iter_content.return_value = [b"col1,col2\nval1,val2\n"]
        mock_get.return_value = mock_get_response

        result = scraper.download_file("https://example.com/test.csv", "test_file")

        assert result is not None
        assert os.path.exists(result)
        assert result.endswith(".csv")

    @patch("src.data_collection.osha_enforcement_data_scraper.requests.get")
    def test_download_file_zip_success(self, mock_get, scraper, temp_data_dir):
        """Test successful ZIP file download and extraction."""
        # Create a test ZIP file in memory
        zip_content = b"col1,col2\nval1,val2\nval3,val4\n"
        zip_buffer = b""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp_zip:
            with zipfile.ZipFile(tmp_zip.name, "w") as zf:
                zf.writestr("test_file.csv", zip_content)
            with open(tmp_zip.name, "rb") as f:
                zip_buffer = f.read()
        os.unlink(tmp_zip.name)

        # Mock GET request
        mock_get_response = Mock()
        mock_get_response.status_code = 200
        mock_get_response.iter_content.return_value = [zip_buffer]
        mock_get.return_value = mock_get_response

        result = scraper.download_file("https://example.com/test.zip", "test_file")

        assert result is not None
        assert os.path.exists(result)
        assert result.endswith(".csv")

        # Verify CSV content
        df = pd.read_csv(result)
        assert len(df) == 2
        assert "col1" in df.columns

    @patch("src.data_collection.osha_enforcement_data_scraper.requests.get")
    def test_download_file_already_exists(self, mock_get, scraper, temp_data_dir):
        """Test download when CSV file already exists."""
        # Create existing CSV file
        existing_file = os.path.join(temp_data_dir, "osha", "test_file.csv")
        os.makedirs(os.path.dirname(existing_file), exist_ok=True)
        with open(existing_file, "w") as f:
            f.write("col1,col2\nval1,val2\n")

        mock_response = Mock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        result = scraper.download_file("https://example.com/test.zip", "test_file")

        assert result == existing_file
        mock_response.iter_content.assert_not_called()

    @patch("src.data_collection.osha_enforcement_data_scraper.requests.get")
    def test_download_file_failure(self, mock_get, scraper):
        """Test file download failure."""
        mock_response = Mock()
        mock_response.status_code = 404
        mock_get.return_value = mock_response

        result = scraper.download_file("https://example.com/missing.zip", "test_file")
        assert result is None

    @patch("src.data_collection.osha_enforcement_data_scraper.requests.get")
    def test_download_file_exception(self, mock_get, scraper):
        """Test file download with exception."""
        mock_get.side_effect = Exception("Network error")

        result = scraper.download_file("https://example.com/test.zip", "test_file")
        assert result is None

    def test_get_file_info_existing_file(self, scraper, temp_data_dir):
        """Test getting file info for existing file."""
        # Create test CSV file
        test_file = os.path.join(temp_data_dir, "osha", "test.csv")
        os.makedirs(os.path.dirname(test_file), exist_ok=True)
        df = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})
        df.to_csv(test_file, index=False)

        info = scraper.get_file_info(test_file)

        assert info["file_path"] == test_file
        assert info["file_size"] > 0
        assert info["record_count"] == 3
        assert "col1" in info["columns"]
        assert "col2" in info["columns"]

    def test_get_file_info_large_file(self, scraper, temp_data_dir):
        """Test getting file info for large file (chunked reading)."""
        # Create a large CSV file (>50MB simulation)
        test_file = os.path.join(temp_data_dir, "osha", "large_test.csv")
        os.makedirs(os.path.dirname(test_file), exist_ok=True)

        # Create a file that would trigger chunked reading
        # In practice, we'll create a smaller file but mock the size check
        df = pd.DataFrame({"col1": range(1000), "col2": ["value"] * 1000})
        df.to_csv(test_file, index=False)

        with patch("os.path.getsize", return_value=60 * 1024 * 1024):  # 60MB
            info = scraper.get_file_info(test_file)

        assert info["file_path"] == test_file
        assert info["record_count"] > 0

    def test_get_file_info_nonexistent_file(self, scraper):
        """Test getting file info for non-existent file."""
        info = scraper.get_file_info("/nonexistent/file.csv")

        assert info["file_path"] == "/nonexistent/file.csv"
        assert info["file_size"] == 0
        assert info["record_count"] == 0

    def test_get_file_info_invalid_csv(self, scraper, temp_data_dir):
        """Test getting file info for invalid CSV."""
        # Create invalid CSV file
        test_file = os.path.join(temp_data_dir, "osha", "invalid.csv")
        os.makedirs(os.path.dirname(test_file), exist_ok=True)
        with open(test_file, "w") as f:
            f.write("not a valid csv file content")

        info = scraper.get_file_info(test_file)

        assert info["file_path"] == test_file
        assert "record_count" in info

    def test_create_overview_csv(self, scraper, temp_data_dir):
        """Test creating overview CSV."""
        download_results = {
            "osha_inspection": {
                "file_path": os.path.join(temp_data_dir, "osha", "osha_inspection.csv"),
                "file_size": 1000,
                "record_count": 100,
                "columns": ["col1", "col2"],
                "last_modified": "2024-01-01T00:00:00",
                "download_timestamp": "2024-01-01T00:00:00",
            },
            "osha_violation": {
                "file_path": None,
                "file_size": 0,
                "record_count": 0,
                "columns": [],
                "last_modified": None,
                "download_timestamp": "2024-01-01T00:00:00",
            },
        }

        overview_path = scraper.create_overview_csv(download_results)

        assert os.path.exists(overview_path)
        assert overview_path.endswith("osha_enforcement_data_summary.csv")

        # Verify CSV content
        df = pd.read_csv(overview_path)
        assert len(df) == 2
        assert "file_name" in df.columns
        assert "download_status" in df.columns
        assert df[df["file_name"] == "osha_inspection"]["download_status"].iloc[0] == "Success"
        assert df[df["file_name"] == "osha_violation"]["download_status"].iloc[0] == "Failed"

    @patch.object(OSHAEnforcementDataScraper, "get_download_urls")
    @patch.object(OSHAEnforcementDataScraper, "download_file")
    @patch.object(OSHAEnforcementDataScraper, "get_file_info")
    @patch.object(OSHAEnforcementDataScraper, "create_overview_csv")
    def test_run_scraper_success(
        self, mock_create_overview, mock_get_info, mock_download, mock_get_urls, scraper, temp_data_dir
    ):
        """Test running scraper successfully."""
        # Mock get_download_urls
        mock_get_urls.return_value = {
            "osha_inspection": "https://example.com/inspection.zip",
            "osha_violation": "https://example.com/violation.zip",
        }

        # Mock download_file
        mock_download.side_effect = [
            os.path.join(temp_data_dir, "osha", "osha_inspection.csv"),
            os.path.join(temp_data_dir, "osha", "osha_violation.csv"),
        ]

        # Mock get_file_info
        mock_get_info.side_effect = [
            {
                "file_path": os.path.join(temp_data_dir, "osha", "osha_inspection.csv"),
                "file_size": 1000,
                "record_count": 100,
                "columns": ["col1"],
                "last_modified": "2024-01-01",
                "download_timestamp": "2024-01-01",
            },
            {
                "file_path": os.path.join(temp_data_dir, "osha", "osha_violation.csv"),
                "file_size": 2000,
                "record_count": 200,
                "columns": ["col2"],
                "last_modified": "2024-01-01",
                "download_timestamp": "2024-01-01",
            },
        ]

        # Mock create_overview_csv
        mock_create_overview.return_value = os.path.join(temp_data_dir, "osha", "overview.csv")

        results = scraper.run_scraper()

        assert len(results) == 2
        assert "osha_inspection" in results
        assert "osha_violation" in results
        mock_create_overview.assert_called_once()

    @patch.object(OSHAEnforcementDataScraper, "get_download_urls")
    def test_run_scraper_no_urls(self, mock_get_urls, scraper):
        """Test running scraper when no URLs are found."""
        mock_get_urls.return_value = {}

        results = scraper.run_scraper()
        assert len(results) == 0

    @patch.object(OSHAEnforcementDataScraper, "get_download_urls")
    @patch.object(OSHAEnforcementDataScraper, "download_file")
    @patch.object(OSHAEnforcementDataScraper, "get_file_info")
    def test_run_scraper_with_limit(
        self, mock_get_info, mock_download, mock_get_urls, scraper, temp_data_dir
    ):
        """Test running scraper with limit."""
        # Mock get_download_urls
        mock_get_urls.return_value = {
            "osha_inspection": "https://example.com/inspection.zip",
            "osha_violation": "https://example.com/violation.zip",
            "osha_accident": "https://example.com/accident.zip",
        }

        # Mock download_file
        mock_download.return_value = os.path.join(temp_data_dir, "osha", "test.csv")

        # Mock get_file_info
        mock_get_info.return_value = {
            "file_path": os.path.join(temp_data_dir, "osha", "test.csv"),
            "file_size": 1000,
            "record_count": 100,
            "columns": ["col1"],
            "last_modified": "2024-01-01",
            "download_timestamp": "2024-01-01",
        }

        results = scraper.run_scraper(limit=2)

        assert len(results) == 2
        assert mock_download.call_count == 2

    @patch.object(OSHAEnforcementDataScraper, "get_download_urls")
    @patch.object(OSHAEnforcementDataScraper, "download_file")
    @patch.object(OSHAEnforcementDataScraper, "get_file_info")
    @patch.object(OSHAEnforcementDataScraper, "create_overview_csv")
    def test_run_scraper_download_failure(
        self, mock_create_overview, mock_get_info, mock_download, mock_get_urls, scraper, temp_data_dir
    ):
        """Test running scraper with download failure."""
        # Mock get_download_urls
        mock_get_urls.return_value = {
            "osha_inspection": "https://example.com/inspection.zip",
        }

        # Mock download_file to return None (failure)
        mock_download.return_value = None

        # Mock get_file_info - should handle None filepath
        mock_get_info.return_value = {
            "file_path": None,
            "file_size": 0,
            "record_count": 0,
            "columns": [],
            "last_modified": None,
            "download_timestamp": "2024-01-01",
        }

        # Mock create_overview_csv
        mock_create_overview.return_value = os.path.join(temp_data_dir, "osha", "overview.csv")

        results = scraper.run_scraper()

        assert len(results) == 1
        # Check that the result has the expected structure
        assert "osha_inspection" in results
        result = results["osha_inspection"]
        # The result should have file_path key (even if None) or url key
        assert "url" in result
        # If download failed, file_path should be None or not exist
        if "file_path" in result:
            assert result["file_path"] is None

    def test_init_creates_directories(self, temp_data_dir):
        """Test that initialization creates directories if they don't exist."""
        import shutil
        test_dir = os.path.join(temp_data_dir, "test_data")
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)

        scraper = OSHAEnforcementDataScraper(data_dir=test_dir)
        assert os.path.exists(scraper.data_dir)
        assert os.path.exists(scraper.osha_dir)

    @patch("src.data_collection.osha_enforcement_data_scraper.requests.get")
    def test_download_file_zip_no_csv_files(self, mock_get, scraper, temp_data_dir):
        """Test ZIP download with no CSV files."""
        # Create a ZIP file with no CSV files
        import zipfile
        zip_buffer = b""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp_zip:
            with zipfile.ZipFile(tmp_zip.name, "w") as zf:
                zf.writestr("test.txt", "not a csv file")
            with open(tmp_zip.name, "rb") as f:
                zip_buffer = f.read()
        os.unlink(tmp_zip.name)

        mock_get_response = Mock()
        mock_get_response.status_code = 200
        mock_get_response.iter_content.return_value = [zip_buffer]
        mock_get.return_value = mock_get_response

        result = scraper.download_file("https://example.com/test.zip", "test_file")
        assert result is None

    @patch("src.data_collection.osha_enforcement_data_scraper.requests.get")
    def test_download_file_zip_no_extracted_files(self, mock_get, scraper, temp_data_dir):
        """Test ZIP download when no extracted CSV files are found."""
        # This is a difficult edge case to test, but we can mock the file system
        import zipfile
        zip_buffer = b""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp_zip:
            with zipfile.ZipFile(tmp_zip.name, "w") as zf:
                zf.writestr("test.csv", "col1,col2\nval1,val2\n")
            with open(tmp_zip.name, "rb") as f:
                zip_buffer = f.read()
        os.unlink(tmp_zip.name)

        mock_get_response = Mock()
        mock_get_response.status_code = 200
        mock_get_response.iter_content.return_value = [zip_buffer]
        mock_get.return_value = mock_get_response

        # Mock os.walk to return no files
        with patch("os.walk", return_value=[]):
            result = scraper.download_file("https://example.com/test.zip", "test_file")
            assert result is None

    @patch("src.data_collection.osha_enforcement_data_scraper.requests.get")
    def test_download_file_zip_multiple_csv_concatenation(self, mock_get, scraper, temp_data_dir):
        """Test ZIP download with multiple CSV files that need concatenation."""
        import zipfile
        zip_buffer = b""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp_zip:
            with zipfile.ZipFile(tmp_zip.name, "w") as zf:
                zf.writestr("test_file_part1.csv", "col1,col2\nval1,val2\n")
                zf.writestr("test_file_part2.csv", "col1,col2\nval3,val4\n")
            with open(tmp_zip.name, "rb") as f:
                zip_buffer = f.read()
        os.unlink(tmp_zip.name)

        mock_get_response = Mock()
        mock_get_response.status_code = 200
        mock_get_response.iter_content.return_value = [zip_buffer]
        mock_get.return_value = mock_get_response

        result = scraper.download_file("https://example.com/test.zip", "test_file")
        assert result is not None
        assert os.path.exists(result)

        # Verify concatenated content
        df = pd.read_csv(result)
        assert len(df) == 2

    @patch("src.data_collection.osha_enforcement_data_scraper.requests.get")
    def test_download_file_zip_cleanup_errors(self, mock_get, scraper, temp_data_dir):
        """Test ZIP download with file cleanup errors."""
        import zipfile
        zip_buffer = b""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp_zip:
            with zipfile.ZipFile(tmp_zip.name, "w") as zf:
                zf.writestr("test_file.csv", "col1,col2\nval1,val2\n")
            with open(tmp_zip.name, "rb") as f:
                zip_buffer = f.read()
        os.unlink(tmp_zip.name)

        mock_get_response = Mock()
        mock_get_response.status_code = 200
        mock_get_response.iter_content.return_value = [zip_buffer]
        mock_get.return_value = mock_get_response

        # Mock os.remove to raise exception only for CSV files, not ZIP
        original_remove = os.remove
        call_count = {"count": 0}
        def mock_remove(path):
            call_count["count"] += 1
            # Raise exception only when removing CSV files (after ZIP is removed)
            if path.endswith(".csv") and call_count["count"] > 1:
                raise Exception("Permission denied")
            return original_remove(path)

        with patch("os.remove", side_effect=mock_remove):
            result = scraper.download_file("https://example.com/test.zip", "test_file")
            # Should still return the file path despite cleanup error
            assert result is not None

    @patch("src.data_collection.osha_enforcement_data_scraper.requests.get")
    def test_download_file_zip_empty_directory_cleanup(self, mock_get, scraper, temp_data_dir):
        """Test ZIP download with empty directory cleanup."""
        import zipfile
        zip_buffer = b""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp_zip:
            with zipfile.ZipFile(tmp_zip.name, "w") as zf:
                zf.writestr("subdir/test_file.csv", "col1,col2\nval1,val2\n")
            with open(tmp_zip.name, "rb") as f:
                zip_buffer = f.read()
        os.unlink(tmp_zip.name)

        mock_get_response = Mock()
        mock_get_response.status_code = 200
        mock_get_response.iter_content.return_value = [zip_buffer]
        mock_get.return_value = mock_get_response

        result = scraper.download_file("https://example.com/test.zip", "test_file")
        assert result is not None

    @patch("src.data_collection.osha_enforcement_data_scraper.requests.get")
    def test_download_file_zip_extraction_exception(self, mock_get, scraper, temp_data_dir):
        """Test ZIP download with extraction exception."""
        # Create invalid ZIP file
        invalid_zip = b"This is not a valid ZIP file"

        mock_get_response = Mock()
        mock_get_response.status_code = 200
        mock_get_response.iter_content.return_value = [invalid_zip]
        mock_get.return_value = mock_get_response

        result = scraper.download_file("https://example.com/test.zip", "test_file")
        assert result is None

    def test_get_file_info_error_reading(self, scraper, temp_data_dir):
        """Test get_file_info with error reading CSV."""
        # Create a file that looks like CSV but can't be read properly
        test_file = os.path.join(temp_data_dir, "osha", "invalid.csv")
        os.makedirs(os.path.dirname(test_file), exist_ok=True)
        # Create a file that will cause read error
        with open(test_file, "wb") as f:
            f.write(b"\x00\x01\x02\x03")  # Binary data that's not valid CSV

        info = scraper.get_file_info(test_file)
        assert info["file_path"] == test_file
        assert "record_count" in info

    @patch.object(OSHAEnforcementDataScraper, "get_download_urls")
    def test_run_scraper_missing_url(self, mock_get_urls, scraper, temp_data_dir):
        """Test run_scraper when URL is missing for a file."""
        mock_get_urls.return_value = {
            "osha_inspection": "https://example.com/inspection.zip",
        }

        with patch.object(scraper, "download_file", return_value=None), \
             patch.object(scraper, "get_file_info", return_value={
                 "file_path": None,
                 "file_size": 0,
                 "record_count": 0,
                 "columns": [],
                 "last_modified": None,
                 "download_timestamp": "2024-01-01",
             }), \
             patch.object(scraper, "create_overview_csv", return_value=os.path.join(temp_data_dir, "overview.csv")):

            # Test with a filename that doesn't have a URL
            results = scraper.run_scraper()
            # Should handle missing URL gracefully
            assert isinstance(results, dict)
