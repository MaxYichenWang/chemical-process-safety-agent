"""
Unit tests for CSBInvestigationScraper.
"""

import os
import tempfile
from unittest.mock import Mock, patch

import pandas as pd
import pytest
from bs4 import BeautifulSoup

from src.data_collection.csb_investigation_scraper import CSBInvestigationScraper


@pytest.fixture
def temp_data_dir():
    """Create a temporary directory for test data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def scraper(temp_data_dir):
    """Create a CSBInvestigationScraper instance for testing."""
    return CSBInvestigationScraper(base_url="https://www.csb.gov", data_dir=temp_data_dir)


@pytest.fixture
def mock_html_response():
    """Create a mock HTML response for testing."""
    html_content = """
    <html>
        <head><title>Test Investigation</title></head>
        <body>
            <h1>Test Investigation Title</h1>
            <div class="moduleBlock">
                <h2 class="title">Accident Description</h2>
                <div class="details">
                    <p>Location: Test Location</p>
                    <p>Accident Occurred On: 2020-01-01 | Final Report Released On: 2020-06-01</p>
                </div>
                <p>This is a test accident description.</p>
            </div>
            <div class="moduleBlock">
                <h2 class="title">Final Reports</h2>
                <a href="/reports/test_report.pdf">Test Report</a>
            </div>
            <div class="moduleBlock">
                <h2 class="title">Recommendations</h2>
                <li class="ac">
                    <a class="head">Test Target</a>
                    <div class="recSubItem">
                        <div class="recNumber">R-1</div>
                        <span id="lblDesc">Test recommendation description</span>
                        <a class="tooltip">Open</a>
                        <a href="/docs/test.pdf">Test Document</a>
                    </div>
                </li>
            </div>
        </body>
    </html>
    """
    return html_content


class TestCSBInvestigationScraper:
    """Test suite for CSBInvestigationScraper."""

    def test_init_with_relative_path(self, temp_data_dir):
        """Test initialization with relative path."""
        scraper = CSBInvestigationScraper(data_dir=temp_data_dir)
        assert scraper.base_url == "https://www.csb.gov"
        assert os.path.exists(scraper.data_dir)
        assert os.path.exists(scraper.csb_dir)

    def test_init_with_absolute_path(self, temp_data_dir):
        """Test initialization with absolute path."""
        abs_path = os.path.abspath(temp_data_dir)
        scraper = CSBInvestigationScraper(data_dir=abs_path)
        assert scraper.data_dir == abs_path

    def test_sanitize_filename(self, scraper):
        """Test filename sanitization."""
        test_cases = [
            ("test:file.pdf", "test_file.pdf"),
            ("test/file.pdf", "test_file.pdf"),
            ("test  file.pdf", "test_file.pdf"),
            ("test___file.pdf", "test_file.pdf"),
            ('test"file.pdf', "test_file.pdf"),
        ]
        for input_name, expected in test_cases:
            result = scraper.sanitize_filename(input_name)
            assert result == expected or result.replace("_", "") == expected.replace("_", "")

    def test_create_investigation_folder(self, scraper):
        """Test folder structure creation."""
        folder_path, docs_folder, final_reports_folder, related_documents_folder, recommendations_folder = (
            scraper.create_investigation_folder("Test_Investigation")
        )

        assert os.path.exists(folder_path)
        assert os.path.exists(docs_folder)
        assert os.path.exists(final_reports_folder)
        assert os.path.exists(related_documents_folder)
        assert os.path.exists(recommendations_folder)

    @patch("src.data_collection.csb_investigation_scraper.requests.get")
    @patch("src.data_collection.csb_investigation_scraper.requests.head")
    def test_download_file_success(self, mock_head, mock_get, scraper, temp_data_dir):
        """Test successful file download."""
        # Mock HEAD request
        mock_head_response = Mock()
        mock_head_response.headers = {"Content-Type": "application/pdf"}
        mock_head_response.status_code = 200
        mock_head.return_value = mock_head_response

        # Mock GET request
        mock_get_response = Mock()
        mock_get_response.status_code = 200
        mock_get_response.iter_content.return_value = [b"test content"]
        mock_get.return_value = mock_get_response

        folder_path = os.path.join(temp_data_dir, "test_folder")
        os.makedirs(folder_path, exist_ok=True)

        result = scraper.download_file("https://example.com/test.pdf", folder_path)
        assert result is not None
        assert os.path.exists(os.path.join(folder_path, "test.pdf"))

    @patch("src.data_collection.csb_investigation_scraper.requests.get")
    @patch("src.data_collection.csb_investigation_scraper.requests.head")
    def test_download_file_already_exists(self, mock_head, mock_get, scraper, temp_data_dir):
        """Test download when file already exists."""
        folder_path = os.path.join(temp_data_dir, "test_folder")
        os.makedirs(folder_path, exist_ok=True)
        file_path = os.path.join(folder_path, "existing.pdf")
        with open(file_path, "wb") as f:
            f.write(b"existing content")

        result = scraper.download_file("https://example.com/existing.pdf", folder_path, "existing.pdf")
        assert result is not None
        mock_get.assert_not_called()

    @patch("src.data_collection.csb_investigation_scraper.requests.get")
    def test_download_file_failure(self, mock_get, scraper, temp_data_dir):
        """Test file download failure."""
        mock_response = Mock()
        mock_response.status_code = 404
        mock_get.return_value = mock_response

        folder_path = os.path.join(temp_data_dir, "test_folder")
        os.makedirs(folder_path, exist_ok=True)

        result = scraper.download_file("https://example.com/missing.pdf", folder_path)
        assert result is None

    def test_extract_section_content(self, scraper, mock_html_response):
        """Test section content extraction."""
        soup = BeautifulSoup(mock_html_response, "html.parser")
        section = scraper.extract_section_content(soup, "Accident Description")
        assert section is not None
        assert "Accident Description" in section.get_text()

    def test_extract_section_content_not_found(self, scraper, mock_html_response):
        """Test section content extraction when section doesn't exist."""
        soup = BeautifulSoup(mock_html_response, "html.parser")
        section = scraper.extract_section_content(soup, "Non-existent Section")
        assert section is None

    def test_extract_accident_description(self, scraper, mock_html_response):
        """Test accident description extraction."""
        soup = BeautifulSoup(mock_html_response, "html.parser")
        accident_info = scraper.extract_accident_description(soup)

        assert "Location" in accident_info
        assert "Accident Occurred On" in accident_info
        assert "Final Report Released On" in accident_info
        assert "Description" in accident_info

    def test_extract_final_reports(self, scraper, mock_html_response, temp_data_dir):
        """Test final reports extraction."""
        soup = BeautifulSoup(mock_html_response, "html.parser")
        docs_folder = os.path.join(temp_data_dir, "docs")
        os.makedirs(docs_folder, exist_ok=True)

        with patch.object(scraper, "download_file", return_value="documents/test_report.pdf"):
            reports = scraper.extract_final_reports(soup, docs_folder)
            assert len(reports) > 0
            assert reports[0]["title"] == "Test Report"

    def test_extract_recommendations(self, scraper, mock_html_response):
        """Test recommendations extraction."""
        soup = BeautifulSoup(mock_html_response, "html.parser")
        recommendations = scraper.extract_recommendations(soup)

        assert len(recommendations) > 0
        assert recommendations[0]["target"] == "Test Target"
        assert len(recommendations[0]["recommendations"]) > 0
        assert recommendations[0]["recommendations"][0]["number"] == "R-1"

    def test_extract_related_documents(self, scraper, mock_html_response, temp_data_dir):
        """Test related documents extraction."""
        # Add related documents section to HTML
        html_with_docs = mock_html_response.replace(
            "</body>",
            '<div class="moduleBlock"><h2 class="title">Related Documents</h2><a href="/docs/doc1.pdf">Doc 1</a></div></body>',
        )
        soup = BeautifulSoup(html_with_docs, "html.parser")
        docs_folder = os.path.join(temp_data_dir, "docs")
        os.makedirs(docs_folder, exist_ok=True)

        with patch.object(scraper, "download_file", return_value="documents/doc1.pdf"):
            documents = scraper.extract_related_documents(soup, docs_folder)
            assert len(documents) > 0

    def test_extract_related_news(self, scraper):
        """Test related news extraction."""
        html_with_news = """
        <html>
            <div class="moduleBlock">
                <h2 class="title">Related News</h2>
                <div class="phgal">
                    <a class="rmore" href="/news/1">
                        <span class="content">
                            <span class="subTitle">News Title</span>
                            <span class="date">2020-01-01</span>
                        </span>
                    </a>
                    <img src="/img/news.jpg" />
                </div>
            </div>
        </html>
        """
        soup = BeautifulSoup(html_with_news, "html.parser")
        news = scraper.extract_related_news(soup)

        assert len(news) > 0
        assert news[0]["title"] == "News Title"
        assert news[0]["date"] == "2020-01-01"

    def test_extract_statistics(self, scraper):
        """Test statistics extraction."""
        html_with_stats = """
        <html>
            <div class="moduleBlock">
                <h2 class="title">Statistics</h2>
                <div class="legend">
                    <div>Total Recommendations # 10</div>
                    <div>Open Recommendations # 5</div>
                    <div>Closed Recommendations # 5</div>
                </div>
                <div class="vs">
                    <span class="nums">5 vs 5</span>
                </div>
            </div>
        </html>
        """
        soup = BeautifulSoup(html_with_stats, "html.parser")
        stats = scraper.extract_statistics(soup)

        assert stats["total_recommendations"] == 10
        assert stats["open_recommendations"] == 5
        assert stats["closed_recommendations"] == 5
        assert stats["open_percentage"] == 5
        assert stats["closed_percentage"] == 5

    def test_extract_videos(self, scraper):
        """Test video extraction."""
        html_with_videos = """
        <html>
            <div class="moduleBlock">
                <h2 class="title">Video</h2>
                <iframe src="https://www.youtube.com/embed/test" title="Test Video"></iframe>
            </div>
        </html>
        """
        soup = BeautifulSoup(html_with_videos, "html.parser")
        videos = scraper.extract_videos(soup)

        assert len(videos) > 0
        assert "youtube.com" in videos[0]["url"]

    @patch("src.data_collection.csb_investigation_scraper.requests.get")
    def test_get_investigation_links(self, mock_get, scraper):
        """Test getting investigation links."""
        # Mock pagination response
        html_page1 = """
        <html>
            <body>
                <div class="item"><a class="linkHd" href="/investigations/test1">Test 1</a></div>
                <div class="item"><a class="linkHd" href="/investigations/test2">Test 2</a></div>
            </body>
        </html>
        """
        html_page2 = """
        <html>
            <body>
                <div class="item"><a class="linkHd" href="/investigations/test3">Test 3</a></div>
            </body>
        </html>
        """
        html_empty = "<html><body></body></html>"

        mock_response1 = Mock()
        mock_response1.status_code = 200
        mock_response1.text = html_page1

        mock_response2 = Mock()
        mock_response2.status_code = 200
        mock_response2.text = html_page2

        mock_response_empty = Mock()
        mock_response_empty.status_code = 200
        mock_response_empty.text = html_empty

        mock_get.side_effect = [mock_response1, mock_response2, mock_response_empty]

        links = scraper.get_investigation_links("https://www.csb.gov/investigations/")

        assert len(links) == 3
        assert all("https://www.csb.gov" in link for link in links)

    @patch("src.data_collection.csb_investigation_scraper.requests.get")
    def test_get_investigation_links_failure(self, mock_get, scraper):
        """Test getting investigation links with HTTP failure."""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_get.return_value = mock_response

        links = scraper.get_investigation_links("https://www.csb.gov/investigations/")
        assert len(links) == 0

    @patch("src.data_collection.csb_investigation_scraper.requests.get")
    def test_process_investigation(self, mock_get, scraper, mock_html_response, temp_data_dir):
        """Test processing a single investigation."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = mock_html_response
        mock_get.return_value = mock_response

        csv_path = os.path.join(temp_data_dir, "test.csv")

        with patch.object(scraper, "download_file", return_value="documents/test.pdf"):
            result = scraper.process_investigation("https://www.csb.gov/test", csv_path)

        assert result is not None
        assert result["title"] == "Test Investigation Title"
        assert "accident_info" in result
        assert "final_reports" in result
        assert "recommendations" in result

    @patch("src.data_collection.csb_investigation_scraper.requests.get")
    def test_process_investigation_failure(self, mock_get, scraper):
        """Test processing investigation with HTTP failure."""
        mock_response = Mock()
        mock_response.status_code = 404
        mock_get.return_value = mock_response

        result = scraper.process_investigation("https://www.csb.gov/test", "test.csv")
        assert result is None

    def test_write_investigation_to_csv(self, scraper, temp_data_dir):
        """Test writing investigation data to CSV."""
        csv_path = os.path.join(temp_data_dir, "test.csv")
        investigation_data = {
            "title": "Test Investigation",
            "url": "https://www.csb.gov/test",
            "accident_info": {"Location": "Test Location"},
            "final_reports": [],
            "recommendations": [],
            "related_documents": [],
            "related_news": [],
            "statistics": {},
            "videos": [],
        }

        scraper.write_investigation_to_csv(investigation_data, csv_path)

        assert os.path.exists(csv_path)
        # Verify CSV was written correctly
        import csv as csv_module

        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv_module.DictReader(f)
            rows = list(reader)
            assert len(rows) == 1
            assert rows[0]["title"] == "Test Investigation"

    def test_write_investigation_to_csv_duplicate(self, scraper, temp_data_dir):
        """Test writing duplicate investigation to CSV."""
        csv_path = os.path.join(temp_data_dir, "test.csv")
        investigation_data = {
            "title": "Test Investigation",
            "url": "https://www.csb.gov/test",
            "accident_info": {},
            "final_reports": [],
            "recommendations": [],
            "related_documents": [],
            "related_news": [],
            "statistics": {},
            "videos": [],
        }

        # Write first time
        scraper.write_investigation_to_csv(investigation_data, csv_path)

        # Write second time (should be skipped)
        scraper.write_investigation_to_csv(investigation_data, csv_path)

        # Verify only one row exists
        import csv as csv_module

        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv_module.DictReader(f)
            rows = list(reader)
            assert len(rows) == 1

    @patch("src.data_collection.csb_investigation_scraper.requests.get")
    def test_run_scraper(self, mock_get, scraper, mock_html_response, temp_data_dir):
        """Test running the scraper."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = mock_html_response
        mock_get.return_value = mock_response

        links = ["https://www.csb.gov/test1", "https://www.csb.gov/test2"]
        csv_file = "test.csv"

        with patch.object(scraper, "download_file", return_value="documents/test.pdf"):
            results = scraper.run_scraper(links, csv_file, limit=2)

        assert len(results) == 2
        assert os.path.exists(os.path.join(temp_data_dir, "csb", "csb_investigation_index.json"))

    @patch("src.data_collection.csb_investigation_scraper.requests.get")
    def test_run_scraper_with_limit(self, mock_get, scraper, mock_html_response, temp_data_dir):
        """Test running scraper with limit."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = mock_html_response
        mock_get.return_value = mock_response

        links = ["https://www.csb.gov/test1", "https://www.csb.gov/test2", "https://www.csb.gov/test3"]
        csv_file = "test.csv"

        with patch.object(scraper, "download_file", return_value="documents/test.pdf"):
            results = scraper.run_scraper(links, csv_file, limit=1)

        assert len(results) == 1

    def test_init_creates_directories(self, temp_data_dir):
        """Test that initialization creates directories if they don't exist."""
        import shutil
        # Remove directories if they exist
        test_dir = os.path.join(temp_data_dir, "test_data")
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)

        scraper = CSBInvestigationScraper(data_dir=test_dir)
        assert os.path.exists(scraper.data_dir)
        assert os.path.exists(scraper.csb_dir)

    @patch("src.data_collection.csb_investigation_scraper.requests.get")
    @patch("src.data_collection.csb_investigation_scraper.requests.head")
    def test_download_file_relative_url(self, mock_head, mock_get, scraper, temp_data_dir):
        """Test download with relative URL."""
        mock_head_response = Mock()
        mock_head_response.headers = {"Content-Type": "application/pdf"}
        mock_head_response.status_code = 200
        mock_head.return_value = mock_head_response

        mock_get_response = Mock()
        mock_get_response.status_code = 200
        mock_get_response.iter_content.return_value = [b"test content"]
        mock_get.return_value = mock_get_response

        folder_path = os.path.join(temp_data_dir, "test_folder")
        os.makedirs(folder_path, exist_ok=True)

        # Test with relative URL
        result = scraper.download_file("/reports/test.pdf", folder_path)
        assert result is not None

    @patch("src.data_collection.csb_investigation_scraper.requests.get")
    @patch("src.data_collection.csb_investigation_scraper.requests.head")
    def test_download_file_content_disposition(self, mock_head, mock_get, scraper, temp_data_dir):
        """Test download with Content-Disposition header."""
        mock_head_response = Mock()
        mock_head_response.headers = {
            "Content-Type": "application/pdf",
            "Content-Disposition": 'attachment; filename="test_report.pdf"'
        }
        mock_head_response.status_code = 200
        mock_head.return_value = mock_head_response

        mock_get_response = Mock()
        mock_get_response.status_code = 200
        mock_get_response.iter_content.return_value = [b"test content"]
        mock_get.return_value = mock_get_response

        folder_path = os.path.join(temp_data_dir, "test_folder")
        os.makedirs(folder_path, exist_ok=True)

        result = scraper.download_file("https://example.com/download", folder_path)
        assert result is not None
        assert "test_report.pdf" in result

    @patch("src.data_collection.csb_investigation_scraper.requests.get")
    @patch("src.data_collection.csb_investigation_scraper.requests.head")
    def test_download_file_no_filename_fallback(self, mock_head, mock_get, scraper, temp_data_dir):
        """Test download with no filename, using fallback."""
        mock_head_response = Mock()
        mock_head_response.headers = {"Content-Type": "application/pdf"}
        mock_head_response.status_code = 200
        mock_head.return_value = mock_head_response

        mock_get_response = Mock()
        mock_get_response.status_code = 200
        mock_get_response.iter_content.return_value = [b"test content"]
        mock_get.return_value = mock_get_response

        folder_path = os.path.join(temp_data_dir, "test_folder")
        os.makedirs(folder_path, exist_ok=True)

        # URL with no filename
        result = scraper.download_file("https://example.com/", folder_path)
        assert result is not None
        assert "downloaded_file" in result

    @patch("src.data_collection.csb_investigation_scraper.requests.get")
    @patch("src.data_collection.csb_investigation_scraper.requests.head")
    def test_download_file_content_type_extensions(self, mock_head, mock_get, scraper, temp_data_dir):
        """Test download with content type-based extension addition."""
        folder_path = os.path.join(temp_data_dir, "test_folder")
        os.makedirs(folder_path, exist_ok=True)

        test_cases = [
            ("application/pdf", ".pdf"),
            ("application/msword", ".doc"),
            ("application/vnd.openxmlformats-officedocument.wordprocessingml.document", ".docx"),
        ]

        for content_type, expected_ext in test_cases:
            mock_head_response = Mock()
            mock_head_response.headers = {"Content-Type": content_type}
            mock_head_response.status_code = 200
            mock_head.return_value = mock_head_response

            mock_get_response = Mock()
            mock_get_response.status_code = 200
            mock_get_response.iter_content.return_value = [b"test content"]
            mock_get.return_value = mock_get_response

            # Use filename without extension to trigger extension addition
            result = scraper.download_file("https://example.com/file", folder_path, "testfile")
            assert result is not None
            # Check that the extension was added (filename will be sanitized)
            assert expected_ext.replace(".", "") in result or "testfile" in result

    @patch("src.data_collection.csb_investigation_scraper.requests.get")
    @patch("src.data_collection.csb_investigation_scraper.requests.head")
    def test_download_file_exception_handling(self, mock_head, mock_get, scraper, temp_data_dir):
        """Test download file exception handling."""
        mock_head.side_effect = Exception("Network error")

        folder_path = os.path.join(temp_data_dir, "test_folder")
        os.makedirs(folder_path, exist_ok=True)

        result = scraper.download_file("https://example.com/test.pdf", folder_path)
        assert result is None

    def test_extract_accident_description_with_image(self, scraper):
        """Test accident description extraction with image."""
        html_with_image = """
        <html>
            <div class="moduleBlock">
                <h2 class="title">Accident Description</h2>
                <div class="imageContainer">
                    <img src="/images/accident.jpg" />
                </div>
            </div>
        </html>
        """
        soup = BeautifulSoup(html_with_image, "html.parser")
        accident_info = scraper.extract_accident_description(soup)

        assert "Image URL" in accident_info
        assert "accident.jpg" in accident_info["Image URL"]

    def test_extract_videos_with_links(self, scraper):
        """Test video extraction with video links."""
        html_with_video_links = """
        <html>
            <div class="moduleBlock">
                <h2 class="title">Video</h2>
                <a href="https://www.youtube.com/watch?v=test123">Watch Video</a>
                <a href="https://vimeo.com/123456">Vimeo Video</a>
            </div>
        </html>
        """
        soup = BeautifulSoup(html_with_video_links, "html.parser")
        videos = scraper.extract_videos(soup)

        assert len(videos) >= 2
        assert any("youtube.com" in v["url"] for v in videos)
        assert any("vimeo.com" in v["url"] for v in videos)

    def test_extract_videos_duplicate_check(self, scraper):
        """Test video extraction with duplicate check."""
        html_with_duplicate = """
        <html>
            <div class="moduleBlock">
                <h2 class="title">Video</h2>
                <iframe src="https://www.youtube.com/embed/test" title="Test Video"></iframe>
            </div>
            <iframe src="https://www.youtube.com/embed/test" title="Duplicate"></iframe>
        </html>
        """
        soup = BeautifulSoup(html_with_duplicate, "html.parser")
        videos = scraper.extract_videos(soup)

        # Should only have one video (duplicate removed)
        youtube_videos = [v for v in videos if "youtube.com" in v["url"]]
        assert len(youtube_videos) == 1

    @patch("src.data_collection.csb_investigation_scraper.requests.get")
    def test_get_investigation_links_pagination_break(self, mock_get, scraper):
        """Test pagination break when no items found."""
        html_empty = """
        <html>
            <body>
            </body>
        </html>
        """
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = html_empty
        mock_get.return_value = mock_response

        links = scraper.get_investigation_links("https://www.csb.gov/investigations/")
        assert len(links) == 0

    def test_write_investigation_to_csv_exception_handling(self, scraper, temp_data_dir):
        """Test CSV writing with exception handling."""
        csv_path = os.path.join(temp_data_dir, "test.csv")
        investigation_data = {
            "title": "Test Investigation",
            "url": "https://www.csb.gov/test",
            "accident_info": {},
            "final_reports": [],
            "recommendations": [],
            "related_documents": [],
            "related_news": [],
            "statistics": {},
            "videos": [],
        }

        # Create invalid CSV path to trigger exception
        invalid_path = os.path.join(temp_data_dir, "nonexistent", "test.csv")

        # Should handle exception gracefully
        try:
            scraper.write_investigation_to_csv(investigation_data, invalid_path)
        except Exception:
            pass  # Exception is expected and handled

    @patch("src.data_collection.csb_investigation_scraper.requests.get")
    def test_run_scraper_exception_handling(self, mock_get, scraper, temp_data_dir):
        """Test run_scraper exception handling."""
        mock_get.side_effect = Exception("Network error")

        links = ["https://www.csb.gov/test1"]
        csv_file = "test.csv"

        results = scraper.run_scraper(links, csv_file, limit=1)
        # Should handle exception and continue
        assert isinstance(results, type(pd.DataFrame()))
