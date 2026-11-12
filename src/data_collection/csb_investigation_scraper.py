"""
CSB Investigation Data Scraper

This module provides functionality to scrape investigation data from the
Chemical Safety Board (CSB) website, including accident descriptions,
final reports, recommendations, and related documents.
"""

import csv
import json
import os
import re
import time
from typing import Dict, List, Optional, Tuple
from urllib.parse import urljoin, urlparse

import pandas as pd
import requests
from bs4 import BeautifulSoup

from src.data_collection.config import (
    CSB_BASE_URL,
    CSB_DATA_FILENAME,
    CSB_LINKS_FILENAME,
    DOWNLOAD_CHUNK_SIZE,
    SCRAPER_SLEEP_DELAY,
    SCRAPER_USER_AGENT,
)


class CSBInvestigationScraper:
    """Scraper for CSB investigation data."""

    def __init__(self, base_url: str = CSB_BASE_URL, data_dir: str = "data/raw"):
        """
        Initialize the CSB scraper.

        Args:
            base_url: Base URL for the CSB website
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
        self.csb_dir = os.path.join(self.data_dir, "csb")
        self.headers = {"User-Agent": SCRAPER_USER_AGENT}

        # Create data directory if it doesn't exist
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
        if not os.path.exists(self.csb_dir):
            os.makedirs(self.csb_dir)

    def sanitize_filename(self, filename: str) -> str:
        """Sanitize filename to be valid on most file systems."""
        # Replace characters that are not allowed in filenames
        for char in [":", "/", "\\", "?", "*", '"', "<", ">", "|"]:
            filename = filename.replace(char, "_")
        # Replace multiple spaces with a single underscore
        filename = re.sub(r"\s+", "_", filename)
        # Remove any leading/trailing spaces or underscores
        filename = filename.strip("_ ")
        return filename

    def create_investigation_folder(
        self, investigation_name: str
    ) -> Tuple[str, str, str, str, str]:
        """Create a folder structure for the investigation."""
        folder_path = os.path.join(self.csb_dir, investigation_name)

        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # Create subfolders
        docs_folder = os.path.join(folder_path, "documents")
        final_reports_folder = os.path.join(docs_folder, "final_reports")
        recommendations_folder = os.path.join(docs_folder, "recommendations")
        related_documents_folder = os.path.join(docs_folder, "related_documents")

        for folder in [
            docs_folder,
            final_reports_folder,
            recommendations_folder,
            related_documents_folder,
        ]:
            if not os.path.exists(folder):
                os.makedirs(folder)

        return (
            folder_path,
            docs_folder,
            final_reports_folder,
            related_documents_folder,
            recommendations_folder,
        )

    def download_file(
        self, url: str, folder_path: str, filename: Optional[str] = None
    ) -> Optional[str]:
        """Download a file and save it to the specified folder."""
        try:
            if not url.startswith(("http:", "https:")):
                url = urljoin(self.base_url, url)

            # Get filename from URL or headers if not provided
            if not filename:
                head_response = requests.head(
                    url, headers=self.headers, allow_redirects=True
                )
                content_disposition = head_response.headers.get("Content-Disposition")
                filename_extracted = False

                if content_disposition and "filename=" in content_disposition:
                    matches = re.findall(r'filename="?([^";]+)"?', content_disposition)
                    if matches:
                        filename = matches[0]
                        filename_extracted = True

                if not filename_extracted:
                    filename = os.path.basename(urlparse(url).path)
                    if not filename:
                        filename = f"downloaded_file_{int(time.time())}"

                # Add extension based on content type
                content_type = head_response.headers.get("Content-Type", "")
                if "." not in filename:
                    if "pdf" in content_type.lower():
                        filename += ".pdf"
                    elif "application/msword" in content_type.lower():
                        filename += ".doc"
                    elif (
                        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                        in content_type.lower()
                    ):
                        filename += ".docx"

            filename = self.sanitize_filename(filename)
            filepath = os.path.join(folder_path, filename)

            # Skip if file already exists
            if os.path.exists(filepath):
                print(f"File already exists, skipping download: {filename}")
                return os.path.join(
                    os.path.basename(os.path.dirname(folder_path)),
                    "documents",
                    filename,
                )

            print(f"Downloading: {url}")
            response = requests.get(url, headers=self.headers, stream=True)

            if response.status_code == 200:
                with open(filepath, "wb") as f:
                    for chunk in response.iter_content(chunk_size=DOWNLOAD_CHUNK_SIZE):
                        f.write(chunk)

                print(f"Downloaded: {filename}")
                return os.path.join(
                    os.path.basename(os.path.dirname(folder_path)),
                    "documents",
                    filename,
                )
            else:
                print(f"Failed to download: {url}, Status code: {response.status_code}")
                return None

        except Exception as e:
            print(f"Error downloading {url}: {e}")
            return None

    def extract_section_content(self, soup: BeautifulSoup, section_title: str):
        """Extract content from a specific section."""
        sections = soup.find_all("h2", class_="title")

        for section in sections:
            if section_title.lower() in section.text.strip().lower():
                parent_block = section.find_parent(
                    "div", class_=lambda c: c and "moduleBlock" in c
                )
                if parent_block:
                    return parent_block

        return None

    def extract_accident_description(self, soup: BeautifulSoup) -> Dict:
        """Extract accident description information."""
        accident_info = {}

        description_block = self.extract_section_content(soup, "Accident Description")

        if description_block:
            details_div = description_block.find("div", class_="details")
            if details_div:
                for p in details_div.find_all("p"):
                    text = p.get_text(strip=True)

                    if ":" in text:
                        key, value = text.split(":", 1)
                        key = key.strip().replace("Location: ", "")
                        value = value.strip().replace("|", "").strip()

                        if (
                            key == "Accident Occurred On"
                            and "Final Report Released On" in value
                        ):
                            parts = value.split("Final Report Released On:")
                            accident_info[key] = parts[0].strip()
                            accident_info["Final Report Released On"] = parts[1].strip()
                        else:
                            accident_info[key] = value

            description_p = description_block.find("p", class_=None)
            if description_p:
                accident_info["Description"] = description_p.get_text(strip=True)

            img_container = description_block.find("div", class_="imageContainer")
            if img_container and img_container.find("img"):
                img_src = img_container.find("img").get("src")
                if img_src:
                    accident_info["Image URL"] = urljoin(self.base_url, img_src)

        return accident_info

    def extract_final_reports(
        self, soup: BeautifulSoup, docs_folder: str
    ) -> List[Dict]:
        """Extract final reports information and download files."""
        reports_info = []

        reports_block = self.extract_section_content(soup, "Final Reports")

        if reports_block:
            report_links = reports_block.find_all("a")

            for link in report_links:
                href = link.get("href")
                if href:
                    report_url = urljoin(self.base_url, href)
                    report_text = link.get_text(strip=True)

                    local_path = self.download_file(report_url, docs_folder)

                    reports_info.append(
                        {
                            "title": report_text,
                            "url": report_url,
                            "local_path": local_path,
                        }
                    )

        return reports_info

    def extract_recommendations(self, soup: BeautifulSoup) -> List[Dict]:
        """Extract recommendations information."""
        recommendations_info = []

        recom_block = self.extract_section_content(soup, "Recommendations")

        if recom_block:
            rec_groups = recom_block.find_all("li", class_="ac")

            for group in rec_groups:
                group_info = {"target": "", "recommendations": []}

                target_a = group.find("a", class_="head")
                if target_a:
                    group_info["target"] = target_a.get_text(strip=True)

                rec_items = group.find_all("div", class_="recSubItem")

                for item in rec_items:
                    rec_info = {
                        "number": "",
                        "description": "",
                        "status": "",
                        "documents": [],
                    }

                    rec_num = item.find("div", class_="recNumber")
                    if rec_num:
                        rec_info["number"] = rec_num.get_text(strip=True)

                    desc_span = item.find("span", id=lambda x: x and "lblDesc" in x)
                    if desc_span:
                        rec_info["description"] = desc_span.get_text(strip=True)

                    status_elem = item.find("a", class_="tooltip")
                    if status_elem:
                        rec_info["status"] = status_elem.get_text(strip=True)

                    doc_links = item.find_all(
                        "a",
                        href=lambda x: x and (x.endswith(".pdf") or "/assets/" in x),
                    )

                    for doc_link in doc_links:
                        if doc_link.get("href") and doc_link.get_text().strip() != "":
                            rec_info["documents"].append(
                                {
                                    "url": urljoin(self.base_url, doc_link.get("href")),
                                    "title": doc_link.get_text(strip=True),
                                }
                            )

                    group_info["recommendations"].append(rec_info)

                recommendations_info.append(group_info)

        return recommendations_info

    def extract_related_documents(
        self, soup: BeautifulSoup, docs_folder: str
    ) -> List[Dict]:
        """Extract related documents information and download files."""
        documents_info = []

        docs_block = self.extract_section_content(soup, "Related Documents")

        if docs_block:
            doc_links = docs_block.find_all("a")

            for link in doc_links:
                href = link.get("href")
                if href:
                    doc_url = urljoin(self.base_url, href)
                    doc_text = link.get_text(strip=True)

                    local_path = self.download_file(doc_url, docs_folder)

                    documents_info.append(
                        {"title": doc_text, "url": doc_url, "local_path": local_path}
                    )

        return documents_info

    def extract_related_news(self, soup: BeautifulSoup) -> List[Dict]:
        """Extract related news information."""
        news_info = []

        news_block = self.extract_section_content(soup, "Related News")

        if news_block:
            news_items = news_block.find_all("div", class_="phgal")

            for item in news_items:
                news_data = {"title": "", "date": "", "url": "", "image_url": ""}

                news_link = item.find("a", class_="rmore")
                if news_link:
                    news_data["url"] = urljoin(self.base_url, news_link.get("href", ""))

                    content_span = news_link.find("span", class_="content")
                    if content_span:
                        title_span = content_span.find("span", class_="subTitle")
                        if title_span:
                            news_data["title"] = title_span.get_text(strip=True)

                        date_span = content_span.find("span", class_="date")
                        if date_span:
                            news_data["date"] = date_span.get_text(strip=True)

                img = item.find("img")
                if img:
                    news_data["image_url"] = urljoin(self.base_url, img.get("src", ""))

                news_info.append(news_data)

        return news_info

    def extract_statistics(self, soup: BeautifulSoup) -> Dict:
        """Extract statistics information."""
        stats_info = {}

        stats_block = self.extract_section_content(soup, "Statistics")

        if stats_block:
            legend_div = stats_block.find("div", class_="legend")
            if legend_div:
                for div in legend_div.find_all("div"):
                    text = div.get_text(strip=True)

                    if "Total Recommendations" in text:
                        match = re.search(r"Total Recommendations\s*#\s*(\d+)", text)
                        if match:
                            stats_info["total_recommendations"] = int(match.group(1))

                    elif "Open Recommendations" in text:
                        match = re.search(r"Open Recommendations\s*#\s*(\d+)", text)
                        if match:
                            stats_info["open_recommendations"] = int(match.group(1))

                    elif "Closed Recommendations" in text:
                        match = re.search(r"Closed Recommendations\s*#\s*(\d+)", text)
                        if match:
                            stats_info["closed_recommendations"] = int(match.group(1))

            vs_div = stats_block.find("div", class_="vs")
            if vs_div:
                nums_span = vs_div.find("span", class_="nums")
                if nums_span:
                    text = nums_span.get_text(strip=True)
                    match = re.search(r"(\d+)\s+vs\s+(\d+)", text)
                    if match:
                        stats_info["open_percentage"] = int(match.group(1))
                        stats_info["closed_percentage"] = int(match.group(2))

        return stats_info

    def extract_videos(self, soup: BeautifulSoup) -> List[Dict]:
        """Extract video links if present."""
        video_info = []

        video_section = self.extract_section_content(soup, "Video")

        if video_section:
            iframes = video_section.find_all("iframe")
            for iframe in iframes:
                src = iframe.get("src", "")
                if src:
                    video_info.append(
                        {
                            "type": "embedded",
                            "url": src,
                            "title": iframe.get("title", "Embedded Video"),
                        }
                    )

            video_links = video_section.find_all(
                "a", href=lambda x: x and ("youtube.com" in x or "vimeo.com" in x)
            )
            for link in video_links:
                video_info.append(
                    {
                        "type": "link",
                        "url": link.get("href"),
                        "title": link.get_text(strip=True) or "Video Link",
                    }
                )

        # Check for videos in other sections
        all_iframes = soup.find_all("iframe")
        for iframe in all_iframes:
            src = iframe.get("src", "")
            if src and ("youtube.com" in src or "vimeo.com" in src):
                existing = any(v["url"] == src for v in video_info)
                if not existing:
                    video_info.append(
                        {
                            "type": "embedded",
                            "url": src,
                            "title": iframe.get("title", "Embedded Video"),
                        }
                    )

        return video_info

    def get_investigation_links(self, base_url: str) -> List[str]:
        """Get all investigation links from the completed investigations page with pagination."""
        print(f"Fetching investigation links from {base_url}")

        all_links = []
        page_num = 1

        while True:
            page_url = f"{base_url}?Type=2&pg={page_num}"
            print(f"Processing page {page_num}: {page_url}")

            response = requests.get(page_url, headers=self.headers)
            if response.status_code != 200:
                print(
                    f"Failed to fetch {page_url}, status code: {response.status_code}"
                )
                break

            soup = BeautifulSoup(response.text, "html.parser")
            items = soup.find_all("div", class_="item")

            if not items:
                print(f"No items found on page {page_num}. End of pagination.")
                break

            links_found = 0

            for item in items:
                link = item.find("a", class_="linkHd")
                if link:
                    href = link.get("href")
                    if href:
                        investigation_url = urljoin(self.base_url, href)
                        if investigation_url not in all_links:
                            all_links.append(investigation_url)
                            links_found += 1

            print(f"Found {links_found} links on page {page_num}")

            if links_found == 0:
                break

            page_num += 1
            time.sleep(SCRAPER_SLEEP_DELAY)

        print(
            f"Found a total of {len(all_links)} investigation links across {page_num} pages"
        )

        # Save links to CSV
        links_df = pd.DataFrame(
            {"number": range(1, len(all_links) + 1), "url": all_links}
        )

        csv_filename = os.path.join(self.csb_dir, CSB_LINKS_FILENAME)
        links_df.to_csv(csv_filename, index=False)
        print(f"Links saved to {csv_filename}")

        return all_links

    def process_investigation(self, url: str, csv_file_path: str) -> Optional[Dict]:
        """Process an investigation page and extract all relevant information."""
        print(f"\nProcessing investigation: {url}")

        response = requests.get(url, headers=self.headers)
        if response.status_code != 200:
            print(f"Failed to fetch {url}, status code: {response.status_code}")
            return None

        soup = BeautifulSoup(response.text, "html.parser")

        # Get the investigation title
        title = soup.find("h1")
        investigation_title = (
            title.get_text(strip=True) if title else "Unknown Investigation"
        )

        # Create folder structure
        folder_name = self.sanitize_filename(investigation_title)
        (
            folder_path,
            docs_folder,
            final_reports_folder,
            related_documents_folder,
            recommendations_folder,
        ) = self.create_investigation_folder(folder_name)

        # Extract information from all sections
        investigation_data = {
            "title": investigation_title,
            "url": url,
            "accident_info": self.extract_accident_description(soup),
            "final_reports": self.extract_final_reports(soup, final_reports_folder),
            "recommendations": self.extract_recommendations(soup),
            "related_documents": self.extract_related_documents(
                soup, related_documents_folder
            ),
            "related_news": self.extract_related_news(soup),
            "statistics": self.extract_statistics(soup),
            "videos": self.extract_videos(soup),
        }

        # Process recommendations documents
        for group in investigation_data["recommendations"]:
            for rec in group["recommendations"]:
                local_docs = []
                for doc in rec["documents"]:
                    local_path = self.download_file(doc["url"], recommendations_folder)
                    if local_path:
                        doc["local_path"] = local_path
                        local_docs.append(doc)
                rec["documents"] = local_docs

        # Save JSON data if it doesn't already exist
        json_path = os.path.join(folder_path, "investigation_data.json")
        if not os.path.exists(json_path):
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(investigation_data, f, indent=4, ensure_ascii=False)
            print(f"Investigation data saved to {json_path}")
        else:
            print(f"Investigation data already exists, skipping: {json_path}")

        # Write to CSV
        self.write_investigation_to_csv(investigation_data, csv_file_path)

        return investigation_data

    def write_investigation_to_csv(self, investigation_data: Dict, csv_file_path: str):
        """Write investigation data to CSV file."""

        def to_snake_case(name):
            s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
            s2 = re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1)
            return re.sub(r"[^a-zA-Z0-9]", "_", s2).lower()

        # Check if this investigation already exists in CSV
        investigation_title = investigation_data.get("title", "")
        investigation_url = investigation_data.get("url", "")

        if os.path.exists(csv_file_path):
            try:
                import csv as csv_module

                with open(csv_file_path, encoding="utf-8") as csvfile:
                    reader = csv_module.DictReader(csvfile)
                    for row in reader:
                        if (
                            row.get("title") == investigation_title
                            or row.get("url") == investigation_url
                        ):
                            print(
                                f"Investigation already exists in CSV, skipping: {investigation_title}"
                            )
                            return
            except Exception as e:
                print(f"Error checking existing CSV entries: {e}")

        csv_data = {}

        # Add basic investigation data
        csv_data["title"] = investigation_data.get("title", "")
        csv_data["url"] = investigation_data.get("url", "")

        # Flatten accident_info
        accident_info = investigation_data.get("accident_info", {})
        for key, value in accident_info.items():
            snake_key = to_snake_case(key)
            csv_data[snake_key] = value

        # Use JSON for complex data structures
        csv_data["final_reports"] = json.dumps(
            investigation_data.get("final_reports", []), ensure_ascii=False
        )
        csv_data["recommendations"] = json.dumps(
            investigation_data.get("recommendations", []), ensure_ascii=False
        )
        csv_data["related_documents"] = json.dumps(
            investigation_data.get("related_documents", []), ensure_ascii=False
        )
        csv_data["related_news"] = json.dumps(
            investigation_data.get("related_news", []), ensure_ascii=False
        )
        csv_data["statistics"] = json.dumps(
            investigation_data.get("statistics", {}), ensure_ascii=False
        )
        csv_data["videos"] = json.dumps(
            investigation_data.get("videos", []), ensure_ascii=False
        )

        # Write to CSV
        file_exists = os.path.isfile(csv_file_path)

        with open(csv_file_path, "a", newline="", encoding="utf-8") as csvfile:
            fieldnames = list(csv_data.keys())
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            if not file_exists:
                writer.writeheader()

            writer.writerow(csv_data)

        print(f"Investigation data appended to {csv_file_path}")

    def run_scraper(
        self, links: List[str], csv_file: str, limit: Optional[int] = None
    ) -> pd.DataFrame:
        """Run the scraper for the provided links with optional limit."""
        all_investigations = []

        csv_file_path = os.path.join(self.csb_dir, csv_file)
        links_to_process = links[:limit] if limit else links

        for i, link in enumerate(links_to_process):
            try:
                print(
                    f"\nProcessing investigation {i+1}/{len(links_to_process)}: {link}"
                )

                if i > 0:
                    time.sleep(2)

                investigation_data = self.process_investigation(link, csv_file_path)

                if investigation_data:
                    investigation_summary = {
                        "title": investigation_data["title"],
                        "url": investigation_data["url"],
                        "folder": self.sanitize_filename(investigation_data["title"]),
                    }
                    all_investigations.append(investigation_summary)

                    print(f"Completed processing: {investigation_data['title']}")

            except Exception as e:
                print(f"Error processing {link}: {e}")

        # Save master index
        index_path = os.path.join(self.csb_dir, "csb_investigation_index.json")
        with open(index_path, "w", encoding="utf-8") as f:
            json.dump(all_investigations, f, indent=4, ensure_ascii=False)

        print(f"\nCompleted scraping {len(all_investigations)} investigations")
        print(f"Master index saved to {index_path}")

        return pd.DataFrame(all_investigations)


def main():
    """Main function to run the CSB scraper."""
    scraper = CSBInvestigationScraper()

    # Get investigation links
    url_base = CSB_BASE_URL + "/investigations/completed-investigations/"
    investigation_links_csv_file_path = os.path.join(
        scraper.csb_dir, CSB_LINKS_FILENAME
    )

    if os.path.exists(investigation_links_csv_file_path):
        links_df = pd.read_csv(investigation_links_csv_file_path)
        investigation_links = links_df["url"].tolist()
        print(f"Loaded {len(investigation_links)} investigation links from CSV file.")
    else:
        investigation_links = scraper.get_investigation_links(url_base)
        print(
            f"Downloaded {len(investigation_links)} investigation links from website."
        )

    # Process investigations
    csv_file_name = CSB_DATA_FILENAME
    results = scraper.run_scraper(investigation_links, csv_file_name)

    return results


if __name__ == "__main__":
    main()
