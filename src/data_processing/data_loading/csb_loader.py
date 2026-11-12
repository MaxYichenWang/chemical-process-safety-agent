"""
CSB Data Loader

This module provides data loading and chunking functionality for CSB investigation reports
and recommendations. It implements TOC-guided hierarchical chunking for final reports and
minimal chunking for recommendations as per the RAG methodology.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import duckdb
import pandas as pd
from tqdm import tqdm

from .. import config
from ..utilities.utils import (
    chunk_text,
    count_tokens,
    create_chunk_metadata,
    generate_chunk_id,
)


class CSBDataLoader:
    """
    Load and chunk CSB investigation reports and recommendations for RAG systems.

    This loader supports:
    - Final reports: TOC-guided hierarchical section-based chunking
    - Recommendations: Minimal chunking (whole-document or paragraph-level)
    - Automatic discovery of reports from file system
    - DuckDB integration for efficient data filtering
    """

    def __init__(
        self,
        data_root: Optional[str] = None,
        chunk_size: int = config.CHUNK_SIZE,
        chunk_overlap: int = config.CHUNK_OVERLAP,
        max_tokens_per_chunk: int = config.MAX_TOKENS_PER_CHUNK,
    ):
        """
        Initialize the CSB data loader.

        Args:
            data_root: Root directory for CSB data (default: data/processed/csb_olmocr)
            chunk_size: Target tokens per chunk (aligned with validator target range)
            chunk_overlap: Overlap tokens between chunks
            max_tokens_per_chunk: Maximum tokens before forcing a split
        """
        if data_root is None:
            data_root = config.CSB_DATA_DIR

        self.data_root = Path(data_root)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_tokens_per_chunk = max_tokens_per_chunk

        # Verify data directories exist
        self._verify_data_structure()

        # Initialize DuckDB connection
        self.conn = duckdb.connect(database=":memory:")

    def _verify_data_structure(self) -> None:
        """Verify that required data directories exist."""
        required_dirs = ["json", "markdown", "toc"]
        for dir_name in required_dirs:
            dir_path = self.data_root / dir_name
            if not dir_path.exists():
                raise FileNotFoundError(
                    f"Required directory not found: {dir_path}. "
                    f"Please ensure CSB data is properly extracted."
                )

    def _discover_reports(self) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
        """
        Discover final reports and recommendations from the file system.

        Returns:
            Tuple of (final_reports_list, recommendations_list)
            Each list contains dicts with 'incident_name' and 'report_name' keys.
        """
        json_dir = self.data_root / "json"
        toc_dir = self.data_root / "toc"

        final_reports = []
        recommendations = []

        # Scan all incident folders in json directory
        if not json_dir.exists():
            return [], []

        for incident_dir in json_dir.iterdir():
            if not incident_dir.is_dir():
                continue

            incident_name = incident_dir.name

            # Find all structured JSON files in this incident folder
            for json_file in incident_dir.glob("*_structured.json"):
                # Extract report_name by removing "_structured.json" suffix
                report_name = json_file.stem.replace("_structured", "")

                # Check if TOC exists (final reports have TOC, recommendations don't)
                toc_path = toc_dir / incident_name / f"{report_name}_toc.json"
                if toc_path.exists():
                    final_reports.append(
                        {
                            "incident_name": incident_name,
                            "report_name": report_name,
                        }
                    )
                else:
                    recommendations.append(
                        {
                            "incident_name": incident_name,
                            "report_name": report_name,
                        }
                    )

        return final_reports, recommendations

    def load_report_json(self, incident_name: str, report_name: str) -> Dict[str, Any]:
        """
        Load a structured JSON file for a specific report.

        Args:
            incident_name: Incident identifier
            report_name: Report filename (without extension)

        Returns:
            Parsed JSON dictionary with pages and metadata
        """
        json_path = (
            self.data_root / "json" / incident_name / f"{report_name}_structured.json"
        )

        if not json_path.exists():
            raise FileNotFoundError(f"JSON file not found: {json_path}")

        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def load_toc(
        self, incident_name: str, report_name: str
    ) -> Optional[Dict[str, Any]]:
        """
        Load the Table of Contents (TOC) for a final report.

        Args:
            incident_name: Incident identifier
            report_name: Report filename (without extension)

        Returns:
            TOC dictionary or None if not available
        """
        toc_path = self.data_root / "toc" / incident_name / f"{report_name}_toc.json"

        if not toc_path.exists():
            return None

        with open(toc_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _flatten_toc(
        self,
        toc_node: Dict[str, Any],
        parent_path: str = "",
        result: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Flatten hierarchical TOC structure into a list of sections.

        Args:
            toc_node: TOC node (recursive structure)
            parent_path: Path of parent sections
            result: Accumulated result list

        Returns:
            List of flattened TOC entries with hierarchy information
        """
        if result is None:
            result = []

        if "title" in toc_node and toc_node["title"] != "Document Root":
            # Build section path
            current_path = (
                f"{parent_path} / {toc_node['title']}"
                if parent_path
                else toc_node["title"]
            )

            section_info = {
                "title": toc_node["title"],
                "level": toc_node.get("level", 0),
                "page_number": toc_node.get("pageNumber"),
                "position": toc_node.get("position"),
                "section_path": current_path,
                "parent_path": parent_path or None,
            }

            result.append(section_info)

            # Recursively process children
            if "children" in toc_node and toc_node["children"]:
                for child in toc_node["children"]:
                    self._flatten_toc(child, current_path, result)
        else:
            # Root node - process children directly
            if "children" in toc_node and toc_node["children"]:
                for child in toc_node["children"]:
                    self._flatten_toc(child, parent_path, result)

        return result

    def _extract_text_by_pages(
        self,
        report_data: Dict[str, Any],
        start_page: int,
        end_page: Optional[int] = None,
    ) -> str:
        """
        Extract text from a range of pages in the report.

        Args:
            report_data: Parsed JSON report data
            start_page: Starting page number (1-indexed)
            end_page: Ending page number (inclusive), or None for single page

        Returns:
            Concatenated text from the page range
        """
        if end_page is None:
            end_page = start_page

        pages = report_data.get("pages", [])
        text_parts = []

        for page in pages:
            page_num = page.get("page_number")
            if start_page <= page_num <= end_page:
                text_parts.append(page.get("text", ""))

        return "\n\n".join(text_parts)

    def chunk_final_report(
        self,
        incident_name: str,
        report_name: str,
        report_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Chunk a final report using TOC-guided hierarchical chunking.

        Args:
            incident_name: Incident identifier
            report_name: Report filename
            report_metadata: Optional metadata from summary CSV

        Returns:
            List of chunk dictionaries with text and metadata
        """
        # Load report data and TOC
        report_data = self.load_report_json(incident_name, report_name)
        toc_data = self.load_toc(incident_name, report_name)

        if toc_data is None:
            # Fallback: chunk without TOC (rare case)
            return self._chunk_without_toc(
                incident_name, report_name, report_data, report_metadata
            )

        # Flatten TOC to get section boundaries
        sections = self._flatten_toc(toc_data)

        # If TOC has no sections (empty or no children), fall back to page-based chunking
        if not sections:
            return self._chunk_without_toc(
                incident_name, report_name, report_data, report_metadata
            )

        # Filter out sections with None page_number and add end page numbers
        valid_sections = []
        for section in sections:
            if section.get("page_number") is not None:
                valid_sections.append(section)

        # If no valid sections after filtering, fall back
        if not valid_sections:
            return self._chunk_without_toc(
                incident_name, report_name, report_data, report_metadata
            )

        sections = valid_sections

        # Add end page numbers for each section
        for i, section in enumerate(sections):
            if i < len(sections) - 1:
                next_page = sections[i + 1].get("page_number")
                if next_page is not None:
                    section["end_page"] = next_page - 1
                else:
                    section["end_page"] = len(report_data.get("pages", []))
            else:
                # Last section goes to end of document
                section["end_page"] = len(report_data.get("pages", []))

        chunks = []

        # Process each section
        for section_idx, section in enumerate(sections):
            # Skip if page_number is still None (shouldn't happen after filtering, but safety check)
            if section.get("page_number") is None:
                continue

            # Extract section text
            section_text = self._extract_text_by_pages(
                report_data, section["page_number"], section["end_page"]
            )

            if not section_text.strip():
                continue

            # Check if section needs splitting
            token_count = count_tokens(section_text)

            if token_count <= self.max_tokens_per_chunk:
                # Single chunk for this section
                chunk_id = generate_chunk_id(
                    "csb_reports", f"{incident_name}_{section['position']}"
                )

                chunk = create_chunk_metadata(
                    chunk_id=chunk_id,
                    corpus="csb",
                    subcorpus="csb_reports",
                    doc_type="csb_final_report",
                    text=section_text,
                    source_path=str(
                        self.data_root
                        / "json"
                        / incident_name
                        / f"{report_name}_structured.json"
                    ),
                    incident_name=incident_name,
                    report_name=report_name,
                    section_title=section["title"],
                    section_level=section["level"],
                    section_path=section["section_path"],
                    parent_section_path=section["parent_path"],
                    page_start=section["page_number"],
                    page_end=section["end_page"],
                    chunk_index=0,
                    total_chunks=1,
                )

                # Add report-level metadata if available
                if report_metadata:
                    chunk.update(
                        {
                            "num_pages": report_metadata.get("num_pages"),
                            "extraction_date": report_metadata.get("extraction_date"),
                            "processing_date": report_metadata.get("processing_date"),
                        }
                    )

                chunks.append(chunk)
            else:
                # Split section into multiple chunks
                section_chunks = chunk_text(
                    section_text, self.chunk_size, self.chunk_overlap
                )

                for chunk_idx, chunk_text_content in enumerate(section_chunks):
                    chunk_id = generate_chunk_id(
                        "csb_reports",
                        f"{incident_name}_{section['position']}",
                        chunk_idx,
                    )

                    chunk = create_chunk_metadata(
                        chunk_id=chunk_id,
                        corpus="csb",
                        subcorpus="csb_reports",
                        doc_type="csb_final_report",
                        text=chunk_text_content,
                        source_path=str(
                            self.data_root
                            / "json"
                            / incident_name
                            / f"{report_name}_structured.json"
                        ),
                        incident_name=incident_name,
                        report_name=report_name,
                        section_title=section["title"],
                        section_level=section["level"],
                        section_path=section["section_path"],
                        parent_section_path=section["parent_path"],
                        page_start=section["page_number"],
                        page_end=section["end_page"],
                        chunk_index=chunk_idx,
                        total_chunks=len(section_chunks),
                    )

                    if report_metadata:
                        chunk.update(
                            {
                                "num_pages": report_metadata.get("num_pages"),
                                "extraction_date": report_metadata.get(
                                    "extraction_date"
                                ),
                                "processing_date": report_metadata.get(
                                    "processing_date"
                                ),
                            }
                        )

                    chunks.append(chunk)

        return chunks

    def _chunk_without_toc(
        self,
        incident_name: str,
        report_name: str,
        report_data: Dict[str, Any],
        report_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Fallback chunking when TOC is not available.

        Simply concatenates all pages and chunks at paragraph boundaries.
        """
        # Extract all text
        all_text = self._extract_text_by_pages(
            report_data, 1, len(report_data.get("pages", []))
        )

        # Chunk the text
        text_chunks = chunk_text(all_text, self.chunk_size, self.chunk_overlap)

        chunks = []
        for chunk_idx, chunk_text_content in enumerate(text_chunks):
            chunk_id = generate_chunk_id(
                "csb_reports", f"{incident_name}_page_based", chunk_idx
            )

            chunk = create_chunk_metadata(
                chunk_id=chunk_id,
                corpus="csb",
                subcorpus="csb_reports",
                doc_type="csb_final_report",
                text=chunk_text_content,
                source_path=str(
                    self.data_root
                    / "json"
                    / incident_name
                    / f"{report_name}_structured.json"
                ),
                incident_name=incident_name,
                report_name=report_name,
                chunk_index=chunk_idx,
                total_chunks=len(text_chunks),
                chunking_method="page_based_fallback",
            )

            if report_metadata:
                chunk.update(
                    {
                        "num_pages": report_metadata.get("num_pages"),
                        "extraction_date": report_metadata.get("extraction_date"),
                        "processing_date": report_metadata.get("processing_date"),
                    }
                )

            chunks.append(chunk)

        return chunks

    def chunk_recommendation(
        self,
        incident_name: str,
        report_name: str,
        recommendation_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Chunk a recommendation document with minimal splitting.

        Args:
            incident_name: Incident identifier
            report_name: Recommendation filename
            recommendation_metadata: Optional metadata from summary CSV

        Returns:
            List of chunk dictionaries (typically 1 chunk per recommendation)
        """
        # Load recommendation data
        report_data = self.load_report_json(incident_name, report_name)

        # Extract simple section metadata (if available) so we preserve some
        # hierarchy information even though we use minimal chunking.
        #
        # The structured JSON for recommendations contains a top-level
        # "sections" array with entries like:
        #   {"title": "Recommendation Text", "level": 1, ...}
        #
        # For RAG, it's useful to at least know the primary section title/level
        # that the recommendation text comes from, so we expose those as
        # `section_title` and `section_level` in the chunk metadata. We keep
        # chunking minimal (usually 1 chunk per document) to match the current
        # data loading methodology.
        primary_section_title: Optional[str] = None
        primary_section_level: Optional[int] = None

        sections = report_data.get("sections") or []
        if isinstance(sections, list) and sections:
            # Prefer the canonical "Recommendation Text" section when present;
            # otherwise fall back to the first top-level section.
            primary_section = None
            for section in sections:
                if (
                    isinstance(section, dict)
                    and section.get("title") == "Recommendation Text"
                ):
                    primary_section = section
                    break

            if primary_section is None and isinstance(sections[0], dict):
                primary_section = sections[0]

            if isinstance(primary_section, dict):
                primary_section_title = primary_section.get("title")
                # Some JSONs may omit level; default to 1 in that case.
                primary_section_level = primary_section.get("level", 1)

        # Extract all text
        all_text = self._extract_text_by_pages(
            report_data, 1, len(report_data.get("pages", []))
        )

        # Check if splitting is needed
        token_count = count_tokens(all_text)

        if token_count <= self.max_tokens_per_chunk:
            # Single chunk for entire recommendation
            chunk_id = generate_chunk_id("csb_recs", f"{incident_name}_{report_name}")

            chunk = create_chunk_metadata(
                chunk_id=chunk_id,
                corpus="csb",
                subcorpus="csb_recs",
                doc_type="csb_recommendation",
                text=all_text,
                source_path=str(
                    self.data_root
                    / "json"
                    / incident_name
                    / f"{report_name}_structured.json"
                ),
                incident_name=incident_name,
                report_name=report_name,
                chunk_index=0,
                total_chunks=1,
                section_title=primary_section_title,
                section_level=primary_section_level,
            )

            if recommendation_metadata:
                chunk.update(
                    {
                        "num_pages": recommendation_metadata.get("num_pages"),
                        "extraction_date": recommendation_metadata.get(
                            "extraction_date"
                        ),
                        "processing_date": recommendation_metadata.get(
                            "processing_date"
                        ),
                    }
                )

            return [chunk]
        else:
            # Split at paragraph boundaries
            text_chunks = chunk_text(all_text, self.chunk_size, self.chunk_overlap)

            chunks = []
            for chunk_idx, chunk_text_content in enumerate(text_chunks):
                chunk_id = generate_chunk_id(
                    "csb_recs", f"{incident_name}_{report_name}", chunk_idx
                )

                chunk = create_chunk_metadata(
                    chunk_id=chunk_id,
                    corpus="csb",
                    subcorpus="csb_recs",
                    doc_type="csb_recommendation",
                    text=chunk_text_content,
                    source_path=str(
                        self.data_root
                        / "json"
                        / incident_name
                        / f"{report_name}_structured.json"
                    ),
                    incident_name=incident_name,
                    report_name=report_name,
                    chunk_index=chunk_idx,
                    total_chunks=len(text_chunks),
                    section_title=primary_section_title,
                    section_level=primary_section_level,
                )

                if recommendation_metadata:
                    chunk.update(
                        {
                            "num_pages": recommendation_metadata.get("num_pages"),
                            "extraction_date": recommendation_metadata.get(
                                "extraction_date"
                            ),
                            "processing_date": recommendation_metadata.get(
                                "processing_date"
                            ),
                        }
                    )

                chunks.append(chunk)

            return chunks

    def load_all_final_reports(
        self, limit: Optional[int] = None, show_progress: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Load and chunk all final reports.

        Args:
            limit: Optional limit on number of reports to process
            show_progress: Whether to show progress bar

        Returns:
            List of all chunks from all final reports
        """
        final_reports, _ = self._discover_reports()

        if limit:
            final_reports = final_reports[:limit]

        all_chunks = []

        iterator = final_reports
        if show_progress:
            iterator = tqdm(
                final_reports,
                desc="Processing final reports",
                total=len(final_reports),
            )

        failed_reports = []
        for report_info in iterator:
            try:
                chunks = self.chunk_final_report(
                    incident_name=report_info["incident_name"],
                    report_name=report_info["report_name"],
                    report_metadata=None,
                )
                if not chunks:
                    failed_reports.append(
                        {
                            "incident": report_info["incident_name"],
                            "report": report_info["report_name"],
                            "reason": "No chunks generated (empty TOC or no content)",
                        }
                    )
                all_chunks.extend(chunks)
            except Exception as e:
                failed_reports.append(
                    {
                        "incident": report_info["incident_name"],
                        "report": report_info["report_name"],
                        "reason": f"{type(e).__name__}: {str(e)}",
                    }
                )
                print(
                    f"Error processing {report_info['incident_name']}/{report_info['report_name']}: {e}"
                )
                continue

        if failed_reports:
            print(f"\n⚠️  Warning: {len(failed_reports)} reports failed to load:")
            for failed in failed_reports[:10]:  # Show first 10
                print(
                    f"  - {failed['incident']}/{failed['report']}: {failed['reason']}"
                )
            if len(failed_reports) > 10:
                print(f"  ... and {len(failed_reports) - 10} more")

        return all_chunks

    def load_all_recommendations(
        self, limit: Optional[int] = None, show_progress: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Load and chunk all recommendations.

        Args:
            limit: Optional limit on number of recommendations to process
            show_progress: Whether to show progress bar

        Returns:
            List of all chunks from all recommendations
        """
        _, recommendations = self._discover_reports()

        if limit:
            recommendations = recommendations[:limit]

        all_chunks = []

        iterator = recommendations
        if show_progress:
            iterator = tqdm(
                recommendations,
                desc="Processing recommendations",
                total=len(recommendations),
            )

        for rec_info in iterator:
            try:
                chunks = self.chunk_recommendation(
                    incident_name=rec_info["incident_name"],
                    report_name=rec_info["report_name"],
                    recommendation_metadata=None,
                )
                all_chunks.extend(chunks)
            except Exception as e:
                print(
                    f"Error processing {rec_info['incident_name']}/{rec_info['report_name']}: {e}"
                )
                continue

        return all_chunks

    def export_chunks_to_parquet(
        self, chunks: List[Dict[str, Any]], output_path: str
    ) -> None:
        """
        Export chunks to Parquet format for storage.

        Args:
            chunks: List of chunk dictionaries
            output_path: Output Parquet file path
        """
        df = pd.DataFrame(chunks)
        df.to_parquet(output_path, index=False, compression="snappy")
        print(f"Exported {len(chunks)} chunks to {output_path}")
