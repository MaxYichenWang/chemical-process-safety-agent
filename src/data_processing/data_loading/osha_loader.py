"""
OSHA Data Loader

This module provides data loading and chunking functionality for OSHA regulations
and enforcement data. It implements section-preserving chunking for regulations and
record-level chunking for enforcement violations and accidents.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import duckdb
import pandas as pd
from tqdm import tqdm

from src.data_processing.config import KEY_OSHA_STANDARDS

from .. import config
from ..utilities.utils import (
    chunk_text,
    count_tokens,
    create_chunk_metadata,
    extract_section_hierarchy,
    generate_chunk_id,
    normalize_standard_number,
)


class OSHARegulationsLoader:
    """
    Load and chunk OSHA regulations with section-preserving strategy.

    Preserves natural section boundaries as primary chunks. Large sections
    (>800 tokens) are split at paragraph boundaries while maintaining metadata.
    """

    def __init__(
        self,
        data_root: Optional[str] = None,
        chunk_size: int = config.CHUNK_SIZE,
        chunk_overlap: int = config.CHUNK_OVERLAP,
    ):
        """
        Initialize the OSHA regulations loader.

        Args:
            data_root: Root directory for OSHA data (default: data/processed/osha)
            chunk_size: Target tokens per chunk for large sections
            chunk_overlap: Overlap tokens between chunks
        """
        if data_root is None:
            data_root = config.OSHA_DATA_DIR

        self.data_root = Path(data_root)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Verify regulations file exists
        self.regulations_path = (
            self.data_root / "regulations" / "regulations_sections.parquet"
        )
        if not self.regulations_path.exists():
            raise FileNotFoundError(
                f"Regulations file not found: {self.regulations_path}"
            )

        # Initialize DuckDB connection
        self.conn = duckdb.connect(database=":memory:")

    def load_regulations(
        self,
        part_number: Optional[str] = None,
        standard_number: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Load OSHA regulations from Parquet with optional filtering.

        Args:
            part_number: Filter by part number (e.g., '1910', '1926')
            standard_number: Filter by specific standard (e.g., '1910.147')
            limit: Limit number of records

        Returns:
            DataFrame with regulation sections
        """
        query = f"SELECT * FROM '{self.regulations_path}'"

        where_clauses = []
        if part_number:
            where_clauses.append(f"part_number = '{part_number}'")
        if standard_number:
            norm_standard = normalize_standard_number(standard_number)
            where_clauses.append(f"standard_number LIKE '{norm_standard}%'")

        if where_clauses:
            query += " WHERE " + " AND ".join(where_clauses)

        if limit:
            query += f" LIMIT {limit}"

        return self.conn.execute(query).df()

    def chunk_regulation_section(
        self, section_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Chunk a single regulation section.

        Args:
            section_data: Dictionary with section data

        Returns:
            List of chunk dictionaries (1 or more depending on section size)
        """
        text = section_data.get("text", "") or section_data.get("section_text", "")
        if not text or not text.strip():
            # Return empty list for sections with no text (these are filtered out)
            # This is expected behavior - empty sections don't need chunks
            return []

        token_count = count_tokens(text)

        # Extract section information
        section_id = section_data.get("section_id", "") or section_data.get(
            "standard_number", ""
        )
        section_hierarchy = extract_section_hierarchy(section_id)

        chunks = []

        if token_count <= self.chunk_size:
            # Small section - use as-is
            chunk_id = generate_chunk_id("osha_reg", section_id)

            chunk = create_chunk_metadata(
                chunk_id=chunk_id,
                corpus="osha_reg",
                subcorpus="osha_reg",
                doc_type="regulation",
                text=text,
                source_path=str(self.regulations_path),
                source_url=section_data.get("url"),
                standard_number=section_data.get("standard_number"),
                section_id=section_id,
                title=section_data.get("title") or section_data.get("section_title"),
                part_number=section_hierarchy["part"],
                section_level=section_hierarchy["level"],
                chunk_index=0,
                total_chunks=1,
            )

            chunks.append(chunk)
        else:
            # Large section - split at paragraph boundaries
            text_chunks = chunk_text(text, self.chunk_size, self.chunk_overlap)

            for chunk_idx, chunk_text_content in enumerate(text_chunks):
                chunk_id = generate_chunk_id("osha_reg", section_id, chunk_idx)

                chunk = create_chunk_metadata(
                    chunk_id=chunk_id,
                    corpus="osha_reg",
                    subcorpus="osha_reg",
                    doc_type="regulation",
                    text=chunk_text_content,
                    source_path=str(self.regulations_path),
                    source_url=section_data.get("url"),
                    standard_number=section_data.get("standard_number"),
                    section_id=section_id,
                    title=section_data.get("title")
                    or section_data.get("section_title"),
                    part_number=section_hierarchy["part"],
                    section_level=section_hierarchy["level"],
                    chunk_index=chunk_idx,
                    total_chunks=len(text_chunks),
                )

                chunks.append(chunk)

        return chunks

    def chunk_violation(self, violation_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Create chunk(s) for a single violation record.

        This implementation supports both single-chunk and multi-chunk records,
        depending on the token length of the violation text.

        Args:
            violation_data: Dictionary with violation data

        Returns:
            List of chunk dictionaries (one or more depending on text length)
        """
        # Use the pre-constructed text field
        text = violation_data.get("text", "")
        if not text or not text.strip():
            return []

        # Generate base chunk ID
        activity_nr = violation_data.get("activity_nr", "unknown")
        citation_id = violation_data.get("citation_id", "unknown")
        base_chunk_id = f"osha_viol_{activity_nr}_{citation_id}"

        # Check length and optionally split into multiple chunks
        token_count = count_tokens(text)
        chunks: List[Dict[str, Any]] = []

        if token_count <= self.max_tokens_per_chunk:
            # Single chunk record
            chunks.append(
                create_chunk_metadata(
                    chunk_id=base_chunk_id,
                    corpus="osha_enf",
                    subcorpus="osha_enf",
                    doc_type="enforcement_violation",
                    text=text,
                    source_path=str(self.enforcement_dir / "violations_text.parquet"),
                    activity_nr=str(activity_nr),
                    citation_id=str(citation_id),
                    standard_number=violation_data.get("standard_number_norm"),
                    viol_type=violation_data.get("viol_type"),
                    initial_penalty=violation_data.get("initial_penalty"),
                    current_penalty=violation_data.get("current_penalty"),
                    site_state=violation_data.get("site_state"),
                    site_city=violation_data.get("site_city"),
                    naics_code=violation_data.get("naics_code"),
                    sic_code=violation_data.get("sic_code"),
                    issuance_date=(
                        str(violation_data.get("issuance_date"))
                        if violation_data.get("issuance_date")
                        else None
                    ),
                    establishment_name=violation_data.get("estab_name"),
                    chunk_index=0,
                    total_chunks=1,
                )
            )
            return chunks

        # Multi-chunk record: split at paragraph boundaries
        text_chunks = chunk_text(text, self.chunk_size, self.chunk_overlap)
        total_chunks = len(text_chunks)

        for idx, chunk_text_content in enumerate(text_chunks):
            chunks.append(
                create_chunk_metadata(
                    chunk_id=f"{base_chunk_id}_chunk_{idx}",
                    corpus="osha_enf",
                    subcorpus="osha_enf",
                    doc_type="enforcement_violation",
                    text=chunk_text_content,
                    source_path=str(self.enforcement_dir / "violations_text.parquet"),
                    activity_nr=str(activity_nr),
                    citation_id=str(citation_id),
                    standard_number=violation_data.get("standard_number_norm"),
                    viol_type=violation_data.get("viol_type"),
                    initial_penalty=violation_data.get("initial_penalty"),
                    current_penalty=violation_data.get("current_penalty"),
                    site_state=violation_data.get("site_state"),
                    site_city=violation_data.get("site_city"),
                    naics_code=violation_data.get("naics_code"),
                    sic_code=violation_data.get("sic_code"),
                    issuance_date=(
                        str(violation_data.get("issuance_date"))
                        if violation_data.get("issuance_date")
                        else None
                    ),
                    establishment_name=violation_data.get("estab_name"),
                    chunk_index=idx,
                    total_chunks=total_chunks,
                )
            )

        return chunks

    def load_all_regulations(
        self,
        part_number: Optional[str] = None,
        limit: Optional[int] = None,
        show_progress: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Load and chunk all regulations.

        Args:
            part_number: Optional filter by part number
            limit: Optional limit on number of sections
            show_progress: Whether to show progress bar

        Returns:
            List of all chunks from regulations
        """
        regulations_df = self.load_regulations(part_number=part_number, limit=limit)

        all_chunks = []
        skipped_sections = []

        iterator = regulations_df.iterrows()
        if show_progress:
            iterator = tqdm(
                list(iterator),
                desc="Processing OSHA regulations",
                total=len(regulations_df),
            )

        for _, row in iterator:
            chunks = self.chunk_regulation_section(row.to_dict())
            if not chunks:
                # Track sections that were skipped (empty text)
                section_id = row.get("section_id", "") or row.get("standard_number", "")
                text_len = len(str(row.get("text", "") or row.get("section_text", "")))
                skipped_sections.append(
                    {"section_id": section_id, "text_length": text_len}
                )
            all_chunks.extend(chunks)

        if skipped_sections:
            print(
                f"\n⚠️  Note: {len(skipped_sections)} sections skipped (empty or whitespace-only text)"
            )
            if len(skipped_sections) <= 20:
                print("  Skipped sections:")
                for skipped in skipped_sections:
                    print(
                        f"    - {skipped['section_id']} ({skipped['text_length']} chars)"
                    )
            else:
                print("  First 10 skipped sections:")
                for skipped in skipped_sections[:10]:
                    print(
                        f"    - {skipped['section_id']} ({skipped['text_length']} chars)"
                    )
                print(f"  ... and {len(skipped_sections) - 10} more")

        return all_chunks

    def export_chunks_to_parquet(
        self, chunks: List[Dict[str, Any]], output_path: str
    ) -> None:
        """
        Export chunks to Parquet format.

        Args:
            chunks: List of chunk dictionaries
            output_path: Output Parquet file path
        """
        df = pd.DataFrame(chunks)
        df.to_parquet(output_path, index=False, compression="snappy")
        print(f"Exported {len(chunks)} regulation chunks to {output_path}")


class OSHAEnforcementLoader:
    """
    Load and chunk OSHA enforcement data (violations, accidents, and general duty violations).

    Uses record-level chunking - each violation, accident, or general duty citation is a single chunk
    with rich metadata from joined tables.

    Text-heavy tables (embedded as chunks):
    - violations_text.parquet: Standard violations
    - accidents_text.parquet: Accident narratives
    - violation_genduty_text.parquet: General Duty Clause [5(a)(1)] violations

    Structured tables (metadata enrichment only, not embedded):
    - inspection.parquet: Inspection metadata
    - violation_event.parquet: Historical penalty changes
    - related_activity.parquet: Inspection relationships
    - strategic_codes.parquet: Program classifications
    - optional_info.parquet: Supplementary metadata
    """

    def __init__(
        self,
        data_root: Optional[str] = None,
        chunk_size: int = config.CHUNK_SIZE,
        chunk_overlap: int = config.CHUNK_OVERLAP,
        max_tokens_per_chunk: int = config.MAX_TOKENS_PER_CHUNK,
    ):
        """
        Initialize the OSHA enforcement loader.

        Args:
            data_root: Root directory for OSHA data (default: data/processed/osha)
            chunk_size: Target tokens per chunk when splitting long records
            chunk_overlap: Overlap tokens between chunks
            max_tokens_per_chunk: Maximum tokens before forcing a split
        """
        if data_root is None:
            data_root = config.OSHA_DATA_DIR

        self.data_root = Path(data_root)
        self.enforcement_dir = self.data_root / "enforcement"

        # Verify enforcement directory exists
        if not self.enforcement_dir.exists():
            raise FileNotFoundError(
                f"Enforcement directory not found: {self.enforcement_dir}"
            )

        # Initialize DuckDB connection and chunking parameters
        self.conn = duckdb.connect(database=":memory:")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_tokens_per_chunk = max_tokens_per_chunk

    def load_violations(
        self,
        state: Optional[str] = None,
        standard_number: Optional[str] = None,
        violation_type: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Load OSHA violations with optional filtering.

        Args:
            state: Filter by state code (e.g., 'CA', 'TX')
            standard_number: Filter by OSHA standard (e.g., '1910.147')
            violation_type: Filter by violation type (e.g., 'Serious', 'Willful')
            limit: Limit number of records

        Returns:
            DataFrame with violations
        """
        violations_path = self.enforcement_dir / "violations_text.parquet"

        query = f"SELECT * FROM '{violations_path}'"

        where_clauses = []
        if state:
            where_clauses.append(f"site_state = '{state}'")
        if standard_number:
            norm_standard = normalize_standard_number(standard_number)
            where_clauses.append(f"standard_number_norm LIKE '{norm_standard}%'")
        if violation_type:
            where_clauses.append(f"viol_type = '{violation_type}'")

        if where_clauses:
            query += " WHERE " + " AND ".join(where_clauses)

        if limit:
            query += f" LIMIT {limit}"

        return self.conn.execute(query).df()

    def load_accidents(
        self,
        state: Optional[str] = None,
        fatality_only: bool = False,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Load OSHA accidents with optional filtering.

        Args:
            state: Filter by state code
            fatality_only: Only include accidents with fatalities
            limit: Limit number of records

        Returns:
            DataFrame with accidents
        """
        accidents_path = self.enforcement_dir / "accidents_text.parquet"

        query = f"SELECT * FROM '{accidents_path}'"

        where_clauses = []
        if state:
            where_clauses.append(f"state_flag = '{state}'")
        if fatality_only:
            where_clauses.append("fatality = 'Y'")

        if where_clauses:
            query += " WHERE " + " AND ".join(where_clauses)

        if limit:
            query += f" LIMIT {limit}"

        return self.conn.execute(query).df()

    def load_general_duty_violations(
        self,
        state: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Load OSHA General Duty Clause [5(a)(1)] violations with optional filtering.

        Args:
            state: Filter by state code (requires join with inspection table)
            limit: Limit number of records

        Returns:
            DataFrame with general duty violations
        """
        genduty_path = self.enforcement_dir / "violation_genduty_text.parquet"

        if not genduty_path.exists():
            raise FileNotFoundError(
                f"General duty violations file not found: {genduty_path}"
            )

        # If state filtering is requested, join with inspection table
        if state:
            inspection_path = self.enforcement_dir / "inspection.parquet"
            if not inspection_path.exists():
                raise FileNotFoundError(
                    f"Inspection file not found for state filtering: {inspection_path}"
                )

            query = f"""
                SELECT gd.*, i.site_state, i.site_city, i.estab_name, i.naics_code
                FROM '{genduty_path}' gd
                LEFT JOIN '{inspection_path}' i
                    ON gd.activity_nr = i.activity_nr
                WHERE i.site_state = '{state}'
            """
        else:
            query = f"SELECT * FROM '{genduty_path}'"

        if limit:
            query += f" LIMIT {limit}"

        return self.conn.execute(query).df()

    def chunk_violation(self, violation_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Create chunk for a single violation record.

        Args:
            violation_data: Dictionary with violation data

        Returns:
            List of chunk dictionaries (one or more depending on text length)
        """
        # Use the pre-constructed text field
        text = violation_data.get("text", "")
        if not text or not text.strip():
            return []

        # Generate base chunk ID
        activity_nr = violation_data.get("activity_nr", "unknown")
        citation_id = violation_data.get("citation_id", "unknown")
        base_chunk_id = f"osha_viol_{activity_nr}_{citation_id}"

        # Check length and optionally split into multiple chunks
        token_count = count_tokens(text)
        chunks: List[Dict[str, Any]] = []

        if token_count <= self.max_tokens_per_chunk:
            # Single chunk record
            chunks.append(
                create_chunk_metadata(
                    chunk_id=base_chunk_id,
                    corpus="osha_enf",
                    subcorpus="osha_enf",
                    doc_type="enforcement_violation",
                    text=text,
                    source_path=str(self.enforcement_dir / "violations_text.parquet"),
                    activity_nr=str(activity_nr),
                    citation_id=str(citation_id),
                    standard_number=violation_data.get("standard_number_norm"),
                    viol_type=violation_data.get("viol_type"),
                    initial_penalty=violation_data.get("initial_penalty"),
                    current_penalty=violation_data.get("current_penalty"),
                    site_state=violation_data.get("site_state"),
                    site_city=violation_data.get("site_city"),
                    naics_code=violation_data.get("naics_code"),
                    sic_code=violation_data.get("sic_code"),
                    issuance_date=(
                        str(violation_data.get("issuance_date"))
                        if violation_data.get("issuance_date")
                        else None
                    ),
                    establishment_name=violation_data.get("estab_name"),
                    chunk_index=0,
                    total_chunks=1,
                )
            )
            return chunks

        # Multi-chunk record: split at paragraph boundaries
        text_chunks = chunk_text(text, self.chunk_size, self.chunk_overlap)
        total_chunks = len(text_chunks)

        for idx, chunk_text_content in enumerate(text_chunks):
            chunks.append(
                create_chunk_metadata(
                    chunk_id=f"{base_chunk_id}_chunk_{idx}",
                    corpus="osha_enf",
                    subcorpus="osha_enf",
                    doc_type="enforcement_violation",
                    text=chunk_text_content,
                    source_path=str(self.enforcement_dir / "violations_text.parquet"),
                    activity_nr=str(activity_nr),
                    citation_id=str(citation_id),
                    standard_number=violation_data.get("standard_number_norm"),
                    viol_type=violation_data.get("viol_type"),
                    initial_penalty=violation_data.get("initial_penalty"),
                    current_penalty=violation_data.get("current_penalty"),
                    site_state=violation_data.get("site_state"),
                    site_city=violation_data.get("site_city"),
                    naics_code=violation_data.get("naics_code"),
                    sic_code=violation_data.get("sic_code"),
                    issuance_date=(
                        str(violation_data.get("issuance_date"))
                        if violation_data.get("issuance_date")
                        else None
                    ),
                    establishment_name=violation_data.get("estab_name"),
                    chunk_index=idx,
                    total_chunks=total_chunks,
                )
            )

        return chunks

    def chunk_accident(self, accident_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Create chunk for a single accident record.

        Args:
            accident_data: Dictionary with accident data

        Returns:
            List of chunk dictionaries
        """
        # Use the pre-constructed text field
        text = accident_data.get("text", "")
        if not text or not text.strip():
            return []

        # Generate base chunk ID
        summary_nr = accident_data.get("summary_nr", "unknown")
        base_chunk_id = f"osha_acc_{summary_nr}"

        # Check length and optionally split into multiple chunks
        token_count = count_tokens(text)
        chunks: List[Dict[str, Any]] = []

        if token_count <= self.max_tokens_per_chunk:
            chunks.append(
                create_chunk_metadata(
                    chunk_id=base_chunk_id,
                    corpus="osha_enf",
                    subcorpus="osha_enf",
                    doc_type="enforcement_accident",
                    text=text,
                    source_path=str(self.enforcement_dir / "accidents_text.parquet"),
                    summary_nr=str(summary_nr),
                    event_date=(
                        str(accident_data.get("event_date"))
                        if accident_data.get("event_date")
                        else None
                    ),
                    state=accident_data.get("state_flag"),
                    fatality=accident_data.get("fatality") == "Y",
                    hospitalization=accident_data.get("hospitalization", 0),
                    amputation=accident_data.get("amputation", 0),
                    event_description=accident_data.get("event_desc"),
                    event_keyword=accident_data.get("event_keyword"),
                    chunk_index=0,
                    total_chunks=1,
                )
            )
            return chunks

        # Multi-chunk record
        text_chunks = chunk_text(text, self.chunk_size, self.chunk_overlap)
        total_chunks = len(text_chunks)

        for idx, chunk_text_content in enumerate(text_chunks):
            chunks.append(
                create_chunk_metadata(
                    chunk_id=f"{base_chunk_id}_chunk_{idx}",
                    corpus="osha_enf",
                    subcorpus="osha_enf",
                    doc_type="enforcement_accident",
                    text=chunk_text_content,
                    source_path=str(self.enforcement_dir / "accidents_text.parquet"),
                    summary_nr=str(summary_nr),
                    event_date=(
                        str(accident_data.get("event_date"))
                        if accident_data.get("event_date")
                        else None
                    ),
                    state=accident_data.get("state_flag"),
                    fatality=accident_data.get("fatality") == "Y",
                    hospitalization=accident_data.get("hospitalization", 0),
                    amputation=accident_data.get("amputation", 0),
                    event_description=accident_data.get("event_desc"),
                    event_keyword=accident_data.get("event_keyword"),
                    chunk_index=idx,
                    total_chunks=total_chunks,
                )
            )

        return chunks

    def chunk_general_duty_violation(
        self, genduty_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Create chunk for a single General Duty Clause violation record.

        Args:
            genduty_data: Dictionary with general duty violation data

        Returns:
            List of chunk dictionaries
        """
        # Use the aggregated text field
        text = genduty_data.get("text", "")
        if not text or not text.strip():
            return []

        # Generate base chunk ID
        activity_nr = genduty_data.get("activity_nr", "unknown")
        citation_id = genduty_data.get("citation_id", "unknown")
        base_chunk_id = f"osha_genduty_{activity_nr}_{citation_id}"

        # Check length and optionally split into multiple chunks
        token_count = count_tokens(text)
        chunks: List[Dict[str, Any]] = []

        if token_count <= self.max_tokens_per_chunk:
            chunks.append(
                create_chunk_metadata(
                    chunk_id=base_chunk_id,
                    corpus="osha_enf",
                    subcorpus="osha_enf",
                    doc_type="enforcement_general_duty",
                    text=text,
                    source_path=str(
                        self.enforcement_dir / "violation_genduty_text.parquet"
                    ),
                    activity_nr=str(activity_nr),
                    citation_id=str(citation_id),
                    standard_number="5(a)(1)",  # General Duty Clause
                    site_state=genduty_data.get("site_state"),
                    site_city=genduty_data.get("site_city"),
                    naics_code=genduty_data.get("naics_code"),
                    establishment_name=genduty_data.get("estab_name"),
                    chunk_index=0,
                    total_chunks=1,
                )
            )
            return chunks

        text_chunks = chunk_text(text, self.chunk_size, self.chunk_overlap)
        total_chunks = len(text_chunks)

        for idx, chunk_text_content in enumerate(text_chunks):
            chunks.append(
                create_chunk_metadata(
                    chunk_id=f"{base_chunk_id}_chunk_{idx}",
                    corpus="osha_enf",
                    subcorpus="osha_enf",
                    doc_type="enforcement_general_duty",
                    text=chunk_text_content,
                    source_path=str(
                        self.enforcement_dir / "violation_genduty_text.parquet"
                    ),
                    activity_nr=str(activity_nr),
                    citation_id=str(citation_id),
                    standard_number="5(a)(1)",  # General Duty Clause
                    site_state=genduty_data.get("site_state"),
                    site_city=genduty_data.get("site_city"),
                    naics_code=genduty_data.get("naics_code"),
                    establishment_name=genduty_data.get("estab_name"),
                    chunk_index=idx,
                    total_chunks=total_chunks,
                )
            )

        return chunks

    def load_all_violations(
        self,
        state: Optional[str] = None,
        standard_number: Optional[str] = None,
        limit: Optional[int] = None,
        show_progress: bool = True,
        batch_size: int = 100000,
    ) -> List[Dict[str, Any]]:
        """
        Load and chunk all violations using batch processing to avoid memory issues.

        Args:
            state: Optional filter by state
            standard_number: Optional filter by standard
            limit: Optional limit on number of records
            show_progress: Whether to show progress bar
            batch_size: Number of records to process per batch (default: 100K)

        Returns:
            List of all violation chunks
        """
        # Get total count without loading all data
        violations_path = self.enforcement_dir / "violations_text.parquet"
        count_query = f"SELECT COUNT(*) as count FROM '{violations_path}'"

        where_clauses = []
        if state:
            where_clauses.append(f"site_state = '{state}'")
        if standard_number:
            norm_standard = normalize_standard_number(standard_number)
            where_clauses.append(f"standard_number_norm LIKE '{norm_standard}%'")

        if where_clauses:
            count_query += " WHERE " + " AND ".join(where_clauses)

        total_count = self.conn.execute(count_query).fetchone()[0]

        if limit:
            total_count = min(total_count, limit)

        all_chunks = []
        offset = 0

        # Process in batches
        if show_progress:
            pbar = tqdm(total=total_count, desc="Processing OSHA violations")

        while offset < total_count:
            current_batch_size = min(batch_size, total_count - offset)

            # Build query with LIMIT and OFFSET
            query = f"SELECT * FROM '{violations_path}'"
            if where_clauses:
                query += " WHERE " + " AND ".join(where_clauses)
            query += f" LIMIT {current_batch_size} OFFSET {offset}"

            # Load batch
            batch_df = self.conn.execute(query).df()

            # Process batch
            for _, row in batch_df.iterrows():
                chunks = self.chunk_violation(row.to_dict())
                if chunks:
                    all_chunks.extend(chunks)

            offset += current_batch_size

            if show_progress:
                pbar.update(current_batch_size)

        if show_progress:
            pbar.close()

        return all_chunks

    def load_all_accidents(
        self,
        state: Optional[str] = None,
        fatality_only: bool = False,
        limit: Optional[int] = None,
        show_progress: bool = True,
        batch_size: int = 50000,
    ) -> List[Dict[str, Any]]:
        """
        Load and chunk all accidents using batch processing.

        Args:
            state: Optional filter by state
            fatality_only: Only include accidents with fatalities
            limit: Optional limit on number of records
            show_progress: Whether to show progress bar
            batch_size: Number of records to process per batch (default: 50K)

        Returns:
            List of all accident chunks
        """
        # Get total count without loading all data
        accidents_path = self.enforcement_dir / "accidents_text.parquet"
        count_query = f"SELECT COUNT(*) as count FROM '{accidents_path}'"

        where_clauses = []
        if state:
            where_clauses.append(f"state_flag = '{state}'")
        if fatality_only:
            where_clauses.append("fatality = 'Y'")

        if where_clauses:
            count_query += " WHERE " + " AND ".join(where_clauses)

        total_count = self.conn.execute(count_query).fetchone()[0]

        if limit:
            total_count = min(total_count, limit)

        all_chunks = []
        offset = 0

        # Process in batches
        if show_progress:
            pbar = tqdm(total=total_count, desc="Processing OSHA accidents")

        while offset < total_count:
            current_batch_size = min(batch_size, total_count - offset)

            # Build query with LIMIT and OFFSET
            query = f"SELECT * FROM '{accidents_path}'"
            if where_clauses:
                query += " WHERE " + " AND ".join(where_clauses)
            query += f" LIMIT {current_batch_size} OFFSET {offset}"

            # Load batch
            batch_df = self.conn.execute(query).df()

            # Process batch
            for _, row in batch_df.iterrows():
                chunks = self.chunk_accident(row.to_dict())
                if chunks:
                    all_chunks.extend(chunks)

            offset += current_batch_size

            if show_progress:
                pbar.update(current_batch_size)

        if show_progress:
            pbar.close()

        return all_chunks

    def load_all_general_duty_violations(
        self,
        state: Optional[str] = None,
        limit: Optional[int] = None,
        show_progress: bool = True,
        batch_size: int = 100000,
    ) -> List[Dict[str, Any]]:
        """
        Load and chunk all General Duty Clause violations using batch processing.

        Args:
            state: Optional filter by state
            limit: Optional limit on number of records
            show_progress: Whether to show progress bar
            batch_size: Number of records to process per batch (default: 100K)

        Returns:
            List of all general duty violation chunks
        """
        genduty_path = self.enforcement_dir / "violation_genduty_text.parquet"

        if not genduty_path.exists():
            raise FileNotFoundError(
                f"General duty violations file not found: {genduty_path}"
            )

        # Get total count
        if state:
            inspection_path = self.enforcement_dir / "inspection.parquet"
            count_query = f"""
                SELECT COUNT(*) as count
                FROM '{genduty_path}' gd
                LEFT JOIN '{inspection_path}' i
                    ON gd.activity_nr = i.activity_nr
                WHERE i.site_state = '{state}'
            """
        else:
            count_query = f"SELECT COUNT(*) as count FROM '{genduty_path}'"

        total_count = self.conn.execute(count_query).fetchone()[0]

        if limit:
            total_count = min(total_count, limit)

        all_chunks = []
        offset = 0

        # Process in batches
        if show_progress:
            pbar = tqdm(
                total=total_count, desc="Processing OSHA general duty violations"
            )

        while offset < total_count:
            current_batch_size = min(batch_size, total_count - offset)

            # Build query with LIMIT and OFFSET
            if state:
                inspection_path = self.enforcement_dir / "inspection.parquet"
                query = f"""
                    SELECT gd.*, i.site_state, i.site_city, i.estab_name, i.naics_code
                    FROM '{genduty_path}' gd
                    LEFT JOIN '{inspection_path}' i
                        ON gd.activity_nr = i.activity_nr
                    WHERE i.site_state = '{state}'
                    LIMIT {current_batch_size} OFFSET {offset}
                """
            else:
                query = f"SELECT * FROM '{genduty_path}' LIMIT {current_batch_size} OFFSET {offset}"

            # Load batch
            batch_df = self.conn.execute(query).df()

            # Process batch
            for _, row in batch_df.iterrows():
                chunks = self.chunk_general_duty_violation(row.to_dict())
                if chunks:
                    all_chunks.extend(chunks)

            offset += current_batch_size

            if show_progress:
                pbar.update(current_batch_size)

        if show_progress:
            pbar.close()

        return all_chunks

    def _build_violations_sample_query(self, count_only: bool = False) -> str:
        """Build the SQL query for PSM-relevant violations sample.

        Joins violations_text with inspection for enrichment and optionally
        with the raw violations CSV for viol_type/initial_penalty filtering.

        Args:
            count_only: If True, return a COUNT(*) query instead of data
        """
        violations_path = self.enforcement_dir / "violations_text.parquet"
        inspection_path = self.enforcement_dir / "inspection.parquet"

        raw_violations_path = (
            self.data_root.parent.parent
            / "raw"
            / "osha"
            / "enforcement"
            / "osha_violation.csv"
        )

        standard_clauses = " OR ".join(
            f"v.standard_number_norm LIKE '{std}%'" for std in KEY_OSHA_STANDARDS
        )

        has_raw = raw_violations_path.exists()

        if has_raw:
            raw_read = f"read_csv('{raw_violations_path}', ignore_errors=true, null_padding=true)"
            from_clause = f"""
                '{violations_path}' v
                LEFT JOIN '{inspection_path}' i ON v.activity_nr = i.activity_nr
                LEFT JOIN {raw_read} rv
                    ON CAST(v.activity_nr AS VARCHAR) = CAST(rv.activity_nr AS VARCHAR)
                    AND CAST(v.citation_id AS VARCHAR) = CAST(rv.citation_id AS VARCHAR)
            """
            extra_filters = (
                "OR rv.viol_type IN ('W', 'R')"
                " OR TRY_CAST(rv.initial_penalty AS DOUBLE) >= ("
                f"  SELECT PERCENTILE_DISC(0.99) WITHIN GROUP (ORDER BY TRY_CAST(initial_penalty AS DOUBLE))"
                f"  FROM {raw_read} WHERE TRY_CAST(initial_penalty AS DOUBLE) IS NOT NULL"
                ")"
            )
            select_extra = ", rv.viol_type, rv.initial_penalty, rv.current_penalty, rv.issuance_date"
        else:
            from_clause = f"""
                '{violations_path}' v
                LEFT JOIN '{inspection_path}' i ON v.activity_nr = i.activity_nr
            """
            has_viol_type = self._column_exists(violations_path, "viol_type")
            has_penalty = self._column_exists(violations_path, "initial_penalty")
            extra_parts = []
            if has_viol_type:
                extra_parts.append("OR v.viol_type IN ('W', 'R')")
            if has_penalty:
                extra_parts.append(
                    f"OR v.initial_penalty >= ("
                    f"  SELECT PERCENTILE_DISC(0.99) WITHIN GROUP (ORDER BY initial_penalty)"
                    f"  FROM '{violations_path}')"
                )
            extra_filters = " ".join(extra_parts)
            select_extra = ""

        where_clause = f"({standard_clauses})"
        if extra_filters:
            where_clause += f" {extra_filters}"

        if count_only:
            return f"SELECT COUNT(*) as count FROM {from_clause} WHERE {where_clause}"

        select_clause = "v.*, i.estab_name, i.site_city, i.site_state, i.naics_code"
        return f"SELECT {select_clause}{select_extra} FROM {from_clause} WHERE {where_clause}"

    def _column_exists(self, parquet_path: Path, column_name: str) -> bool:
        """Check if a column exists in a parquet file."""
        try:
            self.conn.execute(f"SELECT {column_name} FROM '{parquet_path}' LIMIT 0")
            return True
        except Exception:
            return False

    def load_violations_sample(
        self,
        limit: Optional[int] = None,
        batch_size: int = 100000,
    ) -> pd.DataFrame:
        """Load a filtered sample of OSHA violations focused on PSM-relevant records.

        Filter criteria (OR-combined for broader coverage):
        - standard_number_norm starts with any value in KEY_OSHA_STANDARDS
        - viol_type IN ('W', 'R') (willful, repeat) -- from raw CSV if available
        - initial_penalty in top 1% -- from raw CSV if available

        Results are enriched via JOIN with inspection.parquet for establishment
        name, city, state, and NAICS code.

        Args:
            limit: Maximum number of rows to return
            batch_size: Number of rows per batch for memory-safe iteration

        Returns:
            Filtered DataFrame of PSM-relevant violations
        """
        violations_path = self.enforcement_dir / "violations_text.parquet"
        inspection_path = self.enforcement_dir / "inspection.parquet"

        if not violations_path.exists():
            raise FileNotFoundError(f"Violations file not found: {violations_path}")
        if not inspection_path.exists():
            raise FileNotFoundError(f"Inspection file not found: {inspection_path}")

        query = self._build_violations_sample_query()

        if limit:
            query += f" LIMIT {limit}"

        return self.conn.execute(query).df()

    def load_genduty_sample(
        self,
        limit: Optional[int] = None,
        keyword_filter: bool = True,
    ) -> pd.DataFrame:
        """Load a filtered sample of General Duty Clause violations for chemical industry.

        Filter criteria:
        - naics_code starts with '325' (chemicals) or '324' (petroleum/coal)
        - Optionally: keyword match in text for process safety terms

        Results are enriched via JOIN with inspection.parquet.

        Args:
            limit: Maximum number of rows to return
            keyword_filter: If True, also filter by process-safety keywords in text

        Returns:
            Filtered DataFrame of chemical-industry general duty violations
        """
        genduty_path = self.enforcement_dir / "violation_genduty_text.parquet"
        inspection_path = self.enforcement_dir / "inspection.parquet"

        if not genduty_path.exists():
            raise FileNotFoundError(f"General duty file not found: {genduty_path}")
        if not inspection_path.exists():
            raise FileNotFoundError(f"Inspection file not found: {inspection_path}")

        naics_filter = "(i.naics_code LIKE '325%' OR i.naics_code LIKE '324%')"

        if keyword_filter:
            keywords = [
                "chemical",
                "reactive",
                "explosion",
                "release",
                "toxic",
                "process safety",
            ]
            keyword_clauses = " OR ".join(
                f"LOWER(gd.text) LIKE '%{kw}%'" for kw in keywords
            )
            where_clause = f"{naics_filter} OR ({keyword_clauses})"
        else:
            where_clause = naics_filter

        query = f"""
            SELECT gd.*, i.estab_name, i.site_city, i.site_state, i.naics_code
            FROM '{genduty_path}' gd
            LEFT JOIN '{inspection_path}' i
                ON gd.activity_nr = i.activity_nr
            WHERE {where_clause}
        """

        if limit:
            query += f" LIMIT {limit}"

        return self.conn.execute(query).df()

    def load_all_violations_sample(
        self,
        limit: Optional[int] = None,
        show_progress: bool = True,
        batch_size: int = 100000,
    ) -> List[Dict[str, Any]]:
        """Load and chunk all PSM-relevant violation samples.

        Uses load_violations_sample() filter criteria for data retrieval and
        chunk_violation() for each row, with batched iteration for memory safety.

        Args:
            limit: Maximum number of source rows to process
            show_progress: Whether to show a progress bar
            batch_size: Number of rows per batch

        Returns:
            List of chunk dictionaries
        """
        violations_path = self.enforcement_dir / "violations_text.parquet"
        inspection_path = self.enforcement_dir / "inspection.parquet"

        if not violations_path.exists():
            raise FileNotFoundError(f"Violations file not found: {violations_path}")
        if not inspection_path.exists():
            raise FileNotFoundError(f"Inspection file not found: {inspection_path}")

        count_query = self._build_violations_sample_query(count_only=True)
        total_count = self.conn.execute(count_query).fetchone()[0]

        if limit:
            total_count = min(total_count, limit)

        all_chunks: List[Dict[str, Any]] = []
        offset = 0

        if show_progress:
            pbar = tqdm(total=total_count, desc="Processing violations sample")

        while offset < total_count:
            current_batch_size = min(batch_size, total_count - offset)

            data_query = self._build_violations_sample_query(count_only=False)
            data_query += f" LIMIT {current_batch_size} OFFSET {offset}"

            batch_df = self.conn.execute(data_query).df()

            for _, row in batch_df.iterrows():
                chunks = self.chunk_violation(row.to_dict())
                if chunks:
                    all_chunks.extend(chunks)

            offset += current_batch_size

            if show_progress:
                pbar.update(current_batch_size)

        if show_progress:
            pbar.close()

        return all_chunks

    def load_all_genduty_sample(
        self,
        limit: Optional[int] = None,
        show_progress: bool = True,
        batch_size: int = 100000,
    ) -> List[Dict[str, Any]]:
        """Load and chunk all chemical-industry general duty violation samples.

        Uses load_genduty_sample() filter criteria for data retrieval and
        chunk_general_duty_violation() for each row, with batched iteration.

        Args:
            limit: Maximum number of source rows to process
            show_progress: Whether to show a progress bar
            batch_size: Number of rows per batch

        Returns:
            List of chunk dictionaries
        """
        genduty_path = self.enforcement_dir / "violation_genduty_text.parquet"
        inspection_path = self.enforcement_dir / "inspection.parquet"

        if not genduty_path.exists():
            raise FileNotFoundError(f"General duty file not found: {genduty_path}")
        if not inspection_path.exists():
            raise FileNotFoundError(f"Inspection file not found: {inspection_path}")

        naics_filter = "(i.naics_code LIKE '325%' OR i.naics_code LIKE '324%')"
        keywords = [
            "chemical",
            "reactive",
            "explosion",
            "release",
            "toxic",
            "process safety",
        ]
        keyword_clauses = " OR ".join(
            f"LOWER(gd.text) LIKE '%{kw}%'" for kw in keywords
        )
        where_clause = f"{naics_filter} OR ({keyword_clauses})"

        count_query = f"""
            SELECT COUNT(*) as count
            FROM '{genduty_path}' gd
            LEFT JOIN '{inspection_path}' i
                ON gd.activity_nr = i.activity_nr
            WHERE {where_clause}
        """

        total_count = self.conn.execute(count_query).fetchone()[0]

        if limit:
            total_count = min(total_count, limit)

        all_chunks: List[Dict[str, Any]] = []
        offset = 0

        if show_progress:
            pbar = tqdm(total=total_count, desc="Processing general duty sample")

        while offset < total_count:
            current_batch_size = min(batch_size, total_count - offset)

            query = f"""
                SELECT gd.*, i.estab_name, i.site_city, i.site_state, i.naics_code
                FROM '{genduty_path}' gd
                LEFT JOIN '{inspection_path}' i
                    ON gd.activity_nr = i.activity_nr
                WHERE {where_clause}
                LIMIT {current_batch_size} OFFSET {offset}
            """

            batch_df = self.conn.execute(query).df()

            for _, row in batch_df.iterrows():
                chunks = self.chunk_general_duty_violation(row.to_dict())
                if chunks:
                    all_chunks.extend(chunks)

            offset += current_batch_size

            if show_progress:
                pbar.update(current_batch_size)

        if show_progress:
            pbar.close()

        return all_chunks

    def export_chunks_to_parquet(
        self, chunks: List[Dict[str, Any]], output_path: str
    ) -> None:
        """
        Export chunks to Parquet format.

        Args:
            chunks: List of chunk dictionaries
            output_path: Output Parquet file path
        """
        df = pd.DataFrame(chunks)
        df.to_parquet(output_path, index=False, compression="snappy")
        print(f"Exported {len(chunks)} enforcement chunks to {output_path}")
