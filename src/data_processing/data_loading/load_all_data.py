#!/usr/bin/env python3
"""
Load all CSB and OSHA data and export to parquet files.

This script loads all available data from:
- CSB final reports
- CSB recommendations
- OSHA regulations
- OSHA violations
- OSHA accidents

And exports them to parquet files in data/loaded_chunks/
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_processing import (  # noqa: E402
    CSBDataLoader,
    OSHAEnforcementLoader,
    OSHARegulationsLoader,
    config,
)


def create_output_directory():
    """Create output directory for loaded chunks."""
    output_dir = config.CHUNKS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def load_chunks_from_parquet_if_exists(file_path: Path):
    """Load chunks from parquet file if it exists, otherwise return None.

    Args:
        file_path: Path to the parquet file

    Returns:
        DataFrame if file exists, None otherwise
    """
    if file_path.exists():
        print(f"   ⚠ File already exists: {file_path.name}")
        print("   → Loading existing chunks from parquet...")
        df = pd.read_parquet(file_path)
        print(f"   ✓ Loaded {len(df):,} chunks from existing file")
        return df
    return None


def load_all_csb_data(output_dir: Path, results: dict):
    """Load all CSB data (final reports and recommendations)."""
    print("\n" + "=" * 80)
    print("Loading CSB Data")
    print("=" * 80)

    csb_loader = CSBDataLoader()

    # Discover reports to get counts
    print("\n1. Discovering reports...")
    final_reports, recommendations = csb_loader._discover_reports()
    results["csb"] = {
        "total_final_reports": len(final_reports),
        "total_recommendations": len(recommendations),
    }
    print(f"   ✓ Found {len(final_reports)} final reports")
    print(f"   ✓ Found {len(recommendations)} recommendations")

    # Load all final reports
    print("\n2. Loading all final reports...")
    final_reports_path = output_dir / "csb_final_reports_chunks.parquet"

    # Check if file already exists
    existing_chunks = load_chunks_from_parquet_if_exists(final_reports_path)
    if existing_chunks is not None:
        csb_final_chunks = existing_chunks
        duration = 0
        results["csb"]["final_reports"] = {
            "chunks_generated": len(csb_final_chunks),
            "processing_time_seconds": duration,
            "avg_chunks_per_report": (
                len(csb_final_chunks) / results["csb"]["total_final_reports"]
                if results["csb"]["total_final_reports"] > 0
                else 0
            ),
            "loaded_from_existing": True,
        }
    else:
        start_time = datetime.now()
        csb_final_chunks = csb_loader.load_all_final_reports(
            limit=None, show_progress=True
        )
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        results["csb"]["final_reports"] = {
            "chunks_generated": len(csb_final_chunks),
            "processing_time_seconds": duration,
            "avg_chunks_per_report": (
                len(csb_final_chunks) / results["csb"]["total_final_reports"]
                if results["csb"]["total_final_reports"] > 0
                else 0
            ),
            "loaded_from_existing": False,
        }
        print(
            f"   ✓ Generated {len(csb_final_chunks)} chunks in {duration:.1f} seconds"
        )

        # Export final reports chunks
        print("\n3. Exporting final reports chunks to parquet...")
        csb_loader.export_chunks_to_parquet(csb_final_chunks, str(final_reports_path))

    results["csb"]["final_reports"]["output_file"] = str(final_reports_path)

    # Load all recommendations
    print("\n4. Loading all recommendations...")
    rec_path = output_dir / "csb_recommendations_chunks.parquet"

    # Check if file already exists
    existing_chunks = load_chunks_from_parquet_if_exists(rec_path)
    if existing_chunks is not None:
        csb_rec_chunks = existing_chunks
        duration = 0
        results["csb"]["recommendations"] = {
            "chunks_generated": len(csb_rec_chunks),
            "processing_time_seconds": duration,
            "avg_chunks_per_recommendation": (
                len(csb_rec_chunks) / results["csb"]["total_recommendations"]
                if results["csb"]["total_recommendations"] > 0
                else 0
            ),
            "loaded_from_existing": True,
        }
    else:
        start_time = datetime.now()
        csb_rec_chunks = csb_loader.load_all_recommendations(
            limit=None, show_progress=True
        )
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        results["csb"]["recommendations"] = {
            "chunks_generated": len(csb_rec_chunks),
            "processing_time_seconds": duration,
            "avg_chunks_per_recommendation": (
                len(csb_rec_chunks) / results["csb"]["total_recommendations"]
                if results["csb"]["total_recommendations"] > 0
                else 0
            ),
            "loaded_from_existing": False,
        }
        print(f"   ✓ Generated {len(csb_rec_chunks)} chunks in {duration:.1f} seconds")

        # Export recommendations chunks
        print("\n5. Exporting recommendations chunks to parquet...")
        csb_loader.export_chunks_to_parquet(csb_rec_chunks, str(rec_path))

    results["csb"]["recommendations"]["output_file"] = str(rec_path)

    return csb_final_chunks, csb_rec_chunks


def load_all_osha_regulations(output_dir: Path, results: dict):
    """Load all OSHA regulations."""
    print("\n" + "=" * 80)
    print("Loading OSHA Regulations")
    print("=" * 80)

    osha_reg_loader = OSHARegulationsLoader()

    # Get total count
    print("\n1. Counting total regulations...")
    all_regulations = osha_reg_loader.load_regulations(limit=None)
    results["osha_regulations"] = {
        "total_sections": len(all_regulations),
    }
    print(f"   ✓ Found {len(all_regulations)} regulation sections")

    # Load and chunk all regulations
    print("\n2. Loading and chunking all regulations...")
    reg_path = output_dir / "osha_regulations_chunks.parquet"

    # Check if file already exists
    existing_chunks = load_chunks_from_parquet_if_exists(reg_path)
    if existing_chunks is not None:
        osha_reg_chunks = existing_chunks
        duration = 0
        results["osha_regulations"]["chunks_generated"] = len(osha_reg_chunks)
        results["osha_regulations"]["processing_time_seconds"] = duration
        results["osha_regulations"]["avg_chunks_per_section"] = (
            len(osha_reg_chunks) / len(all_regulations)
            if len(all_regulations) > 0
            else 0
        )
        results["osha_regulations"]["loaded_from_existing"] = True
    else:
        start_time = datetime.now()
        osha_reg_chunks = osha_reg_loader.load_all_regulations(
            limit=None, show_progress=True
        )
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        results["osha_regulations"]["chunks_generated"] = len(osha_reg_chunks)
        results["osha_regulations"]["processing_time_seconds"] = duration
        results["osha_regulations"]["avg_chunks_per_section"] = (
            len(osha_reg_chunks) / len(all_regulations)
            if len(all_regulations) > 0
            else 0
        )
        results["osha_regulations"]["loaded_from_existing"] = False
        print(f"   ✓ Generated {len(osha_reg_chunks)} chunks in {duration:.1f} seconds")

        # Export regulations chunks
        print("\n3. Exporting regulations chunks to parquet...")
        osha_reg_loader.export_chunks_to_parquet(osha_reg_chunks, str(reg_path))

    results["osha_regulations"]["output_file"] = str(reg_path)

    return osha_reg_chunks


def load_all_osha_enforcement(
    output_dir: Path,
    results: dict,
    load_violations: bool = False,
    load_violations_sample: bool = True,
    load_genduty_sample: bool = True,
):
    """Load all OSHA enforcement data (violations, accidents, and general duty violations).

    Args:
        output_dir: Directory to save output files
        results: Dictionary to store results
        load_violations: If False, skip loading all violations (default: False)
        load_violations_sample: If True, load PSM-filtered violations sample (default: True)
        load_genduty_sample: If True, load chemical-industry general duty sample (default: True)
    """
    print("\n" + "=" * 80)
    print("Loading OSHA Enforcement Data")
    print("=" * 80)

    osha_enf_loader = OSHAEnforcementLoader()

    # Get total counts (using COUNT queries to avoid loading all data)
    print("\n1. Counting total violations, accidents, and general duty violations...")

    # Use DuckDB COUNT queries instead of loading full DataFrames
    violations_path = osha_enf_loader.enforcement_dir / "violations_text.parquet"
    accidents_path = osha_enf_loader.enforcement_dir / "accidents_text.parquet"
    genduty_path = osha_enf_loader.enforcement_dir / "violation_genduty_text.parquet"

    # Count violations (but don't load if load_violations=False)
    if load_violations:
        violations_count = osha_enf_loader.conn.execute(
            f"SELECT COUNT(*) as count FROM '{violations_path}'"
        ).fetchone()[0]
    else:
        violations_count = osha_enf_loader.conn.execute(
            f"SELECT COUNT(*) as count FROM '{violations_path}'"
        ).fetchone()[0]
        print("   ⚠ Skipping violations loading (load_violations=False)")

    accidents_count = osha_enf_loader.conn.execute(
        f"SELECT COUNT(*) as count FROM '{accidents_path}'"
    ).fetchone()[0]

    # Try to count general duty violations (may not exist in all environments)
    # Only check if load_violations is True
    all_genduty = False
    genduty_count = 0
    if load_violations:
        try:
            genduty_count = osha_enf_loader.conn.execute(
                f"SELECT COUNT(*) as count FROM '{genduty_path}'"
            ).fetchone()[0]
            all_genduty = True  # Flag to indicate file exists
        except Exception:
            print("   ⚠ General duty violations file not found, skipping...")
            all_genduty = False
            genduty_count = 0
    else:
        print("   ⚠ Skipping general duty violations (load_violations=False)")

    results["osha_enforcement"] = {
        "total_violations": violations_count,
        "total_accidents": accidents_count,
        "total_general_duty": genduty_count,
        "violations_loaded": load_violations,
    }
    print(
        f"   ✓ Found {violations_count:,} violations (loading: {'enabled' if load_violations else 'skipped'})"
    )
    print(f"   ✓ Found {accidents_count:,} accidents")
    if load_violations and genduty_count > 0:
        print(f"   ✓ Found {genduty_count:,} general duty violations")

    # Load and chunk all violations (optional)
    osha_viol_chunks = []
    if load_violations:
        print("\n2. Loading and chunking all violations...")
        viol_path = output_dir / "osha_violations_chunks.parquet"

        # Check if file already exists
        existing_chunks = load_chunks_from_parquet_if_exists(viol_path)
        if existing_chunks is not None:
            osha_viol_chunks = existing_chunks
            duration = 0
            results["osha_enforcement"]["violations"] = {
                "chunks_generated": len(osha_viol_chunks),
                "processing_time_seconds": duration,
                "loaded_from_existing": True,
            }
        else:
            start_time = datetime.now()
            osha_viol_chunks = osha_enf_loader.load_all_violations(
                limit=None, show_progress=True
            )
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            results["osha_enforcement"]["violations"] = {
                "chunks_generated": len(osha_viol_chunks),
                "processing_time_seconds": duration,
                "loaded_from_existing": False,
            }
            print(
                f"   ✓ Generated {len(osha_viol_chunks)} chunks in {duration:.1f} seconds"
            )

            # Export violations chunks
            print("\n3. Exporting violations chunks to parquet...")
            osha_enf_loader.export_chunks_to_parquet(osha_viol_chunks, str(viol_path))

        results["osha_enforcement"]["violations"]["output_file"] = str(viol_path)
    else:
        print("\n2. Skipping violations loading (load_violations=False)")
        results["osha_enforcement"]["violations"] = {
            "chunks_generated": 0,
            "processing_time_seconds": 0,
            "skipped": True,
        }

    # Load and chunk all accidents
    step_num = 4 if load_violations else 2
    print(f"\n{step_num}. Loading and chunking all accidents...")
    acc_path = output_dir / "osha_accidents_chunks.parquet"

    # Check if file already exists
    existing_chunks = load_chunks_from_parquet_if_exists(acc_path)
    if existing_chunks is not None:
        osha_acc_chunks = existing_chunks
        duration = 0
        results["osha_enforcement"]["accidents"] = {
            "chunks_generated": len(osha_acc_chunks),
            "processing_time_seconds": duration,
            "loaded_from_existing": True,
        }
    else:
        start_time = datetime.now()
        osha_acc_chunks = osha_enf_loader.load_all_accidents(
            limit=None, show_progress=True
        )
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        results["osha_enforcement"]["accidents"] = {
            "chunks_generated": len(osha_acc_chunks),
            "processing_time_seconds": duration,
            "loaded_from_existing": False,
        }
        print(f"   ✓ Generated {len(osha_acc_chunks)} chunks in {duration:.1f} seconds")

        # Export accidents chunks
        step_num = 5 if load_violations else 3
        print(f"\n{step_num}. Exporting accidents chunks to parquet...")
        osha_enf_loader.export_chunks_to_parquet(osha_acc_chunks, str(acc_path))

    results["osha_enforcement"]["accidents"]["output_file"] = str(acc_path)

    # Load and chunk all general duty violations (if available and load_violations=True)
    osha_genduty_chunks = []
    if load_violations and all_genduty and genduty_count > 0:
        step_num = 6 if load_violations else 4
        print(f"\n{step_num}. Loading and chunking all general duty violations...")
        genduty_path = output_dir / "osha_genduty_chunks.parquet"

        # Check if file already exists
        existing_chunks = load_chunks_from_parquet_if_exists(genduty_path)
        if existing_chunks is not None:
            osha_genduty_chunks = existing_chunks
            duration = 0
            results["osha_enforcement"]["general_duty"] = {
                "chunks_generated": len(osha_genduty_chunks),
                "processing_time_seconds": duration,
                "loaded_from_existing": True,
            }
        else:
            start_time = datetime.now()
            osha_genduty_chunks = osha_enf_loader.load_all_general_duty_violations(
                limit=None, show_progress=True
            )
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            results["osha_enforcement"]["general_duty"] = {
                "chunks_generated": len(osha_genduty_chunks),
                "processing_time_seconds": duration,
                "loaded_from_existing": False,
            }
            print(
                f"   ✓ Generated {len(osha_genduty_chunks)} chunks in {duration:.1f} seconds"
            )

            # Export general duty chunks
            step_num = 7 if load_violations else 5
            print(
                f"\n{step_num}. Exporting general duty violations chunks to parquet..."
            )
            osha_enf_loader.export_chunks_to_parquet(
                osha_genduty_chunks, str(genduty_path)
            )

        results["osha_enforcement"]["general_duty"]["output_file"] = str(genduty_path)

    # Load and chunk violations sample (PSM-filtered subset)
    osha_viol_sample_chunks = []
    if load_violations_sample:
        print("\n   Loading and chunking violations sample (PSM-filtered)...")
        viol_sample_path = output_dir / "osha_violations_sample_chunks.parquet"

        existing_chunks = load_chunks_from_parquet_if_exists(viol_sample_path)
        if existing_chunks is not None:
            osha_viol_sample_chunks = existing_chunks
            duration = 0
            results["osha_enforcement"]["violations_sample"] = {
                "chunks_generated": len(osha_viol_sample_chunks),
                "processing_time_seconds": duration,
                "loaded_from_existing": True,
            }
        else:
            start_time = datetime.now()
            osha_viol_sample_chunks = osha_enf_loader.load_all_violations_sample(
                limit=None, show_progress=True
            )
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            results["osha_enforcement"]["violations_sample"] = {
                "chunks_generated": len(osha_viol_sample_chunks),
                "processing_time_seconds": duration,
                "loaded_from_existing": False,
            }
            print(
                f"   ✓ Generated {len(osha_viol_sample_chunks)} violations sample chunks in {duration:.1f} seconds"
            )

            osha_enf_loader.export_chunks_to_parquet(
                osha_viol_sample_chunks, str(viol_sample_path)
            )

        results["osha_enforcement"]["violations_sample"]["output_file"] = str(
            viol_sample_path
        )
    else:
        print("\n   Skipping violations sample loading (load_violations_sample=False)")
        results["osha_enforcement"]["violations_sample"] = {
            "chunks_generated": 0,
            "processing_time_seconds": 0,
            "skipped": True,
        }

    # Load and chunk general duty sample (chemical-industry filtered)
    osha_genduty_sample_chunks = []
    if load_genduty_sample:
        print("\n   Loading and chunking general duty sample (chemical-industry)...")
        genduty_sample_path = output_dir / "osha_genduty_sample_chunks.parquet"

        existing_chunks = load_chunks_from_parquet_if_exists(genduty_sample_path)
        if existing_chunks is not None:
            osha_genduty_sample_chunks = existing_chunks
            duration = 0
            results["osha_enforcement"]["genduty_sample"] = {
                "chunks_generated": len(osha_genduty_sample_chunks),
                "processing_time_seconds": duration,
                "loaded_from_existing": True,
            }
        else:
            start_time = datetime.now()
            osha_genduty_sample_chunks = osha_enf_loader.load_all_genduty_sample(
                limit=None, show_progress=True
            )
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            results["osha_enforcement"]["genduty_sample"] = {
                "chunks_generated": len(osha_genduty_sample_chunks),
                "processing_time_seconds": duration,
                "loaded_from_existing": False,
            }
            print(
                f"   ✓ Generated {len(osha_genduty_sample_chunks)} general duty sample chunks in {duration:.1f} seconds"
            )

            osha_enf_loader.export_chunks_to_parquet(
                osha_genduty_sample_chunks, str(genduty_sample_path)
            )

        results["osha_enforcement"]["genduty_sample"]["output_file"] = str(
            genduty_sample_path
        )
    else:
        print("\n   Skipping general duty sample loading (load_genduty_sample=False)")
        results["osha_enforcement"]["genduty_sample"] = {
            "chunks_generated": 0,
            "processing_time_seconds": 0,
            "skipped": True,
        }

    return (
        osha_viol_chunks,
        osha_acc_chunks,
        osha_genduty_chunks,
        osha_viol_sample_chunks,
        osha_genduty_sample_chunks,
    )


def generate_results_document(results: dict, output_dir: Path):
    """Generate a comprehensive results document."""
    doc_path = config.CHUNKS_DIR / "DATA_LOADING_VALIDATION_REPORT.md"

    total_chunks = (
        results["csb"]["final_reports"]["chunks_generated"]
        + results["csb"]["recommendations"]["chunks_generated"]
        + results["osha_regulations"]["chunks_generated"]
        + results["osha_enforcement"]["violations"]["chunks_generated"]
        + results["osha_enforcement"]["accidents"]["chunks_generated"]
    )
    # Add general duty violations if they were loaded
    if "general_duty" in results["osha_enforcement"]:
        total_chunks += results["osha_enforcement"]["general_duty"]["chunks_generated"]
    # Add sample chunks
    if "violations_sample" in results["osha_enforcement"]:
        total_chunks += results["osha_enforcement"]["violations_sample"][
            "chunks_generated"
        ]
    if "genduty_sample" in results["osha_enforcement"]:
        total_chunks += results["osha_enforcement"]["genduty_sample"][
            "chunks_generated"
        ]

    total_time = (
        results["csb"]["final_reports"]["processing_time_seconds"]
        + results["csb"]["recommendations"]["processing_time_seconds"]
        + results["osha_regulations"]["processing_time_seconds"]
        + results["osha_enforcement"]["accidents"]["processing_time_seconds"]
    )
    # Add violations processing time if they were loaded
    if results["osha_enforcement"].get("violations_loaded", False):
        total_time += results["osha_enforcement"]["violations"][
            "processing_time_seconds"
        ]
    # Add general duty processing time if they were loaded
    if "general_duty" in results["osha_enforcement"]:
        total_time += results["osha_enforcement"]["general_duty"][
            "processing_time_seconds"
        ]
    # Add sample processing times
    if "violations_sample" in results["osha_enforcement"]:
        total_time += results["osha_enforcement"]["violations_sample"][
            "processing_time_seconds"
        ]
    if "genduty_sample" in results["osha_enforcement"]:
        total_time += results["osha_enforcement"]["genduty_sample"][
            "processing_time_seconds"
        ]

    # Calculate performance metrics (avoid division by zero)
    def _calc_speed(section):
        t = section.get("processing_time_seconds", 0)
        if section.get("skipped", False):
            return "N/A (skipped)"
        if t > 0:
            return f"{section['chunks_generated'] / t:.1f} chunks/second"
        return "N/A (loaded from existing file)"

    csb_final_speed = _calc_speed(results["csb"]["final_reports"])
    csb_rec_speed = _calc_speed(results["csb"]["recommendations"])
    osha_reg_speed = _calc_speed(results["osha_regulations"])
    osha_viol_speed = _calc_speed(results["osha_enforcement"]["violations"])
    osha_acc_speed = _calc_speed(results["osha_enforcement"]["accidents"])
    osha_viol_sample_speed = _calc_speed(
        results["osha_enforcement"].get("violations_sample", {"skipped": True})
    )
    osha_genduty_sample_speed = _calc_speed(
        results["osha_enforcement"].get("genduty_sample", {"skipped": True})
    )

    doc_content = f"""# Data Loading Results

**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary

This document summarizes the complete data loading process for all CSB and OSHA datasets.

### Overall Statistics

- **Total Chunks Generated**: {total_chunks:,}
- **Total Processing Time**: {total_time:.1f} seconds ({total_time/60:.1f} minutes)
- **Output Directory**: `{output_dir}`

## CSB Data

### Final Reports

- **Total Reports**: {results['csb']['total_final_reports']:,}
- **Chunks Generated**: {results['csb']['final_reports']['chunks_generated']:,}
- **Average Chunks per Report**: {results['csb']['final_reports']['avg_chunks_per_report']:.2f}
- **Processing Time**: {results['csb']['final_reports']['processing_time_seconds']:.1f} seconds ({results['csb']['final_reports']['processing_time_seconds']/60:.1f} minutes)
- **Output File**: `{results['csb']['final_reports']['output_file']}`

### Recommendations

- **Total Recommendations**: {results['csb']['total_recommendations']:,}
- **Chunks Generated**: {results['csb']['recommendations']['chunks_generated']:,}
- **Average Chunks per Recommendation**: {results['csb']['recommendations']['avg_chunks_per_recommendation']:.2f}
- **Processing Time**: {results['csb']['recommendations']['processing_time_seconds']:.1f} seconds ({results['csb']['recommendations']['processing_time_seconds']/60:.1f} minutes)
- **Output File**: `{results['csb']['recommendations']['output_file']}`

### CSB Summary

- **Total CSB Chunks**: {results['csb']['final_reports']['chunks_generated'] + results['csb']['recommendations']['chunks_generated']:,}
- **Total CSB Processing Time**: {results['csb']['final_reports']['processing_time_seconds'] + results['csb']['recommendations']['processing_time_seconds']:.1f} seconds

## OSHA Regulations

- **Total Regulation Sections**: {results['osha_regulations']['total_sections']:,}
- **Chunks Generated**: {results['osha_regulations']['chunks_generated']:,}
- **Average Chunks per Section**: {results['osha_regulations']['avg_chunks_per_section']:.2f}
- **Processing Time**: {results['osha_regulations']['processing_time_seconds']:.1f} seconds ({results['osha_regulations']['processing_time_seconds']/60:.1f} minutes)
- **Output File**: `{results['osha_regulations']['output_file']}`

## OSHA Enforcement

### Violations

- **Total Violations**: {results['osha_enforcement']['total_violations']:,}
{"- **Chunks Generated**: " + f"{results['osha_enforcement']['violations']['chunks_generated']:,}" + "\n" if results['osha_enforcement'].get('violations_loaded', False) else "- **Status**: Skipped (load_violations=False)\n"}
{"- **Processing Time**: " + f"{results['osha_enforcement']['violations']['processing_time_seconds']:.1f} seconds ({results['osha_enforcement']['violations']['processing_time_seconds']/60:.1f} minutes)" + "\n" if results['osha_enforcement'].get('violations_loaded', False) else ""}
{"- **Output File**: `" + results['osha_enforcement']['violations']['output_file'] + "`" + "\n" if results['osha_enforcement'].get('violations_loaded', False) and 'output_file' in results['osha_enforcement']['violations'] else ""}

### Accidents

- **Total Accidents**: {results['osha_enforcement']['total_accidents']:,}
- **Chunks Generated**: {results['osha_enforcement']['accidents']['chunks_generated']:,}
- **Processing Time**: {results['osha_enforcement']['accidents']['processing_time_seconds']:.1f} seconds ({results['osha_enforcement']['accidents']['processing_time_seconds']/60:.1f} minutes)
- **Output File**: `{results['osha_enforcement']['accidents']['output_file']}`

{f"### General Duty Violations\n\n- **Total General Duty Violations**: {results['osha_enforcement']['total_general_duty']:,}\n- **Chunks Generated**: {results['osha_enforcement']['general_duty']['chunks_generated']:,}\n- **Processing Time**: {results['osha_enforcement']['general_duty']['processing_time_seconds']:.1f} seconds ({results['osha_enforcement']['general_duty']['processing_time_seconds']/60:.1f} minutes)\n- **Output File**: `{results['osha_enforcement']['general_duty']['output_file']}`\n" if "general_duty" in results['osha_enforcement'] else ""}

{"### Violations Sample (PSM-Filtered)\n\n- **Chunks Generated**: " + f"{results['osha_enforcement']['violations_sample']['chunks_generated']:,}" + "\n- **Processing Time**: " + f"{results['osha_enforcement']['violations_sample']['processing_time_seconds']:.1f} seconds" + "\n- **Output File**: `" + str(results['osha_enforcement']['violations_sample'].get('output_file', 'N/A')) + "`\n" if "violations_sample" in results["osha_enforcement"] and not results["osha_enforcement"]["violations_sample"].get("skipped", False) else "### Violations Sample (PSM-Filtered)\n\n- **Status**: Skipped\n" if "violations_sample" in results["osha_enforcement"] else ""}

{"### General Duty Sample (Chemical-Industry)\n\n- **Chunks Generated**: " + f"{results['osha_enforcement']['genduty_sample']['chunks_generated']:,}" + "\n- **Processing Time**: " + f"{results['osha_enforcement']['genduty_sample']['processing_time_seconds']:.1f} seconds" + "\n- **Output File**: `" + str(results['osha_enforcement']['genduty_sample'].get('output_file', 'N/A')) + "`\n" if "genduty_sample" in results["osha_enforcement"] and not results["osha_enforcement"]["genduty_sample"].get("skipped", False) else "### General Duty Sample (Chemical-Industry)\n\n- **Status**: Skipped\n" if "genduty_sample" in results["osha_enforcement"] else ""}

### OSHA Enforcement Summary

- **Total Enforcement Chunks**: {(results['osha_enforcement']['violations']['chunks_generated'] if results['osha_enforcement'].get('violations_loaded', False) else 0) + results['osha_enforcement']['accidents']['chunks_generated'] + (results['osha_enforcement']['general_duty']['chunks_generated'] if "general_duty" in results['osha_enforcement'] else 0) + (results['osha_enforcement']['violations_sample']['chunks_generated'] if "violations_sample" in results['osha_enforcement'] else 0) + (results['osha_enforcement']['genduty_sample']['chunks_generated'] if "genduty_sample" in results['osha_enforcement'] else 0):,}
- **Total Enforcement Processing Time**: {(results['osha_enforcement']['violations']['processing_time_seconds'] if results['osha_enforcement'].get('violations_loaded', False) else 0) + results['osha_enforcement']['accidents']['processing_time_seconds'] + (results['osha_enforcement']['general_duty']['processing_time_seconds'] if "general_duty" in results['osha_enforcement'] else 0) + (results['osha_enforcement']['violations_sample']['processing_time_seconds'] if "violations_sample" in results['osha_enforcement'] else 0) + (results['osha_enforcement']['genduty_sample']['processing_time_seconds'] if "genduty_sample" in results['osha_enforcement'] else 0):.1f} seconds

## File Structure

All output files are stored in: `{output_dir}`

```
{output_dir.name}/
├── csb_final_reports_chunks.parquet
├── csb_recommendations_chunks.parquet
├── osha_regulations_chunks.parquet
├── osha_violations_chunks.parquet
├── osha_accidents_chunks.parquet
{"├── osha_genduty_chunks.parquet" if "general_duty" in results['osha_enforcement'] else ""}
{"├── osha_violations_sample_chunks.parquet" if "violations_sample" in results['osha_enforcement'] and not results['osha_enforcement']['violations_sample'].get('skipped', False) else ""}
{"└── osha_genduty_sample_chunks.parquet" if "genduty_sample" in results['osha_enforcement'] and not results['osha_enforcement']['genduty_sample'].get('skipped', False) else ""}
```

## Data Schema

All chunks follow the standardized schema defined in `src/data_processing/utilities/utils.py`:

- `chunk_id`: Unique identifier
- `corpus`: High-level corpus (csb, osha_reg, osha_enf)
- `subcorpus`: Sub-corpus identifier
- `doc_type`: Document type
- `text`: Chunk text content
- `token_count`: Number of tokens
- `hash`: SHA256 hash for deduplication
- `source_path`: Path to source file
- Additional corpus-specific metadata fields

## Usage

To load the chunks back:

```python
import pandas as pd

# Load CSB final reports
csb_chunks = pd.read_parquet('data/loaded_chunks/csb_final_reports_chunks.parquet')

# Load OSHA regulations
osha_reg_chunks = pd.read_parquet('data/loaded_chunks/osha_regulations_chunks.parquet')

# Load OSHA violations sample (PSM-filtered)
osha_viol_sample = pd.read_parquet('data/loaded_chunks/osha_violations_sample_chunks.parquet')

# Load OSHA general duty sample (chemical-industry)
osha_genduty_sample = pd.read_parquet('data/loaded_chunks/osha_genduty_sample_chunks.parquet')
```

## Performance Metrics

### Processing Speed

- **CSB Final Reports**: {csb_final_speed}
- **CSB Recommendations**: {csb_rec_speed}
- **OSHA Regulations**: {osha_reg_speed}
- **OSHA Violations**: {osha_viol_speed}
- **OSHA Accidents**: {osha_acc_speed}
- **OSHA Violations Sample**: {osha_viol_sample_speed}
- **OSHA General Duty Sample**: {osha_genduty_sample_speed}

## Notes

- All chunks are exported in Parquet format for efficient storage and retrieval
- Chunks include rich metadata for filtering and provenance tracking
- Processing times may vary based on system resources
- Data loading preserves source metadata and validates the shared chunk schema

## Next Steps

1. Generate embeddings for all chunks using `EmbeddingsGenerator`
2. Store in ChromaDB using `ChromaDBManager`
3. Integrate with RAG pipeline for querying

---

*This document was automatically generated by `load_all_data.py`*
"""

    with open(doc_path, "w") as f:
        f.write(doc_content)

    print(f"\n✓ Results document generated: {doc_path}")

    # Also save JSON summary
    json_path = config.CHUNKS_DIR / "csb_osha_data_loading_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"✓ JSON summary saved: {json_path}")


def main():
    """Main execution function."""
    print("=" * 80)
    print("Loading All CSB and OSHA Data")
    print("=" * 80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    results = {}
    overall_start = datetime.now()

    # Create output directory
    output_dir = create_output_directory()
    print(f"\nOutput directory: {output_dir}")

    try:
        # Load CSB data
        csb_final_chunks, csb_rec_chunks = load_all_csb_data(output_dir, results)

        # Load OSHA regulations
        osha_reg_chunks = load_all_osha_regulations(output_dir, results)

        # Load OSHA enforcement (skip full violations, load filtered samples)
        (
            osha_viol_chunks,
            osha_acc_chunks,
            osha_genduty_chunks,
            osha_viol_sample_chunks,
            osha_genduty_sample_chunks,
        ) = load_all_osha_enforcement(
            output_dir,
            results,
            load_violations=False,
            load_violations_sample=True,
            load_genduty_sample=True,
        )

        # Calculate totals
        overall_end = datetime.now()
        total_duration = (overall_end - overall_start).total_seconds()
        results["overall"] = {
            "total_processing_time_seconds": total_duration,
            "start_time": overall_start.isoformat(),
            "end_time": overall_end.isoformat(),
        }

        # Generate results document
        print("\n" + "=" * 80)
        print("Generating Results Document")
        print("=" * 80)
        generate_results_document(results, output_dir)

        # Final summary
        total_chunks = (
            len(csb_final_chunks)
            + len(csb_rec_chunks)
            + len(osha_reg_chunks)
            + len(osha_viol_chunks)
            + len(osha_acc_chunks)
            + len(osha_genduty_chunks)
            + len(osha_viol_sample_chunks)
            + len(osha_genduty_sample_chunks)
        )

        print("\n" + "=" * 80)
        print("Loading Complete!")
        print("=" * 80)
        print(f"Total chunks generated: {total_chunks:,}")
        print(
            f"Total processing time: {total_duration:.1f} seconds ({total_duration/60:.1f} minutes)"
        )
        print(f"Output directory: {output_dir}")
        print("Results document: data/loaded_chunks/DATA_LOADING_VALIDATION_REPORT.md")
        print("=" * 80)

        return 0

    except Exception as e:
        print(f"\n✗ Error during loading: {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
