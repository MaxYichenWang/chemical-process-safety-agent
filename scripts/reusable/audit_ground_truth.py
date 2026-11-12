"""Audit eval dataset ground_truth_chunks against ChromaDB collections.

Reports:
  (1) chunks that exist in dataset but not in any collection
  (2) near-matches (chunk_N suffix tolerance)
  (3) per-category alignment rate

Usage:
    python scripts/reusable/audit_ground_truth.py [--chromadb-path chromadb] [--dataset data/evaluation/eval_dataset_v1.json]
"""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

COLLECTIONS = ["osha_regulations", "osha_enf", "csb", "csb_recommendations"]


def _normalize_chunk_id(cid: str) -> str:
    """Strip trailing _chunk_N suffix for fuzzy matching."""
    return re.sub(r"_chunk_\d+$", "", cid)


def main():
    parser = argparse.ArgumentParser(description="Audit ground_truth_chunks vs ChromaDB")
    parser.add_argument("--chromadb-path", default="chromadb", help="Path to ChromaDB directory")
    parser.add_argument("--dataset", default="data/evaluation/eval_dataset_v1.json", help="Path to eval dataset")
    args = parser.parse_args()

    import chromadb

    client = chromadb.PersistentClient(path=args.chromadb_path)

    # Build set of all chunk IDs across collections
    all_chunk_ids: set = set()
    all_normalized: dict = {}  # normalized → [original_ids]
    collection_map: dict = {}  # chunk_id → collection_name
    collection_counts: dict = {}

    for coll_name in COLLECTIONS:
        try:
            coll = client.get_collection(coll_name)
            ids = coll.get(include=[])["ids"]
            collection_counts[coll_name] = len(ids)
            for cid in ids:
                all_chunk_ids.add(cid)
                collection_map[cid] = coll_name
                norm = _normalize_chunk_id(cid)
                all_normalized.setdefault(norm, []).append(cid)
        except Exception as e:
            print(f"WARNING: Could not read collection {coll_name}: {e}")
            collection_counts[coll_name] = -1

    print("ChromaDB Collections:")
    for name, count in sorted(collection_counts.items()):
        print(f"  {name}: {count:,} chunks")
    print(f"  Total unique IDs: {len(all_chunk_ids):,}")

    # Load dataset
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"ERROR: Dataset not found at {dataset_path}")
        return

    with open(dataset_path) as f:
        dataset = json.load(f)

    print(f"\nDataset: {len(dataset)} entries from {dataset_path}")

    # Audit each entry
    missing = []
    found = []
    near_misses = []
    category_stats = defaultdict(lambda: {"total": 0, "found": 0, "missing": 0, "fuzzy": 0})

    for i, entry in enumerate(dataset):
        cat = entry.get("category", "unknown")
        gt_chunks = entry.get("ground_truth_chunks", [])
        for chunk_id in gt_chunks:
            category_stats[cat]["total"] += 1

            # Exact match
            if chunk_id in all_chunk_ids:
                found.append((i, chunk_id, cat, collection_map.get(chunk_id, "?")))
                category_stats[cat]["found"] += 1
                continue

            # Fuzzy: strip _chunk_N suffix
            norm = _normalize_chunk_id(chunk_id)
            if norm in all_normalized:
                fuzzy_match = all_normalized[norm][0]
                near_misses.append((i, chunk_id, fuzzy_match, cat))
                category_stats[cat]["fuzzy"] += 1
                category_stats[cat]["found"] += 1
                continue

            # No match
            missing.append((i, chunk_id, cat, entry.get("query", "")[:80]))
            category_stats[cat]["missing"] += 1

    # Report
    print(f"\n{'='*70}")
    print("AUDIT RESULTS")
    print(f"{'='*70}")
    print(f"Exact matches:  {len(found)}")
    print(f"Fuzzy matches:  {len(near_misses)} (suffix tolerance)")
    print(f"MISSING:        {len(missing)} (no match in any collection)")
    total_gt = len(found) + len(near_misses) + len(missing)
    alignment = (len(found) + len(near_misses)) / total_gt * 100 if total_gt > 0 else 0
    print(f"Overall alignment: {alignment:.1f}%")

    print("\nPer-category alignment:")
    print(f"  {'Category':<30} {'Found':>6} {'Fuzzy':>6} {'Miss':>6} {'Total':>6} {'Rate':>6}")
    print(f"  {'-'*28}  {'-'*5}  {'-'*5}  {'-'*5}  {'-'*5}  {'-'*5}")
    for cat in sorted(category_stats):
        s = category_stats[cat]
        rate = s["found"] / s["total"] * 100 if s["total"] > 0 else 0
        print(f"  {cat:<30} {s['found']-s['fuzzy']:>6} {s['fuzzy']:>6} {s['missing']:>6} {s['total']:>6} {rate:>5.0f}%")

    if missing:
        print(f"\n{'='*70}")
        print(f"MISSING CHUNK IDs ({len(missing)}):")
        print(f"{'='*70}")
        for idx, cid, cat, query in missing:
            print(f"  [{idx:>3}] {cat}")
            print(f"        chunk: {cid}")
            print(f"        query: {query}")
            # Suggest closest match by prefix
            prefix = cid.split("_chunk_")[0][:20]
            candidates = [c for c in all_chunk_ids if c.startswith(prefix)]
            if candidates:
                print(f"        candidates: {candidates[:3]}")
            print()

    if near_misses:
        print(f"\n{'='*70}")
        print(f"NEAR MISSES — update dataset to use these IDs ({len(near_misses)}):")
        print(f"{'='*70}")
        for idx, old_id, new_id, cat in near_misses:
            print(f"  [{idx:>3}] {cat}")
            print(f"        {old_id}")
            print(f"     →  {new_id}")

    # Summary recommendation
    if missing:
        print(f"\n{'='*70}")
        print("ACTION REQUIRED:")
        print(f"  {len(missing)} ground_truth_chunks not found in ChromaDB.")
        print("  Options:")
        print("    1. Update eval_dataset_v1.json with correct chunk IDs")
        print("    2. Re-index missing corpora into ChromaDB")
        print("    3. Remove entries with invalid ground truth (last resort)")


if __name__ == "__main__":
    main()
