"""Import per-image label corrections from CSV and update manifests with label_source='human'.

Usage:
    python scripts/import_label_corrections.py --csv path/to/corrections.csv --split holdout
    python scripts/import_label_corrections.py --csv path/to/corrections.csv --split tune

Input CSV Columns required:
    - image_id (or fixture / id)
    - objects (semicolon-separated class names, e.g. "person;car" or empty)
    - pothole (0 or 1, or yes/no)
    - left (WALKABLE / BLOCKED / UNKNOWN / UNSURE)
    - centre (WALKABLE / BLOCKED / UNKNOWN / UNSURE)
    - right (WALKABLE / BLOCKED / UNKNOWN / UNSURE)
    - verified (true / false)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Dict, List

import pandas as pd

VALID_CORRIDOR_STATUSES = {"WALKABLE", "BLOCKED", "UNKNOWN", "UNLABELLED", "UNSURE"}


def import_corrections(csv_path: Path, split: str = "holdout") -> int:
    if not csv_path.exists():
        raise FileNotFoundError(f"Corrections CSV not found: {csv_path}")

    df_corr = pd.read_csv(csv_path)

    # Normalize column names
    col_map = {}
    for c in df_corr.columns:
        norm = c.strip().lower()
        if norm in ("image_id", "fixture", "id", "image"):
            col_map[c] = "id"
        elif norm in ("objects", "expected_objects", "object_list"):
            col_map[c] = "objects"
        elif norm in ("pothole", "pothole_present", "potholes", "pothole yes/no"):
            col_map[c] = "pothole"
        elif norm in ("left", "left_corridor"):
            col_map[c] = "left"
        elif norm in ("centre", "center", "centre_corridor", "center_corridor"):
            col_map[c] = "centre"
        elif norm in ("right", "right_corridor"):
            col_map[c] = "right"
        elif norm in ("verified", "is_verified"):
            col_map[c] = "verified"

    df_corr = df_corr.rename(columns=col_map)
    required = ["id", "objects", "pothole", "left", "centre", "right", "verified"]
    missing = [c for c in required if c not in df_corr.columns]
    if missing:
        raise ValueError(f"Corrections CSV missing required columns: {missing}")

    repo_root = Path(__file__).resolve().parent.parent
    split_dir = repo_root / "tests" / "fixtures" / "real" / split.lower()
    manifest_path = split_dir / "manifest.json"

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found for split '{split}': {manifest_path}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest_items: List[dict] = json.load(f)

    manifest_lookup = {item["id"]: item for item in manifest_items}
    updated_count = 0

    for _, row in df_corr.iterrows():
        img_id = str(row["id"]).strip()
        if img_id not in manifest_lookup:
            print(f"[WARN] Image ID '{img_id}' not found in {split} manifest. Skipping.")
            continue

        item = manifest_lookup[img_id]

        # Parse corridor statuses
        l_stat = str(row["left"]).strip().upper()
        c_stat = str(row["centre"]).strip().upper()
        r_stat = str(row["right"]).strip().upper()

        for st_name, val in [("left", l_stat), ("centre", c_stat), ("right", r_stat)]:
            if val not in VALID_CORRIDOR_STATUSES:
                raise ValueError(f"Invalid status '{val}' for corridor '{st_name}' on {img_id}")

        # Map UNSURE to UNKNOWN
        l_mapped = "UNKNOWN" if l_stat == "UNSURE" else l_stat
        c_mapped = "UNKNOWN" if c_stat == "UNSURE" else c_stat
        r_mapped = "UNKNOWN" if r_stat == "UNSURE" else r_stat

        corridor_dict = {"left": l_mapped, "centre": c_mapped, "right": r_mapped}
        item["corridor_status"] = corridor_dict
        item["expected_walkable"] = corridor_dict

        # Parse objects
        obj_raw = str(row["objects"]) if pd.notna(row["objects"]) else ""
        obj_list = [x.strip() for x in obj_raw.split(";") if x.strip()]
        item["expected_objects"] = obj_list

        # Parse pothole
        ph_raw = str(row["pothole"]).strip().lower()
        has_pothole = ph_raw in ("1", "true", "yes", "y")
        hazards = [h for h in item.get("expected_hazards", []) if h != "pothole"]
        if has_pothole:
            hazards.append("pothole")
        item["expected_hazards"] = hazards

        # Mark human verified
        is_verified = str(row["verified"]).strip().lower() in ("1", "true", "yes", "y")
        item["verified"] = is_verified
        item["label_source"] = "human" if is_verified else "yolov8x_prefill"

        # Also update corresponding individual fixture json file if present
        fixture_json = split_dir / f"{img_id}.json"
        if fixture_json.exists():
            with open(fixture_json, "w", encoding="utf-8") as f_out:
                json.dump(item, f_out, indent=2)

        updated_count += 1

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_items, f, indent=2)

    print(f"Successfully imported {updated_count} human corrections into {split} manifest.")
    return updated_count


def main():
    parser = argparse.ArgumentParser(description="Import human label corrections into manifest")
    parser.add_argument("--csv", required=True, help="Path to corrections CSV")
    parser.add_argument("--split", default="holdout", choices=["holdout", "tune"], help="Target dataset split")
    args = parser.parse_args()

    import_corrections(Path(args.csv), split=args.split)


if __name__ == "__main__":
    main()
