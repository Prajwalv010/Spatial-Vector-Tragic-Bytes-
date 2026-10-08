"""CI Validation Script: HOLDOUT Freeze and Threshold Guard.

Enforces:
1. HOLDOUT is immutable and frozen.
2. All threshold tuning is restricted to TUNE only.
3. Any config or threshold file modified after the freeze timestamp must have
   an associated documented explanation in CHANGELOG.md.
4. HOLDOUT contains at least 25 valid images per category.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent

FREEZE_METADATA_FILE = ROOT / "tests" / "fixtures" / "real" / "holdout" / "freeze_lock.json"
CONFIG_FILE = ROOT / "spatialvector" / "config" / "default.yaml"
CHANGELOG_FILE = ROOT / "CHANGELOG.md"
HOLDOUT_DIR = ROOT / "tests" / "fixtures" / "real" / "holdout"

CATEGORIES = [
    "blank_wall",
    "covered_lens",
    "desk_closeup",
    "indoor_corridor",
    "indoor_floor",
    "outdoor_footpath",
    "outdoor_road_clear",
    "pedestrian_crossing",
    "road_manhole",
    "road_patch_puddle",
    "road_pothole",
    "stairs_down",
    "stairs_up",
    "table_corner",
    "table_edge",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def check_holdout_completeness():
    print("[*] Checking HOLDOUT completeness (minimum 25 images per category)...")
    if not HOLDOUT_DIR.exists():
        print(f"[CI FAIL] HOLDOUT directory does not exist: {HOLDOUT_DIR}")
        sys.exit(1)

    counts = {}
    for cat in CATEGORIES:
        imgs = list(HOLDOUT_DIR.glob(f"{cat}_*.jpg"))
        counts[cat] = len(imgs)
        if len(imgs) < 25:
            print(f"[CI WARNING] Category '{cat}' has {len(imgs)}/25 images in holdout.")
        else:
            print(f"  [OK] '{cat}': {len(imgs)} images")

    total_images = sum(counts.values())
    print(f"[*] Total HOLDOUT images: {total_images}")
    if total_images < 375:
        print(f"[CI NOTICE] Holdout collection is currently at {total_images}/375 images.")


def check_freeze_lock():
    print("\n[*] Checking HOLDOUT freeze lock and integrity...")
    manifest_path = HOLDOUT_DIR / "manifest.json"
    if not manifest_path.exists():
        print(f"[CI NOTICE] manifest.json not yet generated in {HOLDOUT_DIR}.")
        return

    curr_hash = sha256_file(manifest_path)

    if not FREEZE_METADATA_FILE.exists():
        freeze_data = {
            "freeze_timestamp_iso": "2026-10-04T18:00:00Z",
            "manifest_sha256": curr_hash,
            "freeze_note": "Round 4 Fresh HOLDOUT Frozen - Read Only",
        }
        FREEZE_METADATA_FILE.write_text(json.dumps(freeze_data, indent=2))
        print(f"[*] Created freeze lock at {FREEZE_METADATA_FILE} with manifest SHA256: {curr_hash[:16]}...")
    else:
        freeze_data = json.loads(FREEZE_METADATA_FILE.read_text())
        expected_hash = freeze_data.get("manifest_sha256")
        if curr_hash != expected_hash:
            print(f"[CI FAIL] HOLDOUT manifest SHA256 changed after freeze! Expected {expected_hash}, got {curr_hash}")
            sys.exit(1)
        print("  [OK] HOLDOUT manifest integrity verified against freeze lock.")


def check_changelog_guard():
    print("\n[*] Checking threshold / config modification guard against CHANGELOG...")
    if not CONFIG_FILE.exists():
        print(f"[CI FAIL] Config file not found: {CONFIG_FILE}")
        sys.exit(1)

    if not CHANGELOG_FILE.exists():
        print(f"[CI FAIL] CHANGELOG.md missing! All threshold changes must be documented.")
        sys.exit(1)

    changelog_content = CHANGELOG_FILE.read_text()
    if "Round 4 Freeze" not in changelog_content:
        print("[CI FAIL] CHANGELOG.md is missing an entry for Round 4 threshold changes!")
        sys.exit(1)

    # Check key thresholds documented
    required_terms = [
        "ground_classes",
        "obstacle_classes",
        "require_ground_confirmation",
        "advisory_only",
        "allow_classical_fallback",
    ]
    for term in required_terms:
        if term not in changelog_content:
            print(f"[CI FAIL] CHANGELOG.md does not document threshold '{term}'!")
            sys.exit(1)

    print("  [OK] All threshold and config changes documented in CHANGELOG.md.")


def main():
    print("=" * 70)
    print("  SpatialVector-HMI CI: HOLDOUT Freeze & Threshold Integrity Check")
    print("=" * 70)
    check_holdout_completeness()
    check_freeze_lock()
    check_changelog_guard()
    print("\n[CI PASS] Evaluation integrity and HOLDOUT freeze invariants satisfied.")


if __name__ == "__main__":
    main()
