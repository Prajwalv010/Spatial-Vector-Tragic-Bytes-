"""Audit M14 freespace cues on TUNE split only.

Investigates:
- Why clear images fail (which specific cue vetoes: validity gate, YOLO box,
  SegFormer class mix, gradient jump, span difference, continuity scan, dark blob).
- Ensures 0.00% False WALKABLE on hazard categories.
"""
from __future__ import annotations

import json
from pathlib import Path
from collections import Counter, defaultdict

import sys
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
from ultralytics import YOLO

from spatialvector.freespace import FreeSpaceEstimator, CorridorStatus
TUNE_DIR = ROOT / "tests" / "fixtures" / "real" / "tune"

CLEAR_CATEGORIES = {
    "indoor_corridor",
    "indoor_floor",
    "outdoor_footpath",
    "outdoor_road_clear",
}

HAZARD_CATEGORIES = {
    "table_edge",
    "table_corner",
    "stairs_down",
    "stairs_up",
    "blank_wall",
    "desk_closeup",
    "covered_lens",
}


def audit_cues():
    print("=" * 72)
    print("  M14 Cue Breakdown & Veto Audit on TUNE (tests/fixtures/real/tune)")
    print("=" * 72)

    estimator = FreeSpaceEstimator(smoothing_window=1)
    yolo = YOLO("yolov8n.pt")

    clear_fails = Counter()
    clear_passes = 0
    clear_total = 0

    hazard_fails_safe = 0
    hazard_false_walkables = 0
    hazard_total = 0

    veto_details = defaultdict(list)

    for json_file in sorted(TUNE_DIR.glob("*.json")):
        if json_file.name == "manifest.json":
            continue
        meta = json.loads(json_file.read_text())
        cat = meta.get("category", "")
        img_path = json_file.with_suffix(".jpg")
        if not img_path.exists():
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        # Get YOLO bboxes
        yres = yolo(img, verbose=False)[0]
        y_boxes = []
        if yres.boxes is not None and len(yres.boxes) > 0:
            for b in yres.boxes:
                y_boxes.append(tuple(b.xyxy[0].cpu().numpy().tolist()))

        res = estimator.estimate(img, yolo_bboxes=y_boxes)
        centre_stat = res.centre
        centre_reason = res.reasons.get("centre", "")

        is_clear = cat in CLEAR_CATEGORIES
        is_hazard = cat in HAZARD_CATEGORIES

        if is_clear:
            clear_total += 1
            if centre_stat == CorridorStatus.WALKABLE:
                clear_passes += 1
            else:
                # Classify the veto cue
                cue = "other"
                if "frame" in centre_reason or "blurry" in centre_reason or "dark" in centre_reason or "over-exposed" in centre_reason:
                    cue = "Cue A: Frame Validity"
                elif "YOLO" in centre_reason:
                    cue = "Cue E: YOLO Obstacle"
                elif "semantic" in centre_reason or "ground evidence" in centre_reason or "scene obstacle" in centre_reason:
                    cue = "SegFormer Class Mix"
                elif "brightness step" in centre_reason:
                    cue = "Cue B1: Gradient Jump"
                elif "span diff" in centre_reason or "surface mismatch" in centre_reason:
                    cue = "Cue B2: Span Difference"
                elif "ground surface break" in centre_reason or "continuity" in centre_reason:
                    cue = "Cue C: Continuity Scan"
                elif "unexplained dark" in centre_reason:
                    cue = "Cue D: Dark Blob"
                elif "uniform surface" in centre_reason:
                    cue = "Surface Uniformity (std too low)"
                elif "chaotic surface" in centre_reason:
                    cue = "Surface Variance (std too high)"

                clear_fails[cue] += 1
                veto_details[cue].append((meta["id"], cat, centre_reason))

        if is_hazard:
            hazard_total += 1
            # Check all corridors for false walk-forward
            if any(res.get(c) == CorridorStatus.WALKABLE for c in ("left", "centre", "right")):
                hazard_false_walkables += 1
                print(f"[HAZARD FALSE WALKABLE!] {meta['id']} ({cat}): {res.left}, {res.centre}, {res.right}")
            else:
                hazard_fails_safe += 1

    print("\n--- RESULTS ON TUNE ---")
    print(f"Hazard Frames Evaluated: {hazard_total}")
    print(f"Hazard Safe (UNKNOWN / BLOCKED): {hazard_fails_safe} / {hazard_total} ({hazard_fails_safe/max(1,hazard_total):.1%})")
    print(f"Hazard False WALKABLE: {hazard_false_walkables} / {hazard_total} ({hazard_false_walkables/max(1,hazard_total):.1%})")

    print(f"\nClear Frames Evaluated: {clear_total}")
    print(f"Clear Walkable Passed: {clear_passes} / {clear_total} ({clear_passes/max(1,clear_total):.1%})")
    print(f"Clear Vetoed: {clear_total - clear_passes} / {clear_total}")
    print("\nBreakdown of Veto Cues on Clear Images:")
    for cue, cnt in clear_fails.most_common():
        pct = (cnt / clear_total) * 100
        print(f"  - {cue:35s}: {cnt:3d} frames ({pct:5.1f}%)")

    print("\nSample Veto Reasons:")
    for cue, samples in veto_details.items():
        print(f"\n[{cue}] (showing up to 3):")
        for fid, cat, rsn in samples[:3]:
            print(f"   {fid} ({cat}): {rsn}")


if __name__ == '__main__':
    audit_cues()
