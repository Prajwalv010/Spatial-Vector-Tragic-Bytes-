"""Safety benchmark measuring transition latency on sudden hazard appearances.

Simulates sequential walking:
1. Frames 1..N: Clear walkable footpath
2. Frame N+1: Sudden appearance of drop-off / stairs / table-edge
Measures detection latency from appearance to first non-WALKABLE (UNKNOWN / BLOCKED) decision
for seg_frame_skip=0 vs seg_frame_skip=1.
"""
from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Dict, List, Tuple

import sys
root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))

import cv2
import numpy as np

from spatialvector.freespace.corridor_estimator import FreeSpaceEstimator, CorridorStatus


def run_frame_skip_safety_test() -> Dict[str, any]:
    root = Path(__file__).resolve().parent.parent
    tune_dir = root / "tests" / "fixtures" / "real" / "tune"

    # Find walkable clear footpath
    footpath_imgs = sorted(list(tune_dir.glob("outdoor_footpath*.jpg")))
    edge_imgs = sorted(list(tune_dir.glob("table_edge*.jpg")))
    stairs_imgs = sorted(list(tune_dir.glob("stairs_down*.jpg")))

    est_check = FreeSpaceEstimator(seg_frame_skip=0)
    clear_img = None
    clear_name = ""
    for p in footpath_imgs:
        im = cv2.imread(str(p))
        if im is None:
            continue
        est_check.reset()
        res = est_check.estimate(im)
        if res.centre == CorridorStatus.WALKABLE:
            clear_img = im
            clear_name = p.name
            break

    if clear_img is None:
        raise RuntimeError("No walkable footpath image found in TUNE set for priming.")

    test_hazards = [
        ("table_edge", edge_imgs[0]),
        ("stairs_down", stairs_imgs[0]),
    ]

    results = {}

    for h_name, h_path in test_hazards:
        h_img = cv2.imread(str(h_path))
        results[h_name] = {}

        for skip in [0, 1]:
            # Prime with 5 clear frames
            estimator = FreeSpaceEstimator(seg_frame_skip=skip, smoothing_window=1)
            for _ in range(5):
                estimator.estimate(clear_img)

            # Test latency when hazard appears on frame 6
            # For skip=1, frame 6 is an odd/even step relative to internal frame counter
            # We measure frame latency until decision is not WALKABLE
            latency_frames = 0
            decision = CorridorStatus.WALKABLE
            reasons = []

            # Step frame-by-frame with hazard image
            for step in range(1, 4):
                res = estimator.estimate(h_img)
                if res.centre != CorridorStatus.WALKABLE and latency_frames == 0:
                    latency_frames = step
                    decision = res.centre
                    reasons.append(res.reasons.get("centre", ""))
                    break

            # 63.3ms is measured mean latency per frame from live viewer benchmark
            latency_ms = latency_frames * 63.3

            results[h_name][f"skip_{skip}"] = {
                "hazard_file": h_path.name,
                "latency_frames": latency_frames,
                "latency_ms": latency_ms,
                "decision": decision.value,
                "reason": reasons[0] if reasons else "",
            }

    out_path = root / "frame_skip_safety_benchmark.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    return results


def main():
    print("Running Frame-Skip Safety Benchmark on TUNE sequences...")
    res = run_frame_skip_safety_test()
    for h_name, data in res.items():
        print(f"\nHazard: {h_name}")
        for skip_key, d in data.items():
            print(f"  {skip_key}: Latency = {d['latency_frames']} frame ({d['latency_ms']:.1f} ms) | Status = {d['decision']} | Reason = {d['reason']}")
    print("\nSaved benchmark results to frame_skip_safety_benchmark.json")


if __name__ == "__main__":
    main()
