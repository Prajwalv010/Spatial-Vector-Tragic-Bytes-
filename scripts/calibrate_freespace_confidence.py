"""Calibrate FreeSpaceEstimator confidence scores on the TUNE split.

Evaluates FreeSpaceEstimator on all 360 TUNE images.
Collects predicted centre corridor confidence scores and checks whether
predicted status matches ground truth (is_correct: bool).
Computes reliability table in bins: [0.0-0.2), [0.2-0.4), [0.4-0.6), [0.6-0.8), [0.8-1.0].
Saves output to calibration_reliability_tune.json.
"""
import json
import time
from pathlib import Path
import sys

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))

import cv2
import numpy as np

from spatialvector.freespace.corridor_estimator import FreeSpaceEstimator, CorridorStatus

def main():
    root = Path(__file__).resolve().parent.parent
    tune_manifest = root / "tests" / "fixtures" / "real" / "tune" / "manifest.json"
    tune_img_dir = root / "tests" / "fixtures" / "real" / "tune"

    with open(tune_manifest, "r", encoding="utf-8") as f:
        items = json.load(f)

    estimator = FreeSpaceEstimator(seg_frame_skip=0)

    results = []
    print(f"Running Freespace confidence calibration on {len(items)} TUNE images...")

    for item in items:
        cat = item["category"]
        img_name = item["image_file"]
        img_path = tune_img_dir / img_name
        if not img_path.is_file():
            img_path = tune_img_dir / cat / img_name
        if not img_path.is_file():
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        # Single frame evaluation — reset history between independent images
        estimator.reset()
        res = estimator.estimate(img)

        pred_status = res.centre
        pred_conf = res.centre_conf
        expected_status_str = item.get("corridor_status", "UNKNOWN")
        expected_status = getattr(CorridorStatus, expected_status_str, CorridorStatus.UNKNOWN)

        # Ground truth correctness definition:
        # If expected is WALKABLE, prediction is correct if pred_status == WALKABLE
        # If expected is BLOCKED/UNKNOWN (hazard/non-walkable), prediction is correct if pred_status != WALKABLE
        if expected_status == CorridorStatus.WALKABLE:
            is_correct = (pred_status == CorridorStatus.WALKABLE)
        else:
            is_correct = (pred_status != CorridorStatus.WALKABLE)

        results.append({
            "id": item["id"],
            "category": cat,
            "pred_status": pred_status.value,
            "pred_conf": float(pred_conf),
            "expected_status": expected_status_str,
            "is_correct": bool(is_correct),
        })

    # Reliability diagram binning
    bins = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
    table = []

    for b_low, b_high in bins:
        bin_items = [r for r in results if (b_low <= r["pred_conf"] < b_high or (b_high == 1.0 and r["pred_conf"] == 1.0))]
        count = len(bin_items)
        if count > 0:
            avg_conf = float(np.mean([r["pred_conf"] for r in bin_items]))
            obs_acc = float(np.mean([1.0 if r["is_correct"] else 0.0 for r in bin_items]))
        else:
            avg_conf = (b_low + b_high) / 2.0
            obs_acc = 0.0

        table.append({
            "bin": f"[{b_low:.1f}, {b_high:.1f})",
            "count": count,
            "mean_confidence": round(avg_conf, 4),
            "observed_accuracy": round(obs_acc, 4),
            "gap": round(abs(avg_conf - obs_acc), 4) if count > 0 else 0.0,
        })

    out_file = root / "calibration_reliability_tune.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({"total_samples": len(results), "reliability_table": table, "per_sample": results}, f, indent=2)

    print("\nCalibration Reliability Table on TUNE (N = {}):".format(len(results)))
    print(f"{'Bin':<14} | {'Count':<6} | {'Mean Conf':<10} | {'Observed Acc':<12} | {'Gap':<8}")
    print("-" * 60)
    for row in table:
        print(f"{row['bin']:<14} | {row['count']:<6} | {row['mean_confidence']:<10.4f} | {row['observed_accuracy']:<12.4f} | {row['gap']:<8.4f}")

if __name__ == "__main__":
    main()
