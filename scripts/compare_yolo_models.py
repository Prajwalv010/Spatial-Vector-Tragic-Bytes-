"""Compare YOLOv8n, YOLOv8s, and YOLOv8m on TUNE set with corrected per-image ground truth.

Measures:
- Per-class precision, recall, F1 on walking-relevant classes:
  person, bicycle, car, motorcycle, bus, truck, bench, dog, backpack, handbag, suitcase, chair, dining table.
- Mean end-to-end FPS (including preprocessing, inference, postprocessing).
- Model parameter count / latency trade-off.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import time
from collections import defaultdict

import cv2
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
TUNE_DIR = ROOT / "tests" / "fixtures" / "real" / "tune"

WALKING_RELEVANT_CLASSES = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "bus",
    "truck",
    "bench",
    "dog",
    "backpack",
    "handbag",
    "suitcase",
    "chair",
    "dining table",
    "couch",
    "bottle",
    "laptop",
]

MODELS_TO_EVAL = ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"]


def eval_model(model_name: str, manifest: list) -> dict:
    print(f"\n[*] Evaluating {model_name} on TUNE set ({len(manifest)} images)...")
    model = YOLO(model_name)

    class_tp = defaultdict(int)
    class_fp = defaultdict(int)
    class_fn = defaultdict(int)

    latencies = []

    for item in manifest:
        img_path = TUNE_DIR / item["image_file"]
        if not img_path.exists():
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        exp_objs = set(item.get("expected_objects", []))

        t0 = time.perf_counter()
        res = model(img, conf=0.35, verbose=False)[0]
        dt = time.perf_counter() - t0
        latencies.append(dt)

        det_objs = set()
        if res.boxes is not None and len(res.boxes) > 0:
            for b in res.boxes:
                cname = res.names[int(b.cls[0])]
                det_objs.add(cname)

        # Update confusion metrics for walking-relevant classes
        for c in WALKING_RELEVANT_CLASSES:
            is_exp = c in exp_objs
            is_det = c in det_objs
            if is_exp and is_det:
                class_tp[c] += 1
            elif (not is_exp) and is_det:
                class_fp[c] += 1
            elif is_exp and (not is_det):
                class_fn[c] += 1

    mean_latency_ms = float(np.mean(latencies)) * 1000.0
    mean_fps = 1000.0 / mean_latency_ms if mean_latency_ms > 0 else 0.0

    # Calculate metrics
    per_class_metrics = {}
    total_tp = sum(class_tp.values())
    total_fp = sum(class_fp.values())
    total_fn = sum(class_fn.values())

    micro_prec = total_tp / max(1, total_tp + total_fp)
    micro_rec = total_tp / max(1, total_tp + total_fn)
    micro_f1 = (2 * micro_prec * micro_rec) / max(1e-6, micro_prec + micro_rec)

    for c in WALKING_RELEVANT_CLASSES:
        tp = class_tp[c]
        fp = class_fp[c]
        fn = class_fn[c]
        prec = tp / max(1, tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / max(1, tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / max(1e-6, prec + rec) if (prec + rec) > 0 else 0.0
        per_class_metrics[c] = {
            "tp": tp, "fp": fp, "fn": fn,
            "precision": prec, "recall": rec, "f1": f1
        }

    return {
        "model": model_name,
        "mean_latency_ms": mean_latency_ms,
        "fps": mean_fps,
        "micro_precision": micro_prec,
        "micro_recall": micro_rec,
        "micro_f1": micro_f1,
        "per_class": per_class_metrics,
    }


def main():
    print("=" * 72)
    print("  SpatialVector-HMI: YOLO Model Comparison on TUNE Ground Truth")
    print("=" * 72)

    manifest_path = TUNE_DIR / "manifest.json"
    manifest = json.loads(manifest_path.read_text())

    results = []
    for m in MODELS_TO_EVAL:
        results.append(eval_model(m, manifest))

    print("\n" + "=" * 72)
    print("  SUMMARY COMPARISON TABLE")
    print("=" * 72)
    print(f"{'Model':<12} | {'FPS':>6} | {'Latency':>10} | {'Micro Prec':>10} | {'Micro Rec':>10} | {'Micro F1':>8}")
    print("-" * 72)
    for r in results:
        print(f"{r['model']:<12} | {r['fps']:6.1f} | {r['mean_latency_ms']:8.2f}ms | {r['micro_precision']:10.1%} | {r['micro_recall']:10.1%} | {r['micro_f1']:8.3f}")

    print("\n" + "=" * 72)
    print("  PER-CLASS BREAKDOWN (Precision / Recall / F1)")
    print("=" * 72)
    for c in WALKING_RELEVANT_CLASSES:
        row_str = f"{c:<14}"
        for r in results:
            pcm = r["per_class"][c]
            row_str += f" | {r['model']}: P={pcm['precision']:.0%} R={pcm['recall']:.0%} F1={pcm['f1']:.2f}"
        print(row_str)

    # Save to JSON for report generation
    out_path = ROOT / "yolo_model_comparison_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\n[DONE] Saved detailed results to {out_path.name}")


if __name__ == '__main__':
    main()
