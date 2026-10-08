"""Evaluation Harness for SpatialVector-HMI.

Evaluates M14 (Freespace), M13 (Ground Hazard Detector), and M02 (Object Detection)
on ground-truth fixture frames.

Reports:
- False walk-forward rate (must be 0 on edge/corner/dark object/covered lens/wall/stairs)
- Walkable precision and recall
- Pothole detection precision and recall
- Object detection precision and recall per class
- M14 + M13 + YOLO latency (FPS)
- Annotated failure images saved to eval_failures/
"""
import argparse
import csv
import json
from pathlib import Path
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from spatialvector.freespace import CorridorStatus, FreeSpaceEstimator, FreespaceResult
from spatialvector.hazards import GroundHazardDetector, HazardDetection

CORRIDORS = ("left", "centre", "right")


def load_fixture(png_path: Path) -> Tuple[Optional[np.ndarray], Optional[dict]]:
    label_path = png_path.with_suffix(".json")
    if not label_path.exists():
        return None, None
    img = cv2.imread(str(png_path))
    label = json.loads(label_path.read_text())
    return img, label


def run_yolo(yolo_model, img: np.ndarray, conf: float = 0.30):
    if yolo_model is None:
        return [], []
    try:
        results = yolo_model(img, conf=conf, verbose=False)
        boxes, classes = [], []
        for r in results:
            if r.boxes is None or len(r.boxes) == 0:
                continue
            for i in range(len(r.boxes)):
                xyxy = r.boxes.xyxy[i].cpu().numpy().tolist()
                cls_id = int(r.boxes.cls[i].cpu().numpy())
                cls_name = r.names.get(cls_id, str(cls_id))
                boxes.append(tuple(xyxy))
                classes.append(cls_name)
        return boxes, classes
    except Exception:
        return [], []


def run_m14(
    estimator: FreeSpaceEstimator,
    img: np.ndarray,
    yolo_bboxes: Optional[List[Tuple[float, float, float, float]]] = None,
) -> Tuple[FreespaceResult, float]:
    t0 = time.perf_counter()
    result = estimator.estimate(img, yolo_bboxes=yolo_bboxes)
    fps = 1.0 / max(1e-6, time.perf_counter() - t0)
    estimator.reset()
    return result, fps


def run_m13(
    detector: Optional[GroundHazardDetector],
    img: np.ndarray,
    yolo_bboxes: Optional[List[Tuple[float, float, float, float]]] = None,
    freespace_result: Optional[object] = None,
) -> Tuple[bool, List[HazardDetection]]:
    if detector is None:
        return False, []
    try:
        detections = detector.detect(img, yolo_bboxes=yolo_bboxes, freespace_result=freespace_result)
        detector.reset()
        return len(detections) > 0, detections
    except Exception:
        return False, []


def evaluate(
    fixtures_dir: Path,
    out_csv: Path,
    pothole_model: Optional[Path] = None,
    yolo_model_path: Optional[Path] = None,
    save_failures: bool = True,
) -> None:
    fixtures_dir = fixtures_dir.resolve()
    image_files = sorted([f for f in fixtures_dir.glob("*.*") if f.suffix.lower() in (".png", ".jpg", ".jpeg")])
    if not image_files:
        print(f"[WARN] No image fixtures found in {fixtures_dir}")
        return

    # Load YOLO detector
    yolo_model = None
    if yolo_model_path and yolo_model_path.exists():
        try:
            from ultralytics import YOLO
            yolo_model = YOLO(str(yolo_model_path))
            print(f"[YOLO] Loaded object detector: {yolo_model_path}")
        except Exception as e:
            print(f"[WARN] Could not load YOLO detector: {e}")

    # Initialise modules (single-frame evaluation)
    estimator = FreeSpaceEstimator(
        smoothing_window=1,
        ground_classes=[3, 6, 11, 28, 52],
        obstacle_classes=[0, 1, 2, 7, 8, 10, 12, 14, 15, 31, 32, 33, 53, 56, 59, 64, 80, 102, 121],
    )
    detector: Optional[GroundHazardDetector] = None
    if pothole_model and pothole_model.exists():
        try:
            detector = GroundHazardDetector(
                model_path=str(pothole_model),
                persistence_required=1,
                enable_heuristic_fallback=False,
                allow_heuristic=False,
                require_ground_confirmation=True,
                advisory_only=True,
            )
            print(f"[M13] Loaded ground hazard detector: {pothole_model}")
        except Exception as e:
            print(f"[WARN] Could not load hazard detector: {e}")
            detector = GroundHazardDetector(persistence_required=1, enable_heuristic_fallback=False, allow_heuristic=False, require_ground_confirmation=True, advisory_only=True)
    else:
        detector = GroundHazardDetector(persistence_required=1, enable_heuristic_fallback=False, allow_heuristic=False, require_ground_confirmation=True, advisory_only=True)

    rows: List[dict] = []
    totals: Dict[str, int] = {
        "walkable_tp": 0, "walkable_fp": 0, "walkable_fn": 0,
        "blocked_unk_tp": 0, "blocked_unk_fp": 0, "blocked_unk_fn": 0,
        "pothole_tp": 0, "pothole_fp": 0, "pothole_fn": 0,
        "pothole_negatives": 0,
        "obj_tp": 0, "obj_fp": 0, "obj_fn": 0,
        "hazard_frames": 0, "hazard_false_walkable": 0,
        "clear_frames": 0, "clear_walkable_tp": 0,
    }
    fps_list: List[float] = []
    pipeline_fps_list: List[float] = []
    failures_dir = Path("eval_failures")
    if save_failures:
        failures_dir.mkdir(parents=True, exist_ok=True)

    for img_path in image_files:
        img, label = load_fixture(img_path)
        if img is None or label is None:
            print(f"[SKIP] {img_path.name} — missing or unreadable")
            continue

        raw_walkable = label.get("expected_walkable", False)
        if isinstance(raw_walkable, dict):
            exp_walkable = raw_walkable
        elif isinstance(raw_walkable, bool):
            if raw_walkable:
                exp_walkable = {"left": "WALKABLE", "centre": "WALKABLE", "right": "WALKABLE"}
            else:
                default_stat = label.get("corridor_status", "UNKNOWN" if label.get("is_hazard_category") else "BLOCKED")
                exp_walkable = {"left": default_stat, "centre": default_stat, "right": default_stat}
        else:
            exp_walkable = {"left": "UNKNOWN", "centre": "UNKNOWN", "right": "UNKNOWN"}

        exp_pothole = ("pothole" in label.get("expected_hazards", [])) or bool(label.get("expected_potholes", False))
        exp_objects: list = label.get("expected_objects", [])
        cat_name = label.get("category", "")
        # Step 3 Prompt: "must be 0 on edge, ledge, stairs, wall, desk close-up, covered lens"
        is_m14_hazard_cat = cat_name in (
            "table_edge", "table_corner", "stairs_down", "stairs_up", "blank_wall", "desk_closeup", "covered_lens"
        )
        is_clear_cat = cat_name in ("indoor_corridor", "indoor_floor", "outdoor_footpath", "outdoor_road_clear")

        # Measure full pipeline latency
        t_pipe_0 = time.perf_counter()

        # Run YOLO object detection
        yolo_boxes, got_objects = run_yolo(yolo_model, img)

        # Run M14 with YOLO bounding boxes
        fs_result, fps = run_m14(estimator, img, yolo_bboxes=yolo_boxes)
        fps_list.append(fps)

        # Run M13 with YOLO bounding boxes and two-stage freespace ground confirmation
        got_pothole, hazard_dets = run_m13(detector, img, yolo_bboxes=yolo_boxes, freespace_result=fs_result)

        t_pipe_end = time.perf_counter()
        pipeline_fps = 1.0 / max(1e-6, t_pipe_end - t_pipe_0)
        pipeline_fps_list.append(pipeline_fps)

        fixture_has_failure = False

        # Hazard categories safety check: must be 0 false WALKABLE on edge, stairs, wall, desk, lens
        if is_m14_hazard_cat:
            totals["hazard_frames"] += 1
            if any(fs_result.get(c) == CorridorStatus.WALKABLE for c in CORRIDORS):
                totals["hazard_false_walkable"] += 1
                fixture_has_failure = True

        # Clear categories recall check
        if is_clear_cat:
            totals["clear_frames"] += 1
            if fs_result.get("centre") == CorridorStatus.WALKABLE:
                totals["clear_walkable_tp"] += 1

        # Per-corridor metrics
        for corridor in CORRIDORS:
            exp_str = exp_walkable.get(corridor, "UNKNOWN")
            got_status: CorridorStatus = fs_result.get(corridor)

            exp_w = (exp_str == "WALKABLE")
            got_w = (got_status == CorridorStatus.WALKABLE)
            exp_b = (exp_str in ("BLOCKED", "UNKNOWN"))
            got_b = (got_status in (CorridorStatus.BLOCKED, CorridorStatus.UNKNOWN))

            tp_w = int(exp_w and got_w)
            fp_w = int((not exp_w) and got_w)   # false walk-forward (safety-critical)
            fn_w = int(exp_w and (not got_w))

            totals["walkable_tp"] += tp_w
            totals["walkable_fp"] += fp_w
            totals["walkable_fn"] += fn_w

            tp_b = int(exp_b and got_b)
            fp_b = int((not exp_b) and got_b)
            fn_b = int(exp_b and (not got_b))
            totals["blocked_unk_tp"] += tp_b

            if fp_w == 1:
                fixture_has_failure = True

            rows.append({
                "fixture":            img_path.stem,
                "category":           cat_name,
                "corridor":           corridor,
                "expected_walkable":  exp_str,
                "got_walkable":       got_status.value,
                "conf":               f"{getattr(fs_result, corridor + '_conf', 0.0):.3f}",
                "reason":             fs_result.reasons.get(corridor, ""),
                "tp_w": tp_w, "fp_w": fp_w, "fn_w": fn_w,
                "tp_b": tp_b, "fp_b": fp_b, "fn_b": fn_b,
                "pothole_expected":   int(exp_pothole),
                "pothole_got":        int(got_pothole),
                "objects_expected":   ";".join(exp_objects),
                "objects_got":        ";".join(got_objects),
                "fps_m14":            f"{fps:.1f}",
            })

        # Pothole metrics
        p_tp = int(exp_pothole and got_pothole)
        p_fp = int((not exp_pothole) and got_pothole)
        p_fn = int(exp_pothole and (not got_pothole))
        totals["pothole_tp"] += p_tp
        totals["pothole_fp"] += p_fp
        totals["pothole_fn"] += p_fn
        if not exp_pothole:
            totals["pothole_negatives"] += 1

        if p_fp or p_fn:
            fixture_has_failure = True

        # Object metrics
        matched_exp = set()
        for go in got_objects:
            if go in exp_objects and go not in matched_exp:
                totals["obj_tp"] += 1
                matched_exp.add(go)
            else:
                totals["obj_fp"] += 1
        totals["obj_fn"] += len(exp_objects) - len(matched_exp)

        # Save annotated image on failure
        if save_failures and fixture_has_failure:
            ann = img.copy()
            h, w = ann.shape[:2]
            y0, y1 = int(h * 0.55), int(h * 0.95)
            cv2.rectangle(ann, (0, y0), (w, y1), (255, 255, 0), 1)
            stat_text = f"L:{fs_result.left.value} C:{fs_result.centre.value} R:{fs_result.right.value}"
            cv2.putText(ann, stat_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(ann, f"PH got:{got_pothole} exp:{exp_pothole}", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            for box in yolo_boxes:
                bx1, by1, bx2, by2 = map(int, box)
                cv2.rectangle(ann, (bx1, by1), (bx2, by2), (0, 255, 0), 2)
            out_fail_path = failures_dir / f"fail_{img_path.name}"
            cv2.imwrite(str(out_fail_path), ann)

    # ── Write CSV ─────────────────────────────────────────────────────────────
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    # Write sha256 checksum next to output CSV for provenance verification
    import hashlib
    sha256_hash = hashlib.sha256(out_csv.read_bytes()).hexdigest()
    sha256_file = out_csv.with_suffix(out_csv.suffix + ".sha256")
    sha256_file.write_text(sha256_hash.strip() + "\n", encoding="utf-8")
    print(f"  SHA-256 written to        : {sha256_file} ({sha256_hash[:16]}...)")

    # ── Summary calculation ───────────────────────────────────────────────────
    def safe_div(n, d):
        return n / d if d else 0.0

    wtp = totals["walkable_tp"]
    wfp = totals["walkable_fp"]
    wfn = totals["walkable_fn"]
    walkable_precision = safe_div(wtp, wtp + wfp)
    walkable_recall    = safe_div(wtp, wtp + wfn)

    ptp = totals["pothole_tp"]
    pfp = totals["pothole_fp"]
    pfn = totals["pothole_fn"]
    ph_precision = safe_div(ptp, ptp + pfp)
    ph_recall    = safe_div(ptp, ptp + pfn)

    otp = totals["obj_tp"]
    ofp = totals["obj_fp"]
    ofn = totals["obj_fn"]
    obj_precision = safe_div(otp, otp + ofp)
    obj_recall    = safe_div(otp, otp + ofn)

    mean_m14_fps = float(np.mean(fps_list)) if fps_list else 0.0
    mean_pipeline_fps = float(np.mean(pipeline_fps_list)) if pipeline_fps_list else 0.0

    hazard_total = totals["hazard_frames"]
    hazard_fw = totals["hazard_false_walkable"]
    hazard_fw_rate = safe_div(hazard_fw, hazard_total)

    clear_total = totals["clear_frames"]
    clear_rec = safe_div(totals["clear_walkable_tp"], clear_total)

    p_neg = totals["pothole_negatives"]
    ph_fpr_on_neg = safe_div(pfp, p_neg)

    print("\n" + "=" * 68)
    print("  SpatialVector-HMI Full Evaluation Report (Ground Truth HOLDOUT)")
    print("=" * 68)
    print(f"  Fixtures evaluated        : {len(image_files)}")
    print(f"  Corridor checks           : {len(rows)}")
    print()
    print("  [M14 WALKABLE GROUND]")
    print(f"  Overall WALKABLE precision: {walkable_precision:.2%}  (FP={wfp} corridor checks)")
    print(f"  Overall WALKABLE recall   : {walkable_recall:.2%}  (FN={wfn})")
    print(f"  Hazard Categories Evaluated: {hazard_total} frames (table edge/corner, stairs, wall, desk, lens)")
    print(f"  Hazard False WALKABLE Rate : {hazard_fw_rate:.2%} ({hazard_fw}/{hazard_total})  [TARGET: 0.00%]")
    print(f"  Clear Categories Recall   : {clear_rec:.2%} ({totals['clear_walkable_tp']}/{clear_total})  [TARGET: >= 80%]")
    print()
    print("  [M13 GROUND HAZARDS / POTHOLE]")
    print(f"  Pothole precision         : {ph_precision:.2%}  (TP={ptp}, FP={pfp})")
    print(f"  Pothole recall            : {ph_recall:.2%}  (FN={pfn})")
    print(f"  Pothole FP on Negatives   : {ph_fpr_on_neg:.2%} ({pfp}/{p_neg})  [TARGET: <= 5.0%]")
    print()
    if yolo_model:
        print("  [M02 OBJECT DETECTION (YOLO)]")
        print(f"  Object precision          : {obj_precision:.2%}  (TP={otp}, FP={ofp})")
        print(f"  Object recall             : {obj_recall:.2%}  (FN={ofn})")
        print()
    print("  [SYSTEM PERFORMANCE]")
    print(f"  M14 SegFormer Speed       : {mean_m14_fps:.1f} FPS ({1000.0/max(mean_m14_fps, 1):.1f} ms)")
    print(f"  Full Pipeline Speed       : {mean_pipeline_fps:.1f} FPS ({1000.0/max(mean_pipeline_fps, 1):.1f} ms)  [TARGET: >= 15 FPS]")
    print(f"  Results saved to          : {out_csv}")
    print("=" * 68)

    if hazard_fw > 0 or wfp > 0:
        print(f"\n[CRITICAL] {wfp} FALSE WALK-FORWARD(S) — M14 said WALKABLE when unsafe:")
        for row in rows:
            if row["fp_w"] == 1:
                print(f"    fixture={row['fixture']}  category={row.get('category')}  corridor={row['corridor']}  reason={row['reason']}")
    else:
        print("\n[SUCCESS] ZERO FALSE WALK-FORWARDS ON HAZARDS! (Safety invariant held 100%)")


def main():
    ap = argparse.ArgumentParser(description="SpatialVector pipeline evaluator")
    ap.add_argument("--fixtures", default="tests/fixtures/real", help="Directory of PNG+JSON fixtures")
    ap.add_argument("--out", default="eval_results.csv", help="Output CSV path")
    ap.add_argument("--pothole-model", default="pothole_yolov8.pt", help="Path to pothole .pt model")
    ap.add_argument("--yolo-model", default="yolov8n.pt", help="Path to COCO YOLO detector")
    args = ap.parse_args()

    evaluate(
        fixtures_dir=Path(args.fixtures),
        out_csv=Path(args.out),
        pothole_model=Path(args.pothole_model) if args.pothole_model else None,
        yolo_model_path=Path(args.yolo_model) if args.yolo_model else None,
    )


if __name__ == "__main__":
    main()
