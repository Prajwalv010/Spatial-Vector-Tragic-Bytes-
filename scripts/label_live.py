"""OpenCV interactive keyboard labelling tool for human ground-truth annotation.

Usage:
    python scripts/label_live.py --source tests/fixtures/real/holdout --out human_labels.csv
    python scripts/label_live.py --source sessions --out human_labels.csv

Controls:
    [1] Cycle Left Corridor:   SAFE (WALKABLE) -> UNSAFE (BLOCKED) -> UNSURE
    [2] Cycle Centre Corridor: SAFE (WALKABLE) -> UNSAFE (BLOCKED) -> UNSURE
    [3] Cycle Right Corridor:  SAFE (WALKABLE) -> UNSAFE (BLOCKED) -> UNSURE
    [P] Toggle Pothole:        NO -> YES
    [O] Edit Objects:          Input comma/semicolon-separated objects in console
    [S] / [Enter] Save & Next: Save current human annotation with verified=true
    [N] Skip:                  Skip frame without saving
    [Q] / [ESC] Quit:          Exit labelling tool

Honesty Rule:
    The agent must not auto-label walkability. The operator marks each corridor.
    Output CSV format is 100% compatible with scripts/import_label_corrections.py:
    image_id,objects,pothole,left,centre,right,verified
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# Ensure repo root on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from spatialvector.freespace import FreeSpaceEstimator, CorridorStatus
from spatialvector.hazards import GroundHazardDetector

CORRIDOR_STATES = ["SAFE", "UNSAFE", "UNSURE"]
STATE_TO_WALKABLE = {
    "SAFE": "WALKABLE",
    "UNSAFE": "BLOCKED",
    "UNSURE": "UNKNOWN",
}


def find_frames(source_dir: Path) -> List[Path]:
    """Find all image frames under source_dir (or subdirectories if sessions/)."""
    exts = {".png", ".jpg", ".jpeg"}
    frames = [p for p in source_dir.rglob("*.*") if p.suffix.lower() in exts]
    # Filter out annotated failure images or temp files
    frames = [p for p in frames if not p.name.startswith("fail_") and not p.name.startswith(".")]
    return sorted(frames)


def label_frames(
    source_dir: Path,
    out_csv: Path,
    pothole_model: Optional[Path] = None,
    yolo_model_path: Optional[Path] = None,
    headless: bool = False,
) -> None:
    frames = find_frames(source_dir)
    if not frames:
        print(f"[!] No frames found in {source_dir}")
        return

    print(f"[*] Found {len(frames)} frames in {source_dir}")

    # Check for existing labels to allow resuming
    existing_labels: Dict[str, dict] = {}
    if out_csv.exists():
        try:
            with open(out_csv, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    existing_labels[row.get("image_id", row.get("id", ""))] = row
            print(f"[*] Loaded {len(existing_labels)} existing annotations from {out_csv}")
        except Exception as e:
            print(f"[!] Could not read existing {out_csv}: {e}")

    # Load model inference helpers for displaying model's CURRENT decision (advisory only)
    estimator = FreeSpaceEstimator(smoothing_window=1)
    detector = None
    if pothole_model and pothole_model.exists():
        try:
            detector = GroundHazardDetector(model_path=str(pothole_model), advisory_only=True)
        except Exception:
            pass

    yolo_model = None
    if yolo_model_path and yolo_model_path.exists():
        try:
            from ultralytics import YOLO
            yolo_model = YOLO(str(yolo_model_path))
        except Exception:
            pass

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["image_id", "objects", "pothole", "left", "centre", "right", "verified"]

    if headless:
        print("[!] Headless mode requested: displaying frame summary only.")
        return

    cv2.namedWindow("SpatialVector Human Labelling Tool", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("SpatialVector Human Labelling Tool", 1024, 680)

    idx = 0
    while idx < len(frames):
        frame_path = frames[idx]
        image_id = frame_path.stem

        img = cv2.imread(str(frame_path))
        if img is None:
            print(f"[SKIP] Corrupt frame: {frame_path}")
            idx += 1
            continue

        # Run model to get current decision for operator awareness
        model_fs = estimator.estimate(img)
        model_ph = False
        if detector:
            try:
                dets = detector.detect(img)
                model_ph = len(dets) > 0
            except Exception:
                pass

        # Initial operator state: load existing if available, else blank/UNSURE
        if image_id in existing_labels:
            prev = existing_labels[image_id]
            human_left = "SAFE" if prev.get("left") == "WALKABLE" else ("UNSAFE" if prev.get("left") == "BLOCKED" else "UNSURE")
            human_centre = "SAFE" if prev.get("centre") == "WALKABLE" else ("UNSAFE" if prev.get("centre") == "BLOCKED" else "UNSURE")
            human_right = "SAFE" if prev.get("right") == "WALKABLE" else ("UNSAFE" if prev.get("right") == "BLOCKED" else "UNSURE")
            human_ph = str(prev.get("pothole", "0")).lower() in ("1", "yes", "true")
            human_objects = prev.get("objects", "")
        else:
            human_left = "UNSURE"
            human_centre = "UNSURE"
            human_right = "UNSURE"
            human_ph = False
            human_objects = ""

        # Loop on the current frame until Save [S] or Skip [N] or Quit [Q]
        while True:
            display = img.copy()
            dh, dw = display.shape[:2]

            # Header info
            cv2.rectangle(display, (0, 0), (dw, 64), (18, 22, 28), -1)
            title = f"Frame {idx+1}/{len(frames)}: {frame_path.name}"
            cv2.putText(display, title, (14, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 210, 255), 2, cv2.LINE_AA)

            # Model decision line
            m_text = f"Model Decision: L:{model_fs.left.value} C:{model_fs.centre.value} R:{model_fs.right.value} | Pothole:{int(model_ph)}"
            cv2.putText(display, m_text, (14, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (160, 175, 190), 1, cv2.LINE_AA)

            # Corridor split guides
            cx1 = int(dw * 0.38)
            cx2 = int(dw * 0.62)
            cv2.line(display, (cx1, 64), (cx1, dh - 80), (70, 80, 95), 1, cv2.LINE_AA)
            cv2.line(display, (cx2, 64), (cx2, dh - 80), (70, 80, 95), 1, cv2.LINE_AA)

            # Bottom human annotation panel
            cv2.rectangle(display, (0, dh - 80), (dw, dh), (14, 18, 24), -1)
            cv2.line(display, (0, dh - 80), (dw, dh - 80), (0, 180, 240), 1)

            def state_color(st):
                if st == "SAFE":
                    return (80, 230, 130)
                if st == "UNSAFE":
                    return (60, 60, 255)
                return (0, 195, 255)

            # Human state text
            cv2.putText(display, f"[1] Left: {human_left}", (20, dh - 48), cv2.FONT_HERSHEY_SIMPLEX, 0.50, state_color(human_left), 2, cv2.LINE_AA)
            cv2.putText(display, f"[2] Centre: {human_centre}", (cx1 + 20, dh - 48), cv2.FONT_HERSHEY_SIMPLEX, 0.50, state_color(human_centre), 2, cv2.LINE_AA)
            cv2.putText(display, f"[3] Right: {human_right}", (cx2 + 20, dh - 48), cv2.FONT_HERSHEY_SIMPLEX, 0.50, state_color(human_right), 2, cv2.LINE_AA)

            ph_color = (0, 210, 255) if human_ph else (160, 170, 180)
            cv2.putText(display, f"[P] Pothole: {'YES' if human_ph else 'NO'}", (20, dh - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, ph_color, 1, cv2.LINE_AA)
            cv2.putText(display, f"[O] Objects: {human_objects or 'none'}", (cx1 + 20, dh - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 205, 215), 1, cv2.LINE_AA)
            cv2.putText(display, "[S] Save | [N] Skip | [Q] Quit", (dw - 240, dh - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 210, 255), 1, cv2.LINE_AA)

            cv2.imshow("SpatialVector Human Labelling Tool", display)
            key = cv2.waitKey(0) & 0xFF

            if key in (ord("q"), 27):  # Q or ESC
                print("[*] Labelling session exited by operator.")
                cv2.destroyAllWindows()
                return

            elif key == ord("1"):
                curr_idx = CORRIDOR_STATES.index(human_left)
                human_left = CORRIDOR_STATES[(curr_idx + 1) % len(CORRIDOR_STATES)]

            elif key == ord("2"):
                curr_idx = CORRIDOR_STATES.index(human_centre)
                human_centre = CORRIDOR_STATES[(curr_idx + 1) % len(CORRIDOR_STATES)]

            elif key == ord("3"):
                curr_idx = CORRIDOR_STATES.index(human_right)
                human_right = CORRIDOR_STATES[(curr_idx + 1) % len(CORRIDOR_STATES)]

            elif key in (ord("p"), ord("P")):
                human_ph = not human_ph

            elif key in (ord("o"), ord("O")):
                print("\nEnter objects visible (comma or semicolon separated, e.g. 'person,chair'): ", end="", flush=True)
                obj_input = sys.stdin.readline().strip()
                human_objects = ";".join(x.strip() for x in obj_input.replace(",", ";").split(";") if x.strip())
                print(f"Set objects: {human_objects}")

            elif key in (ord("s"), ord("S"), 13):  # S or Enter -> Save & Next
                existing_labels[image_id] = {
                    "image_id": image_id,
                    "objects": human_objects,
                    "pothole": 1 if human_ph else 0,
                    "left": STATE_TO_WALKABLE[human_left],
                    "centre": STATE_TO_WALKABLE[human_centre],
                    "right": STATE_TO_WALKABLE[human_right],
                    "verified": "true",
                }
                # Write CSV after every saved image for safety against crashes
                with open(out_csv, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    for row in existing_labels.values():
                        writer.writerow({k: row.get(k, "") for k in fieldnames})
                print(f"[SAVED] {image_id} -> L:{human_left} C:{human_centre} R:{human_right} PH:{human_ph} OBJ:{human_objects}")
                idx += 1
                break

            elif key in (ord("n"), ord("N"), 32):  # N or Space -> Skip
                print(f"[SKIPPED] {image_id}")
                idx += 1
                break

    cv2.destroyAllWindows()
    print(f"\n[DONE] Labelling complete! Annotated {len(existing_labels)} frames saved to {out_csv}")


def main():
    parser = argparse.ArgumentParser(description="Human ground-truth labelling tool")
    parser.add_argument("--source", default="tests/fixtures/real/holdout", help="Source folder of images/sessions")
    parser.add_argument("--out", default="human_labels.csv", help="Output corrections CSV path")
    parser.add_argument("--pothole-model", default="pothole_yolov8.pt", help="Path to pothole model (advisory display)")
    parser.add_argument("--yolo-model", default="yolov8n.pt", help="Path to YOLO model")
    parser.add_argument("--headless", action="store_true", help="Run without opening GUI window")
    args = parser.parse_args()

    label_frames(
        source_dir=Path(args.source),
        out_csv=Path(args.out),
        pothole_model=Path(args.pothole_model) if args.pothole_model else None,
        yolo_model_path=Path(args.yolo_model) if args.yolo_model else None,
        headless=args.headless,
    )


if __name__ == "__main__":
    main()
