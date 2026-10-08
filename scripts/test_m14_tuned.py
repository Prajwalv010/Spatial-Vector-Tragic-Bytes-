"""Test tuned M14 freespace logic on TUNE split only.

Verifies:
1. False WALKABLE rate on hazard categories remains 0.00% (or <= 0.01).
2. WALKABLE recall on clear categories reaches >= 80%.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
from collections import Counter

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
from ultralytics import YOLO

from spatialvector.freespace.corridor_estimator import FreeSpaceEstimator, CorridorStatus, _IMAGENET_MEAN, _IMAGENET_STD

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

# Ground-blocking YOLO classes for a walking pedestrian
GROUND_BLOCKING_CLASSES = {
    "person", "bicycle", "car", "motorcycle", "bus", "truck",
    "bench", "dog", "chair", "couch", "bed", "dining table", "suitcase", "fire hydrant"
}


class TunedFreeSpaceEstimator(FreeSpaceEstimator):
    """Refined M14 estimator with configurable classes and ground-informed relaxation."""

    def __init__(
        self,
        ground_classes=None,
        obstacle_classes=None,
        allow_classical_fallback: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.ground_classes = set(ground_classes) if ground_classes else {3, 6, 11, 28, 52}
        self.obstacle_classes = set(obstacle_classes) if obstacle_classes else {
            0, 1, 2, 7, 8, 10, 12, 14, 15, 31, 32, 33, 53, 56, 59, 64, 80, 102, 121
        }
        self.allow_classical_fallback = allow_classical_fallback

    def _run_segmentation(self, frame_bgr: np.ndarray) -> dict:
        if self._seg_model is None:
            return {}
        try:
            import torch
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
            norm = (resized - _IMAGENET_MEAN) / _IMAGENET_STD
            tensor = torch.from_numpy(norm.transpose(2, 0, 1)).unsqueeze(0)

            with torch.no_grad():
                out = self._seg_model(tensor)
            pred = torch.argmax(out.logits, dim=1)[0].numpy()
            ph, pw = pred.shape

            roi_pred = pred[int(ph * self.ground_y_start):, :]
            full_o_frac = float(np.mean(np.isin(roi_pred, list(self.obstacle_classes))))

            cx_left = int(pw * self.left_x_end)
            cx_right = int(pw * self.right_x_start)
            slices = {
                "left": roi_pred[:, :cx_left],
                "centre": roi_pred[:, cx_left:cx_right],
                "right": roi_pred[:, cx_right:],
            }

            res = {"full_obstacle_frac": full_o_frac}
            for name, sl in slices.items():
                if sl.size == 0:
                    res[name] = {"ground_frac": 0.0, "obstacle_frac": 0.0}
                    continue
                g = float(np.mean(np.isin(sl, list(self.ground_classes))))
                o = float(np.mean(np.isin(sl, list(self.obstacle_classes))))
                res[name] = {"ground_frac": g, "obstacle_frac": o}
            return res
        except Exception:
            return {}

    def _analyse_corridor(
        self,
        crop: np.ndarray,
        name: str,
        full_h: int, full_w: int,
        yolo_bboxes: list,
        y0: int, y1: int,
        corridor_x0: int, corridor_x1: int,
        c_seg: dict,
        full_o_frac: float,
    ):
        # 1. Cue E: YOLO Obstacle Veto (filtered for ground-blocking objects)
        if yolo_bboxes:
            for item in yolo_bboxes:
                if isinstance(item, tuple) and len(item) == 2:
                    bbox, cls_name = item
                    if cls_name not in GROUND_BLOCKING_CLASSES:
                        continue
                else:
                    bbox = item
                bx1, by1, bx2, by2 = bbox
                h_overlap = bx1 < corridor_x1 and bx2 > corridor_x0
                # Must penetrate at least 15% into the ground ROI and have non-negligible height
                corridor_h = y1 - y0
                min_penetration_y = y0 + corridor_h * 0.15
                v_overlap = by2 > min_penetration_y and by1 < y1
                box_h_in_roi = min(by2, y1) - max(by1, y0)
                if h_overlap and v_overlap and box_h_in_roi > 15:
                    return CorridorStatus.BLOCKED, 0.95, "YOLO obstacle in ground corridor"

        g_frac = c_seg.get("ground_frac", None)
        o_frac = c_seg.get("obstacle_frac", None)

        # 2. Semantic Segmentation Cues
        is_verified_ground = False
        if g_frac is not None and o_frac is not None:
            # Table/stairs/wall obstacle in corridor
            if o_frac > 0.35:
                return CorridorStatus.BLOCKED, 0.85, f"semantic obstacle {o_frac:.0%} (wall/table/stairs)"
            if o_frac > self.seg_max_obstacle and g_frac < 0.40:
                return CorridorStatus.BLOCKED, 0.85, f"semantic obstacle {o_frac:.0%} (wall/table/stairs)"

            # If full scene is >35% obstacle and corridor itself has low ground (<30%), veto
            if full_o_frac > 0.35 and g_frac < 0.30:
                return CorridorStatus.BLOCKED, 0.80, f"scene obstacle {full_o_frac:.0%} (stairs/wall)"

            # Positive ground confirmation
            if g_frac >= 0.30 and o_frac <= 0.25:
                is_verified_ground = True
            elif g_frac < 0.20:
                return CorridorStatus.UNKNOWN, 0.10, f"insufficient ground evidence ({g_frac:.0%})"

        # 3. Classical Surface Uniformity Check
        std = float(np.std(crop))
        if std < self.min_ground_std and not is_verified_ground:
            return CorridorStatus.UNKNOWN, 0.0, f"uniform surface (std={std:.1f})"
        if std > self.max_ground_std:
            return CorridorStatus.UNKNOWN, 0.0, f"chaotic surface (std={std:.1f})"

        # 4. Cue B: Gradient Jump & Span Difference Check
        h = crop.shape[0]
        if h < 8:
            return CorridorStatus.UNKNOWN, 0.10, "crop too short"

        row_means = np.mean(crop, axis=1).astype(np.float32)
        diffs = np.abs(np.diff(row_means))
        max_jump = float(np.max(diffs)) if diffs.size > 0 else 0.0

        j_thresh = self.gradient_jump_thresh * 1.50 if is_verified_ground else self.gradient_jump_thresh
        if max_jump > j_thresh:
            return CorridorStatus.UNKNOWN, 0.10, f"brightness step {max_jump:.1f} px (table edge?)"

        # Span Difference
        third = h // 3
        top_band = crop[:third, :]
        bot_band = crop[-third:, :]
        if top_band.size > 0 and bot_band.size > 0:
            span_diff = abs(float(np.mean(top_band)) - float(np.mean(bot_band)))
            s_thresh = 52.0 if is_verified_ground else 30.0
            if span_diff > s_thresh:
                return CorridorStatus.UNKNOWN, 0.10, f"surface mismatch ahead (span diff {span_diff:.1f} px)"

        # 5. Cue C: Continuity Scan
        bw = self.continuity_window_rows
        if h >= bw * 2:
            band_means = []
            row = h
            while row - bw >= 0:
                band_means.append(float(np.mean(crop[row - bw: row, :])))
                row -= bw
            if len(band_means) >= self.continuity_min_bands:
                max_allowed_delta = 35.0 if is_verified_ground else self.continuity_max_delta
                for i in range(len(band_means) - 1):
                    delta = abs(band_means[i+1] - band_means[i])
                    if delta > max_allowed_delta:
                        return CorridorStatus.UNKNOWN, 0.10, f"ground surface break (delta={delta:.1f} at band {i})"

        # 6. Cue D: Dark Blob Gate
        dark_pixels = np.sum(crop < self.dark_blob_abs_thresh)
        dark_frac = float(dark_pixels) / max(crop.size, 1)
        max_dark_frac = 0.35 if is_verified_ground else self.dark_blob_frac_thresh
        if dark_frac > max_dark_frac:
            return CorridorStatus.UNKNOWN, 0.10, f"unexplained dark region ({dark_frac:.0%} of corridor)"

        conf = 0.85 if is_verified_ground else 0.65
        m_str = f"{g_frac:.0%}" if g_frac is not None else "n/a"
        return CorridorStatus.WALKABLE, conf, f"ground confirmed (model={m_str})"


def main():
    estimator = TunedFreeSpaceEstimator(
        ground_classes=[3, 6, 11, 28, 52],
        obstacle_classes=[0, 1, 2, 7, 8, 10, 12, 14, 15, 31, 32, 33, 53, 56, 59, 64, 80, 102, 121],
        smoothing_window=1,
    )
    yolo = YOLO("yolov8n.pt")

    clear_total = 0
    clear_walkable = 0

    hazard_total = 0
    hazard_false_walkable = 0

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

        # Get YOLO detections with class labels
        yres = yolo(img, verbose=False)[0]
        y_items = []
        if yres.boxes is not None and len(yres.boxes) > 0:
            for b in yres.boxes:
                cname = yres.names[int(b.cls[0])]
                xyxy = tuple(b.xyxy[0].cpu().numpy().tolist())
                y_items.append((xyxy, cname))

        res = estimator.estimate(img, yolo_bboxes=y_items)

        if cat in CLEAR_CATEGORIES:
            clear_total += 1
            if res.centre == CorridorStatus.WALKABLE:
                clear_walkable += 1

        if cat in HAZARD_CATEGORIES:
            hazard_total += 1
            if any(res.get(c) == CorridorStatus.WALKABLE for c in ("left", "centre", "right")):
                hazard_false_walkable += 1
                print(f"[TUNED HAZARD FALSE WALKABLE] {meta['id']} ({cat}): L={res.left}, C={res.centre}, R={res.right}")

    print("\n" + "=" * 70)
    print("  TUNED M14 EVALUATION ON TUNE SPLIT")
    print("=" * 70)
    print(f"Hazard Frames: {hazard_total}")
    print(f"Hazard False WALKABLE: {hazard_false_walkable} / {hazard_total} ({hazard_false_walkable/max(1,hazard_total):.2%})")
    print(f"Clear Frames: {clear_total}")
    print(f"Clear Walkable Recall: {clear_walkable} / {clear_total} ({clear_walkable/max(1,clear_total):.1%})")


if __name__ == '__main__':
    main()
