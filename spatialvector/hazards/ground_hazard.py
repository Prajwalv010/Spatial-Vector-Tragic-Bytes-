"""M13 — Ground Hazard Detector.

Detects ground-level hazards (potholes, surface damage) in the lower region
of the camera frame using a conservative rule-based CV pipeline.

Design decisions:
- Uses a FIXED absolute dark-threshold (not Otsu) to avoid firing on any image.
- Requires minimum persistence across N frames before promoting a detection.
- Gates against YOLO tracks: discards candidates that overlap tracked objects.
- Can accept an optional ONNX model for trained pothole detection as primary cue.
- Output is a list of HazardDetection dataclasses, not raw dicts.
- All thresholds are configurable at construction time (or via YAML config dict).

Distance estimation (ground-plane geometry):
  dist = (camera_height_m * focal_length_px) / max(1, bbox_bottom_px - horizon_row_px)
  where:
    focal_length_px  = focal_length_ratio * frame_width
    horizon_row_px   = frame_height * (0.5 - tan(pitch_rad))
  This requires camera_height_m, camera_pitch_deg, and focal_length_ratio from
  default.yaml (hazard_detector section). When any parameter is None / 0.0, the
  returned dist_m is None and is labelled "distance:unknown" in the UI and logs.
  The constant 5.0m–0.4m linear fallback has been removed (it was not calibrated).
- Uses a FIXED absolute dark-threshold (not Otsu) to avoid firing on any image.
- Requires minimum persistence across N frames before promoting a detection.
- Gates against YOLO tracks: discards candidates that overlap tracked objects.
- Can accept an optional ONNX model for trained pothole detection as primary cue.
- Output is a list of HazardDetection dataclasses, not raw dicts.
- All thresholds are configurable at construction time (or via YAML config dict).

This module ONLY detects and outputs hazards.
It does NOT write to the risk engine, voice engine, or haptic output.
Those are the responsibility of the guidance layer (M15).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np
import math

logger = logging.getLogger(__name__)


# ─── Output Schema ──────────────────────────────────────────────────────────

@dataclass
class HazardDetection:
    """A single confirmed ground hazard detection."""
    hazard_class: str          # "pothole" | "manhole" | "surface_damage" | "anomaly"
    bbox_xyxy: Tuple[float, float, float, float]   # in original image coordinates
    confidence: float          # 0.0 – 1.0; from heuristic scoring or model output
    dist_m: Optional[float]    # geometric distance in metres, or None if uncalibratable
    dist_provenance: str       # "ground_plane_geometry" | "unknown"
    source: str                # "heuristic" | "model" | "possible surface anomaly (unverified)"
    frame_id: int
    timestamp: float
    persistence_frames: int = 1   # how many consecutive frames this candidate has been seen
    is_verified: bool = False


def _cand_overlap_with_yolo(cand_bbox: Tuple[float,float,float,float], yolo_bbox: Tuple[float,float,float,float]) -> float:
    """Compute overlap metric between candidate box and a YOLO box.

    Returns the maximum of:
    - standard IoU
    - containment ratio: intersection / cand_area
    This ensures that if a candidate is inside a large YOLO box (e.g. bed, chair, desk, bag),
    the containment ratio is high (~1.0) and triggers gating.
    """
    ix1 = max(cand_bbox[0], yolo_bbox[0]); iy1 = max(cand_bbox[1], yolo_bbox[1])
    ix2 = min(cand_bbox[2], yolo_bbox[2]); iy2 = min(cand_bbox[3], yolo_bbox[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    cand_area = max(1e-6, (cand_bbox[2] - cand_bbox[0]) * (cand_bbox[3] - cand_bbox[1]))
    yolo_area = max(1e-6, (yolo_bbox[2] - yolo_bbox[0]) * (yolo_bbox[3] - yolo_bbox[1]))
    iou = inter / max(1e-6, cand_area + yolo_area - inter)
    containment = inter / cand_area
    return max(iou, containment)


def _iou(a: Tuple[float,float,float,float], b: Tuple[float,float,float,float]) -> float:
    """Compute IoU between two (x1,y1,x2,y2) boxes."""
    ix1 = max(a[0], b[0]);  iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]);  iy2 = min(a[3], b[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = max(1e-6, (a[2]-a[0]) * (a[3]-a[1]))
    area_b = max(1e-6, (b[2]-b[0]) * (b[3]-b[1]))
    return inter / (area_a + area_b - inter)


# ─── Ground Hazard Detector ─────────────────────────────────────────────────

class GroundHazardDetector:
    """M13 — Ground Hazard Detector.

    Runs a conservative multi-criteria CV heuristic on the lower ground region
    of each frame to detect pothole-like dark depressions.

    Key safety properties:
    - Fixed threshold (not Otsu): the detector does NOT fire on every image.
    - YOLO overlap gating: any candidate overlapping a tracked object is discarded.
    - Persistence gate: candidates must survive N consecutive frames before output.
    - Single CLAHE pass (the caller's enhanced image is already one pass;
      pass raw=True to receive the raw frame instead).
    - All parameters exposed and configurable.
    """

    def __init__(
        self,
        # Ground ROI vertical limits (fraction of frame height)
        roi_y_start: float = 0.58,
        roi_y_end: float = 0.90,
        roi_x_margin: float = 0.07,
        # Fixed absolute dark threshold (0–255). NOT Otsu.
        # Pixels darker than this relative to the local mean will be flagged.
        dark_offset: int = 30,            # flag pixels darker than (mean − dark_offset)
        dark_min_abs: int = 40,           # never flag pixels above this absolute value
        # Shape / size gates
        min_area_frac: float = 0.006,     # min fraction of ROI area
        max_area_frac: float = 0.30,      # max fraction of ROI area
        max_aspect: float = 4.5,          # max width/height ratio
        min_aspect: float = 0.22,         # min width/height ratio
        max_solidity: float = 0.88,       # max solidity (regular = not pothole)
        min_solidity: float = 0.30,       # min solidity (noise / very jagged shapes)
        min_edge_density: float = 5.0,    # minimum mean edge density inside bbox
        # Confidence scoring weights
        w_area: float = 0.35,
        w_edge: float = 0.40,
        w_shape: float = 0.25,
        min_conf: float = 0.45,           # detections below this confidence are dropped (higher than object threshold)
        # YOLO overlap gating
        max_iou_with_yolo: float = 0.15,  # discard if IoU with any YOLO track > this
        # Persistence gate (consecutive frames required before promoting)
        persistence_required: int = 2,
        # Max candidates returned per frame
        max_detections: int = 3,
        # Optional trained ONNX/YOLO model path (None = heuristic only)
        model_path: Optional[str] = None,
        # Whether to run heuristic fallback when a trained model is loaded
        enable_heuristic_fallback: bool = False,
        # Whether heuristic is permitted when NO trained model is loaded
        # (Safety critical: default False; unverified heuristic must not produce pothole alerts)
        allow_heuristic: bool = False,
        # Two-stage rule: require freespace road/ground confirmation before accepting potholes
        require_ground_confirmation: bool = True,
        # Advisory only mode: potholes alert visually/audibly but do not halt user navigation
        advisory_only: bool = True,
        # Camera geometry for ground-plane distance estimation (from default.yaml)
        camera_height_m: Optional[float] = None,
        camera_pitch_deg: Optional[float] = None,
        focal_length_ratio: Optional[float] = None,
    ):
        self.roi_y_start = roi_y_start
        self.roi_y_end   = roi_y_end
        self.roi_x_margin = roi_x_margin
        self.dark_offset = dark_offset
        self.dark_min_abs = dark_min_abs
        self.min_area_frac = min_area_frac
        self.max_area_frac = max_area_frac
        self.max_aspect   = max_aspect
        self.min_aspect   = min_aspect
        self.max_solidity = max_solidity
        self.min_solidity = min_solidity
        self.min_edge_density = min_edge_density
        self.w_area  = w_area
        self.w_edge  = w_edge
        self.w_shape = w_shape
        self.min_conf = min_conf
        self.max_iou_with_yolo = max_iou_with_yolo
        self.persistence_required = persistence_required
        self.max_detections = max_detections
        self.enable_heuristic_fallback = enable_heuristic_fallback
        self.allow_heuristic = allow_heuristic
        self.require_ground_confirmation = require_ground_confirmation
        self.advisory_only = advisory_only

        # Persistence tracking: list of (bbox_xyxy, consecutive_count)
        self._persistence: List[Tuple[Tuple, int]] = []

        # Optional trained model
        self._model = None
        if model_path:
            self._load_model(model_path)

        # Camera geometry for ground-plane distance estimation
        # Read from hazard_detector section of default.yaml (passed as constructor args)
        self.camera_height_m: Optional[float] = camera_height_m
        self.camera_pitch_deg: Optional[float] = camera_pitch_deg
        self.focal_length_ratio: Optional[float] = focal_length_ratio

        mode_str = "trained-model" if self._model else ("heuristic-unverified" if allow_heuristic else "disabled (no model)")
        if self.camera_height_m and self.focal_length_ratio:
            dist_mode = "ground-plane geometry"
        else:
            dist_mode = "distance:UNKNOWN (camera geometry not configured)"
        logger.info("GroundHazardDetector (M13) initialised — mode=%s, persistence=%d, distance=%s",
                    mode_str, persistence_required, dist_mode)

    @property
    def has_trained_model(self) -> bool:
        """True if a verified trained pothole model is loaded."""
        return self._model is not None

    def _load_model(self, path: str):
        """Load optional trained ONNX/YOLO pothole model."""
        try:
            from ultralytics import YOLO
            self._model = YOLO(path)
            logger.info("M13: Loaded trained pothole model from %s", path)
        except Exception as e:
            logger.warning("M13: Could not load pothole model from %s: %s. Model disabled.", path, e)
            self._model = None

    # ── Public API ────────────────────────────────────────────────────────────

    def detect(
        self,
        frame_bgr: np.ndarray,
        yolo_bboxes: Optional[List[Tuple[float,float,float,float]]] = None,
        frame_id: int = 0,
        timestamp: Optional[float] = None,
        freespace_result: Optional[object] = None,
    ) -> List[HazardDetection]:
        """Detect ground hazards in one frame.

        Args:
            frame_bgr: Raw (un-enhanced or lightly-enhanced) BGR image.
            yolo_bboxes: List of (x1,y1,x2,y2) boxes from YOLO detector/tracker.
                         Candidates overlapping these are discarded.
            frame_id: Frame index for persistence tracking.
            timestamp: Monotonic timestamp (defaults to time.monotonic()).
            freespace_result: Optional FreespaceResult from M14. When require_ground_confirmation
                              is True, candidates on non-ground/unverified surfaces are rejected.

        Returns:
            List of confirmed HazardDetection objects (sorted by confidence).
        """
        if timestamp is None:
            timestamp = time.monotonic()

        # Two-stage rule: require freespace road/ground confirmation
        if self.require_ground_confirmation and freespace_result is not None:
            c_stat = getattr(freespace_result, "centre", None)
            c_conf = getattr(freespace_result, "centre_conf", 0.0)
            reasons = getattr(freespace_result, "reasons", {})
            c_reason = reasons.get("centre", "") if isinstance(reasons, dict) else ""
            # If scene is completely non-ground (e.g. wall, table edge, covered lens, stairs)
            is_valid_ground_surface = (
                (c_stat is not None and str(c_stat).endswith("WALKABLE"))
                or ("ground confirmed" in c_reason)
                or (c_conf >= 0.35 and not any(k in c_reason for k in ("wall", "table", "stairs", "dark", "blurry", "step")))
            )
            if not is_valid_ground_surface:
                # Do not hallucinate potholes on blank walls, tables, or covered lenses
                return []

        # If a trained model is loaded, use it as primary detector
        if self._model is not None:
            candidates = self._model_detect(frame_bgr, frame_id, timestamp)
            if self.enable_heuristic_fallback and self.allow_heuristic:
                h_candidates = self._heuristic_detect(frame_bgr)
                candidates = self._merge_candidates(h_candidates, candidates)
        else:
            if self.allow_heuristic:
                candidates = self._heuristic_detect(frame_bgr)
                for c in candidates:
                    c['source'] = 'possible surface anomaly (unverified)'
                    c['hazard_class'] = 'anomaly'
            else:
                candidates = []

        # Gate against YOLO detections (discard if inside or heavily overlapping an object)
        if yolo_bboxes:
            candidates = [
                c for c in candidates
                if all(_cand_overlap_with_yolo(c['bbox'], yb) <= self.max_iou_with_yolo for yb in yolo_bboxes)
            ]

        # Confidence filter
        candidates = [c for c in candidates if c['conf'] >= self.min_conf]

        # Persistence gate
        candidates = self._apply_persistence(candidates)

        # Sort and cap
        candidates.sort(key=lambda c: c['conf'], reverse=True)
        candidates = candidates[:self.max_detections]

        return [
            HazardDetection(
                hazard_class=c.get('hazard_class', 'pothole'),
                bbox_xyxy=c['bbox'],
                confidence=c['conf'],
                dist_m=c.get('dist_m', None),
                dist_provenance=c.get('dist_provenance', 'unknown'),
                source=c.get('source', 'heuristic'),
                frame_id=frame_id,
                timestamp=timestamp,
                persistence_frames=c.get('persist', 1),
                is_verified=(c.get('source') == 'model'),
            )
            for c in candidates
        ]

    # ── Ground-Plane Distance Estimation ─────────────────────────────────────

    def _compute_ground_plane_distance(
        self,
        fy2: float,
        frame_h: int,
        frame_w: int,
    ) -> Tuple[Optional[float], str]:
        """Compute hazard distance from ground-plane pinhole geometry.

        Model: camera at height H (m) above a flat ground plane, depressed
        by pitch_deg below the horizon.

            horizon_px = frame_h * (0.5 - tan(pitch_rad))
            focal_px   = focal_length_ratio * frame_w
            dist_m     = (H * focal_px) / max(1, fy2 - horizon_px)

        Returns (dist_m, provenance_str).
        When parameters are missing: returns (None, "unknown").
        Callers MUST log provenance_str and show it in any UI displaying dist_m.
        """
        H   = self.camera_height_m
        pit = self.camera_pitch_deg
        flr = self.focal_length_ratio

        if not (H and H > 0 and pit is not None and flr and flr > 0):
            return None, "unknown"

        pitch_rad  = math.radians(pit)
        focal_px   = flr * frame_w
        horizon_px = frame_h * (0.5 - math.tan(pitch_rad))
        denom      = max(1.0, fy2 - horizon_px)
        dist_m     = float(np.clip((H * focal_px) / denom, 0.2, 20.0))
        return dist_m, "ground_plane_geometry"

    # ── Heuristic Pipeline ────────────────────────────────────────────────────

    def _heuristic_detect(self, frame_bgr: np.ndarray) -> List[dict]:
        """Conservative rule-based pothole detection using fixed threshold."""
        h, w = frame_bgr.shape[:2]
        y_start = int(h * self.roi_y_start)
        y_end   = int(h * self.roi_y_end)
        x_margin = int(w * self.roi_x_margin)

        roi = frame_bgr[y_start:y_end, x_margin:w - x_margin]
        if roi.size == 0:
            return []

        # Single CLAHE pass on grayscale (caller may have already done colour enhancement;
        # we do a fresh grayscale normalisation here for the hazard detector only)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(6, 6))
        gray_eq = clahe.apply(gray)

        # ── FIXED THRESHOLD (not Otsu) ────────────────────────────────────────
        # Pixels that are significantly darker than the local ROI mean.
        # This does NOT fire on every image — a uniformly bright surface produces
        # very few pixels below the threshold.
        local_mean = float(np.mean(gray_eq))
        threshold = max(20, int(local_mean - self.dark_offset))
        threshold = min(threshold, self.dark_min_abs)

        _, dark_mask = cv2.threshold(gray_eq, threshold, 255, cv2.THRESH_BINARY_INV)

        # Morphological cleanup
        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 6))
        k_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 4))
        dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, k_close)
        dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN,  k_open)

        # Edge density map (texture strength proxy)
        blurred = cv2.GaussianBlur(gray_eq, (5, 5), 1.5)
        edges   = cv2.Canny(blurred, 25, 75)
        edge_map = cv2.GaussianBlur(edges.astype(np.float32), (13, 13), 4)

        contours, _ = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        roi_area = roi.shape[0] * roi.shape[1]
        candidates = []

        for cnt in contours:
            area = cv2.contourArea(cnt)

            # Size gate
            if area < roi_area * self.min_area_frac or area > roi_area * self.max_area_frac:
                continue

            x, y, bw, bh = cv2.boundingRect(cnt)

            # Aspect ratio gate
            aspect = bw / max(bh, 1)
            if aspect > self.max_aspect or aspect < self.min_aspect:
                continue

            # Solidity gate (irregular blobs only)
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            solidity = area / max(hull_area, 1.0)
            if solidity > self.max_solidity or solidity < self.min_solidity:
                continue

            # Edge density inside bbox (potholes have strong boundary edges)
            edge_crop = edge_map[y:y+bh, x:x+bw]
            mean_edge = float(np.mean(edge_crop)) if edge_crop.size > 0 else 0.0
            if mean_edge < self.min_edge_density:
                continue

            # Additional check: candidate should NOT cover >40% uniform background
            # (a shadow has little internal texture variation)
            roi_crop = gray_eq[y:y+bh, x:x+bw]
            if roi_crop.size > 0:
                internal_std = float(np.std(roi_crop))
                if internal_std < 8.0:   # near-uniform patch — likely shadow / floor
                    continue

            # Map to full-frame coordinates
            fx1 = float(x + x_margin)
            fy1 = float(y + y_start)
            fx2 = float(fx1 + bw)
            fy2 = float(fy1 + bh)

            # Ground-plane distance estimation
            dist_m, dist_provenance = self._compute_ground_plane_distance(fy2, h, w)

            # Confidence score (weighted combination of cues)
            area_score  = min(1.0, area / (roi_area * 0.06))
            edge_score  = min(1.0, mean_edge / 25.0)
            # Ideal pothole solidity ~0.65 (irregular but not fragmented)
            shape_score = max(0.0, 1.0 - abs(solidity - 0.65) / 0.40)
            conf = float(np.clip(
                self.w_area  * area_score +
                self.w_edge  * edge_score +
                self.w_shape * shape_score,
                0.0, 0.95
            ))

            candidates.append({
                'bbox': (fx1, fy1, fx2, fy2),
                'conf': conf,
                'dist_m': dist_m,
                'dist_provenance': dist_provenance,
                'source': 'heuristic',
            })

        return candidates

    def _model_detect(self, frame_bgr: np.ndarray, frame_id: int, timestamp: float) -> List[dict]:
        """Run trained ONNX/YOLO pothole model (placeholder for future trained model)."""
        if self._model is None:
            return []
        try:
            h, w = frame_bgr.shape[:2]
            results = self._model(frame_bgr, conf=self.min_conf, verbose=False)
            out = []
            for r in results:
                if r.boxes is None or len(r.boxes) == 0:
                    continue
                for i in range(len(r.boxes)):
                    xyxy = r.boxes.xyxy[i].cpu().numpy().tolist()
                    conf = float(r.boxes.conf[i].cpu().numpy())
                    fx1, fy1, fx2, fy2 = float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])

                    # Ground ROI gate: candidate must lie on the ground plane
                    fy_center_norm = (fy1 + fy2) / 2.0 / max(h, 1)
                    if fy_center_norm < self.roi_y_start or fy2 < h * (self.roi_y_start * 0.9):
                        continue

                    # Ground-plane distance estimation
                    dist_m, dist_provenance = self._compute_ground_plane_distance(fy2, h, w)

                    out.append({
                        'bbox': (fx1, fy1, fx2, fy2),
                        'conf': conf,
                        'dist_m': dist_m,
                        'dist_provenance': dist_provenance,
                        'source': 'model',
                    })
            return out
        except Exception as e:
            logger.warning("M13: Trained model inference error: %s", e)
            return []

    def _merge_candidates(self, heuristic: List[dict], model: List[dict]) -> List[dict]:
        """Merge heuristic and model candidates; model wins on overlapping regions."""
        merged = list(model)  # model output takes priority
        for h in heuristic:
            if not any(_iou(h['bbox'], m['bbox']) > 0.3 for m in model):
                merged.append(h)
        return merged

    # ── Persistence Gate ──────────────────────────────────────────────────────

    def _apply_persistence(self, candidates: List[dict]) -> List[dict]:
        """Keep only candidates seen in >= persistence_required consecutive frames.

        Candidates not seen this frame lose their count; new ones start at 1.
        Returns candidates that meet the persistence threshold.
        """
        # Match each new candidate to a persisted one by IoU
        new_persistence: List[Tuple[Tuple, int]] = []
        promoted: List[dict] = []

        for cand in candidates:
            best_match = None
            best_iou = 0.3  # minimum IoU to count as "same" candidate
            for (old_bbox, count) in self._persistence:
                iou = _iou(cand['bbox'], old_bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_match = (old_bbox, count)

            new_count = (best_match[1] + 1) if best_match else 1
            new_persistence.append((cand['bbox'], new_count))
            cand['persist'] = new_count

            if new_count >= self.persistence_required:
                promoted.append(cand)

        self._persistence = new_persistence
        return promoted

    def reset(self):
        """Clear persistence state (call when source changes)."""
        self._persistence = []
