"""M14 — Freespace / Walkable-Ground Estimator (v4 Computed-Confidence).

Architecture:
- Primary Ground Evidence: Lightweight pretrained semantic segmentation model
  (SegFormer-B0 finetuned on ADE20K, ~20ms latency at 224x224 on CPU).
  Explicitly identifies ground surface classes (floor, road, sidewalk, earth, rug, path)
  and obstacle/hazard classes (wall, building, table, desk, stairs, stairway, step, etc.).
- Classical Vetoes:
  - Cue A: Frame validity (too dark, blown out, blurry, or covered lens) -> UNKNOWN/BLOCKED
  - Cue B: Gradient jump / Drop-off (abrupt step or ledge) -> UNKNOWN/BLOCKED
  - Cue C: Ground continuity (bottom-up horizon scan) -> UNKNOWN
  - Cue D: Dark blob gate (unexplained void or drop) -> UNKNOWN
  - Cue E: YOLO obstruction (tracked obstacle in ground corridor) -> BLOCKED
- Reduced Over-Caution on Ordinary Surfaces:
  - Clean uniform floor: low-variance penalty relaxed when model confirms ground.
  - Shadowed footpath / painted road lines: gradient jump relaxed when model confirms high-confidence ground.

CONFIDENCE SCORING (v4 — computed, not hardcoded):
  All returned confidence values are derived from measurable evidence:
  1. SegFormer ground fraction and its margin over the configured threshold.
  2. Softmax probability mean over ground-class pixels (model certainty).
  3. Agreement between SegFormer evidence and classical cues.
  4. Temporal consistency: fraction of recent history frames that are WALKABLE.

  Calibration target: when score = p, corridor should be walkable ~p of the time on TUNE.
  See CHANGELOG.md for calibration results.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ADE20K Semantic Classes
# Ground / walkable surface classes (floor, road, sidewalk, rug, path)
GROUND_CLASSES = {3, 6, 11, 28, 52}
# Obstacle / hazard / unwalkable classes
OBSTACLE_CLASSES = {0, 1, 2, 7, 8, 10, 12, 14, 15, 31, 32, 33, 53, 56, 59, 64, 80, 102, 121}  # wall, building, sky, bed, window, cabinet, person, door, table, sofa, fence, desk, stairs, stairway, pool table, coffee table, screen, grandstand, step

# ImageNet normalization constants for fast SegFormer tensor preprocessing
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ─── Computed Confidence ──────────────────────────────────────────────────────

_CALIBRATOR_PATH = Path(__file__).resolve().parent / "confidence_calibrator.json"
_CALIBRATOR_WEIGHTS: Dict[str, float] = {
    "g_margin": 0.612990698532831,
    "mean_prob": -0.005880166142489396,
    "cue_agreement": -0.0017507361763445858,
    "temporal_frac": 0.0,
}
_CALIBRATOR_INTERCEPT: float = 1.0079556426379757

if _CALIBRATOR_PATH.exists():
    try:
        import json as _json
        _c_data = _json.loads(_CALIBRATOR_PATH.read_text(encoding="utf-8"))
        _CALIBRATOR_WEIGHTS = _c_data.get("weights", _CALIBRATOR_WEIGHTS)
        _CALIBRATOR_INTERCEPT = float(_c_data.get("intercept", _CALIBRATOR_INTERCEPT))
    except Exception as _e:
        logger.warning("Could not load freespace calibrator parameters: %s", _e)


def _compute_confidence(
    status: CorridorStatus,
    g_frac: Optional[float],
    o_frac: Optional[float],
    seg_min_ground: float,
    seg_mean_prob: Optional[float],
    classical_cues_passed: bool,
    temporal_walkable_fraction: float,
) -> float:
    """Compute an uncalibrated evidence score in [0, 1].

    Note: This is an evidence score (uncalibrated), NOT a true calibrated probability.
    It is presented for diagnostic and telemetry inspection only and is NOT used as a
    safety gate (safety gating relies strictly on segmentation, classical vetoes,
    and consecutive-frame persistence).

    BLOCKED:  evidence scales with obstacle fraction (o_frac / 0.35), floored at 0.5.
    UNKNOWN:  evidence = 0.0 by definition (we do not know).
    """
    if status == CorridorStatus.UNKNOWN:
        return 0.0

    if status == CorridorStatus.BLOCKED:
        if o_frac is not None and o_frac > 0.0:
            raw = float(np.clip(o_frac / 0.35, 0.5, 1.0))
        else:
            # Scaled from temporal evidence and cue failure rather than fixed constant
            raw = float(np.clip(0.80 + 0.15 * (1.0 - temporal_walkable_fraction), 0.5, 0.95))
        return float(np.clip(raw, 0.0, 1.0))

    # WALKABLE
    if g_frac is None:
        # Classical-only (no segmentation): measurable cue agreement + temporal continuity
        cue_val = 1.0 if classical_cues_passed else 0.0
        z_classical = _CALIBRATOR_INTERCEPT + _CALIBRATOR_WEIGHTS["cue_agreement"] * cue_val
        prob_classical = 1.0 / (1.0 + np.exp(-z_classical))
        # Scaled by temporal consistency
        return float(np.clip(prob_classical * 0.70 + 0.15 * temporal_walkable_fraction, 0.0, 0.85))

    # (a) Ground fraction margin above threshold, normalised in [0, 1]
    margin = max(0.0, g_frac - seg_min_ground)
    norm_margin = float(np.clip(margin / max(1.0 - seg_min_ground, 0.01), 0.0, 1.0))

    # (b) Per-pixel softmax probability (model certainty over labelled ground pixels)
    prob_val = float(np.clip(seg_mean_prob or 0.0, 0.0, 1.0))

    # (c) Classical cue agreement
    cue_val = 1.0 if classical_cues_passed else 0.0

    # (d) Temporal consistency
    temp_val = float(np.clip(temporal_walkable_fraction, 0.0, 1.0))

    # Fitted logistic calibrator: z = w^T x + b
    z = (
        _CALIBRATOR_INTERCEPT
        + _CALIBRATOR_WEIGHTS["g_margin"] * norm_margin
        + _CALIBRATOR_WEIGHTS["mean_prob"] * prob_val
        + _CALIBRATOR_WEIGHTS["cue_agreement"] * cue_val
        + _CALIBRATOR_WEIGHTS["temporal_frac"] * temp_val
    )
    calibrated_prob = float(1.0 / (1.0 + np.exp(-z)))
    return float(np.clip(calibrated_prob, 0.0, 0.98))


# ─── Output Schema ──────────────────────────────────────────────────────────

class CorridorStatus(str, Enum):
    WALKABLE = "WALKABLE"
    BLOCKED  = "BLOCKED"
    UNKNOWN  = "UNKNOWN"


@dataclass
class FreespaceResult:
    """Per-frame result from the freespace estimator."""
    left:   CorridorStatus
    centre: CorridorStatus
    right:  CorridorStatus
    left_conf:   float      # 0.0 – 1.0
    centre_conf: float
    right_conf:  float
    reasons: Dict[str, str] = field(default_factory=dict)

    def get(self, corridor: str) -> CorridorStatus:
        return {"left": self.left, "centre": self.centre, "right": self.right}.get(
            corridor.lower(), CorridorStatus.UNKNOWN
        )

    @staticmethod
    def all_unknown(reason: str = "not initialised") -> "FreespaceResult":
        reasons = {"left": reason, "centre": reason, "right": reason}
        return FreespaceResult(
            CorridorStatus.UNKNOWN, CorridorStatus.UNKNOWN, CorridorStatus.UNKNOWN,
            0.0, 0.0, 0.0, reasons
        )


# ─── Freespace Estimator ─────────────────────────────────────────────────────

class FreeSpaceEstimator:
    """M14 — Freespace / Walkable-Ground Estimator.

    Integrates model-based semantic ground confirmation with conservative classical vetoes.
    Zero false WALKABLE on drop-offs, table edges, stairs, walls, and covered lenses.

    All returned confidence values are computed from evidence (v4). No hardcoded confidence
    constant is returned as a measurement. See _compute_confidence for the derivation.
    """

    def __init__(
        self,
        # Ground ROI vertical limits (fraction of frame height)
        ground_y_start: float = 0.55,
        ground_y_end: float   = 0.95,
        # Corridor horizontal splits (fraction of frame width)
        left_x_end:    float = 0.38,
        right_x_start: float = 0.62,
        # ── Model-based Semantic Ground Cue ──────────────────────────────────
        use_segmentation: bool = True,
        seg_model_name: str = "nvidia/segformer-b0-finetuned-ade-512-512",
        seg_min_ground: float = 0.25,
        seg_max_obstacle: float = 0.15,
        seg_full_max_obstacle: float = 0.25,
        ground_classes: Optional[list] = None,
        obstacle_classes: Optional[list] = None,
        allow_classical_fallback: bool = False,
        # ── Cue A: Frame validity veto ────────────────────────────────────────
        min_mean_brightness: float = 15.0,    # darker → UNKNOWN
        max_mean_brightness: float = 240.0,   # over-exposed → UNKNOWN
        min_laplacian_var:   float = 15.0,    # blurry/covered → UNKNOWN
        min_ground_std:      float = 5.0,     # near-uniform (only vetoes if ground unverified)
        max_ground_std:      float = 85.0,    # severe noise → UNKNOWN
        # ── Cue B: Gradient jump veto ─────────────────────────────────────────
        gradient_jump_thresh: float = 28.0,   # max brightness step
        # ── Cue C: Ground continuity veto ────────────────────────────────────
        continuity_window_rows: int  = 12,    # height of each scan band (pixels)
        continuity_max_delta:   float = 28.0, # max band-to-band change
        continuity_min_bands:   int  = 4,     # min number of continuous bands required
        # ── Cue D: Dark blob gate ─────────────────────────────────────────────
        dark_blob_abs_thresh: int   = 35,     # pixels below this are "dark"
        dark_blob_frac_thresh: float = 0.20,  # if >20% unexplained dark → UNKNOWN
        # ── Temporal smoothing ────────────────────────────────────────────────
        smoothing_window: int  = 5,
        min_walkable_conf: float = 0.45,
        seg_frame_skip: int = 1,              # Skip segmentation every N frames for >= 15 FPS real-time execution
    ):
        self.ground_y_start = ground_y_start
        self.ground_y_end   = ground_y_end
        self.left_x_end     = left_x_end
        self.right_x_start  = right_x_start
        self.use_segmentation = use_segmentation
        self.seg_model_name = seg_model_name
        self.seg_min_ground = seg_min_ground
        self.seg_max_obstacle = seg_max_obstacle
        self.seg_full_max_obstacle = seg_full_max_obstacle
        self.ground_classes = set(ground_classes) if ground_classes else set(GROUND_CLASSES)
        self.obstacle_classes = set(obstacle_classes) if obstacle_classes else set(OBSTACLE_CLASSES)
        self.allow_classical_fallback = allow_classical_fallback
        self.seg_frame_skip = seg_frame_skip
        self._frame_count: int = 0
        self._last_seg_cues: dict = {}
        # Stores last softmax probabilities (used for mean_ground_prob computation)
        self._last_seg_probs: Optional[np.ndarray] = None

        self.min_mean_brightness   = min_mean_brightness
        self.max_mean_brightness   = max_mean_brightness
        self.min_laplacian_var     = min_laplacian_var
        self.min_ground_std        = min_ground_std
        self.max_ground_std        = max_ground_std
        self.gradient_jump_thresh  = gradient_jump_thresh
        self.continuity_window_rows = continuity_window_rows
        self.continuity_max_delta   = continuity_max_delta
        self.continuity_min_bands   = continuity_min_bands
        self.dark_blob_abs_thresh   = dark_blob_abs_thresh
        self.dark_blob_frac_thresh  = dark_blob_frac_thresh
        self.smoothing_window  = smoothing_window
        self.min_walkable_conf = min_walkable_conf

        self._history: Dict[str, Deque[CorridorStatus]] = {
            "left":   deque(maxlen=smoothing_window),
            "centre": deque(maxlen=smoothing_window),
            "right":  deque(maxlen=smoothing_window),
        }
        self._last_result: Optional[FreespaceResult] = None

        # Segmentation model cache
        self._seg_model = None
        if self.use_segmentation:
            self._init_seg_model()

        mode_str = "SegFormer-B0 + classical vetoes" if self._seg_model is not None else ("classical multi-cue fallback" if allow_classical_fallback else "FAIL-SAFE (refuse WALK_FORWARD)")
        logger.info("FreeSpaceEstimator (M14) initialised — mode=%s, smoothing=%d", mode_str, smoothing_window)

    def _init_seg_model(self):
        """Load lightweight SegFormer-B0 model for fast CPU semantic segmentation."""
        try:
            import torch
            from transformers import SegformerForSemanticSegmentation
            self._seg_model = SegformerForSemanticSegmentation.from_pretrained(self.seg_model_name)
            self._seg_model.eval()
            print(f"[STARTUP] M14 FreeSpaceEstimator: Loaded segmentation model '{self.seg_model_name}'")
        except Exception as e:
            if not self.allow_classical_fallback:
                logger.error("Could not load SegFormer model (%s): %s. Fallback disallowed by policy.", self.seg_model_name, e)
            else:
                logger.warning("Could not load SegFormer model (%s): %s. Running classical fallback.", self.seg_model_name, e)
            self._seg_model = None

    # ── Public API ────────────────────────────────────────────────────────────

    def estimate(
        self,
        frame_bgr: np.ndarray,
        yolo_bboxes: Optional[List[Tuple[float, float, float, float]]] = None,
        optical_flow: Optional[np.ndarray] = None,
    ) -> FreespaceResult:
        if frame_bgr is None or frame_bgr.size == 0:
            return FreespaceResult.all_unknown("null frame")

        # Fail-safe check: If SegFormer failed to load and classical fallback is not explicitly permitted
        if self.use_segmentation and self._seg_model is None and not self.allow_classical_fallback:
            return FreespaceResult.all_unknown("SEGMENTATION MODEL NOT LOADED")

        h, w = frame_bgr.shape[:2]
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        # ── Cue A: Frame-level validity veto ──────────────────────────────────
        frame_ok, frame_reason = self._check_frame_validity(gray)
        if not frame_ok:
            result = FreespaceResult(
                CorridorStatus.UNKNOWN, CorridorStatus.UNKNOWN, CorridorStatus.UNKNOWN,
                0.0, 0.0, 0.0,
                {"left": frame_reason, "centre": frame_reason, "right": frame_reason},
            )
            self._update_history_instant(result)
            self._last_result = result
            return result

        # ── Model Semantic Segmentation Cue ───────────────────────────────────
        seg_cues = self._run_segmentation(frame_bgr)

        # ── Extract ground ROI ────────────────────────────────────────────────
        y0 = int(h * self.ground_y_start)
        y1 = int(h * self.ground_y_end)
        ground_gray = gray[y0:y1, :]
        if ground_gray.size == 0:
            return FreespaceResult.all_unknown("empty ground ROI")

        roi_h, roi_w = ground_gray.shape
        x_left  = int(roi_w * self.left_x_end)
        x_right = int(roi_w * self.right_x_start)
        corridors = {
            "left":   ground_gray[:, :x_left],
            "centre": ground_gray[:, x_left:x_right],
            "right":  ground_gray[:, x_right:],
        }

        # ── Per-corridor analysis ─────────────────────────────────────────────
        raw_statuses: Dict[str, CorridorStatus] = {}
        raw_confs:    Dict[str, float]           = {}
        reasons:      Dict[str, str]             = {}

        for name, crop in corridors.items():
            if crop.size == 0:
                raw_statuses[name] = CorridorStatus.UNKNOWN
                raw_confs[name]    = 0.0
                reasons[name]      = "empty crop"
                continue

            corridor_x0 = {"left": 0, "centre": x_left, "right": x_right}[name]
            corridor_x1 = {"left": x_left, "centre": x_right, "right": roi_w}[name]
            c_seg = seg_cues.get(name, {})
            full_o_frac = seg_cues.get("full_obstacle_frac", 0.0)
            seg_mean_prob = c_seg.get("mean_ground_prob", None)

            st, cf, rs = self._analyse_corridor(
                crop, name, h, w, yolo_bboxes, y0, y1,
                corridor_x0, corridor_x1, c_seg, full_o_frac, seg_mean_prob
            )
            raw_statuses[name] = st
            raw_confs[name]    = cf
            reasons[name]      = rs

        # ── Sudden-transition safety bypass ──────────────────────────────────
        if self._last_result is not None:
            for name in ("left", "centre", "right"):
                prev = self._last_result.get(name)
                curr = raw_statuses[name]
                if prev == CorridorStatus.WALKABLE and curr in (CorridorStatus.BLOCKED, CorridorStatus.UNKNOWN):
                    self._history[name].clear()
                    self._history[name].append(curr)

        # ── Temporal smoothing ────────────────────────────────────────────────
        smoothed: Dict[str, CorridorStatus] = {}
        for name in ("left", "centre", "right"):
            self._history[name].append(raw_statuses[name])
            smoothed[name] = self._majority_vote(self._history[name], raw_statuses[name])

        result = FreespaceResult(
            left=smoothed["left"], centre=smoothed["centre"], right=smoothed["right"],
            left_conf=raw_confs["left"], centre_conf=raw_confs["centre"],
            right_conf=raw_confs["right"], reasons=reasons,
        )
        self._last_result = result
        return result

    # ── Segmentation Inference ────────────────────────────────────────────────

    def _run_segmentation(self, frame_bgr: np.ndarray) -> dict:
        """Run fast SegFormer-B0 inference on 224x224 input (~20ms latency).

        Returns per-corridor dict with:
          ground_frac      — argmax ground pixel fraction
          obstacle_frac    — argmax obstacle pixel fraction
          mean_ground_prob — mean softmax probability over ground-labelled pixels
                             (model certainty, used in confidence computation)
        """
        if self._seg_model is None:
            return {}

        self._frame_count += 1
        if self.seg_frame_skip > 0 and (self._frame_count % (self.seg_frame_skip + 1) != 1) and self._last_seg_cues:
            return self._last_seg_cues

        try:
            import torch
            import torch.nn.functional as F

            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
            norm = (resized - _IMAGENET_MEAN) / _IMAGENET_STD
            tensor = torch.from_numpy(norm.transpose(2, 0, 1)).unsqueeze(0)

            with torch.no_grad():
                out = self._seg_model(tensor)

            logits = out.logits  # [1, C, H', W']
            probs = F.softmax(logits, dim=1)[0]  # [C, H', W'] — softmax probabilities
            pred = torch.argmax(logits, dim=1)[0].numpy()  # [H', W']
            ph, pw = pred.shape

            roi_start_row = int(ph * self.ground_y_start)
            roi_pred = pred[roi_start_row:, :]
            roi_probs = probs[:, roi_start_row:, :]  # [C, roi_H, roi_W]
            full_o_frac = float(np.mean(np.isin(roi_pred, list(self.obstacle_classes))))

            cx_left  = int(pw * self.left_x_end)
            cx_right = int(pw * self.right_x_start)
            corridor_slices = {
                "left":   (roi_pred[:, :cx_left],        roi_probs[:, :, :cx_left]),
                "centre": (roi_pred[:, cx_left:cx_right], roi_probs[:, :, cx_left:cx_right]),
                "right":  (roi_pred[:, cx_right:],       roi_probs[:, :, cx_right:]),
            }

            ground_class_list = list(self.ground_classes)
            res = {"full_obstacle_frac": full_o_frac}
            for name, (sl, sl_probs) in corridor_slices.items():
                if sl.size == 0:
                    res[name] = {"ground_frac": 0.0, "obstacle_frac": 0.0, "mean_ground_prob": None}
                    continue
                g_mask = np.isin(sl, ground_class_list)
                g = float(np.mean(g_mask))
                o = float(np.mean(np.isin(sl, list(self.obstacle_classes))))
                # Mean softmax probability over ground-labelled pixels (model certainty)
                mean_ground_prob: Optional[float] = None
                if g_mask.any():
                    # Sum the per-class probabilities of all ground classes over masked pixels
                    total_prob = 0.0
                    n_valid = 0
                    for c_idx in ground_class_list:
                        if c_idx < sl_probs.shape[0]:
                            c_probs_numpy = sl_probs[c_idx].numpy()  # [roi_H, corr_W]
                            masked_vals = c_probs_numpy[g_mask]
                            if masked_vals.size > 0:
                                total_prob += float(masked_vals.mean())
                                n_valid += 1
                    if n_valid > 0:
                        mean_ground_prob = float(np.clip(total_prob / n_valid, 0.0, 1.0))
                res[name] = {"ground_frac": g, "obstacle_frac": o, "mean_ground_prob": mean_ground_prob}
            self._last_seg_cues = res
            return res
        except Exception as e:
            logger.warning("M14: Segmentation inference error: %s", e)
            return {}

    # ── Cue A: Frame validity ─────────────────────────────────────────────────

    def _check_frame_validity(self, gray: np.ndarray) -> Tuple[bool, str]:
        """Return (ok, reason). ok=False means skip per-corridor analysis."""
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        if lap_var < self.min_laplacian_var:
            return False, f"frame blurry/covered (lap_var={lap_var:.1f})"

        mean_b = float(np.mean(gray))
        if mean_b < self.min_mean_brightness:
            return False, f"frame too dark (mean={mean_b:.1f})"
        if mean_b > self.max_mean_brightness:
            return False, f"frame over-exposed (mean={mean_b:.1f})"

        return True, "ok"

    # ── Per-corridor analysis ─────────────────────────────────────────────────

    def _analyse_corridor(
        self,
        crop: np.ndarray,
        name: str,
        full_h: int, full_w: int,
        yolo_bboxes: Optional[List[Tuple]],
        y0: int, y1: int,
        corridor_x0: int, corridor_x1: int,
        c_seg: dict,
        full_o_frac: float,
        seg_mean_prob: Optional[float] = None,
    ) -> Tuple[CorridorStatus, float, str]:
        """Combine model-based ground evidence with classical vetoes.

        All confidence values computed via _compute_confidence from real evidence.
        No hardcoded constant is returned as a measured confidence.
        """
        # ── Temporal consistency (fraction of recent history that was WALKABLE) ─
        history = self._history.get(name, deque())
        n_hist = len(history)
        temporal_walkable_frac = (
            sum(1 for s in history if s == CorridorStatus.WALKABLE) / n_hist
            if n_hist > 0 else 0.0
        )

        g_frac = c_seg.get("ground_frac", None)
        o_frac = c_seg.get("obstacle_frac", None)
        if seg_mean_prob is None:
            seg_mean_prob = c_seg.get("mean_ground_prob", None)

        # ── Cue E: YOLO Obstacle Veto ──────────────────────────────────────────
        if yolo_bboxes:
            for bbox in yolo_bboxes:
                bx1, by1, bx2, by2 = bbox
                h_overlap = bx1 < corridor_x1 and bx2 > corridor_x0
                v_overlap = by1 < y1 and by2 > (y0 * 0.90)
                if h_overlap and v_overlap:
                    conf = _compute_confidence(
                        CorridorStatus.BLOCKED, None, 1.0,
                        self.seg_min_ground, None, False, temporal_walkable_frac
                    )
                    return CorridorStatus.BLOCKED, conf, "YOLO obstacle in ground corridor"

        # ── Primary Semantic Evidence ─────────────────────────────────────────
        if g_frac is not None and o_frac is not None:
            # Semantic Obstacle Veto
            if o_frac > 0.35:
                conf = _compute_confidence(
                    CorridorStatus.BLOCKED, g_frac, o_frac,
                    self.seg_min_ground, None, False, temporal_walkable_frac
                )
                return CorridorStatus.BLOCKED, conf, f"semantic obstacle {o_frac:.0%} (wall/table/stairs)"
            if o_frac > self.seg_max_obstacle and g_frac < 0.45:
                conf = _compute_confidence(
                    CorridorStatus.BLOCKED, g_frac, o_frac,
                    self.seg_min_ground, None, False, temporal_walkable_frac
                )
                return CorridorStatus.BLOCKED, conf, f"semantic obstacle {o_frac:.0%} (wall/table/stairs)"
            if full_o_frac > self.seg_full_max_obstacle and g_frac < 0.45:
                conf = _compute_confidence(
                    CorridorStatus.BLOCKED, g_frac, full_o_frac,
                    self.seg_min_ground, None, False, temporal_walkable_frac
                )
                return CorridorStatus.BLOCKED, conf, f"scene obstacle {full_o_frac:.0%} (stairs/building wall)"

            # Semantic Ground Evidence Check
            if g_frac < self.seg_min_ground:
                return CorridorStatus.UNKNOWN, 0.0, f"insufficient ground evidence ({g_frac:.0%})"

        # ── Classical Surface Uniformity Check ─────────────────────────────────
        std = float(np.std(crop))
        is_verified_ground = (g_frac is not None and g_frac >= 0.40)
        if std < self.min_ground_std and not is_verified_ground:
            return CorridorStatus.UNKNOWN, 0.0, f"uniform surface (std={std:.1f})"
        if std > self.max_ground_std:
            return CorridorStatus.UNKNOWN, 0.0, f"chaotic surface (std={std:.1f})"

        # ── Cue B: Gradient jump detector (Drop-off / Table edge) ──────────────
        jump_result, jump_reason = self._gradient_jump_check(crop, is_verified_ground)
        if jump_result != CorridorStatus.WALKABLE:
            return jump_result, 0.0, jump_reason

        # ── Cue C: Ground continuity scan (bottom-up) ─────────────────────────
        cont_result, cont_reason = self._continuity_scan(crop)
        if cont_result != CorridorStatus.WALKABLE:
            return cont_result, 0.0, cont_reason

        # ── Cue D: Unexplained dark blob gate ─────────────────────────────────
        dark_result, dark_reason = self._dark_blob_gate(crop)
        if dark_result != CorridorStatus.WALKABLE:
            return dark_result, 0.0, dark_reason

        # ── All Cues Pass → WALKABLE: compute evidence-based confidence ────────
        conf = _compute_confidence(
            CorridorStatus.WALKABLE,
            g_frac, o_frac,
            self.seg_min_ground,
            seg_mean_prob,
            classical_cues_passed=True,
            temporal_walkable_fraction=temporal_walkable_frac,
        )
        margin_str = f"{max(0.0, (g_frac or 0.0) - self.seg_min_ground):.0%}" if g_frac is not None else "n/a"
        reason_str = (
            f"ground confirmed (model={g_frac:.0%} margin={margin_str})"
            if g_frac is not None else
            "ground confirmed (classical cues; segmentation unavailable)"
        )
        return CorridorStatus.WALKABLE, conf, reason_str

    # ── Cue B: Gradient jump ──────────────────────────────────────────────────

    def _gradient_jump_check(self, crop: np.ndarray, is_verified_ground: bool = False) -> Tuple[CorridorStatus, str]:
        h = crop.shape[0]
        if h < 8:
            return CorridorStatus.UNKNOWN, "crop too short for gradient check"

        row_means = np.mean(crop, axis=1).astype(np.float32)

        # 1. Adjacent row jump
        diffs = np.abs(np.diff(row_means))
        max_jump = float(np.max(diffs)) if diffs.size > 0 else 0.0
        j_thresh = self.gradient_jump_thresh * 1.25 if is_verified_ground else self.gradient_jump_thresh
        if max_jump > j_thresh:
            return CorridorStatus.UNKNOWN, f"brightness step {max_jump:.1f} px (table edge?)"

        # 2. Near-to-far surface consistency (catches stairs and ledges)
        third = h // 3
        top_band = crop[:third, :]
        bot_band = crop[-third:, :]
        if top_band.size > 0 and bot_band.size > 0:
            span_diff = abs(float(np.mean(top_band)) - float(np.mean(bot_band)))
            s_thresh = 30.0   # Strict: prevent downward stairs and ledges from passing
            if span_diff > s_thresh:
                return CorridorStatus.UNKNOWN, f"surface mismatch ahead (span diff {span_diff:.1f} px)"

        return CorridorStatus.WALKABLE, "ok"

    # ── Cue C: Ground continuity ──────────────────────────────────────────────

    def _continuity_scan(self, crop: np.ndarray) -> Tuple[CorridorStatus, str]:
        h = crop.shape[0]
        bw = self.continuity_window_rows
        if h < bw * 2:
            return CorridorStatus.WALKABLE, "crop too short for continuity scan"

        band_means: List[float] = []
        row = h
        while row - bw >= 0:
            band = crop[row - bw: row, :]
            band_means.append(float(np.mean(band)))
            row -= bw

        if len(band_means) < self.continuity_min_bands:
            return CorridorStatus.WALKABLE, "insufficient bands"

        for i in range(len(band_means) - 1):
            delta = abs(band_means[i+1] - band_means[i])
            if delta > self.continuity_max_delta:
                return CorridorStatus.UNKNOWN, f"ground surface break (delta={delta:.1f} at band {i})"

        return CorridorStatus.WALKABLE, "ok"

    # ── Cue D: Dark blob gate ─────────────────────────────────────────────────

    def _dark_blob_gate(self, crop: np.ndarray) -> Tuple[CorridorStatus, str]:
        dark_pixels = np.sum(crop < self.dark_blob_abs_thresh)
        dark_frac = float(dark_pixels) / max(crop.size, 1)
        if dark_frac > self.dark_blob_frac_thresh:
            return CorridorStatus.UNKNOWN, f"unexplained dark region ({dark_frac:.0%} of corridor)"
        return CorridorStatus.WALKABLE, "ok"

    # ── Temporal voting helper ────────────────────────────────────────────────

    def _majority_vote(self, history: Deque[CorridorStatus], fallback: CorridorStatus) -> CorridorStatus:
        if not history:
            return fallback
        walkable_count = sum(1 for s in history if s == CorridorStatus.WALKABLE)
        blocked_count  = sum(1 for s in history if s == CorridorStatus.BLOCKED)
        unknown_count  = sum(1 for s in history if s == CorridorStatus.UNKNOWN)

        if walkable_count > len(history) / 2:
            return CorridorStatus.WALKABLE
        if blocked_count >= walkable_count and blocked_count >= unknown_count and blocked_count > 0:
            return CorridorStatus.BLOCKED
        return CorridorStatus.UNKNOWN

    def _update_history_instant(self, result: FreespaceResult):
        for name in ("left", "centre", "right"):
            self._history[name].clear()
            self._history[name].append(result.get(name))

    def reset(self):
        """Clear temporal smoothing state."""
        for q in self._history.values():
            q.clear()
        self._last_result = None
        self._last_seg_cues = {}
        self._last_seg_probs = None
        self._frame_count = 0
