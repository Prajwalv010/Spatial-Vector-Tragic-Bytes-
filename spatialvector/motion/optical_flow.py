"""
M04 — Optical Flow & Focus of Expansion (FOE)

Estimates sparse Lucas-Kanade optical flow between consecutive frames and computes
the Focus of Expansion (FOE): the image point from which apparent motion radiates
when the camera is moving forward. The FOE is a geometric proxy for travel direction.

Architectural rule: this module outputs geometry ONLY — no risk scores, no warnings,
no collision logic. That is M07's job.

Failure modes handled:
  - Low texture (few trackable points) → flow_quality drops; still valid output.
  - Motion blur → forward-backward error rejection catches most bad tracks.
  - Pure camera rotation → flow field is parallel, not radial → foe_confidence degrades.
  - Dynamic foreground (people in frame) → RANSAC-style outlier rejection down-weights them.
"""

from __future__ import annotations

import logging
import math
import time
from typing import Optional

import cv2
import numpy as np

from spatialvector.motion.schemas import FlowResult

logger = logging.getLogger(__name__)


class OpticalFlowEstimator:
    """Sparse Lucas-Kanade optical flow + FOE estimation for consecutive frame pairs.

    All thresholds are configurable — never hardcoded. See config/default.yaml
    for the live values; defaults here are safe fallbacks for testing.
    """

    def __init__(
        self,
        max_corners: int = 200,
        quality_level: float = 0.01,
        min_distance: float = 7.0,
        fb_error_threshold_px: float = 2.0,
        reseed_below_point_count: int = 50,
        lk_win_size: tuple[int, int] = (21, 21),
        lk_max_level: int = 3,
    ):
        """
        Args:
            max_corners: max seed points from goodFeaturesToTrack each re-seed.
            quality_level: goodFeaturesToTrack quality ratio.
            min_distance: min pixel distance between seed points.
            fb_error_threshold_px: forward-backward consistency check — tracks with
                error > this threshold in pixels are rejected as unreliable.
            reseed_below_point_count: if tracked point count drops below this, re-seed.
            lk_win_size: Lucas-Kanade search window size.
            lk_max_level: LK pyramid levels.
        """
        self.max_corners = max_corners
        self.quality_level = quality_level
        self.min_distance = min_distance
        self.fb_error_threshold_px = fb_error_threshold_px
        self.reseed_below_point_count = reseed_below_point_count
        self.lk_win_size = lk_win_size
        self.lk_max_level = lk_max_level

        self._lk_params = dict(
            winSize=lk_win_size,
            maxLevel=lk_max_level,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )

        self._prev_gray: Optional[np.ndarray] = None
        self._prev_pts: Optional[np.ndarray] = None  # shape (N, 1, 2) float32

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, frame_bgr: np.ndarray, frame_id: int, timestamp: float) -> FlowResult:
        """Process one frame. Returns FlowResult with flow vectors and FOE.

        On the very first call (no previous frame) returns an empty FlowResult
        with flow_quality=0 and NaN FOE — not an error.
        """
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        if self._prev_gray is None or self._prev_gray.shape != gray.shape:
            # Bootstrap or resolution shift: re-seed initial points, return empty result
            self._prev_gray = gray
            self._prev_pts = self._detect_features(gray)
            return FlowResult(
                frame_id=frame_id,
                timestamp=timestamp,
                flow_vectors=[],
                flow_quality=0.0,
                foe_x=float("nan"),
                foe_y=float("nan"),
                foe_confidence=0.0,
            )

        # Track seed points from previous frame
        good_src, good_dst, num_rejected = self._track_points(self._prev_gray, gray, self._prev_pts)

        num_survived = len(good_src)
        num_total = len(self._prev_pts) if self._prev_pts is not None else 1
        flow_quality = float(num_survived) / max(float(num_total), 1.0)

        # Build flow vector list (x0, y0, dx, dy)
        flow_vectors: list[tuple[float, float, float, float]] = []
        for src, dst in zip(good_src, good_dst):
            x0, y0 = float(src[0]), float(src[1])
            dx, dy = float(dst[0] - src[0]), float(dst[1] - src[1])
            flow_vectors.append((x0, y0, dx, dy))

        # Estimate FOE from the surviving flow vectors
        foe_x, foe_y, foe_confidence = self._estimate_foe(
            flow_vectors, frame_bgr.shape[1], frame_bgr.shape[0]
        )

        # Re-seed if point count is low
        if num_survived < self.reseed_below_point_count:
            self._prev_pts = self._detect_features(gray)
            logger.debug(
                f"[M04] Re-seeded features at frame {frame_id}: "
                f"{num_survived} points survived, re-seeded {len(self._prev_pts) if self._prev_pts is not None else 0}"
            )
        else:
            # Carry forward the successfully tracked points as next prev_pts
            self._prev_pts = good_dst.reshape(-1, 1, 2).astype(np.float32)

        self._prev_gray = gray

        return FlowResult(
            frame_id=frame_id,
            timestamp=timestamp,
            flow_vectors=flow_vectors,
            flow_quality=flow_quality,
            foe_x=foe_x,
            foe_y=foe_y,
            foe_confidence=foe_confidence,
        )

    def reset(self):
        """Reset internal state — call when source changes or tracking breaks."""
        self._prev_gray = None
        self._prev_pts = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _detect_features(self, gray: np.ndarray) -> np.ndarray:
        """Seed feature points using Shi-Tomasi corner detector."""
        pts = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=self.max_corners,
            qualityLevel=self.quality_level,
            minDistance=self.min_distance,
        )
        if pts is None:
            pts = np.empty((0, 1, 2), dtype=np.float32)
        return pts  # shape (N, 1, 2)

    def _track_points(
        self,
        prev_gray: np.ndarray,
        curr_gray: np.ndarray,
        prev_pts: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, int]:
        """Forward-backward LK flow with consistency check.

        Returns (good_src, good_dst, num_rejected) where arrays are shape (M, 2) float32.
        """
        if prev_pts is None or len(prev_pts) == 0:
            return np.empty((0, 2), np.float32), np.empty((0, 2), np.float32), 0

        try:
            # Forward pass: prev → curr
            fwd_pts, fwd_status, _ = cv2.calcOpticalFlowPyrLK(
                prev_gray, curr_gray, prev_pts, None, **self._lk_params
            )
            # Backward pass: curr → prev (for consistency check)
            bwd_pts, bwd_status, _ = cv2.calcOpticalFlowPyrLK(
                curr_gray, prev_gray, fwd_pts, None, **self._lk_params
            )
        except cv2.error as e:
            logger.warning(f"[M04] Optical flow calculation failed: {e}")
            return np.empty((0, 2), np.float32), np.empty((0, 2), np.float32), 0

        fwd_status = fwd_status.ravel()
        bwd_status = bwd_status.ravel()

        # Forward-backward error: how far did we drift when going back?
        fb_errors = np.linalg.norm(
            prev_pts.reshape(-1, 2) - bwd_pts.reshape(-1, 2), axis=1
        )
        good_mask = (
            (fwd_status == 1)
            & (bwd_status == 1)
            & (fb_errors < self.fb_error_threshold_px)
        )

        num_rejected = int(np.sum(~good_mask))
        good_src = prev_pts.reshape(-1, 2)[good_mask]
        good_dst = fwd_pts.reshape(-1, 2)[good_mask]
        return good_src, good_dst, num_rejected

    def _estimate_foe(
        self,
        flow_vectors: list[tuple[float, float, float, float]],
        frame_w: int,
        frame_h: int,
    ) -> tuple[float, float, float]:
        """Estimate Focus of Expansion via RANSAC-style least-squares line intersection.

        Each flow vector (x0, y0, dx, dy) defines a ray: the line through (x0, y0)
        in direction (dx, dy), extended backward, should pass through the FOE under
        pure translational forward motion.

        Returns (foe_x, foe_y, confidence) where confidence is 0..1.
        Returns (nan, nan, 0.0) if there are too few vectors or flow is dominated by
        rotation (parallel vectors, not radial expansion).
        """
        if len(flow_vectors) < 4:
            return float("nan"), float("nan"), 0.0

        vecs = np.array(flow_vectors, dtype=np.float64)  # (N, 4)
        x0, y0 = vecs[:, 0], vecs[:, 1]
        dx, dy = vecs[:, 2], vecs[:, 3]

        # Magnitude filter: discard very small motion vectors (noise / static points)
        magnitudes = np.sqrt(dx**2 + dy**2)
        mag_mask = magnitudes > 0.3
        if mag_mask.sum() < 4:
            return float("nan"), float("nan"), 0.0

        x0, y0, dx, dy = x0[mag_mask], y0[mag_mask], dx[mag_mask], dy[mag_mask]

        # Check for rotation-dominated flow: if vectors are nearly parallel,
        # there's no radial expansion center — degrade confidently rather than fabricate one.
        angles = np.arctan2(dy, dx)
        angle_std = _circular_std(angles)
        if angle_std < 0.15:
            # Vectors are all pointing in roughly the same direction → pure rotation or lateral slide
            return float("nan"), float("nan"), max(0.0, 1.0 - (0.15 - angle_std) / 0.15)

        # Build linear system: each flow line is ax + by = c
        # Line through (x0, y0) in direction (dx, dy): normal is (-dy, dx)
        # So: -dy*(X - x0) + dx*(Y - y0) = 0  →  -dy*X + dx*Y = -dy*x0 + dx*y0
        A_col0 = -dy        # coefficient for X (FOE_x)
        A_col1 = dx         # coefficient for Y (FOE_y)
        b_vec = -dy * x0 + dx * y0

        A = np.column_stack([A_col0, A_col1])  # (N, 2)

        # Outlier rejection: RANSAC-style — run least-squares, find inliers, refit.
        foe_pt, inlier_mask = _ransac_line_intersection(A, b_vec, n_iter=30, inlier_thresh_px=15.0)
        if foe_pt is None or inlier_mask.sum() < 4:
            return float("nan"), float("nan"), 0.0

        foe_x, foe_y = float(foe_pt[0]), float(foe_pt[1])

        # Confidence: fraction of inliers × quality-of-fit signal
        inlier_ratio = float(inlier_mask.sum()) / float(len(inlier_mask))

        # Penalize if the FOE is wildly outside the frame (> 2× frame dimensions away)
        cx, cy = frame_w / 2.0, frame_h / 2.0
        dist_from_center = math.hypot(foe_x - cx, foe_y - cy)
        max_dist = math.hypot(frame_w, frame_h)
        distance_penalty = min(1.0, dist_from_center / (2.0 * max_dist))

        confidence = inlier_ratio * (1.0 - distance_penalty)
        return foe_x, foe_y, float(np.clip(confidence, 0.0, 1.0))


# ------------------------------------------------------------------
# Module-level geometry helpers (pure functions, no state)
# ------------------------------------------------------------------

def _circular_std(angles: np.ndarray) -> float:
    """Circular standard deviation of an array of angles (radians)."""
    sin_mean = np.mean(np.sin(angles))
    cos_mean = np.mean(np.cos(angles))
    R = math.sqrt(sin_mean**2 + cos_mean**2)  # mean resultant length
    return float(math.sqrt(-2.0 * math.log(max(R, 1e-9))))


def _ransac_line_intersection(
    A: np.ndarray,
    b: np.ndarray,
    n_iter: int = 30,
    inlier_thresh_px: float = 15.0,
) -> tuple[Optional[np.ndarray], np.ndarray]:
    """RANSAC least-squares intersection of flow lines.

    Each row of A and b represents the equation A[i] @ [foe_x, foe_y] = b[i].
    Returns (best_foe_point, inlier_mask) or (None, empty_mask).
    """
    n = len(b)
    if n < 4:
        return None, np.zeros(n, dtype=bool)

    best_inliers = np.zeros(n, dtype=bool)
    best_count = 0

    rng = np.random.default_rng(42)

    for _ in range(n_iter):
        # Sample 2 lines → solve for intersection
        idx = rng.choice(n, size=2, replace=False)
        A_s, b_s = A[idx], b[idx]
        try:
            pt = np.linalg.solve(A_s, b_s)
        except np.linalg.LinAlgError:
            continue

        # Residuals: distance from each line to this candidate FOE
        residuals = np.abs(A @ pt - b) / np.sqrt(A[:, 0]**2 + A[:, 1]**2 + 1e-9)
        inliers = residuals < inlier_thresh_px

        if inliers.sum() > best_count:
            best_count = int(inliers.sum())
            best_inliers = inliers

    if best_count < 4:
        return None, best_inliers

    # Refit with all inliers
    A_in, b_in = A[best_inliers], b[best_inliers]
    result, _, _, _ = np.linalg.lstsq(A_in, b_in, rcond=None)
    return result, best_inliers
