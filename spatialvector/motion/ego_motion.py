"""
M05 — Ego-Motion Compensation

Takes raw optical flow (FlowResult from M04) and an IMU sample (IMUSample from imu_reader.py)
and subtracts the rotational flow component caused by camera/body movement, producing a
MotionState with corrected flow that better represents world-object motion.

This module handles ROTATION compensation only — NOT full 6-DOF odometry.
ego_rotation is a rotation-rate estimate (rad/s), NOT a translation or position.

Mandatory fallback behavior (per master prompt, Section 7):
  When motion_quality == "DEGRADED" (IMU absent, stale, or drift too high):
    - corrected_flow = raw flow (passed through unchanged)
    - fallback_active = True
    - ego_rotation = (0.0, 0.0, 0.0)
  Downstream (M06, M07) MUST check fallback_active before trusting flow quality.

Config values driving fallback decisions (all in default.yaml):
  ego_motion.fallback_flow_quality_threshold: 0.3
      Below this flow_quality value, motion_quality → DEGRADED regardless of IMU.
  ego_motion.fallback_caution_widen_factor: 1.5
      Documents how conservative M07 should treat geometry when fallback is active.
      Stored here so the decision is locked during the build, not improvised later.
  imu.max_timestamp_drift_s: 0.05
      If |t_video - t_imu| > this, treat IMU sample as stale.
"""

from __future__ import annotations

import logging
import math
import time
from typing import Optional

import numpy as np

from spatialvector.motion.schemas import FlowResult, IMUSample, MotionState

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Fallback threshold — DECIDED DURING BUILD, per master prompt requirement.
# If flow_quality < this, motion_quality = DEGRADED regardless of IMU health.
# Value 0.3: means if fewer than 30% of seed points survived tracking, we
# don't trust the flow field for ego-motion correction.
# ---------------------------------------------------------------------------
_DEFAULT_FALLBACK_FLOW_QUALITY_THRESHOLD = 0.3

# ---------------------------------------------------------------------------
# Max IMU timestamp drift before treating sample as stale.
# 50ms = ~1.5 frames at 30fps — reasonable for loose serial sync.
# ---------------------------------------------------------------------------
_DEFAULT_MAX_TIMESTAMP_DRIFT_S = 0.05


class EgoMotionCompensator:
    """Subtracts rotational flow caused by camera/body rotation from the raw flow field.

    When motion_quality == "DEGRADED":
      - corrected_flow = raw flow_vectors (no correction applied)
      - fallback_active = True
      - ego_rotation = (0, 0, 0)

    Gyro bias drift: a running mean of recent "near-zero-motion" samples is tracked
    to detect and compensate slow drift, but full Kalman filtering is out of scope here.
    """

    def __init__(
        self,
        fallback_flow_quality_threshold: float = _DEFAULT_FALLBACK_FLOW_QUALITY_THRESHOLD,
        max_timestamp_drift_s: float = _DEFAULT_MAX_TIMESTAMP_DRIFT_S,
        approx_focal_length_px: Optional[float] = None,
    ):
        """
        Args:
            fallback_flow_quality_threshold: below this, motion_quality → DEGRADED.
            max_timestamp_drift_s: max allowed |t_video - t_imu| before IMU is stale.
            approx_focal_length_px: optional focal length for metric rotation subtraction.
                                    None = use image-diagonal heuristic (good enough for demos).
        """
        self.fallback_flow_quality_threshold = fallback_flow_quality_threshold
        self.max_timestamp_drift_s = max_timestamp_drift_s
        self.approx_focal_length_px = approx_focal_length_px

        # Gyro bias estimation — rolling accumulator of low-activity frames
        self._bias_accumulator: list[tuple[float, float, float]] = []
        self._bias: tuple[float, float, float] = (0.0, 0.0, 0.0)
        self._last_imu_t_arrival: Optional[float] = None

    def compensate(
        self,
        flow: FlowResult,
        imu_sample: Optional[IMUSample],
        frame_timestamp: float,
        frame_w: int,
        frame_h: int,
    ) -> MotionState:
        """Produce MotionState from FlowResult + IMUSample.

        Args:
            flow: FlowResult from M04 for this frame.
            imu_sample: most recent IMUSample, or None if IMU is not available.
            frame_timestamp: time.monotonic() of the video frame (from Frame.t_capture).
            frame_w, frame_h: frame dimensions in pixels (needed for rotation projection).

        Returns:
            MotionState with corrected flow and quality flags.
        """
        # --- Determine IMU validity ---
        imu_valid, imu_reason = self._check_imu_validity(imu_sample, frame_timestamp)

        # --- Flow quality gate ---
        flow_ok = flow.flow_quality >= self.fallback_flow_quality_threshold

        if not imu_valid or not flow_ok:
            reason = imu_reason if not imu_valid else f"flow_quality={flow.flow_quality:.2f} below threshold"
            logger.debug(f"[M05] DEGRADED: {reason} — passing raw flow through")
            return MotionState(
                timestamp=frame_timestamp,
                ego_rotation=(0.0, 0.0, 0.0),
                corrected_flow=list(flow.flow_vectors),  # raw, unchanged
                foe_x=flow.foe_x,
                foe_y=flow.foe_y,
                flow_quality=flow.flow_quality,
                foe_confidence=flow.foe_confidence,      # carry forward, even in degraded mode
                motion_quality="DEGRADED",
                fallback_active=True,
            )

        # --- Extract and debias gyro ---
        raw_gyro = imu_sample.gyro_xyz  # type: ignore[union-attr]
        debiased_gyro = (
            raw_gyro[0] - self._bias[0],
            raw_gyro[1] - self._bias[1],
            raw_gyro[2] - self._bias[2],
        )
        self._update_bias(debiased_gyro)

        # --- Subtract rotational flow component ---
        corrected = self._subtract_rotation(
            flow.flow_vectors, debiased_gyro, frame_w, frame_h
        )

        return MotionState(
            timestamp=frame_timestamp,
            ego_rotation=debiased_gyro,
            corrected_flow=corrected,
            foe_x=flow.foe_x,
            foe_y=flow.foe_y,
            flow_quality=flow.flow_quality,
            foe_confidence=flow.foe_confidence,          # carry forward from M04
            motion_quality="OK",
            fallback_active=False,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_imu_validity(
        self, imu_sample: Optional[IMUSample], frame_timestamp: float
    ) -> tuple[bool, str]:
        """Return (valid, reason_string)."""
        if imu_sample is None:
            return False, "IMU sample is None (not started?)"
        if imu_sample.status != "OK":
            return False, f"IMU status={imu_sample.status}"
        drift = abs(frame_timestamp - imu_sample.t_arrival)
        if drift > self.max_timestamp_drift_s:
            logger.warning(
                f"[M05] IMU timestamp drift {drift*1000:.1f}ms > {self.max_timestamp_drift_s*1000:.0f}ms threshold"
            )
            return False, f"IMU timestamp drift {drift*1000:.1f}ms"
        return True, "OK"

    def _subtract_rotation(
        self,
        flow_vectors: list[tuple[float, float, float, float]],
        gyro_xyz: tuple[float, float, float],
        frame_w: int,
        frame_h: int,
    ) -> list[tuple[float, float, float, float]]:
        """Project gyro rotation into expected image-space flow and subtract it.

        For a camera rotating at angular velocity (ωx, ωy, ωz) rad/s, the predicted
        image-space displacement of a point (x, y) relative to image center is:
            Δx ≈ -ωz * f + ωy * (x - cx)
            Δy ≈  ωz * 0 - ωx * (y - cy)
        where f is focal length in pixels and (cx, cy) is principal point.

        We use dt ≈ 1/30 sec (one frame) for the integration — no accumulated state.
        This is a first-order approximation; good enough for short demo clips.
        """
        if not flow_vectors:
            return []

        cx, cy = frame_w / 2.0, frame_h / 2.0
        f = self.approx_focal_length_px or math.sqrt(frame_w**2 + frame_h**2)

        # dt approximation: 1 frame at ~30fps
        dt = 1.0 / 30.0

        gx, gy, gz = gyro_xyz  # rad/s (roll, pitch, yaw)

        corrected: list[tuple[float, float, float, float]] = []
        for x0, y0, dx, dy in flow_vectors:
            # Predicted rotational displacement at this point
            xn = x0 - cx
            yn = y0 - cy
            pred_dx = (-gz * f + gy * xn) * dt
            pred_dy = (gz * 0.0 - gx * yn) * dt

            corr_dx = dx - pred_dx
            corr_dy = dy - pred_dy
            corrected.append((x0, y0, corr_dx, corr_dy))

        return corrected

    def _update_bias(self, gyro: tuple[float, float, float]):
        """Accumulate gyro samples for slow drift estimation.

        Only samples with very low angular velocity count toward bias estimation
        (heuristic: if the user appears nearly stationary, assume gyro should read ~0).
        """
        mag = math.sqrt(gyro[0]**2 + gyro[1]**2 + gyro[2]**2)
        if mag < 0.02:  # rad/s — threshold for "nearly stationary"
            self._bias_accumulator.append(gyro)
            if len(self._bias_accumulator) > 200:
                self._bias_accumulator.pop(0)
            if len(self._bias_accumulator) >= 30:
                arr = np.array(self._bias_accumulator)
                mean = arr.mean(axis=0)
                self._bias = (float(mean[0]), float(mean[1]), float(mean[2]))
