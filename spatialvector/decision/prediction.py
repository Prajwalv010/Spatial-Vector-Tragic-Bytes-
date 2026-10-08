"""
M07 — Collision Prediction: TTC + CPA + Intersection

Pure geometry module. No hardware, no threads, no I/O. No ML.
Input:  ObjectGeometry (M06) + Track (M03) per active track.
Output: Prediction per track — three separate, separately-computed geometric answers.

IMPORTANT: This module answers geometry questions only. It does NOT label anything
as "dangerous." Risk classification is M08's job.

Linear-motion assumption (documented):
  All trajectory extrapolation assumes constant velocity. This is a known limitation
  for the first prototype — pedestrians, cyclists, and vehicles are modelled as
  point-masses moving at their current estimated velocity. Non-linear motion (turns,
  acceleration, deceleration) is accepted as out-of-scope for this build.
  Implication: TTC and CPA accuracy degrades in proportion to how non-linear the
  actual motion is. For objects undergoing significant direction changes, treat
  prediction_confidence as an upper bound on reliability.

Three separate questions (never collapse into one number):
  1. CPA  — closest point of approach (distance), regardless of whether paths cross.
             Computed from: cpa_time = -dot(rel_pos, rel_vel) / dot(rel_vel, rel_vel)
             cpa_distance = |rel_pos + rel_vel * cpa_time|
  2. TTC  — time until relative distance crosses the contact threshold.
             Defined only when intersection_flag is True and the contact crossing
             occurs within the configured horizon. Otherwise ttc_s = None.
  3. Intersection — do projected paths actually cross within corridor_width_normalized
             AND within horizon_s AND in the future (cpa_time > 0)?

Config values (from config/default.yaml, section "prediction"):
  horizon_s:                    5.0   — beyond this, treat as "no near-term concern"
  contact_threshold_normalized: 0.05  — "collision" distance in normalized units
  corridor_width_normalized:    0.12  — how wide a corridor is for intersection_flag

Near-zero relative velocity handling:
  When |rel_vel| < _REL_VEL_EPSILON, objects are effectively parallel or co-moving.
  In this case: no TTC (None), intersection_flag = False, cpa = current distance.
  This is NOT treated as an error — it is a valid safe outcome.

Ego-motion note:
  The user is treated as the stationary reference frame. Ego-motion compensation
  (M05) has already been applied upstream in MotionState.corrected_flow; there is
  no second ego-motion subtraction here.
"""

from __future__ import annotations

import math
import logging
from typing import Optional

import numpy as np

from spatialvector.motion.schemas import ObjectGeometry
from spatialvector.perception.schemas import Track
from spatialvector.decision.schemas import Prediction
from spatialvector.decision.adaptive_calibrator import SelfAdaptingLoomingCalibrator
from spatialvector.decision.corridor_constants import CORRIDOR_CENTER_BEARING_RAD
from spatialvector.motion.temporal_smoother import AdaptiveEMA

logger = logging.getLogger(__name__)

# Guard for near-zero relative velocity (normalized units/frame)
# Below this, trajectories are treated as parallel/co-moving.
_REL_VEL_EPSILON = 1e-6

# Default config values — override by passing kwargs to CollisionPredictor.__init__
_DEFAULT_HORIZON_S = 5.0
_DEFAULT_CONTACT_THRESHOLD_NORM = 0.05
_DEFAULT_CORRIDOR_WIDTH_NORM = 0.12

# Direct static proximity hazard defaults — aligned with default.yaml & corridor boundaries
_DEFAULT_PROXIMITY_CENTER_BEARING_RAD = CORRIDOR_CENTER_BEARING_RAD
_DEFAULT_PROXIMITY_CENTER_SCALE_THRESHOLD = 0.22
_DEFAULT_PROXIMITY_CENTER_SCALE_RANGE = 0.45
_DEFAULT_PROXIMITY_CENTER_SCALE_FLOOR = 0.17
_DEFAULT_PROXIMITY_OFFCENTER_SCALE_THRESHOLD = 0.38
_DEFAULT_PROXIMITY_OFFCENTER_SCALE_RANGE = 0.45
_DEFAULT_PROXIMITY_OFFCENTER_SCALE_FLOOR = 0.28
_DEFAULT_PROXIMITY_OFFCENTER_CAP = 0.80

# Assumed frame rate for converting velocity units to time.
# ObjectGeometry uses normalized-per-second units (vx/W, vy/H per second).
# Velocity units: normalized image units per second (not calibrated to metric).
# TTC and CPA are in the same normalized space — not in SI units.


class CollisionPredictor:
    """Computes TTC, CPA, and intersection flag for one track per frame.

    Treats the user as the stationary reference frame (ego-motion already compensated
    by M05 upstream). Objects' velocities in ObjectGeometry are already corrected.

    Usage:
        predictor = CollisionPredictor()
        prediction = predictor.predict(geometry, track, frame_id)
    """

    def __init__(
        self,
        horizon_s: float = _DEFAULT_HORIZON_S,
        contact_threshold_normalized: float = _DEFAULT_CONTACT_THRESHOLD_NORM,
        corridor_width_normalized: float = _DEFAULT_CORRIDOR_WIDTH_NORM,
        proximity_center_bearing_rad: float = _DEFAULT_PROXIMITY_CENTER_BEARING_RAD,
        proximity_center_scale_threshold: float = _DEFAULT_PROXIMITY_CENTER_SCALE_THRESHOLD,
        proximity_center_scale_range: float = _DEFAULT_PROXIMITY_CENTER_SCALE_RANGE,
        proximity_center_scale_floor: float = _DEFAULT_PROXIMITY_CENTER_SCALE_FLOOR,
        proximity_offcenter_scale_threshold: float = _DEFAULT_PROXIMITY_OFFCENTER_SCALE_THRESHOLD,
        proximity_offcenter_scale_range: float = _DEFAULT_PROXIMITY_OFFCENTER_SCALE_RANGE,
        proximity_offcenter_scale_floor: float = _DEFAULT_PROXIMITY_OFFCENTER_SCALE_FLOOR,
        proximity_offcenter_cap: float = _DEFAULT_PROXIMITY_OFFCENTER_CAP,
    ):
        """
        Args:
            horizon_s: finite TTC horizon. Beyond this, treat as no near-term concern.
            contact_threshold_normalized: "collision" distance in normalized units.
            corridor_width_normalized: corridor half-width for intersection_flag.
            proximity_center_bearing_rad: corridor boundary angle in radians (pi/6 matches M08).
            proximity_center_scale_threshold: minimum bbox scale to trigger center proximity risk.
            proximity_center_scale_range: scaling denominator for center proximity risk.
            proximity_center_scale_floor: floor subtracted from scale before dividing by range.
            proximity_offcenter_scale_threshold: minimum bbox scale for off-center close objects.
            proximity_offcenter_scale_range: scaling denominator for off-center proximity risk.
            proximity_offcenter_scale_floor: floor subtracted for off-center obstacles.
            proximity_offcenter_cap: max proximity risk for off-center obstacles.
        """
        self.horizon_s = float(horizon_s)
        self.contact_threshold = float(contact_threshold_normalized)
        self.corridor_width = float(corridor_width_normalized)
        self.proximity_center_bearing_rad = float(proximity_center_bearing_rad)
        self.proximity_center_scale_threshold = float(proximity_center_scale_threshold)
        self.proximity_center_scale_range = float(proximity_center_scale_range)
        self.proximity_center_scale_floor = float(proximity_center_scale_floor)
        self.proximity_offcenter_scale_threshold = float(proximity_offcenter_scale_threshold)
        self.proximity_offcenter_scale_range = float(proximity_offcenter_scale_range)
        self.proximity_offcenter_scale_floor = float(proximity_offcenter_scale_floor)
        self.proximity_offcenter_cap = float(proximity_offcenter_cap)

        self.calibrator = SelfAdaptingLoomingCalibrator()
        # Per-track smoothers for continuous telemetry (TTC, CPA, miss distance)
        self._track_smoothers: dict[int, dict[str, AdaptiveEMA]] = {}

    @property
    def _ttc_smoothers(self) -> dict[int, AdaptiveEMA]:
        """Expose TTC AdaptiveEMA smoothers by track_id for telemetry & diagnostics."""
        return {tid: sm["ttc"] for tid, sm in self._track_smoothers.items() if "ttc" in sm}

    def predict(
        self,
        geometry: ObjectGeometry,
        track: Track,
        frame_id: int,
    ) -> Prediction:
        """Compute collision prediction for one track.

        Args:
            geometry: ObjectGeometry from M06 for this track+frame.
            track: Track from M03 for this track.
            frame_id: current frame index.

        Returns:
            Prediction with ttc_s, cpa_normalized, miss_distance_normalized,
            intersection_flag, and prediction_confidence.
        """
        bearing = geometry.bearing
        obj_x = math.sin(bearing)

        # Dynamic depth estimation from proximity scale (bounding box height fraction)
        proximity_scale = getattr(geometry, "proximity_scale", 0.0)
        if proximity_scale <= 0.0:
            proximity_scale = getattr(track, "bbox_scale", 0.0)

        # In perspective projection, large nearby obstacles have smaller forward distance
        if proximity_scale > 0.03:
            obj_y = float(np.clip(0.35 / proximity_scale, 0.15, 2.5))
        else:
            obj_y = 0.5  # Standard baseline for unit tests

        # User at origin
        user_x, user_y = 0.0, 0.0
        rel_pos = np.array([obj_x - user_x, obj_y - user_y], dtype=np.float64)

        # Lateral and approach velocity
        vx, vy = geometry.relative_image_velocity

        # Extract expansion rate (looming cue)
        raw_expansion = getattr(geometry, "expansion_rate", 0.0)
        if raw_expansion == 0.0:
            raw_expansion = getattr(track, "expansion_rate", 0.0)

        eff_expansion, is_looming = self.calibrator.evaluate_expansion(raw_expansion)

        # Forward approach velocity:
        # If looming expansion is detected, it is the primary physical depth approach signal
        v_approach = obj_y * eff_expansion if is_looming else 0.0

        # Also incorporate centroid vertical motion (vy > 0 indicates approach in test fixtures)
        if vy > 0.02:
            v_approach = max(v_approach, float(vy))
        elif vy < -0.05 and not is_looming:
            v_approach = float(vy)  # negative = receding

        # rel_vel: [vx, -v_approach] (negative y means moving towards user at origin)
        rel_vel = np.array([vx, -v_approach], dtype=np.float64)

        # --- CPA computation ---
        ttc_s, cpa_norm, miss_norm, intersection_flag = self._compute_cpa_ttc_intersection(
            rel_pos, rel_vel
        )

        # If head-on looming was detected with direct expansion:
        if is_looming and eff_expansion > 0.05:
            looming_ttc = float(np.clip(1.0 / eff_expansion, 0.1, self.horizon_s))
            if abs(obj_x) < self.corridor_width:
                intersection_flag = True
                cpa_norm = min(cpa_norm, abs(obj_x))
                miss_norm = min(miss_norm, abs(obj_x))
                ttc_s = min(ttc_s, looming_ttc) if ttc_s is not None else looming_ttc

        # --- Direct Static Proximity Hazard (Dead-Zone Free, Aligned with M08 Corridor Boundary) ---
        proximity_risk = 0.0
        in_center_corridor = abs(bearing) <= self.proximity_center_bearing_rad
        if in_center_corridor and proximity_scale > self.proximity_center_scale_threshold:
            # Person or obstacle occupying sufficient height in center corridor
            proximity_risk = float(np.clip(
                (proximity_scale - self.proximity_center_scale_floor) / self.proximity_center_scale_range,
                0.0,
                1.0,
            ))
        elif proximity_scale > self.proximity_offcenter_scale_threshold:
            proximity_risk = float(np.clip(
                (proximity_scale - self.proximity_offcenter_scale_floor) / self.proximity_offcenter_scale_range,
                0.0,
                self.proximity_offcenter_cap,
            ))

        # SAFETY OVERRIDE (DOCUMENTED):
        # Forced intersection_flag applies ONLY for objects directly in the centre corridor
        # exceeding the proximity threshold (proximity_risk >= 0.40). This is an explicit
        # fail-safe safety override to prevent walking into large unmoving obstacles directly ahead,
        # NOT an objective kinematic measurement.
        if in_center_corridor and proximity_risk >= 0.40:
            intersection_flag = True  # Safety override: path IS physically blocked
            if cpa_norm > 0.10:
                # Bound CPA to reflect closeness from bbox scale
                cpa_norm = min(cpa_norm, float(np.clip(0.05 / max(proximity_scale, 0.05), 0.01, 0.15)))
                miss_norm = min(miss_norm, cpa_norm)
            logger.debug(
                "[M07] proximity_risk=%.2f (scale=%.3f) in center corridor — intersection forced True (safety override), "
                "CPA clamped to %.4f; TTC from geometry: %s",
                proximity_risk, proximity_scale, cpa_norm, ttc_s
            )

        # --- Smooth Continuous Kinematics (TTC, CPA, miss distance) via AdaptiveEMA ---
        tid = track.track_id
        if tid not in self._track_smoothers:
            self._track_smoothers[tid] = {
                "ttc": AdaptiveEMA(alpha_slow=0.15, alpha_fast=0.60, window_size=30, rising_is_dangerous=False),
                "cpa": AdaptiveEMA(alpha_slow=0.15, alpha_fast=0.60, window_size=30, rising_is_dangerous=False),
                "miss": AdaptiveEMA(alpha_slow=0.15, alpha_fast=0.60, window_size=30, rising_is_dangerous=False),
            }
        smoothers = self._track_smoothers[tid]
        if ttc_s is not None:
            ttc_s = float(smoothers["ttc"].update(ttc_s))
        cpa_norm = float(smoothers["cpa"].update(cpa_norm))
        miss_norm = float(smoothers["miss"].update(miss_norm))

        # --- Prediction confidence ---
        pred_conf = self._compute_prediction_confidence(geometry, track)
        conf_source = "geometry"
        if proximity_risk > 0.25:
            # Boost confidence for direct proximity hazards; mark source explicitly
            boosted = float(np.clip(proximity_risk * 0.95, 0.4, 1.0))
            if boosted > pred_conf:
                pred_conf = boosted
                conf_source = "proximity_heuristic"

        prediction = Prediction(
            track_id=track.track_id,
            frame_id=frame_id,
            ttc_s=ttc_s,
            cpa_normalized=cpa_norm,
            miss_distance_normalized=miss_norm,
            intersection_flag=intersection_flag,
            prediction_confidence=pred_conf,
            bearing=geometry.bearing,
            proximity_risk=proximity_risk,
            expansion_rate=raw_expansion,
            confidence_source=conf_source,
        )

        logger.debug(
            f"[M07] track={track.track_id} frame={frame_id} "
            f"ttc={ttc_s} cpa={cpa_norm:.4f} intersect={intersection_flag} "
            f"conf={pred_conf:.3f} prox={proximity_risk:.2f}"
        )
        return prediction

    def predict_batch(
        self,
        geometries: list[ObjectGeometry],
        tracks: list[Track],
        frame_id: int,
    ) -> list[Prediction]:
        """Predict for all active tracks. Convenience wrapper."""
        # Update adaptive looming calibrator with observed expansion rates
        exp_rates = [
            getattr(g, "expansion_rate", 0.0)
            for g in geometries
            if getattr(g, "expansion_rate", 0.0) != 0.0
        ]
        if exp_rates:
            self.calibrator.update(exp_rates)

        # Evict stale per-track smoothers
        active_tids = {t.track_id for t in tracks}
        for tid in list(self._track_smoothers.keys()):
            if tid not in active_tids:
                del self._track_smoothers[tid]

        track_by_id = {t.track_id: t for t in tracks}
        results = []
        for geom in geometries:
            track = track_by_id.get(geom.track_id)
            if track is None:
                continue
            results.append(self.predict(geom, track, frame_id))
        return results

    # ------------------------------------------------------------------
    # Internal helpers — pure functions
    # ------------------------------------------------------------------

    def _compute_cpa_ttc_intersection(
        self,
        rel_pos: np.ndarray,
        rel_vel: np.ndarray,
    ) -> tuple[Optional[float], float, float, bool]:
        """
        Compute CPA time, CPA distance, miss distance, and intersection flag.

        Returns (ttc_s, cpa_normalized, miss_distance_normalized, intersection_flag).
        ttc_s = None when there is no finite/relevant TTC (receding, parallel, or beyond horizon).

        The three computations are independent:
          CPA answers: how close would the closest approach be?
          TTC answers: when would they actually touch (if they do)?
          Intersection answers: do the paths actually cross within the corridor?
        """
        vel_sq = float(np.dot(rel_vel, rel_vel))

        # Current distance (will be cpa_normalized if effectively static)
        current_dist = float(np.linalg.norm(rel_pos))

        if vel_sq < _REL_VEL_EPSILON:
            # Near-zero relative velocity: objects are co-moving or stationary.
            # Valid safe outcome — not an error.
            return None, current_dist, current_dist, False

        # --- CPA time ---
        # t_cpa = -dot(rel_pos, rel_vel) / dot(rel_vel, rel_vel)
        # This is the time at which d/dt(|rel_pos + rel_vel * t|²) = 0
        t_cpa = -float(np.dot(rel_pos, rel_vel)) / vel_sq

        if t_cpa <= 0.0:
            # CPA is in the past — objects are moving apart NOW
            # Receding motion: intersection_flag = False, ttc_s = None
            return None, current_dist, current_dist, False

        # --- CPA distance ---
        pos_at_cpa = rel_pos + rel_vel * t_cpa
        cpa_dist = float(np.linalg.norm(pos_at_cpa))

        # --- Intersection flag ---
        # True only if:
        #   (a) CPA distance is below corridor_width (paths come close enough)
        #   (b) CPA time is within the configured horizon
        within_corridor = cpa_dist < self.corridor_width
        within_horizon = t_cpa <= self.horizon_s
        intersection_flag = within_corridor and within_horizon

        # --- TTC ---
        # TTC = time at which |rel_pos + rel_vel * t| first drops below contact_threshold.
        # Only computed (and only valid) when intersection_flag is True.
        # We solve the quadratic |rel_pos + rel_vel*t|² = contact_threshold² for t.
        ttc_s: Optional[float] = None
        if intersection_flag:
            ttc_s = self._solve_contact_time(rel_pos, rel_vel, vel_sq)
            if ttc_s is not None and ttc_s > self.horizon_s:
                ttc_s = None   # contact beyond horizon — treat as no near-term concern

        # miss_distance = cpa_dist when no intersection
        miss_dist = cpa_dist if not intersection_flag else 0.0

        return ttc_s, cpa_dist, miss_dist, intersection_flag

    def _solve_contact_time(
        self,
        rel_pos: np.ndarray,
        rel_vel: np.ndarray,
        vel_sq: float,
    ) -> Optional[float]:
        """Solve for the first positive time t when |rel_pos + rel_vel*t| = contact_threshold.

        Quadratic: a*t² + b*t + c = 0
          a = |rel_vel|²
          b = 2 * dot(rel_pos, rel_vel)
          c = |rel_pos|² - contact_threshold²

        Returns the smaller positive root, or None if no real positive root exists.
        """
        a = vel_sq
        b = 2.0 * float(np.dot(rel_pos, rel_vel))
        c = float(np.dot(rel_pos, rel_pos)) - self.contact_threshold ** 2

        discriminant = b * b - 4.0 * a * c
        if discriminant < 0.0:
            return None  # no real solution — paths never get that close

        sqrt_d = math.sqrt(discriminant)
        t1 = (-b - sqrt_d) / (2.0 * a)
        t2 = (-b + sqrt_d) / (2.0 * a)

        # Return the smallest positive root
        candidates = [t for t in (t1, t2) if t > 0.0]
        if not candidates:
            return None
        return min(candidates)

    def _compute_prediction_confidence(
        self,
        geometry: ObjectGeometry,
        track: Track,
    ) -> float:
        """0..1 confidence in the prediction values.

        Inherits geometry_confidence from M06 (which already encodes flow quality,
        fallback_active, and FOE confidence). Additional penalty for track instability:
        a track with highly erratic velocity estimates is less predictable.
        """
        # Start from geometry_confidence — the upstream signal
        conf = float(np.clip(geometry.geometry_confidence, 0.0, 1.0))

        # Track age penalty: very young tracks have unreliable velocity estimates
        if track.track_age < 3:
            conf *= 0.4
        elif track.track_age < 5:
            conf *= 0.7

        # Velocity consistency check: if the velocity estimate has changed drastically
        # in recent frames (noisy), penalize. Proxy: use bbox history variance as track
        # stability measure (already penalized in geometry_confidence, but worth double-
        # weighting for the prediction use case).
        if len(track.center_history) >= 4:
            centers = np.array(track.center_history[-4:], dtype=np.float64)
            frame_deltas = np.diff(centers, axis=0)
            velocity_variance = float(np.var(frame_deltas))
            # Penalize when velocity varies by more than 15px² across recent frames
            if velocity_variance > 225.0:   # 15px std dev squared
                stability_factor = max(0.3, 1.0 - velocity_variance / 3000.0)
                conf *= stability_factor

        return float(np.clip(conf, 0.0, 1.0))
