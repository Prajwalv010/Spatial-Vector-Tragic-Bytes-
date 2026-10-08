"""
M06 — Motion & Geometry

Pure function module. No hardware access, no threads, no I/O.
Input: Track objects (M03) + MotionState (M05).
Output: ObjectGeometry per active track.

All coordinates normalized to 0..1 so output is resolution-independent.
A 720p and 1080p feed of the same scene must produce identical bearing/velocity values.

Architectural rule: this module contains NO risk logic, NO thresholds for danger,
NO "too close" checks. It outputs geometric quantities only. M07 interprets them.

Failure modes handled:
  - NaN FOE (undefined) → foe_containment = False, geometry_confidence reduced
  - Track bbox jitter → output is clamped; geometry_confidence drops proportionally
  - Young tracks (< 3 frames) → geometry_confidence penalized (velocity not reliable yet)
  - Resolution/FOV change → normalized coordinates absorb it automatically
"""

from __future__ import annotations

import math
import logging
from typing import Optional

import numpy as np

from spatialvector.perception.schemas import Track
from spatialvector.motion.schemas import MotionState, ObjectGeometry

logger = logging.getLogger(__name__)

# FOE containment radius as fraction of frame diagonal (normalized).
# If FOE-to-object distance is within this, foe_containment = True.
# Decided here during build (not a TODO) so M07 has a concrete value to inherit.
_FOE_CONTAINMENT_RADIUS_NORM = 0.15   # 15% of frame diagonal


def compute_object_geometry(
    track: Track,
    motion_state: MotionState,
    frame_w: int,
    frame_h: int,
    foe_containment_radius_norm: float = _FOE_CONTAINMENT_RADIUS_NORM,
) -> ObjectGeometry:
    """Compute geometric quantities for one track given the current MotionState.

    Args:
        track: active Track from M03.
        motion_state: MotionState from M05 for this frame.
        frame_w, frame_h: frame dimensions for normalization.
        foe_containment_radius_norm: normalized radius around FOE for containment check.

    Returns:
        ObjectGeometry with resolution-independent normalized values.
    """
    frame_diag = math.sqrt(frame_w**2 + frame_h**2)

    # ----------------------------------------------------------------
    # 1. Bearing — angle relative to image center, resolution-independent
    # ----------------------------------------------------------------
    if len(track.center_history) == 0:
        # Degenerate track — no position data
        return ObjectGeometry(
            track_id=track.track_id,
            frame_id=track.last_seen_frame_id,
            bearing=0.0,
            relative_image_velocity=(0.0, 0.0),
            motion_vector=(0.0, 0.0),
            foe_containment=False,
            geometry_confidence=0.0,
        )

    cx_frame, cy_frame = frame_w / 2.0, frame_h / 2.0
    obj_cx, obj_cy = track.center_history[-1]

    # Normalize object position to -0.5..0.5
    norm_x = (obj_cx - cx_frame) / frame_w   # -0.5 to 0.5
    norm_y = (obj_cy - cy_frame) / frame_h   # -0.5 to 0.5

    bearing = math.atan2(norm_x, 1.0)   # radians; 0 = dead center, ±π/2 at frame edges

    # ----------------------------------------------------------------
    # 2. Relative image velocity — normalized by frame dimensions
    # ----------------------------------------------------------------
    raw_vx, raw_vy = track.estimated_image_velocity
    norm_vx = raw_vx / frame_w   # fraction of frame width per second
    norm_vy = raw_vy / frame_h   # fraction of frame height per second
    # Clamp to prevent single-frame spikes from propagating to M07
    norm_vx = float(np.clip(norm_vx, -2.0, 2.0))
    norm_vy = float(np.clip(norm_vy, -2.0, 2.0))

    # ----------------------------------------------------------------
    # 3. Motion vector — normalized direction of movement
    # ----------------------------------------------------------------
    vel_mag = math.sqrt(norm_vx**2 + norm_vy**2)
    if vel_mag > 1e-9:
        mv_x = norm_vx / vel_mag
        mv_y = norm_vy / vel_mag
    else:
        mv_x, mv_y = 0.0, 0.0

    # ----------------------------------------------------------------
    # 4. FOE containment — is the object near the travel direction?
    # ----------------------------------------------------------------
    foe_containment = False
    foe_x, foe_y = motion_state.foe_x, motion_state.foe_y
    if not (math.isnan(foe_x) or math.isnan(foe_y)):
        foe_norm_x = (foe_x - cx_frame) / frame_w
        foe_norm_y = (foe_y - cy_frame) / frame_h
        dist_to_foe = math.sqrt((norm_x - foe_norm_x)**2 + (norm_y - foe_norm_y)**2)
        # foe_containment_radius_norm is defined relative to half-frame (0..0.5 range)
        foe_containment = dist_to_foe < foe_containment_radius_norm

    # ----------------------------------------------------------------
    # 5. Proximity scale & expansion rate (Looming cues)
    # ----------------------------------------------------------------
    expansion_rate = getattr(track, "expansion_rate", 0.0)
    proximity_scale = getattr(track, "bbox_scale", 0.0)
    if proximity_scale <= 0.0 and track.bbox_history:
        latest_bbox = track.bbox_history[-1]
        bh = abs(latest_bbox[3] - latest_bbox[1])
        proximity_scale = float(np.clip(bh / max(frame_h, 1.0), 0.0, 1.0))

    # ----------------------------------------------------------------
    # 6. Geometry confidence — aggregate signal quality
    # ----------------------------------------------------------------
    confidence = _compute_confidence(
        track=track,
        flow_quality=motion_state.flow_quality,
        fallback_active=motion_state.fallback_active,
        foe_confidence=motion_state.foe_confidence,   # real confidence signal, not a pixel coord
    )

    return ObjectGeometry(
        track_id=track.track_id,
        frame_id=track.last_seen_frame_id,
        bearing=bearing,
        relative_image_velocity=(norm_vx, norm_vy),
        motion_vector=(mv_x, mv_y),
        foe_containment=foe_containment,
        geometry_confidence=confidence,
        expansion_rate=expansion_rate,
        proximity_scale=proximity_scale,
    )


def compute_geometry_batch(
    tracks: list[Track],
    motion_state: MotionState,
    frame_w: int,
    frame_h: int,
) -> list[ObjectGeometry]:
    """Convenience wrapper — compute ObjectGeometry for all active tracks."""
    return [
        compute_object_geometry(track, motion_state, frame_w, frame_h)
        for track in tracks
    ]


# ------------------------------------------------------------------
# Internal helpers — pure functions
# ------------------------------------------------------------------

def _compute_confidence(
    track: Track,
    flow_quality: float,
    fallback_active: bool,
    foe_confidence: float,
) -> float:
    """Aggregate confidence score 0..1 for ObjectGeometry outputs.

    Penalizes:
      - Young tracks (< 3 frames): velocity estimate unreliable
      - Low flow_quality from M04
      - Fallback mode active (IMU unavailable)
      - High bbox jitter (large variance in recent positions)
      - Low foe_confidence from M04 (FOE poorly estimated)

    foe_confidence floor: 0.3 — a poor FOE estimate reduces but does not eliminate
    confidence for objects that don't rely on FOE alignment. An object clearly moving
    toward the user center can still be meaningful even without a well-defined FOE.
    """
    conf = 1.0

    # Young track penalty
    if track.track_age < 3:
        conf *= 0.4
    elif track.track_age < 6:
        conf *= 0.7

    # Flow quality
    conf *= float(np.clip(flow_quality, 0.0, 1.0))

    # Fallback penalty — we don't know how much rotation remains in flow
    if fallback_active:
        conf *= 0.5

    # Bbox jitter: if center history varies wildly, velocity is noisy
    if len(track.center_history) >= 3:
        centers = np.array(track.center_history[-5:], dtype=np.float64)
        jitter = float(np.std(np.diff(centers, axis=0)))
        # Penalize if avg frame-to-frame jump std is > 30px
        if jitter > 30.0:
            conf *= max(0.2, 1.0 - jitter / 200.0)

    # FOE confidence — clamp and apply with a floor so objects that don't
    # rely on FOE alignment aren't zeroed out entirely.
    # Floor of 0.3: a fully-uncertain FOE reduces confidence to at most 30% of
    # what it would otherwise be; it does not zero it.
    foe_factor = float(np.clip(foe_confidence, 0.0, 1.0))
    foe_factor = max(foe_factor, 0.3)
    conf *= foe_factor

    return float(np.clip(conf, 0.0, 1.0))
