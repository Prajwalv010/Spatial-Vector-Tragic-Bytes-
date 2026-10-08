"""
Motion chain shared data contracts for SpatialVector-HMI.

These schemas define the outputs of M04 (optical flow), M05 (IMU/ego-motion),
and M06 (geometry) — designed to feed directly into M07 (TTC/risk) without refactoring.

Field names here are final. Do not rename without updating M07.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class FlowResult:
    """Output of M04 OpticalFlowEstimator for one frame pair.

    flow_vectors: list of (x0, y0, dx, dy) per tracked point, image-space pixels.
    flow_quality:  0..1 fraction of seed points that survived tracking + outlier rejection.
    foe_x, foe_y: Focus of Expansion in image coordinates. float('nan') if undefined this frame.
    foe_confidence: 0..1 confidence in the FOE estimate; low for pure-rotation or low-texture frames.
    """

    frame_id: int
    timestamp: float                                          # time.monotonic()
    flow_vectors: list[tuple[float, float, float, float]]    # (x0, y0, dx, dy) per point
    flow_quality: float                                       # 0..1
    foe_x: float                                             # image px; NaN if undefined
    foe_y: float                                             # image px; NaN if undefined
    foe_confidence: float                                    # 0..1


@dataclass
class IMUSample:
    """One sample from the MPU-6050 (or simulated IMU) arriving over serial/BLE.

    t_arrival:  time.monotonic() when this sample was received by the laptop.
    t_device:   raw device timestamp if available (seconds); kept for drift analysis only.
                Do NOT assume t_arrival == t_device — they're on different clocks.
    gyro_xyz:   angular velocity in rad/s (x=roll, y=pitch, z=yaw).
    accel_xyz:  optional linear acceleration in m/s² (not used for ego-rotation, kept for future).
    status:     "OK" | "DISCONNECTED" | "INVALID"
    """

    t_arrival: float
    t_device: Optional[float]
    gyro_xyz: tuple[float, float, float]
    accel_xyz: Optional[tuple[float, float, float]] = None
    status: str = "OK"


@dataclass
class MotionState:
    """Output of M05 EgoMotionCompensator for one frame.

    ego_rotation:    rotation rate in rad/s estimated from IMU (roll, pitch, yaw).
                     This is rotation-rate ONLY — NOT position or translation.
    corrected_flow:  flow_vectors with the rotational component subtracted out.
                     When fallback_active is True, this is raw (uncorrected) flow.
    foe_x, foe_y:    FOE carried forward from FlowResult (possibly refined after correction).
    flow_quality:    carried from FlowResult.
    foe_confidence:  0..1 confidence in the FOE estimate; carried directly from
                     FlowResult.foe_confidence without modification. Low for pure-rotation
                     or low-texture frames. M06 uses this to scale geometry_confidence.
    motion_quality:  "OK" when IMU data is fresh and correction applied;
                     "DEGRADED" when IMU absent, stale, or correction untrusted.
    fallback_active: True when running on conservative fallback (raw uncorrected flow).
                     Downstream modules (M07) must treat geometry outputs with more caution
                     when this flag is True — do not assume metric accuracy.
    """

    timestamp: float
    ego_rotation: tuple[float, float, float]                  # rad/s (roll, pitch, yaw)
    corrected_flow: list[tuple[float, float, float, float]]   # (x0, y0, dx, dy)
    foe_x: float
    foe_y: float
    flow_quality: float
    foe_confidence: float                                      # 0..1 carried from FlowResult
    motion_quality: str                                        # "OK" | "DEGRADED"
    fallback_active: bool


@dataclass
class ObjectGeometry:
    """Per-object geometric quantities output by M06, one per active track per frame.

    bearing:                 Angle in radians relative to image center (0 = dead center,
                             positive = right). Computed from normalized bbox center.
                             Resolution-independent (normalized 0..1 before arctan).
    relative_image_velocity: Normalized image-space velocity (vx/W, vy/H) so a 720p and 1080p
                             camera produce identical values for the same physical motion.
    motion_vector:           (dx, dy) normalized direction of object movement in image-space.
    foe_containment:         True if the object's current position or trajectory is within
                             foe_containment_radius_norm of the FOE point (see geometry.py).
                             A cue, not a hard safety call — M07's job to interpret.
    geometry_confidence:     0..1 confidence in these geometry values. Drops when:
                               - track bbox jitter is high
                               - flow_quality is low
                               - track_age is very young (< 3 frames)
    """

    track_id: int
    frame_id: int
    bearing: float                                            # radians relative to image center
    relative_image_velocity: tuple[float, float]             # normalized 0..1
    motion_vector: tuple[float, float]                       # normalized direction
    foe_containment: bool
    geometry_confidence: float                               # 0..1
    expansion_rate: float = 0.0                              # 1/sec rate of scale growth (optical looming)
    proximity_scale: float = 0.0                             # 0..1 fraction of frame height occupied by bbox
