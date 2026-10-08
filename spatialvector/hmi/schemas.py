"""Shared data contracts for HMI and Output/Safety-Net Chain (M10, M11, M12).

Modules:
- M10: ArduinoStatus
- M11: TelemetryMessage
- M12: SessionRecord

Pipeline health reporting (M11):
  The `pipeline_health` field in TelemetryMessage MUST be populated by the caller
  with real status queries. Do NOT leave it as the default empty dict in production.
  Use build_pipeline_health() to construct it from live subsystem objects.
  Status values: "OK" | "SIMULATED" | "DISCONNECTED" | "DEGRADED" | "UNKNOWN".
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


@dataclass
class ArduinoStatus:
    """M10 status, polled or pushed from the firmware side."""
    connected: bool = False
    last_ack_t: Optional[float] = None        # time.monotonic() of last ACK received
    last_command_sent: Optional[str] = None   # pattern_id of the last command actually written to serial
    motor_test_result: Optional[Dict[str, str]] = None  # per-motor pass/fail from startup self-test

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TelemetryMessage:
    """M11 output — one JSON message per frame, sent to the phone over WebSocket."""
    session_id: str
    ts: float
    frame_id: int
    tracks: List[Dict[str, Any]] = field(default_factory=list)
    risk_state: Dict[str, Any] = field(default_factory=dict)
    haptic: Dict[str, Any] = field(default_factory=dict)
    pipeline_health: Dict[str, str] = field(default_factory=dict)
    # ^ Must be populated via build_pipeline_health(); empty = not yet reported.
    # Status values: "OK" | "SIMULATED" | "DISCONNECTED" | "DEGRADED" | "UNKNOWN".
    measurement_provenance: Dict[str, str] = field(default_factory=dict)
    # ^ Per-field provenance: "measured" | "estimated" | "unavailable".
    # Safety-critical numbers must carry a provenance entry. Example:
    #   {"pothole_dist_m": "ground_plane_geometry", "imu": "SIMULATED"}
    frame_width: int = 640
    frame_height: int = 480

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def build_track_telemetry(tracks: List[Any], pred_map: Dict[int, Any]) -> List[Dict[str, Any]]:
    """Builds the full per-track telemetry payload — every field the dashboard reads.
    
    Includes:
    - track_id, class_name, bbox ([x1, y1, x2, y2]), track_confidence
    - cpa, ttc_s, intersect (bool), pred_conf, bearing
    - relative_velocity: (vx, vy) image velocity in px/sec
    """
    result = []
    for t in tracks:
        pred = pred_map.get(t.track_id)
        result.append({
            "track_id": t.track_id,
            "class_name": t.class_name,
            "bbox": list(t.bbox_history[-1]) if getattr(t, "bbox_history", None) else None,
            "track_confidence": getattr(t, "track_confidence", 0.0),
            "cpa": getattr(pred, "cpa_normalized", None),
            "ttc_s": getattr(pred, "ttc_s", None),
            "intersect": bool(getattr(pred, "intersection_flag", False)),
            "pred_conf": getattr(pred, "prediction_confidence", None),
            "confidence_source": getattr(pred, "confidence_source", "geometry"),
            "bearing": getattr(pred, "bearing", None),
            "relative_velocity": getattr(t, "estimated_image_velocity", (0.0, 0.0)),
        })
    return result


def build_pipeline_health(
    frame_source=None,   # object with .is_running() -> bool
    imu_reader=None,     # IMUReader or SimulatedIMUReader with .get_latest() -> IMUSample|None
    arduino=None,        # ArduinoInterface with .get_status() -> ArduinoStatus
) -> Dict[str, str]:
    """Build pipeline_health dict from real subsystem status queries.

    Never hardcodes "OK". Each status is derived from a live query:
      camera   — frame_source.is_running()
      imu      — imu_sample.status ("OK" | "SIMULATED" | "DISCONNECTED" | "INVALID")
      arduino  — arduino_status.connected

    Args:
        frame_source: M01 FrameSource (or None if not provided).
        imu_reader:   M05 IMUReader or SimulatedIMUReader (or None).
        arduino:      M10 ArduinoInterface (or None).

    Returns dict with keys "camera", "imu", "arduino".
    """
    health: Dict[str, str] = {}

    # Camera
    if frame_source is not None:
        try:
            running = frame_source.is_running()
            health["camera"] = "OK" if running else "DEGRADED"
        except Exception:
            health["camera"] = "UNKNOWN"
    else:
        health["camera"] = "UNKNOWN"

    # IMU
    if imu_reader is not None:
        try:
            sample = imu_reader.get_latest()
            if sample is None:
                health["imu"] = "UNKNOWN"
            else:
                s = getattr(sample, "status", "UNKNOWN")
                # Pass through: "OK", "SIMULATED", "DISCONNECTED", "INVALID"
                health["imu"] = s
        except Exception:
            health["imu"] = "UNKNOWN"
    else:
        health["imu"] = "UNKNOWN"

    # Arduino
    if arduino is not None:
        try:
            status = arduino.get_status()
            health["arduino"] = "OK" if getattr(status, "connected", False) else "DISCONNECTED"
        except Exception:
            health["arduino"] = "UNKNOWN"
    else:
        health["arduino"] = "UNKNOWN"

    return health


@dataclass
class SessionRecord:
    """M12 — one line in the JSONL session log."""
    session_id: str
    record_type: str  # "header" | "frame" | "detection" | "track" | "motion" | "prediction" | "risk" | "haptic" | "imu"
    ts: float
    frame_id: Optional[int] = None
    payload: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
