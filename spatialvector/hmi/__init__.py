"""SpatialVector-HMI — Modules M10, M11, M12.

M10: Arduino Haptic Interface & Firmware
M11: Local Telemetry Gateway & Phone Dashboard
M12: Session Logger, Replay & Evaluation Harness
"""

from spatialvector.hmi.schemas import (
    ArduinoStatus,
    TelemetryMessage,
    SessionRecord,
    build_track_telemetry,
)

__all__ = [
    "ArduinoStatus",
    "TelemetryMessage",
    "SessionRecord",
    "build_track_telemetry",
]
