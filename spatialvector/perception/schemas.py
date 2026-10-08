from dataclasses import dataclass
import numpy as np


@dataclass
class Frame:
    frame_id: int
    image: np.ndarray          # BGR, as returned by OpenCV
    t_capture: float           # monotonic clock, seconds
    fps_estimate: float


@dataclass
class Detection:
    detection_id: int
    frame_id: int
    class_id: int
    class_name: str
    bbox_xyxy: tuple[float, float, float, float]
    confidence: float
    timestamp: float


@dataclass
class Track:
    track_id: int
    class_name: str
    bbox_history: list[tuple[float, float, float, float]]   # most recent last
    center_history: list[tuple[float, float]]
    estimated_image_velocity: tuple[float, float]            # px/sec, image-space
    track_age: int             # frames since first seen
    track_confidence: float
    last_seen_frame_id: int
    expansion_rate: float = 0.0                             # 1/sec rate of scale change (looming cue)
    bbox_scale: float = 0.0                                 # 0..1 fraction of frame height occupied by bbox

