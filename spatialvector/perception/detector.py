import logging
import time
from typing import Optional, Union
import numpy as np

from .schemas import Frame, Detection

logger = logging.getLogger(__name__)


class ObjectDetector:
    """M02 - Object Detection.

    Wraps a lightweight YOLO model (nano variant) to detect objects on input Frames.
    Extracts structured Detection objects with monotonic timestamps.
    Separates raw detections from filtered detections, preserving confidence and class metadata.
    Explicitly logs filtered-out classes without discarding them silently.
    Note: Detection confidence is statistical probability of visual presence, NEVER collision risk.
    """

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        confidence_threshold: float = 0.4,
        class_filter: Optional[list[str]] = None,
        device: Optional[str] = None,
    ):
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.class_filter = [c.lower() for c in class_filter] if class_filter else None
        self.device = device

        # Ultralytics model instance
        self._model = None
        self._load_model()

        # Diagnostics / statistics
        self.total_frames_processed = 0
        self.total_detections_produced = 0
        self.filtered_out_counts: dict[str, int] = {}
        self.status = "OK"

    def _load_model(self):
        try:
            from ultralytics import YOLO
            logger.info(f"Loading YOLO model from: {self.model_path}")
            self._model = YOLO(self.model_path)
            self.status = "OK"
        except Exception as e:
            self.status = "DEGRADED"
            logger.error(f"Failed to load YOLO model: {e}")
            raise e

    def detect_raw(self, frame: Union[Frame, np.ndarray]) -> list[Detection]:
        """Perform object detection returning ALL candidate detections before confidence and class filtering."""
        if self._model is None:
            self.status = "DEGRADED"
            return []

        img = frame.image if isinstance(frame, Frame) else frame
        frame_id = frame.frame_id if isinstance(frame, Frame) else self.total_frames_processed
        timestamp = frame.t_capture if isinstance(frame, Frame) else time.monotonic()

        try:
            # Run inference with conf=0.01 to get raw candidate boxes for tracking / recovery / inspection
            results = self._model(
                source=img,
                conf=0.01,
                device=self.device,
                verbose=False,
            )
        except Exception as e:
            logger.error(f"Inference error on frame {frame_id}: {e}")
            self.status = "DEGRADED"
            return []

        self.status = "OK"
        raw_detections: list[Detection] = []
        det_id_counter = 0

        for r in results:
            boxes = r.boxes
            if boxes is None or len(boxes) == 0:
                continue

            for i in range(len(boxes)):
                xyxy = boxes.xyxy[i].cpu().numpy().tolist()
                conf = float(boxes.conf[i].cpu().numpy())
                cls_id = int(boxes.cls[i].cpu().numpy())
                cls_name = str(self._model.names.get(cls_id, f"class_{cls_id}"))

                raw_detections.append(
                    Detection(
                        detection_id=det_id_counter,
                        frame_id=frame_id,
                        class_id=cls_id,
                        class_name=cls_name,
                        bbox_xyxy=(float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])),
                        confidence=conf,
                        timestamp=timestamp,
                    )
                )
                det_id_counter += 1

        return raw_detections

    def detect(self, frame: Union[Frame, np.ndarray]) -> list[Detection]:
        """Runs detector and applies documented confidence and class filtering.
        Filtered-out detections are counted/logged, never silently vanished.
        """
        raw = self.detect_raw(frame)
        self.total_frames_processed += 1

        filtered_detections: list[Detection] = []
        for det in raw:
            # 1. Confidence filter check
            if det.confidence < self.confidence_threshold:
                continue

            # 2. Class filter check
            if self.class_filter is not None:
                if det.class_name.lower() not in self.class_filter:
                    self.filtered_out_counts[det.class_name] = (
                        self.filtered_out_counts.get(det.class_name, 0) + 1
                    )
                    continue

            filtered_detections.append(det)

        self.total_detections_produced += len(filtered_detections)
        return filtered_detections

