import collections
import logging
import queue
import threading
import time
from typing import Optional, Union

import cv2
import numpy as np

from .schemas import Frame

logger = logging.getLogger(__name__)


class FrameSource:
    """M01 - Frame Acquisition & Timebase.

    Acquires frames from a webcam or video file on a dedicated thread,
    timestamping with time.monotonic(), maintaining a bounded ring buffer (drop-oldest),
    measuring actual FPS, and handling reconnection/disconnection states.
    """

    STATUS_OK = "OK"
    STATUS_DEGRADED = "DEGRADED"
    STATUS_DISCONNECTED = "DISCONNECTED"

    def __init__(
        self,
        source: Union[int, str] = 0,
        resolution: tuple[int, int] = (1280, 720),
        target_fps: int = 30,
        queue_size: int = 5,
        reconnect_interval: float = 1.0,
        loop_video: bool = False,
    ):
        self.source = source
        self.target_width, self.target_height = resolution
        self.target_fps = target_fps
        self.queue_size = queue_size
        self.reconnect_interval = reconnect_interval
        self.loop_video = loop_video

        # Check if source is a network stream (HTTP/RTSP/UDP)
        self.is_network_stream = isinstance(self.source, str) and any(
            str(self.source).lower().startswith(p) for p in ("http://", "https://", "rtsp://", "udp://")
        )
        self.is_vdo_ninja = isinstance(self.source, str) and "vdo.ninja" in str(self.source).lower()
        self._vdo_capture = None

        # Internal queue: non-blocking, bounded
        self._queue: queue.Queue[Frame] = queue.Queue(maxsize=self.queue_size)

        # Threading state
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        # Diagnostic state
        self.status = self.STATUS_DISCONNECTED
        self.frames_captured = 0
        self.frames_dropped = 0
        self.fps_estimate = 0.0

        # Monotonic time tracking
        self._timestamps = collections.deque(maxlen=30)
        self._last_capture_time: Optional[float] = None
        self._expected_interval = 1.0 / self.target_fps if self.target_fps > 0 else 0.033

        # OpenCV capture
        self._cap: Optional[cv2.VideoCapture] = None

    def _open_capture(self) -> bool:
        """Attempt to open or re-open the video capture device."""
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None

        try:
            if isinstance(self.source, int):
                # On Windows, cv2.CAP_DSHOW provides fast startup for webcams
                cap = cv2.VideoCapture(self.source, cv2.CAP_DSHOW)
                if not cap.isOpened():
                    cap = cv2.VideoCapture(self.source)
            else:
                cap = cv2.VideoCapture(str(self.source))

            if cap.isOpened():
                if isinstance(self.source, int):
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.target_width)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.target_height)
                    if self.target_fps > 0:
                        cap.set(cv2.CAP_PROP_FPS, self.target_fps)
                elif self.is_network_stream:
                    # Live network streams (IP Webcam, DroidCam, RTSP) should keep buffer size minimal
                    # to prevent OpenCV from accumulating a multi-second frame backlog
                    try:
                        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    except Exception:
                        pass
                self._cap = cap
                self.status = self.STATUS_OK
                logger.info(f"Successfully opened video source: {self.source}")
                return True
            else:
                self.status = self.STATUS_DISCONNECTED
                logger.warning(f"Failed to open video source: {self.source}")
                return False
        except Exception as e:
            self.status = self.STATUS_DISCONNECTED
            logger.error(f"Exception opening video source {self.source}: {e}")
            return False

    def _capture_loop(self):
        """Worker thread loop acquiring frames with monotonic timestamps."""
        while not self._stop_event.is_set():
            if self._cap is None or not self._cap.isOpened():
                self.status = self.STATUS_DISCONNECTED
                success = self._open_capture()
                if not success:
                    time.sleep(self.reconnect_interval)
                    continue

            ret, img = self._cap.read()
            t_capture = time.monotonic()

            if not ret or img is None:
                # Handle end of video or camera disconnect
                if isinstance(self.source, str) and self.loop_video:
                    # Video file loop
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                elif isinstance(self.source, str) and not self.loop_video:
                    # End of non-looping video file
                    logger.info("Video file reached EOF.")
                    self.status = self.STATUS_DISCONNECTED
                    break
                else:
                    # Camera read failure / disconnect
                    logger.warning("Camera read returned False (possible disconnect).")
                    self.status = self.STATUS_DISCONNECTED
                    if self._cap:
                        self._cap.release()
                        self._cap = None
                    time.sleep(self.reconnect_interval)
                    continue

            self._process_and_enqueue(img, t_capture)

            # If playing back a local video file, pace according to target_fps so we don't rush through
            if isinstance(self.source, str) and not self.is_network_stream and self.target_fps > 0:
                elapsed = time.monotonic() - t_capture
                sleep_time = (1.0 / self.target_fps) - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self.status = self.STATUS_DISCONNECTED

    def _process_and_enqueue(self, img: np.ndarray, t_capture: float):
        """Process an acquired frame, compute rolling FPS, and put into bounded queue."""
        self.status = self.STATUS_OK
        frame_id = self.frames_captured
        self.frames_captured += 1

        # Check timestamp drift (>3x expected inter-frame interval)
        if self._last_capture_time is not None:
            dt = t_capture - self._last_capture_time
            if dt > (3.0 * self._expected_interval):
                logger.warning(
                    f"Timestamp drift detected: frame interval {dt:.4f}s > 3x expected {self._expected_interval:.4f}s"
                )
        self._last_capture_time = t_capture

        # Compute actual rolling FPS from monotonic timestamps
        self._timestamps.append(t_capture)
        if len(self._timestamps) > 1:
            duration = self._timestamps[-1] - self._timestamps[0]
            if duration > 0:
                self.fps_estimate = (len(self._timestamps) - 1) / duration

        frame = Frame(
            frame_id=frame_id,
            image=img,
            t_capture=t_capture,
            fps_estimate=self.fps_estimate,
        )

        # Bounded queue: drop oldest if full
        while True:
            try:
                self._queue.put_nowait(frame)
                break
            except queue.Full:
                try:
                    _ = self._queue.get_nowait()
                    self.frames_dropped += 1
                    logger.debug(f"Frame queue full; dropped oldest frame. Total dropped: {self.frames_dropped}")
                except queue.Empty:
                    pass

    def start(self) -> "FrameSource":
        """Start the capture thread."""
        if self.is_vdo_ninja:
            if self._vdo_capture is None:
                from .vdo_ninja_source import VdoNinjaCapture

                def _on_vdo_status_change(connected: bool):
                    if connected:
                        self.status = self.STATUS_OK
                    else:
                        self.status = self.STATUS_DISCONNECTED

                self._vdo_capture = VdoNinjaCapture(
                    str(self.source),
                    on_frame=self._process_and_enqueue,
                    on_status_change=_on_vdo_status_change,
                    target_fps=self.target_fps,
                )
                self._vdo_capture.start()
            return self

        if self._thread is not None and self._thread.is_alive():
            return self

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        return self

    def stop(self):
        """Stop the capture thread and release resources."""
        if self._vdo_capture is not None:
            self._vdo_capture.stop()
            self._vdo_capture = None

        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self.status = self.STATUS_DISCONNECTED

    def get_frame(self, timeout: Optional[float] = 0.5) -> Optional[Frame]:
        """Fetch the next available frame from the bounded queue."""
        # Active disconnect tracking for VDO.Ninja
        if self.is_vdo_ninja and self._vdo_capture is not None:
            if not self._vdo_capture.check_health():
                self.status = self.STATUS_DISCONNECTED

        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            if self.is_vdo_ninja and self._vdo_capture is not None and not self._vdo_capture.check_health():
                self.status = self.STATUS_DISCONNECTED
            return None

    def __iter__(self):
        return self

    def __next__(self) -> Frame:
        frame = self.get_frame(timeout=1.0)
        if frame is None:
            if not self.is_running() and self._queue.empty():
                raise StopIteration
        return frame

    def is_running(self) -> bool:
        if self.is_vdo_ninja:
            return self._vdo_capture is not None and self._vdo_capture.is_alive()
        return self._thread is not None and self._thread.is_alive()

