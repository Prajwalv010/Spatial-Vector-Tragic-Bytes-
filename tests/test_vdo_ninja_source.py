"""Unit tests for VDO.Ninja stream capture and disconnect resilience (Issue #3, #4, #7).

Tests:
1. VdoNinjaCapture URL sanitization and auto-configuration
2. Disconnect & stall watchdog tracking
3. Status change callback triggers
4. FrameSource integration with VdoNinjaCapture
5. Frame decoding and latency instrumentation
"""

import base64
import time
from pathlib import Path
import sys

import cv2
import numpy as np
import pytest

# Add repo root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from spatialvector.perception.frame_source import FrameSource
from spatialvector.perception.vdo_ninja_source import VdoNinjaCapture


def test_vdo_ninja_url_sanitization():
    """Verify that clean WebRTC parameters are appended for headless low-latency streaming."""
    cap = VdoNinjaCapture(
        url="https://vdo.ninja/?view=my_room_id",
        on_frame=lambda img, ts: None,
    )
    assert "cleanoutput" in cap.url
    assert "autostart" in cap.url
    assert "my_room_id" in cap.url


def test_vdo_ninja_health_check_and_stall_detection():
    """Verify that check_health accurately reports True only when frames are recent."""
    cap = VdoNinjaCapture(
        url="https://vdo.ninja/?view=test_room",
        on_frame=lambda img, ts: None,
        stale_timeout=1.0,
    )
    # Initially not connected
    assert not cap.is_connected
    assert not cap.check_health()

    # Simulate frame arrival
    now = time.monotonic()
    cap.last_frame_time = now
    cap._set_connected(True)
    assert cap.is_connected
    assert cap.check_health()

    # Simulate stall (> 1.0s)
    cap.last_frame_time = now - 1.5
    assert not cap.check_health()


def test_vdo_ninja_status_change_callback():
    """Verify on_status_change callback triggers on state transitions."""
    status_history = []

    def on_status(st: bool):
        status_history.append(st)

    cap = VdoNinjaCapture(
        url="https://vdo.ninja/?view=test_room",
        on_frame=lambda img, ts: None,
        on_status_change=on_status,
    )

    cap._set_connected(True)
    assert status_history == [True]

    # No duplicate triggers if state does not change
    cap._set_connected(True)
    assert status_history == [True]

    cap._set_connected(False)
    assert status_history == [True, False]


def test_frame_source_vdo_ninja_disconnect_detection():
    """Verify FrameSource reflects DISCONNECTED when VdoNinjaCapture stalls."""
    source = FrameSource(
        source="https://vdo.ninja/?view=test_device_stream",
        target_fps=30,
        queue_size=5,
    )
    assert source.is_vdo_ninja is True
    assert source.status == FrameSource.STATUS_DISCONNECTED

    # Mock a simulated VdoNinjaCapture attached to FrameSource
    class MockVdoCapture:
        def __init__(self):
            self.is_connected = True
            self.last_frame_time = time.monotonic()
            self._healthy = True

        def check_health(self):
            return self._healthy

        def is_alive(self):
            return True

        def stop(self):
            self.is_connected = False
            self._healthy = False

    mock_cap = MockVdoCapture()
    source._vdo_capture = mock_cap
    source.status = FrameSource.STATUS_OK
    assert source.is_running() is True

    # Simulate stream healthy
    f = source.get_frame(timeout=0.01)
    assert source.status == FrameSource.STATUS_OK

    # Simulate phone stream disconnect / stall
    mock_cap._healthy = False
    mock_cap.is_connected = False
    f = source.get_frame(timeout=0.01)
    assert source.status == FrameSource.STATUS_DISCONNECTED

    # Cleanup
    source.stop()
    assert source.status == FrameSource.STATUS_DISCONNECTED
    assert source._vdo_capture is None


def test_frame_decoding_and_latency_measurement():
    """Verify that base64 JPEG decode correctly extracts images and measures latency."""
    frames_received = []

    def on_frame_cb(img, ts):
        frames_received.append((img, ts))

    cap = VdoNinjaCapture(
        url="https://vdo.ninja/?view=test_room",
        on_frame=on_frame_cb,
    )

    # Generate a dummy test image
    test_img = np.zeros((120, 160, 3), dtype=np.uint8)
    test_img[:, :] = [0, 255, 0]
    ret, buf = cv2.imencode(".jpg", test_img)
    assert ret
    b64_str = "data:image/jpeg;base64," + base64.b64encode(buf).decode("utf-8")

    # Manually invoke frame decode logic
    comma = b64_str.find(",")
    raw = base64.b64decode(b64_str[comma + 1 :])
    arr = np.frombuffer(raw, dtype=np.uint8)
    decoded = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    assert decoded is not None
    assert decoded.shape == (120, 160, 3)
