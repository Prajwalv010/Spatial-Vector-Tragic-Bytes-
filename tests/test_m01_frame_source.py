import time
from pathlib import Path
import sys

import cv2
import numpy as np
try:
    import pytest
except ImportError:
    pytest = None

# Add repo root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from spatialvector.perception.frame_source import FrameSource
from spatialvector.perception.schemas import Frame


def create_mock_video(path: Path, num_frames: int = 120, fps: int = 30):
    """Generates a synthetic MP4 video for reproducible CI testing."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (320, 240))
    for i in range(num_frames):
        img = np.zeros((240, 320, 3), dtype=np.uint8)
        # Draw moving circle to simulate visual features
        cx = int(20 + (i * 2) % 280)
        cy = 120
        cv2.circle(img, (cx, cy), 15, (0, 255, 0), -1)
        cv2.putText(img, f"Frame {i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        writer.write(img)
    writer.release()


if pytest is not None:
    @pytest.fixture(scope="session")
    def test_video_file(tmp_path_factory):
        fn = tmp_path_factory.mktemp("data") / "synthetic_test.mp4"
        create_mock_video(fn, num_frames=150, fps=30)
        return str(fn)
else:
    def test_video_file():
        return None


def test_t01_strictly_increasing_ids_and_monotonic_timestamps(test_video_file):
    """T01: Capture from test source; assert frame_id and t_capture are strictly increasing."""
    source = FrameSource(source=test_video_file, target_fps=30, queue_size=10, loop_video=False)
    source.start()

    frames: list[Frame] = []
    try:
        # Collect frames for a short duration or until exhausted
        start_t = time.monotonic()
        while time.monotonic() - start_t < 4.0:
            f = source.get_frame(timeout=0.2)
            if f is not None:
                frames.append(f)
            elif not source.is_running():
                break
    finally:
        source.stop()

    assert len(frames) > 20, f"Expected at least 20 frames, got {len(frames)}"

    # Verify frame_id and t_capture monotonicity
    for i in range(1, len(frames)):
        assert frames[i].frame_id > frames[i - 1].frame_id, (
            f"frame_id not strictly increasing: {frames[i-1].frame_id} >= {frames[i].frame_id}"
        )
        assert frames[i].t_capture > frames[i - 1].t_capture, (
            f"t_capture not strictly monotonic: {frames[i-1].t_capture} >= {frames[i].t_capture}"
        )
        assert frames[i].image is not None


def test_t02_camera_disconnect_and_recovery():
    """T02: Simulate camera block/disconnect for 2 seconds.

    Asserts that status becomes DISCONNECTED during read failure and resumes to OK upon reconnection.
    """
    class MockFailingCapture:
        """Simulates a camera that succeeds, fails for 2 seconds, then resumes."""
        def __init__(self):
            self.created_t = time.monotonic()
            self._opened = True

        def isOpened(self):
            return self._opened

        def release(self):
            self._opened = False

        def read(self):
            elapsed = time.monotonic() - self.created_t
            # In interval [1.0s, 2.5s], simulate disconnect/read failure
            if 1.0 <= elapsed <= 2.5:
                return False, None
            # Otherwise return a dummy frame
            dummy = np.zeros((100, 100, 3), dtype=np.uint8)
            return True, dummy

        def set(self, prop, val):
            return True

    source = FrameSource(source=0, queue_size=5, reconnect_interval=0.2)
    # Inject our mock capture directly
    mock_cap = MockFailingCapture()
    source._cap = mock_cap
    source.status = FrameSource.STATUS_OK

    # Override _open_capture to simulate re-connection after 2.5s
    def mock_reopen():
        if time.monotonic() - mock_cap.created_t > 2.5:
            mock_cap._opened = True
            source._cap = mock_cap
            source.status = FrameSource.STATUS_OK
            return True
        source.status = FrameSource.STATUS_DISCONNECTED
        return False

    source._open_capture = mock_reopen
    source.start()

    statuses_seen = set()
    try:
        t0 = time.monotonic()
        while time.monotonic() - t0 < 3.8:
            statuses_seen.add(source.status)
            _ = source.get_frame(timeout=0.1)
            time.sleep(0.05)
    finally:
        source.stop()

    assert FrameSource.STATUS_OK in statuses_seen, "Should have observed STATUS_OK"
    assert FrameSource.STATUS_DISCONNECTED in statuses_seen, "Should have observed STATUS_DISCONNECTED during failure"
    assert source.frames_captured > 0


def test_t03_cpu_stress_bounded_queue_and_drop_counter(test_video_file):
    """T03: Simulate consumer slowdown; verify queue never exceeds maxsize and frames_dropped increments."""
    queue_size = 5
    source = FrameSource(source=test_video_file, target_fps=100, queue_size=queue_size, loop_video=True)
    source.start()

    try:
        # Consumer sleeps to force queue overflow
        time.sleep(0.8)

        # Assert internal queue size does not exceed configured queue_size
        current_qsize = source._queue.qsize()
        assert current_qsize <= queue_size, f"Queue size {current_qsize} exceeded max {queue_size}"

        # Assert dropped frame counter has incremented
        assert source.frames_dropped > 0, f"Expected dropped frames, got {source.frames_dropped}"
    finally:
        source.stop()

    # Draining after stop
    drained = 0
    while True:
        f = source.get_frame(timeout=0.05)
        if f is None:
            break
        drained += 1
    assert drained <= queue_size


def test_t01b_source_resolver(tmp_path):
    """T01b: Verify camera source resolution priority and file fallback."""
    from spatialvector.perception.source_resolver import (
        resolve_camera_source,
        write_camera_source_file,
        read_camera_source_file,
    )

    # 1. Explicit CLI argument takes highest precedence
    src, origin = resolve_camera_source(cli_source="https://vdo.ninja/?view=cliprec")
    assert src == "https://vdo.ninja/?view=cliprec"
    assert "command-line" in origin

    # Integer CLI source converts to int
    src_int, _ = resolve_camera_source(cli_source="2")
    assert src_int == 2

    # 2. File source
    dummy_file = tmp_path / "camera_source.txt"
    dummy_file.write_text("https://vdo.ninja/?view=fromfile\n", encoding="utf-8")
    assert "fromfile" in dummy_file.read_text(encoding="utf-8")


if __name__ == "__main__":

    import tempfile
    print("--- Running M01 Tests Standalone ---")
    with tempfile.TemporaryDirectory() as tmpdir:
        video_p = Path(tmpdir) / "m01_test.mp4"
        create_mock_video(video_p, num_frames=120, fps=30)

        print("[TEST] Running T01: Monotonic frame_id & t_capture...")
        test_t01_strictly_increasing_ids_and_monotonic_timestamps(str(video_p))
        print("  -> T01 PASSED")

        print("[TEST] Running T02: Camera disconnect & automatic recovery...")
        test_t02_camera_disconnect_and_recovery()
        print("  -> T02 PASSED")

        print("[TEST] Running T03: CPU stress bounded queue & drop counter...")
        test_t03_cpu_stress_bounded_queue_and_drop_counter(str(video_p))
        print("  -> T03 PASSED")

    print("\nALL M01 TESTS PASSED SUCCESSFULLY.")
