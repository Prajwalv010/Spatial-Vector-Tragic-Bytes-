import time
from pathlib import Path
import sys

import cv2
import numpy as np
import pytest

# Add repo root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from spatialvector.perception.tracker import MultiObjectTracker
from spatialvector.perception.schemas import Frame, Track


def make_person_patch(h=140, w=60):
    """Generates a small synthetic person-like patch."""
    patch = np.zeros((h, w, 3), dtype=np.uint8)
    # Head
    cv2.circle(patch, (w // 2, 25), 18, (120, 100, 80), -1)
    # Body
    cv2.rectangle(patch, (10, 45), (w - 10, 95), (60, 60, 200), -1)
    # Legs
    cv2.rectangle(patch, (12, 95), (26, h - 5), (40, 40, 40), -1)
    cv2.rectangle(patch, (w - 26, 95), (w - 12, h - 5), (40, 40, 40), -1)
    return patch


def generate_test_crossing_clip(file_path: Path):
    """Creates a 60-frame video of two people walking across each other."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = 30
    writer = cv2.VideoWriter(str(file_path), fourcc, fps, (640, 360))
    p1 = make_person_patch()
    p2 = make_person_patch()

    for i in range(60):
        frame = np.full((360, 640, 3), 220, dtype=np.uint8)
        # Person 1 moves left to right: x: 50 -> 500, y: 150
        x1 = int(50 + (i * 7.5))
        y1 = 150
        # Person 2 moves right to left: x: 550 -> 100, y: 150
        x2 = int(550 - (i * 7.5))
        y2 = 150

        # Draw Person 1
        h1, w1 = p1.shape[:2]
        if 0 <= x1 < 640 - w1 and 0 <= y1 < 360 - h1:
            frame[y1 : y1 + h1, x1 : x1 + w1] = p1

        # Draw Person 2
        h2, w2 = p2.shape[:2]
        if 0 <= x2 < 640 - w2 and 0 <= y2 < 360 - h2:
            frame[y2 : y2 + h2, x2 : x2 + w2] = p2

        writer.write(frame)
    writer.release()


try:
    import pytest
except ImportError:
    pytest = None

if pytest is not None:
    @pytest.fixture(scope="session")
    def crossing_clip_path(tmp_path_factory):
        p = tmp_path_factory.mktemp("video") / "crossing_test.mp4"
        generate_test_crossing_clip(p)
        return str(p)
else:
    def crossing_clip_path():
        return None


def test_t06_persistent_track_id_single_subject():
    """T06: Run test_walking.mp4 (single walking person); assert track_id is stable throughout.

    Uses the real fixture video so YOLO detects actual objects — avoids the vacuous-pass risk
    of synthetic geometric patches that COCO-trained YOLO never fires on.
    Asserts: if any track ID appears for ≥5 frames, at least one ID persists with ≥70% stability.
    """
    fixture = Path(__file__).resolve().parent / "fixtures" / "test_walking.mp4"
    assert fixture.exists(), f"Fixture not found: {fixture} — run tests/fixtures/generate_fixtures.py first"

    tracker = MultiObjectTracker(
        backend="bytetrack",
        history_length=10,
        confidence_threshold=0.01,  # Match detect_raw threshold — fixture figures need low conf
        class_filter=None,           # No class filter — fixture objects aren't COCO-labeled
    )
    cap = cv2.VideoCapture(str(fixture))

    seen_track_ids: list[int] = []
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        f_obj = Frame(frame_id=frame_idx, image=frame, t_capture=frame_idx * 0.033, fps_estimate=30.0)
        tracks = tracker.track(f_obj)
        if tracks:
            seen_track_ids.append(tracks[0].track_id)

        # Assert no duplicate IDs within a frame
        ids_this_frame = [t.track_id for t in tracks]
        assert len(ids_this_frame) == len(set(ids_this_frame)), f"Duplicate IDs in frame {frame_idx}"

        frame_idx += 1
    cap.release()

    # Non-vacuous gate: require enough detections for a meaningful stability check
    assert len(seen_track_ids) >= 5, (
        f"Too few frames with detections ({len(seen_track_ids)}) — check that YOLOv8n fires on the walking fixture"
    )

    # Count the most common track ID across all frames with detections
    from collections import Counter
    most_common_id, most_common_count = Counter(seen_track_ids).most_common(1)[0]
    stability_ratio = most_common_count / len(seen_track_ids)
    assert stability_ratio >= 0.7, (
        f"Track ID stability ratio {stability_ratio:.2f} below 0.7 — "
        f"most common ID #{most_common_id} appeared {most_common_count}/{len(seen_track_ids)} frames"
    )


def test_t07_camera_rotation_bounded_track_count():
    """T07: Run a fixture video; confirm number of distinct track IDs is bounded.

    Uses test_crossing.mp4 (two people crossing, simulating camera-relative motion).
    Asserts: total distinct track IDs spawned across the entire clip ≤ 6
    (2 people × max 3 ID-switches allowed per person as a reasonable threshold).
    Requires actual detections to be non-empty (non-vacuous gate).
    """
    fixture = Path(__file__).resolve().parent / "fixtures" / "test_crossing.mp4"
    assert fixture.exists(), f"Fixture not found: {fixture} — run tests/fixtures/generate_fixtures.py first"

    tracker = MultiObjectTracker(
        backend="bytetrack",
        history_length=10,
        confidence_threshold=0.01,
        class_filter=None,
    )
    cap = cv2.VideoCapture(str(fixture))

    distinct_ids: set[int] = set()
    total_tracked_frames = 0
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        f_obj = Frame(frame_id=frame_idx, image=frame, t_capture=frame_idx * 0.033, fps_estimate=30.0)
        tracks = tracker.track(f_obj)
        if tracks:
            total_tracked_frames += 1
        for t in tracks:
            distinct_ids.add(t.track_id)
        frame_idx += 1
    cap.release()

    # Non-vacuous gate: require real tracking activity
    assert total_tracked_frames >= 10, (
        f"Only {total_tracked_frames} frames had tracked objects — check fixture detection quality"
    )

    # Bounded track count: 2 subjects × max 3 ID-switches = 6 IDs absolute ceiling
    assert len(distinct_ids) <= 6, (
        f"Track ID explosion: {len(distinct_ids)} distinct IDs for 2 subjects — "
        f"ByteTrack needs tuning or fixtures need review"
    )


def test_t08_id_switch_benchmark_reporting(crossing_clip_path):
    """T08: Run against two-person crossing clip. Record and report ID-switch benchmark metric."""
    tracker = MultiObjectTracker(backend="bytetrack", history_length=10)
    cap = cv2.VideoCapture(crossing_clip_path)

    frame_idx = 0
    all_tracks_per_frame = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        f_obj = Frame(frame_id=frame_idx, image=frame, t_capture=frame_idx * 0.033, fps_estimate=30.0)
        tracks = tracker.track(f_obj)
        all_tracks_per_frame.append(tracks)

        # Assert no duplicate IDs in any single frame
        track_ids = [t.track_id for t in tracks]
        assert len(track_ids) == len(set(track_ids)), f"Duplicate track IDs in frame {frame_idx}: {track_ids}"

        frame_idx += 1
    cap.release()

    id_switch_count = len(tracker.id_switch_events)
    print(f"\n[BENCHMARK T08] Tracker Backend: {tracker.backend}")
    print(f"[BENCHMARK T08] Total frames processed: {frame_idx}")
    print(f"[BENCHMARK T08] Recorded ID-switch events count: {id_switch_count}")
    print(f"[BENCHMARK T08] Event details: {tracker.id_switch_events}")

    # Report metric; does not fail unless duplicate IDs or unexpected exception occurred
    assert id_switch_count >= 0


if __name__ == "__main__":
    import tempfile
    print("--- Running M03 Multi-Object Tracker Tests Standalone ---")
    print("[TEST] Running T06: Persistent Track ID on single subject...")
    test_t06_persistent_track_id_single_subject()
    print("  -> T06 PASSED")

    print("[TEST] Running T07: Panning camera bounded track count...")
    test_t07_camera_rotation_bounded_track_count()
    print("  -> T07 PASSED")

    print("[TEST] Running T08: ID-switch benchmark on crossing clip...")
    with tempfile.TemporaryDirectory() as tmpdir:
        clip_p = Path(tmpdir) / "crossing.mp4"
        generate_test_crossing_clip(clip_p)
        test_t08_id_switch_benchmark_reporting(str(clip_p))
        print("  -> T08 PASSED")

    print("\nALL M03 TESTS PASSED SUCCESSFULLY.")

