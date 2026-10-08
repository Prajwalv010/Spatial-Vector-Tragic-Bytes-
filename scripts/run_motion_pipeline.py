"""
Gate B — Full Motion Chain Live Demo
M01 → M02 → M03 → M04 → M05 → M06

Runs the complete pipeline and renders:
  - Tracked bounding boxes + IDs (from M03)
  - Sparse flow vectors (from M04, subsampled for readability)
  - FOE point (from M04, carried into M05 MotionState)
  - Per-track bearing and geometry_confidence text (from M06)
  - motion_quality / fallback_active status bar

Usage:
    python scripts/run_motion_pipeline.py               # webcam
    python scripts/run_motion_pipeline.py --source tests/fixtures/test_crossing.mp4
    python scripts/run_motion_pipeline.py --source 0 --no-view   # headless
    python scripts/run_motion_pipeline.py --source video.mp4 --imu-port COM3

Gate B verification:
    Run against a recorded walking clip. Confirm:
    1. flow_quality and motion_quality don't flap wildly frame-to-frame.
    2. If no IMU is connected (default), motion_quality=DEGRADED + fallback_active=True shown.
    3. No crash under normal conditions.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Ensure repo root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from spatialvector.perception.frame_source import FrameSource
from spatialvector.perception.detector import ObjectDetector
from spatialvector.perception.tracker import MultiObjectTracker
from spatialvector.motion.optical_flow import OpticalFlowEstimator
from spatialvector.motion.imu_reader import IMUReader, SimulatedIMUReader
from spatialvector.motion.ego_motion import EgoMotionCompensator
from spatialvector.motion.geometry import compute_geometry_batch


def parse_args():
    p = argparse.ArgumentParser(description="SpatialVector-HMI Gate B — Motion Chain Pipeline")
    p.add_argument("--source", default="0",
                   help="Camera index (default 0) or path to video file")
    p.add_argument("--imu-port", default=None,
                   help="Serial port for IMU (e.g. COM3 or /dev/ttyUSB0). "
                        "Omit to run IMU-less (motion_quality=DEGRADED).")
    p.add_argument("--imu-baud", type=int, default=115200)
    p.add_argument("--sim-imu", action="store_true",
                   help="Use a simulated IMU (zero gyro) for testing without hardware")
    p.add_argument("--sim-imu-disconnect-after", type=float, default=None,
                   help="If using --sim-imu, disconnect after this many seconds (to test fallback)")
    p.add_argument("--no-view", action="store_true",
                   help="Headless mode — no OpenCV window, just console output")
    p.add_argument("--tracker", default="bytetrack",
                   choices=["bytetrack", "botsort"])
    return p.parse_args()


def main():
    args = parse_args()

    # --- Source ---
    source = int(args.source) if args.source.isdigit() else args.source
    is_network = isinstance(source, str) and any(
        source.lower().startswith(p) for p in ("http://", "https://", "rtsp://", "udp://")
    )
    loop_video = isinstance(source, str) and not is_network  # loop recorded files

    # --- M01: Frame source ---
    frame_src = FrameSource(
        source=source,
        target_fps=30.0,
        queue_size=5,
        loop_video=loop_video,
    )

    # --- M02: Detector ---
    detector = ObjectDetector(
        model_path="yolov8n.pt",
        confidence_threshold=0.4,
        class_filter=["person", "bicycle", "car", "motorcycle", "chair"],
    )

    # --- M03: Tracker ---
    tracker = MultiObjectTracker(
        backend=args.tracker,
        history_length=10,
        confidence_threshold=0.4,
    )

    # --- M04: Optical flow ---
    flow_estimator = OpticalFlowEstimator(
        max_corners=200,
        quality_level=0.01,
        min_distance=7.0,
        fb_error_threshold_px=2.0,
        reseed_below_point_count=50,
    )

    # --- M05: IMU reader + ego-motion compensator ---
    imu_reader = None
    if args.imu_port:
        imu_reader = IMUReader(port=args.imu_port, baud_rate=args.imu_baud)
        imu_reader.start()
        print(f"[M05] IMU reader started on {args.imu_port}")
    elif args.sim_imu:
        imu_reader = SimulatedIMUReader(
            gyro_fn=lambda t: (0.0, 0.0, 0.0),
            rate_hz=100.0,
            inject_disconnect_after_s=args.sim_imu_disconnect_after,
        )
        imu_reader.start()
        print("[M05] Simulated IMU started (zero gyro)")
    else:
        print("[M05] No IMU configured — motion_quality will be DEGRADED (fallback mode)")

    compensator = EgoMotionCompensator(
        fallback_flow_quality_threshold=0.3,
        max_timestamp_drift_s=0.05,
    )

    # --- Start frame source ---
    frame_src.start()
    print(f"[M01] Frame source started: {source}")
    print("Press 'q' to quit.\n")

    frame_count = 0
    fps_history: list[float] = []
    t_start = time.monotonic()
    consecutive_timeouts = 0

    try:
        while True:
            frame_obj = frame_src.get_frame(timeout=1.0)
            if frame_obj is None:
                consecutive_timeouts += 1
                if frame_src.is_running() and consecutive_timeouts < 5:
                    # Still warming up camera hardware
                    continue
                print(f"\n[Gate B] No frames received from source '{source}' (status: {frame_src.status}).")
                if isinstance(source, int) or str(source).isdigit():
                    print("[Gate B] Hint: If you do not have a physical webcam connected at index "
                          f"{source}, run with a video fixture:\n"
                          "        python scripts/run_motion_pipeline.py --source tests/fixtures/test_crossing.mp4\n")
                break

            consecutive_timeouts = 0
            img = frame_obj.image
            h, w = img.shape[:2]
            t_frame = frame_obj.t_capture

            # M02: detect (used for display info, M03 runs its own inference)
            # detections = detector.detect(frame_obj.image)  # optional

            # M03: track
            tracks = tracker.track(frame_obj)

            # M04: optical flow
            flow_result = flow_estimator.update(img, frame_obj.frame_id, t_frame)

            # M05: ego-motion compensation
            imu_sample = imu_reader.get_latest() if imu_reader else None
            motion_state = compensator.compensate(
                flow=flow_result,
                imu_sample=imu_sample,
                frame_timestamp=t_frame,
                frame_w=w,
                frame_h=h,
            )

            # M06: geometry for each track
            geometries = compute_geometry_batch(tracks, motion_state, w, h)

            frame_count += 1
            elapsed = time.monotonic() - t_start
            fps = frame_count / max(elapsed, 0.001)

            # ---- Console log ----
            foe_str = (
                f"FOE=({motion_state.foe_x:.0f},{motion_state.foe_y:.0f})"
                if not math.isnan(motion_state.foe_x)
                else "FOE=undefined"
            )
            fallback_str = " FALLBACK" if motion_state.fallback_active else ""
            print(
                f"[{frame_count:4d}] {motion_state.motion_quality}{fallback_str} | "
                f"flow_q={flow_result.flow_quality:.2f} | {foe_str} | "
                f"tracks={len(tracks)} | fps={fps:.1f}",
                end="\r",
            )

            if args.no_view:
                continue

            # ---- Render overlay ----
            canvas = img.copy()
            _draw_flow_vectors(canvas, flow_result.flow_vectors, subsample=10)
            _draw_foe(canvas, motion_state.foe_x, motion_state.foe_y,
                      flow_result.foe_confidence)
            _draw_tracks_and_geometry(canvas, tracks, geometries)
            _draw_status_bar(canvas, motion_state, flow_result, fps, frame_count)

            cv2.imshow("SpatialVector-HMI Gate B", canvas)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    finally:
        frame_src.stop()
        if imu_reader is not None:
            imu_reader.stop()
        cv2.destroyAllWindows()
        print(f"\n[Gate B] Done. Processed {frame_count} frames.")


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

def _draw_flow_vectors(
    canvas: np.ndarray,
    flow_vectors: list[tuple[float, float, float, float]],
    subsample: int = 8,
    scale: float = 3.0,
):
    """Draw a subsampled set of flow vectors as arrows."""
    for i, (x0, y0, dx, dy) in enumerate(flow_vectors):
        if i % subsample != 0:
            continue
        x1 = int(x0 + dx * scale)
        y1 = int(y0 + dy * scale)
        cv2.arrowedLine(
            canvas, (int(x0), int(y0)), (x1, y1),
            color=(0, 200, 255), thickness=1, tipLength=0.3
        )


def _draw_foe(canvas: np.ndarray, foe_x: float, foe_y: float, confidence: float):
    """Draw the Focus of Expansion point (if defined)."""
    if math.isnan(foe_x) or math.isnan(foe_y):
        return
    h, w = canvas.shape[:2]
    px, py = int(np.clip(foe_x, 0, w - 1)), int(np.clip(foe_y, 0, h - 1))
    alpha = int(np.clip(confidence, 0.1, 1.0) * 255)
    color = (0, int(alpha), 255)  # cyan fading by confidence
    cv2.circle(canvas, (px, py), 10, color, 2)
    cv2.drawMarker(canvas, (px, py), color, cv2.MARKER_CROSS, 20, 2)
    cv2.putText(canvas, f"FOE conf={confidence:.2f}", (px + 12, py - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)


def _draw_tracks_and_geometry(canvas, tracks, geometries):
    """Draw bounding boxes, IDs, bearing, and geometry_confidence."""
    geom_by_id = {g.track_id: g for g in geometries}
    for track in tracks:
        if not track.bbox_history:
            continue
        x1, y1, x2, y2 = [int(v) for v in track.bbox_history[-1]]
        color = _track_color(track.track_id)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)

        geom = geom_by_id.get(track.track_id)
        if geom:
            bearing_deg = math.degrees(geom.bearing)
            label = (
                f"ID{track.track_id} | brg={bearing_deg:+.0f}° | "
                f"gc={geom.geometry_confidence:.2f}"
            )
            if geom.foe_containment:
                label += " [FOE]"
        else:
            label = f"ID{track.track_id}"

        cv2.putText(canvas, label, (x1, max(y1 - 6, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        # Draw center trail
        for cx, cy in track.center_history[-8:]:
            cv2.circle(canvas, (int(cx), int(cy)), 2, color, -1)


def _draw_status_bar(canvas, motion_state, flow_result, fps, frame_count):
    """Draw a status bar at the top of the frame."""
    h, w = canvas.shape[:2]
    bar_h = 28
    color = (0, 180, 0) if motion_state.motion_quality == "OK" else (0, 60, 220)
    cv2.rectangle(canvas, (0, 0), (w, bar_h), color, -1)

    fallback_tag = " | FALLBACK ACTIVE" if motion_state.fallback_active else ""
    foe_str = (
        f"FOE=({motion_state.foe_x:.0f},{motion_state.foe_y:.0f})"
        if not math.isnan(motion_state.foe_x) else "FOE=undefined"
    )
    text = (
        f"Frame {frame_count} | {motion_state.motion_quality}{fallback_tag} | "
        f"flow_q={flow_result.flow_quality:.2f} | {foe_str} | FPS={fps:.1f}"
    )
    cv2.putText(canvas, text, (6, 19),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)


def _track_color(track_id: int) -> tuple[int, int, int]:
    """Deterministic color per track ID."""
    palette = [
        (255, 80, 80), (80, 255, 80), (80, 80, 255),
        (255, 200, 0), (0, 200, 255), (200, 0, 255),
        (255, 100, 200), (100, 255, 200),
    ]
    return palette[track_id % len(palette)]


if __name__ == "__main__":
    main()
