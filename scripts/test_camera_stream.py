"""
Standalone Camera & Stream Diagnostic Utility
Test any video source: local webcam (0, 1), DroidCam, or network stream (HTTP/RTSP/IP Webcam).

Usage:
    # 1. Local or DroidCam Virtual Webcam (index 0, 1, 2, etc.)
    python scripts/test_camera_stream.py --source 0
    python scripts/test_camera_stream.py --source 1

    # 2. IP Webcam (Android app streaming over WiFi)
    python scripts/test_camera_stream.py --source "http://192.168.1.10:8080/video"

    # 3. DroidCam Direct Stream (without desktop client)
    python scripts/test_camera_stream.py --source "http://192.168.1.10:4747/video"

    # 4. Headless mode (no OpenCV GUI window, prints live telemetry)
    python scripts/test_camera_stream.py --source "http://192.168.1.10:8080/video" --no-view
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from spatialvector.perception.frame_source import FrameSource


def parse_args():
    p = argparse.ArgumentParser(description="Test video source (Webcam / DroidCam / IP Stream)")
    p.add_argument(
        "--source",
        default="0",
        help="Device index (e.g. 0, 1) or stream URL (e.g. http://192.168.1.x:8080/video)",
    )
    p.add_argument(
        "--no-view",
        action="store_true",
        help="Run in console-only mode without opening cv2.imshow GUI window",
    )
    p.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Optional maximum frames to capture before exiting (0 = run until 'q' or Ctrl+C)",
    )
    return p.parse_args()


def main():
    args = parse_args()

    # Convert digit string to int
    raw_source = args.source
    source = int(raw_source) if raw_source.isdigit() else raw_source

    print("=" * 65)
    print(f"  SPATIALVECTOR-HMI CAMERA / STREAM TEST HARNESS")
    print("=" * 65)
    print(f"  Source       : {source} (type: {type(source).__name__})")
    print(f"  GUI View     : {'Disabled (--no-view)' if args.no_view else 'Enabled'}")
    print(f"  Controls     : Press 'q' on video window or Ctrl+C in terminal to stop")
    print("=" * 65)

    frame_src = FrameSource(source=source, target_fps=30, queue_size=5)
    frame_src.start()

    time.sleep(0.5)  # brief startup delay for capture thread

    frames_received = 0
    t_start = time.monotonic()
    last_stat_time = t_start

    try:
        if not args.no_view:
            import cv2
        else:
            cv2 = None
    except ImportError:
        cv2 = None
        args.no_view = True
        print("[INFO] OpenCV GUI not available or cv2 missing. Running in console mode.")

    try:
        while True:
            frame_obj = frame_src.get_frame(timeout=2.0)
            now = time.monotonic()

            if frame_obj is None:
                status = frame_src.status
                print(f"[WAIT] Waiting for frames from '{source}' (source status: {status})...")
                time.sleep(0.5)
                continue

            frames_received += 1
            img = frame_obj.image
            h, w = img.shape[:2]

            # Print telemetry every ~1.0 second
            if now - last_stat_time >= 1.0:
                dt = now - t_start
                avg_fps = frames_received / dt if dt > 0 else 0.0
                print(
                    f"[{now - t_start:6.1f}s] Frame #{frame_obj.frame_id:05d} | "
                    f"Res: {w}x{h} | "
                    f"Est FPS: {frame_obj.fps_estimate:5.1f} | "
                    f"Avg FPS: {avg_fps:5.1f} | "
                    f"Dropped: {frame_src.frames_dropped} | "
                    f"Status: {frame_src.status}"
                )
                last_stat_time = now

            if cv2 is not None and not args.no_view:
                # Render diagnostics banner
                cv2.putText(
                    img,
                    f"SRC: {source} | {w}x{h} | {frame_obj.fps_estimate:.1f} FPS",
                    (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )
                cv2.imshow("SpatialVector Camera Stream Test (Press 'q' to quit)", img)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:
                    print("\n[INFO] User pressed 'q' — exiting.")
                    break

            if args.max_frames > 0 and frames_received >= args.max_frames:
                print(f"\n[INFO] Reached max frames limit ({args.max_frames}) — exiting.")
                break

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
    finally:
        frame_src.stop()
        if cv2 is not None and not args.no_view:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
        total_dt = time.monotonic() - t_start
        fps_overall = frames_received / total_dt if total_dt > 0 else 0.0
        print("-" * 65)
        print(f"Total Frames Received : {frames_received}")
        print(f"Total Frames Dropped  : {frame_src.frames_dropped}")
        print(f"Session Duration      : {total_dt:.2f}s")
        print(f"Overall Average FPS   : {fps_overall:.2f}")
        print("-" * 65)


if __name__ == "__main__":
    main()
