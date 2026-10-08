import argparse
import logging
from pathlib import Path
import sys
import time

import cv2
import yaml

# Add repository root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from spatialvector.perception.frame_source import FrameSource
from spatialvector.perception.detector import ObjectDetector
from spatialvector.perception.tracker import MultiObjectTracker

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("run_pipeline")


def main():
    parser = argparse.ArgumentParser(description="Run SpatialVector-HMI M01-M03 Perception Pipeline")
    parser.add_argument("--source", type=str, default=None, help="Video source (index or path to .mp4)")
    parser.add_argument("--tracker", type=str, default=None, choices=["bytetrack", "botsort"], help="Tracking backend")
    parser.add_argument("--no-view", action="store_true", help="Run headlessly without cv2.imshow")
    parser.add_argument("--config", type=str, default=None, help="Path to config yaml")
    args = parser.parse_args()

    # Load configuration
    config_file = args.config or Path(__file__).resolve().parent.parent / "spatialvector" / "config" / "default.yaml"
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    video_source = args.source if args.source is not None else config["camera"]["device_index"]
    try:
        video_source = int(video_source)
    except (ValueError, TypeError):
        video_source = str(video_source)

    is_network = isinstance(video_source, str) and any(
        video_source.lower().startswith(p) for p in ("http://", "https://", "rtsp://", "udp://")
    )

    logger.info(f"Initializing M01 FrameSource (source={video_source})...")
    source = FrameSource(
        source=video_source,
        resolution=tuple(config["camera"]["resolution"]),
        target_fps=config["camera"]["target_fps"],
        queue_size=config["camera"]["queue_size"],
        loop_video=isinstance(video_source, str) and not is_network,
    )

    logger.info(f"Initializing M02 ObjectDetector (model={config['detector']['model_path']})...")
    detector = ObjectDetector(
        model_path=config["detector"]["model_path"],
        confidence_threshold=config["detector"]["confidence_threshold"],
        class_filter=config["detector"]["class_filter"],
    )

    logger.info(f"Initializing M03 MultiObjectTracker (backend={tracker_backend})...")
    tracker = MultiObjectTracker(
        backend=tracker_backend,
        history_length=config["tracker"]["history_length"],
        max_missed_frames=config["tracker"]["max_missed_frames"],
        model_path=config["detector"]["model_path"],
        confidence_threshold=config["detector"]["confidence_threshold"],
        class_filter=config["detector"]["class_filter"],
    )

    source.start()
    logger.info("Perception Pipeline running. Press 'q' in preview window to exit.")

    try:
        while True:
            frame = source.get_frame(timeout=1.0)
            if frame is None:
                if not source.is_running():
                    logger.info("Source stopped. Terminating pipeline.")
                    break
                continue

            # M03 Tracking Update
            tracks = tracker.track(frame)
            active_track_count = len(tracks)

            # Display on-screen preview if not headless
            if not args.no_view:
                annotated_img = frame.image.copy()

                # Overlay HUD metrics
                hud_text_1 = (
                    f"Frame: {frame.frame_id} | Status: {source.status} | "
                    f"FPS: {frame.fps_estimate:.1f} | Dropped: {source.frames_dropped}"
                )
                hud_text_2 = f"Active Tracks: {active_track_count} | Tracker: {tracker.backend.upper()}"

                cv2.putText(annotated_img, hud_text_1, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(annotated_img, hud_text_2, (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                # Draw tracks, bboxes, center histories, and velocity vectors
                for trk in tracks:
                    bx1, by1, bx2, by2 = map(int, trk.bbox_history[-1])
                    tid = trk.track_id
                    cls_name = trk.class_name
                    conf = trk.track_confidence
                    vx, vy = trk.estimated_image_velocity

                    # Bounding box
                    cv2.rectangle(annotated_img, (bx1, by1), (bx2, by2), (0, 200, 255), 2)

                    # Label
                    label = f"ID #{tid} {cls_name} ({conf:.2f})"
                    cv2.putText(annotated_img, label, (bx1, max(20, by1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

                    # Draw center trail history
                    centers = [tuple(map(int, pt)) for pt in trk.center_history]
                    for i in range(1, len(centers)):
                        cv2.line(annotated_img, centers[i - 1], centers[i], (255, 100, 0), 2)

                    # Draw image velocity vector
                    if len(centers) > 0:
                        cx, cy = centers[-1]
                        # Scale velocity vector for visual display (e.g. 0.3s prediction arrow)
                        arrow_end = (int(cx + vx * 0.3), int(cy + vy * 0.3))
                        cv2.arrowedLine(annotated_img, (cx, cy), arrow_end, (0, 0, 255), 2, tipLength=0.25)

                cv2.imshow("SpatialVector-HMI — Perception Pipeline (Gate A)", annotated_img)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:
                    logger.info("User requested exit.")
                    break
            else:
                if frame.frame_id % 30 == 0:
                    logger.info(
                        f"Frame {frame.frame_id} | FPS: {frame.fps_estimate:.1f} | "
                        f"Dropped: {source.frames_dropped} | Tracks: {active_track_count}"
                    )

    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user.")
    finally:
        source.stop()
        if not args.no_view:
            cv2.destroyAllWindows()
        logger.info("Pipeline shutdown cleanly.")


if __name__ == "__main__":
    main()

