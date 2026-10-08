import json
import logging
from pathlib import Path
import sys

import cv2
import yaml

# Add repository root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from spatialvector.perception.detector import ObjectDetector
from spatialvector.perception.frame_source import FrameSource

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("dump_detections")


def main():
    config_path = Path(__file__).resolve().parent.parent / "spatialvector" / "config" / "default.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Use test fixture or webcam
    test_fixture = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "test_crossing.mp4"
    if test_fixture.exists():
        video_src = str(test_fixture)
        logger.info(f"Using test fixture video: {video_src}")
    else:
        from spatialvector.perception.source_resolver import resolve_camera_source
        video_src, origin = resolve_camera_source()
        logger.info(f"Using camera source from {origin}: {video_src}")


    source = FrameSource(
        source=video_src,
        resolution=tuple(config["camera"]["resolution"]),
        target_fps=config["camera"]["target_fps"],
        queue_size=config["camera"]["queue_size"],
        loop_video=False,
    )

    detector = ObjectDetector(
        model_path=config["detector"]["model_path"],
        confidence_threshold=config["detector"]["confidence_threshold"],
        class_filter=config["detector"]["class_filter"],
    )

    output_json_path = Path(__file__).resolve().parent.parent / "detections_dump.json"
    dumped_records = []

    logger.info("Starting M01 -> M02 standalone run (no M03 tracker loaded)...")
    source.start()

    max_frames = 60
    processed = 0

    try:
        while processed < max_frames:
            frame = source.get_frame(timeout=1.0)
            if frame is None:
                if not source.is_running():
                    break
                continue

            detections = detector.detect(frame)
            processed += 1

            frame_record = {
                "frame_id": frame.frame_id,
                "t_capture": frame.t_capture,
                "fps_estimate": frame.fps_estimate,
                "detections": [
                    {
                        "detection_id": d.detection_id,
                        "class_name": d.class_name,
                        "confidence": d.confidence,
                        "bbox_xyxy": d.bbox_xyxy,
                    }
                    for d in detections
                ],
            }
            dumped_records.append(frame_record)

            if processed % 15 == 0:
                logger.info(f"Processed frame {frame.frame_id} | Detections found: {len(detections)}")

    finally:
        source.stop()

    with open(output_json_path, "w") as f:
        json.dump(dumped_records, f, indent=2)

    logger.info(f"Successfully dumped {len(dumped_records)} frame detections to {output_json_path}")
    logger.info(f"Class filter stats: {detector.filtered_out_counts}")
    print(f"DONE: Detections dumped to {output_json_path}")


if __name__ == "__main__":
    main()

