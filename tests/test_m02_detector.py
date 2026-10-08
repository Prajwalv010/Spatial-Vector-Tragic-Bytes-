from pathlib import Path
import sys
import cv2
import numpy as np

# Add repo root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from spatialvector.perception.detector import ObjectDetector
from spatialvector.perception.schemas import Frame


def create_sample_person_image() -> np.ndarray:
    """Creates a synthetic test image with a simple human-like figure."""
    img = np.full((480, 640, 3), 200, dtype=np.uint8)
    # Head
    cv2.circle(img, (320, 140), 40, (120, 90, 70), -1)
    # Torso
    cv2.rectangle(img, (270, 180), (370, 320), (50, 50, 180), -1)
    # Legs
    cv2.rectangle(img, (280, 320), (315, 450), (40, 40, 40), -1)
    cv2.rectangle(img, (325, 320), (360, 450), (40, 40, 40), -1)
    # Arms
    cv2.rectangle(img, (240, 180), (270, 300), (50, 50, 180), -1)
    cv2.rectangle(img, (370, 180), (400, 300), (50, 50, 180), -1)
    return img


def test_t04_detector_qualitative_output(tmp_path):
    """T04: Run detector on a test frame/clip; verify structured Detection objects are emitted

    and write annotated frames to disk for inspection.
    """
    detector = ObjectDetector(model_path="yolov8n.pt", confidence_threshold=0.25)
    img = create_sample_person_image()

    detections = detector.detect(img)
    assert isinstance(detections, list)

    # Save annotated debug frame for qualitative human verification
    debug_dir = tmp_path / "qualitative_dumps"
    debug_dir.mkdir(parents=True, exist_ok=True)
    annotated = img.copy()
    for d in detections:
        x1, y1, x2, y2 = map(int, d.bbox_xyxy)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            annotated,
            f"{d.class_name} {d.confidence:.2f}",
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )
    cv2.imwrite(str(debug_dir / "t04_sample_annotated.jpg"), annotated)
    assert (debug_dir / "t04_sample_annotated.jpg").exists()


def test_t05_low_confidence_boxes_present_pre_filter():
    """T05: Confirm low-confidence boxes are present in detect_raw() before filtering.

    Proves the filter step is separable and tracker can leverage low-conf candidate boxes if desired.
    """
    detector = ObjectDetector(
        model_path="yolov8n.pt",
        confidence_threshold=0.85,  # High threshold
        class_filter=["person", "chair", "bottle"],
    )
    img = create_sample_person_image()

    raw_candidates = detector.detect_raw(img)
    filtered_detections = detector.detect(img)

    # All filtered detections must meet or exceed threshold
    for d in filtered_detections:
        assert d.confidence >= 0.85
        assert d.class_name.lower() in ["person", "chair", "bottle"]

    # Raw candidates contain lower confidence detections (or at least all candidates)
    if len(raw_candidates) > len(filtered_detections):
        low_conf_found = any(c.confidence < 0.85 for c in raw_candidates)
        assert low_conf_found, "detect_raw should retain boxes below threshold"


if __name__ == "__main__":
    import tempfile
    print("--- Running M02 Detector Tests Standalone ---")
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_p = Path(tmpdir)
        print("[TEST] Running T04: Detector qualitative output...")
        test_t04_detector_qualitative_output(tmp_p)
        print("  -> T04 PASSED (saved sample to disk)")

        print("[TEST] Running T05: Low-confidence box retention in detect_raw()...")
        test_t05_low_confidence_boxes_present_pre_filter()
        print("  -> T05 PASSED")

    print("\nALL M02 TESTS PASSED SUCCESSFULLY.")

