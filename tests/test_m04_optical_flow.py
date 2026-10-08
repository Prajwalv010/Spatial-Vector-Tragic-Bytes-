"""
Tests for M04 — OpticalFlowEstimator

T09: Forward-walking synthetic sequence → FOE near scene center, foe_confidence above floor.
T10: Lateral translation (parallel to wall) → should NOT produce high-confidence forward FOE.
T11: Pure camera rotation, static scene → foe_confidence degrades predictably (no crash, no wild FOE).

Runnable with:
    pytest tests/test_m04_optical_flow.py -v -s
    python -u tests/test_m04_optical_flow.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from spatialvector.motion.optical_flow import OpticalFlowEstimator


# ---------------------------------------------------------------------------
# Synthetic scene generators
# ---------------------------------------------------------------------------

def _make_forward_zoom_sequence(n_frames: int = 40, w: int = 640, h: int = 480) -> list[np.ndarray]:
    """Simulate forward walking: a grid of dots that expand outward from center each frame.

    Each frame the dot positions move radially away from the center (cx, cy),
    which means the true FOE is at the image center.
    """
    cx, cy = w / 2.0, h / 2.0
    # Initial dot grid — avoid edges so dots don't leave frame too fast
    base_xs = np.linspace(cx * 0.3, cx * 1.7, 12)
    base_ys = np.linspace(cy * 0.3, cy * 1.7, 10)
    grid_x, grid_y = np.meshgrid(base_xs, base_ys)
    base_pts = np.column_stack([grid_x.ravel(), grid_y.ravel()])

    frames = []
    zoom = 1.0
    zoom_rate = 0.025  # radial expansion per frame
    for i in range(n_frames):
        img = np.ones((h, w, 3), dtype=np.uint8) * 40  # dark gray background
        # Scale points radially from center
        scaled = (base_pts - np.array([cx, cy])) * zoom + np.array([cx, cy])
        for px, py in scaled:
            px_i, py_i = int(round(px)), int(round(py))
            if 3 <= px_i < w - 3 and 3 <= py_i < h - 3:
                cv2.circle(img, (px_i, py_i), 3, (220, 220, 220), -1)
        frames.append(img)
        zoom += zoom_rate
    return frames


def _make_lateral_translation_sequence(n_frames: int = 40, w: int = 640, h: int = 480) -> list[np.ndarray]:
    """Simulate walking parallel to a wall: points slide uniformly to the right.

    Under pure lateral translation the flow field is parallel — there is no FOE
    within the frame (it's at infinity to the left). foe_confidence should be low.
    """
    base_pts = []
    for x in np.linspace(30, w - 30, 15):
        for y in np.linspace(30, h - 30, 12):
            base_pts.append([x, y])
    base_pts = np.array(base_pts, dtype=np.float32)

    frames = []
    shift = 0.0
    shift_rate = 8.0  # px per frame laterally
    for i in range(n_frames):
        img = np.ones((h, w, 3), dtype=np.uint8) * 40
        pts = base_pts.copy()
        pts[:, 0] += shift
        for px, py in pts:
            px_i, py_i = int(round(px)), int(round(py))
            if 3 <= px_i < w - 3 and 3 <= py_i < h - 3:
                cv2.circle(img, (px_i, py_i), 3, (220, 220, 220), -1)
        frames.append(img)
        shift += shift_rate
    return frames


def _make_rotation_sequence(n_frames: int = 40, w: int = 640, h: int = 480) -> list[np.ndarray]:
    """Simulate pure camera rotation: all points rotate around image center.

    Under pure rotation there's no FOE — the expansion point is undefined.
    The module should report low foe_confidence, not produce a high-conf fake FOE.
    """
    cx, cy = w / 2.0, h / 2.0
    base_pts = []
    for x in np.linspace(50, w - 50, 14):
        for y in np.linspace(50, h - 50, 10):
            base_pts.append([x - cx, y - cy])
    base_pts = np.array(base_pts, dtype=np.float32)

    frames = []
    angle = 0.0
    angle_rate = 0.04  # radians per frame (about 2.3°/frame)
    for i in range(n_frames):
        img = np.ones((h, w, 3), dtype=np.uint8) * 40
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        rot_pts = np.column_stack([
            base_pts[:, 0] * cos_a - base_pts[:, 1] * sin_a + cx,
            base_pts[:, 0] * sin_a + base_pts[:, 1] * cos_a + cy,
        ])
        for px, py in rot_pts:
            px_i, py_i = int(round(px)), int(round(py))
            if 3 <= px_i < w - 3 and 3 <= py_i < h - 3:
                cv2.circle(img, (px_i, py_i), 3, (220, 220, 220), -1)
        frames.append(img)
        angle += angle_rate
    return frames


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_t09_forward_walking_foe_near_center():
    """T09: Forward walking → FOE should be near image center with decent confidence."""
    frames = _make_forward_zoom_sequence(n_frames=40)
    h, w = frames[0].shape[:2]
    cx, cy = w / 2.0, h / 2.0

    estimator = OpticalFlowEstimator(
        max_corners=150,
        quality_level=0.01,
        min_distance=5.0,
        fb_error_threshold_px=3.0,
        reseed_below_point_count=20,
    )

    foe_xs, foe_ys, confidences = [], [], []
    for i, frame in enumerate(frames):
        result = estimator.update(frame, frame_id=i, timestamp=i * 0.033)
        if not (math.isnan(result.foe_x) or math.isnan(result.foe_y)):
            if result.foe_confidence > 0.05:  # above noise floor
                foe_xs.append(result.foe_x)
                foe_ys.append(result.foe_y)
                confidences.append(result.foe_confidence)

    # Must have produced at least some usable FOE estimates
    assert len(foe_xs) >= 5, (
        f"T09: Only {len(foe_xs)} frames had a valid FOE estimate — check that the forward zoom sequence "
        f"generates trackable features."
    )

    # Median FOE should be within 20% of frame dimensions from center
    tolerance_x = w * 0.20
    tolerance_y = h * 0.20
    median_foe_x = float(np.median(foe_xs))
    median_foe_y = float(np.median(foe_ys))

    assert abs(median_foe_x - cx) < tolerance_x, (
        f"T09: Median FOE x={median_foe_x:.1f} is {abs(median_foe_x - cx):.1f}px from center "
        f"(tolerance {tolerance_x:.0f}px). True center is x={cx:.0f}."
    )
    assert abs(median_foe_y - cy) < tolerance_y, (
        f"T09: Median FOE y={median_foe_y:.1f} is {abs(median_foe_y - cy):.1f}px from center "
        f"(tolerance {tolerance_y:.0f}px). True center is y={cy:.0f}."
    )

    # Confidence should be above a sane floor for this clean synthetic sequence
    max_confidence = max(confidences)
    assert max_confidence >= 0.10, (
        f"T09: Max foe_confidence={max_confidence:.3f} — expected at least 0.10 for a clean forward-zoom sequence."
    )


def test_t10_lateral_translation_no_false_forward_foe():
    """T10: Lateral walking (parallel translation) → module should NOT report high-confidence FOE.

    Under pure lateral translation, flow vectors are all parallel (no radial expansion),
    so the FOE is undefined or at infinity — foe_confidence should be low.
    """
    frames = _make_lateral_translation_sequence(n_frames=40)

    estimator = OpticalFlowEstimator(
        max_corners=150,
        quality_level=0.01,
        min_distance=5.0,
        fb_error_threshold_px=3.0,
        reseed_below_point_count=20,
    )

    high_confidence_foe_count = 0
    total_valid = 0
    for i, frame in enumerate(frames):
        result = estimator.update(frame, frame_id=i, timestamp=i * 0.033)
        if not (math.isnan(result.foe_x) or math.isnan(result.foe_y)):
            total_valid += 1
            if result.foe_confidence > 0.35:  # "high confidence" threshold
                high_confidence_foe_count += 1

    # Should not produce mostly high-confidence FOEs for a lateral slide
    # Allow a few (flow vectors at edges can create noisy intersection estimates)
    high_conf_fraction = high_confidence_foe_count / max(total_valid, 1)
    assert high_conf_fraction <= 0.4, (
        f"T10: {high_confidence_foe_count}/{total_valid} frames had high-confidence FOE "
        f"({high_conf_fraction:.1%}) during pure lateral translation — should be low."
    )


def test_t11_pure_rotation_degrades_foe_confidence():
    """T11: Pure camera rotation → foe_confidence should degrade, not produce a wild high-conf FOE.

    Under pure rotation, the flow field is circular — no radial expansion point exists.
    The module must not crash and must not emit a high-confidence FOE.
    """
    frames = _make_rotation_sequence(n_frames=40)

    estimator = OpticalFlowEstimator(
        max_corners=150,
        quality_level=0.01,
        min_distance=5.0,
        fb_error_threshold_px=3.0,
        reseed_below_point_count=20,
    )

    high_conf_count = 0
    total_frames = 0
    crashed = False
    for i, frame in enumerate(frames):
        try:
            result = estimator.update(frame, frame_id=i, timestamp=i * 0.033)
        except Exception as exc:
            crashed = True
            pytest.fail(f"T11: OpticalFlowEstimator raised exception during rotation sequence: {exc}")

        total_frames += 1
        if not (math.isnan(result.foe_x) or math.isnan(result.foe_y)):
            if result.foe_confidence > 0.5:
                high_conf_count += 1

    assert not crashed, "T11: Module crashed during pure-rotation sequence"

    # Very few (or none) high-confidence FOE estimates during rotation
    high_conf_fraction = high_conf_count / max(total_frames - 1, 1)  # -1 for bootstrap frame
    assert high_conf_fraction <= 0.2, (
        f"T11: {high_conf_count}/{total_frames} frames produced high-confidence FOE "
        f"({high_conf_fraction:.1%}) during pure rotation — module should degrade, not fabricate a FOE."
    )


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("--- Running M04 Optical Flow Tests Standalone ---")
    print("[TEST] T09: Forward walking FOE estimation...")
    test_t09_forward_walking_foe_near_center()
    print("  -> T09 PASSED")

    print("[TEST] T10: Lateral translation — no false forward FOE...")
    test_t10_lateral_translation_no_false_forward_foe()
    print("  -> T10 PASSED")

    print("[TEST] T11: Pure rotation degrades FOE confidence...")
    test_t11_pure_rotation_degrades_foe_confidence()
    print("  -> T11 PASSED")

    print("\nALL M04 TESTS PASSED.")
