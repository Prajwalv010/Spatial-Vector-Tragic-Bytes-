"""
Tests for M06 — Geometry (ObjectGeometry computation)

T14: Known track positions → computed bearings match hand-expected values within tolerance.
T15: Noisy bounding boxes → outputs remain bounded (no NaN, no crashes, no wild values).

Runnable with:
    pytest tests/test_m06_geometry.py -v -s
    python -u tests/test_m06_geometry.py
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from spatialvector.perception.schemas import Track
from spatialvector.motion.schemas import MotionState
from spatialvector.motion.geometry import compute_object_geometry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FRAME_W, FRAME_H = 640, 480


def _make_track(
    track_id: int,
    centers: list[tuple[float, float]],
    velocity: tuple[float, float] = (0.0, 0.0),
    track_age: int = 10,
) -> Track:
    """Build a Track with given center history."""
    bboxes = [(cx - 20, cy - 40, cx + 20, cy + 40) for cx, cy in centers]
    return Track(
        track_id=track_id,
        class_name="person",
        bbox_history=bboxes,
        center_history=centers,
        estimated_image_velocity=velocity,
        track_age=track_age,
        track_confidence=0.85,
        last_seen_frame_id=42,
    )


def _make_motion_state(
    foe_x: float = float("nan"),
    foe_y: float = float("nan"),
    flow_quality: float = 0.8,
    fallback_active: bool = False,
    foe_confidence: float = 1.0,
) -> MotionState:
    return MotionState(
        timestamp=time.monotonic(),
        ego_rotation=(0.0, 0.0, 0.0),
        corrected_flow=[],
        foe_x=foe_x,
        foe_y=foe_y,
        flow_quality=flow_quality,
        foe_confidence=foe_confidence,
        motion_quality="OK" if not fallback_active else "DEGRADED",
        fallback_active=fallback_active,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_t14_known_bearings():
    """T14: Hand-computed expected bearings for specific track positions.

    Layout (640×480 frame):
      - Dead center (320, 240)       → bearing ≈ 0.0 rad
      - Right edge center (608, 240) → bearing ≈ atan2(0.45, 1) ≈ 0.423 rad (positive)
      - Left edge center (32, 240)   → bearing ≈ atan2(-0.45, 1) ≈ -0.423 rad (negative)
      - Top center (320, 48)         → bearing ≈ 0.0 (dead ahead, above center — bearing is horizontal)
    """
    state = _make_motion_state(foe_x=320.0, foe_y=240.0, flow_quality=0.8)

    cases = [
        # (center_x, center_y, expected_bearing_rad, tolerance_rad, description)
        (320.0, 240.0, 0.0,                     0.05, "dead center"),
        (608.0, 240.0, math.atan2(0.45, 1.0),   0.10, "right edge"),
        (32.0,  240.0, math.atan2(-0.45, 1.0),  0.10, "left edge"),
        (320.0,  48.0, 0.0,                     0.05, "top center"),
    ]

    for cx, cy, expected, tol, desc in cases:
        track = _make_track(
            track_id=1,
            centers=[(cx, cy)],
            velocity=(0.0, 0.0),
            track_age=10,
        )
        geom = compute_object_geometry(track, state, FRAME_W, FRAME_H)

        assert not math.isnan(geom.bearing), f"T14 [{desc}]: bearing is NaN"
        assert abs(geom.bearing - expected) < tol, (
            f"T14 [{desc}]: bearing={geom.bearing:.4f} rad, "
            f"expected≈{expected:.4f} rad (tol={tol:.3f})"
        )


def test_t14b_normalized_velocity_is_resolution_independent():
    """T14b: Same physical velocity at 720p and 1080p must produce the same normalized output."""
    # A track moving at 100px/s horizontally on 640×480
    velocity_640 = (100.0, 0.0)
    # Same physical motion on 1280×960 would be 200px/s
    velocity_1280 = (200.0, 0.0)

    track_640 = _make_track(1, [(320.0, 240.0)], velocity=velocity_640)
    track_1280 = _make_track(2, [(640.0, 480.0)], velocity=velocity_1280)
    state = _make_motion_state(flow_quality=0.8)

    geom_640 = compute_object_geometry(track_640, state, 640, 480)
    geom_1280 = compute_object_geometry(track_1280, state, 1280, 960)

    # relative_image_velocity[0] = vx / frame_w should be identical
    assert abs(geom_640.relative_image_velocity[0] - geom_1280.relative_image_velocity[0]) < 0.01, (
        f"T14b: Normalized velocities differ between resolutions: "
        f"{geom_640.relative_image_velocity[0]:.4f} vs {geom_1280.relative_image_velocity[0]:.4f}"
    )


def test_t14c_foe_containment_logic():
    """T14c: Object at FOE position → containment=True; object far from FOE → containment=False."""
    # FOE at center of frame
    foe_x, foe_y = 320.0, 240.0
    state = _make_motion_state(foe_x=foe_x, foe_y=foe_y, flow_quality=0.8)

    # Track exactly at FOE → containment True
    track_at_foe = _make_track(1, [(foe_x, foe_y)], track_age=10)
    geom_in = compute_object_geometry(track_at_foe, state, FRAME_W, FRAME_H)
    assert geom_in.foe_containment is True, "T14c: Object at FOE should have containment=True"

    # Track at far corner → containment False
    track_far = _make_track(2, [(600.0, 10.0)], track_age=10)
    geom_out = compute_object_geometry(track_far, state, FRAME_W, FRAME_H)
    assert geom_out.foe_containment is False, "T14c: Object far from FOE should have containment=False"

    # NaN FOE → containment always False
    state_nan_foe = _make_motion_state(foe_x=float("nan"), foe_y=float("nan"))
    track_center = _make_track(3, [(320.0, 240.0)], track_age=10)
    geom_nan = compute_object_geometry(track_center, state_nan_foe, FRAME_W, FRAME_H)
    assert geom_nan.foe_containment is False, "T14c: NaN FOE should always give containment=False"


def test_t15_noisy_bbox_outputs_bounded():
    """T15: Perturbed bounding boxes → no NaN, no exceptions, confidence stays in [0, 1].

    This is a robustness test — accuracy is not checked, only stability.
    """
    rng = np.random.default_rng(123)
    state = _make_motion_state(foe_x=320.0, foe_y=240.0, flow_quality=0.6)

    noise_levels = [0.0, 5.0, 20.0, 80.0, 200.0]  # pixel std dev of bbox noise

    for noise_std in noise_levels:
        for trial in range(20):
            # Build a track with noisy center history
            base_cx, base_cy = 320.0, 240.0
            noisy_centers = [
                (
                    float(base_cx + rng.normal(0, noise_std)),
                    float(base_cy + rng.normal(0, noise_std)),
                )
                for _ in range(8)
            ]
            # Noisy velocity
            noisy_vx = float(rng.normal(0, noise_std * 10))
            noisy_vy = float(rng.normal(0, noise_std * 10))

            track = _make_track(
                track_id=trial,
                centers=noisy_centers,
                velocity=(noisy_vx, noisy_vy),
                track_age=max(1, trial % 15),
            )

            try:
                geom = compute_object_geometry(track, state, FRAME_W, FRAME_H)
            except Exception as exc:
                pytest.fail(
                    f"T15: compute_object_geometry raised {type(exc).__name__} with "
                    f"noise_std={noise_std}, trial={trial}: {exc}"
                )

            # No NaN/Inf in any output field
            assert not math.isnan(geom.bearing), f"T15: bearing is NaN at noise_std={noise_std}"
            assert not math.isinf(geom.bearing), f"T15: bearing is Inf at noise_std={noise_std}"
            assert not any(math.isnan(v) for v in geom.relative_image_velocity), \
                f"T15: relative_image_velocity has NaN at noise_std={noise_std}"
            assert not any(math.isnan(v) for v in geom.motion_vector), \
                f"T15: motion_vector has NaN at noise_std={noise_std}"
            assert 0.0 <= geom.geometry_confidence <= 1.0, \
                f"T15: geometry_confidence={geom.geometry_confidence} out of [0,1]"


def test_t15b_empty_track_no_crash():
    """T15b: Track with no center history → should return a zero-confidence ObjectGeometry, not crash."""
    track = Track(
        track_id=99,
        class_name="unknown",
        bbox_history=[],
        center_history=[],
        estimated_image_velocity=(0.0, 0.0),
        track_age=0,
        track_confidence=0.0,
        last_seen_frame_id=0,
    )
    state = _make_motion_state()

    try:
        geom = compute_object_geometry(track, state, FRAME_W, FRAME_H)
    except Exception as exc:
        pytest.fail(f"T15b: crashed on empty track: {exc}")

    assert geom.geometry_confidence == 0.0, "T15b: empty track should have zero confidence"


def test_t15c_foe_confidence_affects_geometry_confidence():
    """T15c (regression): geometry_confidence must be measurably lower when foe_confidence is
    low vs. high, holding all other inputs constant.

    This is the exact class of bug that "did it crash" tests won't catch: the original
    geometry.py accepted foe_confidence but never used it, so confidence was always
    identical regardless of FOE quality. This test guards against that regression.
    """
    track = _make_track(
        track_id=1,
        centers=[(320.0, 240.0), (325.0, 235.0), (330.0, 230.0)],
        velocity=(50.0, -50.0),
        track_age=10,  # mature track, no young-track penalty
    )

    # High FOE confidence — baseline
    state_high_foe = _make_motion_state(
        foe_x=320.0, foe_y=240.0, flow_quality=0.8, foe_confidence=1.0
    )
    # Low FOE confidence — only this changes
    state_low_foe = _make_motion_state(
        foe_x=320.0, foe_y=240.0, flow_quality=0.8, foe_confidence=0.05
    )

    geom_high = compute_object_geometry(track, state_high_foe, FRAME_W, FRAME_H)
    geom_low = compute_object_geometry(track, state_low_foe, FRAME_W, FRAME_H)

    assert geom_high.geometry_confidence > 0.0, "T15c: high-foe conf should be positive"
    assert geom_low.geometry_confidence > 0.0, "T15c: low-foe conf should still be positive (floor applied)"
    assert geom_high.geometry_confidence > geom_low.geometry_confidence, (
        f"T15c: geometry_confidence did NOT drop for low foe_confidence. "
        f"high={geom_high.geometry_confidence:.4f}, low={geom_low.geometry_confidence:.4f} "
        f"(they should differ because _compute_confidence now uses foe_confidence)"
    )
    # Quantitative: the low-foe case should be meaningfully lower (floor=0.3 means ~30% reduction at minimum)
    ratio = geom_low.geometry_confidence / geom_high.geometry_confidence
    assert ratio < 0.95, (
        f"T15c: low foe_confidence only reduced geometry_confidence by {(1-ratio)*100:.1f}%, "
        f"expected at least 5% reduction. Ratio={ratio:.4f}. "
        f"high={geom_high.geometry_confidence:.4f}, low={geom_low.geometry_confidence:.4f}"
    )


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("--- Running M06 Geometry Tests Standalone ---")
    print("[TEST] T14: Known bearing values...")
    test_t14_known_bearings()
    print("  -> T14 PASSED")

    print("[TEST] T14b: Resolution-independent normalized velocity...")
    test_t14b_normalized_velocity_is_resolution_independent()
    print("  -> T14b PASSED")

    print("[TEST] T14c: FOE containment logic...")
    test_t14c_foe_containment_logic()
    print("  -> T14c PASSED")

    print("[TEST] T15: Noisy bbox outputs bounded (robustness)...")
    test_t15_noisy_bbox_outputs_bounded()
    print("  -> T15 PASSED")

    print("[TEST] T15b: Empty track → no crash...")
    test_t15b_empty_track_no_crash()
    print("  -> T15b PASSED")

    print("[TEST] T15c (regression): foe_confidence affects geometry_confidence...")
    test_t15c_foe_confidence_affects_geometry_confidence()
    print("  -> T15c PASSED")

    print("\nALL M06 TESTS PASSED.")
