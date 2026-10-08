"""
Tests for M07 — Collision Prediction

T16: Head-on approach → TTC values decrease monotonically across frames.
T17: Parallel motion  → intersection_flag=False for the whole sequence.
T18: Crossing trajectory → finite TTC produced, CPA below corridor width.
T19: Receding motion  → intersection_flag=False AND ttc_s=None explicitly.

All tests use purely synthetic ObjectGeometry + Track fixtures.
No camera, no YOLO, no live hardware required.

Gate C verification: run these tests without any live pipeline components.

Runnable with:
    pytest tests/test_m07_prediction.py -v -s
    python -u tests/test_m07_prediction.py
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

from spatialvector.motion.schemas import ObjectGeometry
from spatialvector.perception.schemas import Track
from spatialvector.decision.prediction import CollisionPredictor


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

FRAME_W, FRAME_H = 640, 480


def _make_track(
    track_id: int,
    centers: list[tuple[float, float]],
    velocity: tuple[float, float] = (0.0, 0.0),
    track_age: int = 10,
) -> Track:
    bboxes = [(cx - 20, cy - 40, cx + 20, cy + 40) for cx, cy in centers]
    return Track(
        track_id=track_id,
        class_name="person",
        bbox_history=bboxes,
        center_history=centers,
        estimated_image_velocity=velocity,
        track_age=track_age,
        track_confidence=0.85,
        last_seen_frame_id=0,
    )


def _make_geometry(
    track_id: int,
    bearing: float,
    vx_norm: float,
    vy_norm: float,
    foe_containment: bool = False,
    geometry_confidence: float = 0.85,
    frame_id: int = 0,
) -> ObjectGeometry:
    """Build a synthetic ObjectGeometry directly, bypassing M06."""
    vel_mag = math.sqrt(vx_norm ** 2 + vy_norm ** 2)
    if vel_mag > 1e-9:
        mv = (vx_norm / vel_mag, vy_norm / vel_mag)
    else:
        mv = (0.0, 0.0)
    return ObjectGeometry(
        track_id=track_id,
        frame_id=frame_id,
        bearing=bearing,
        relative_image_velocity=(vx_norm, vy_norm),
        motion_vector=mv,
        foe_containment=foe_containment,
        geometry_confidence=geometry_confidence,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_t16_head_on_approach_ttc_decreases():
    """T16: Synthetic object approaching head-on → TTC decreases monotonically.

    Setup: object starts at bearing=0 (dead center), moving toward camera
    (positive vy in image = downward = approaching in our depth model).
    At each synthetic frame, the object is "closer" — TTC should shrink.
    """
    predictor = CollisionPredictor(
        horizon_s=5.0,
        contact_threshold_normalized=0.05,
        corridor_width_normalized=0.15,
    )

    # Simulate 8 frames of approach: object moving toward camera at constant rate.
    # In M07's frame: vy > 0 in image → approaching (vy_norm = positive).
    # We keep bearing=0 (dead center, intersection guaranteed) and increase apparent size
    # by increasing velocity slightly each frame to represent constant-approach.
    approach_velocity = 0.20   # normalized units/second, toward camera

    ttc_values = []
    for frame_id in range(8):
        # Each frame: object is "closer" — represent this by a slightly higher velocity
        # (in a real pipeline, the image velocity grows as the object fills more of the frame)
        current_v = approach_velocity * (1.0 + frame_id * 0.05)

        centers = [(320.0, float(240 + frame_id * 10))]  # moving down in image = approaching
        track = _make_track(1, centers, velocity=(0.0, current_v * FRAME_H), track_age=10 + frame_id)

        geom = _make_geometry(
            track_id=1,
            bearing=0.0,         # dead center
            vx_norm=0.0,
            vy_norm=current_v,   # positive vy = approaching
            foe_containment=True,
            geometry_confidence=0.9,
            frame_id=frame_id,
        )

        pred = predictor.predict(geom, track, frame_id)
        assert pred.track_id == 1
        assert pred.frame_id == frame_id

        if pred.ttc_s is not None:
            ttc_values.append(pred.ttc_s)

    # Must have produced at least some finite TTC values for a head-on approach
    assert len(ttc_values) >= 4, (
        f"T16: Expected at least 4 finite TTC values for head-on approach, got {len(ttc_values)}. "
        f"Values: {ttc_values}"
    )

    # TTC must be monotonically decreasing (or at least non-increasing with tolerance)
    for i in range(len(ttc_values) - 1):
        assert ttc_values[i] >= ttc_values[i + 1] - 0.01, (
            f"T16: TTC not monotonically decreasing at step {i}: "
            f"{ttc_values[i]:.4f} → {ttc_values[i+1]:.4f}. Full sequence: {ttc_values}"
        )
    print(f"  T16: TTC sequence = {[f'{v:.3f}s' for v in ttc_values]}")


def test_t17_parallel_motion_no_intersection():
    """T17: Object moving parallel to user path → intersection_flag=False for all frames.

    Setup: object at bearing=-0.4 rad (to the left), moving purely horizontally
    (vx_norm nonzero, vy_norm = 0.0). It's moving alongside the user, not toward them.
    Regardless of how close it gets laterally, paths never cross.
    """
    predictor = CollisionPredictor(
        horizon_s=5.0,
        contact_threshold_normalized=0.05,
        corridor_width_normalized=0.15,
    )

    # Object moving left-to-right alongside user — parallel, not toward
    parallel_speed = 0.15   # normalized units/second, horizontal

    for frame_id in range(12):
        # Object drifts horizontally, stays at same "depth" (vy_norm = 0)
        bearing_offset = math.atan2(-0.3 + frame_id * 0.02, 1.0)  # drifting slightly
        centers = [(float(200 + frame_id * 8), 240.0)]
        track = _make_track(1, centers, velocity=(parallel_speed * FRAME_W, 0.0), track_age=10)

        geom = _make_geometry(
            track_id=1,
            bearing=bearing_offset,
            vx_norm=parallel_speed,   # moving horizontally
            vy_norm=0.0,               # no depth change
            foe_containment=False,
            geometry_confidence=0.85,
            frame_id=frame_id,
        )

        pred = predictor.predict(geom, track, frame_id)

        assert pred.intersection_flag is False, (
            f"T17 [frame {frame_id}]: Expected intersection_flag=False for parallel motion, "
            f"got True. bearing={bearing_offset:.3f}, vx={parallel_speed:.3f}, vy=0.0"
        )

    print("  T17: intersection_flag=False for all 12 parallel frames. PASSED.")


def test_t18_crossing_trajectory_produces_finite_ttc():
    """T18: Object on genuine crossing trajectory → finite TTC, small CPA distance.

    Setup: object at bearing=+0.3 rad (right of center), moving toward center
    (negative vx to move left, and positive vy to approach). Path will cross user's.
    """
    predictor = CollisionPredictor(
        horizon_s=5.0,
        contact_threshold_normalized=0.05,
        corridor_width_normalized=0.15,
    )

    # Object crosses from right-center toward center of frame while approaching
    geom = _make_geometry(
        track_id=2,
        bearing=0.3,              # ~17 degrees right of center
        vx_norm=-0.08,            # moving leftward (toward center)
        vy_norm=0.12,             # approaching (positive vy = depth decrease)
        foe_containment=True,
        geometry_confidence=0.88,
        frame_id=0,
    )
    track = _make_track(2, [(360.0, 200.0), (355.0, 210.0)], velocity=(-0.08 * FRAME_W, 0.12 * FRAME_H))

    pred = predictor.predict(geom, track, 0)

    # CPA should be small (object crosses the path)
    assert pred.cpa_normalized < predictor.corridor_width, (
        f"T18: Expected CPA < corridor_width ({predictor.corridor_width:.3f}), "
        f"got cpa={pred.cpa_normalized:.4f}"
    )

    # intersection_flag should be True
    assert pred.intersection_flag is True, (
        f"T18: Expected intersection_flag=True for crossing trajectory. "
        f"cpa={pred.cpa_normalized:.4f}, corridor_width={predictor.corridor_width:.3f}"
    )

    # Should have a finite TTC within the horizon
    assert pred.ttc_s is not None, (
        f"T18: Expected finite ttc_s for crossing trajectory, got None. "
        f"intersection_flag={pred.intersection_flag}, cpa={pred.cpa_normalized:.4f}"
    )
    assert pred.ttc_s > 0.0, f"T18: ttc_s must be positive, got {pred.ttc_s}"
    assert pred.ttc_s <= predictor.horizon_s, (
        f"T18: ttc_s={pred.ttc_s:.3f}s exceeds horizon {predictor.horizon_s}s"
    )

    print(f"  T18: intersection_flag={pred.intersection_flag}, "
          f"ttc={pred.ttc_s:.3f}s, cpa={pred.cpa_normalized:.4f}. PASSED.")


def test_t19_receding_object_no_ttc():
    """T19: Object moving away → intersection_flag=False AND ttc_s=None.

    These must be explicit sentinel values, not defaulted-safe numbers that look
    like real measurements. A receding object should produce NO collision signal.
    """
    predictor = CollisionPredictor(
        horizon_s=5.0,
        contact_threshold_normalized=0.05,
        corridor_width_normalized=0.15,
    )

    # Object moving away: vy_norm < 0 (negative vy = receding in our depth convention)
    for frame_id in range(6):
        geom = _make_geometry(
            track_id=3,
            bearing=0.05,           # nearly center
            vx_norm=0.0,
            vy_norm=-0.15,          # receding (negative = moving away)
            foe_containment=False,
            geometry_confidence=0.85,
            frame_id=frame_id,
        )
        track = _make_track(3, [(325.0, float(240 - frame_id * 15))],
                            velocity=(0.0, -0.15 * FRAME_H), track_age=10)

        pred = predictor.predict(geom, track, frame_id)

        assert pred.intersection_flag is False, (
            f"T19 [frame {frame_id}]: Expected intersection_flag=False for receding object, got True"
        )
        assert pred.ttc_s is None, (
            f"T19 [frame {frame_id}]: Expected ttc_s=None for receding object, "
            f"got ttc_s={pred.ttc_s}. This should be None — not a defaulted-safe number."
        )

    print("  T19: intersection_flag=False, ttc_s=None for all 6 receding frames. PASSED.")


def test_t16b_near_zero_relative_velocity_no_crash():
    """T16b (robustness): Near-zero relative velocity → no division by zero, no NaN."""
    predictor = CollisionPredictor()
    geom = _make_geometry(1, bearing=0.0, vx_norm=0.0, vy_norm=0.0,
                          geometry_confidence=0.8, frame_id=0)
    track = _make_track(1, [(320.0, 240.0)], velocity=(0.0, 0.0), track_age=5)

    try:
        pred = predictor.predict(geom, track, 0)
    except Exception as exc:
        pytest.fail(f"T16b: near-zero velocity caused exception: {exc}")

    assert pred.ttc_s is None, "T16b: zero velocity should produce ttc_s=None"
    assert pred.intersection_flag is False, "T16b: zero velocity should produce intersection_flag=False"
    assert not math.isnan(pred.cpa_normalized), "T16b: cpa_normalized must not be NaN"
    assert not math.isnan(pred.prediction_confidence), "T16b: confidence must not be NaN"
    print(f"  T16b: near-zero velocity handled cleanly. cpa={pred.cpa_normalized:.4f}. PASSED.")


def test_t16c_looming_approach_pure_expansion_detected():
    """T16c: Pure head-on approach where centroid doesn't move (vx=0, vy=0) but bbox expands."""
    predictor = CollisionPredictor(
        horizon_s=5.0,
        contact_threshold_normalized=0.05,
        corridor_width_normalized=0.15,
    )

    track = _make_track(track_id=1, centers=[(320.0, 240.0)], velocity=(0.0, 0.0), track_age=10)
    track.expansion_rate = 0.45  # expanding at 45%/sec
    track.bbox_scale = 0.35

    geom = _make_geometry(
        track_id=1,
        bearing=0.0,
        vx_norm=0.0,
        vy_norm=0.0,  # Zero centroid velocity
        foe_containment=True,
        geometry_confidence=0.9,
    )
    geom.expansion_rate = 0.45
    geom.proximity_scale = 0.35

    pred = predictor.predict(geom, track, frame_id=1)

    assert pred.intersection_flag is True, "Pure looming expansion must trigger intersection_flag=True"
    assert pred.ttc_s is not None, "Pure looming expansion must produce finite TTC"
    assert pred.ttc_s < 4.0, f"Expected TTC < 4.0s for 45%/s expansion, got {pred.ttc_s}s"
    assert pred.cpa_normalized < 0.15, f"Expected small CPA for head-on looming, got {pred.cpa_normalized}"


def test_t16d_static_large_obstacle_center_proximity():
    """T16d: Static large obstacle occupying center corridor produces immediate proximity risk."""
    predictor = CollisionPredictor(
        horizon_s=5.0,
        contact_threshold_normalized=0.05,
        corridor_width_normalized=0.15,
    )

    track = _make_track(track_id=2, centers=[(320.0, 240.0)], velocity=(0.0, 0.0), track_age=15)
    track.expansion_rate = 0.0
    track.bbox_scale = 0.55  # Fills 55% of height right in front of chest

    geom = _make_geometry(
        track_id=2,
        bearing=0.0,
        vx_norm=0.0,
        vy_norm=0.0,
        foe_containment=False,
        geometry_confidence=0.9,
    )
    geom.expansion_rate = 0.0
    geom.proximity_scale = 0.55

    pred = predictor.predict(geom, track, frame_id=1)

    assert pred.proximity_risk > 0.50, f"Expected proximity_risk > 0.50 for 55% height object, got {pred.proximity_risk}"
    assert pred.intersection_flag is True, "Large static center obstacle must block the path"
    # Step 7 anti-fabrication: Stationary obstacles do not have a synthetic TTC fabricated
    # without approach motion; risk engine acts directly on proximity_risk and intersection_flag.
    assert pred.ttc_s is None, "Stationary obstacle must not fabricate synthetic TTC without motion"


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("--- Running M07 Collision Prediction Tests (Gate C) ---")
    print("[TEST] T16: Head-on approach TTC decreases monotonically...")
    test_t16_head_on_approach_ttc_decreases()
    print("  -> T16 PASSED")

    print("[TEST] T17: Parallel motion → intersection_flag=False...")
    test_t17_parallel_motion_no_intersection()
    print("  -> T17 PASSED")

    print("[TEST] T18: Crossing trajectory → finite TTC + small CPA...")
    test_t18_crossing_trajectory_produces_finite_ttc()
    print("  -> T18 PASSED")

    print("[TEST] T19: Receding motion → ttc_s=None, intersection_flag=False...")
    test_t19_receding_object_no_ttc()
    print("  -> T19 PASSED")

    print("[TEST] T16b: Near-zero velocity → no crash...")
    test_t16b_near_zero_relative_velocity_no_crash()
    print("  -> T16b PASSED")

    print("[TEST] T16c: Looming pure expansion detected...")
    test_t16c_looming_approach_pure_expansion_detected()
    print("  -> T16c PASSED")

    print("[TEST] T16d: Static large obstacle center proximity...")
    test_t16d_static_large_obstacle_center_proximity()
    print("  -> T16d PASSED")

    print("\nALL M07 TESTS PASSED (Gate C verified — no camera/YOLO required).")
