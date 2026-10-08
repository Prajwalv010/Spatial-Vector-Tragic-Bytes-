"""
Tests for M05 — EgoMotionCompensator + IMU ingestion

T12: Simulated camera rotation with matching synthetic gyro → corrected_flow has lower
     rotational component than raw input flow.

T13: IMU drops mid-run → module enters DEGRADED, fallback_active=True,
     corrected_flow = raw flow (no silent failure, no crash).

Runnable with:
    pytest tests/test_m05_imu.py -v -s
    python -u tests/test_m05_imu.py
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from spatialvector.motion.schemas import FlowResult, IMUSample
from spatialvector.motion.ego_motion import EgoMotionCompensator
from spatialvector.motion.imu_reader import SimulatedIMUReader, _parse_imu_line


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_rotation_flow(
    gz_rad_s: float,
    frame_w: int,
    frame_h: int,
    n_points: int = 80,
    focal_px: float = None,
    dt: float = 1.0 / 30.0,
) -> FlowResult:
    """Build a FlowResult whose vectors match what a pure-yaw rotation would produce.

    For a point (x0, y0) relative to center, under yaw ω_z:
        dx ≈ -ω_z * f * dt
        dy ≈  0  (for pure yaw)
    All vectors should be uniformly horizontal — no expansion → pure rotation.
    """
    if focal_px is None:
        focal_px = math.sqrt(frame_w**2 + frame_h**2)

    cx, cy = frame_w / 2.0, frame_h / 2.0
    rng = np.random.default_rng(7)
    xs = rng.uniform(50, frame_w - 50, n_points)
    ys = rng.uniform(50, frame_h - 50, n_points)

    # Predicted rotation displacement at each point (pure yaw, no roll/pitch)
    pred_dx = -gz_rad_s * focal_px * dt
    pred_dy = 0.0

    flow_vectors = [(float(x), float(y), pred_dx, pred_dy) for x, y in zip(xs, ys)]
    return FlowResult(
        frame_id=0,
        timestamp=time.monotonic(),
        flow_vectors=flow_vectors,
        flow_quality=0.85,
        foe_x=float("nan"),
        foe_y=float("nan"),
        foe_confidence=0.05,
    )


def _make_imu_sample_ok(gz: float, t_now: float = None) -> IMUSample:
    """Build a valid IMUSample for the current monotonic time."""
    t = t_now if t_now is not None else time.monotonic()
    return IMUSample(
        t_arrival=t,
        t_device=None,
        gyro_xyz=(0.0, 0.0, gz),
        accel_xyz=None,
        status="OK",
    )


def _rotational_energy(flow_vectors: list[tuple[float, float, float, float]]) -> float:
    """Measure mean magnitude of flow vectors — proxy for rotational content."""
    if not flow_vectors:
        return 0.0
    mags = [math.sqrt(dx**2 + dy**2) for _, _, dx, dy in flow_vectors]
    return float(np.mean(mags))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_t12_rotation_corrected_by_imu():
    """T12: Pure-rotation flow + matching gyro → corrected_flow has lower magnitude than raw.

    We simulate a camera yawing at gz=0.3 rad/s. The flow vectors encode exactly
    the rotational displacement at each point. When we feed the matching gyro signal
    to EgoMotionCompensator, it should subtract most of that rotation and leave
    corrected_flow with much lower magnitude.
    """
    frame_w, frame_h = 640, 480
    gz = 0.3  # rad/s yaw

    flow = _make_rotation_flow(gz_rad_s=gz, frame_w=frame_w, frame_h=frame_h)
    t_now = time.monotonic()
    imu = _make_imu_sample_ok(gz=gz, t_now=t_now)

    compensator = EgoMotionCompensator(
        fallback_flow_quality_threshold=0.3,
        max_timestamp_drift_s=0.10,
    )

    state = compensator.compensate(
        flow=flow,
        imu_sample=imu,
        frame_timestamp=t_now,
        frame_w=frame_w,
        frame_h=frame_h,
    )

    assert state.motion_quality == "OK", (
        f"T12: Expected motion_quality=OK, got {state.motion_quality}. "
        f"Check IMU timestamp or flow_quality threshold."
    )
    assert not state.fallback_active, "T12: fallback_active should be False when IMU is valid"

    raw_energy = _rotational_energy(flow.flow_vectors)
    corrected_energy = _rotational_energy(state.corrected_flow)

    # Corrected flow should have meaningfully less rotational energy than raw
    # Allow up to 60% remaining (gyro approximation is imperfect; this is first-order)
    assert corrected_energy < raw_energy * 0.70, (
        f"T12: Corrected flow energy ({corrected_energy:.3f}) should be < 70% of raw "
        f"({raw_energy:.3f}) — ego-motion subtraction had little effect."
    )


def test_t13_imu_disconnect_enters_degraded():
    """T13: IMU goes DISCONNECTED mid-run → DEGRADED state, fallback_active=True, no crash.

    - First call: valid IMU → motion_quality=OK
    - Second call: DISCONNECTED IMU → motion_quality=DEGRADED, corrected_flow = raw flow
    - Third call: None IMU (not started) → also DEGRADED
    """
    frame_w, frame_h = 640, 480
    gz = 0.1

    compensator = EgoMotionCompensator(
        fallback_flow_quality_threshold=0.3,
        max_timestamp_drift_s=0.10,
    )

    t_now = time.monotonic()
    flow = _make_rotation_flow(gz_rad_s=gz, frame_w=frame_w, frame_h=frame_h)
    flow.timestamp = t_now

    # --- Round 1: valid IMU → OK ---
    good_imu = _make_imu_sample_ok(gz=gz, t_now=t_now)
    state_ok = compensator.compensate(flow, good_imu, t_now, frame_w, frame_h)
    assert state_ok.motion_quality == "OK", f"T13 setup: first call should be OK, got {state_ok.motion_quality}"

    # --- Round 2: DISCONNECTED IMU → DEGRADED ---
    disconnected_imu = IMUSample(
        t_arrival=time.monotonic(),
        t_device=None,
        gyro_xyz=(0.0, 0.0, 0.0),
        status="DISCONNECTED",
    )
    t_now2 = time.monotonic()
    flow2 = _make_rotation_flow(gz_rad_s=gz, frame_w=frame_w, frame_h=frame_h)
    state_degraded = compensator.compensate(flow2, disconnected_imu, t_now2, frame_w, frame_h)

    assert state_degraded.motion_quality == "DEGRADED", (
        f"T13: Expected DEGRADED after DISCONNECTED IMU, got {state_degraded.motion_quality}"
    )
    assert state_degraded.fallback_active is True, (
        "T13: fallback_active must be True when motion_quality is DEGRADED"
    )

    # Corrected flow should equal the raw flow (pass-through, no correction applied)
    assert len(state_degraded.corrected_flow) == len(flow2.flow_vectors), (
        "T13: In fallback mode, corrected_flow should have same length as raw flow_vectors"
    )
    for raw_vec, corr_vec in zip(flow2.flow_vectors, state_degraded.corrected_flow):
        assert raw_vec == corr_vec, (
            f"T13: In fallback mode, corrected_flow must equal raw flow. "
            f"Got {corr_vec} vs raw {raw_vec}"
        )

    # --- Round 3: None IMU (hardware not started) → also DEGRADED, no crash ---
    t_now3 = time.monotonic()
    flow3 = _make_rotation_flow(gz_rad_s=gz, frame_w=frame_w, frame_h=frame_h)
    try:
        state_none = compensator.compensate(flow3, None, t_now3, frame_w, frame_h)
    except Exception as exc:
        pytest.fail(f"T13: compensate() crashed when IMU sample is None: {exc}")

    assert state_none.motion_quality == "DEGRADED", (
        f"T13: None IMU should produce DEGRADED, got {state_none.motion_quality}"
    )
    assert state_none.fallback_active is True


def test_t12b_parse_imu_line():
    """Bonus: validate the serial line parser handles all documented wire formats."""
    sample = _parse_imu_line("GYRO:0.012,-0.003,0.001")
    assert sample is not None and sample.status == "OK"
    assert abs(sample.gyro_xyz[0] - 0.012) < 1e-6

    sample_ts = _parse_imu_line("GYRO:0.1,0.2,0.3,T:12345")
    assert sample_ts is not None and sample_ts.status == "OK"
    assert sample_ts.t_device is not None
    assert abs(sample_ts.t_device - 12.345) < 1e-4

    sample_accel = _parse_imu_line("GYRO:0.1,0.2,0.3,0.01,-9.8,0.05,T:99")
    assert sample_accel is not None and sample_accel.status == "OK"
    assert sample_accel.accel_xyz is not None
    assert abs(sample_accel.accel_xyz[1] - (-9.8)) < 1e-4

    bad_sample = _parse_imu_line("GARBAGE")
    assert bad_sample is not None and bad_sample.status == "INVALID"


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("--- Running M05 IMU / Ego-Motion Tests Standalone ---")
    print("[TEST] T12: Rotation corrected by matching gyro signal...")
    test_t12_rotation_corrected_by_imu()
    print("  -> T12 PASSED")

    print("[TEST] T12b: IMU serial line parser validation...")
    test_t12b_parse_imu_line()
    print("  -> T12b PASSED")

    print("[TEST] T13: IMU disconnect → DEGRADED fallback (no crash)...")
    test_t13_imu_disconnect_enters_degraded()
    print("  -> T13 PASSED")

    print("\nALL M05 TESTS PASSED.")
