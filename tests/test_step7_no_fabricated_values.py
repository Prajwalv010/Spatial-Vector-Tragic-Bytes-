"""Step 7 — Anti-Fabrication Test Suite.

Verifies that all safety-critical numerical outputs from core modules are
computed from real evidence rather than being hardcoded constants.

Test IDs:
  T7-01: FreeSpaceEstimator confidence is never a hardcoded constant
  T7-02: GroundHazardDetector returns dist_m=None when geometry is unconfigured
  T7-03: GroundHazardDetector _compute_ground_plane_distance uses real geometry
  T7-04: SimulatedIMUReader emits status=\"SIMULATED\" not \"OK\"
  T7-05: build_pipeline_health returns real status from subsystem queries
  T7-06: CollisionPredictor does not fabricate TTC when no motion measurable
  T7-07: _compute_confidence varies with inputs (not a constant function)
  T7-08: Corridor confidence is 0.0 for UNKNOWN regardless of inputs
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ──────────────────────────────────────────────────────────────────────────
# T7-07 / T7-08: _compute_confidence varies with evidence
# ──────────────────────────────────────────────────────────────────────────

def test_t7_07_compute_confidence_varies_with_evidence():
    """T7-07: _compute_confidence must produce different values for different g_frac.

    A constant function would return the same value regardless of g_frac.
    This test checks that higher ground fraction consistently yields higher confidence.
    """
    from spatialvector.freespace.corridor_estimator import _compute_confidence, CorridorStatus

    WALKABLE = CorridorStatus.WALKABLE
    seg_min = 0.25

    # Varying ground fractions should produce monotonically increasing confidence
    confs = [
        _compute_confidence(WALKABLE, g_frac=g, o_frac=0.0,
                            seg_min_ground=seg_min, seg_mean_prob=0.85,
                            classical_cues_passed=True, temporal_walkable_fraction=0.8)
        for g in (0.26, 0.40, 0.60, 0.80, 0.95)
    ]

    # Confidence must not all be the same value
    assert len(set(round(c, 3) for c in confs)) > 1, (
        f"_compute_confidence returned constant: {confs}. "
        "It must vary with ground fraction."
    )

    # Confidence must increase with ground fraction
    for i in range(len(confs) - 1):
        assert confs[i] <= confs[i+1] + 0.001, (
            f"Confidence did not increase monotonically: {confs}"
        )

    # Must be bounded in [0, 1]
    for c in confs:
        assert 0.0 <= c <= 1.0, f"Confidence out of bounds: {c}"


def test_t7_08_unknown_status_always_returns_zero_confidence():
    """T7-08: UNKNOWN status must return 0.0 confidence regardless of other inputs."""
    from spatialvector.freespace.corridor_estimator import _compute_confidence, CorridorStatus

    for g_frac in (0.0, 0.5, 1.0):
        for prob in (None, 0.5, 0.99):
            conf = _compute_confidence(
                CorridorStatus.UNKNOWN, g_frac=g_frac, o_frac=0.0,
                seg_min_ground=0.25, seg_mean_prob=prob,
                classical_cues_passed=True, temporal_walkable_fraction=1.0
            )
            assert conf == 0.0, (
                f"UNKNOWN status returned non-zero confidence: {conf} "
                f"(g_frac={g_frac}, prob={prob})"
            )


def test_t7_07b_blocked_confidence_scales_with_obstacle_frac():
    """T7-07b: BLOCKED confidence must scale with o_frac, not be a fixed constant."""
    from spatialvector.freespace.corridor_estimator import _compute_confidence, CorridorStatus

    BLOCKED = CorridorStatus.BLOCKED
    confs = [
        _compute_confidence(BLOCKED, g_frac=0.0, o_frac=o,
                            seg_min_ground=0.25, seg_mean_prob=None,
                            classical_cues_passed=False, temporal_walkable_fraction=0.0)
        for o in (0.10, 0.20, 0.35, 0.50, 0.80)
    ]
    # Must vary
    assert len(set(round(c, 3) for c in confs)) > 1, (
        f"BLOCKED confidence is constant: {confs}. It must scale with obstacle fraction."
    )
    # Must be monotonically non-decreasing
    for i in range(len(confs) - 1):
        assert confs[i] <= confs[i+1] + 0.001, f"BLOCKED confidence not monotonic: {confs}"


# ──────────────────────────────────────────────────────────────────────────
# T7-02 / T7-03: Ground hazard distance is geometric or None
# ──────────────────────────────────────────────────────────────────────────

def test_t7_02_dist_m_is_none_when_geometry_unconfigured():
    """T7-02: When camera geometry params are missing, dist_m must be None."""
    from spatialvector.hazards.ground_hazard import GroundHazardDetector

    detector = GroundHazardDetector(
        allow_heuristic=True,
        camera_height_m=None,   # deliberately missing
        camera_pitch_deg=None,
        focal_length_ratio=None,
    )
    dist_m, provenance = detector._compute_ground_plane_distance(fy2=400.0, frame_h=480, frame_w=640)
    assert dist_m is None, f"Expected None when geometry unconfigured, got: {dist_m}"
    assert provenance == "unknown"


def test_t7_03_dist_m_uses_geometry_formula():
    """T7-03: With valid geometry params, dist_m must match the expected formula."""
    from spatialvector.hazards.ground_hazard import GroundHazardDetector

    H   = 1.30  # metres
    pit = 15.0  # degrees
    flr = 0.72  # focal_length_ratio
    W   = 1280
    HT  = 720

    detector = GroundHazardDetector(
        allow_heuristic=True,
        camera_height_m=H,
        camera_pitch_deg=pit,
        focal_length_ratio=flr,
    )
    fy2 = 600.0  # bottom of a bbox at row 600 of 720
    dist_m, provenance = detector._compute_ground_plane_distance(fy2=fy2, frame_h=HT, frame_w=W)

    # Compute expected value independently
    focal_px   = flr * W
    horizon_px = HT * (0.5 - math.tan(math.radians(pit)))
    expected   = float(np.clip((H * focal_px) / max(1.0, fy2 - horizon_px), 0.2, 20.0))

    assert dist_m is not None, "Expected a numeric distance, got None"
    assert provenance == "ground_plane_geometry"
    assert abs(dist_m - expected) < 0.001, (
        f"Computed distance {dist_m:.4f} does not match expected {expected:.4f}"
    )


def test_t7_03b_dist_m_varies_with_bbox_position():
    """T7-03b: Distance must vary with bbox position (not a constant function)."""
    from spatialvector.hazards.ground_hazard import GroundHazardDetector

    detector = GroundHazardDetector(
        allow_heuristic=True,
        camera_height_m=1.30,
        camera_pitch_deg=15.0,
        focal_length_ratio=0.72,
    )
    dists = [
        detector._compute_ground_plane_distance(fy2=float(y), frame_h=720, frame_w=1280)[0]
        for y in (450, 500, 550, 600, 650, 700)
    ]
    # All must be non-None
    assert all(d is not None for d in dists), f"Some distances are None: {dists}"
    # Must vary (not constant)
    assert len(set(round(d, 2) for d in dists)) > 1, (
        f"Distances are constant across positions: {dists}. Formula not varying with fy2."
    )
    # Must decrease as fy2 increases (closer to camera = closer hazard)
    for i in range(len(dists) - 1):
        assert dists[i] >= dists[i+1], f"Distance not decreasing with fy2: {dists}"


# ──────────────────────────────────────────────────────────────────────────
# T7-04: SimulatedIMUReader emits SIMULATED status
# ──────────────────────────────────────────────────────────────────────────

def test_t7_04_simulated_imu_emits_simulated_status():
    """T7-04: SimulatedIMUReader must emit status='SIMULATED', never 'OK'."""
    from spatialvector.motion.imu_reader import SimulatedIMUReader

    reader = SimulatedIMUReader(rate_hz=50.0)
    reader.start()
    time.sleep(0.05)  # let one sample arrive
    sample = reader.get_latest()
    reader.stop()

    assert sample is not None, "SimulatedIMUReader returned no sample"
    assert sample.status == "SIMULATED", (
        f"Expected status='SIMULATED', got '{sample.status}'. "
        "SimulatedIMUReader must never report 'OK' as if it were real hardware."
    )


# ──────────────────────────────────────────────────────────────────────────
# T7-05: build_pipeline_health returns real status
# ──────────────────────────────────────────────────────────────────────────

def test_t7_05_pipeline_health_uses_real_subsystem_queries():
    """T7-05: build_pipeline_health must query real subsystem objects."""
    from spatialvector.hmi.schemas import build_pipeline_health
    from spatialvector.motion.schemas import IMUSample

    # Mock a running frame source
    fs_running = MagicMock()
    fs_running.is_running.return_value = True

    # Mock a stopped frame source
    fs_stopped = MagicMock()
    fs_stopped.is_running.return_value = False

    # Mock IMU returning SIMULATED sample
    imu_mock = MagicMock()
    simulated_sample = IMUSample(t_arrival=0.0, t_device=None,
                                 gyro_xyz=(0.0, 0.0, 0.0), status="SIMULATED")
    imu_mock.get_latest.return_value = simulated_sample

    # Mock disconnected arduino
    arduino_mock = MagicMock()
    arduino_status = MagicMock()
    arduino_status.connected = False
    arduino_mock.get_status.return_value = arduino_status

    health = build_pipeline_health(
        frame_source=fs_running,
        imu_reader=imu_mock,
        arduino=arduino_mock,
    )

    # Camera must reflect actual running state (not hardcoded)
    assert health["camera"] == "OK", f"Expected OK for running camera, got {health['camera']}"

    health2 = build_pipeline_health(
        frame_source=fs_stopped,
        imu_reader=imu_mock,
        arduino=arduino_mock,
    )
    assert health2["camera"] == "DEGRADED", f"Expected DEGRADED for stopped camera, got {health2['camera']}"

    # IMU must reflect SIMULATED from sample status
    assert health["imu"] == "SIMULATED", (
        f"Expected SIMULATED for SimulatedIMUReader, got {health['imu']}"
    )

    # Arduino must reflect disconnected state
    assert health["arduino"] == "DISCONNECTED", (
        f"Expected DISCONNECTED when arduino.connected=False, got {health['arduino']}"
    )


def test_t7_05b_pipeline_health_unknowns_when_subsystems_none():
    """T7-05b: When subsystems are None, health must be UNKNOWN (not OK)."""
    from spatialvector.hmi.schemas import build_pipeline_health

    health = build_pipeline_health(frame_source=None, imu_reader=None, arduino=None)
    for key in ("camera", "imu", "arduino"):
        assert health[key] == "UNKNOWN", (
            f"Expected UNKNOWN when {key} subsystem is None, got {health[key]}"
        )


# ──────────────────────────────────────────────────────────────────────────
# T7-06: CollisionPredictor does not fabricate TTC from proximity alone
# ──────────────────────────────────────────────────────────────────────────

def test_t7_06_no_synthetic_ttc_when_no_motion():
    """T7-06: When relative velocity is near-zero (static object), ttc_s must be None.

    The previous code fabricated TTC from proximity_scale alone when no motion
    was detected. This verifies that the fix is in place: a stationary large
    object should not receive a synthetic TTC.
    """
    from spatialvector.decision.prediction import CollisionPredictor
    from spatialvector.motion.schemas import ObjectGeometry
    from spatialvector.perception.schemas import Track

    predictor = CollisionPredictor()

    # Construct a geometry for a large, stationary object in center corridor
    geom = ObjectGeometry(
        track_id=1,
        frame_id=1,
        bearing=0.0,                  # dead centre
        relative_image_velocity=(0.0, 0.0),  # zero velocity
        motion_vector=(0.0, 0.0),
        foe_containment=False,
        expansion_rate=0.0,           # no looming
        proximity_scale=0.50,         # large bbox (50% of frame height)
        geometry_confidence=0.8,
    )

    track = Track(
        track_id=1,
        class_name="person",
        track_age=10,
        center_history=[(320, 400)] * 10,
        bbox_history=[(200, 200, 440, 600)] * 10,
        estimated_image_velocity=(0.0, 0.0),
        track_confidence=0.85,
        last_seen_frame_id=1,
    )

    pred = predictor.predict(geom, track, frame_id=1)

    # With zero relative velocity, _compute_cpa_ttc_intersection must return ttc_s=None
    # The proximity_risk block must NOT fabricate a TTC from proximity_scale
    assert pred.ttc_s is None, (
        f"Expected ttc_s=None for static large object (no measurable motion), "
        f"got ttc_s={pred.ttc_s:.2f}. Fabricated TTC has been removed."
    )

    # intersection_flag can be True (path IS obstructed by a large object)
    assert pred.intersection_flag is True, (
        f"Expected intersection_flag=True for large object in center, got {pred.intersection_flag}"
    )


# ──────────────────────────────────────────────────────────────────────────
# T7-01: FreeSpaceEstimator classical-only confidence is bounded below max
# ──────────────────────────────────────────────────────────────────────────

def test_t7_01_freespace_confidence_never_claims_full_certainty():
    """T7-01: No WALKABLE confidence must reach 1.0 (certainty not achievable with 2D camera)."""
    from spatialvector.freespace.corridor_estimator import _compute_confidence, CorridorStatus

    # Maximally favourable inputs
    conf = _compute_confidence(
        CorridorStatus.WALKABLE,
        g_frac=1.0, o_frac=0.0,
        seg_min_ground=0.25, seg_mean_prob=1.0,
        classical_cues_passed=True, temporal_walkable_fraction=1.0,
    )
    assert conf < 1.0, (
        f"WALKABLE confidence must be < 1.0 (single camera cannot be certain), got {conf}"
    )
    assert conf >= 0.70, (
        f"WALKABLE confidence with all favourable evidence should be >= 0.70, got {conf}"
    )


def test_t7_01b_hardcoded_values_not_present_in_module_source():
    """T7-01b: The old hardcoded constants must no longer appear in corridor_estimator.py."""
    import re
    src_path = Path(__file__).resolve().parent.parent / "spatialvector" / "freespace" / "corridor_estimator.py"
    src = src_path.read_text(encoding="utf-8")

    forbidden_patterns = [
        r"return CorridorStatus\.BLOCKED,\s*0\.95",  # old YOLO veto hardcode
        r"return CorridorStatus\.BLOCKED,\s*0\.85",  # old semantic blocked hardcode
        r"return CorridorStatus\.BLOCKED,\s*0\.80",  # old full scene obstacle hardcode
        r"return CorridorStatus\.WALKABLE,\s*0\.80",  # old walk confirmed hardcode
        r"return CorridorStatus\.WALKABLE,\s*0\.65",  # old walk unverified hardcode
        r"return.*0\.10.*'insufficient ground",      # old 0.10 for UNKNOWN
    ]
    violations = []
    for pattern in forbidden_patterns:
        matches = re.findall(pattern, src)
        if matches:
            violations.append(f"Pattern found: {pattern!r} -> {matches}")

    assert not violations, (
        "Hardcoded confidence constants found in corridor_estimator.py:\n" +
        "\n".join(violations)
    )
