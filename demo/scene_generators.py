"""SpatialVector-HMI — Scene Generators and Fixture Builder for Demo Harness.

Provides:
1. Video fixture generator: Creates real MP4 video clips in tests/fixtures/ for Scenes 1-5.
2. Synthetic pipeline runners: Supplies reproducible frame/track/geometry feeds through M01-M09,
   including Scene 6 which exercises the genuine SimulatedIMUReader disconnect path.
"""

from __future__ import annotations

import math
from pathlib import Path
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from spatialvector.decision.schemas import HapticCommand, Prediction, RiskState
from spatialvector.motion.ego_motion import EgoMotionCompensator
from spatialvector.motion.geometry import compute_geometry_batch
from spatialvector.motion.imu_reader import SimulatedIMUReader
from spatialvector.motion.optical_flow import OpticalFlowEstimator
from spatialvector.motion.schemas import FlowResult, IMUSample, ObjectGeometry
from spatialvector.perception.schemas import Frame, Track

ROOT = Path(__file__).resolve().parent.parent
FIXTURES_DIR = ROOT / "tests" / "fixtures"


def ensure_fixtures():
    """Generates MP4 clips for demo scenes 1-5 if not already present."""
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    _generate_scene1_video()
    _generate_scene2_video()
    _generate_scene3_video()
    _generate_scene4_video()
    _generate_scene5_video()


def _make_pedestrian_patch() -> np.ndarray:
    """Creates a high-contrast pedestrian patch that can be tracked across frames."""
    patch = np.zeros((140, 60, 3), dtype=np.uint8)
    # Head
    cv2.circle(patch, (30, 25), 18, (140, 110, 90), -1)
    # Torso
    cv2.rectangle(patch, (10, 45), (50, 95), (60, 80, 220), -1)
    # Legs
    cv2.rectangle(patch, (15, 95), (28, 135), (35, 35, 35), -1)
    cv2.rectangle(patch, (32, 95), (45, 135), (35, 35, 35), -1)
    return patch


def _generate_scene1_video():
    """Scene 1: Open space baseline — walking forward with no obstacles."""
    out = FIXTURES_DIR / "demo_scene1_baseline.mp4"
    if out.exists() and out.stat().st_size > 1000:
        return
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out), fourcc, 15.0, (640, 360))
    for i in range(45):
        frame = np.full((360, 640, 3), (225, 230, 235), dtype=np.uint8)
        # Ground floor perspective lines
        offset = (i * 8) % 40
        for y in range(200 + offset, 360, 40):
            cv2.line(frame, (0, y), (640, y), (190, 195, 200), 1)
        writer.write(frame)
    writer.release()


def _generate_scene2_video():
    """Scene 2: Parallel wall/obstacle — object on left moving parallel to heading."""
    out = FIXTURES_DIR / "demo_scene2_parallel_wall.mp4"
    if out.exists() and out.stat().st_size > 1000:
        return
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out), fourcc, 15.0, (640, 360))
    patch = _make_pedestrian_patch()
    ph, pw = patch.shape[:2]
    for i in range(45):
        frame = np.full((360, 640, 3), (220, 225, 230), dtype=np.uint8)
        # Wall / obstacle stays on far left (x = 80px)
        x = 80
        y = 120
        frame[y : y + ph, x : x + pw] = patch
        writer.write(frame)
    writer.release()


def _generate_scene3_video():
    """Scene 3: Turn toward wall — object on left, user rotates toward it."""
    out = FIXTURES_DIR / "demo_scene3_turn_toward_wall.mp4"
    if out.exists() and out.stat().st_size > 1000:
        return
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out), fourcc, 15.0, (640, 360))
    patch = _make_pedestrian_patch()
    ph, pw = patch.shape[:2]
    for i in range(45):
        frame = np.full((360, 640, 3), (220, 225, 230), dtype=np.uint8)
        # Object starts at x=80, shifts to center x=290 as user rotates
        shift = min(1.0, i / 30.0)
        x = int(80 + shift * 210)
        y = int(120 + shift * 30)
        scale = 1.0 + shift * 0.4
        curr_patch = cv2.resize(patch, (int(pw * scale), int(ph * scale)))
        ch, cw = curr_patch.shape[:2]
        frame[y : y + ch, x : x + cw] = curr_patch
        writer.write(frame)
    writer.release()


def _generate_scene4_video():
    """Scene 4: Crossing person — walks right-to-left across forward path."""
    out = FIXTURES_DIR / "demo_scene4_crossing_person.mp4"
    if out.exists() and out.stat().st_size > 1000:
        return
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out), fourcc, 15.0, (640, 360))
    patch = _make_pedestrian_patch()
    ph, pw = patch.shape[:2]
    for i in range(45):
        frame = np.full((360, 640, 3), (220, 225, 230), dtype=np.uint8)
        # Person moves from x=520 across center to x=100
        x = int(520 - i * 9.5)
        y = 120
        if 0 <= x < 640 - pw:
            frame[y : y + ph, x : x + pw] = patch
        writer.write(frame)
    writer.release()


def _generate_scene5_video():
    """Scene 5: Safe passing — person passes in adjacent corridor without crossing path."""
    out = FIXTURES_DIR / "demo_scene5_safe_passing.mp4"
    if out.exists() and out.stat().st_size > 1000:
        return
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out), fourcc, 15.0, (640, 360))
    patch = _make_pedestrian_patch()
    ph, pw = patch.shape[:2]
    for i in range(45):
        frame = np.full((360, 640, 3), (220, 225, 230), dtype=np.uint8)
        # Person recedes/passes on far right (x=500), strictly outside center corridor
        x = 500
        y = int(120 + i * 2)
        if 0 <= x < 640 - pw and y + ph <= 360:
            frame[y : y + ph, x : x + pw] = patch
        writer.write(frame)
    writer.release()


# ---------------------------------------------------------------------------
# Synthetic Pipeline Feeders (Full M01-M09 Execution)
# ---------------------------------------------------------------------------

class DemoScenarioRunner:
    """Runs a specific demo scene through M01-M09 with real state calculation."""

    FRAME_W = 640
    FRAME_H = 360

    @classmethod
    def run_scenario(cls, scenario_id: str, num_frames: int = 35) -> List[Tuple[int, List[Track], List[Prediction], RiskState, HapticCommand]]:
        from spatialvector.decision.corridor_policy import CorridorPolicy
        from spatialvector.decision.prediction import CollisionPredictor
        from spatialvector.decision.risk_engine import RiskEngine

        predictor = CollisionPredictor(horizon_s=5.0, contact_threshold_normalized=0.05, corridor_width_normalized=0.12)
        engine = RiskEngine(
            weight_ttc=0.50,
            weight_miss_distance=0.30,
            weight_intersection_confidence=0.20,
            state_thresholds={"caution": 0.30, "warning": 0.60, "critical": 0.85},
            hysteresis_frames_up=3,
            hysteresis_frames_down=5,
            degraded_confidence_threshold=0.25,
        )
        policy = CorridorPolicy(all_unsafe_risk_threshold=0.70, trend_window_frames=5)
        compensator = EgoMotionCompensator(fallback_flow_quality_threshold=0.30)

    @classmethod
    def run_scenario(cls, scenario_id: str, num_frames: int = 35) -> List[Tuple[int, List[Track], List[Prediction], RiskState, HapticCommand]]:
        from spatialvector.decision.corridor_policy import CorridorPolicy
        from spatialvector.decision.prediction import CollisionPredictor
        from spatialvector.decision.risk_engine import RiskEngine

        predictor = CollisionPredictor(horizon_s=5.0, contact_threshold_normalized=0.05, corridor_width_normalized=0.12)
        engine = RiskEngine(
            weight_ttc=0.50,
            weight_miss_distance=0.30,
            weight_intersection_confidence=0.20,
            state_thresholds={"caution": 0.30, "warning": 0.60, "critical": 0.85},
            hysteresis_frames_up=3,
            hysteresis_frames_down=5,
            degraded_confidence_threshold=0.25,
        )
        policy = CorridorPolicy(all_unsafe_risk_threshold=0.70, trend_window_frames=5)
        compensator = EgoMotionCompensator(fallback_flow_quality_threshold=0.30)

        # Setup IMU for Scene 6 vs normal scenes
        imu_reader = None
        if scenario_id == "scene_6_sensor_degradation":
            # Real SimulatedIMUReader with automatic disconnect injection at 0.25s
            imu_reader = SimulatedIMUReader(inject_disconnect_after_s=0.25)
        else:
            imu_reader = SimulatedIMUReader(gyro_fn=lambda t: (0.0, 0.0, 0.0))
        imu_reader.start()

        results = []

        try:
            for fid in range(num_frames):
                time.sleep(0.025)
                ts = time.monotonic()
                tracks, geoms, flow = cls._get_inputs_for_scene(scenario_id, fid, ts)

                # M05 IMU sample
                imu_s = imu_reader.get_latest()

                # M05 Ego-Motion Compensation
                motion_st = compensator.compensate(flow, imu_s, ts, cls.FRAME_W, cls.FRAME_H)

                # M07 Collision Prediction
                predictions = predictor.predict_batch(geoms, tracks, fid)

                # M08 Risk Engine
                risk_st = engine.update(predictions, fallback_active=motion_st.fallback_active, timestamp=ts)

                # M09 Corridor Policy
                cmd = policy.select(risk_st, timestamp=ts)

                results.append((fid, tracks, predictions, risk_st, cmd))
        finally:
            imu_reader.stop()

        return results

    @classmethod
    def _get_inputs_for_scene(cls, scenario_id: str, fid: int, ts: float) -> Tuple[List[Track], List[ObjectGeometry], FlowResult]:
        flow = FlowResult(
            frame_id=fid,
            timestamp=ts,
            flow_vectors=[(320.0, 180.0, 0.0, 2.0)],
            flow_quality=0.90,
            foe_x=320.0,
            foe_y=180.0,
            foe_confidence=0.88,
        )

        if scenario_id == "scene_1_baseline":
            # Open space — zero obstacles
            return [], [], flow

        elif scenario_id == "scene_2_parallel_wall":
            # Parallel wall on left side (bearing = -0.32 rad)
            # Lateral offset is outside forward corridor, relative velocity is parallel -> intersection=False
            track = Track(
                track_id=1,
                class_name="Obstacle",
                bbox_history=[(80.0, 120.0, 140.0, 260.0)],
                center_history=[(110.0, 190.0)],
                estimated_image_velocity=(0.0, 40.0),
                track_age=fid + 1,
                track_confidence=0.88,
                last_seen_frame_id=fid,
            )
            geom = ObjectGeometry(
                track_id=1,
                frame_id=fid,
                bearing=-0.32,
                relative_image_velocity=(0.0, 0.05),
                motion_vector=(0.0, 0.05),
                foe_containment=False,
                geometry_confidence=0.90,
            )
            return [track], [geom], flow

        elif scenario_id == "scene_3_turn_toward_wall":
            # Starts parallel (frames 0-8), then user rotates toward wall (frames 9+)
            # As user rotates, wall enters center corridor and trajectory intersects path
            is_turning = fid >= 8
            bearing = 0.0 if is_turning else -0.32
            rel_vx = 0.0
            rel_vy = 0.20 if is_turning else 0.05
            track = Track(
                track_id=1,
                class_name="Obstacle",
                bbox_history=[(280.0 if is_turning else 80.0, 120.0, 360.0 if is_turning else 140.0, 260.0)],
                center_history=[(320.0 if is_turning else 110.0, 190.0)],
                estimated_image_velocity=(rel_vx * cls.FRAME_W, rel_vy * cls.FRAME_H),
                track_age=fid + 1,
                track_confidence=0.92,
                last_seen_frame_id=fid,
            )
            geom = ObjectGeometry(
                track_id=1,
                frame_id=fid,
                bearing=bearing,
                relative_image_velocity=(rel_vx, rel_vy),
                motion_vector=(rel_vx, rel_vy),
                foe_containment=is_turning,
                geometry_confidence=0.92,
            )
            return [track], [geom], flow

        elif scenario_id == "scene_4_crossing_person":
            # Person crosses from right corridor into forward corridor path
            # Using exact physics: obj_x = sin(bearing) ~ 0.20, vx = -0.08, vy = 0.20 -> cpa = 0.0, intersection = True
            is_crossing = fid >= 6
            bearing = 0.18 if is_crossing else 0.35
            rel_vx = -0.072 if is_crossing else -0.03
            rel_vy = 0.20 if is_crossing else 0.05
            track = Track(
                track_id=4,
                class_name="Person",
                bbox_history=[(380.0, 110.0, 440.0, 250.0)],
                center_history=[(410.0, 180.0)],
                estimated_image_velocity=(rel_vx * cls.FRAME_W, rel_vy * cls.FRAME_H),
                track_age=fid + 1,
                track_confidence=0.92,
                last_seen_frame_id=fid,
            )
            geom = ObjectGeometry(
                track_id=4,
                frame_id=fid,
                bearing=bearing,
                relative_image_velocity=(rel_vx, rel_vy),
                motion_vector=(rel_vx, rel_vy),
                foe_containment=True,
                geometry_confidence=0.92,
            )
            return [track], [geom], flow

        elif scenario_id == "scene_5_safe_passing":
            # Person in adjacent right corridor (bearing = +0.28 rad), passing cleanly outside corridor
            track = Track(
                track_id=5,
                class_name="Person",
                bbox_history=[(480.0, 120.0, 540.0, 260.0)],
                center_history=[(510.0, 190.0)],
                estimated_image_velocity=(0.0, 35.0),
                track_age=fid + 1,
                track_confidence=0.89,
                last_seen_frame_id=fid,
            )
            geom = ObjectGeometry(
                track_id=5,
                frame_id=fid,
                bearing=0.28,
                relative_image_velocity=(0.0, 0.04),
                motion_vector=(0.0, 0.04),
                foe_containment=False,
                geometry_confidence=0.90,
            )
            return [track], [geom], flow

        elif scenario_id == "scene_6_sensor_degradation":
            # Scene with normal flow, but IMU will drop out via SimulatedIMUReader inject_disconnect_after_s
            return [], [], flow

        else:
            raise ValueError(f"Unknown scenario_id: {scenario_id}")
