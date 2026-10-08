import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from spatialvector.decision.schemas import HapticCommand, Prediction, RiskState
from spatialvector.motion.schemas import MotionState
from spatialvector.perception.schemas import Track
from run_camera_prediction_viewer import PredictionViewer


@pytest.fixture
def dummy_viewer():
    v = PredictionViewer.__new__(PredictionViewer)
    v.show_hud = True
    v.show_telemetry = True
    v.view_mode = "AUTO"
    v.tracker_type = "bytetrack"
    v.predictor = None
    v.win_name = "TestWindow"
    return v


def test_autoscale_render_portrait_frame(dummy_viewer):
    """Verify that portrait frame renders properly across AUTO, MAX, and STUDIO modes."""
    portrait_img = np.zeros((540, 304, 3), dtype=np.uint8)

    tracks = [
        Track(
            track_id=511,
            class_name="person",
            bbox_history=[(20.0, 150.0, 140.0, 480.0)],
            center_history=[(80.0, 315.0)],
            estimated_image_velocity=(0.0, 0.0),
            track_age=15,
            track_confidence=0.88,
            last_seen_frame_id=100,
        ),
        Track(
            track_id=503,
            class_name="person",
            bbox_history=[(110.0, 160.0, 230.0, 490.0)],
            center_history=[(170.0, 325.0)],
            estimated_image_velocity=(0.0, 0.0),
            track_age=25,
            track_confidence=0.91,
            last_seen_frame_id=100,
        ),
        Track(
            track_id=508,
            class_name="chair",
            bbox_history=[(90.0, 230.0, 210.0, 450.0)],
            center_history=[(150.0, 340.0)],
            estimated_image_velocity=(0.0, 0.0),
            track_age=10,
            track_confidence=0.78,
            last_seen_frame_id=100,
        ),
    ]

    predictions = [
        Prediction(
            track_id=511,
            frame_id=100,
            ttc_s=None,
            cpa_normalized=0.85,
            miss_distance_normalized=0.85,
            intersection_flag=False,
            prediction_confidence=0.70,
            proximity_risk=0.45,
        ),
        Prediction(
            track_id=503,
            frame_id=100,
            ttc_s=2.0,
            cpa_normalized=0.05,
            miss_distance_normalized=0.05,
            intersection_flag=True,
            prediction_confidence=0.85,
            proximity_risk=0.80,
        ),
        Prediction(
            track_id=508,
            frame_id=100,
            ttc_s=2.4,
            cpa_normalized=0.05,
            miss_distance_normalized=0.05,
            intersection_flag=True,
            prediction_confidence=0.80,
            proximity_risk=0.75,
        ),
    ]

    risk = RiskState(
        timestamp=0.0,
        global_risk=0.79,
        state="WARNING",
        reason_codes=["proximity:0.43:track_511", "ttc_low:2.0s:track_503"],
        corridor_risks={"left": 0.0, "center": 1.0, "right": 0.0},
        confidence=0.55,
        recommended_horizon_s=5.0,
    )

    cmd = HapticCommand(
        timestamp=0.0,
        direction="LEFT",
        urgency=4,
        pattern_id="LEFT_FAST",
        duration_ms=200,
    )

    motion = MotionState(
        timestamp=0.0,
        ego_rotation=(0.0, 0.0, 0.0),
        corrected_flow=[],
        foe_x=float("nan"),
        foe_y=float("nan"),
        flow_quality=1.0,
        foe_confidence=0.5,
        motion_quality="OK",
        fallback_active=False,
    )

    # Test AUTO mode with standard 1280x720 canvas
    dummy_viewer.view_mode = "AUTO"
    canvas_auto = dummy_viewer.render_overlay(
        img=portrait_img,
        tracks=tracks,
        predictions=predictions,
        risk=risk,
        cmd=cmd,
        motion=motion,
        fps=20.5,
        frame_id=100,
        target_w=1280,
        target_h=720,
    )
    assert canvas_auto.shape == (720, 1280, 3)

    # Test MAX mode (full video focus, floating HUD)
    dummy_viewer.view_mode = "MAX"
    canvas_max = dummy_viewer.render_overlay(
        img=portrait_img,
        tracks=tracks,
        predictions=predictions,
        risk=risk,
        cmd=cmd,
        motion=motion,
        fps=20.5,
        frame_id=101,
        target_w=860,
        target_h=920,
    )
    assert canvas_max.shape == (920, 860, 3)

    # Test STUDIO mode (fixed sidebars)
    dummy_viewer.view_mode = "STUDIO"
    canvas_studio = dummy_viewer.render_overlay(
        img=portrait_img,
        tracks=tracks,
        predictions=predictions,
        risk=risk,
        cmd=cmd,
        motion=motion,
        fps=20.5,
        frame_id=102,
        target_w=1280,
        target_h=720,
    )
    assert canvas_studio.shape == (720, 1280, 3)


def test_autoscale_landscape_frame(dummy_viewer):
    """Verify that landscape frame renders properly."""
    landscape_img = np.zeros((720, 1280, 3), dtype=np.uint8)
    risk = RiskState(
        timestamp=0.0,
        global_risk=0.10,
        state="SAFE",
        reason_codes=[],
        corridor_risks={"left": 0.0, "center": 0.0, "right": 0.0},
        confidence=0.90,
        recommended_horizon_s=5.0,
    )
    cmd = HapticCommand(timestamp=0.0, direction="NONE", urgency=0, pattern_id="OFF", duration_ms=0)
    motion = MotionState(
        timestamp=0.0,
        ego_rotation=(0.0, 0.0, 0.0),
        corrected_flow=[],
        foe_x=float("nan"),
        foe_y=float("nan"),
        flow_quality=1.0,
        foe_confidence=0.5,
        motion_quality="OK",
        fallback_active=False,
    )

    canvas = dummy_viewer.render_overlay(
        img=landscape_img,
        tracks=[],
        predictions=[],
        risk=risk,
        cmd=cmd,
        motion=motion,
        fps=30.0,
        frame_id=1,
        target_w=1280,
        target_h=720,
    )
    assert canvas.shape == (720, 1280, 3)


def test_badge_collision_avoidance(dummy_viewer):
    """Verify that multiple tracks with identical Y positions avoid label collisions without crashing."""
    portrait_img = np.zeros((540, 304, 3), dtype=np.uint8)

    # 4 overlapping tracks at almost the exact same Y position (clumped scenario)
    tracks = [
        Track(
            track_id=1,
            class_name="person",
            bbox_history=[(50.0, 180.0, 150.0, 400.0)],
            center_history=[(100.0, 290.0)],
            estimated_image_velocity=(0.0, 0.0),
            track_age=10,
            track_confidence=0.9,
            last_seen_frame_id=5,
        ),
        Track(
            track_id=2,
            class_name="person",
            bbox_history=[(60.0, 182.0, 160.0, 410.0)],
            center_history=[(110.0, 295.0)],
            estimated_image_velocity=(0.0, 0.0),
            track_age=10,
            track_confidence=0.9,
            last_seen_frame_id=5,
        ),
        Track(
            track_id=3,
            class_name="chair",
            bbox_history=[(70.0, 185.0, 170.0, 380.0)],
            center_history=[(120.0, 280.0)],
            estimated_image_velocity=(0.0, 0.0),
            track_age=10,
            track_confidence=0.8,
            last_seen_frame_id=5,
        ),
    ]

    predictions = [
        Prediction(track_id=1, frame_id=5, ttc_s=1.5, cpa_normalized=0.04, miss_distance_normalized=0.04, intersection_flag=True, prediction_confidence=0.9),
        Prediction(track_id=2, frame_id=5, ttc_s=1.8, cpa_normalized=0.05, miss_distance_normalized=0.05, intersection_flag=True, prediction_confidence=0.85),
        Prediction(track_id=3, frame_id=5, ttc_s=2.2, cpa_normalized=0.06, miss_distance_normalized=0.06, intersection_flag=False, prediction_confidence=0.8),
    ]

    risk = RiskState(
        timestamp=0.0,
        global_risk=0.85,
        state="CRITICAL",
        reason_codes=["ttc_low:1.5s:track_1"],
        corridor_risks={"left": 0.0, "center": 1.0, "right": 0.0},
        confidence=0.9,
        recommended_horizon_s=5.0,
    )

    cmd = HapticCommand(timestamp=0.0, direction="STOP", urgency=5, pattern_id="STOP_CRITICAL", duration_ms=300)
    motion = MotionState(
        timestamp=0.0,
        ego_rotation=(0.0, 0.0, 0.0),
        corrected_flow=[],
        foe_x=float("nan"),
        foe_y=float("nan"),
        flow_quality=1.0,
        foe_confidence=0.5,
        motion_quality="OK",
        fallback_active=False,
    )

    canvas = dummy_viewer.render_overlay(
        img=portrait_img,
        tracks=tracks,
        predictions=predictions,
        risk=risk,
        cmd=cmd,
        motion=motion,
        fps=20.0,
        frame_id=5,
        target_w=860,
        target_h=920,
    )
    assert canvas.shape == (920, 860, 3)

