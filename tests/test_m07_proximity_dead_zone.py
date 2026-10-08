"""
Regression test for Bug 1: Proximity dead zone elimination and M07/M08 corridor agreement.

Verifies that:
1. An obstacle at bearing = 0.40 rad (inside [0.35, 0.5236] rad / ~23 deg) with proximity_scale = 0.35
   now produces proximity_risk > 0.0 (previously returned 0.0 due to dead zone).
2. The corridor assigned to this prediction by RiskEngine._assign_corridor_from_prediction
   is 'center' by construction (sharing corridor_constants.CORRIDOR_CENTER_BEARING_RAD = pi/6).
3. Close-range detection at conversational distance (proximity_scale = 0.25, bearing = 0.10)
   correctly scores proximity_risk > 0.0 with the tuned 0.22 threshold (previously 0.30).
"""

import math
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from spatialvector.motion.schemas import ObjectGeometry
from spatialvector.perception.schemas import Track
from spatialvector.decision.prediction import CollisionPredictor
from spatialvector.decision.risk_engine import RiskEngine
from spatialvector.decision.corridor_constants import CORRIDOR_CENTER_BEARING_RAD


def test_proximity_dead_zone_elimination():
    predictor = CollisionPredictor()
    risk_engine = RiskEngine()

    # Track with bearing 0.40 rad (~22.9°), which was in the dead zone:
    # 0.35 < 0.40 < 0.5236 (pi/6)
    # Old code checked abs(bearing) < 0.35 and proximity_scale > 0.30 -> False!
    # Old offcenter branch checked proximity_scale > 0.45 -> False!
    # Resulted in proximity_risk = 0.0.
    track = Track(
        track_id=1,
        class_name="person",
        bbox_history=[(100, 100, 200, 268)],  # height = 168 / 480 = 0.35
        center_history=[(150.0, 184.0)],
        estimated_image_velocity=(0.0, 0.0),
        bbox_scale=0.35,
        track_age=10,
        track_confidence=0.90,
        last_seen_frame_id=1,
    )
    geom = ObjectGeometry(
        track_id=1,
        frame_id=1,
        bearing=0.40,
        relative_image_velocity=(0.0, 0.0),
        motion_vector=(0.0, 0.0),
        foe_containment=False,
        geometry_confidence=0.90,
    )

    p = predictor.predict(geom, track, frame_id=1)

    # 1. Proximity risk must be non-zero
    assert p.proximity_risk > 0.0, (
        f"Obstacle in center corridor at bearing 0.40 rad with scale 0.35 "
        f"must score proximity_risk > 0.0, got {p.proximity_risk}"
    )

    # 2. Risk engine corridor assignment must agree that bearing 0.40 is 'center'
    assigned_corridor = risk_engine._assign_corridor_from_prediction(p)
    assert assigned_corridor == "center", (
        f"Expected corridor 'center' for bearing {p.bearing:.3f} rad "
        f"(boundary is {CORRIDOR_CENTER_BEARING_RAD:.3f} rad), got {assigned_corridor}"
    )


def test_proximity_conversational_distance_triggers():
    predictor = CollisionPredictor()

    # Person standing in front at conversational distance: scale = 0.25 (> 0.22 threshold)
    track = Track(
        track_id=2,
        class_name="person",
        bbox_history=[(100, 100, 200, 220)],
        center_history=[(150.0, 160.0)],
        estimated_image_velocity=(0.0, 0.0),
        bbox_scale=0.25,
        track_age=10,
        track_confidence=0.90,
        last_seen_frame_id=1,
    )
    geom = ObjectGeometry(
        track_id=2,
        frame_id=1,
        bearing=0.10,
        relative_image_velocity=(0.0, 0.0),
        motion_vector=(0.0, 0.0),
        foe_containment=False,
        geometry_confidence=0.90,
    )

    p = predictor.predict(geom, track, frame_id=1)
    assert p.proximity_risk > 0.0, f"Scale 0.25 must trigger proximity risk > 0, got {p.proximity_risk}"
