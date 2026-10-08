"""SpatialVector-HMI — Demo Day Scenario Definitions (Section 12 / Gate G).

Encodes the 6 thesis-proving demonstration scenes into structured, checkable expectations.
Used by validate_scenarios.py and run_gate_g.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class DemoScenario:
    """Structured contract defining expected behavior for one demo scene."""

    id: str
    name: str
    physical_action: str
    source: str                                  # path to a recorded clip, OR "synthetic:<generator_name>"
    expect_intersection: Optional[bool]          # None = don't assert on this field
    expect_state_in: List[str]                   # acceptable steady-state RiskState.state values
    expect_haptic_pattern_prefix: Optional[str]  # e.g. "ALL_CLEAR", "STOP", "CENTER", "LEFT"
    expect_min_tracks: int                       # fail loudly if fewer tracks are ever observed (anti-vacuous rule)
    judge_takeaway: str                          # core thesis takeaway for judges


SCENARIOS: List[DemoScenario] = [
    DemoScenario(
        id="scene_1_baseline",
        name="Baseline — open space",
        physical_action="Walk in open space, no obstacles",
        source="tests/fixtures/demo_scene1_baseline.mp4",
        expect_intersection=None,
        expect_state_in=["SAFE"],
        expect_haptic_pattern_prefix="ALL_CLEAR",
        expect_min_tracks=0,  # legitimately 0 is correct — empty corridor
        judge_takeaway="The system does not constantly alert.",
    ),
    DemoScenario(
        id="scene_2_parallel_wall",
        name="Parallel wall — close but safe",
        physical_action="Walk beside a wall at close but safe lateral offset",
        source="tests/fixtures/demo_scene2_parallel_wall.mp4",
        expect_intersection=False,
        expect_state_in=["SAFE", "CAUTION"],
        expect_haptic_pattern_prefix=None,  # silent or low caution — check state, not haptic
        expect_min_tracks=1,                # must actually detect the wall/object
        judge_takeaway="Nearby is not automatically dangerous.",
    ),
    DemoScenario(
        id="scene_3_turn_toward_wall",
        name="Turn toward wall — trajectory shift",
        physical_action="Rotate body toward the obstacle",
        source="tests/fixtures/demo_scene3_turn_toward_wall.mp4",
        expect_intersection=True,
        expect_state_in=["WARNING", "CRITICAL"],
        expect_haptic_pattern_prefix="LEFT",  # M09 selects clear side corridor to steer user away from center obstacle
        expect_min_tracks=1,
        judge_takeaway="Risk changes because trajectory changes, not distance.",
    ),
    DemoScenario(
        id="scene_4_crossing_person",
        name="Crossing person — dynamic intercept",
        physical_action="A person walks across the future path",
        source="tests/fixtures/demo_scene4_crossing_person.mp4",
        expect_intersection=True,
        expect_state_in=["WARNING", "CRITICAL"],
        expect_haptic_pattern_prefix="LEFT",  # steering clear of threat on right/center
        expect_min_tracks=1,
        judge_takeaway="Dynamic obstacles are predicted, not just detected.",
    ),
    DemoScenario(
        id="scene_5_safe_passing",
        name="Safe passing — adjacent corridor",
        physical_action="A person passes just outside the safety corridor",
        source="tests/fixtures/demo_scene5_safe_passing.mp4",
        expect_intersection=False,
        expect_state_in=["SAFE", "CAUTION"],
        expect_haptic_pattern_prefix=None,
        expect_min_tracks=1,
        judge_takeaway="Same distance can produce different decisions depending on trajectory.",
    ),
    DemoScenario(
        id="scene_6_sensor_degradation",
        name="Sensor degradation — IMU dropout",
        physical_action="Briefly interrupt IMU / create low-flow scene",
        source="synthetic:imu_dropout",
        expect_intersection=None,
        expect_state_in=["DEGRADED"],
        expect_haptic_pattern_prefix="STOP",  # STOP_SLOW degraded pattern
        expect_min_tracks=0,
        judge_takeaway="System exposes uncertainty rather than hiding it.",
    ),
]
