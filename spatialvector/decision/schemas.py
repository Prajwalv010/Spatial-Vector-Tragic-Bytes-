"""
Decision chain shared data contracts for SpatialVector-HMI.

These schemas define the outputs of M07 (Collision Prediction), M08 (Risk Engine),
and M09 (Corridor Policy / Haptic Policy).

Architectural rules encoded here:
  - Prediction answers geometry questions only. It does NOT say "this is dangerous."
  - RiskState is the single source of truth for warning state. No other module may
    independently decide "this is a WARNING."
  - HapticCommand is the only module output that specifies a motor pattern. M10 is a
    dumb consumer of this struct.

Field names here are final. Do not rename without updating downstream modules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Prediction:
    """Output of M07 for one track, one frame.

    Three separately-computed geometric answers:
      ttc_s:                  HOW SOON would paths cross (timing, not probability).
                              None = no finite/relevant TTC — receding, parallel, or beyond horizon.
                              IMPORTANT: TTC is a timing answer, not a collision probability.
                              Do NOT interpret a low TTC as "high probability of collision."
                              intersection_flag and prediction_confidence carry the "how sure are
                              we this matters" information; TTC alone does not.
      cpa_normalized:         Closest Point of Approach distance (normalized geometry units).
                              This is the minimum distance the two trajectories would reach
                              if projected forward, regardless of whether paths literally cross.
      miss_distance_normalized: Perpendicular miss distance when paths don't intersect.
                              Equal to cpa_normalized when intersection_flag is False.
      intersection_flag:      Do projected paths actually cross within the configured horizon?
                              True only if CPA distance is below corridor_width_normalized AND
                              CPA time is positive and within horizon_s.
      prediction_confidence:  0..1 confidence in these values. Derives from geometry_confidence
                              (M06) and trajectory stability. When low, M08 should treat this
                              prediction cautiously.
    """

    track_id: int
    frame_id: int
    ttc_s: Optional[float]              # None = no finite/relevant TTC
    cpa_normalized: float               # closest approach distance, normalized units
    miss_distance_normalized: float     # perpendicular miss if no intersection
    intersection_flag: bool             # do projected paths cross within horizon?
    prediction_confidence: float        # 0..1
    bearing: float = 0.0               # horizontal bearing (radians) from M06, for corridor assignment
    proximity_risk: float = 0.0        # 0..1 direct proximity hazard (large obstacle in path)
    expansion_rate: float = 0.0        # 1/sec looming expansion rate
    confidence_source: str = "geometry" # "geometry" | "proximity_heuristic"


@dataclass
class RiskState:
    """Output of M08 — the SINGLE SOURCE OF TRUTH for warning state.

    No other module (M07, M09, dashboard) is permitted to independently compute a
    warning state. Everything downstream reads RiskState.state and nothing else.

    state values:
      "SAFE"      — no significant risk detected
      "CAUTION"   — risk is present but below warning threshold
      "WARNING"   — risk above warning threshold, sustained through hysteresis window
      "CRITICAL"  — risk above critical threshold, sustained
      "DEGRADED"  — insufficient confidence to produce a meaningful assessment;
                    downstream should treat this as "don't know," not "SAFE"
    """

    timestamp: float
    global_risk: float                        # 0..1 aggregate risk across all tracked objects
    state: str                                # "SAFE"|"CAUTION"|"WARNING"|"CRITICAL"|"DEGRADED"
    reason_codes: list[str]                   # e.g. ["ttc_low:track_4", "intersection:track_4"]
    corridor_risks: dict[str, float]          # {"left": 0.1, "center": 0.7, "right": 0.2}
    confidence: float                         # 0..1 — how much to trust this assessment
    recommended_horizon_s: float             # how far ahead this assessment is valid for


@dataclass
class HapticCommand:
    """Output of M09 — the ONLY module permitted to decide haptic output.

    M10 (Arduino interface) is a dumb consumer of this struct. It looks up
    pattern_id in its motor table and executes it. M10 does not compute risk.

    direction values:
      "LEFT"    — route left is safest, warn about threat on right/center
      "CENTER"  — threat is approaching from ahead, route center is blocked
      "RIGHT"   — route right is safest, warn about threat on left/center
      "STOP"    — all corridors are unsafe; halt and re-assess

    urgency: 1 (informational) .. 5 (critical/stop)
    pattern_id: named pattern for M10 to look up, e.g. "LEFT_FAST", "STOP_CRITICAL"
    duration_ms: how long M10 should run this pattern before expecting the next command
    """

    timestamp: float
    direction: str                            # "LEFT"|"CENTER"|"RIGHT"|"STOP"
    urgency: int                              # 1..5
    pattern_id: str                           # named pattern, e.g. "LEFT_FAST"
    duration_ms: int
