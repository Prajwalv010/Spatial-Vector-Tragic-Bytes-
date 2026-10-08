"""
Tests for M09 — Safe-Corridor Selector + Haptic Policy

T23: Risk only on left → safest corridor is chosen (not hardcoded as "right").
T24: Risk concentrated in center → a side direction is preferred over center.
T25: High risk in all corridors → STOP/critical pattern, not a normal directional command.

All tests use hand-constructed RiskState values — no camera, no full pipeline required.

Integration rule verified in these tests:
  - pattern_id and direction always come from M09. Tests assert they are set.
  - The STOP pattern (T25) is visibly distinct from normal directional commands:
    direction="STOP", urgency=5, pattern_id="STOP_CRITICAL".

Runnable with:
    pytest tests/test_m09_corridor_policy.py -v -s
    python -u tests/test_m09_corridor_policy.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from spatialvector.decision.schemas import RiskState
from spatialvector.decision.corridor_policy import CorridorPolicy


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _make_risk_state(
    state: str = "WARNING",
    global_risk: float = 0.65,
    corridor_risks: dict | None = None,
    confidence: float = 0.85,
    timestamp: float = 0.0,
) -> RiskState:
    if corridor_risks is None:
        corridor_risks = {"left": 0.3, "center": 0.7, "right": 0.1}
    return RiskState(
        timestamp=timestamp,
        global_risk=global_risk,
        state=state,
        reason_codes=["test"],
        corridor_risks=corridor_risks,
        confidence=confidence,
        recommended_horizon_s=5.0,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_t23_risk_on_left_safest_corridor_chosen():
    """T23: Risk only on the left → the safest remaining corridor is chosen.

    IMPORTANT: We do NOT hardcode the expectation as "always right."
    We assert the chosen direction corresponds to whichever corridor has the
    minimum risk in the RiskState, which is the correct behavior.
    """
    policy = CorridorPolicy(all_unsafe_risk_threshold=0.7)

    corridor_risks = {"left": 0.75, "center": 0.20, "right": 0.15}
    rs = _make_risk_state(
        state="WARNING",
        global_risk=0.70,
        corridor_risks=corridor_risks,
    )

    cmd = policy.select(rs, timestamp=0.0)

    # The chosen direction must be the corridor with minimum risk
    min_corridor = min(corridor_risks, key=corridor_risks.get)  # type: ignore[arg-type]
    expected_direction = min_corridor.upper()

    assert cmd.direction == expected_direction, (
        f"T23: Expected direction={expected_direction} (lowest risk corridor), "
        f"got direction={cmd.direction}. corridor_risks={corridor_risks}"
    )

    # Must not direct toward the highest-risk corridor
    max_corridor = max(corridor_risks, key=corridor_risks.get)  # type: ignore[arg-type]
    assert cmd.direction != max_corridor.upper(), (
        f"T23: M09 directed toward the HIGHEST risk corridor ({max_corridor.upper()}). "
        f"corridor_risks={corridor_risks}"
    )

    # Must have a real pattern_id
    assert cmd.pattern_id, "T23: pattern_id must not be empty"
    assert cmd.urgency >= 1 and cmd.urgency <= 5, f"T23: urgency {cmd.urgency} out of range 1-5"

    print(f"  T23: direction={cmd.direction}, urgency={cmd.urgency}, pattern={cmd.pattern_id}. "
          f"(min corridor={expected_direction}). PASSED.")


def test_t24_center_risk_side_preferred():
    """T24: Risk concentrated in center → a side direction is chosen over center.

    When the center corridor has the highest risk, the output direction must be
    LEFT or RIGHT, not CENTER.
    """
    policy = CorridorPolicy(all_unsafe_risk_threshold=0.7)

    corridor_risks = {"left": 0.25, "center": 0.80, "right": 0.30}
    rs = _make_risk_state(
        state="WARNING",
        global_risk=0.70,
        corridor_risks=corridor_risks,
    )

    cmd = policy.select(rs, timestamp=0.0)

    assert cmd.direction != "CENTER", (
        f"T24: Expected M09 to choose a side corridor when center is blocked "
        f"(center_risk=0.80), but got direction={cmd.direction}. "
        f"corridor_risks={corridor_risks}"
    )
    assert cmd.direction in ("LEFT", "RIGHT"), (
        f"T24: Direction should be LEFT or RIGHT, got {cmd.direction}"
    )

    # The chosen side must be the lower-risk side
    # left=0.25 < right=0.30, so LEFT is expected
    best_side_risk = min(corridor_risks["left"], corridor_risks["right"])
    if best_side_risk == corridor_risks["left"]:
        expected_side = "LEFT"
    else:
        expected_side = "RIGHT"

    assert cmd.direction == expected_side, (
        f"T24: Wrong side chosen. Expected {expected_side} (risk={best_side_risk:.2f}), "
        f"got {cmd.direction}. corridor_risks={corridor_risks}"
    )

    print(f"  T24: direction={cmd.direction} (side preferred over blocked center). PASSED.")


def test_t25_all_corridors_blocked_stop_pattern():
    """T25: High risk in ALL corridors → distinct STOP/critical pattern.

    The STOP pattern must be visibly distinct from a normal directional command:
      - direction = "STOP"
      - urgency = 5 (maximum)
      - pattern_id contains "STOP_CRITICAL"
    This is NOT just a normal command with high urgency — it is qualitatively different.
    """
    policy = CorridorPolicy(all_unsafe_risk_threshold=0.7)

    # All corridors above the all_unsafe_threshold
    corridor_risks = {"left": 0.85, "center": 0.90, "right": 0.78}
    rs = _make_risk_state(
        state="CRITICAL",
        global_risk=0.90,
        corridor_risks=corridor_risks,
    )

    cmd = policy.select(rs, timestamp=0.0)

    assert cmd.direction == "STOP", (
        f"T25: Expected direction='STOP' when all corridors are blocked, "
        f"got direction='{cmd.direction}'. corridor_risks={corridor_risks}"
    )
    assert cmd.urgency == 5, (
        f"T25: Expected urgency=5 for STOP pattern, got urgency={cmd.urgency}"
    )
    assert "STOP" in cmd.pattern_id, (
        f"T25: pattern_id must contain 'STOP', got '{cmd.pattern_id}'. "
        f"The STOP pattern must be distinct from directional commands."
    )
    assert "CRITICAL" in cmd.pattern_id, (
        f"T25: pattern_id must contain 'CRITICAL' for all-corridors-blocked case, "
        f"got '{cmd.pattern_id}'. "
        f"This must be qualitatively different from a normal directional warning."
    )

    print(f"  T25: direction={cmd.direction}, urgency={cmd.urgency}, "
          f"pattern_id={cmd.pattern_id}. "
          f"STOP/CRITICAL is distinct from directional commands. PASSED.")


def test_t25b_stop_distinct_from_directional_high_urgency():
    """T25b: Confirm STOP_CRITICAL is NOT produced for a high-urgency directional warning.

    A single dangerous corridor with high risk should produce a directional command
    (e.g. LEFT_FAST or RIGHT_CRITICAL), NOT STOP_CRITICAL. STOP is reserved for
    the all-corridors-blocked case only.
    """
    policy = CorridorPolicy(all_unsafe_risk_threshold=0.7)

    # Only center is blocked — left and right are clear
    corridor_risks = {"left": 0.10, "center": 0.90, "right": 0.15}
    rs = _make_risk_state(
        state="CRITICAL",
        global_risk=0.88,
        corridor_risks=corridor_risks,
    )

    cmd = policy.select(rs, timestamp=0.0)

    assert cmd.direction != "STOP", (
        f"T25b: STOP pattern should NOT be issued when left and right corridors are clear. "
        f"Got direction={cmd.direction}. corridor_risks={corridor_risks}"
    )
    assert cmd.pattern_id != "STOP_CRITICAL", (
        f"T25b: STOP_CRITICAL pattern should only be issued when ALL corridors are blocked. "
        f"Here left=0.10 and right=0.15 are clear. Got pattern_id={cmd.pattern_id}"
    )

    print(f"  T25b: Directional command issued correctly (not STOP) when side corridors are clear. "
          f"direction={cmd.direction}, pattern={cmd.pattern_id}. PASSED.")


def test_t23b_degraded_state_produces_command():
    """T23b: DEGRADED risk state produces distinct DEGRADED_WARN pattern."""
    policy = CorridorPolicy()
    rs = _make_risk_state(state="DEGRADED", global_risk=0.0,
                          corridor_risks={"left": 0.0, "center": 0.0, "right": 0.0},
                          confidence=0.1)

    cmd = policy.select(rs, timestamp=0.0)

    assert cmd is not None, "T23b: Must produce a command in DEGRADED state"
    assert cmd.direction == "STOP"
    assert cmd.urgency == 2, "T23b: DEGRADED urgency must be 2"
    assert cmd.pattern_id == "DEGRADED_WARN", f"T23b: Expected DEGRADED_WARN, got {cmd.pattern_id}"
    print(f"  T23b: DEGRADED -> direction={cmd.direction}, urgency={cmd.urgency}, pattern={cmd.pattern_id}. PASSED.")


def test_t23c_safe_clear_state_produces_all_clear():
    """T23c: Confirmed SAFE state produces distinct ALL_CLEAR pattern with urgency=1."""
    policy = CorridorPolicy()
    rs = _make_risk_state(state="SAFE", global_risk=0.0,
                          corridor_risks={"left": 0.0, "center": 0.0, "right": 0.0},
                          confidence=1.0)

    cmd = policy.select(rs, timestamp=0.0)

    assert cmd is not None, "T23c: Must produce a command in SAFE clear state"
    assert cmd.direction == "STOP"
    assert cmd.urgency == 1, "T23c: SAFE clear urgency must be 1"
    assert cmd.pattern_id == "ALL_CLEAR", f"T23c: Expected ALL_CLEAR, got {cmd.pattern_id}"
    print(f"  T23c: SAFE clear -> direction={cmd.direction}, urgency={cmd.urgency}, pattern={cmd.pattern_id}. PASSED.")


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("--- Running M09 Corridor Policy Tests ---")
    print("[TEST] T23: Risk on left → safest corridor chosen (not hardcoded)...")
    test_t23_risk_on_left_safest_corridor_chosen()
    print("  -> T23 PASSED")

    print("[TEST] T24: Center risk → side corridor preferred...")
    test_t24_center_risk_side_preferred()
    print("  -> T24 PASSED")

    print("[TEST] T25: All corridors blocked → STOP_CRITICAL pattern...")
    test_t25_all_corridors_blocked_stop_pattern()
    print("  -> T25 PASSED")

    print("[TEST] T25b: STOP_CRITICAL NOT issued for single-corridor danger...")
    test_t25b_stop_distinct_from_directional_high_urgency()
    print("  -> T25b PASSED")

    print("[TEST] T23b: DEGRADED state still produces a command...")
    test_t23b_degraded_state_produces_command()
    print("  -> T23b PASSED")

    print("\nALL M09 TESTS PASSED.")
