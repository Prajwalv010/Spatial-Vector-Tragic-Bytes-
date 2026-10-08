"""
Tests for M08 — Risk Engine & State Machine

T20: Approaching → close → receding sequence → state transitions in expected order.
T21: Same closest distance, different trajectories (crossing vs. parallel) →
     crossing produces higher risk (core thesis: distance ≠ danger).
T22: Noisy sequence hovering near threshold → state does NOT flicker (hysteresis works).

All tests use hand-constructed Prediction sequences injected directly into M08.
No camera, no YOLO, no live pipeline required (Gate D).

Runnable with:
    pytest tests/test_m08_risk_engine.py -v -s
    python -u tests/test_m08_risk_engine.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from spatialvector.decision.schemas import Prediction
from spatialvector.decision.risk_engine import RiskEngine


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _make_prediction(
    track_id: int = 1,
    frame_id: int = 0,
    ttc_s: Optional[float] = None,
    cpa_normalized: float = 0.5,
    miss_distance_normalized: float = 0.5,
    intersection_flag: bool = False,
    prediction_confidence: float = 0.85,
) -> Prediction:
    return Prediction(
        track_id=track_id,
        frame_id=frame_id,
        ttc_s=ttc_s,
        cpa_normalized=cpa_normalized,
        miss_distance_normalized=miss_distance_normalized,
        intersection_flag=intersection_flag,
        prediction_confidence=prediction_confidence,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_t20_state_transitions_in_order():
    """T20: Feed approaching → close → receding sequence.

    Expected state progression:
      SAFE → (may hit CAUTION) → WARNING/CRITICAL (at closest) → back toward SAFE.

    We assert the sequence of distinct states seen is ordered (no skipping from
    SAFE to CRITICAL and back in one frame — hysteresis prevents that).
    """
    engine = RiskEngine(
        hysteresis_frames_up=2,    # faster for test
        hysteresis_frames_down=3,
        state_thresholds={"caution": 0.3, "warning": 0.5, "critical": 0.85},
    )

    states_seen = []

    # Phase 1: approaching (15 frames) — risk rises
    for i in range(15):
        approach_fraction = i / 14.0  # 0 → 1
        ttc = max(0.2, 4.0 - approach_fraction * 3.5)   # 4.0s → 0.5s
        cpa = max(0.02, 0.4 - approach_fraction * 0.35)  # 0.4 → 0.05
        miss = cpa
        intersect = cpa < 0.10

        pred = _make_prediction(
            frame_id=i,
            ttc_s=ttc if intersect else None,
            cpa_normalized=cpa,
            miss_distance_normalized=miss,
            intersection_flag=intersect,
            prediction_confidence=0.88,
        )
        rs = engine.update([pred], timestamp=float(i))
        states_seen.append(rs.state)

    # Phase 2: receding (15 frames) — risk falls
    for i in range(15):
        recede_fraction = i / 14.0
        cpa = 0.05 + recede_fraction * 0.35   # 0.05 → 0.4
        miss = cpa

        pred = _make_prediction(
            frame_id=15 + i,
            ttc_s=None,
            cpa_normalized=cpa,
            miss_distance_normalized=miss,
            intersection_flag=False,
            prediction_confidence=0.88,
        )
        rs = engine.update([pred], timestamp=float(15 + i))
        states_seen.append(rs.state)

    # Assertions: must have seen at least CAUTION or WARNING at peak
    state_order_value = {"SAFE": 0, "CAUTION": 1, "WARNING": 2, "CRITICAL": 3, "DEGRADED": -1}
    peak_state = max((s for s in states_seen if s != "DEGRADED"),
                     key=lambda s: state_order_value.get(s, 0),
                     default="SAFE")

    assert peak_state in ("CAUTION", "WARNING", "CRITICAL"), (
        f"T20: Expected to reach at least CAUTION during approach. Peak state was {peak_state}. "
        f"States sequence: {states_seen}"
    )

    # Final state must have de-escalated back toward SAFE (not stuck at WARNING/CRITICAL)
    final_state = states_seen[-1]
    final_value = state_order_value.get(final_state, 0)
    peak_value = state_order_value.get(peak_state, 0)
    assert final_value < peak_value, (
        f"T20: State did not de-escalate after receding. Peak={peak_state}, Final={final_state}. "
        f"States: {states_seen}"
    )

    # No impossible state skips (e.g. SAFE directly to CRITICAL in one step without hysteresis)
    # Check that adjacent states in the escalation phase differ by at most one level
    escalation_states = states_seen[:15]
    for i in range(len(escalation_states) - 1):
        a = state_order_value.get(escalation_states[i], 0)
        b = state_order_value.get(escalation_states[i + 1], 0)
        assert abs(b - a) <= 1, (
            f"T20: Impossible state jump from {escalation_states[i]} to {escalation_states[i+1]} "
            f"at frame {i}. Hysteresis should prevent skipping more than one level."
        )

    print(f"  T20: States = {states_seen}")
    print(f"  T20: Peak={peak_state}, Final={final_state}. PASSED.")


def test_t21_same_distance_different_risk():
    """T21: Core thesis test — distance is NOT the same as danger.

    Two scenarios with the SAME closest distance (same CPA):
      Scenario A: crossing trajectory (intersection_flag=True) → higher risk
      Scenario B: parallel trajectory (intersection_flag=False) → lower risk

    Assert: Scenario A risk > Scenario B risk.
    """
    # Use a fresh engine each time (no state carry-over)
    engine_a = RiskEngine(hysteresis_frames_up=1, hysteresis_frames_down=1)
    engine_b = RiskEngine(hysteresis_frames_up=1, hysteresis_frames_down=1)

    same_cpa = 0.08   # same closest distance in both scenarios

    # Scenario A: crossing — paths intersect, low TTC
    pred_a = _make_prediction(
        track_id=1,
        frame_id=0,
        ttc_s=1.5,                  # paths will cross in 1.5 seconds
        cpa_normalized=same_cpa,
        miss_distance_normalized=0.0,
        intersection_flag=True,     # CROSSING
        prediction_confidence=0.88,
    )

    # Scenario B: parallel — same CPA but paths never intersect
    pred_b = _make_prediction(
        track_id=1,
        frame_id=0,
        ttc_s=None,                  # no TTC — parallel
        cpa_normalized=same_cpa,
        miss_distance_normalized=same_cpa,
        intersection_flag=False,     # PARALLEL
        prediction_confidence=0.88,
    )

    rs_a = engine_a.update([pred_a], timestamp=0.0)
    rs_b = engine_b.update([pred_b], timestamp=0.0)

    assert rs_a.global_risk > rs_b.global_risk, (
        f"T21: CRITICAL THESIS FAILURE — crossing (risk={rs_a.global_risk:.4f}) should be "
        f"riskier than parallel (risk={rs_b.global_risk:.4f}) at the SAME CPA={same_cpa:.3f}. "
        f"Distance alone does not determine danger."
    )

    print(f"  T21: crossing risk={rs_a.global_risk:.4f} > parallel risk={rs_b.global_risk:.4f} "
          f"(same CPA={same_cpa}). Core thesis validated. PASSED.")


def test_t22_hysteresis_prevents_flickering():
    """T22: Noisy sequence hovering near threshold → state must NOT flicker.

    Feed a sequence of predictions whose risk alternates just above and below the
    CAUTION threshold (0.3). Count state transitions — must be small (≤ 3 over
    the whole sequence), not one-per-frame.
    """
    engine = RiskEngine(
        hysteresis_frames_up=3,
        hysteresis_frames_down=5,
        state_thresholds={"caution": 0.3, "warning": 0.6, "critical": 0.85},
    )

    # Noisy sequence: risk alternates around the caution threshold (0.3)
    # This would cause 1-per-frame flickering without hysteresis.
    rng = np.random.default_rng(42)
    noisy_risks = [0.28 + rng.uniform(-0.06, 0.06) for _ in range(30)]   # ~0.22..0.34

    states_seen = []
    for i, target_risk in enumerate(noisy_risks):
        # Construct a prediction that would produce approximately target_risk in the engine.
        # target_risk ≈ w_miss * miss_score (simplest lever: vary miss_distance).
        # miss_score = 1 - miss/0.5, so miss = (1 - target_risk/0.3) * 0.5
        miss = max(0.01, (1.0 - target_risk / 0.3) * 0.5)

        pred = _make_prediction(
            frame_id=i,
            ttc_s=None,
            cpa_normalized=miss,
            miss_distance_normalized=miss,
            intersection_flag=False,
            prediction_confidence=0.85,
        )
        rs = engine.update([pred], timestamp=float(i))
        states_seen.append(rs.state)

    # Count state transitions
    transitions = sum(1 for i in range(len(states_seen) - 1)
                      if states_seen[i] != states_seen[i + 1])

    assert transitions <= 3, (
        f"T22: Too many state transitions ({transitions}) in a noisy sequence. "
        f"Hysteresis should keep this to ≤ 3 over 30 frames near threshold. "
        f"States: {states_seen}"
    )

    print(f"  T22: {transitions} state transitions over 30 noisy frames (<= 3 required). "
          f"States: {states_seen}")
    print("  T22: Hysteresis working correctly. PASSED.")


def test_t22b_degraded_when_low_confidence():
    """T22b: If all predictions have very low confidence → state = DEGRADED."""
    engine = RiskEngine(degraded_confidence_threshold=0.25)

    # Very low confidence predictions — engine should declare DEGRADED
    low_conf_preds = [
        _make_prediction(track_id=i, prediction_confidence=0.1)
        for i in range(3)
    ]
    rs = engine.update(low_conf_preds, fallback_active=True, timestamp=0.0)

    assert rs.state == "DEGRADED", (
        f"T22b: Expected DEGRADED state for low-confidence + fallback_active=True input. "
        f"Got state={rs.state}, confidence={rs.confidence:.4f}"
    )
    print(f"  T22b: DEGRADED correctly issued for low-confidence predictions. PASSED.")


def test_t22c_empty_scene_healthy_pipeline_is_safe():
    """T22c: Empty scene with healthy pipeline must be SAFE, not DEGRADED.

    When there are 0 predictions:
      - If fallback_active=False (sensors healthy, clear path): state must be SAFE, global_risk=0.0.
      - If fallback_active=True (sensors failed, degraded): state must be DEGRADED.
    """
    engine = RiskEngine(degraded_confidence_threshold=0.25)

    # Healthy sensors + empty hallway -> SAFE
    rs_healthy = engine.update([], fallback_active=False, timestamp=0.0)
    assert rs_healthy.state == "SAFE", (
        f"T22c: Empty scene with healthy sensors should be SAFE, got {rs_healthy.state} "
        f"(confidence={rs_healthy.confidence}, reasons={rs_healthy.escalation_reasons})"
    )
    assert rs_healthy.global_risk == 0.0, f"T22c: Expected risk 0.0, got {rs_healthy.global_risk}"
    assert rs_healthy.confidence == 1.0, f"T22c: Expected confidence 1.0, got {rs_healthy.confidence}"

    # Degraded sensors + empty predictions -> DEGRADED
    rs_degraded = engine.update([], fallback_active=True, timestamp=1.0)
    assert rs_degraded.state == "DEGRADED", (
        f"T22c: Fallback active should yield DEGRADED, got {rs_degraded.state}"
    )
    assert rs_degraded.confidence == 0.0, f"T22c: Expected confidence 0.0, got {rs_degraded.confidence}"
    print("  T22c: Empty scene produces SAFE when healthy and DEGRADED when sensor fails. PASSED.")


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("--- Running M08 Risk Engine Tests ---")
    print("[TEST] T20: State machine transitions in correct order...")
    test_t20_state_transitions_in_order()
    print("  -> T20 PASSED")

    print("[TEST] T21: Same distance, different risk (core thesis)...")
    test_t21_same_distance_different_risk()
    print("  -> T21 PASSED")

    print("[TEST] T22: Hysteresis prevents flickering at threshold...")
    test_t22_hysteresis_prevents_flickering()
    print("  -> T22 PASSED")

    print("[TEST] T22b: Low confidence → DEGRADED state...")
    test_t22b_degraded_when_low_confidence()
    print("  -> T22b PASSED")

    print("[TEST] T22c: Empty scene healthy vs degraded...")
    test_t22c_empty_scene_healthy_pipeline_is_safe()
    print("  -> T22c PASSED")

    print("\nALL M08 TESTS PASSED.")

