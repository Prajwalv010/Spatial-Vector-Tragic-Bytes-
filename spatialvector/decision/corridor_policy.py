"""
M09 — Safe-Corridor Selector + Haptic Policy

The ONLY module permitted to decide direction, urgency, and pattern_id for haptic output.
M08 describes risk; M09 turns that into a concrete motor command. M10 is a dumb consumer.

Responsibilities:
  1. Select the safest corridor from RiskState.corridor_risks.
  2. Compute urgency from risk magnitude AND trend (rising risk is more urgent
     than stationary risk at the same absolute level).
  3. Issue a distinct STOP/CRITICAL pattern when all corridors are unsafe.
  4. Map (direction, urgency) to a named pattern_id that M10 can look up.

Rules enforced in code:
  - Reads corridor_risks from RiskState as-is. Does NOT re-derive its own
    "which corridor is dangerous" logic — that's M08's job.
  - Direction = lowest-risk corridor. Not a binary present/absent check.
  - When all corridors exceed all_unsafe_risk_threshold → STOP pattern.
    The STOP pattern is visibly distinct from a directional warning: direction="STOP",
    pattern_id="STOP_CRITICAL", urgency=5.
  - Urgency 1-5 is a rising scale. Urgency 5 = critical/stop. Pattern names encode
    both direction and urgency tier so M10 can tune vibration intensity/frequency.

Config values (from config/default.yaml, section "corridor_policy"):
  num_corridors:             3     — left/center/right
  urgency_levels:            5
  all_unsafe_risk_threshold: 0.7   — if every corridor ≥ this, issue STOP pattern
  trend_window_frames:       5     — rolling window for rise/fall trend

Pattern ID naming convention (for M10 lookup table):
  "{DIRECTION}_{TIER}"
  where TIER = "SLOW" (urgency 1-2), "MED" (urgency 3), "FAST" (urgency 4), "CRITICAL" (urgency 5)
  Examples: "LEFT_SLOW", "CENTER_MED", "RIGHT_FAST", "STOP_CRITICAL"

Duration policy:
  - STOP: 300ms (short, repeated)
  - urgency 4-5: 200ms
  - urgency 1-3: 400ms (longer, lower frequency)
"""

from __future__ import annotations

import logging
import time
from collections import deque
from typing import Optional

import numpy as np

from spatialvector.decision.schemas import RiskState, HapticCommand

logger = logging.getLogger(__name__)

# Default config values — mirror default.yaml corridor_policy section
_DEFAULT_ALL_UNSAFE_THRESHOLD = 0.7
_DEFAULT_TREND_WINDOW = 5
_DEFAULT_URGENCY_LEVELS = 5

# Pattern ID mapping: (direction, urgency_tier) → pattern_id
# urgency_tier: 1-2="SLOW", 3="MED", 4="FAST", 5="CRITICAL"
_URGENCY_TIER = {1: "SLOW", 2: "SLOW", 3: "MED", 4: "FAST", 5: "CRITICAL"}

# Duration per urgency level (ms)
_DURATION_MS = {1: 400, 2: 400, 3: 300, 4: 200, 5: 150}


class CorridorPolicy:
    """Selects the safest corridor and generates a HapticCommand.

    Maintains a short rolling history of corridor risks to detect rising trends,
    which increases urgency beyond what the raw risk number alone would suggest.

    Usage:
        policy = CorridorPolicy()
        cmd = policy.select(risk_state)
    """

    def __init__(
        self,
        all_unsafe_risk_threshold: float = _DEFAULT_ALL_UNSAFE_THRESHOLD,
        trend_window_frames: int = _DEFAULT_TREND_WINDOW,
        urgency_levels: int = _DEFAULT_URGENCY_LEVELS,
    ):
        """
        Args:
            all_unsafe_risk_threshold: if all corridors exceed this, issue STOP.
                Configured value: 0.7.
            trend_window_frames: rolling window size for rise/fall trend detection.
                Configured value: 5.
            urgency_levels: number of urgency levels (1..N). Configured value: 5.
        """
        self.all_unsafe_threshold = all_unsafe_risk_threshold
        self.trend_window = trend_window_frames
        self.urgency_levels = urgency_levels

        # Rolling risk history per corridor for trend detection
        self._history: dict[str, deque[float]] = {
            "left": deque(maxlen=trend_window_frames),
            "center": deque(maxlen=trend_window_frames),
            "right": deque(maxlen=trend_window_frames),
        }

    def select(self, risk_state: RiskState, timestamp: Optional[float] = None) -> HapticCommand:
        """Select the safest corridor and generate a HapticCommand.

        Args:
            risk_state: RiskState from M08 — the single source of truth.
            timestamp: override timestamp (defaults to time.monotonic()).

        Returns:
            HapticCommand describing which motor pattern M10 should fire.
        """
        ts = timestamp if timestamp is not None else time.monotonic()
        corridor_risks = risk_state.corridor_risks

        # Update rolling history
        for corr in ("left", "center", "right"):
            self._history[corr].append(corridor_risks.get(corr, 0.0))

        # --- DEGRADED state: produce a distinct safe fallback command ---
        if risk_state.state == "DEGRADED":
            return HapticCommand(
                timestamp=ts,
                direction="STOP",
                urgency=2,
                pattern_id="DEGRADED_WARN",
                duration_ms=350,
            )

        # --- SAFE state: no command (send a "clear" pulse) ---
        if risk_state.state == "SAFE" and risk_state.global_risk < 0.1:
            return HapticCommand(
                timestamp=ts,
                direction="STOP",
                urgency=1,
                pattern_id="ALL_CLEAR",   # M10 interprets urgency=1 direction=STOP as "all clear"
                duration_ms=200,
            )

        # --- All-corridors-blocked check ---
        left_risk = corridor_risks.get("left", 0.0)
        center_risk = corridor_risks.get("center", 0.0)
        right_risk = corridor_risks.get("right", 0.0)

        all_unsafe = (
            left_risk >= self.all_unsafe_threshold
            and center_risk >= self.all_unsafe_threshold
            and right_risk >= self.all_unsafe_threshold
        )

        if all_unsafe:
            return HapticCommand(
                timestamp=ts,
                direction="STOP",
                urgency=5,
                pattern_id="STOP_CRITICAL",
                duration_ms=150,
            )

        # --- Normal case: pick lowest-risk corridor ---
        risks = {"left": left_risk, "center": center_risk, "right": right_risk}
        best_corridor = min(risks, key=risks.get)    # type: ignore[arg-type]
        direction = best_corridor.upper()

        # --- Urgency computation ---
        # Base urgency from global_risk level
        base_urgency = self._risk_to_base_urgency(risk_state.global_risk, risk_state.state)

        # Trend boost: if the highest-risk corridor's risk is rising, increase urgency
        worst_corridor = max(risks, key=risks.get)   # type: ignore[arg-type]
        trend = self._compute_trend(worst_corridor)
        urgency_boost = 1 if trend > 0.05 else 0     # rising trend adds 1 urgency level

        urgency = int(np.clip(base_urgency + urgency_boost, 1, self.urgency_levels))

        # --- Pattern ID ---
        tier = _URGENCY_TIER.get(urgency, "MED")
        pattern_id = f"{direction}_{tier}"
        duration_ms = _DURATION_MS.get(urgency, 300)

        logger.debug(
            f"[M09] direction={direction} urgency={urgency} pattern={pattern_id} "
            f"global_risk={risk_state.global_risk:.3f} trend={trend:+.4f}"
        )

        return HapticCommand(
            timestamp=ts,
            direction=direction,
            urgency=urgency,
            pattern_id=pattern_id,
            duration_ms=duration_ms,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _risk_to_base_urgency(self, global_risk: float, state: str) -> int:
        """Map global_risk and state to a base urgency level 1..5.

        State takes priority over raw risk for the CRITICAL case:
          SAFE     → 1
          CAUTION  → 2
          WARNING  → 3-4 (risk-dependent)
          CRITICAL → 5
        """
        if state == "CRITICAL":
            return 5
        elif state == "WARNING":
            # Scale 3..4 within warning band (0.6..0.85)
            return 4 if global_risk >= 0.75 else 3
        elif state == "CAUTION":
            return 2
        else:
            return 1

    def _compute_trend(self, corridor: str) -> float:
        """Compute the risk trend for a corridor over the rolling window.

        Returns a positive value if risk is rising, negative if falling.
        Trend is measured as: mean(last_half) - mean(first_half) of the rolling window.
        """
        history = list(self._history.get(corridor, []))
        if len(history) < 2:
            return 0.0

        mid = len(history) // 2
        first_half = history[:mid] if mid > 0 else [history[0]]
        last_half = history[mid:] if mid < len(history) else [history[-1]]

        return float(np.mean(last_half)) - float(np.mean(first_half))
