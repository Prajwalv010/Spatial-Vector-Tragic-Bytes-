"""T15 — Tests for M15 NavigationDecisionEngine.

Verifies the core safety invariant:
  "Walk Forward" is only ever issued when the centre corridor is positively
  WALKABLE for >= N consecutive frames, AND risk is low, AND no hazard blocks it.
  All other states produce STOP, CAUTION, or a steering command.

Tests:
1. UNKNOWN freespace → never WALK_FORWARD
2. BLOCKED freespace → STOP
3. WALKABLE but insufficient consecutive frames → UNCERTAIN_STOP (not WALK_FORWARD)
4. WALKABLE for N consecutive frames, low risk → WALK_FORWARD
5. WALKABLE but obstacle risk high → move command, not WALK_FORWARD
6. WALKABLE but pothole close → UNCERTAIN_STOP
7. All corridors blocked → STOP
8. HapticCommand LEFT with UNKNOWN side → UNCERTAIN_STOP, not MOVE_LEFT
9. Schema: GuidanceDecision has action, reason, voice_text, confidence
10. action enum values are the human-readable strings used by overlay
"""

import pytest
from dataclasses import dataclass
from typing import Dict, List, Optional

from spatialvector.freespace import CorridorStatus, FreespaceResult
from spatialvector.guidance import GuidanceDecision, GuidanceAction, NavigationDecisionEngine
from spatialvector.hazards import HazardDetection
import time


# ── Minimal stubs ─────────────────────────────────────────────────────────────

@dataclass
class FakeRisk:
    corridor_risks: Dict[str, float]
    global_risk: float
    confidence: float = 0.95


@dataclass
class FakeCmd:
    direction: str


def make_risk(l=0.0, c=0.0, r=0.0, g=0.0):
    return FakeRisk({"left": l, "center": c, "right": r}, g)


def make_cmd(direction="NONE"):
    return FakeCmd(direction)


def make_freespace(l=CorridorStatus.WALKABLE, c=CorridorStatus.WALKABLE, r=CorridorStatus.WALKABLE,
                   lc=0.8, cc=0.8, rc=0.8):
    return FreespaceResult(l, c, r, lc, cc, rc,
                           {"left": "ok", "centre": "ok", "right": "ok"})


def make_unknown_fs():
    return FreespaceResult.all_unknown("not confirmed")


def make_pothole(dist_m=1.5, conf=0.75):
    return HazardDetection(
        hazard_class="pothole",
        bbox_xyxy=(100.0, 300.0, 200.0, 400.0),
        confidence=conf,
        dist_m=dist_m,
        dist_provenance="ground_plane_geometry",
        source="heuristic",
        frame_id=0,
        timestamp=time.monotonic(),
    )


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestNavigationDecisionEngine:

    def setup_method(self):
        self.engine = NavigationDecisionEngine(forward_consec_frames_required=3)

    def _run_n_frames(self, n, risk=None, cmd=None, freespace=None, hazards=None):
        risk = risk or make_risk()
        cmd  = cmd  or make_cmd("NONE")
        freespace = freespace or make_freespace()
        hazards   = hazards or []
        last = None
        for _ in range(n):
            last = self.engine.decide(risk, cmd, freespace, hazards)
        return last

    # ── Core safety invariant ─────────────────────────────────────────────────

    def test_unknown_freespace_never_walk_forward(self):
        """UNKNOWN centre freespace must never produce WALK_FORWARD."""
        decision = self._run_n_frames(10, freespace=make_unknown_fs())
        assert decision.action != GuidanceAction.WALK_FORWARD, (
            "UNKNOWN freespace must never result in WALK_FORWARD"
        )

    def test_blocked_freespace_produces_stop(self):
        """BLOCKED centre freespace must produce STOP."""
        blocked_fs = make_freespace(c=CorridorStatus.BLOCKED, cc=0.9)
        decision = self._run_n_frames(5, freespace=blocked_fs)
        assert decision.action == GuidanceAction.STOP, (
            "BLOCKED centre corridor must produce STOP"
        )

    def test_insufficient_walkable_streak_is_not_walk_forward(self):
        """Less than N consecutive WALKABLE frames must not produce WALK_FORWARD."""
        self.engine.reset()
        decision = self._run_n_frames(2, freespace=make_freespace())  # 2 < 3 required
        assert decision.action != GuidanceAction.WALK_FORWARD, (
            "< N consecutive WALKABLE frames must not produce WALK_FORWARD"
        )

    def test_n_consecutive_walkable_produces_walk_forward(self):
        """Exactly N consecutive WALKABLE frames with low risk → WALK_FORWARD."""
        self.engine.reset()
        decision = self._run_n_frames(3, freespace=make_freespace())
        assert decision.action == GuidanceAction.WALK_FORWARD, (
            "After exactly N WALKABLE frames, should get WALK_FORWARD"
        )

    def test_walkable_streak_resets_on_blocked_frame(self):
        """After a BLOCKED frame, streak resets and WALK_FORWARD requires N new frames."""
        self.engine.reset()
        self._run_n_frames(3, freespace=make_freespace())  # reach WALK_FORWARD
        # Inject one BLOCKED frame
        self.engine.decide(make_risk(), make_cmd(), make_freespace(c=CorridorStatus.BLOCKED, cc=0.9), [])
        # Now only 1 subsequent WALKABLE frame
        decision = self.engine.decide(make_risk(), make_cmd(), make_freespace(), [])
        assert decision.action != GuidanceAction.WALK_FORWARD, (
            "Streak must reset after BLOCKED frame"
        )

    # ── Hazard blocking ───────────────────────────────────────────────────────

    def test_pothole_close_blocks_walk_forward(self):
        """A near pothole (< pothole_block_dist_m) must not allow WALK_FORWARD when blocking is enabled."""
        engine = NavigationDecisionEngine(forward_consec_frames_required=3, pothole_advisory_only=False)
        # Prime streak to WALKABLE state
        for _ in range(3):
            d = engine.decide(make_risk(), make_cmd(), make_freespace(), [make_pothole(dist_m=1.0)])
        assert d.action != GuidanceAction.WALK_FORWARD, (
            "Nearby pothole must prevent WALK_FORWARD"
        )

    def test_far_pothole_does_not_block_walk_forward(self):
        """A distant pothole (> pothole_block_dist_m) should not block WALK_FORWARD."""
        self.engine.reset()
        for _ in range(3):
            d = self.engine.decide(make_risk(), make_cmd(), make_freespace(), [make_pothole(dist_m=5.0)])
        assert d.action == GuidanceAction.WALK_FORWARD, (
            "Distant pothole should not block WALK_FORWARD"
        )

    # ── Risk engine blocking ───────────────────────────────────────────────────

    def test_high_centre_risk_blocks_walk_forward(self):
        """High centre corridor risk should produce a steering command, not WALK_FORWARD."""
        self.engine.reset()
        high_risk = make_risk(c=0.80, g=0.50)
        for _ in range(5):
            d = self.engine.decide(high_risk, make_cmd(), make_freespace(), [])
        assert d.action != GuidanceAction.WALK_FORWARD

    def test_all_blocked_risk_produces_stop(self):
        """All corridors at high risk → STOP."""
        self.engine.reset()
        all_risk = make_risk(l=0.80, c=0.80, r=0.80, g=0.90)
        d = self.engine.decide(all_risk, make_cmd(), make_freespace(), [])
        assert d.action == GuidanceAction.STOP

    # ── Haptic evasion ────────────────────────────────────────────────────────

    def test_haptic_left_with_unknown_side_is_caution(self):
        """HapticCommand LEFT with UNKNOWN left corridor → UNCERTAIN_STOP, not MOVE_LEFT."""
        self.engine.reset()
        mixed_fs = make_freespace(l=CorridorStatus.UNKNOWN, lc=0.0, c=CorridorStatus.WALKABLE, cc=0.8)
        d = self.engine.decide(make_risk(), make_cmd("LEFT"), mixed_fs, [])
        assert d.action != GuidanceAction.MOVE_LEFT, (
            "Cannot move LEFT if left corridor is UNKNOWN"
        )
        assert d.action == GuidanceAction.UNCERTAIN_STOP

    def test_haptic_left_with_walkable_side_produces_move_left(self):
        """HapticCommand LEFT with WALKABLE left corridor → MOVE_LEFT."""
        self.engine.reset()
        d = self.engine.decide(make_risk(), make_cmd("LEFT"), make_freespace(), [])
        assert d.action == GuidanceAction.MOVE_LEFT

    # ── Schema ───────────────────────────────────────────────────────────────

    def test_guidance_decision_schema(self):
        """GuidanceDecision has all required fields."""
        d = self.engine.decide(make_risk(), make_cmd(), make_freespace(), [])
        assert isinstance(d, GuidanceDecision)
        assert isinstance(d.action, GuidanceAction)
        assert isinstance(d.reason, str) and len(d.reason) > 0
        assert isinstance(d.voice_text, str) and len(d.voice_text) > 0
        assert 0.0 <= d.confidence <= 1.0
        assert isinstance(d.banner_color_bgr, tuple) and len(d.banner_color_bgr) == 3
        assert d.nav_icon in ("UP", "LEFT", "RIGHT", "STOP")

    def test_guidance_action_values_are_human_readable(self):
        """GuidanceAction enum values are the strings shown in the UI."""
        assert GuidanceAction.WALK_FORWARD.value == "WALK FORWARD"
        assert GuidanceAction.MOVE_LEFT.value == "MOVE LEFT"
        assert GuidanceAction.MOVE_RIGHT.value == "MOVE RIGHT"
        assert GuidanceAction.STOP.value == "STOP"
        assert GuidanceAction.UNCERTAIN_STOP.value == "CAUTION"

    # ── Reset ─────────────────────────────────────────────────────────────────

    def test_reset_clears_streak(self):
        """reset() clears the consecutive WALKABLE streak."""
        self._run_n_frames(3, freespace=make_freespace())
        self.engine.reset()
        d = self.engine.decide(make_risk(), make_cmd(), make_freespace(), [])
        assert d.action != GuidanceAction.WALK_FORWARD
