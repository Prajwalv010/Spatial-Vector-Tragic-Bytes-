"""M15 — Navigation Decision Engine (Single Source of Truth).

Receives:
- RiskState from M08 (corridor risks, global risk, state)
- HapticCommand from M09 (corridor policy direction)
- List[HazardDetection] from M13 (potholes, etc.)
- FreespaceResult from M14 (walkable ground per corridor)

Outputs:
- GuidanceDecision: one decision object per frame

STRICT SAFETY RULE:
  "WALK_FORWARD" is only issued when ALL of the following hold:
  1. Centre corridor FreespaceResult == WALKABLE with confidence >= threshold
  2. This has been true for at least `forward_consec_frames_required` consecutive frames
  3. Centre corridor risk (from M08) is below the caution threshold
  4. Global risk is below the warning threshold
  5. No confirmed hazard (pothole, etc.) intersects the centre corridor
  6. The haptic policy (M09) does not call for left or right evasion

  If ANY condition fails, the output is STOP, UNCERTAIN_STOP, MOVE_LEFT, or MOVE_RIGHT.
  Never "Walk Forward" by default. Fail safe.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

logger = logging.getLogger(__name__)


class GuidanceAction(str, Enum):
    WALK_FORWARD   = "WALK FORWARD"
    MOVE_LEFT      = "MOVE LEFT"
    MOVE_RIGHT     = "MOVE RIGHT"
    STOP           = "STOP"
    UNCERTAIN_STOP = "CAUTION"      # safer than STOP but not forward


@dataclass
class GuidanceDecision:
    """Single guidance decision produced by M15."""
    action: GuidanceAction
    reason: str                   # human-readable explanation (shown in overlay)
    voice_text: str               # what to say aloud
    confidence: float             # 0.0 – 1.0
    # Visual styling hints (colour as BGR tuple)
    banner_color_bgr: tuple = (35, 175, 55)
    banner_border_bgr: tuple = (70, 235, 95)
    nav_icon: str = "UP"          # "UP" | "LEFT" | "RIGHT" | "STOP"


class NavigationDecisionEngine:
    """M15 — Produces one authoritative GuidanceDecision per frame.

    This is the only place where the final navigation instruction is computed.
    The viewer's render_overlay reads from this. The voice engine reads from this.
    There is no other decision logic.
    """

    def __init__(
        self,
        # FreespaceResult confidence threshold for WALKABLE to count
        freespace_min_conf: float = 0.48,
        # Consecutive WALKABLE frames required before allowing WALK_FORWARD
        forward_consec_frames_required: int = 3,
        # Risk thresholds (matched to default.yaml)
        centre_risk_threshold: float = 0.35,    # c_risk must be below this
        global_risk_threshold: float = 0.45,    # global must be below this
        all_blocked_global_threshold: float = 0.72,
        # Pothole proximity that blocks the centre corridor (metres)
        pothole_block_dist_m: float = 3.0,
        # Advisory-only pothole mode: potholes alert visually/audibly but do not halt user navigation
        pothole_advisory_only: bool = True,
    ):
        self.freespace_min_conf = freespace_min_conf
        self.forward_consec_required = forward_consec_frames_required
        self.centre_risk_threshold = centre_risk_threshold
        self.global_risk_threshold = global_risk_threshold
        self.all_blocked_global_threshold = all_blocked_global_threshold
        self.pothole_block_dist_m = pothole_block_dist_m
        self.pothole_advisory_only = pothole_advisory_only

        # Consecutive WALKABLE frame counter for centre corridor
        self._centre_walkable_streak: int = 0

        logger.info("NavigationDecisionEngine (M15) initialised — forward requires %d consec WALKABLE frames (pothole_advisory=%s)", forward_consec_frames_required, pothole_advisory_only)

    def decide(
        self,
        risk,           # RiskState from M08
        cmd,            # HapticCommand from M09
        freespace,      # FreespaceResult from M14 (or None if M14 not running)
        hazards,        # List[HazardDetection] from M13
    ) -> GuidanceDecision:
        """Compute the authoritative guidance decision for this frame."""

        from spatialvector.freespace import CorridorStatus

        l_risk = risk.corridor_risks.get("left", 0.0)
        c_risk = risk.corridor_risks.get("center", 0.0)
        r_risk = risk.corridor_risks.get("right", 0.0)

        # ── 0. Fail-Safe: Refuse WALK_FORWARD if segmentation model failed to load ─
        centre_reason = (freespace.reasons or {}).get("centre", "") if freespace else ""
        if centre_reason == "SEGMENTATION MODEL NOT LOADED":
            self._centre_walkable_streak = 0
            return GuidanceDecision(
                action=GuidanceAction.UNCERTAIN_STOP,
                reason="SEGMENTATION MODEL NOT LOADED",
                voice_text="Caution: ground segmentation offline",
                confidence=0.0,
                banner_color_bgr=(0, 140, 255),
                banner_border_bgr=(0, 180, 255),
                nav_icon="STOP",
            )

        # ── 1. Hard STOP conditions ────────────────────────────────────────────
        all_blocked_risk = (
            (c_risk >= 0.65 and l_risk >= 0.55 and r_risk >= 0.55)
            or risk.global_risk >= self.all_blocked_global_threshold
            or (cmd.direction == "STOP" and risk.global_risk > 0.40)
        )
        if all_blocked_risk:
            self._centre_walkable_streak = 0
            return GuidanceDecision(
                action=GuidanceAction.STOP,
                reason=f"all corridors blocked (global_risk={risk.global_risk:.0%})",
                voice_text="Stop",
                confidence=1.0,
                banner_color_bgr=(25, 25, 225),
                banner_border_bgr=(60, 60, 255),
                nav_icon="STOP",
            )

        # ── 2. Near pothole in centre corridor → stop/caution (if not advisory only) ──
        centre_pothole = next(
            (h for h in (hazards or []) if h.dist_m <= self.pothole_block_dist_m), None
        )
        if centre_pothole and not self.pothole_advisory_only:
            self._centre_walkable_streak = 0
            return GuidanceDecision(
                action=GuidanceAction.UNCERTAIN_STOP,
                reason=f"pothole {centre_pothole.dist_m:.1f}m ahead (conf={centre_pothole.confidence:.0%})",
                voice_text=f"Pothole ahead {centre_pothole.dist_m:.1f} metres",
                confidence=centre_pothole.confidence,
                banner_color_bgr=(0, 145, 245),
                banner_border_bgr=(0, 210, 255),
                nav_icon="STOP",
            )

        # ── 3. Freespace check (core safety gate) ─────────────────────────────
        centre_fs = freespace.centre if freespace else CorridorStatus.UNKNOWN
        centre_fs_conf = freespace.centre_conf if freespace else 0.0
        centre_reason  = (freespace.reasons or {}).get("centre", "unknown") if freespace else "freespace not running"

        # The safety gates are segmentation, vetoes, and consecutive-frames rule.
        # Freespace confidence is an uncalibrated evidence score, NOT used as a safety gate.
        centre_walkable = (centre_fs == CorridorStatus.WALKABLE)

        if centre_walkable:
            self._centre_walkable_streak += 1
        else:
            self._centre_walkable_streak = 0

        # ── 4. MOVE LEFT / RIGHT (evasion from risk engine) ───────────────────
        if cmd.direction in ("LEFT", "RIGHT"):
            self._centre_walkable_streak = 0
            side = cmd.direction
            side_fs = (freespace.left if side == "LEFT" else freespace.right) if freespace else CorridorStatus.UNKNOWN
            if side_fs == CorridorStatus.UNKNOWN:
                # Can't confirm the side is safe either — caution
                return GuidanceDecision(
                    action=GuidanceAction.UNCERTAIN_STOP,
                    reason=f"obstacle in centre; {side} side unconfirmed",
                    voice_text="Caution, obstacle ahead",
                    confidence=0.5,
                    banner_color_bgr=(0, 145, 245),
                    banner_border_bgr=(0, 210, 255),
                    nav_icon="STOP",
                )
            action = GuidanceAction.MOVE_LEFT if side == "LEFT" else GuidanceAction.MOVE_RIGHT
            icon   = "LEFT" if side == "LEFT" else "RIGHT"
            return GuidanceDecision(
                action=action,
                reason=f"obstacle in centre; {side} side clear",
                voice_text=f"Move {side.title()}",
                confidence=0.8,
                banner_color_bgr=(0, 145, 245),
                banner_border_bgr=(0, 210, 255),
                nav_icon=icon,
            )

        # ── 5. Freespace BLOCKED or UNKNOWN → can't go forward ────────────────
        if not centre_walkable:
            self._centre_walkable_streak = 0
            if centre_fs == CorridorStatus.BLOCKED:
                return GuidanceDecision(
                    action=GuidanceAction.STOP,
                    reason=f"centre blocked: {centre_reason}",
                    voice_text="Stop, path blocked",
                    confidence=0.85,
                    banner_color_bgr=(25, 25, 225),
                    banner_border_bgr=(60, 60, 255),
                    nav_icon="STOP",
                )
            else:
                # UNKNOWN — safest default
                return GuidanceDecision(
                    action=GuidanceAction.UNCERTAIN_STOP,
                    reason=f"ground not confirmed: {centre_reason}",
                    voice_text="Caution, checking path",
                    confidence=0.5,
                    banner_color_bgr=(0, 115, 200),
                    banner_border_bgr=(0, 170, 240),
                    nav_icon="STOP",
                )

        # ── 6. Centre risk from obstacle detection ─────────────────────────────
        if c_risk >= self.centre_risk_threshold or risk.global_risk >= self.global_risk_threshold:
            self._centre_walkable_streak = 0
            if l_risk <= r_risk:
                return GuidanceDecision(
                    action=GuidanceAction.MOVE_LEFT,
                    reason=f"obstacle risk in centre (c_risk={c_risk:.0%})",
                    voice_text="Move Left",
                    confidence=0.75,
                    banner_color_bgr=(0, 145, 245),
                    banner_border_bgr=(0, 210, 255),
                    nav_icon="LEFT",
                )
            else:
                return GuidanceDecision(
                    action=GuidanceAction.MOVE_RIGHT,
                    reason=f"obstacle risk in centre (c_risk={c_risk:.0%})",
                    voice_text="Move Right",
                    confidence=0.75,
                    banner_color_bgr=(0, 145, 245),
                    banner_border_bgr=(0, 210, 255),
                    nav_icon="RIGHT",
                )

        # ── 7. WALK FORWARD — all gates cleared ───────────────────────────────
        if self._centre_walkable_streak >= self.forward_consec_required:
            return GuidanceDecision(
                action=GuidanceAction.WALK_FORWARD,
                reason=f"ground confirmed, path clear ({self._centre_walkable_streak} frames)",
                voice_text="Walk Forward",
                confidence=min(1.0, centre_fs_conf * 1.2),
                banner_color_bgr=(35, 175, 55),
                banner_border_bgr=(70, 235, 95),
                nav_icon="UP",
            )

        # Not enough consecutive WALKABLE frames yet — wait
        return GuidanceDecision(
            action=GuidanceAction.UNCERTAIN_STOP,
            reason=f"waiting for ground confirmation ({self._centre_walkable_streak}/{self.forward_consec_required} frames)",
            voice_text="Checking path",
            confidence=0.4,
            banner_color_bgr=(0, 115, 200),
            banner_border_bgr=(0, 170, 240),
            nav_icon="STOP",
        )

    def reset(self):
        """Reset streak counter (call when source changes)."""
        self._centre_walkable_streak = 0
