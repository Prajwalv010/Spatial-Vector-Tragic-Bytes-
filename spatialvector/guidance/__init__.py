"""spatialvector.guidance — Single-Source-of-Truth Navigation Decision (M15).

Produces ONE GuidanceDecision per frame. The viewer, voice engine, haptic
output, and telemetry all read from this single object. There is NO duplicated
decision logic anywhere else.

Core safety rule: "Walk Forward" requires the centre corridor to be positively
WALKABLE for several consecutive frames AND no tracked obstacle or confirmed
hazard intersects it. If the centre is UNKNOWN or BLOCKED, the output is
always a stop or caution variant — never forward.
"""
from .decision import GuidanceDecision, GuidanceAction, NavigationDecisionEngine

__all__ = ["GuidanceDecision", "GuidanceAction", "NavigationDecisionEngine"]
