"""spatialvector.freespace — Walkable Ground Estimation (M14).

Estimates whether the ground ahead is walkable, blocked, or unknown,
per corridor (left, centre, right).

Core safety rule: UNKNOWN is the default. The system must positively confirm
WALKABLE before allowing "Walk Forward". Absence of obstacles is NOT evidence
of walkability. Fail safe, never fail open.
"""
from .corridor_estimator import FreeSpaceEstimator, CorridorStatus, FreespaceResult

__all__ = ["FreeSpaceEstimator", "CorridorStatus", "FreespaceResult"]
