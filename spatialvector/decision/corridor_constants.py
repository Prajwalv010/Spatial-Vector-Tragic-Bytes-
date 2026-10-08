"""SpatialVector-HMI — Canonical Corridor Bearing Constants.

Single source of truth for corridor angular boundaries shared across M07 (Prediction)
and M08 (Risk Engine). Eliminates silent dead-zone discrepancies where M07 and M08
disagree on what constitutes the center walking path.
"""

from __future__ import annotations

import math

# Corridor bearing boundary (radians):
# The field of view is split into three sectors:
#   Left:   bearing < -CORRIDOR_CENTER_BEARING_RAD
#   Center: -CORRIDOR_CENTER_BEARING_RAD <= bearing <= CORRIDOR_CENTER_BEARING_RAD
#   Right:  bearing > CORRIDOR_CENTER_BEARING_RAD
# pi/6 (~30 degrees, 0.5236 rad) splits a typical ~90 degree horizontal FOV into thirds.
CORRIDOR_CENTER_BEARING_RAD: float = math.pi / 6.0
