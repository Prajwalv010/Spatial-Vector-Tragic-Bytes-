"""spatialvector.hazards — Ground Hazard Detection (M13).

Detects ground-level hazards such as potholes, manholes and surface damage.
Outputs structured HazardDetection objects consumed by the guidance layer.
Never reads from or writes to the risk engine directly; it provides data forward only.
"""
from .ground_hazard import GroundHazardDetector, HazardDetection

__all__ = ["GroundHazardDetector", "HazardDetection"]
