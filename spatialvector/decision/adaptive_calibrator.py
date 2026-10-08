"""SpatialVector-HMI — Online Self-Adapting Looming Calibrator

Learns the per-session ambient noise floor of bounding box jitter, walking sway,
and camera FOV dynamics. Dynamically separates true obstacle approach (looming)
from chest-mount gait fluctuations and detector boundary noise.
"""

from __future__ import annotations

import collections
import math
from typing import Dict, List, Tuple
import numpy as np


class SelfAdaptingLoomingCalibrator:
    """Online adaptive estimator for scale-change and looming detection.

    Maintains a rolling history of expansion rates across background tracks,
    computing the session's noise floor (mean and variance). Adapts the detection
    threshold so the system remains sensitive to approaching obstacles while
    filtering out wearer body motion and detector jitter.
    """

    def __init__(
        self,
        window_size: int = 80,
        min_threshold: float = 0.05,
        max_threshold: float = 0.25,
        k_sigma: float = 2.0,
    ):
        self.window_size = window_size
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.k_sigma = k_sigma

        self._history = collections.deque(maxlen=window_size)
        self._mean_floor: float = 0.0
        self._std_floor: float = 0.02
        self._adaptive_threshold: float = min_threshold
        self._total_samples: int = 0

    def update(self, rates: List[float]):
        """Update rolling noise floor statistics with observed expansion rates."""
        for r in rates:
            if math.isfinite(r) and abs(r) < 3.0:
                self._history.append(float(r))
                self._total_samples += 1

        if len(self._history) >= 10:
            arr = np.array(self._history, dtype=np.float64)
            # Use median and MAD (Median Absolute Deviation) for outlier-robust noise floor
            med = float(np.median(arr))
            mad = float(np.median(np.abs(arr - med)))
            std_est = max(0.015, mad * 1.4826)

            self._mean_floor = med
            self._std_floor = std_est

            # Dynamic threshold: median noise floor + k * std
            thresh = med + self.k_sigma * std_est
            self._adaptive_threshold = float(np.clip(thresh, self.min_threshold, self.max_threshold))

    def evaluate_expansion(self, raw_rate: float) -> Tuple[float, bool]:
        """Evaluate raw expansion rate against the adaptive threshold.

        Returns:
            (effective_rate, is_approaching):
                effective_rate: sanitized positive expansion rate for TTC calculation.
                is_approaching: True if rate exceeds the adaptive noise threshold.
        """
        if not math.isfinite(raw_rate) or raw_rate <= 0.0:
            return 0.0, False

        is_approaching = raw_rate >= self._adaptive_threshold
        effective_rate = max(0.0, raw_rate - max(0.0, self._mean_floor)) if is_approaching else 0.0
        return float(effective_rate), bool(is_approaching)

    @property
    def current_threshold(self) -> float:
        return self._adaptive_threshold

    @property
    def noise_floor(self) -> float:
        return self._mean_floor

    @property
    def noise_std(self) -> float:
        return self._std_floor

    def get_status(self) -> Dict[str, float]:
        return {
            "noise_mean": float(self._mean_floor),
            "noise_std": float(self._std_floor),
            "threshold": float(self._adaptive_threshold),
            "samples": int(self._total_samples),
        }
