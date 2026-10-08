"""SpatialVector-HMI — Adaptive Temporal Smoother.

Provides self-adapting exponential moving average (AdaptiveEMA) with asymmetric
attack/decay and robust median/MAD noise tracking. Prevents telemetry jitter and
false alarms without adding lag to genuine rapid obstacle approach events.
"""

from __future__ import annotations

import collections
import math
from typing import Optional
import numpy as np


class AdaptiveEMA:
    """Exponential moving average with a self-adapting smoothing factor.

    - Learns the recent frame-to-frame noise level (median absolute deviation) of the
      raw signal, analogous to SelfAdaptingLoomingCalibrator.
    - When the raw signal's frame-to-frame jump is within the learned noise band, smooths
      HARD (slow alpha) to suppress detector/torso jitter.
    - When the raw signal moves well outside the learned noise band (a real approach event),
      smooths LIGHT (fast alpha) to avoid lagging behind rapid hazards.
    - Employs asymmetric attack/decay: escalating threats respond with short attack,
      while resolving threats decay more conservatively.
    """

    def __init__(
        self,
        alpha_slow: float = 0.15,
        alpha_fast: float = 0.60,
        window_size: int = 30,
        rising_is_dangerous: bool = True,
        initial_value: Optional[float] = None,
    ):
        """
        Args:
            alpha_slow: conservative EMA factor used when variations are within ambient noise.
            alpha_fast: responsive EMA factor used when variations exceed ambient noise.
            window_size: sample count for rolling median/MAD noise level estimation.
            rising_is_dangerous: True if increasing values denote higher threat (e.g. risk, proximity_scale);
                                 False if decreasing values denote higher threat (e.g. TTC, CPA).
            initial_value: optional seed value.
        """
        self.alpha_slow = float(alpha_slow)
        self.alpha_fast = float(alpha_fast)
        self.window_size = int(window_size)
        self.rising_is_dangerous = bool(rising_is_dangerous)

        self._diff_history: collections.deque[float] = collections.deque(maxlen=self.window_size)
        self._smoothed_value: Optional[float] = float(initial_value) if initial_value is not None else None
        self._last_raw: Optional[float] = float(initial_value) if initial_value is not None else None
        self._noise_std: float = 0.02
        self._current_alpha: float = self.alpha_slow
        self._sample_count: int = 0

    def update(self, raw_value: Optional[float]) -> Optional[float]:
        """Incorporate a new raw observation and return the smoothed estimate.

        Args:
            raw_value: current raw signal measurement (or None).

        Returns:
            Smoothed signal value (or None if uninitialized and raw_value is None).
        """
        if raw_value is None or not math.isfinite(raw_value):
            return self._smoothed_value

        val = float(raw_value)
        self._sample_count += 1

        if self._smoothed_value is None:
            self._smoothed_value = val
            self._last_raw = val
            return self._smoothed_value

        # Track frame-to-frame jump magnitude
        if self._last_raw is not None:
            diff = abs(val - self._last_raw)
            if math.isfinite(diff):
                self._diff_history.append(diff)
        self._last_raw = val

        # Estimate ambient noise floor via median and MAD
        if len(self._diff_history) >= 5:
            arr = np.array(self._diff_history, dtype=np.float64)
            med = float(np.median(arr))
            mad = float(np.median(np.abs(arr - med)))
            self._noise_std = max(0.005, mad * 1.4826)

        noise_band = 2.0 * self._noise_std
        delta = val - self._smoothed_value
        abs_delta = abs(delta)

        # Asymmetric alpha selection
        if abs_delta <= noise_band:
            # Jitter within ambient noise floor: apply heavy smoothing
            alpha = self.alpha_slow
        else:
            # True trajectory deviation: check if threat is escalating
            if self.rising_is_dangerous:
                # Escalating threat: rapid attack; de-escalating threat: slower release
                alpha = self.alpha_fast if delta > 0 else self.alpha_slow
            else:
                # Lowering value is dangerous (e.g. TTC dropping rapidly): rapid attack
                alpha = self.alpha_fast if delta < 0 else self.alpha_slow

        self._current_alpha = alpha
        self._smoothed_value = (1.0 - alpha) * self._smoothed_value + alpha * val
        return float(self._smoothed_value)

    @property
    def current_alpha(self) -> float:
        """Current smoothing factor alpha (between alpha_slow and alpha_fast)."""
        return self._current_alpha

    @property
    def noise_std(self) -> float:
        """Current estimated noise standard deviation (MAD-derived)."""
        return self._noise_std

    @property
    def value(self) -> Optional[float]:
        """Most recent smoothed value."""
        return self._smoothed_value

    def reset(self, initial_value: Optional[float] = None):
        """Reset internal history and seed value."""
        self._diff_history.clear()
        self._smoothed_value = float(initial_value) if initial_value is not None else None
        self._last_raw = float(initial_value) if initial_value is not None else None
        self._current_alpha = self.alpha_slow
        self._sample_count = 0
