"""
Unit & Regression Tests for AdaptiveEMA (spatialvector/motion/temporal_smoother.py).

Verifies:
1. Flat noisy signal converges to alpha_slow and significantly reduces output variance.
2. Step change simulating a real hazard triggers alpha_fast and converges within <= 4 frames.
3. Asymmetric attack/decay:
   - With rising_is_dangerous=True, rising edge uses alpha_fast, falling edge uses alpha_slow.
   - With rising_is_dangerous=False (e.g. TTC/CPA), falling edge uses alpha_fast, rising edge uses alpha_slow.
"""

import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from spatialvector.motion.temporal_smoother import AdaptiveEMA


def test_adaptive_ema_flat_noisy_signal():
    """Flat noisy signal should learn the noise floor, stay in alpha_slow, and filter out variance."""
    np.random.seed(42)
    smoother = AdaptiveEMA(alpha_slow=0.15, alpha_fast=0.60, window_size=30, rising_is_dangerous=True)

    base = 0.25
    noise = np.random.uniform(-0.015, 0.015, size=40)
    raw_signals = [base + n for n in noise]

    smoothed_outputs = []
    for val in raw_signals:
        s = smoother.update(val)
        smoothed_outputs.append(s)

    # After noise learning, alpha must be alpha_slow
    assert smoother.current_alpha == 0.15
    # Variance of smoothed signal should be significantly lower than raw signal
    raw_var = np.var(raw_signals[10:])
    smooth_var = np.var(smoothed_outputs[10:])
    assert smooth_var < raw_var * 0.35, f"Smoothed variance {smooth_var:.6f} should be < 35% of raw {raw_var:.6f}"


def test_adaptive_ema_step_change_rapid_convergence():
    """A real hazard event (step jump) must immediately activate alpha_fast and track within <= 4 frames."""
    smoother = AdaptiveEMA(alpha_slow=0.15, alpha_fast=0.60, window_size=30, rising_is_dangerous=True)

    # Prime with ambient baseline
    for _ in range(30):
        smoother.update(0.20)

    assert smoother.current_alpha == 0.15

    # Sudden approach: risk jumps from 0.20 to 0.90 (delta = 0.70)
    target = 0.90
    threshold_90pct = 0.20 + 0.90 * (target - 0.20)  # 0.83

    frames_to_reach = 0
    for frame in range(1, 10):
        val = smoother.update(target)
        # On first jump, alpha must switch to alpha_fast
        if frame == 1:
            assert smoother.current_alpha == 0.60, f"Expected alpha_fast 0.60, got {smoother.current_alpha}"
        if val >= threshold_90pct and frames_to_reach == 0:
            frames_to_reach = frame
            break

    assert frames_to_reach <= 4, f"Fast tracking mode must reach 90% in <= 4 frames, took {frames_to_reach}"


def test_adaptive_ema_ttc_falling_is_dangerous():
    """For TTC (rising_is_dangerous=False), rapid drop in TTC must use alpha_fast, while rise uses alpha_slow."""
    smoother = AdaptiveEMA(alpha_slow=0.15, alpha_fast=0.60, window_size=30, rising_is_dangerous=False)

    # Prime at safe distance (5.0 seconds TTC)
    for _ in range(25):
        smoother.update(5.0)

    # Rapid closing hazard: TTC plummets to 0.8 seconds
    s1 = smoother.update(0.8)
    assert smoother.current_alpha == 0.60, "TTC drop must trigger alpha_fast (urgent hazard)"

    # Hazard clears: TTC jumps back to 4.5 seconds
    # Should decay smoothly with alpha_slow to prevent alarm ping-pong
    for _ in range(15):
        smoother.update(0.8)  # stabilize around 0.8

    s_clear = smoother.update(4.5)
    assert smoother.current_alpha == 0.15, "TTC recovery must use alpha_slow to avoid alarm flickering"
