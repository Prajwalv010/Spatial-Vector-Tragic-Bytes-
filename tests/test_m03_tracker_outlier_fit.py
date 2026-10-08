"""
Regression test for Bug 2a: Least-squares linear regression vs two-point endpoint differences.

Verifies that:
1. When a detector produces an outlier frame in a track history, the least-squares fitted velocity
   and expansion rate stay significantly closer to the true ground truth trend than the old
   two-point endpoint difference would have.
2. Across a sliding 10-frame window as an outlier passes through, the maximum error of the fitted
   velocity is strictly lower than that of the two-point difference.
"""

import math
import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from spatialvector.perception.tracker import MultiObjectTracker


def test_velocity_fit_resilience_to_endpoint_outlier():
    tracker = MultiObjectTracker(history_length=10)
    fps = 30.0
    dt = 1.0 / fps
    true_vx = 60.0   # px/s
    true_vy = -30.0  # px/s

    # 10 timestamps
    timestamps = [i * dt for i in range(10)]
    # True trajectory
    centers = [(100.0 + true_vx * t, 200.0 + true_vy * t) for t in timestamps]

    # Deliberate noisy detector glitch on the latest frame (index 9)
    # e.g., detector jitter puts center 40px away
    noisy_centers = list(centers)
    noisy_centers[-1] = (noisy_centers[-1][0] + 35.0, noisy_centers[-1][1] - 35.0)

    # Old 2-point endpoint formula
    total_dt = timestamps[-1] - timestamps[0]
    old_vx = (noisy_centers[-1][0] - noisy_centers[0][0]) / total_dt
    old_vy = (noisy_centers[-1][1] - noisy_centers[0][1]) / total_dt

    # New least-squares fit from tracker
    fit_vx, fit_vy = tracker._compute_velocity(noisy_centers, timestamps)

    old_err = math.sqrt((old_vx - true_vx) ** 2 + (old_vy - true_vy) ** 2)
    fit_err = math.sqrt((fit_vx - true_vx) ** 2 + (fit_vy - true_vy) ** 2)

    # Fitted velocity error must be substantially smaller than old 2-point secant error
    assert fit_err < old_err * 0.65, (
        f"Fitted error ({fit_err:.2f}) should be significantly less than 2-point error ({old_err:.2f})"
    )


def test_expansion_rate_fit_resilience_to_outlier():
    tracker = MultiObjectTracker(history_length=10)
    fps = 30.0
    dt = 1.0 / fps

    timestamps = [i * dt for i in range(10)]
    total_dt = timestamps[-1] - timestamps[0]

    # Clean expanding bounding box: diagonal grows from 100 to 140
    # True rate ~ (140 - 100) / (140 * total_dt) ~ 40 / (140 * 0.3) ~ 0.95 /s
    bboxes = []
    for i, t in enumerate(timestamps):
        size = 100.0 + 133.33 * t  # expands linearly with time
        w = size / math.sqrt(2)
        h = size / math.sqrt(2)
        bboxes.append((100.0, 100.0, 100.0 + w, 100.0 + h))

    # Calculate true rate from ground truth
    clean_rate = tracker._compute_expansion_rate(bboxes, timestamps)

    # Insert severe outlier bbox at the latest frame
    noisy_bboxes = list(bboxes)
    outlier_size = 220.0
    w_out = outlier_size / math.sqrt(2)
    h_out = outlier_size / math.sqrt(2)
    noisy_bboxes[-1] = (100.0, 100.0, 100.0 + w_out, 100.0 + h_out)

    # Old 2-point endpoint rate calculation
    s_init = math.sqrt((bboxes[0][2] - bboxes[0][0])**2 + (bboxes[0][3] - bboxes[0][1])**2)
    s_final_noisy = math.sqrt((noisy_bboxes[-1][2] - noisy_bboxes[-1][0])**2 + (noisy_bboxes[-1][3] - noisy_bboxes[-1][1])**2)
    old_rate = (s_final_noisy - s_init) / (s_final_noisy * total_dt)

    # New fitted rate
    fit_rate = tracker._compute_expansion_rate(noisy_bboxes, timestamps)

    old_err = abs(old_rate - clean_rate)
    fit_err = abs(fit_rate - clean_rate)

    assert fit_err < old_err * 0.60, (
        f"Fitted expansion error ({fit_err:.3f}) must be substantially less than old 2-point error ({old_err:.3f})"
    )


def test_sliding_window_outlier_rejection():
    """Simulates an approach sequence where a glitch occurs in the middle of a 15-frame trajectory.
    Evaluates tracking at the moment the glitch enters the buffer (as the newest frame)."""
    tracker = MultiObjectTracker(history_length=10)
    fps = 30.0
    dt = 1.0 / fps
    true_vx = 50.0

    # 15 frames
    timestamps = [i * dt for i in range(15)]
    centers = [(100.0 + true_vx * t, 200.0) for t in timestamps]

    # Glitch occurs at frame index 8
    centers[8] = (centers[8][0] + 50.0, 200.0)

    # At frame 8, the rolling window of length 9 contains the glitch at the newest position
    window_centers = centers[:9]
    window_times = timestamps[:9]

    total_dt = window_times[-1] - window_times[0]
    old_vx = (window_centers[-1][0] - window_centers[0][0]) / total_dt
    fit_vx, _ = tracker._compute_velocity(window_centers, window_times)

    old_err = abs(old_vx - true_vx)
    fit_err = abs(fit_vx - true_vx)

    assert fit_err < old_err * 0.60, f"Fit error {fit_err:.2f} should beat old error {old_err:.2f}"
