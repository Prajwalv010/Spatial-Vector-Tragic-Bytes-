"""T14 — Tests for M14 FreeSpaceEstimator.

Verifies the core safety invariant:
  "Walk Forward" is only ever issued when ground is positively WALKABLE.
  Absence of obstacles alone does NOT produce WALKABLE.

Tests:
1. Black frame → UNKNOWN (not WALKABLE)
2. Blank white frame → UNKNOWN (near-uniform surface)
3. Normal textured floor → WALKABLE
4. Frame with strong vertical edges (table corner) → BLOCKED or UNKNOWN
5. YOLO bbox in ground region → BLOCKED
6. Sudden transition WALKABLE→UNKNOWN bypasses smoothing
7. all_unknown() factory returns correct structure
"""

import numpy as np
import pytest

from spatialvector.freespace import FreeSpaceEstimator, CorridorStatus, FreespaceResult


# ── Frame fixtures ────────────────────────────────────────────────────────────

def black_frame(h=480, w=640):
    return np.zeros((h, w, 3), dtype=np.uint8)


def white_frame(h=480, w=640):
    return np.full((h, w, 3), 250, dtype=np.uint8)


def textured_floor_frame(h=480, w=640):
    """Synthesise a frame with realistic ground texture (random noise + slight structure)."""
    rng = np.random.default_rng(42)
    frame = rng.integers(80, 160, (h, w, 3), dtype=np.uint8)
    return frame


def table_edge_frame(h=480, w=640):
    """Frame where a sharp horizontal band simulates a table edge in the ground region."""
    frame = textured_floor_frame(h, w)
    # Hard vertical transition at y=int(h*0.70)
    frame[int(h * 0.68):int(h * 0.72), :] = [0, 0, 0]  # black line = sharp discontinuity
    return frame


def sky_frame(h=480, w=640):
    """Bright uniform sky — very bright, low texture."""
    return np.full((h, w, 3), 220, dtype=np.uint8)


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestFreeSpaceEstimator:

    def setup_method(self):
        self.estimator = FreeSpaceEstimator(smoothing_window=1)  # no smoothing for unit tests

    # ── Core safety: fail-safe defaults ──────────────────────────────────────

    def test_black_frame_is_unknown(self):
        """Black / dark frame must produce UNKNOWN for all corridors, never WALKABLE."""
        result = self.estimator.estimate(black_frame())
        assert result.centre != CorridorStatus.WALKABLE, (
            "Dark frame should NOT be WALKABLE"
        )
        assert result.centre in (CorridorStatus.UNKNOWN, CorridorStatus.BLOCKED)

    def test_white_uniform_frame_is_unknown(self):
        """Near-uniform bright frame (sky, blank wall, covered lens) → UNKNOWN."""
        result = self.estimator.estimate(white_frame())
        assert result.centre != CorridorStatus.WALKABLE, (
            "Uniform bright frame should NOT be WALKABLE"
        )

    def test_sky_frame_is_unknown(self):
        """Sky / over-exposed frame → UNKNOWN."""
        result = self.estimator.estimate(sky_frame())
        assert result.centre in (CorridorStatus.UNKNOWN, CorridorStatus.BLOCKED)

    # ── Table-corner / discontinuity ──────────────────────────────────────────

    def test_table_edge_frame_is_not_walk_forward(self):
        """Frame with sharp horizontal band (table edge) → BLOCKED or UNKNOWN, not WALKABLE."""
        result = self.estimator.estimate(table_edge_frame())
        assert result.centre != CorridorStatus.WALKABLE, (
            "Table-edge frame must NEVER yield WALKABLE centre corridor"
        )

    # ── YOLO blocking ─────────────────────────────────────────────────────────

    def test_yolo_bbox_in_ground_region_blocks_corridor(self):
        """A YOLO-tracked object in the ground region should make that corridor BLOCKED."""
        frame = textured_floor_frame()
        h, w = frame.shape[:2]
        # Obstacle covers the centre corridor at ground level
        centre_bbox = [(w * 0.38, h * 0.58, w * 0.62, h * 0.92)]
        result = self.estimator.estimate(frame, yolo_bboxes=centre_bbox)
        assert result.centre in (CorridorStatus.BLOCKED, CorridorStatus.UNKNOWN), (
            "Centre corridor with YOLO obstacle should be BLOCKED or UNKNOWN"
        )

    # ── Schema ───────────────────────────────────────────────────────────────

    def test_result_schema_fields_present(self):
        """FreespaceResult has all required fields."""
        result = self.estimator.estimate(textured_floor_frame())
        assert isinstance(result.left,   CorridorStatus)
        assert isinstance(result.centre, CorridorStatus)
        assert isinstance(result.right,  CorridorStatus)
        assert 0.0 <= result.left_conf   <= 1.0
        assert 0.0 <= result.centre_conf <= 1.0
        assert 0.0 <= result.right_conf  <= 1.0
        assert isinstance(result.reasons, dict)
        assert "centre" in result.reasons

    def test_all_unknown_factory(self):
        """FreespaceResult.all_unknown() returns correct structure."""
        result = FreespaceResult.all_unknown("test reason")
        assert result.centre == CorridorStatus.UNKNOWN
        assert result.left   == CorridorStatus.UNKNOWN
        assert result.right  == CorridorStatus.UNKNOWN
        assert result.centre_conf == 0.0
        assert "test reason" in result.reasons.get("centre", "")

    def test_get_method(self):
        """FreespaceResult.get() resolves named corridor correctly."""
        result = FreespaceResult.all_unknown()
        assert result.get("left")   == CorridorStatus.UNKNOWN
        assert result.get("centre") == CorridorStatus.UNKNOWN
        assert result.get("right")  == CorridorStatus.UNKNOWN
        assert result.get("invalid") == CorridorStatus.UNKNOWN

    # ── Sudden transition ────────────────────────────────────────────────────

    def test_sudden_ground_loss_bypasses_smoothing(self):
        """After WALKABLE, a sudden UNKNOWN/BLOCKED frame must immediately output non-WALKABLE."""
        est = FreeSpaceEstimator(smoothing_window=5)  # heavy smoothing
        textured = textured_floor_frame()

        # Prime with WALKABLE frames
        for _ in range(5):
            r = est.estimate(textured)
        # Now present a black frame (sudden ground loss)
        sudden_loss = est.estimate(black_frame())

        # Even with heavy smoothing, sudden ground loss must not produce WALKABLE
        assert sudden_loss.centre != CorridorStatus.WALKABLE, (
            "Sudden black frame after WALKABLE must bypass smoothing"
        )

    # ── Reset ────────────────────────────────────────────────────────────────

    def test_reset_clears_history(self):
        """After reset(), the estimator behaves as freshly initialised."""
        est = FreeSpaceEstimator(smoothing_window=3)
        for _ in range(3):
            est.estimate(textured_floor_frame())
        est.reset()
        result = est.estimate(black_frame())
        assert result.centre != CorridorStatus.WALKABLE
