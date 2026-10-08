"""Tests for Module M12 — Logger, Replay & Evaluation Harness.

Tests:
  T32: Determinism test — Replay recorded session 10 times, assert identical output sequence.
  T33: Corruption resilience — Corrupt record raises clear, informative error.
  T34: A/B evaluation — Compare baseline vs modified risk-engine weights on same recorded session.
"""

import json
from pathlib import Path
import sys
import tempfile
import time

import pytest

# Add repository root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from spatialvector.decision.risk_engine import RiskEngine
from spatialvector.decision.schemas import Prediction
from spatialvector.hmi.logger import SessionLogger
from spatialvector.hmi.replay import SessionReplayer, compare_replays, parse_session_file


def _record_sample_session(session_dir: Path, session_id: str = "test_run", num_frames: int = 30) -> Path:
    """Helper to record a synthetic session using SessionLogger."""
    logger = SessionLogger(session_id=session_id, base_dir=session_dir, config_snapshot={"mode": "test"})

    for i in range(num_frames):
        ts = 1000.0 + i * 0.033
        # Scenario: Object approaches and crosses at frame 15
        is_approaching = (i >= 10 and i <= 20)
        ttc = 1.8 if is_approaching else None
        cpa = 0.05 if is_approaching else 0.6
        intersect = is_approaching

        pred = Prediction(
            track_id=1,
            frame_id=i,
            ttc_s=ttc,
            cpa_normalized=cpa,
            miss_distance_normalized=cpa,
            intersection_flag=intersect,
            prediction_confidence=0.88,
            bearing=0.0,
        )


        logger.log_event("prediction", pred, frame_id=i, ts=ts)
        logger.log_event("motion", {"fallback_active": False}, frame_id=i, ts=ts)

    logger.close()
    return session_dir / session_id / "session.jsonl"


def test_t32_replay_determinism(tmp_path):
    """T32: Replay the same recorded session 10 times.

    Asserts that the output sequence of RiskState and HapticCommands is
    100% deterministic and identical across all 10 runs.
    """
    session_file = _record_sample_session(tmp_path, session_id="det_session", num_frames=30)
    assert session_file.exists()

    replayer = SessionReplayer(session_file)

    # First baseline run
    baseline = replayer.run_replay()
    assert baseline["total_frames"] == 30
    assert len(baseline["states"]) == 30

    # Repeat 9 more times and assert identical match
    for iteration in range(2, 11):
        rerun = replayer.run_replay()
        assert rerun["states"] == baseline["states"], f"Divergence in states on run {iteration}"
        assert rerun["global_risks"] == baseline["global_risks"], f"Divergence in risk scores on run {iteration}"

        # Commands must match direction and urgency
        for f_idx, (cmd_base, cmd_rerun) in enumerate(zip(baseline["commands"], rerun["commands"])):
            assert cmd_base.direction == cmd_rerun.direction, f"Direction mismatch at frame {f_idx}"
            assert cmd_base.urgency == cmd_rerun.urgency, f"Urgency mismatch at frame {f_idx}"
            assert cmd_base.pattern_id == cmd_rerun.pattern_id, f"Pattern mismatch at frame {f_idx}"

    print(f"  T32: Verified 10 identical runs across {baseline['total_frames']} frames. PASSED.")


def test_t33_corrupted_record_reports_clear_error(tmp_path):
    """T33: Corrupt one record in a JSONL session.

    Asserts that the parser detects the corruption and raises a clear ValueError
    specifying the line number and error reason.
    """
    session_file = _record_sample_session(tmp_path, session_id="corrupt_session", num_frames=10)

    # Read lines
    lines = session_file.read_text(encoding="utf-8").splitlines()

    # Corrupt line 4 with malformed JSON
    corrupted_lines = list(lines)
    corrupted_lines[3] = '{"session_id": "corrupt_session", "record_type": "prediction", INVALID_JSON_HERE'
    bad_file_1 = tmp_path / "bad_json.jsonl"
    bad_file_1.write_text("\n".join(corrupted_lines) + "\n", encoding="utf-8")

    with pytest.raises(ValueError) as excinfo1:
        parse_session_file(bad_file_1)
    assert "Corrupt record at line 4" in str(excinfo1.value)
    assert "Malformed JSON" in str(excinfo1.value)

    # Corrupt line 5 with missing required field (no record_type)
    corrupted_lines2 = list(lines)
    corrupted_lines2[4] = json.dumps({"session_id": "test", "ts": 100.0, "payload": {}})
    bad_file_2 = tmp_path / "missing_field.jsonl"
    bad_file_2.write_text("\n".join(corrupted_lines2) + "\n", encoding="utf-8")

    with pytest.raises(ValueError) as excinfo2:
        parse_session_file(bad_file_2)
    assert "Corrupt record at line 5" in str(excinfo2.value)
    assert "Missing required field 'record_type'" in str(excinfo2.value)

    print("  T33: Corrupt records report specific line numbers and failure reasons. PASSED.")


def test_t34_ab_evaluation_different_weights(tmp_path):
    """T34: Run one recorded session through decision chain twice with different weights.

    Asserts that the metrics report detects a measurable, quantifiable difference.
    """
    session_file = _record_sample_session(tmp_path, session_id="ab_session", num_frames=30)
    replayer = SessionReplayer(session_file)

    # 1. Baseline engine: default caution threshold = 0.30
    engine_baseline = RiskEngine(state_thresholds={"caution": 0.30, "warning": 0.60, "critical": 0.85})
    eval_a = replayer.run_replay(risk_engine=engine_baseline)

    # 2. Hyper-sensitive engine: lower caution threshold = 0.15, fast hysteresis
    engine_sensitive = RiskEngine(
        state_thresholds={"caution": 0.15, "warning": 0.35, "critical": 0.65},
        weight_ttc=0.60,
        weight_miss_distance=0.20,
        weight_intersection_confidence=0.20,
        hysteresis_frames_up=1,
        hysteresis_frames_down=1,
    )
    eval_b = replayer.run_replay(risk_engine=engine_sensitive)


    # Compare
    diff = compare_replays(eval_a, eval_b)
    print(f"\n  [T34 A/B Report] Compared frames: {diff['compared_frames']}")
    print(f"  [T34 A/B Report] Divergent states: {diff['divergent_state_count']} ({diff['divergence_percentage']:.1f}%)")
    print(f"  [T34 A/B Report] Mean risk diff: {diff['mean_risk_difference']:.4f}")
    print(f"  [T34 A/B Report] Baseline Dist: {diff['distribution_a']}")
    print(f"  [T34 A/B Report] Sensitive Dist: {diff['distribution_b']}")

    assert diff["compared_frames"] == 30
    assert diff["divergent_state_count"] > 0, "A/B comparison failed to detect difference between distinct models"
    assert diff["mean_risk_difference"] > 0.0, "Mean risk difference should be greater than zero"


if __name__ == "__main__":
    print("--- Running M12 Tests ---")
    with tempfile.TemporaryDirectory() as td:
        p = Path(td)
        test_t32_replay_determinism(p)
        print("  -> T32 PASSED")
        test_t33_corrupted_record_reports_clear_error(p)
        print("  -> T33 PASSED")
        test_t34_ab_evaluation_different_weights(p)
        print("  -> T34 PASSED")
    print("\nALL M12 TESTS PASSED.")
