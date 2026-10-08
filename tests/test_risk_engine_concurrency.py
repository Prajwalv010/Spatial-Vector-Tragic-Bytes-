"""Tests for M08 RiskEngine Concurrency & Live-Threshold Reconfiguration.

Simulates high-frequency updates from the primary decision loop on Thread A
while Thread B performs repeated atomic configuration updates, asserting:
1. Zero unhandled exceptions or torn/partial reads.
2. Threshold mutations take immediate, consistent effect on subsequent frames.
3. Server-side validation rejects corrupt/out-of-bounds configurations without destabilizing the engine.
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from spatialvector.decision.risk_engine import RiskEngine, RiskEngineConfig
from spatialvector.decision.schemas import Prediction


def _make_dummy_prediction(ttc: float = 2.0, cpa: float = 0.1, intersect: bool = True, conf: float = 0.9) -> Prediction:
    return Prediction(
        track_id=1,
        frame_id=1,
        ttc_s=ttc,
        cpa_normalized=cpa,
        miss_distance_normalized=cpa,
        intersection_flag=intersect,
        prediction_confidence=conf,
        bearing=0.0,
    )


def test_concurrency_stress_reads_and_writes():
    """Tight-loop concurrency test: 500+ updates on reader thread while writer mutates config."""
    engine = RiskEngine()
    stop_event = threading.Event()
    exceptions = []
    read_count = 0
    write_count = 0

    def reader_loop():
        nonlocal read_count
        preds = [_make_dummy_prediction()]
        try:
            while not stop_event.is_set():
                rs = engine.update(preds, fallback_active=False)
                assert rs.state in ("SAFE", "CAUTION", "WARNING", "CRITICAL", "DEGRADED")
                read_count += 1
                time.sleep(0.001)
        except Exception as exc:
            exceptions.append(exc)

    def writer_loop():
        nonlocal write_count
        configs = [
            RiskEngineConfig(
                weight_ttc=0.4,
                weight_miss_distance=0.4,
                weight_intersection_confidence=0.2,
                state_thresholds={"caution": 0.25, "warning": 0.55, "critical": 0.80},
                hysteresis_frames_up=2,
                hysteresis_frames_down=4,
            ),
            RiskEngineConfig(
                weight_ttc=0.6,
                weight_miss_distance=0.2,
                weight_intersection_confidence=0.2,
                state_thresholds={"caution": 0.35, "warning": 0.65, "critical": 0.90},
                hysteresis_frames_up=3,
                hysteresis_frames_down=5,
            ),
            RiskEngineConfig(
                weight_ttc=0.5,
                weight_miss_distance=0.3,
                weight_intersection_confidence=0.2,
                state_thresholds={"caution": 0.30, "warning": 0.60, "critical": 0.85},
                hysteresis_frames_up=1,
                hysteresis_frames_down=1,
            ),
        ]
        try:
            for i in range(120):
                if stop_event.is_set():
                    break
                cfg = configs[i % len(configs)]
                engine.update_config(cfg)
                write_count += 1
                time.sleep(0.003)
        except Exception as exc:
            exceptions.append(exc)

    t_reader = threading.Thread(target=reader_loop, daemon=True)
    t_writer = threading.Thread(target=writer_loop, daemon=True)

    t_reader.start()
    t_writer.start()

    t_writer.join(timeout=3.0)
    stop_event.set()
    t_reader.join(timeout=1.0)

    assert len(exceptions) == 0, f"Encountered concurrency exceptions: {exceptions}"
    assert read_count > 100, f"Expected at least 100 reads, got {read_count}"
    assert write_count > 30, f"Expected at least 30 writes, got {write_count}"


def test_threshold_change_takes_effect_on_subsequent_frame():
    """Validates that changing thresholds changes risk classification deterministically on the next frame."""
    engine = RiskEngine(
        state_thresholds={"caution": 0.30, "warning": 0.60, "critical": 0.85},
        hysteresis_frames_up=1,
        hysteresis_frames_down=1,
    )

    pred = _make_dummy_prediction(ttc=3.0, cpa=0.2, intersect=True, conf=0.85)

    # Frame 1: Standard thresholds
    rs1 = engine.update([pred])
    initial_risk = rs1.global_risk
    assert 0.40 <= initial_risk <= 0.70

    # Increase warning threshold so initial_risk is now strictly CAUTION
    new_cfg = RiskEngineConfig(
        state_thresholds={"caution": 0.20, "warning": 0.95, "critical": 0.99},
        hysteresis_frames_up=1,
        hysteresis_frames_down=1,
    )
    engine.update_config(new_cfg)

    # Frame 2: Must evaluate with new threshold
    rs2 = engine.update([pred])
    assert rs2.state == "CAUTION", f"Expected CAUTION with warning=0.95, got {rs2.state}"


def test_config_validation_rejects_corrupt_thresholds():
    """Validates that invalid thresholds are rejected before mutating engine state."""
    engine = RiskEngine()
    current_cfg = engine.config

    # Inverted thresholds: warning < caution
    with pytest.raises(ValueError, match="strictly ascending"):
        engine.update_config({
            "state_thresholds": {"caution": 0.70, "warning": 0.50, "critical": 0.85}
        })

    # Excessive weights sum > 1.05
    with pytest.raises(ValueError, match="Sum of weights"):
        engine.update_config({
            "weight_ttc": 0.8,
            "weight_miss_distance": 0.5,
            "weight_intersection_confidence": 0.2,
        })

    # Hysteresis <= 0
    with pytest.raises(ValueError, match="hysteresis_frames_up"):
        engine.update_config({"hysteresis_frames_up": 0})

    # Engine state preserved
    assert engine.config == current_cfg
