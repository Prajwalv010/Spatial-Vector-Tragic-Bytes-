"""Regression tests for telemetry payload completeness (T38, T39, T40).

Guarantees:
- T38: build_track_telemetry preserves intersection_flag as 'intersect' == True.
- T39: build_track_telemetry preserves prediction_confidence == 0.0 without dropping or nulling it.
- T40: Every track property accessed in dashboard.js updateInspector() has a matching key in build_track_telemetry(),
       and every haptic property accessed in updateHapticTelemetry() matches HapticCommand fields.
"""

from __future__ import annotations

import inspect
import re
import sys
from pathlib import Path
from dataclasses import asdict

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spatialvector.decision.schemas import HapticCommand, Prediction
from spatialvector.hmi.schemas import TelemetryMessage, build_track_telemetry
from spatialvector.perception.schemas import Track


def _make_track(track_id: int = 1, class_name: str = "Pedestrian") -> Track:
    return Track(
        track_id=track_id,
        class_name=class_name,
        bbox_history=[(100.0, 150.0, 220.0, 380.0)],
        center_history=[(160.0, 265.0)],
        estimated_image_velocity=(-2.5, 4.0),
        track_age=12,
        track_confidence=0.88,
        last_seen_frame_id=42,
    )


def _make_prediction(
    track_id: int = 1,
    intersection_flag: bool = True,
    prediction_confidence: float = 0.85,
    ttc_s: float = 2.1,
    cpa_normalized: float = 0.35,
) -> Prediction:
    return Prediction(
        track_id=track_id,
        frame_id=42,
        ttc_s=ttc_s,
        cpa_normalized=cpa_normalized,
        miss_distance_normalized=0.12,
        intersection_flag=intersection_flag,
        prediction_confidence=prediction_confidence,
        bearing=-0.15,
    )


def test_t38_intersection_flag_in_payload():
    """T38: Assert build_track_telemetry preserves intersection_flag=True as 'intersect'."""
    track = _make_track(track_id=7)
    pred = _make_prediction(track_id=7, intersection_flag=True)

    payload = build_track_telemetry([track], {7: pred})
    assert len(payload) == 1
    item = payload[0]

    assert item["track_id"] == 7
    assert item["intersect"] is True, "Expected 'intersect' key to be True when intersection_flag=True"


def test_t39_legitimate_zero_prediction_confidence_preserved():
    """T39: Assert prediction_confidence=0.0 is preserved as 0.0, not None or dropped."""
    track = _make_track(track_id=8)
    pred = _make_prediction(track_id=8, prediction_confidence=0.0)

    payload = build_track_telemetry([track], {8: pred})
    assert len(payload) == 1
    item = payload[0]

    assert item["pred_conf"] == 0.0, f"Expected pred_conf to be 0.0, got {item.get('pred_conf')}"
    assert item["pred_conf"] is not None


def test_t40_dashboard_js_keys_covered_by_payload():
    """T40: Assert all track.* and haptic.* properties referenced in dashboard.js

    have matching keys in build_track_telemetry() or HapticCommand.
    """
    repo_root = Path(__file__).resolve().parent.parent
    dashboard_js = repo_root / "web" / "dashboard" / "dashboard.js"
    assert dashboard_js.exists(), f"dashboard.js not found at {dashboard_js}"

    js_code = dashboard_js.read_text(encoding="utf-8")

    # 1. Extract track.<prop> references from updateInspector()
    inspector_match = re.search(r"function updateInspector\(\)\s*\{(.*?)\n    \}", js_code, re.DOTALL)
    assert inspector_match, "Could not locate updateInspector() function in dashboard.js"
    inspector_body = inspector_match.group(1)
    track_refs = set(re.findall(r"\btrack\.(\w+)", inspector_body))

    # Generate sample payload to inspect keys
    sample_track = _make_track(1)
    sample_pred = _make_prediction(1)
    payload = build_track_telemetry([sample_track], {1: sample_pred})
    payload_keys = set(payload[0].keys())

    missing_track_keys = track_refs - payload_keys
    assert not missing_track_keys, (
        f"dashboard.js updateInspector() references track keys missing from telemetry payload: {missing_track_keys}"
    )

    # 2. Extract haptic.<prop> references from updateHapticTelemetry()
    haptic_match = re.search(r"function updateHapticTelemetry\(haptic\)\s*\{(.*?)\n    \}", js_code, re.DOTALL)
    assert haptic_match, "Could not locate updateHapticTelemetry() function in dashboard.js"
    haptic_body = haptic_match.group(1)
    haptic_refs = set(re.findall(r"\bhaptic\.(\w+)", haptic_body))

    dummy_cmd = HapticCommand(
        timestamp=0.0,
        direction="STOP",
        urgency=1,
        pattern_id="ALL_CLEAR",
        duration_ms=200,
    )
    haptic_keys = set(asdict(dummy_cmd).keys())

    missing_haptic_keys = haptic_refs - haptic_keys
    assert not missing_haptic_keys, (
        f"dashboard.js updateHapticTelemetry() references haptic keys missing from HapticCommand: {missing_haptic_keys}"
    )
