"""Tests for TelemetryServer Settings API and Session Export Endpoints.

Verifies:
1. GET /api/settings/risk-thresholds returns active config.
2. POST /api/settings/risk-thresholds updates running engine config immediately.
3. POST /api/settings/risk-thresholds rejects invalid bounds with HTTP 400.
4. GET /api/sessions returns recorded session list.
5. GET /api/sessions/{session_id}/export returns JSON array or CSV format.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from spatialvector.decision.risk_engine import RiskEngine
from spatialvector.hmi.logger import SessionLogger
from spatialvector.hmi.telemetry_server import TelemetryServer


@pytest.fixture
def client_and_engine(tmp_path):
    server = TelemetryServer(port=8999)
    engine = RiskEngine()
    server.set_risk_engine(engine)
    client = TestClient(server.app)
    return client, engine


def test_get_risk_thresholds(client_and_engine):
    client, engine = client_and_engine
    resp = client.get("/api/settings/risk-thresholds")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "OK"
    assert data["source"] == "live_engine"
    assert "state_thresholds" in data["config"]
    assert data["config"]["state_thresholds"]["caution"] == 0.30


def test_post_risk_thresholds_valid(client_and_engine):
    client, engine = client_and_engine
    payload = {
        "weight_ttc": 0.45,
        "weight_miss_distance": 0.35,
        "weight_intersection_confidence": 0.20,
        "state_thresholds": {"caution": 0.28, "warning": 0.58, "critical": 0.82},
        "hysteresis_frames_up": 2,
        "hysteresis_frames_down": 4,
        "degraded_confidence_threshold": 0.30,
        "horizon_s": 6.0,
    }
    resp = client.post("/api/settings/risk-thresholds", json=payload)
    assert resp.status_code == 200
    res_data = resp.json()
    assert res_data["status"] == "OK"
    assert res_data["config"]["weight_ttc"] == 0.45

    # Verify live engine updated immediately
    assert engine.w_ttc == 0.45
    assert engine.thresholds["caution"] == 0.28
    assert engine.hyst_up == 2
    assert engine.horizon_s == 6.0


def test_post_risk_thresholds_invalid_rejection(client_and_engine):
    client, engine = client_and_engine
    # Inverted thresholds
    payload = {
        "state_thresholds": {"caution": 0.80, "warning": 0.40, "critical": 0.90}
    }
    resp = client.post("/api/settings/risk-thresholds", json=payload)
    assert resp.status_code == 400
    assert "strictly ascending" in resp.json()["detail"]


def test_session_list_and_export(tmp_path, monkeypatch):
    # Create mock session
    session_id = f"test_export_{int(tmp_path.stat().st_mtime)}"
    logger = SessionLogger(session_id=session_id, base_dir=tmp_path)
    logger.log_event("risk", {"state": "WARNING", "global_risk": 0.65, "corridor_risks": {"left": 0.1, "center": 0.7, "right": 0.2}, "confidence": 0.9, "reason_codes": ["ttc_low:2.1s"]}, frame_id=1)
    logger.log_event("haptic", {"direction": "LEFT", "urgency": 3, "pattern_id": "LEFT_MED", "duration_ms": 300}, frame_id=1)
    logger.close()

    server = TelemetryServer(port=8998)
    client = TestClient(server.app)

    # Monkeypatch base_dir for logger queries in server routes
    import spatialvector.hmi.logger as log_mod
    orig_list = log_mod.list_sessions
    orig_json = log_mod.export_session_json
    orig_csv = log_mod.export_session_csv

    monkeypatch.setattr(log_mod, "list_sessions", lambda base_dir=None: orig_list(base_dir=tmp_path))
    monkeypatch.setattr(log_mod, "export_session_json", lambda s, base_dir=None: orig_json(s, base_dir=tmp_path))
    monkeypatch.setattr(log_mod, "export_session_csv", lambda s, base_dir=None: orig_csv(s, base_dir=tmp_path))

    # Test list
    res_list = client.get("/api/sessions")
    assert res_list.status_code == 200
    sessions = res_list.json()["sessions"]
    assert any(s["session_id"] == session_id for s in sessions)

    # Test JSON export
    res_json = client.get(f"/api/sessions/{session_id}/export?format=json")
    assert res_json.status_code == 200
    assert res_json.headers["content-type"] == "application/json"
    records = res_json.json()
    assert len(records) >= 2

    # Test CSV export
    res_csv = client.get(f"/api/sessions/{session_id}/export?format=csv")
    assert res_csv.status_code == 200
    assert "text/csv" in res_csv.headers["content-type"]
    csv_text = res_csv.text
    assert "global_risk" in csv_text
    assert "corridor_center" in csv_text
    assert "LEFT_MED" in csv_text
