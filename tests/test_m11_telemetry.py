"""Tests for Module M11 — Local Telemetry Gateway & Phone Dashboard.

Tests:
  T29: Phone client connects, disconnects, and reconnects mid-session; server keeps running.
  T30: End-to-end telemetry latency benchmark (server broadcast -> client receipt).
  T31: Removability test: pipeline and haptics run with TelemetryServer completely absent.
"""

import asyncio
import json
from pathlib import Path
import sys
import time

import pytest
import websockets

# Add repository root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from spatialvector.decision.schemas import HapticCommand, RiskState
from spatialvector.hmi.arduino_interface import SimulatedArduinoInterface
from spatialvector.hmi.schemas import TelemetryMessage
from spatialvector.hmi.telemetry_server import TelemetryServer


def _make_telemetry_msg(frame_id: int = 1, state: str = "SAFE", risk: float = 0.0) -> TelemetryMessage:
    return TelemetryMessage(
        session_id="test_session_123",
        ts=time.monotonic(),
        frame_id=frame_id,
        tracks=[{"track_id": 1, "class_name": "person", "cpa": 0.5, "ttc_s": 2.5}],
        risk_state={
            "state": state,
            "global_risk": risk,
            "confidence": 0.95,
            "corridor_risks": {"left": 0.0, "center": risk, "right": 0.0},
            "reason_codes": ["ttc_low:2.5s"],
        },
        haptic={
            "direction": "LEFT",
            "urgency": 3,
            "pattern_id": "LEFT_MED",
            "duration_ms": 300,
        },
        pipeline_health={"camera": "OK", "imu": "OK", "arduino": "OK"},
    )


def test_t29_client_disconnect_and_reconnect():
    """T29: Simulate phone client disconnecting and reconnecting mid-stream.

    Asserts that server never crashes and pipeline broadcasts continue seamlessly.
    """
    server = TelemetryServer(host="127.0.0.1", port=8901)
    server.start()

    async def client_lifecycle():
        uri = "ws://127.0.0.1:8901/ws/telemetry"

        # 1. Connect Client 1
        async with websockets.connect(uri) as ws1:
            # Broadcast frame 1
            server.broadcast(_make_telemetry_msg(frame_id=1))
            msg1 = await asyncio.wait_for(ws1.recv(), timeout=2.0)
            data1 = json.loads(msg1)
            assert data1["frame_id"] == 1

        # Client 1 abruptly disconnected here (context manager exited)
        time.sleep(0.1)

        # Broadcast frames while no client is connected — must NOT crash
        for fid in range(2, 6):
            server.broadcast(_make_telemetry_msg(frame_id=fid))

        # 2. Reconnect Client 2
        async with websockets.connect(uri) as ws2:
            server.broadcast(_make_telemetry_msg(frame_id=6))
            msg2 = await asyncio.wait_for(ws2.recv(), timeout=2.0)
            data2 = json.loads(msg2)
            if data2["frame_id"] == 5:
                # Received cached latest state on connect; next is live broadcast
                msg3 = await asyncio.wait_for(ws2.recv(), timeout=2.0)
                data3 = json.loads(msg3)
                assert data3["frame_id"] == 6
            else:
                assert data2["frame_id"] == 6


    try:
        asyncio.run(client_lifecycle())
    finally:
        server.stop()


def test_t30_telemetry_latency_benchmark():
    """T30: Measure end-to-end telemetry transport latency.

    Broadcasts 20 messages, measures time from pre-broadcast monotonic timestamp
    to client receipt, and reports average latency in milliseconds.
    """
    server = TelemetryServer(host="127.0.0.1", port=8902)
    server.start()

    latencies_ms = []

    async def measure():
        uri = "ws://127.0.0.1:8902/ws/telemetry"
        async with websockets.connect(uri) as ws:
            for i in range(20):
                t_send = time.monotonic()
                msg = _make_telemetry_msg(frame_id=i)
                msg.ts = t_send
                server.broadcast(msg)

                raw = await asyncio.wait_for(ws.recv(), timeout=2.0)
                t_recv = time.monotonic()

                # Calculate one-way local transport latency
                lat_ms = (t_recv - t_send) * 1000.0
                latencies_ms.append(lat_ms)
                await asyncio.sleep(0.01)

    try:
        asyncio.run(measure())
    finally:
        server.stop()

    assert len(latencies_ms) == 20
    avg_lat = sum(latencies_ms) / len(latencies_ms)
    max_lat = max(latencies_ms)
    print(f"\n  [T30 Benchmark] Telemetry Latency: avg={avg_lat:.2f}ms, max={max_lat:.2f}ms over 20 messages")

    # Local WebSocket roundtrip should easily be < 50ms
    assert avg_lat < 50.0, f"Average telemetry latency too high: {avg_lat:.2f}ms"


def test_t31_removability_pipeline_runs_without_telemetry():
    """T31: Run the full M07->M10 decision & haptics loop with NO telemetry server.

    Asserts that the safety loop executes with zero dependency on M11.
    """
    # Initialize M10 interface
    arduino = SimulatedArduinoInterface(queue_size=10, watchdog_timeout_ms=500)
    arduino.start()

    # Note: TelemetryServer is NOT started!
    telemetry_server = None

    try:
        # Simulate 50 frames passing through decision & actuation
        commands_sent = 0
        for i in range(50):
            # Synthetic command from M09
            cmd = HapticCommand(
                timestamp=time.monotonic(),
                direction="LEFT" if i % 2 == 0 else "RIGHT",
                urgency=(i % 5) + 1,
                pattern_id=f"PAT_{i}",
                duration_ms=250,
            )

            # Send to M10
            arduino.send(cmd)
            commands_sent += 1

            # Broadcast if telemetry existed (guards must allow None)
            if telemetry_server is not None:
                telemetry_server.broadcast(_make_telemetry_msg(frame_id=i))

        # Assert all 50 commands were successfully processed by M10
        assert commands_sent == 50
        status = arduino.get_status()
        assert status.connected is True
        assert status.last_command_sent == "PAT_49"

    finally:
        arduino.stop()


if __name__ == "__main__":
    print("--- Running M11 Tests ---")
    test_t29_client_disconnect_and_reconnect()
    print("  -> T29 PASSED")
    test_t30_telemetry_latency_benchmark()
    print("  -> T30 PASSED")
    test_t31_removability_pipeline_runs_without_telemetry()
    print("  -> T31 PASSED")
    print("\nALL M11 TESTS PASSED.")
