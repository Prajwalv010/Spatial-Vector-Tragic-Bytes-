"""Tests for Module M10 — Arduino Haptic Interface.

Tests:
  T26: Direct command test for left/center/right motors
  T27: Command timeout returns motors to safe idle (watchdog test)
  T28: 1000-command rapid sequence (no deadlock, bounded queue, non-blocking)
"""

import time
from pathlib import Path
import sys

import pytest

# Add repository root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from spatialvector.decision.schemas import HapticCommand
from spatialvector.hmi.arduino_interface import (
    SimulatedArduinoInterface,
    format_wire_command,
)


def _make_cmd(direction: str, urgency: int = 3, pattern_id: str = "TEST_PAT", duration_ms: int = 300) -> HapticCommand:
    return HapticCommand(
        timestamp=time.monotonic(),
        direction=direction,
        urgency=urgency,
        pattern_id=pattern_id,
        duration_ms=duration_ms,
    )


def test_t26_direct_commands_all_three_motors():
    """T26: Direct command test for left/center/right/stop motors.

    Verifies wire formatting and simulated motor activation across channels.
    """
    sim = SimulatedArduinoInterface(queue_size=10, watchdog_timeout_ms=500)
    sim.start()

    try:
        # 1. Test LEFT motor
        cmd_left = _make_cmd("LEFT", urgency=4, pattern_id="LEFT_FAST", duration_ms=200)
        wire_left = format_wire_command(cmd_left)
        assert wire_left == "CMD,LEFT,4,LEFT_FAST,200\n"

        sim.send(cmd_left)
        motors = sim.get_motors()
        assert motors["L"] > 0, f"Expected Left motor > 0, got {motors}"
        assert motors["C"] == 0, f"Expected Center motor 0, got {motors}"
        assert motors["R"] == 0, f"Expected Right motor 0, got {motors}"

        # 2. Test CENTER motor
        cmd_center = _make_cmd("CENTER", urgency=3, pattern_id="CENTER_MED", duration_ms=300)
        sim.send(cmd_center)
        motors = sim.get_motors()
        assert motors["C"] > 0, f"Expected Center motor > 0, got {motors}"
        assert motors["L"] == 0, f"Expected Left motor 0, got {motors}"
        assert motors["R"] == 0, f"Expected Right motor 0, got {motors}"

        # 3. Test RIGHT motor
        cmd_right = _make_cmd("RIGHT", urgency=2, pattern_id="RIGHT_SLOW", duration_ms=400)
        sim.send(cmd_right)
        motors = sim.get_motors()
        assert motors["R"] > 0, f"Expected Right motor > 0, got {motors}"
        assert motors["L"] == 0, f"Expected Left motor 0, got {motors}"
        assert motors["C"] == 0, f"Expected Center motor 0, got {motors}"

        # 4. Test STOP (Critical)
        cmd_stop = _make_cmd("STOP", urgency=5, pattern_id="STOP_CRITICAL", duration_ms=500)
        sim.send(cmd_stop)
        motors = sim.get_motors()
        assert motors["L"] > 0 and motors["C"] > 0 and motors["R"] > 0, f"Expected all motors > 0, got {motors}"

        # Status check
        status = sim.get_status()
        assert status.connected is True
        assert status.last_command_sent == "STOP_CRITICAL"
        assert status.last_ack_t is not None

        # Self-test check
        res = sim.run_self_test()
        assert res.get("L") == "PASS" and res.get("C") == "PASS" and res.get("R") == "PASS"

    finally:
        sim.stop()


def test_t27_watchdog_timeout_returns_motors_to_idle():
    """T27: Command timeout returns motors to safe idle.

    If laptop stops sending commands, the firmware watchdog must turn off motors.
    """
    sim = SimulatedArduinoInterface(queue_size=10, watchdog_timeout_ms=300)  # 300ms watchdog
    sim.start()

    try:
        # Trigger motor vibration
        cmd = _make_cmd("LEFT", urgency=4, pattern_id="LEFT_FAST", duration_ms=1000)
        sim.send(cmd)

        # Immediate check: motor is active
        motors_active = sim.get_motors()
        assert motors_active["L"] > 0, f"Motor should be active right after send: {motors_active}"

        # Sleep past the watchdog timeout (300ms + 50ms buffer)
        time.sleep(0.35)

        # Motor must now be 0 (watchdog kicked in)
        motors_idle = sim.get_motors()
        assert motors_idle == {"L": 0, "C": 0, "R": 0}, (
            f"Watchdog failed to idle motors after timeout! Current: {motors_idle}"
        )
    finally:
        sim.stop()


def test_t28_rapid_1000_commands_no_deadlock():
    """T28: Send 1000 commands in rapid sequence.

    Asserts:
    1. No deadlock occurs.
    2. Internal queue never grows unboundedly (bounded to queue_size).
    3. Processing completes in sub-second time (high throughput, non-blocking).
    """
    queue_size = 10
    sim = SimulatedArduinoInterface(queue_size=queue_size, watchdog_timeout_ms=500)
    sim.start()

    try:
        t_start = time.monotonic()
        for i in range(1000):
            cmd = _make_cmd(
                direction="LEFT" if i % 2 == 0 else "RIGHT",
                urgency=(i % 5) + 1,
                pattern_id=f"PAT_{i}",
                duration_ms=200,
            )
            sim.send(cmd)

        elapsed = time.monotonic() - t_start

        # Assert no deadlock and fast execution
        assert elapsed < 1.0, f"1000 rapid sends took too long ({elapsed:.3f}s) — possible lock contention"

        # Assert queue is bounded
        q_size = sim._cmd_queue.qsize()
        assert q_size <= queue_size, f"Queue grew to {q_size}, exceeding maxsize {queue_size}"

        # Sent count matches
        assert sim.sent_count == 1000, f"Expected 1000 sent, got {sim.sent_count}"

    finally:
        sim.stop()


if __name__ == "__main__":
    print("--- Running M10 Tests ---")
    test_t26_direct_commands_all_three_motors()
    print("  -> T26 PASSED")
    test_t27_watchdog_timeout_returns_motors_to_idle()
    print("  -> T27 PASSED")
    test_t28_rapid_1000_commands_no_deadlock()
    print("  -> T28 PASSED")
    print("\nALL M10 TESTS PASSED.")
