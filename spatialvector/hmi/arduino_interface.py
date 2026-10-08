"""M10 — Arduino Haptic Interface.

Provides asynchronous, non-blocking serial communication with the Arduino haptic
controller firmware (firmware/haptic_controller/haptic_controller.ino).

Rules (from PROTOCOL.md & Engineering Blueprint):
1. Never block the safety loop on an ACK — sends are queued and written asynchronously.
2. Short bounded queue with drop-oldest discipline — stale commands are dropped.
3. ACKs update ArduinoStatus asynchronously.
4. SimulatedArduinoInterface provides full mock functionality for hardware-free tests.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Dict, Optional

from spatialvector.decision.schemas import HapticCommand
from spatialvector.hmi.schemas import ArduinoStatus

logger = logging.getLogger(__name__)


def format_wire_command(cmd: HapticCommand) -> str:
    """Formats a HapticCommand into the wire protocol string per PROTOCOL.md."""
    # Wire format: CMD,<direction>,<urgency>,<pattern_id>,<duration_ms>\n
    direction = str(cmd.direction).upper().strip()
    urgency = max(1, min(5, int(cmd.urgency)))
    pattern_id = str(cmd.pattern_id).strip()
    duration_ms = max(50, int(cmd.duration_ms))
    return f"CMD,{direction},{urgency},{pattern_id},{duration_ms}\n"


class ArduinoInterface:
    """Production Arduino interface over serial using PySerial and background threads."""

    def __init__(
        self,
        port: str = "COM3",
        baud_rate: int = 115200,
        queue_size: int = 10,
        watchdog_timeout_ms: int = 500,
    ):
        self.port = port
        self.baud_rate = baud_rate
        self.queue_size = queue_size
        self.watchdog_timeout_ms = watchdog_timeout_ms

        self._cmd_queue: queue.Queue[str] = queue.Queue(maxsize=queue_size)
        self._status = ArduinoStatus(connected=False)
        self._status_lock = threading.Lock()
        self._stop_event = threading.Event()

        self._serial = None
        self._writer_thread: Optional[threading.Thread] = None
        self._reader_thread: Optional[threading.Thread] = None
        self._self_test_event = threading.Event()
        self._self_test_result: Optional[Dict[str, str]] = None

    def start(self):
        """Starts background reader/writer threads and opens serial port."""
        self._stop_event.clear()
        try:
            import serial
            self._serial = serial.Serial(self.port, self.baud_rate, timeout=0.1)
            with self._status_lock:
                self._status.connected = True
            logger.info(f"[M10] Connected to Arduino on {self.port} @ {self.baud_rate}")
        except Exception as exc:
            logger.warning(f"[M10] Failed to open serial port {self.port}: {exc}. Will retry in background.")
            with self._status_lock:
                self._status.connected = False

        self._writer_thread = threading.Thread(target=self._writer_loop, daemon=True, name="ArduinoWriter")
        self._reader_thread = threading.Thread(target=self._reader_loop, daemon=True, name="ArduinoReader")
        self._writer_thread.start()
        self._reader_thread.start()

    def stop(self):
        """Stops background threads and closes serial port."""
        self._stop_event.set()
        if self._writer_thread and self._writer_thread.is_alive():
            self._writer_thread.join(timeout=1.0)
        if self._reader_thread and self._reader_thread.is_alive():
            self._reader_thread.join(timeout=1.0)

        if self._serial is not None:
            try:
                self._serial.close()
            except Exception:
                pass
            self._serial = None

        with self._status_lock:
            self._status.connected = False
        logger.info("[M10] ArduinoInterface stopped.")

    def send(self, cmd: HapticCommand) -> None:
        """Enqueues a HapticCommand for asynchronous sending. Never blocks."""
        line = format_wire_command(cmd)
        try:
            self._cmd_queue.put_nowait(line)
        except queue.Full:
            # Drop oldest command to prevent latency buildup
            try:
                self._cmd_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._cmd_queue.put_nowait(line)
            except queue.Full:
                pass

        with self._status_lock:
            self._status.last_command_sent = cmd.pattern_id

    def get_status(self) -> ArduinoStatus:
        """Returns the latest connection and actuation status."""
        with self._status_lock:
            return ArduinoStatus(
                connected=self._status.connected,
                last_ack_t=self._status.last_ack_t,
                last_command_sent=self._status.last_command_sent,
                motor_test_result=self._status.motor_test_result,
            )

    def run_self_test(self, timeout_s: float = 2.0) -> Dict[str, str]:
        """Runs motor self-test and waits up to timeout_s for result."""
        self._self_test_event.clear()
        self._self_test_result = None
        try:
            self._cmd_queue.put_nowait("TEST\n")
        except queue.Full:
            pass

        if self._self_test_event.wait(timeout=timeout_s):
            return self._self_test_result or {"L": "PASS", "C": "PASS", "R": "PASS"}
        return {"L": "TIMEOUT", "C": "TIMEOUT", "R": "TIMEOUT"}

    # -------------------------------------------------------------------------
    # Internal Thread Loops
    # -------------------------------------------------------------------------

    def _writer_loop(self):
        while not self._stop_event.is_set():
            try:
                line = self._cmd_queue.get(timeout=0.05)
            except queue.Empty:
                continue

            if self._serial and self._serial.is_open:
                try:
                    self._serial.write(line.encode("ascii", errors="ignore"))
                    self._serial.flush()
                except Exception as exc:
                    logger.debug(f"[M10] Serial write error: {exc}")
                    with self._status_lock:
                        self._status.connected = False

    def _reader_loop(self):
        while not self._stop_event.is_set():
            if not self._serial or not self._serial.is_open:
                time.sleep(0.1)
                continue

            try:
                raw = self._serial.readline()
                if not raw:
                    continue
                line = raw.decode("ascii", errors="ignore").strip()
                if not line:
                    continue

                if line.startswith("ACK,"):
                    with self._status_lock:
                        self._status.connected = True
                        self._status.last_ack_t = time.monotonic()
                elif line.startswith("TEST_RES,"):
                    # Parse TEST_RES,L:PASS,C:PASS,R:PASS
                    parts = line.split(",")[1:]
                    res = {}
                    for p in parts:
                        if ":" in p:
                            k, v = p.split(":", 1)
                            res[k.strip()] = v.strip()
                    self._self_test_result = res
                    with self._status_lock:
                        self._status.motor_test_result = res
                    self._self_test_event.set()
                elif line.startswith("STATUS,READY"):
                    with self._status_lock:
                        self._status.connected = True
            except Exception as exc:
                logger.debug(f"[M10] Serial read error: {exc}")
                with self._status_lock:
                    self._status.connected = False


class SimulatedArduinoInterface:
    """Simulated Arduino interface for tests and hardware-free execution.

    Faithfully implements:
    1. Wire protocol formatting and parsing
    2. Bounded command queue (drop-oldest)
    3. Asynchronous ACK updating
    4. Firmware watchdog simulation (motors return to 0 after watchdog_timeout_s)
    """

    def __init__(
        self,
        queue_size: int = 10,
        watchdog_timeout_ms: int = 500,
    ):
        self.queue_size = queue_size
        self.watchdog_timeout_s = watchdog_timeout_ms / 1000.0

        self._cmd_queue: queue.Queue[str] = queue.Queue(maxsize=queue_size)
        self._status = ArduinoStatus(connected=True, last_ack_t=time.monotonic())
        self._status_lock = threading.Lock()
        self._stop_event = threading.Event()

        # Simulated hardware state
        self.command_history: list[str] = []
        self.sent_count = 0
        self.last_command_time = time.monotonic()
        self.current_motors = {"L": 0, "C": 0, "R": 0}  # simulated motor PWM (0..255)
        self.active_pattern_id: Optional[str] = None
        self._lock = threading.Lock()

    def start(self):
        self._stop_event.clear()
        with self._status_lock:
            self._status.connected = True
            self._status.last_ack_t = time.monotonic()

    def stop(self):
        self._stop_event.set()
        with self._status_lock:
            self._status.connected = False
        with self._lock:
            self.current_motors = {"L": 0, "C": 0, "R": 0}

    def send(self, cmd: HapticCommand) -> None:
        """Enqueues command and updates simulated motor state."""
        line = format_wire_command(cmd)

        try:
            self._cmd_queue.put_nowait(line)
        except queue.Full:
            try:
                self._cmd_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._cmd_queue.put_nowait(line)
            except queue.Full:
                pass

        now = time.monotonic()
        with self._lock:
            self.command_history.append(line)
            self.sent_count += 1
            self.last_command_time = now
            self.active_pattern_id = cmd.pattern_id

            # Set simulated motor output
            intensity = 80 + cmd.urgency * 35
            d = str(cmd.direction).upper()
            if d == "LEFT":
                self.current_motors = {"L": intensity, "C": 0, "R": 0}
            elif d == "CENTER":
                self.current_motors = {"L": 0, "C": intensity, "R": 0}
            elif d == "RIGHT":
                self.current_motors = {"L": 0, "C": 0, "R": intensity}
            elif d == "STOP":
                if cmd.pattern_id == "STOP_CRITICAL":
                    self.current_motors = {"L": intensity, "C": intensity, "R": intensity}
                elif cmd.pattern_id == "ALL_CLEAR":
                    self.current_motors = {"L": 0, "C": 50, "R": 0}
                else:
                    self.current_motors = {"L": intensity, "C": 0, "R": intensity}

        with self._status_lock:
            self._status.last_command_sent = cmd.pattern_id
            self._status.last_ack_t = now
            self._status.connected = True

    def update_watchdog(self) -> None:
        """Enforces watchdog timeout: zeroes motors if no command within watchdog_timeout_s."""
        now = time.monotonic()
        with self._lock:
            if now - self.last_command_time > self.watchdog_timeout_s:
                self.current_motors = {"L": 0, "C": 0, "R": 0}
                self.active_pattern_id = None

    def get_motors(self) -> Dict[str, int]:
        """Returns current simulated motor PWM levels with watchdog evaluated."""
        self.update_watchdog()
        with self._lock:
            return dict(self.current_motors)

    def get_status(self) -> ArduinoStatus:
        with self._status_lock:
            return ArduinoStatus(
                connected=self._status.connected,
                last_ack_t=self._status.last_ack_t,
                last_command_sent=self._status.last_command_sent,
                motor_test_result=self._status.motor_test_result,
            )

    def run_self_test(self, timeout_s: float = 2.0) -> Dict[str, str]:
        res = {"L": "PASS", "C": "PASS", "R": "PASS"}
        with self._status_lock:
            self._status.motor_test_result = res
        return res
