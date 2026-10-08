"""
M05 — IMU Reader (hardware ingestion side)

Reads gyroscope (+ optional accelerometer) samples from an MPU-6050 connected
to an Arduino, streaming over serial (USB or BLE). Each sample is timestamped on
arrival using time.monotonic() — this is the canonical laptop-side timestamp used
for sync with video frames. The raw device timestamp (if the firmware sends one)
is stored separately for drift analysis, but is NOT used as the ground-truth clock.

Failure handling (matches M01 FrameSource discipline):
  - Serial port not found / not opened → status = "DISCONNECTED", keep retrying
  - Malformed packet / parse error       → status = "INVALID", skip sample
  - Timeout with no new samples          → status = "DISCONNECTED"

Threading: reads in a background thread, exposes latest sample via `get_latest()`.

Wire format expected from Arduino firmware (one line per sample, newline-terminated):
    GYRO:<gx>,<gy>,<gz>[,<ax>,<ay>,<az>][,T:<device_ts_ms>]
Example:
    GYRO:0.012,-0.003,0.001,T:12345
    GYRO:0.012,-0.003,0.001,0.01,-9.8,0.05,T:12346

If no serial hardware is available, use SimulatedIMUReader (same interface) for testing.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Optional

from spatialvector.motion.schemas import IMUSample

logger = logging.getLogger(__name__)

_SAMPLE_TIMEOUT_S = 0.5   # if no sample arrives within this window, mark DISCONNECTED


class IMUReader:
    """Background-threaded IMU reader for MPU-6050 over serial.

    Usage:
        reader = IMUReader(port="/dev/ttyUSB0", baud_rate=115200)
        reader.start()
        sample = reader.get_latest()   # returns IMUSample or None
        reader.stop()
    """

    def __init__(
        self,
        port: str = "/dev/ttyUSB0",
        baud_rate: int = 115200,
        low_pass_alpha: float = 0.8,
    ):
        """
        Args:
            port: serial port name (e.g. "COM3" on Windows, "/dev/ttyUSB0" on Linux).
            baud_rate: serial baud rate (must match Arduino firmware).
            low_pass_alpha: EMA coefficient for gyro low-pass filter.
                            0.0 = heavy filtering (very smooth), 1.0 = no filtering.
                            Default 0.8 = light smoothing to reduce chest-harness vibration.
        """
        self.port = port
        self.baud_rate = baud_rate
        self.low_pass_alpha = low_pass_alpha

        self._latest_sample: Optional[IMUSample] = None
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._filtered_gyro: Optional[tuple[float, float, float]] = None

    def start(self):
        """Start background reader thread."""
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._read_loop, daemon=True, name="IMUReader")
        self._thread.start()
        logger.info(f"[M05] IMUReader started on {self.port} @ {self.baud_rate} baud")

    def stop(self):
        """Stop background reader thread."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def get_latest(self) -> Optional[IMUSample]:
        """Return the most recent IMUSample, or None if no sample has arrived yet."""
        with self._lock:
            return self._latest_sample

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _read_loop(self):
        """Main serial read loop — runs in background thread."""
        import importlib.util
        if importlib.util.find_spec("serial") is None:
            logger.error("[M05] pyserial not installed — IMUReader cannot open serial port. "
                         "Install with: pip install pyserial. Using DISCONNECTED status.")
            self._set_disconnected()
            return

        import serial  # type: ignore
        ser: Optional[serial.Serial] = None

        while not self._stop_event.is_set():
            try:
                if ser is None or not ser.is_open:
                    ser = serial.Serial(self.port, self.baud_rate, timeout=_SAMPLE_TIMEOUT_S)
                    logger.info(f"[M05] Serial port {self.port} opened.")

                line = ser.readline().decode("ascii", errors="replace").strip()
                if not line:
                    # Timeout — no data
                    self._set_disconnected()
                    continue

                sample = _parse_imu_line(line, low_pass_alpha=self.low_pass_alpha,
                                         prev_filtered=self._filtered_gyro)
                if sample is not None and sample.status == "OK":
                    self._filtered_gyro = sample.gyro_xyz
                with self._lock:
                    self._latest_sample = sample

            except Exception as exc:
                logger.warning(f"[M05] Serial error: {exc} — retrying in 0.5s")
                self._set_disconnected()
                try:
                    if ser is not None:
                        ser.close()
                except Exception:
                    pass
                ser = None
                time.sleep(0.5)

        if ser is not None:
            try:
                ser.close()
            except Exception:
                pass

    def _set_disconnected(self):
        """Write a DISCONNECTED sentinel sample so callers always get a defined status."""
        with self._lock:
            self._latest_sample = IMUSample(
                t_arrival=time.monotonic(),
                t_device=None,
                gyro_xyz=(0.0, 0.0, 0.0),
                accel_xyz=None,
                status="DISCONNECTED",
            )


def _parse_imu_line(
    line: str,
    low_pass_alpha: float = 0.8,
    prev_filtered: Optional[tuple[float, float, float]] = None,
) -> Optional[IMUSample]:
    """Parse one ASCII line from Arduino firmware into IMUSample.

    Expected formats:
        GYRO:<gx>,<gy>,<gz>
        GYRO:<gx>,<gy>,<gz>,T:<device_ms>
        GYRO:<gx>,<gy>,<gz>,<ax>,<ay>,<az>
        GYRO:<gx>,<gy>,<gz>,<ax>,<ay>,<az>,T:<device_ms>

    Returns None on parse failure (caller should treat as INVALID).
    """
    t_arrival = time.monotonic()
    try:
        if not line.upper().startswith("GYRO:"):
            return IMUSample(t_arrival=t_arrival, t_device=None,
                             gyro_xyz=(0.0, 0.0, 0.0), status="INVALID")

        payload = line[5:]  # strip "GYRO:"
        t_device: Optional[float] = None

        # Extract optional device timestamp "T:<ms>"
        if ",T:" in payload:
            parts = payload.rsplit(",T:", 1)
            payload = parts[0]
            t_device = float(parts[1]) / 1000.0  # ms → seconds

        fields = [float(f) for f in payload.split(",")]
        if len(fields) < 3:
            return IMUSample(t_arrival=t_arrival, t_device=t_device,
                             gyro_xyz=(0.0, 0.0, 0.0), status="INVALID")

        raw_gyro = (fields[0], fields[1], fields[2])
        accel: Optional[tuple[float, float, float]] = None
        if len(fields) >= 6:
            accel = (fields[3], fields[4], fields[5])

        # Apply exponential moving average low-pass filter (gyro noise from chest harness)
        if prev_filtered is not None:
            a = low_pass_alpha
            filtered_gyro = (
                a * raw_gyro[0] + (1.0 - a) * prev_filtered[0],
                a * raw_gyro[1] + (1.0 - a) * prev_filtered[1],
                a * raw_gyro[2] + (1.0 - a) * prev_filtered[2],
            )
        else:
            filtered_gyro = raw_gyro

        return IMUSample(
            t_arrival=t_arrival,
            t_device=t_device,
            gyro_xyz=filtered_gyro,
            accel_xyz=accel,
            status="OK",
        )

    except (ValueError, IndexError) as exc:
        logger.debug(f"[M05] Failed to parse IMU line '{line}': {exc}")
        return IMUSample(t_arrival=t_arrival, t_device=None,
                         gyro_xyz=(0.0, 0.0, 0.0), status="INVALID")


# ------------------------------------------------------------------
# Simulated IMU — for testing without hardware
# ------------------------------------------------------------------

class SimulatedIMUReader:
    """Simulates an IMU stream from a callable — for tests and demos.

    All samples emitted by this reader carry status="SIMULATED" (not "OK").
    Callers and the UI MUST distinguish SIMULATED from real hardware ("OK").
    Display "IMU: SIMULATED" whenever status=="SIMULATED" is seen.

    Usage:
        def my_gyro_fn(t): return (0.01 * math.sin(t), 0.0, 0.0)
        reader = SimulatedIMUReader(gyro_fn=my_gyro_fn)
        reader.start()
        sample = reader.get_latest()
    """

    def __init__(
        self,
        gyro_fn=None,    # callable(t_monotonic) → (gx, gy, gz) rad/s
        rate_hz: float = 100.0,
        inject_disconnect_after_s: Optional[float] = None,
    ):
        """
        Args:
            gyro_fn: function of current time returning (gx, gy, gz).
                     Defaults to zero motion.
            rate_hz: simulated sample rate.
            inject_disconnect_after_s: if set, emit DISCONNECTED after this many seconds.
        """
        self.gyro_fn = gyro_fn or (lambda t: (0.0, 0.0, 0.0))
        self.rate_hz = rate_hz
        self.inject_disconnect_after_s = inject_disconnect_after_s

        self._latest_sample: Optional[IMUSample] = None
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._start_time: Optional[float] = None

    def start(self):
        self._stop_event.clear()
        self._start_time = time.monotonic()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="SimIMU")
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def get_latest(self) -> Optional[IMUSample]:
        with self._lock:
            return self._latest_sample

    def _loop(self):
        interval = 1.0 / self.rate_hz
        while not self._stop_event.is_set():
            t = time.monotonic()
            elapsed = t - (self._start_time or t)

            if (self.inject_disconnect_after_s is not None
                    and elapsed > self.inject_disconnect_after_s):
                sample = IMUSample(
                    t_arrival=t,
                    t_device=None,
                    gyro_xyz=(0.0, 0.0, 0.0),
                    status="DISCONNECTED",
                )
            else:
                gyro = self.gyro_fn(t)
                sample = IMUSample(
                    t_arrival=t,
                    t_device=elapsed,
                    gyro_xyz=gyro,
                    status="SIMULATED",  # Not real hardware — callers/UI must show "IMU: SIMULATED"
                )

            with self._lock:
                self._latest_sample = sample
            time.sleep(interval)
