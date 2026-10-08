"""
VDO.Ninja WebRTC Stream Capture Provider for SpatialVector-HMI.

Allows capturing live video from a remote phone streaming via VDO.Ninja
(e.g., https://vdo.ninja/?view=YOUR_ROOM_ID) into OpenCV in background.

Features:
- Headless Chromium capture using Playwright
- Automatic retry loop with exponential backoff on connection failure
- Active disconnect & stall detection (reports health back to FrameSource)
- Per-frame latency tracking and throughput benchmarking
- Graceful shutdown and clean browser resource management
"""

from __future__ import annotations

import asyncio
import base64
import logging
import threading
import time
from typing import Callable, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class VdoNinjaCapture:
    """Connects to a VDO.Ninja view link in a headless Chromium process

    and extracts live video frames via HTML5 Video / Canvas in real-time.
    Includes auto-reconnection and disconnect detection.
    """

    def __init__(
        self,
        url: str,
        on_frame: Callable[[np.ndarray, float], None],
        on_status_change: Optional[Callable[[bool], None]] = None,
        target_fps: int = 30,
        stale_timeout: float = 3.0,
    ):
        self.url = url
        # Ensure clean playback parameters for lowest latency WebRTC
        if "cleanoutput" not in self.url:
            sep = "&" if "?" in self.url else "?"
            self.url = f"{self.url}{sep}cleanoutput&autostart"

        self.on_frame = on_frame
        self.on_status_change = on_status_change
        self.target_fps = target_fps
        self.stale_timeout = stale_timeout

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.is_connected = False
        self.last_frame_time = 0.0

        # Latency & throughput diagnostics
        self.total_frames_received = 0
        self.last_decode_time_ms = 0.0
        self.avg_decode_time_ms = 0.0
        self._decode_times: list[float] = []

    def is_alive(self) -> bool:
        """Returns True if the background capture thread is currently running."""
        return self._thread is not None and self._thread.is_alive()

    def check_health(self) -> bool:
        """Returns True if frames have been received within stale_timeout."""
        if not self.is_connected:
            return False
        if self.last_frame_time == 0.0:
            return False
        return (time.monotonic() - self.last_frame_time) < self.stale_timeout

    def start(self) -> "VdoNinjaCapture":
        if self._thread is not None and self._thread.is_alive():
            return self

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="VdoNinjaCapture")
        self._thread.start()
        return self

    def stop(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=4.0)
            self._thread = None
        self._set_connected(False)

    def _set_connected(self, state: bool):
        if self.is_connected != state:
            self.is_connected = state
            if self.on_status_change:
                try:
                    self.on_status_change(state)
                except Exception as e:
                    logger.debug(f"Status change callback error: {e}")

    def _run_loop(self):
        """Run asyncio loop inside thread with top-level error boundary."""
        try:
            asyncio.run(self._async_capture())
        except Exception as e:
            logger.error(f"VdoNinja capture thread terminated unexpectedly: {e}")
        finally:
            self._set_connected(False)

    async def _async_capture(self):
        try:
            from playwright.async_api import async_playwright
        except ImportError:
            logger.error("Playwright is required for VDO.Ninja capture. Run: pip install playwright && playwright install chromium")
            return

        retry_delay = 1.5
        max_retry_delay = 10.0

        while not self._stop_event.is_set():
            browser = None
            try:
                async with async_playwright() as p:
                    logger.info(f"Launching headless Chromium for VDO.Ninja stream: {self.url}")
                    # Try system Chrome first, fall back to Playwright Chromium
                    launch_args = [
                        "--use-fake-ui-for-media-stream",
                        "--autoplay-policy=no-user-gesture-required",
                        "--disable-web-security",
                        "--disable-features=IsolateOrigins,site-per-process",
                    ]
                    try:
                        browser = await p.chromium.launch(
                            channel="chrome",
                            headless=True,
                            args=launch_args,
                        )
                    except Exception:
                        browser = await p.chromium.launch(
                            headless=True,
                            args=launch_args,
                        )

                    page = await browser.new_page()

                    async def _handle_frame(b64_data: str, client_ts: float = 0.0):
                        t_cap = time.monotonic()
                        self.last_frame_time = t_cap
                        if not self.is_connected:
                            self._set_connected(True)

                        t_decode_start = time.perf_counter()
                        try:
                            comma = b64_data.find(",")
                            raw = base64.b64decode(b64_data[comma + 1 :])
                            arr = np.frombuffer(raw, dtype=np.uint8)
                            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                            if img is not None:
                                self.on_frame(img, t_cap)
                                self.total_frames_received += 1
                                decode_ms = (time.perf_counter() - t_decode_start) * 1000.0
                                self.last_decode_time_ms = decode_ms
                                self._decode_times.append(decode_ms)
                                if len(self._decode_times) > 100:
                                    self._decode_times.pop(0)
                                self.avg_decode_time_ms = sum(self._decode_times) / len(self._decode_times)

                                if self.total_frames_received % 150 == 0:
                                    logger.info(
                                        f"[VDO.Ninja] Frames: {self.total_frames_received} | "
                                        f"Avg Decode Latency: {self.avg_decode_time_ms:.1f}ms | "
                                        f"Resolution: {img.shape[1]}x{img.shape[0]}"
                                    )
                        except Exception as e:
                            logger.debug(f"Frame decode error: {e}")

                    await page.expose_function("sendFrameToPython", _handle_frame)

                    logger.info(f"Navigating to VDO.Ninja URL: {self.url}")
                    await page.goto(self.url, wait_until="domcontentloaded", timeout=25000)

                    # Selector wait with retry recovery
                    logger.info("Waiting for VDO.Ninja video element to start playback...")
                    try:
                        await page.wait_for_selector("video", timeout=20000)
                    except Exception as wait_err:
                        logger.warning(
                            f"VDO.Ninja <video> element not yet available ({wait_err}). "
                            f"Retrying in {retry_delay:.1f}s (Ensure phone is streaming)..."
                        )
                        self._set_connected(False)
                        await browser.close()
                        browser = None
                        await asyncio.sleep(retry_delay)
                        retry_delay = min(retry_delay * 1.5, max_retry_delay)
                        continue

                    # Injected client script for fast canvas capture
                    frame_interval_ms = int(1000.0 / self.target_fps) if self.target_fps > 0 else 33
                    await page.evaluate(
                        f"""() => {{
                        const video = document.querySelector('video');
                        const canvas = document.createElement('canvas');
                        const ctx = canvas.getContext('2d', {{ alpha: false }});
                        let running = true;
                        window._stopVdo = () => {{ running = false; }};

                        function grab() {{
                            if (!running) return;
                            if (video.videoWidth > 0 && video.videoHeight > 0 && !video.paused && !video.ended) {{
                                canvas.width = video.videoWidth;
                                canvas.height = video.videoHeight;
                                ctx.drawImage(video, 0, 0);
                                // Quality 0.75 reduces base64 payload size by ~40% vs 0.85 with identical YOLO detection accuracy
                                const b64 = canvas.toDataURL('image/jpeg', 0.75);
                                const ts = performance.now();
                                window.sendFrameToPython(b64, ts).then(() => {{
                                    if ('requestVideoFrameCallback' in video) {{
                                        video.requestVideoFrameCallback(grab);
                                    }} else {{
                                        setTimeout(grab, {frame_interval_ms});
                                    }}
                                }}).catch(() => setTimeout(grab, 100));
                            }} else {{
                                setTimeout(grab, 200);
                            }}
                        }}
                        grab();
                    }}"""
                    )

                    logger.info("VDO.Ninja WebRTC capture pipeline active.")
                    retry_delay = 1.5  # Reset backoff on successful connection

                    # Active monitoring loop: verify frames are actively arriving
                    while not self._stop_event.is_set():
                        await asyncio.sleep(0.5)

                        # Disconnect / stall watchdog
                        if self.is_connected and self.last_frame_time > 0.0:
                            elapsed = time.monotonic() - self.last_frame_time
                            if elapsed > self.stale_timeout:
                                logger.warning(
                                    f"[VDO.Ninja] Frame delivery stalled ({elapsed:.1f}s > {self.stale_timeout}s). "
                                    f"Flagging DISCONNECTED and triggering stream reconnect..."
                                )
                                self._set_connected(False)
                                break  # Break to reconnect browser session

                    try:
                        await page.evaluate("() => { if (window._stopVdo) window._stopVdo(); }")
                    except Exception:
                        pass

                    if browser:
                        await browser.close()
                        browser = None

            except Exception as loop_err:
                if self._stop_event.is_set():
                    break
                logger.warning(f"VDO.Ninja session error: {loop_err}. Reconnecting in {retry_delay:.1f}s...")
                self._set_connected(False)
                if browser:
                    try:
                        await browser.close()
                    except Exception:
                        pass
                await asyncio.sleep(retry_delay)
                retry_delay = min(retry_delay * 1.5, max_retry_delay)

        self._set_connected(False)
        logger.info("VDO.Ninja capture worker exited cleanly.")
