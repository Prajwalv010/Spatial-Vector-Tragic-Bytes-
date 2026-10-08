"""SpatialVector-HMI — Master System Launcher (run.py)

Unified single command to launch the entire SpatialVector-HMI system:
1. Telemetry Gateway & Web Dashboard Server (FastAPI + WebSocket on port 8081)
2. Interactive Mobile & Desktop Dashboard (web/dashboard served over HTTP)
3. Camera Stream Resolver (VDO.Ninja WebRTC, USB webcam, or RTSP/file)
4. Decision & Safety Pipeline (M01-M09 Perception, Flow, Prediction, Risk Engine, Corridor Policy)
5. Haptic Output & Real-Time Telemetry Broadcasting (M10-M12)

Usage:
    python run.py                     # Live mode (reads source from camera_source.txt)
    python run.py --synthetic         # Standalone synthetic demo mode (no camera needed)
    python run.py --source 0          # Local webcam
    python run.py --port 8081         # Custom dashboard port
    python run.py --no-browser        # Do not automatically launch browser
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import logging
from pathlib import Path
import shutil
import sys
import time
import webbrowser

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from spatialvector.decision.corridor_policy import CorridorPolicy
from spatialvector.decision.prediction import CollisionPredictor
from spatialvector.decision.risk_engine import RiskEngine
from spatialvector.decision.schemas import HapticCommand, Prediction, RiskState
from spatialvector.hmi.arduino_interface import ArduinoInterface, SimulatedArduinoInterface
from spatialvector.hmi.schemas import TelemetryMessage, build_track_telemetry, build_pipeline_health
from spatialvector.hmi.telemetry_server import TelemetryServer
from spatialvector.motion.schemas import ObjectGeometry
from spatialvector.perception.schemas import Frame, Track
from spatialvector.perception.source_resolver import resolve_camera_source

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("SpatialVector-Runner")


def parse_args():
    p = argparse.ArgumentParser(description="SpatialVector-HMI — Master System Launcher")
    p.add_argument("--port", type=int, default=8081, help="Port for dashboard & telemetry (default: 8081)")
    p.add_argument("--host", default="0.0.0.0", help="Host address (default: 0.0.0.0)")
    p.add_argument("--source", default=None, help="Camera index, video file, or VDO.Ninja URL (default: reads camera_source.txt)")
    p.add_argument("--arduino-port", default=None, help="Arduino serial port (e.g. /dev/ttyACM0 or COM3). If not set, uses simulated interface.")
    p.add_argument("--sim-arduino", action="store_true", help="Force simulated Arduino even if port is set")
    p.add_argument("--synthetic", action="store_true", help="Run in synthetic demo mode (no camera needed)")
    p.add_argument("--fps", type=float, default=15.0, help="Target pipeline FPS (default: 15.0)")
    p.add_argument("--delay", type=float, default=0.03, help="Pacing delay in seconds (default: 0.03s)")
    p.add_argument("--view", action="store_true", help="Open local OpenCV debug window")
    p.add_argument("--no-browser", action="store_true", help="Do not automatically open default browser")
    return p.parse_args()


def check_prerequisites(is_vdo_ninja: bool):
    """Verifies that optional or required libraries are present."""
    if is_vdo_ninja:
        try:
            import playwright  # noqa: F401
        except ImportError:
            print("\n[!] WARNING: Playwright is required for VDO.Ninja phone streaming.")
            print("    Run: pip install playwright && playwright install chromium\n")


class SyntheticTrackGenerator:
    """Generates dynamic, realistic synthetic trajectories for standalone demo mode."""

    def __init__(self):
        self.frame_id = 0

    def step(self) -> tuple[list[Track], list[Prediction], RiskState, HapticCommand]:
        self.frame_id += 1
        t = (self.frame_id % 120) / 120.0  # 8-second cycle at 15 FPS

        # Approaching obstacle in center corridor
        dist = max(0.5, 6.0 * (1.0 - t))
        ttc = max(0.8, dist / 2.2)
        cpa = 0.35 + 0.1 * (1.0 - t)
        risk = 0.85 if t > 0.4 else (0.2 + t * 1.5)
        state = "CRITICAL" if risk > 0.80 else ("WARNING" if risk > 0.50 else "CAUTION")

        tracks = [
            Track(
                track_id=2,
                class_name="Scooter",
                bbox_history=[(160.0, 80.0, 240.0, 190.0)],
                center_history=[(200.0, 135.0)],
                estimated_image_velocity=(0.0, 45.0),
                track_age=self.frame_id,
                track_confidence=0.88,
                last_seen_frame_id=self.frame_id,
            )
        ]
        predictions = [
            Prediction(
                track_id=2,
                frame_id=self.frame_id,
                ttc_s=ttc,
                cpa_normalized=cpa,
                miss_distance_normalized=cpa,
                intersection_flag=True,
                prediction_confidence=0.91,
                bearing=0.035,
            )
        ]
        now = time.monotonic()
        risk_state = RiskState(
            timestamp=now,
            global_risk=risk,
            state=state,
            reason_codes=[f"hazard:track_2:ttc={ttc:.1f}s", "center_corridor_blocked"],
            corridor_risks={"left": 0.12, "center": round(risk, 2), "right": 0.23},
            confidence=0.91,
            recommended_horizon_s=5.0,
        )
        haptic_cmd = HapticCommand(
            timestamp=now,
            direction="LEFT",
            urgency=3 if state == "WARNING" else (4 if state == "CRITICAL" else 1),
            pattern_id=f"LEFT_{'FAST' if state == 'CRITICAL' else 'MEDIUM'}",
            duration_ms=320,
        )
        return tracks, predictions, risk_state, haptic_cmd


def build_waiting_telemetry_message(
    session_id: str,
    ts: float,
    frame_count: int,
    consecutive_misses: int,
    threshold: int = 20,
) -> TelemetryMessage:
    """Constructs an honest, un-fabricated telemetry message during camera acquisition gaps."""
    camera_health = "CONNECTING" if consecutive_misses < threshold else "DISCONNECTED"
    reason = "waiting_for_camera" if camera_health == "CONNECTING" else "camera_disconnected"
    return TelemetryMessage(
        session_id=session_id,
        ts=ts,
        frame_id=frame_count,
        tracks=[],
        risk_state=asdict(RiskState(
            timestamp=ts,
            global_risk=0.0,
            state="DEGRADED",
            reason_codes=[reason],
            corridor_risks={"left": 0.0, "center": 0.0, "right": 0.0},
            confidence=0.0,
            recommended_horizon_s=0.0,
        )),
        haptic=asdict(HapticCommand(
            timestamp=ts,
            direction="STOP",
            urgency=1,
            pattern_id="DEGRADED_WARN",
            duration_ms=200,
        )),
        pipeline_health={"camera": camera_health, "imu": "UNKNOWN", "arduino": "UNKNOWN"},
    )


def main():
    args = parse_args()

    print("\n" + "=" * 74)
    print("      SpatialVector-HMI — Autonomous Navigation & Safety System       ")
    print("                 Nirmaan 2026 Hackathon Master Runner                 ")
    print("=" * 74)

    session_id = f"sess_{int(time.time())}"

    # 1. Initialize Telemetry & Web Server
    print(f"[*] Starting Telemetry Server on http://{args.host}:{args.port}")
    telemetry_server = TelemetryServer(host=args.host, port=args.port)
    telemetry_server.start()

    dash_url = f"http://localhost:{args.port}/"
    ws_url = f"ws://localhost:{args.port}/ws/telemetry"
    print(f"[*] Dashboard URL:  {dash_url}")
    print(f"[*] WebSocket Feed: {ws_url}")

    # 2. Launch Browser
    if not args.no_browser:
        print(f"[*] Opening dashboard in browser: {dash_url}")
        try:
            webbrowser.open(dash_url)
        except Exception:
            pass

    # 3. Decision Pipeline Setup
    predictor = CollisionPredictor(horizon_s=5.0, contact_threshold_normalized=0.05, corridor_width_normalized=0.12)
    engine = RiskEngine(
        weight_ttc=0.50,
        weight_miss_distance=0.30,
        weight_intersection_confidence=0.20,
        state_thresholds={"caution": 0.30, "warning": 0.60, "critical": 0.85},
        hysteresis_frames_up=3,
        hysteresis_frames_down=5,
        degraded_confidence_threshold=0.25,
    )
    telemetry_server.set_risk_engine(engine)
    policy = CorridorPolicy(all_unsafe_risk_threshold=0.70, trend_window_frames=5)
    # Use real ArduinoInterface if --arduino-port is given, else simulated
    if not args.sim_arduino and args.arduino_port:
        arduino = ArduinoInterface(port=args.arduino_port, baud_rate=115200)
        print(f"[*] Arduino: REAL hardware on {args.arduino_port}")
    else:
        arduino = SimulatedArduinoInterface()
        if args.sim_arduino:
            print("[*] Arduino: SIMULATED (--sim-arduino flag set)")
        else:
            print("[*] Arduino: SIMULATED (no --arduino-port given; use --arduino-port /dev/ttyACM0 for real hardware)")
    arduino.start()

    simple_url = f"http://localhost:{args.port}/app_simple/"
    print(f"[*] Simplified 3-Tab UI: {simple_url}")

    # 4. Pipeline Execution
    if args.synthetic:
        print("\n[*] MODE: Synthetic Demo (Gate C/D/E)")
        print("[*] Generating simulated obstacles and broadcasting telemetry...")
        synth = SyntheticTrackGenerator()
        try:
            while True:
                ts = time.monotonic()
                tracks, predictions, risk_state, cmd = synth.step()
                arduino.send(cmd)

                pred_map = {p.track_id: p for p in predictions}
                telemetry_server.broadcast(TelemetryMessage(
                    session_id=session_id,
                    ts=ts,
                    frame_id=synth.frame_id,
                    tracks=build_track_telemetry(tracks, pred_map),
                    risk_state=asdict(risk_state),
                    haptic=asdict(cmd),
                    pipeline_health={"camera": "OK (Synthetic)", "imu": "SIMULATED", "arduino": "SIMULATED"},
                    frame_width=640,
                    frame_height=480,
                ))
                time.sleep(args.delay)
        except KeyboardInterrupt:
            print("\n[*] Stopping runner...")
    else:
        # Live Pipeline Execution
        from spatialvector.motion.ego_motion import EgoMotionCompensator
        from spatialvector.motion.geometry import compute_geometry_batch
        from spatialvector.motion.imu_reader import SimulatedIMUReader
        from spatialvector.motion.optical_flow import OpticalFlowEstimator
        from spatialvector.perception.frame_source import FrameSource
        from spatialvector.perception.tracker import MultiObjectTracker

        src, origin = resolve_camera_source(cli_source=args.source)
        is_vdo = isinstance(src, str) and "vdo.ninja" in src.lower()
        check_prerequisites(is_vdo)

        print(f"\n[*] MODE: Live Perception Pipeline")
        print(f"[*] Camera Source: {src}")
        print(f"[*] Source Origin: {origin}")
        print("[*] Press Ctrl+C to stop pipeline\n")

        is_network = isinstance(src, str) and any(src.lower().startswith(p) for p in ("http://", "https://", "rtsp://"))
        frame_src = FrameSource(
            source=src,
            target_fps=args.fps,
            queue_size=5,
            loop_video=(not is_network and isinstance(src, str)),
        )
        tracker = MultiObjectTracker(backend="bytetrack", history_length=10, confidence_threshold=0.4)
        flow_est = OpticalFlowEstimator(max_corners=200, quality_level=0.01, min_distance=7.0)
        imu = SimulatedIMUReader()
        imu.start()
        compensator = EgoMotionCompensator(fallback_flow_quality_threshold=0.30)

        frame_src.start()
        frame_count = 0
        consecutive_misses = 0
        last_risk_state = None
        last_cmd = None

        CAMERA_WAIT_WARN_THRESHOLD = 20     # ~10s at 0.5s timeout for VDO.Ninja/Chromium cold start

        try:
            while True:
                f_obj = frame_src.get_frame(timeout=0.5)
                ts = time.monotonic()

                if f_obj is None:
                    consecutive_misses += 1
                    # Broadcast honest waiting/degraded state — NEVER fabricated data
                    msg = build_waiting_telemetry_message(
                        session_id=session_id,
                        ts=ts,
                        frame_count=frame_count,
                        consecutive_misses=consecutive_misses,
                        threshold=CAMERA_WAIT_WARN_THRESHOLD,
                    )
                    telemetry_server.broadcast(msg)
                    # NOTE: arduino.send() is deliberately NOT called here to prevent haptic spamming
                    time.sleep(args.delay)
                    continue

                consecutive_misses = 0

                # Live Frame Processed
                img = f_obj.image
                h, w = img.shape[:2]
                frame_count += 1

                tracks = tracker.track(f_obj)
                flow = flow_est.update(img, f_obj.frame_id, ts)
                imu_s = imu.get_latest()
                motion_st = compensator.compensate(flow, imu_s, ts, w, h)
                geometries = compute_geometry_batch(tracks, motion_st, w, h)

                predictions = predictor.predict_batch(geometries, tracks, f_obj.frame_id)
                risk_state = engine.update(predictions, fallback_active=motion_st.fallback_active, timestamp=ts)
                cmd = policy.select(risk_state, timestamp=ts)

                # Only dispatch to arduino on state/pattern transition
                if last_cmd is None or cmd.pattern_id != last_cmd.pattern_id or cmd.direction != last_cmd.direction:
                    arduino.send(cmd)
                    last_cmd = cmd
                last_risk_state = risk_state

                pred_map = {p.track_id: p for p in predictions}
                telemetry_server.broadcast(TelemetryMessage(
                    session_id=session_id,
                    ts=ts,
                    frame_id=f_obj.frame_id,
                    tracks=build_track_telemetry(tracks, pred_map),
                    risk_state=asdict(risk_state),
                    haptic=asdict(cmd),
                    pipeline_health=build_pipeline_health(
                        frame_source=frame_src,
                        imu_reader=imu,
                        arduino=arduino,
                    ),
                    frame_width=w,
                    frame_height=h,
                ))

                # Push current frame to MJPEG endpoint so dashboard can show /api/video
                telemetry_server.push_frame(img)

                if frame_count % 30 == 0:
                    print(
                        f"[Frame {f_obj.frame_id:4d}] State: {risk_state.state:<8} | "
                        f"Risk: {risk_state.global_risk:.2f} | "
                        f"Tracks: {len(tracks)} | "
                        f"Haptic: {cmd.pattern_id}"
                    )

                time.sleep(args.delay)

        except KeyboardInterrupt:
            print("\n[*] Stopping pipeline...")
        finally:
            frame_src.stop()
            imu.stop()

    telemetry_server.stop()
    arduino.stop()
    print("[*] SpatialVector-HMI shutdown complete.")


if __name__ == "__main__":
    main()
