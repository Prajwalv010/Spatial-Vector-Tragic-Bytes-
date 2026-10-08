"""SpatialVector-HMI — Master Pipeline Demo Entrypoint (M01 → M12).

Integrates the full predictive collision intelligence system:
  Perception:  M01 FrameSource -> M02 Detector -> M03 Tracker
  Motion:      M04 Optical Flow -> M05 IMU -> M06 Ego-Motion & Geometry
  Decision:    M07 Collision Prediction -> M08 Risk Engine -> M09 Corridor Policy
  Output/HMI:  M10 Arduino Haptic Controller -> M11 Phone Telemetry -> M12 Session Logger

Satisfies Gates E, F, and G:
  Gate E: Full live pipeline with continuous M12 recording.
  Gate F: M11 phone dashboard mirrors state without becoming a safety dependency.
  Gate G: Full demo scenarios run cleanly from cold start.
"""

from __future__ import annotations

import argparse
import logging
import math
from dataclasses import asdict
from pathlib import Path
import sys
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import yaml

# Add repository root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from spatialvector.decision.corridor_policy import CorridorPolicy
from spatialvector.decision.prediction import CollisionPredictor
from spatialvector.decision.risk_engine import RiskEngine
from spatialvector.decision.schemas import HapticCommand, Prediction, RiskState
from spatialvector.hmi.arduino_interface import ArduinoInterface, SimulatedArduinoInterface
from spatialvector.hmi.logger import SessionLogger
from spatialvector.hmi.schemas import TelemetryMessage, build_track_telemetry
from spatialvector.hmi.telemetry_server import TelemetryServer
from spatialvector.motion.schemas import ObjectGeometry
from spatialvector.perception.schemas import Frame, Track
from spatialvector.perception.source_resolver import resolve_camera_source

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("SpatialVector-Full")


# ---------------------------------------------------------------------------
# Argument Parser
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="SpatialVector-HMI Master Pipeline Demo (M01 → M12)"
    )
    p.add_argument("--source", default=None,
                   help="Camera index, video path, or VDO.Ninja URL (reads camera_source.txt by default)")
    p.add_argument("--prompt", action="store_true",
                   help="Prompt interactively for camera / VDO.Ninja link and save it")
    p.add_argument("--synthetic", action="store_true",
                   help="Run Gate C/D/E synthetic scenario (no camera, no YOLO required)")
    p.add_argument("--no-view", action="store_true",
                   help="Headless mode (no OpenCV GUI window)")
    p.add_argument("--fps", type=float, default=15.0,
                   help="Target capture FPS (default: 15.0 for smooth, stable processing)")
    p.add_argument("--delay", type=float, default=0.03,
                   help="Pacing sleep delay between frames in seconds (default: 0.03s)")

    # M05 IMU options
    p.add_argument("--sim-imu", action="store_true", help="Use simulated IMU (default if no port)")
    p.add_argument("--imu-port", default=None, help="Serial port for IMU (e.g. COM3 or /dev/ttyUSB0)")

    # M10 Arduino options
    p.add_argument("--sim-arduino", action="store_true", help="Use simulated Arduino haptics")
    p.add_argument("--arduino-port", default=None, help="Serial port for Arduino (e.g. COM4 or /dev/ttyACM0)")

    # M11 Telemetry options
    p.add_argument("--no-telemetry", action="store_true",
                   help="Disable M11 telemetry server completely (verifies removability)")
    p.add_argument("--telemetry-port", type=int, default=8080,
                   help="Port for phone telemetry web dashboard (default: 8080)")

    # M12 Logger options
    p.add_argument("--no-log", action="store_true",
                   help="Disable M12 session recording")
    p.add_argument("--session-dir", default="sessions",
                   help="Base directory for JSONL session recordings (default: sessions/)")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Synthetic Scenario Generator (Gate C/D/E)
# ---------------------------------------------------------------------------

class SyntheticScenario:
    """Generates synthetic ObjectGeometry and Track objects for testing without camera/YOLO."""
    FRAME_W = 640
    FRAME_H = 480

    def get_tracks_and_geometries(self, fid: int) -> Tuple[List[Track], List[ObjectGeometry]]:
        phase = self._get_phase(fid)
        if phase == "approach":
            return self._approaching_object(fid)
        elif phase == "recede":
            return self._receding_object(fid, offset=30)
        elif phase == "parallel":
            return self._parallel_object(fid, offset=50)
        elif phase == "crossing":
            return self._crossing_objects(fid, offset=70)
        else:
            return [], []

    def _get_phase(self, fid: int) -> str:
        if fid <= 30:
            return "approach"
        elif fid <= 50:
            return "recede"
        elif fid <= 70:
            return "parallel"
        elif fid <= 90:
            return "crossing"
        else:
            return "clear"

    def _make_track(self, tid: int, cx: float, cy: float, vx: float, vy: float) -> Track:
        return Track(
            track_id=tid,
            class_name="person",
            bbox_history=[(cx - 25, cy - 50, cx + 25, cy + 50)],
            center_history=[(cx, cy)],
            estimated_image_velocity=(vx, vy),
            track_age=15,
            track_confidence=0.88,
            last_seen_frame_id=0,
        )

    def _make_geom(self, tid: int, bearing: float, vx_n: float, vy_n: float, foe_cont: bool, conf: float, fid: int) -> ObjectGeometry:
        mag = math.sqrt(vx_n**2 + vy_n**2)
        mv = (vx_n / mag, vy_n / mag) if mag > 1e-9 else (0.0, 0.0)
        return ObjectGeometry(
            track_id=tid, frame_id=fid, bearing=bearing,
            relative_image_velocity=(vx_n, vy_n), motion_vector=mv,
            foe_containment=foe_cont, geometry_confidence=conf,
        )

    def _approaching_object(self, fid: int):
        vy = 0.12 + (fid / 30.0) * 0.15
        g = self._make_geom(1, bearing=0.0, vx_n=0.0, vy_n=vy, foe_cont=True, conf=0.92, fid=fid)
        t = self._make_track(1, 320, 200 + fid * 6, 0, vy * self.FRAME_H)
        return [t], [g]

    def _receding_object(self, fid: int, offset: int):
        vy = -(0.10 + ((fid - offset) / 20.0) * 0.05)
        g = self._make_geom(1, bearing=0.0, vx_n=0.0, vy_n=vy, foe_cont=False, conf=0.85, fid=fid)
        t = self._make_track(1, 320, 380 - (fid - offset) * 4, 0, vy * self.FRAME_H)
        return [t], [g]

    def _parallel_object(self, fid: int, offset: int):
        g = self._make_geom(2, bearing=-0.45, vx_n=0.0, vy_n=0.0, foe_cont=False, conf=0.85, fid=fid)
        t = self._make_track(2, 100, 240, 0, 0)
        return [t], [g]

    def _crossing_objects(self, fid: int, offset: int):
        t_prog = (fid - offset) / 20.0
        # Object crossing from right to left toward center
        cx = 500 - t_prog * 200
        cy = 180 + t_prog * 120
        g = self._make_geom(3, bearing=0.15, vx_n=-0.15, vy_n=0.12, foe_cont=True, conf=0.90, fid=fid)
        t = self._make_track(3, cx, cy, -0.15 * self.FRAME_W, 0.12 * self.FRAME_H)
        return [t], [g]


# ---------------------------------------------------------------------------
# GUI Overlay Renderer
# ---------------------------------------------------------------------------

def _draw_overlay(
    img: np.ndarray,
    tracks: List[Track],
    predictions: List[Prediction],
    risk_state: RiskState,
    cmd: HapticCommand,
    session_id: str,
    fps: float,
) -> np.ndarray:
    canvas = img.copy()
    h, w = canvas.shape[:2]

    # Top Status Bar
    state_colors = {
        "SAFE": (0, 180, 0), "CAUTION": (0, 190, 220),
        "WARNING": (0, 130, 255), "CRITICAL": (0, 0, 220), "DEGRADED": (180, 0, 180),
    }
    bar_color = state_colors.get(risk_state.state, (100, 100, 100))
    cv2.rectangle(canvas, (0, 0), (w, 36), bar_color, -1)

    cr = risk_state.corridor_risks
    status_text = (
        f"STATE: {risk_state.state} | Global Risk: {risk_state.global_risk:.3f} | "
        f"Corridors: L={cr.get('left',0):.2f} C={cr.get('center',0):.2f} R={cr.get('right',0):.2f} | "
        f"{fps:.1f} FPS"
    )
    cv2.putText(canvas, status_text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)

    # Tracks and Bounding Boxes
    pred_map = {p.track_id: p for p in predictions}
    for tr in tracks:
        if not tr.bbox_history:
            continue
        bx = tr.bbox_history[-1]
        x1, y1, x2, y2 = int(bx[0]), int(bx[1]), int(bx[2]), int(bx[3])
        p = pred_map.get(tr.track_id)

        if p and p.intersection_flag:
            color = (0, 0, 240)  # Red - collision course
        elif p and p.cpa_normalized < 0.25:
            color = (0, 165, 255)  # Orange - caution
        else:
            color = (0, 210, 0)  # Green - safe

        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        ttc_str = f"TTC:{p.ttc_s:.1f}s" if (p and p.ttc_s is not None) else "TTC:None"
        lbl = f"#{tr.track_id} {tr.class_name} | {ttc_str}"
        cv2.putText(canvas, lbl, (x1, max(20, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    # Bottom Haptic Banner
    cv2.rectangle(canvas, (0, h - 30), (w, h), (25, 25, 25), -1)
    haptic_str = (
        f"M10 HAPTIC: Direction={cmd.direction} | Urgency={cmd.urgency}/5 | "
        f"Pattern={cmd.pattern_id} ({cmd.duration_ms}ms) | sess:{session_id[:8]}"
    )
    cv2.putText(canvas, haptic_str, (10, h - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 255), 1, cv2.LINE_AA)

    return canvas


# ---------------------------------------------------------------------------
# Pipeline Builders
# ---------------------------------------------------------------------------

def build_decision_chain(config: dict):
    m07_cfg = config.get("prediction", {})
    m08_cfg = config.get("risk_engine", {})
    m09_cfg = config.get("corridor_policy", {})

    predictor = CollisionPredictor(
        horizon_s=m07_cfg.get("horizon_s", 5.0),
        contact_threshold_normalized=m07_cfg.get("contact_threshold_normalized", 0.05),
        corridor_width_normalized=m07_cfg.get("corridor_width_normalized", 0.12),
    )
    engine = RiskEngine(
        weight_ttc=m08_cfg.get("weight_ttc", 0.5),
        weight_miss_distance=m08_cfg.get("weight_miss_distance", 0.3),
        weight_intersection_confidence=m08_cfg.get("weight_intersection_confidence", 0.2),
        state_thresholds=m08_cfg.get("state_thresholds"),
        hysteresis_frames_up=m08_cfg.get("hysteresis_frames_up", 3),
        hysteresis_frames_down=m08_cfg.get("hysteresis_frames_down", 5),
        degraded_confidence_threshold=m08_cfg.get("degraded_confidence_threshold", 0.25),
    )
    policy = CorridorPolicy(
        all_unsafe_risk_threshold=m09_cfg.get("all_unsafe_risk_threshold", 0.70),
        trend_window_frames=m09_cfg.get("trend_window_frames", 5),
    )
    return predictor, engine, policy


# ---------------------------------------------------------------------------
# Main Runner
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # Load configuration
    cfg_path = ROOT / "spatialvector" / "config" / "default.yaml"
    config = {}
    if cfg_path.exists():
        with open(cfg_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}

    # Initialize M12 Session Logger
    logger_instance = None
    session_id = f"demo_{int(time.time())}"
    if not args.no_log:
        logger_instance = SessionLogger(
            session_id=session_id,
            base_dir=args.session_dir,
            config_snapshot=config,
        )

    # Initialize M11 Telemetry Server
    telemetry_server = None
    if not args.no_telemetry:
        tel_cfg = config.get("telemetry", {})
        host = tel_cfg.get("host", "0.0.0.0")
        port = args.telemetry_port or tel_cfg.get("port", 8080)
        telemetry_server = TelemetryServer(host=host, port=port)
        telemetry_server.start()
        print(f"[M11] Phone Dashboard live at: http://localhost:{port}/")

    # Initialize M10 Arduino Interface
    arduino = None
    ard_cfg = config.get("arduino", {})
    if args.sim_arduino or args.synthetic or (args.arduino_port is None and not ard_cfg.get("serial_port")):
        arduino = SimulatedArduinoInterface(
            queue_size=ard_cfg.get("command_queue_size", 10),
            watchdog_timeout_ms=ard_cfg.get("watchdog_timeout_ms", 500),
        )
    else:
        port = args.arduino_port or ard_cfg.get("serial_port", "COM3")
        arduino = ArduinoInterface(
            port=port,
            baud_rate=ard_cfg.get("baud_rate", 115200),
            queue_size=ard_cfg.get("command_queue_size", 10),
            watchdog_timeout_ms=ard_cfg.get("watchdog_timeout_ms", 500),
        )
    arduino.start()


    # Build decision chain (M07, M08, M09)
    predictor, engine, policy = build_decision_chain(config)

    print("=" * 70)
    print("SpatialVector-HMI -- Full Pipeline Live Demo (M01 -> M12)")
    print(f"Session ID: {session_id}")
    print(f"Mode: {'SYNTHETIC (Gate C/D/E)' if args.synthetic else 'LIVE CAMERA'}")
    print("=" * 70)


    # Run Loop
    t_start = time.monotonic()
    frame_count = 0

    try:
        if args.synthetic:
            # Synthetic Mode: runs through multi-object scenarios deterministically
            scenario = SyntheticScenario()
            total_synthetic_frames = 100

            for fid in range(total_synthetic_frames):
                ts = time.monotonic()
                tracks, geometries = scenario.get_tracks_and_geometries(fid)

                predictions = predictor.predict_batch(geometries, tracks, fid)
                risk_state = engine.update(predictions, fallback_active=False, timestamp=ts)
                cmd = policy.select(risk_state, timestamp=ts)

                # Send to M10 Arduino
                arduino.send(cmd)

                # Broadcast to M11 Telemetry
                if telemetry_server:
                    pred_map = {p.track_id: p for p in predictions}
                    telemetry_server.broadcast(TelemetryMessage(
                        session_id=session_id,
                        ts=ts,
                        frame_id=fid,
                        tracks=build_track_telemetry(tracks, pred_map),
                        risk_state=asdict(risk_state),
                        haptic=asdict(cmd),
                        pipeline_health={"camera": "OK", "imu": "OK", "arduino": "OK"},
                        frame_width=640,
                        frame_height=480,
                    ))

                # Record in M12 Logger
                if logger_instance:
                    logger_instance.log_event("prediction_batch", predictions, frame_id=fid, ts=ts)
                    logger_instance.log_event("risk", risk_state, frame_id=fid, ts=ts)
                    logger_instance.log_event("haptic", cmd, frame_id=fid, ts=ts)


                frame_count += 1

                # GUI Display
                if not args.no_view:
                    blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    canvas = _draw_overlay(blank_frame, tracks, predictions, risk_state, cmd, session_id, fps=args.fps)
                    cv2.imshow("SpatialVector-HMI Master Pipeline", canvas)
                    wait_ms = max(1, int(args.delay * 1000))
                    if cv2.waitKey(wait_ms) & 0xFF == ord('q'):
                        break
                elif args.delay > 0:
                    time.sleep(args.delay)

        else:
            # Live Camera Mode (M01-M06)
            from spatialvector.motion.ego_motion import EgoMotionCompensator
            from spatialvector.motion.geometry import compute_geometry_batch
            from spatialvector.motion.imu_reader import IMUReader, SimulatedIMUReader
            from spatialvector.motion.optical_flow import OpticalFlowEstimator
            from spatialvector.perception.detector import ObjectDetector
            from spatialvector.perception.frame_source import FrameSource
            from spatialvector.perception.tracker import MultiObjectTracker

            # Resolve camera source
            src, origin = resolve_camera_source(cli_source=args.source, interactive=args.prompt)
            print(f"[Pipeline] Video source: {src} (resolved from {origin})")

            is_net = isinstance(src, str) and any(src.lower().startswith(p) for p in ("http://", "https://", "rtsp://"))
            frame_source = FrameSource(source=src, target_fps=args.fps, queue_size=5, loop_video=(not is_net and isinstance(src, str)))

            det_cfg = config.get("detector", {})
            detector = ObjectDetector(
                model_path=det_cfg.get("model_path", "yolov8n.pt"),
                confidence_threshold=det_cfg.get("confidence_threshold", 0.4),
                class_filter=det_cfg.get("class_filter", ["person", "bicycle", "car", "chair"]),
            )
            tracker = MultiObjectTracker(backend="bytetrack", history_length=10, confidence_threshold=0.4)
            flow_est = OpticalFlowEstimator(max_corners=200, quality_level=0.01, min_distance=7.0)

            imu_reader = None
            if args.sim_imu or (args.imu_port is None and not config.get("imu", {}).get("serial_port")):
                imu_reader = SimulatedIMUReader()
            else:
                imu_reader = IMUReader(port=args.imu_port or config.get("imu", {}).get("serial_port", "COM3"))
            imu_reader.start()

            compensator = EgoMotionCompensator(
                fallback_flow_quality_threshold=config.get("ego_motion", {}).get("fallback_flow_quality_threshold", 0.30),
            )

            frame_source.start()
            consecutive_timeouts = 0

            while True:
                f_obj = frame_source.get_frame(timeout=1.0)
                if f_obj is None:
                    consecutive_timeouts += 1
                    if consecutive_timeouts > 15:
                        print("[Pipeline] No frames received — exiting.")
                        break
                    continue
                consecutive_timeouts = 0

                img = f_obj.image
                h, w = img.shape[:2]
                ts = f_obj.t_capture

                # M02 + M03
                detections = detector.detect(img, f_obj.frame_id, ts)
                tracks = tracker.update(detections, f_obj.frame_id, ts)

                # M04 + M05 + M06
                flow = flow_est.update(img, f_obj.frame_id, ts)
                imu_s = imu_reader.get_latest() if imu_reader else None
                motion_st = compensator.compensate(flow, imu_s, ts, w, h)
                geometries = compute_geometry_batch(tracks, motion_st, w, h)

                # M07 + M08 + M09
                predictions = predictor.predict_batch(geometries, tracks, f_obj.frame_id)
                risk_state = engine.update(predictions, fallback_active=motion_st.fallback_active, timestamp=ts)
                cmd = policy.select(risk_state, timestamp=ts)

                # M10 Haptics
                arduino.send(cmd)

                # M11 Telemetry
                if telemetry_server:
                    pred_map = {p.track_id: p for p in predictions}
                    telemetry_server.broadcast(TelemetryMessage(
                        session_id=session_id,
                        ts=ts,
                        frame_id=f_obj.frame_id,
                        tracks=build_track_telemetry(tracks, pred_map),
                        risk_state=asdict(risk_state),
                        haptic=asdict(cmd),
                        pipeline_health={
                            "camera": "OK" if frame_source.is_running() else "DISCONNECTED",
                            "imu": "OK" if (imu_s and imu_s.status == "OK") else "DISCONNECTED",
                            "arduino": "OK" if arduino.get_status().connected else "DISCONNECTED",
                        },
                        frame_width=w,
                        frame_height=h,
                    ))

                # M12 Logger
                if logger_instance:
                    logger_instance.log_event("detections", detections, frame_id=f_obj.frame_id, ts=ts)
                    logger_instance.log_event("predictions", predictions, frame_id=f_obj.frame_id, ts=ts)
                    logger_instance.log_event("risk", risk_state, frame_id=f_obj.frame_id, ts=ts)
                    logger_instance.log_event("haptic", cmd, frame_id=f_obj.frame_id, ts=ts)


                frame_count += 1

                # GUI Display
                if not args.no_view:
                    fps_calc = frame_count / max(0.1, time.monotonic() - t_start)
                    canvas = _draw_overlay(img, tracks, predictions, risk_state, cmd, session_id, fps=fps_calc)
                    cv2.imshow("SpatialVector-HMI Master Pipeline", canvas)
                    wait_ms = max(1, int(args.delay * 1000))
                    if cv2.waitKey(wait_ms) & 0xFF == ord('q'):
                        break
                elif args.delay > 0:
                    time.sleep(args.delay)

    finally:
        elapsed = max(0.01, time.monotonic() - t_start)
        print("\n" + "=" * 70)
        print(f"[Master Pipeline] Completed {frame_count} frames in {elapsed:.1f}s ({frame_count/elapsed:.1f} FPS)")
        if arduino:
            arduino.stop()
        if telemetry_server:
            telemetry_server.stop()
        if logger_instance:
            logger_instance.close()
            print(f"[M12 Logger] Recorded session saved to: {logger_instance.session_file}")
        if not args.no_view:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
        print("=" * 70)


if __name__ == "__main__":
    main()
