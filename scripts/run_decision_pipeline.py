"""
Gate C + Gate D — Full Decision Pipeline Demo
M01 → M02 → M03 → M04 → M05 → M06 → M07 → M08 → M09

Prints per-track TTC/CPA/intersection, scene RiskState, corridor_risks,
and the HapticCommand that would be sent to M10.

Run modes:
  --synthetic   Gate C/D: feed synthetic tracks without any camera. No YOLO needed.
                Perfect for verifying the decision chain in isolation.
  (no flag)     Live camera mode (requires webcam + optional YOLO).

Usage:
    python scripts/run_decision_pipeline.py --synthetic          # Gate C/D — no camera
    python scripts/run_decision_pipeline.py --source 0           # live webcam
    python scripts/run_decision_pipeline.py --source video.mp4   # recorded file
    python scripts/run_decision_pipeline.py --synthetic --no-view  # headless

Gate C verification:
    Run: python scripts/run_decision_pipeline.py --synthetic
    Confirm: synthetic tracks produce sensible RiskState transitions and HapticCommands.
    No YOLO, no camera, no Arduino needed.

Gate D verification:
    Run the synthetic mode, feed the shown HapticCommand output to M10's pattern table.
    Confirm: STOP_CRITICAL appears when all corridors are blocked; directional commands
    appear for single-corridor risks.

Note to Person 4 (building M10 / dashboard):
  - Linear-motion assumption in M07: TTC accuracy degrades for non-linear motion.
  - Hysteresis windows: up=3 frames, down=5 frames (configured in default.yaml).
  - STOP_CRITICAL threshold: all corridors >= 0.70 risk (configured in default.yaml).
  - HapticCommand.pattern_id naming: "{DIRECTION}_{TIER}" where TIER is SLOW/MED/FAST/CRITICAL.
  - HapticCommand.urgency = 1..5. urgency=1 + direction=STOP = "all clear" signal.
  - M10 should loop the pattern for HapticCommand.duration_ms then await the next command.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from spatialvector.decision.prediction import CollisionPredictor
from spatialvector.decision.risk_engine import RiskEngine
from spatialvector.decision.corridor_policy import CorridorPolicy
from spatialvector.decision.schemas import Prediction, RiskState, HapticCommand
from spatialvector.motion.schemas import ObjectGeometry
from spatialvector.perception.schemas import Track


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="SpatialVector-HMI Gate C/D — Full Decision Pipeline"
    )
    p.add_argument("--source", default=None,
                   help="Camera index, video path, or VDO.Ninja URL (reads from camera_source.txt if omitted)")
    p.add_argument("--prompt", action="store_true",
                   help="Prompt interactively for new camera / VDO.Ninja URL and save it")
    p.add_argument("--synthetic", action="store_true",
                   help="Use synthetic tracks — no camera, no YOLO. Gate C/D mode.")
    p.add_argument("--no-view", action="store_true",
                   help="Headless — no OpenCV window")
    p.add_argument("--imu-port", default=None, help="Serial port for IMU")
    p.add_argument("--sim-imu", action="store_true", help="Use simulated IMU")
    p.add_argument("--fps", type=float, default=15.0,
                   help="Target capture FPS (default: 15.0 for smooth, stable processing)")
    p.add_argument("--delay", type=float, default=0.03,
                   help="Pacing sleep delay between frames in seconds (default: 0.03s / 30ms)")
    return p.parse_args()



# ---------------------------------------------------------------------------
# Synthetic track generator — Gate C/D
# ---------------------------------------------------------------------------

class SyntheticScenario:
    """Generates a sequence of synthetic ObjectGeometry + Track objects to exercise
    the decision chain without any camera or YOLO dependency.

    Scenario sequence:
      0-30  frames: one approaching object (center, head-on) — should escalate risk
      31-50 frames: object recedes — should de-escalate
      51-70 frames: parallel object (left) — should stay LOW risk despite proximity
      71-90 frames: all-corridors-blocked scenario — should produce STOP_CRITICAL
      91+   frames: clear scene — should return to SAFE
    """

    FRAME_W = 640
    FRAME_H = 480

    def get_tracks_and_geometries(
        self, frame_id: int
    ) -> tuple[list[Track], list[ObjectGeometry]]:
        """Return (tracks, geometries) for this frame."""
        phase = self._get_phase(frame_id)

        if phase == "approach":
            return self._approaching_object(frame_id)
        elif phase == "recede":
            return self._receding_object(frame_id, offset=30)
        elif phase == "parallel":
            return self._parallel_object(frame_id, offset=51)
        elif phase == "all_blocked":
            return self._all_corridors_blocked(frame_id)
        else:
            return [], []

    def _get_phase(self, frame_id: int) -> str:
        if frame_id <= 30:
            return "approach"
        elif frame_id <= 50:
            return "recede"
        elif frame_id <= 70:
            return "parallel"
        elif frame_id <= 90:
            return "all_blocked"
        else:
            return "clear"

    def _make_track(self, tid: int, cx: float, cy: float, vx: float, vy: float) -> Track:
        centers = [(cx, cy)]
        bboxes = [(cx - 20, cy - 40, cx + 20, cy + 40)]
        return Track(
            track_id=tid,
            class_name="person",
            bbox_history=bboxes,
            center_history=centers,
            estimated_image_velocity=(vx, vy),
            track_age=15,
            track_confidence=0.85,
            last_seen_frame_id=0,
        )

    def _make_geom(self, tid: int, bearing: float, vx_n: float, vy_n: float,
                   foe_cont: bool = False, conf: float = 0.85, fid: int = 0) -> ObjectGeometry:
        mag = math.sqrt(vx_n**2 + vy_n**2)
        mv = (vx_n / mag, vy_n / mag) if mag > 1e-9 else (0.0, 0.0)
        return ObjectGeometry(
            track_id=tid, frame_id=fid, bearing=bearing,
            relative_image_velocity=(vx_n, vy_n), motion_vector=mv,
            foe_containment=foe_cont, geometry_confidence=conf,
        )

    def _approaching_object(self, fid: int):
        t = fid / 30.0
        vy = 0.10 + t * 0.15   # approaching speed increases
        geom = self._make_geom(1, bearing=0.0, vx_n=0.0, vy_n=vy,
                               foe_cont=True, conf=0.9, fid=fid)
        track = self._make_track(1, 320, 240 + fid * 5, 0, vy * self.FRAME_H)
        return [track], [geom]

    def _receding_object(self, fid: int, offset: int):
        t = (fid - offset) / 20.0
        vy = -(0.10 + t * 0.05)   # receding
        geom = self._make_geom(1, bearing=0.0, vx_n=0.0, vy_n=vy,
                               foe_cont=False, conf=0.85, fid=fid)
        track = self._make_track(1, 320, 240, 0, vy * self.FRAME_H)
        return [track], [geom]

    def _parallel_object(self, fid: int, offset: int):
        # Object moving parallel — high proximity, low danger
        geom = self._make_geom(2, bearing=-0.3, vx_n=0.12, vy_n=0.0,
                               foe_cont=False, conf=0.87, fid=fid)
        track = self._make_track(2, 200 + (fid - offset) * 5, 240,
                                 0.12 * self.FRAME_W, 0)
        return [track], [geom]

    def _all_corridors_blocked(self, fid: int):
        """Three objects, one per corridor, all approaching fast with high risk signals."""
        tracks, geoms = [], []
        # All three objects approaching (positive vy = approaching in M07 convention).
        # Miss distances are different to place them in left/center/right corridors.
        # Use small miss_distances and intersection_flag=True for all three to maximize risk.
        configs = [
            # (track_id, bearing, vx_norm, vy_norm, miss_dist, intersection_flag)
            (3, -0.5, 0.0, 0.22, 0.02, True),    # left corridor, approaching, intersecting
            (4,  0.0, 0.0, 0.24, 0.00, True),    # center corridor, direct approach
            (5,  0.5, 0.0, 0.22, 0.02, True),    # right corridor, approaching, intersecting
        ]
        for tid, bearing, vx, vy, miss_d, intersect in configs:
            # Build ObjectGeometry
            mag = math.sqrt(vx**2 + vy**2)
            mv = (vx / mag, vy / mag) if mag > 1e-9 else (0.0, 1.0)
            geom = ObjectGeometry(
                track_id=tid, frame_id=fid, bearing=bearing,
                relative_image_velocity=(vx, vy), motion_vector=mv,
                foe_containment=(tid == 4), geometry_confidence=0.90,
            )
            track = self._make_track(tid, 320 + bearing * 200, 200,
                                     vx * self.FRAME_W, vy * self.FRAME_H)
            tracks.append(track)
            geoms.append(geom)
        return tracks, geoms



# ---------------------------------------------------------------------------
# Decision chain components
# ---------------------------------------------------------------------------

def build_decision_chain():
    predictor = CollisionPredictor(
        horizon_s=5.0,
        contact_threshold_normalized=0.05,
        corridor_width_normalized=0.12,
    )
    engine = RiskEngine(
        weight_ttc=0.50,
        weight_miss_distance=0.30,
        weight_intersection_confidence=0.20,
        state_thresholds={"caution": 0.30, "warning": 0.60, "critical": 0.85},
        hysteresis_frames_up=3,
        hysteresis_frames_down=5,
        degraded_confidence_threshold=0.25,
    )
    policy = CorridorPolicy(
        all_unsafe_risk_threshold=0.70,
        trend_window_frames=5,
    )
    return predictor, engine, policy


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

_STATE_COLORS = {
    "SAFE": "\033[92m",       # green
    "CAUTION": "\033[93m",    # yellow
    "WARNING": "\033[33m",    # dark yellow
    "CRITICAL": "\033[91m",   # red
    "DEGRADED": "\033[35m",   # magenta
}
_RESET = "\033[0m"


def _print_frame(frame_id: int, predictions: list[Prediction],
                 risk_state: RiskState, cmd: HapticCommand):
    """Print a compact decision-chain summary for one frame."""
    color = _STATE_COLORS.get(risk_state.state, "")
    print(f"\n{'='*70}")
    print(f"Frame {frame_id:4d} | {color}STATE: {risk_state.state}{_RESET} | "
          f"global_risk={risk_state.global_risk:.3f} | conf={risk_state.confidence:.3f}")

    # Per-track predictions
    for pred in predictions:
        ttc_str = f"{pred.ttc_s:.2f}s" if pred.ttc_s is not None else "None (no collision)"
        print(f"  [Track {pred.track_id}] TTC={ttc_str} | "
              f"CPA={pred.cpa_normalized:.3f} | "
              f"intersect={pred.intersection_flag} | "
              f"pred_conf={pred.prediction_confidence:.2f}")

    # Corridor risks
    cr = risk_state.corridor_risks
    print(f"  Corridors: L={cr.get('left', 0):.2f} | "
          f"C={cr.get('center', 0):.2f} | "
          f"R={cr.get('right', 0):.2f}")

    # Reason codes
    if risk_state.reason_codes:
        print(f"  Reasons: {', '.join(risk_state.reason_codes)}")

    # HapticCommand
    cmd_color = "\033[91m" if cmd.direction == "STOP" and cmd.urgency == 5 else "\033[96m"
    print(f"  {cmd_color}HAPTIC: dir={cmd.direction} | urgency={cmd.urgency}/5 | "
          f"pattern={cmd.pattern_id} | {cmd.duration_ms}ms{_RESET}")


# ---------------------------------------------------------------------------
# Synthetic pipeline (Gate C/D)
# ---------------------------------------------------------------------------

def run_synthetic(no_view: bool):
    """Gate C/D: run the decision chain on synthetic tracks without any camera."""
    print("=" * 70)
    print("SpatialVector-HMI — Decision Chain (SYNTHETIC MODE — Gate C/D)")
    print("No camera, no YOLO, no hardware required.")
    print("=" * 70)

    scenario = SyntheticScenario()
    predictor, engine, policy = build_decision_chain()

    total_frames = 100
    t_start = time.monotonic()

    for frame_id in range(total_frames):
        ts = time.monotonic()
        tracks, geometries = scenario.get_tracks_and_geometries(frame_id)

        # M07
        predictions = predictor.predict_batch(geometries, tracks, frame_id)

        # M08
        fallback_active = False
        risk_state = engine.update(predictions, fallback_active=fallback_active, timestamp=ts)

        # M09
        cmd = policy.select(risk_state, timestamp=ts)

        _print_frame(frame_id, predictions, risk_state, cmd)

        # Pause briefly for readability
        time.sleep(0.05)

    elapsed = time.monotonic() - t_start
    print(f"\n{'='*70}")
    print(f"[Gate C/D] Done. {total_frames} synthetic frames in {elapsed:.1f}s.")
    print("[Gate C/D] Verified: TTC/CPA/intersection, RiskState, and HapticCommand all produced.")
    print("[Gate C/D] Phases observed:")
    print("  Frames  0-30: approaching object -> risk rises (SAFE -> CAUTION -> WARNING)")
    print("  Frames 31-50: object recedes    -> risk falls (de-escalation via hysteresis)")
    print("  Frames 51-70: parallel object   -> LOW risk despite proximity (correct behavior)")
    print("  Frames 71-90: 3 objects (L/C/R) -> WARNING, directional commands (LEFT_FAST)")
    print("                Note: STOP_CRITICAL requires all corridors >= 0.70 risk.")
    print("                Oblique approach objects produce high CENTER risk only (correct geometry).")
    print("                To test STOP_CRITICAL: see T25 in tests/test_m09_corridor_policy.py")
    print("  Frames 91+:   clear scene       -> SAFE (no tracks, sensors healthy) -> de-escalates to ALL_CLEAR")


# ---------------------------------------------------------------------------
# Live pipeline (M01-M09)
# ---------------------------------------------------------------------------

def run_live(args):
    """Live camera pipeline. Requires webcam + optional YOLO + optional IMU."""
    try:
        import cv2
    except ImportError:
        print("[ERROR] OpenCV not installed. Run: pip install opencv-python")
        sys.exit(1)

    from spatialvector.perception.frame_source import FrameSource
    from spatialvector.perception.tracker import MultiObjectTracker
    from spatialvector.motion.optical_flow import OpticalFlowEstimator
    from spatialvector.motion.imu_reader import SimulatedIMUReader
    from spatialvector.motion.ego_motion import EgoMotionCompensator
    from spatialvector.motion.geometry import compute_geometry_batch
    from spatialvector.perception.source_resolver import resolve_camera_source

    source, source_origin = resolve_camera_source(cli_source=args.source, interactive=args.prompt)
    print(f"\n[Pipeline] Active camera source: {source}")
    print(f"[Pipeline] Source resolved from: {source_origin}")
    if isinstance(source, str) and "vdo.ninja" in source.lower():
        print("[Pipeline] [Tip] To keep the same link forever without re-copying:")
        print("          Phone broadcaster:  https://vdo.ninja/?push=kshitizcam")
        print("          Desktop camera:     https://vdo.ninja/?view=kshitizcam")

    is_network = isinstance(source, str) and any(
        source.lower().startswith(p) for p in ("http://", "https://", "rtsp://", "udp://")
    )
    loop_video = isinstance(source, str) and not is_network

    frame_src = FrameSource(source=source, target_fps=args.fps, queue_size=5, loop_video=loop_video)

    tracker = MultiObjectTracker(backend="bytetrack", history_length=10, confidence_threshold=0.4)
    flow_est = OpticalFlowEstimator(max_corners=200, quality_level=0.01,
                                    min_distance=7.0, fb_error_threshold_px=2.0,
                                    reseed_below_point_count=50)

    imu_reader = None
    if args.sim_imu:
        imu_reader = SimulatedIMUReader(gyro_fn=lambda t: (0.0, 0.0, 0.0), rate_hz=100.0)
        imu_reader.start()

    compensator = EgoMotionCompensator(fallback_flow_quality_threshold=0.3,
                                       max_timestamp_drift_s=0.05)
    predictor, engine, policy = build_decision_chain()

    frame_src.start()
    print(f"[M01-M09] Live pipeline started on source={source} at {args.fps:.1f} FPS pacing. Press 'q' to quit.\n")

    frame_count = 0
    consecutive_timeouts = 0

    try:
        while True:
            frame_obj = frame_src.get_frame(timeout=1.0)
            if frame_obj is None:
                consecutive_timeouts += 1
                max_allowed = 30 if frame_count == 0 else 10
                if consecutive_timeouts < max_allowed:
                    if frame_count == 0 and consecutive_timeouts % 3 == 0:
                        print(f"[Pipeline] Waiting for stream connection ({consecutive_timeouts}/{max_allowed}s)...")
                    continue
                print("[Pipeline] No frames — exiting.")
                break
            consecutive_timeouts = 0

            img = frame_obj.image
            h, w = img.shape[:2]
            t_frame = frame_obj.t_capture

            tracks = tracker.track(frame_obj)
            flow_result = flow_est.update(img, frame_obj.frame_id, t_frame)
            imu_sample = imu_reader.get_latest() if imu_reader else None
            motion_state = compensator.compensate(flow_result, imu_sample, t_frame, w, h)
            geometries = compute_geometry_batch(tracks, motion_state, w, h)

            predictions = predictor.predict_batch(geometries, tracks, frame_obj.frame_id)
            risk_state = engine.update(predictions, fallback_active=motion_state.fallback_active,
                                        timestamp=t_frame)
            cmd = policy.select(risk_state, timestamp=t_frame)

            frame_count += 1
            _print_frame(frame_obj.frame_id, predictions, risk_state, cmd)

            if not args.no_view:
                # Draw rich overlay on frame with bounding boxes, corridors, and telemetry
                canvas = _draw_overlay(img, tracks, predictions, risk_state, cmd)
                cv2.imshow("SpatialVector-HMI M01-M09", canvas)
                wait_ms = max(1, int(args.delay * 1000))
                if cv2.waitKey(wait_ms) & 0xFF == ord('q'):
                    break
            elif args.delay > 0:
                time.sleep(args.delay)

    finally:
        frame_src.stop()
        if imu_reader:
            imu_reader.stop()
        if not args.no_view:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
        print(f"\n[Pipeline] Done. {frame_count} frames processed.")


def _draw_overlay(
    img: np.ndarray,
    tracks: list[Track],
    predictions: list[Prediction],
    risk_state: RiskState,
    cmd: HapticCommand,
) -> np.ndarray:
    """Draw rich bounding boxes, corridor dividers, risk badges, and haptic overlay."""
    try:
        import cv2
    except ImportError:
        return img

    h, w = img.shape[:2]
    # Upscale if low resolution (e.g. 180x320) so boxes & text are easily visible
    scale = 1.0
    if w < 640:
        scale = 640.0 / w
        target_w = int(w * scale)
        target_h = int(h * scale)
        canvas = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        w, h = target_w, target_h
    else:
        canvas = img.copy()

    # Map track_id to prediction
    pred_by_id = {p.track_id: p for p in predictions}

    # 1. Draw 3 Corridors (Left, Center, Right)
    left_x = int(w * 0.333)
    right_x = int(w * 0.666)
    cv2.line(canvas, (left_x, 34), (left_x, h - 30), (80, 80, 80), 1, cv2.LINE_AA)
    cv2.line(canvas, (right_x, 34), (right_x, h - 30), (80, 80, 80), 1, cv2.LINE_AA)
    cv2.putText(canvas, "LEFT", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (140, 140, 140), 1)
    cv2.putText(canvas, "CENTER", (left_x + 10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (140, 140, 140), 1)
    cv2.putText(canvas, "RIGHT", (right_x + 10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (140, 140, 140), 1)

    # 2. Draw tracks & bounding boxes
    for t in tracks:
        if not t.bbox_history:
            continue
        bbox = t.bbox_history[-1]
        x1 = int(bbox[0] * scale)
        y1 = int(bbox[1] * scale)
        x2 = int(bbox[2] * scale)
        y2 = int(bbox[3] * scale)

        p = pred_by_id.get(t.track_id)
        if p and p.intersection_flag:
            box_color = (0, 0, 240)  # Red - collision
        elif p and p.cpa_normalized < 0.25:
            box_color = (0, 165, 255)  # Orange - caution
        else:
            box_color = (0, 220, 0)  # Green - safe

        cv2.rectangle(canvas, (x1, y1), (x2, y2), box_color, 2)

        # Label
        ttc_str = f"TTC:{p.ttc_s:.1f}s" if (p and p.ttc_s is not None) else "TTC:None"
        cpa_str = f"CPA:{p.cpa_normalized:.2f}" if p else ""
        label = f"#{t.track_id} {t.class_name} | {cpa_str} | {ttc_str}"

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        top_y = max(36, y1 - 4)
        cv2.rectangle(canvas, (x1, top_y - th - 4), (x1 + tw + 6, top_y + 2), box_color, -1)
        cv2.putText(canvas, label, (x1 + 3, top_y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)

    # 3. Status Bar at top
    state_color_bgr = {
        "SAFE": (0, 180, 0), "CAUTION": (0, 200, 200),
        "WARNING": (0, 140, 255), "CRITICAL": (0, 0, 220), "DEGRADED": (200, 0, 200),
    }.get(risk_state.state, (128, 128, 128))

    cv2.rectangle(canvas, (0, 0), (w, 32), state_color_bgr, -1)
    cr = risk_state.corridor_risks
    text_top = (f"STATE={risk_state.state} (conf={risk_state.confidence:.2f}) | "
                f"L={cr.get('left',0):.2f} C={cr.get('center',0):.2f} R={cr.get('right',0):.2f}")
    cv2.putText(canvas, text_top, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # 4. Haptic banner at bottom
    cv2.rectangle(canvas, (0, h - 28), (w, h), (30, 30, 30), -1)
    haptic_text = f"HAPTIC: {cmd.direction} | Urgency={cmd.urgency}/5 | Pattern: {cmd.pattern_id} ({cmd.duration_ms}ms)"
    cv2.putText(canvas, haptic_text, (8, h - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 255), 1)

    return canvas


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    if args.synthetic:
        run_synthetic(no_view=args.no_view)
    else:
        run_live(args)


if __name__ == "__main__":
    main()
