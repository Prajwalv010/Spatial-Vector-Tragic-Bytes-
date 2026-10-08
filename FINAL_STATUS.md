# FINAL STATUS: SpatialVector-HMI System Audit & Operational Truth

This document defines the final status, verified scope, measured capabilities, failure modes, and unverified assumptions of the SpatialVector-HMI assistive navigation pipeline.

---

## 1. What is Measured and Where (File + Exact Command)

Every figure reported in this system is computed directly from raw reproducible outputs without fabricated or hard-coded assumptions:

| Measurement Scope | Source File | Exact Reproducibility Command |
|---|---|---|
| **Regenerate HOLDOUT CSV & SHA-256 Checksum** | `scripts/evaluate.py` | `python scripts/evaluate.py --fixtures tests/fixtures/real/holdout --out eval_holdout_results.csv --pothole-model pothole_yolov8.pt --yolo-model yolov8n.pt` |
| **Full HOLDOUT Evaluation Metrics & Provenance Check** | `eval_holdout_results.csv`, `eval_holdout_results.csv.sha256` | `python scripts/compute_metrics.py --csv eval_holdout_results.csv` |
| **Interactive Human Labelling Tool** | `scripts/label_live.py` | `python scripts/label_live.py --source tests/fixtures/real/holdout --out human_labels.csv` |
| **Live Overlay Viewer FPS (Hardware/Display)** | `live_viewer_fps_benchmark.json` | `python scripts/measure_live_viewer_fps.py --frames 120` |
| **Frame-Skip Safety Latency & Transition Delay** | `frame_skip_safety_benchmark.json` | `python scripts/benchmark_frame_skip_safety.py` |
| **Test Suite Invariants (129 unit/integration tests)** | `tests/` | `python -m pytest tests/` |

---

## 2. What is Human-Verified and How Many Images

- **Current Human-Verified Images**: **0 / 375** (HOLDOUT), **0 / 360** (TUNE).
- **Label State & Provenance**:
  - All object bounding boxes and labels currently in the static fixture datasets were pre-filled via `yolov8x` pseudo-labelling (`label_source: "yolov8x_prefill"`, `verified: false`).
  - Road surface categories (`road_manhole`, `road_patch_puddle`, `road_pothole`, `pedestrian_crossing`) originally defaulted to `BLOCKED`. Because this default was an unverified assumption, all corridor labels for these 100 images have been explicitly marked **`UNLABELLED`** until verified by a human operator.
  - Review sheets are committed for operator inspection at `tests/review_sheets/review_tune.html` and `tests/review_sheets/review_holdout.html`.
  - Built `scripts/label_live.py`, an interactive OpenCV keyboard tool allowing operators to mark corridors (`SAFE` / `UNSAFE` / `UNSURE`), pothole presence, and visible objects without automated agent labeling.
  - The committed importer `scripts/import_label_corrections.py` is ready to ingest operator CSV corrections and set `label_source: "human"`, `verified: true`.
  - In `scripts/compute_metrics.py`, metrics are explicitly split: all unlabelled images are excluded from walkable precision/recall/calibration, and human-verified metrics are reported as **`NO DATA (0 images verified)`** until human review is completed.

---

## 3. Known Failure Categories with Exact Counts

The evaluation on the fresh, frozen 375-image HOLDOUT benchmark reveals the following measured failure rates against stated safety targets:

| Evaluation Criterion | Stated Target | Measured Result (HOLDOUT) | Status | Specific Failure Breakdown |
|---|---|---|:---:|---|
| **Hazard False WALKABLE (Per-Image: Any Corridor)** | **0.00%** | **2.29%** (4 / 175 images, Wilson 95% UB: **5.73%**) | **FAIL** | 4 hazard frames emitted `WALKABLE` on at least one corridor: `blank_wall_019`, `blank_wall_023`, `stairs_up_016`, `table_edge_009`. |
| **Hazard False WALKABLE (Centre Corridor Only)** | **0.00%** | **0.57%** (1 / 175 images, Wilson 95% UB: **3.17%**) | **FAIL** | 1 hazard frame emitted `WALKABLE` directly ahead: `stairs_up_016`. |
| **Clear Corridor Recall (Centre Corridor)** | $\ge 80.0\%$ | **25.00%** (25 / 100 images) | **FAIL** | 75 truly clear images failed to reach single-frame centre corridor `WALKABLE` due to strict conservative classical surface vetoes. |
| **Clear Corridor Recall (Any Corridor)** | $\ge 80.0\%$ | **33.00%** (33 / 100 images) | **FAIL** | 67 clear images failed to reach `WALKABLE` on any corridor in single-frame evaluation. |
| **Safety Precision (Truly Clear among Predicted WALKABLE)** | Informational | **45.67%** (58 / 127 decisions) | **MEASURED** | Across all 127 corridor decisions predicted `WALKABLE` on labelled scenes, 58 were clear surfaces, 69 were on unverified road/hazard categories. |
| **Expected Calibration Error (ECE on WALKABLE)** | $\le 10.0\%$ & beats const baseline | **36.22%** (AUC=0.551, Const 0.82 ECE=36.33%) | **NOT MEANINGFUL** | Fitted Logistic Regression produces scores in [0.73, 0.84] with near-random discrimination (AUC 0.551); a constant 0.82 baseline achieves essentially identical ECE. Not used as a safety gate. |
| **Pothole Recall on Positives** | $\ge 60.0\%$ | **28.00%** (7 / 25 positive images) | **FAIL** | Pothole detector detected 7 of 25 road pothole images (18 missed). |
| **Pothole FPR on Negatives** | $\le 5.0\%$ | **3.43%** (12 / 350 negative images) | **PASS** | 12 false pothole detections across 350 negative images. |

---

## 4. What Could Not Be Verified (Honesty Audit)

The following subsystem components could **not** be verified with genuine real-world evidence in this repository:
1. **User Live Walking Footage**:
   - There are currently fewer than 100 recorded user frames in `sessions/` captured with `--record-session` on real camera hardware. Demo video cutouts and synthetic sequence clips have been purged. Live sequence metrics cannot be claimed until real walking recordings are collected.
2. **Physical IMU Sensor Hardware**:
   - No physical MPU-6050 serial hardware is connected. All tests and live scripts run using `SimulatedIMUReader` (which correctly reports status `SIMULATED`, never `OK`). Real inertial sensor response to human head/chest walking gait has not been verified.
3. **Physical Haptic Feedback**:
   - Arduino serial actuator hardware (vibration motors / coin buzzers) was simulated in tests. User perceptual latency and tactile discernment between patterns (`STOP`, `CAUTION`, `ALL_CLEAR`) have not been tested with human subjects.
4. **End-to-End Blind Pedestrian Walking Trials**:
   - No field trials with visually impaired or blind participants have been performed. Obstacle negotiation in complex dynamic street environments is not verified.

---

## 5. One-Page Operator Demo Script

### System Capabilities (What the Device Actually Does)
- **Monocular Obstacle Tracking**: Detects common objects (pedestrians, vehicles, chairs, obstacles) using YOLOv8n and tracks trajectories over time.
- **Directional Proximity Alerting**: Alerts the user when a large obstacle occupies the center corridor directly ahead.
- **Conservative Ground Surface Analysis**: Inspects ground texture, gradients, and semantic segmentation (SegFormer-B0) to assess walkable paths.
- **Advisory Ground Hazard Alerts**: Overlays detected surface depressions with a dashed warning badge marked `POSSIBLE POTHOLE (advisory)` when geometry indicates a depression on the ground plane.
- **Fail-Safe Fallbacks**: Displays `dist: unknown` when monocular camera geometry is uncalibrated, and marks `IMU: SIMULATED` when hardware sensors are absent.

### System Boundaries (What the Device Does NOT Do)
- **It is NOT a Replacement for a White Cane or Guide Dog**: The system has a measured 2.29% false-WALKABLE rate on hazard scenes (such as upward stairs and drop-off table edges) and cannot guarantee safety on steps or drops.
- **It Does NOT Measure True Metric Depths with One Camera**: Without stereo cameras or LiDAR, object distances are approximations based on bounding box heights under a calibrated pinhole model; unlisted classes or truncated boxes display `dist: unknown`.
- **It Does NOT Guarantee Detection of All Ground Depressions**: Pothole recall is 28.0% (missing 18 out of 25 potholes on the HOLDOUT test set).
- **It Requires Steady Illumination**: Extremely dark rooms, blown-out backlight, or covered lenses trigger an `UNKNOWN` warning state.
