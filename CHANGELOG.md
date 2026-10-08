# SpatialVector-HMI Changelog

All notable changes, threshold modifications, and architectural updates are documented in this file.

## [Round 4 Freeze] - 2026-10-04

### Evaluation Pipeline & Integrity
- **Per-Image Ground Truth**: Replaced all category-level default labels across 360 TUNE images (`tests/fixtures/real/tune/*.json`) with per-image verified labels. Pre-filled with YOLOv8x and manually audited: 300 out of 360 labels were corrected for specific COCO objects present.
- **Fixture Mismatch Purge**: Fixed 9 category mismatches discovered during audit (indoor floors showing vertical walls, table edges showing lakes, blank wall showing football pitch) and replaced them with verified category footage.
- **Fresh HOLDOUT Generation**: Generated an independent, un-burned HOLDOUT split of 375 images (25 images per category across 15 categories) from Wikimedia Commons with perceptual dHash deduplication (Hamming distance >= 6 against TUNE) to guarantee zero data leakage.
- **Holdout Freeze Check (`scripts/check_holdout_freeze.py`)**: Added CI-style freeze validation script. Prevents modifying thresholds or configs after the holdout freeze date without an explicit changelog entry.

### Configuration & Thresholds (`spatialvector/config/default.yaml`)
- **SegFormer ADE20K Ground Classes (`ground_classes`)**: Configured explicitly as `[3, 6, 11, 28, 52]` (`floor`, `road`, `sidewalk`, `rug`, `path`). Removed class 13 (`earth`) from ground classes to maintain 0.00% False WALKABLE on outdoor dirt stairs/ledges.
- **SegFormer ADE20K Obstacle Classes (`obstacle_classes`)**: Configured explicitly as `[0, 1, 2, 7, 8, 10, 12, 14, 15, 31, 32, 33, 53, 56, 59, 64, 80, 102, 121]` (walls, buildings, sky, furniture, stairs, steps).
- **Two-Stage Pothole Rule (`hazard_detector.require_ground_confirmation: true`)**: Pothole model candidates are only confirmed if the Freespace Estimator confirms the presence of road/ground surface (`g_frac >= 0.25`).
- **Pothole Advisory Only (`hazard_detector.advisory_only: true`)**: Pothole detections are displayed on HUD and announced via voice but do not override or stop WALK_FORWARD decisions unless verified with high certainty on navigable ground.
- **Fail-Safe SegFormer Fallback (`freespace.allow_classical_fallback: false`)**: If the deep segmentation model fails to load, Freespace Estimator outputs `UNKNOWN` with reason `"SEGMENTATION MODEL NOT LOADED"`, refusing to issue `WALK_FORWARD`.

### Packaging & Dependencies
- **Pinned Dependencies**: Pinned `transformers>=4.40.0,<5.0.0` and `huggingface_hub>=0.20.0,<1.0.0` in `requirements.txt`.
- **Model Self-Check**: Implemented startup self-check in `run_camera_prediction_viewer.py` and `models/download_verify_model.py` verifying YOLOv8, pothole detector, and SegFormer weights with checksums.

### Step 7 — Removal of Fabricated or Assumed Values
- **Freespace Confidence (`spatialvector/freespace/corridor_estimator.py`)**: Replaced all hardcoded constants (0.95, 0.85, 0.80, 0.10, 0.0) with a computed evidence-based confidence formula combining SegFormer ground fraction margin, class softmax probabilities, cross-cue agreement (Sobel texture + brightness), and temporal hysteresis. Unknown corridors strictly output 0.00 confidence.
- **Pothole Distance Calibration & Provenance (`spatialvector/hazards/ground_hazard.py`)**: Eliminated arbitrary 5.0m to 0.4m linear bounding box mappings. Pothole distance is computed via pinhole flat-ground geometry ($Z = f_y \cdot H_{cam} / \Delta y$) only when optical parameters and camera mounting height/pitch are configured; otherwise explicitly outputs `dist_m = None` and `dist_provenance = "unknown"`.
- **IMU Sensor Source Disambiguation (`spatialvector/motion/imu.py`, `spatialvector/hmi/schemas.py`)**: Replaced `is_simulated: bool` with `status: Literal["HARDWARE", "SIMULATED", "DISCONNECTED"]`. Real and synthetic IMU feeds are explicitly tagged so downstream telemetry cannot present synthetic sinusoids as physical body motion.
- **Real Pipeline Health Subsystem Checks (`spatialvector/hmi/schemas.py`, `run.py`)**: Replaced all hardcoded `{"camera": "OK", "imu": "OK", ...}` dictionaries with `build_pipeline_health()` querying real subsystems (FrameSource, IMUReader, Detector, SegFormer model status). Returns `"DISCONNECTED"` or `"UNKNOWN"` when hardware is absent or unconfigured.
- **Stationary Obstacle TTC Invariant (`spatialvector/decision/prediction.py`)**: Proximity risk for large stationary obstacles no longer fabricates a synthetic collision horizon (`ttc_s = None` without approach velocity), preventing fabricated time-to-contact values.
- **Anti-Fabrication CI Test Suite (`tests/test_step7_no_fabricated_values.py`)**: 12 dedicated automated tests verifying that no hardcoded confidence constants or fabricated physical distances are emitted by core estimation and guidance modules.

