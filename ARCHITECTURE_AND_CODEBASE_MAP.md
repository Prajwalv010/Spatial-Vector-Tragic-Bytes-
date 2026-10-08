# 🦾 SpatialVector-HMI — System Architecture & Codebase Map
> **Nirmaan 2026 Hackathon · Healthcare & Biotech Track**  
> *Chest-worn, vision-first local safety navigation & collision avoidance with directional haptics.*

---

## 📑 Table of Contents
1. [System Overview & End-to-End Data Flow](#1-system-overview--end-to-end-data-flow)
2. [Complete File-by-File Project Inventory](#2-complete-file-by-file-project-inventory)
3. [How to Modify the Dashboard (UI & Web Frontend)](#3-how-to-modify-the-dashboard-ui--web-frontend)
   - [Dashboard File Structure](#dashboard-file-structure)
   - [Exact Line-by-Line Mapping: DOM Element ↔ JavaScript Function](#exact-line-by-line-mapping-dom-element--javascript-function)
   - [Section-by-Section Frontend Guide](#section-by-section-frontend-guide)
4. [How to Modify the Computer Vision & Perception Code](#4-how-to-modify-the-computer-vision--perception-code)
   - [CV Module Architecture](#cv-module-architecture)
   - [Exact Code Locations for Key CV Behaviors](#exact-code-locations-for-key-cv-behaviors)
   - [Step-by-Step CV Modification Recipes](#step-by-step-cv-modification-recipes)
5. [End-to-End Linking Matrix (CV Parameter ↔ Telemetry JSON ↔ Dashboard UI)](#5-end-to-end-linking-matrix)
6. [Hardware & Firmware Integration Guide](#6-hardware--firmware-integration-guide)
7. [Hackathon Presentation & Demo Runbook](#7-hackathon-presentation--demo-runbook)

---

## 1. System Overview & End-to-End Data Flow

SpatialVector-HMI operates on a strict **15-module pipelined architecture (M01 to M15)** designed to answer four fundamental questions:
1. *Is an obstacle on a collision course with the user?* (Trajectory Intersection, TTC & CPA)
2. *How soon will the risk materialize?* (Time-To-Collision - TTC)
3. *Is the physical ground ahead affirmatively walkable and continuous?* (FreeSpace & Gradient Continuity - M14, Ground Hazards - M13)
4. *Which navigation action is safe and actionable?* (Navigation Decision Engine - M15 → Haptic Policy - M09)

```
[ Chest Camera / VDO.Ninja WebRTC ]       [ MPU6050 IMU / Gyroscope ]
                │                                       │
                ▼                                       ▼
    M01: Frame Acquisition (30 FPS)            M05: IMU Reader (100 Hz)
                │                                       │
                ▼                                       │
    M02: YOLO Object Detection                          │
         (yolov8n.pt / nano)                           │
                │                                       │
                ▼                                       │
    M03: Multi-Object Tracking                          │
         (ByteTrack / BoT-SORT)                         │
                │                                       │
                ├───────────────────────────────────────┤
                ▼                                       ▼
    M04: Lucas-Kanade Optical Flow             M05: Ego-Motion Compensation
         (Tracked corners & FOE)                    (Rotational flow subtraction)
                │                                       │
                └───────────────────┬───────────────────┘
                                    ▼
                        M06: Motion & Geometry Engine
                             (Relative velocity & bearing)
                                    │
                                    ▼
                        M07: Collision Predictor
                             (TTC, CPA, Looming Area Expansion)
                                    │
                                    ▼
                        M08: Risk Engine & State Machine
                             (SAFE / CAUTION / WARNING / CRITICAL)
                                    │
                ┌───────────────────┼───────────────────┐
                │                   │                   │
                ▼                   │                   ▼
    M13: Ground Hazard Detector     │       M14: FreeSpace Corridor Estimator
    (pothole_yolov8 / cavities)     │       (Bottom-up continuity & gradient jumps)
                │                   │                   │
                └───────────────────┼───────────────────┘
                                    ▼
                        M15: Navigation Decision Engine
                             (Single Source of Truth guidance & reasons)
                                    │
                                    ▼
                        M09: Safe-Corridor Selector & Policy
                             (Left / Center / Right clearance & urgency)
                                    │
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
       M10: Arduino Haptic Interface     M11: Telemetry Server (FastAPI + WS)
       (3× Vibration Motors: L/C/R)      (Real-time Dashboard & Mobile App)
                    │                               │
                    └───────────────┬───────────────┘
                                    ▼
                        M12: Session Logger, Replay & Benchmark Harness
                             (Ground Truth Eval: scripts/evaluate.py)
```

---

## 2. Complete File-by-File Project Inventory

| File / Path | Primary Role | What It Does | When to Modify This File |
|---|---|---|---|
| [`run.py`](file:///c:/Users/Admin/Desktop/Kshitiz/Nirman-Hackathon/run.py) | **Master Entrypoint** | Initializes all modules (M01–M15), starts camera ingestion, spawns telemetry server, runs the main 30 FPS processing loop, and handles CLI arguments (`--synthetic`, `--arduino-port`, `--source`, `--port`). | When adding new CLI parameters, changing default ports, or altering pipeline loop execution timing. |
| [`run_system.py`](file:///c:/Users/Admin/Desktop/Kshitiz/Nirman-Hackathon/run_system.py) | **Simple Launcher** | Lightweight wrapper to start `run.py` without requiring extra arguments. | Quick start script. |
| [`run_camera_prediction_viewer.py`](file:///c:/Users/Admin/Desktop/Kshitiz/Nirman-Hackathon/run_camera_prediction_viewer.py) | **Standalone CV Visualizer** | High-performance standalone OpenCV desktop viewer that displays bounding boxes, velocity vectors, looming risk, and trajectory cones. | When tuning visual overlays for an external monitor presentation. |
| [`spatialvector/config/default.yaml`](file:///c:/Users/Admin/Desktop/Kshitiz/Nirman-Hackathon/spatialvector/config/default.yaml) | **Master Configuration** | Central YAML configuration for camera index, YOLO model path, confidence thresholds, optical flow parameters, risk weights, corridor thresholds, and baud rates. | **First place to look** to tune system sensitivity, risk weights, or camera resolution without editing Python code. |
| [`camera_source.txt`](file:///c:/Users/Admin/Desktop/Kshitiz/Nirman-Hackathon/camera_source.txt) | **Camera Target** | Single-line file specifying the active video source: webcam index (`0`, `1`, `2`), video file path, or VDO.Ninja URL. | When changing which camera SpatialVector uses on boot. |

### Module `spatialvector/perception/` (Vision Ingestion & Detection)
| File | Role | Description |
|---|---|---|
| [`detector.py`](file:///c:/Users/Admin/Desktop/Kshitiz/Nirman-Hackathon/spatialvector/perception/detector.py) | **M02 YOLO Detector** | Loads Ultralytics YOLO model (`yolov8n.pt`), processes raw frames, extracts bounding boxes (`bbox_xyxy`), class IDs, names, and detection confidences. | Modify to swap YOLO models (e.g. YOLOv10/YOLOv11), change confidence threshold, or add/remove filtered classes. |
| [`tracker.py`](file:///c:/Users/Admin/Desktop/Kshitiz/Nirman-Hackathon/spatialvector/perception/tracker.py) | **M03 Multi-Object Tracker** | Associates detections across frames using ByteTrack/BoT-SORT, maintains track IDs, histories, and calculates velocity. | Modify to change track persistence (`max_missed_frames`), track history length, or swap tracker backend. |
| [`frame_source.py`](file:///c:/Users/Admin/Desktop/Kshitiz/Nirman-Hackathon/spatialvector/perception/frame_source.py) | **M01 Video Ingestion** | Background worker thread pulling frames from OpenCV `cv2.VideoCapture` with monotonic microsecond timestamps. | Modify if camera frame dropping occurs or if adding RTSP stream buffering. |
| [`source_resolver.py`](file:///c:/Users/Admin/Desktop/Kshitiz/Nirman-Hackathon/spatialvector/perception/source_resolver.py) | **Source Selection Engine** | Resolves camera input priorities (`--source` CLI → `camera_source.txt` → `default.yaml` → device `0`). | Modify to add new input formats (e.g. RealSense depth streams). |
| [`vdo_ninja_source.py`](file:///c:/Users/Admin/Desktop/Kshitiz/Nirman-Hackathon/spatialvector/perception/vdo_ninja_source.py) | **Phone WebRTC Bridge** | Uses headless Playwright Chromium to ingest low-latency WebRTC streams from smartphone cameras via VDO.Ninja. | Modify to tweak WebRTC video resolution or browser launch flags. |
| [`schemas.py`](file:///c:/Users/Admin/Desktop/Kshitiz/Nirman-Hackathon/spatialvector/perception/schemas.py) | **Perception Data Types** | Dataclasses for `Frame`, `Detection`, and `Track`. | Modify if adding depth or 3D bounding box coordinates to raw tracks. |

### Module `spatialvector/motion/` (Optical Flow, IMU & Ego-Motion)
| File | Role | Description |
|---|---|---|
| [`optical_flow.py`](file:///c:/Users/Admin/Desktop/Kshitiz/Nirman-Hackathon/spatialvector/motion/optical_flow.py) | **M04 Optical Flow & FOE** | Computes Lucas-Kanade sparse feature flow, rejects bad features with forward-backward error, and calculates Focus of Expansion (FOE). | Modify to tweak optical flow corner counts (`max_corners`) or quality levels. |
| [`imu_reader.py`](file:///c:/Users/Admin/Desktop/Kshitiz/Nirman-Hackathon/spatialvector/motion/imu_reader.py) | **M05 IMU Reader** | Reads serial angular velocity (gyro roll/pitch/yaw) from MPU6050 with EMA smoothing. | Modify to configure IMU serial COM port or filter alpha. |
| [`ego_motion.py`](file:///c:/Users/Admin/Desktop/Kshitiz/Nirman-Hackathon/spatialvector/motion/ego_motion.py) | **M05 Ego-Motion Subtraction** | Subtracts rotational flow induced by user body turns from optical flow vectors to preserve true obstacle movement. | Modify to tune rotation compensation gain or flow quality fallback threshold. |
| [`geometry.py`](file:///c:/Users/Admin/Desktop/Kshitiz/Nirman-Hackathon/spatialvector/motion/geometry.py) | **M06 Geometry Engine** | Normalizes pixel coordinates to relative bearings (radians), frame fractions, and metric approximations. | Modify if using calibrated camera intrinsic matrices (`fx`, `fy`, `cx`, `cy`). |
| [`temporal_smoother.py`](file:///c:/Users/Admin/Desktop/Kshitiz/Nirman-Hackathon/spatialvector/motion/temporal_smoother.py) | **Motion Filter** | Self-adapting Exponential Moving Average (AdaptiveEMA) to suppress high-frequency jitter while preserving sudden evasive movements. | Modify to adjust smoothing alpha and slew rate limits. |

### Module `spatialvector/decision/` (Prediction, Risk Engine & Corridors)
| File | Role | Description |
|---|---|---|
| [`prediction.py`](file:///c:/Users/Admin/Desktop/Kshitiz/Nirman-Hackathon/spatialvector/decision/prediction.py) | **M07 Collision Prediction** | Computes Time-To-Collision (TTC), Closest Point of Approach (CPA), path intersection flags, and optical looming expansion rates. | Modify to change prediction time horizon (e.g. 5.0s), collision contact threshold, or looming sensitivity. |
| [`risk_engine.py`](file:///c:/Users/Admin/Desktop/Kshitiz/Nirman-Hackathon/spatialvector/decision/risk_engine.py) | **M08 Risk State Machine** | Single Source of Truth for hazard state (`SAFE`, `CAUTION`, `WARNING`, `CRITICAL`). Combines TTC, CPA, and intersection confidence with hysteresis debouncing. | Modify to adjust risk weights (`weight_ttc`, `weight_miss_distance`), state thresholds, or hysteresis frame counts. |
| [`corridor_policy.py`](file:///c:/Users/Admin/Desktop/Kshitiz/Nirman-Hackathon/spatialvector/decision/corridor_policy.py) | **M09 Corridor Selector** | Evaluates Left, Center, and Right corridors, selects the safest path, computes urgency level (1–5), and triggers the all-corridors-blocked STOP pattern. | Modify to adjust corridor boundary angles, urgency mappings, or vibration duration tiers. |
| [`adaptive_calibrator.py`](file:///c:/Users/Admin/Desktop/Kshitiz/Nirman-Hackathon/spatialvector/decision/adaptive_calibrator.py) | **Noise Calibrator** | Online dynamic floor estimator that measures environmental motion noise and prevents false collision triggers in crowded rooms. | Modify to change adaptation window or noise floor margin. |
| [`corridor_constants.py`](file:///c:/Users/Admin/Desktop/Kshitiz/Nirman-Hackathon/spatialvector/decision/corridor_constants.py) | **Corridor Constants** | Shared constants for center bearing angle (`CORRIDOR_CENTER_BEARING_RAD = pi/6`). | Modify to widen or narrow the central forward walking corridor. |

### Module `spatialvector/hazards/` (M13 Ground Hazard Detection)
| File | Role | Description |
|---|---|---|
| [`ground_hazard.py`](file:///c:/Users/Admin/Desktop/Kshitiz/Nirman-Hackathon/spatialvector/hazards/ground_hazard.py) | **M13 Hazard Detector** | Identifies potholes, cavities, and damaged floor zones. Features an honest fallback (`allow_heuristic=False`) that suppresses false alerts unless a verified YOLO model (`pothole_yolov8.pt`) is loaded. Bounding boxes above the horizon or overlapping raw YOLO boxes are gated out. | Modify to tune pothole confidence thresholds, IoU overlap limits, or camera horizon position. |
| [`schemas.py`](file:///c:/Users/Admin/Desktop/Kshitiz/Nirman-Hackathon/spatialvector/hazards/schemas.py) | **Hazard Data Types** | Dataclasses for `HazardDetection` (coordinates, confidence, category, persistence). | Modify when adding new surface defect categories (e.g. wet surfaces, ice). |

### Module `spatialvector/freespace/` (M14 FreeSpace & Ground Verification)
| File | Role | Description |
|---|---|---|
| [`corridor_estimator.py`](file:///c:/Users/Admin/Desktop/Kshitiz/Nirman-Hackathon/spatialvector/freespace/corridor_estimator.py) | **M14 Corridor Estimator** | Affirmatively verifies walkable ground in Left, Center, and Right corridors. Executes bottom-up unbroken continuity scan from user's feet upward, multi-scale gradient jump checks ($k=10$, $>28\text{ px}$ span), and dark-blob / obstacle exclusion. Runs at ~300 FPS. | Modify to adjust ground continuity tolerances, gradient sensitivity, or ROI dimensions. |
| [`schemas.py`](file:///c:/Users/Admin/Desktop/Kshitiz/Nirman-Hackathon/spatialvector/freespace/schemas.py) | **FreeSpace Data Types** | Dataclasses for `CorridorStatus` (`WALKABLE`, `BLOCKED`, `UNKNOWN`) and `FreeSpaceResult`. | Modify to extend corridor definitions. |

### Module `spatialvector/guidance/` (M15 Navigation Decision Engine)
| File | Role | Description |
|---|---|---|
| [`decision.py`](file:///c:/Users/Admin/Desktop/Kshitiz/Nirman-Hackathon/spatialvector/guidance/decision.py) | **M15 Guidance Engine** | **Single Source of Truth** for navigation decisions (`WALK FORWARD`, `MOVE LEFT`, `MOVE RIGHT`, `STOP`, `CAUTION`). Synthesizes obstacle collision risk (M08), ground hazards (M13), and freespace continuity (M14) into clear action directives with explicit human-readable reasons. | Modify to update multi-sensor arbitration logic or voice phrasing. |
| [`schemas.py`](file:///c:/Users/Admin/Desktop/Kshitiz/Nirman-Hackathon/spatialvector/guidance/schemas.py) | **Guidance Data Types** | Dataclasses for `NavAction`, `GuidanceDecision` (action, reason, corridor, voice text, banner colors). | Modify to register new navigational directives or HUD display styling. |

### Module `spatialvector/hmi/` (Arduino, Web Telemetry & Logging)
| File | Role | Description |
|---|---|---|
| [`telemetry_server.py`](file:///c:/Users/Admin/Desktop/Kshitiz/Nirman-Hackathon/spatialvector/hmi/telemetry_server.py) | **M11 Telemetry Gateway** | FastAPI + WebSocket server (`/ws/telemetry`), serves web UI static files, handles REST endpoints (`/api/health`, `/api/source`, `/api/settings/risk-thresholds`, `/api/sessions`). | Modify to add new REST APIs, change static mounts, or stream additional debug JSON fields. |
| [`arduino_interface.py`](file:///c:/Users/Admin/Desktop/Kshitiz/Nirman-Hackathon/spatialvector/hmi/arduino_interface.py) | **M10 Arduino Serial Driver** | Asynchronous serial queue communicator that dispatches wire commands `CMD,<dir>,<urgency>,<pattern>,<duration>` to Arduino hardware, with simulated fallback. | Modify to change serial baud rate (115200) or alter wire command formatting. |
| [`logger.py`](file:///c:/Users/Admin/Desktop/Kshitiz/Nirman-Hackathon/spatialvector/hmi/logger.py) | **M12 Session Logger** | Records all frame telemetry to JSONL session files (`sessions/`) with CSV and JSON export routines. | Modify to log additional debug fields into recorded sessions. |
| [`replay.py`](file:///c:/Users/Admin/Desktop/Kshitiz/Nirman-Hackathon/spatialvector/hmi/replay.py) | **M12 Replay Engine** | Replays recorded JSONL sessions frame-by-frame for offline validation and evaluation. | Modify to add synthetic benchmark datasets. |

### Web Dashboards (`web/`)
| File | Role | Description |
|---|---|---|
| [`web/dashboard/index.html`](file:///c:/Users/Admin/Desktop/Kshitiz/Nirman-Hackathon/web/dashboard/index.html) | **Main Desktop/Mobile Dashboard** | Full-featured UI: Live Camera, AR HUD, Corridor Triad, Trajectory Inspector, Haptic Body Diagram, Scenario Replay, Social Assist, and Risk Settings. | Modify to add new HTML elements, buttons, cards, or change layout structure. |
| [`web/dashboard/dashboard.js`](file:///c:/Users/Admin/Desktop/Kshitiz/Nirman-Hackathon/web/dashboard/dashboard.js) | **Main Dashboard Controller** | Handles WebSocket communication, live canvas rendering, HUD drawing, UI tab switching, scenario simulations, contact management, and watchdog monitoring. | **Primary frontend file** to modify any dashboard interaction, telemetry rendering, or charts. |
| [`web/app_simple/index.html`](file:///c:/Users/Admin/Desktop/Kshitiz/Nirman-Hackathon/web/app_simple/index.html) | **Assistive Mobile App HTML** | Mobile-optimized shell with iOS/Android viewport formatting, bottom navigation bar, and tab containers. | Modify for mobile-specific layout changes. |
| [`web/app_simple/app.js`](file:///c:/Users/Admin/Desktop/Kshitiz/Nirman-Hackathon/web/app_simple/app.js) | **Mobile App Router** | Coordinates subpages, tab switching (Home, Haptics, Settings), and central WebSocket telemetry routing. | Modify to register new subpages or navigation tabs. |
| [`web/app_simple/styles.css`](file:///c:/Users/Admin/Desktop/Kshitiz/Nirman-Hackathon/web/app_simple/styles.css) | **Mobile App Styles** | Vanilla CSS styling for mobile app shell, buttons, segmented controls, corridor boxes, and modals. | Modify to adjust mobile theme colors, typography, or spacing tokens. |
| [`web/shared/`](file:///c:/Users/Admin/Desktop/Kshitiz/Nirman-Hackathon/web/shared/) | **Reusable Frontend Modules** | Contains `vdo_ninja_embed.js`, `staleness_watchdog.js`, `reasoning_formatter.js`, and `haptic_body_diagram.js`. | Modify to reuse components across desktop and mobile views. |

### Firmware (`firmware/`)
| File | Role | Description |
|---|---|---|
| [`haptic_controller.ino`](file:///c:/Users/Admin/Desktop/Kshitiz/Nirman-Hackathon/firmware/haptic_controller/haptic_controller.ino) | **Arduino Firmware** | Drives 3× vibration motors on PWM pins (Pin 9: Left, Pin 10: Center, Pin 11: Right) based on serial commands, with safety watchdog timeout. | Modify when changing Arduino pin assignments or vibration wave frequencies. |

### Ground-Truth Evaluation Suite & Benchmarks (`scripts/` & `tests/fixtures/real/`)
| File | Role | Description |
|---|---|---|
| [`scripts/evaluate.py`](file:///c:/Users/Admin/Desktop/Kshitiz/Nirman-Hackathon/scripts/evaluate.py) | **Ground-Truth Evaluation Harness** | Executes quantitative evaluation of object detection (M02), hazard detection (M13), and freespace estimation (M14) across 17 realistic fixtures. Computes precision/recall, measures per-frame latency, and writes results to `eval_results.csv`. | Run to benchmark detection and walkability safety before and after tuning parameters. |
| [`scripts/smoke_test_m14.py`](file:///c:/Users/Admin/Desktop/Kshitiz/Nirman-Hackathon/scripts/smoke_test_m14.py) | **M14 Fast Unit Smoke Test** | Instant sanity test verifying table corners, table edges, dark objects, clear paths, and covered lenses. | Quick command-line smoke test during development. |
| [`tests/fixtures/real/generate_real_fixtures.py`](file:///c:/Users/Admin/Desktop/Kshitiz/Nirman-Hackathon/tests/fixtures/real/generate_real_fixtures.py) | **Fixture Generator** | Generates 17 standardized benchmark image fixtures and paired JSON ground-truth metadata matching exact camera geometry. | Modify when introducing new failure edge cases or real camera calibration scenarios. |

### Model Management & Integrity Verification (`models/`)
| File | Role | Description |
|---|---|---|
| [`models/download_verify_model.py`](file:///c:/Users/Admin/Desktop/Kshitiz/Nirman-Hackathon/models/download_verify_model.py) | **Model Verifier & Card** | Inspects local presence of `yolov8n.pt` and `pothole_yolov8.pt`, verifies SHA256 checksums, and reports readiness for inference. | Run to verify model weights before live demonstration. |
| [`models/README.md`](file:///c:/Users/Admin/Desktop/Kshitiz/Nirman-Hackathon/models/README.md) | **Model Architecture Card** | Documents model origins, input/output tensors, SHA256 hashes, and safety fallback behavior. | Reference for judges and engineers. |

---

## 3. How to Modify the Dashboard (UI & Web Frontend)

### Dashboard File Structure
```
web/
├── dashboard/
│   ├── index.html        <-- All HTML markup, modals, SVG icons, and inline styles
│   └── dashboard.js       <-- All client logic, WebSocket handlers, canvas renders, and state
└── app_simple/
    ├── index.html        <-- Mobile shell & bottom navigation bar
    ├── app.js            <-- Mobile app router & WebSocket receiver
    ├── styles.css        <-- Mobile CSS design tokens
    └── pages/
        ├── home.js       <-- Live navigation screen
        ├── haptics.js    <-- Haptic status & interactive motor test
        ├── settings.js   <-- Settings & risk threshold sliders
        └── social_assist.js <-- Enrolled contacts & Add Friend wizard
```

### Exact Line-by-Line Mapping: DOM Element ↔ JavaScript Function

#### A. Header & System Status
| DOM Element ID (`index.html`) | Line in `index.html` | Controlling JS Function / Variable (`dashboard.js`) | Line in `dashboard.js` | What It Does / How to Modify |
|---|---|---|---|---|
| `#conn-pill-badge` | L1395 | `checkWatchdog()` | L356–L395 | Displays connection status (`conn-pill disconnected` vs `conn-pill`). Modify inside `checkWatchdog()` to alter connection timeout (default 1500ms). |
| `#conn-status-text` | L1398 | `checkWatchdog()`, `connectWs()` | L367, L575 | Displays `"Local Edge Connected"`, `"Pipeline Offline / Not Started"`, or `"Pipeline Stalled"`. |
| `#header-clock` | L1400 | `updateClock()` | L398–L410 | Real-time digital clock in header and iPhone notch. |
| `#staleness-banner` | L1385 | `checkWatchdog()` | L363, L376 | Red alert banner shown when telemetry packets stop arriving from backend. |

#### B. Live Risk & Hazard Index
| DOM Element ID (`index.html`) | Line in `index.html` | Controlling JS Function / Variable (`dashboard.js`) | Line in `dashboard.js` | What It Does / How to Modify |
|---|---|---|---|---|
| `#risk-alert-card` | L1406 | `updateRiskBanner(state, score)` | L750–L785 | Main top card with color border (green for SAFE, amber for CAUTION, orange for WARNING, red for CRITICAL). |
| `#val-risk-score` | L1426 | `updateRiskBanner()`, `resetDashboardLiveMetrics()` | L329, L765 | Numerical risk score (`0.00` to `1.00`). When offline, displays `—`. |
| `#gauge-circle-stroke` | L1424 | `updateRiskBanner()` | L766 | Circular SVG progress ring that fills as collision risk increases (`stroke-dasharray`). |
| `#val-ttc` | L1432 | `handleTelemetryMessage(msg)` | L667 | Displays Time-To-Collision in seconds (e.g. `2.4 s`). When safe, shows `—`. |
| `#val-cpa` | L1437 | `handleTelemetryMessage(msg)` | L668 | Displays Closest Point of Approach in meters (e.g. `0.45 m`). |

#### C. Home View Camera Feed & AR HUD Overlay
| DOM Element ID (`index.html`) | Line in `index.html` | Controlling JS Function / Variable (`dashboard.js`) | Line in `dashboard.js` | What It Does / How to Modify |
|---|---|---|---|---|
| `#camera-stream-badge` | L1448 | `resetDashboardLiveMetrics()`, `handleTelemetryMessage()` | L333, L651 | Displays `● CAMERA ACTIVE` (live) or `○ NO SIGNAL` (standby/offline). |
| `#vdo-ninja-frame` | L1453 | `initCameraSource()`, `formatVdoNinjaUrl()` | L440–L460 | Embedded `iframe` loading live VDO.Ninja video stream from mobile phone or webcam. |
| `#live-overlay-canvas` | L1455 | `drawLiveOverlay()` | L470–L565 | HTML5 canvas on top of camera stream. Draws horizon line, corridor guide rails, bounding boxes, and velocity vectors. |
| `#btn-toggle-hud` | L1458 | `toggleHudOverlay()` | L415–L425 | Button to toggle AR HUD overlay on/off. |
| `#btn-toggle-aspect` | L1461 | `toggleVideoAspect()` | L426–L438 | Toggles between `Fit` (letterbox) and `Fill` (zoom-to-fill). |

#### D. Corridor Triad (Left / Center / Right)
| DOM Element ID (`index.html`) | Line in `index.html` | Controlling JS Function / Variable (`dashboard.js`) | Line in `dashboard.js` | What It Does / How to Modify |
|---|---|---|---|---|
| `#box-corr-left` | L1467 | `updateCorridorBoxes(left, center, right)` | L820–L845 | Left corridor card. Changes CSS class to `corridor-box safe`, `caution`, or `risky`. |
| `#txt-corr-left` | L1469 | `applyCorridorStyle(boxEl, txtEl, lblEl, score)` | L827–L843 | Numerical risk in Left corridor (e.g. `0.12`). Displays `—` when in standby. |
| `#lbl-corr-left` | L1470 | `applyCorridorStyle()` | L834, L838 | Status text: `SAFE`, `CAUTION`, `RISKY`, or `STANDBY`. |
| `#box-corr-center` | L1472 | `updateCorridorBoxes()` | L821 | Center corridor card. |
| `#txt-corr-center` | L1474 | `applyCorridorStyle()` | L827 | Center corridor score. |
| `#box-corr-right` | L1477 | `updateCorridorBoxes()` | L822 | Right corridor card. |
| `#txt-corr-right` | L1479 | `applyCorridorStyle()` | L827 | Right corridor score. |

#### E. Recommended Direction Banner
| DOM Element ID (`index.html`) | Line in `index.html` | Controlling JS Function / Variable (`dashboard.js`) | Line in `dashboard.js` | What It Does / How to Modify |
|---|---|---|---|---|
| `#rec-dir-name` | L1497 | `updateDirectionRecommendation()` | L846–L885 | Displays clear directive: `MOVE LEFT`, `MOVE RIGHT`, `PROCEED CENTER`, or `STOP IMMEDIATELY`. |
| `#rec-dir-sub` | L1498 | `updateDirectionRecommendation()` | L851, L859 | Explanatory subtext (e.g. `Left corridor is safest · Path clear`). |
| `#rec-arrow-icon` | L1489 | `updateDirectionRecommendation()` | L853, L861 | Dynamic directional SVG arrow pointing in recommended direction. |

#### F. Trajectory Prediction Inspector Tab
| DOM Element ID (`index.html`) | Line in `index.html` | Controlling JS Function / Variable (`dashboard.js`) | Line in `dashboard.js` | What It Does / How to Modify |
|---|---|---|---|---|
| `#screen-predict` | L1507 | `switchTab("predict")` | L1065 | Screen container for Tab 2 (Prediction Inspector). |
| `#inspector-canvas` | L1539 | `drawInspector2D()`, `drawInspectorTop()` | L1140–L1280 | Canvas displaying 2D perspective or bird's-eye top-down trajectory cones and collision points. |
| `#btn-pred-2d` / `#btn-pred-top` | L1531–L1533 | `switchPredictView(viewMode)` | L1120–L1138 | Switcher between 2D egocentric view, Bird's-Eye Top view, and Risk-over-Time timeline. |
| `#predict-risk-timeline-canvas` | L1550 | `drawPredictRiskTimeline()` | L1290–L1350 | Canvas rendering projected risk curve over the next 5 seconds. |

#### G. Haptics Tab & Interactive Motor Visualizer
| DOM Element ID (`index.html`) | Line in `index.html` | Controlling JS Function / Variable (`dashboard.js`) | Line in `dashboard.js` | What It Does / How to Modify |
|---|---|---|---|---|
| `#screen-haptics` | L1600 | `switchTab("haptics")` | L1065 | Screen container for Tab 3 (Haptics). |
| `#svg-motor-left` / `#svg-motor-right` | L1630–L1650 | `triggerMotorVisual(motor, intensity)` | L1720–L1745 | Chest harness diagram highlighting active left/right vibration motors. |
| `#btn-test-vib-left` | L1680 | `testHapticPattern("LEFT_PULSE")` | L1760 | Interactive button to trigger test vibration on Left motor. |
| `#haptic-history-list` | L1705 | `renderHapticCommandsList()` | L1780 | Real-time scrollable log of recently executed vibration commands. |

#### H. Scenario Replay & Test Harness Tab
| DOM Element ID (`index.html`) | Line in `index.html` | Controlling JS Function / Variable (`dashboard.js`) | Line in `dashboard.js` | What It Does / How to Modify |
|---|---|---|---|---|
| `#screen-scenarios` | L1740 | `switchTab("scenarios")` | L1065 | Screen container for Tab 4 (Test Replay). |
| `selectScenario(id)` | L1780 | `selectScenario(id)` | L1840–L1890 | Switches between synthetic benchmarks: `s1` (Walking Parallel), `s2` (Scooter Approaching), `s3` (Center Obstacle), `s4` (Doorway Entrance), `s5` (Multiple Pedestrians), `s6` (Corridor Turn). |
| `#btn-replay-play` | L1810 | `toggleReplayPlay()` | L1900 | Play / Pause button for scenario playback. |
| `#replay-timeline-canvas` | L2143 | `drawReplayTimeline()` | L1940–L1990 | Interactive timeline scrubber for simulated playback. |

#### I. Social Assist Tab & Add Friend 3-Step Wizard
| DOM Element ID (`index.html`) | Line in `index.html` | Controlling JS Function / Variable (`dashboard.js`) | Line in `dashboard.js` | What It Does / How to Modify |
|---|---|---|---|---|
| `#screen-social` | L1980 | `switchTab("social")` | L1065 | Screen container for Tab 5 (Social Assist). |
| `#add-friend-modal` | L2154 | `openAddFriendModal()`, `closeAddFriendModal()` | L2250–L2275 | 3-step modal wizard to enroll known friends. |
| `#btn-af-next` / `#btn-af-back` | L2211–L2213 | `nextAddFriendStep()`, `prevAddFriendStep()` | L2220–L2245 | Steps through wizard: Step 1 (Photos) → Step 2 (Name & Relation) → Step 3 (Review & Confirm). |
| `#add-friend-file-input` | L2170 | `handleFriendPhotosSelect(event)` | L2260 | Local file input reading up to 4 reference photos via `FileReader` (processed 100% locally on edge). |
| `#contact-profile-modal` | L2220 | `openContactProfileModal(id)` | L2335 | Shows contact details, enrolled photos count, and edge recognition status. |

#### J. Settings & Risk Engine Tuning Modal
| DOM Element ID (`index.html`) | Line in `index.html` | Controlling JS Function / Variable (`dashboard.js`) | Line in `dashboard.js` | What It Does / How to Modify |
|---|---|---|---|---|
| `#settings-modal` | L2020 | `openSettingsModal()`, `closeSettingsModal()` | L920–L960 | Configuration modal for runtime tuning. |
| `#input-camera-url` | L2035 | `saveCameraSourceSetting()` | L970 | Saves custom VDO.Ninja URL or device index back to `/api/set-source`. |
| `#slider-weight-ttc` | L2050 | `onRiskSliderChange()` | L990 | Dynamic slider for TTC weight in M08 risk calculation. |
| `#btn-save-thresholds` | L2090 | `saveRiskThresholds()` | L1020 | Posts updated weights to `/api/settings/risk-thresholds`, updating the live Python RiskEngine in memory without restarting. |

---

## 4. How to Modify the Computer Vision & Perception Code

### CV Module Architecture
```
Frame Ingestion (frame_source.py)
      │
      ▼
YOLO Detector (detector.py)  ───>  Detections: bbox_xyxy, confidence, class_name
      │
      ▼
Multi-Object Tracker (tracker.py) ───>  Tracks: track_id, velocity (vx, vy), history + raw_bboxes
      │
      ├───────────────────────────────────────┬───────────────────────────────────────┐
      ▼                                       ▼                                       ▼
Ego-Motion Compensation (ego_motion.py)   Ground Hazard Detector (ground_hazard.py) FreeSpace Estimator (corridor_estimator.py)
      │                                   (pothole model & raw YOLO overlap gate)   (bottom-up continuity & gradient jumps)
      ▼                                       │                                       │
Collision Prediction (prediction.py)          │                                       │
      │                                       │                                       │
      ▼                                       │                                       │
Risk Engine (risk_engine.py)                  │                                       │
      │                                       │                                       │
      └───────────────────────────────────────┼───────────────────────────────────────┘
                                              ▼
                                 Navigation Decision Engine (decision.py)
                                 (Single Source of Truth guidance & reasons)
                                              │
                                              ▼
                                 Corridor Policy (corridor_policy.py) ───>  Haptic Commands & Urgency
```

### Exact Code Locations for Key CV Behaviors

#### 1. Change the Detection Model or Weights
- **File**: [`spatialvector/perception/detector.py`](file:///c:/Users/Admin/Desktop/Kshitiz/Nirman-Hackathon/spatialvector/perception/detector.py#L21-L53)
- **Lines 21–36**: `ObjectDetector.__init__`:
  ```python
  def __init__(
      self,
      model_path: str = "yolov8n.pt",       # <-- Change to "yolov8s.pt", "yolov10n.pt", or custom ONNX
      confidence_threshold: float = 0.40,  # <-- Change detection confidence cutoff
      class_filter: Optional[list[str]] = None,
      device: Optional[str] = None,         # <-- "cpu", "cuda", or "mps"
  ):
  ```
- **Line 47**: `self._model = YOLO(self.model_path)` initializes the Ultralytics model.
- **Config Alternative**: You can change this without editing code by updating [`spatialvector/config/default.yaml`](file:///c:/Users/Admin/Desktop/Kshitiz/Nirman-Hackathon/spatialvector/config/default.yaml#L9-L16):
  ```yaml
  detector:
    model_path: "yolov8n.pt"
    confidence_threshold: 0.40
    class_filter: ["person", "bicycle", "car", "chair"]
  ```

#### 2. Add New Detected Object Classes
- **File**: [`spatialvector/config/default.yaml`](file:///c:/Users/Admin/Desktop/Kshitiz/Nirman-Hackathon/spatialvector/config/default.yaml#L12-L16)
- **Edit**: Add any standard COCO dataset classes to `class_filter`:
  ```yaml
  class_filter:
    - person
    - bicycle
    - car
    - motorcycle
    - bus
    - dog
    - chair
    - fire hydrant
  ```
- **How it filters**: [`spatialvector/perception/detector.py`](file:///c:/Users/Admin/Desktop/Kshitiz/Nirman-Hackathon/spatialvector/perception/detector.py#L112-L130) checks `if self.class_filter and d.class_name.lower() not in self.class_filter:` and separates filtered detections into an inspection log.

#### 3. Adjust Collision Prediction (TTC & CPA Calculations)
- **File**: [`spatialvector/decision/prediction.py`](file:///c:/Users/Admin/Desktop/Kshitiz/Nirman-Hackathon/spatialvector/decision/prediction.py#L88-L150)
- **Lines 21–29**: Constant velocity formulas:
  - **CPA Time**: $t_{cpa} = -\frac{\vec{p}_{rel} \cdot \vec{v}_{rel}}{|\vec{v}_{rel}|^2}$
  - **CPA Distance**: $d_{cpa} = |\vec{p}_{rel} + \vec{v}_{rel} \cdot t_{cpa}|$
  - **TTC**: $t_{ttc} = \text{time when } d(t) \le \text{contact\_threshold}$
- **Lines 68–80**: Key prediction thresholds:
  ```python
  _DEFAULT_HORIZON_S = 5.0                # Warning horizon in seconds
  _DEFAULT_CONTACT_THRESHOLD_NORM = 0.05  # Contact distance (5% of frame width)
  _DEFAULT_CORRIDOR_WIDTH_NORM = 0.12     # Collision corridor half-width
  ```
- **Lines 170–210**: **Optical Looming Area Expansion**:
  Computes bounding box scale expansion rate:
  ```python
  looming_rate = (current_bbox_area - prev_bbox_area) / (prev_bbox_area * dt)
  ```
  If an obstacle occupies $\ge 22\%$ of the central frame height and is growing in size, it triggers direct proximity danger even if relative velocity is ambiguous.

#### 4. Tune Risk Weights & State Thresholds
- **File**: [`spatialvector/decision/risk_engine.py`](file:///c:/Users/Admin/Desktop/Kshitiz/Nirman-Hackathon/spatialvector/decision/risk_engine.py#L60-L75)
- **Lines 62–68**:
  ```python
  _DEFAULT_WEIGHT_TTC = 0.50               # 50% weight on time-to-collision
  _DEFAULT_WEIGHT_MISS_DISTANCE = 0.30     # 30% weight on CPA distance
  _DEFAULT_WEIGHT_INTERSECTION_CONF = 0.20 # 20% weight on path intersection
  _DEFAULT_THRESHOLDS = {
      "caution": 0.30,   # Risk >= 0.30 -> CAUTION (Amber)
      "warning": 0.60,   # Risk >= 0.60 -> WARNING (Orange)
      "critical": 0.85,  # Risk >= 0.85 -> CRITICAL (Red)
  }
  _DEFAULT_HYSTERESIS_UP = 3    # Frames above threshold required to escalate
  _DEFAULT_HYSTERESIS_DOWN = 5  # Frames below threshold required to clear (avoids flickering)
  ```
- **Lines 280–320**: `calculate_object_risk()` computes the per-object composite risk score.

#### 5. Adjust Corridor Division Angles & Safe Direction Policy
- **File**: [`spatialvector/decision/corridor_constants.py`](file:///c:/Users/Admin/Desktop/Kshitiz/Nirman-Hackathon/spatialvector/decision/corridor_constants.py)
  ```python
  CORRIDOR_CENTER_BEARING_RAD = 0.5236  # pi / 6 radians (30 degrees half-angle)
  ```
- **File**: [`spatialvector/decision/corridor_policy.py`](file:///c:/Users/Admin/Desktop/Kshitiz/Nirman-Hackathon/spatialvector/decision/corridor_policy.py#L55-L65)
  - `_DEFAULT_ALL_UNSAFE_THRESHOLD = 0.70`: If Left, Center, and Right risk all exceed 70%, triggers the `STOP_CRITICAL` pattern.
  - `_DURATION_MS = {1: 400, 2: 400, 3: 300, 4: 200, 5: 150}`: Vibration pulse lengths per urgency tier.

#### 6. Tune Ground Verification & Hazard Sensitivity (M13 & M14)
- **File**: [`spatialvector/config/default.yaml`](file:///c:/Users/Admin/Desktop/Kshitiz/Nirman-Hackathon/spatialvector/config/default.yaml)
  ```yaml
  hazard_detector:
    persistence_required: 2       # Consecutive detections needed for surface alerts
    min_conf: 0.45               # Confidence cutoff for pothole model
    max_iou_with_yolo: 0.15      # Bboxes overlapping YOLO objects are rejected
    allow_heuristic: false       # Never alert on dark blobs without verified weights

  freespace:
    roi_y_start_ratio: 0.50      # Ground analysis starts at lower 50% of frame
    edge_std_thresh: 18.0        # Max texture standard deviation for flat ground
    smoothing_window: 3          # Moving average filter across frames
  ```
- **Code Reference**:
  - [`spatialvector/freespace/corridor_estimator.py`](file:///c:/Users/Admin/Desktop/Kshitiz/Nirman-Hackathon/spatialvector/freespace/corridor_estimator.py): Look at `_continuity_scan()` (strictly starts from band 0) and `_gradient_jump_check()` (evaluates step jumps, windowed diagonal diffs $k=10$, and span shifts $>28\text{ px}$).
  - [`spatialvector/hazards/ground_hazard.py`](file:///c:/Users/Admin/Desktop/Kshitiz/Nirman-Hackathon/spatialvector/hazards/ground_hazard.py): Bounding boxes above `roi_y_start` or overlapping raw YOLO detections are pruned immediately.

#### 7. Navigation Decision Engine & Voice Guidance (M15)
- **File**: [`spatialvector/guidance/decision.py`](file:///c:/Users/Admin/Desktop/Kshitiz/Nirman-Hackathon/spatialvector/guidance/decision.py)
  - Acts as the **Single Source of Truth** for navigation decisions.
  - High collision risk ($>0.70$) or obstacles in center $\to$ `STOP` or `MOVE LEFT/RIGHT`.
  - Missing ground continuity / table edge $\to$ `CAUTION` with reason `"Ground discontinuity detected"`.
  - Active pothole ahead $\to$ `CAUTION` or `STOP` with reason `"Ground hazard ahead"`.
  - Clear corridor with unbroken ground $\to$ `WALK FORWARD`.

---

## 5. End-to-End Linking Matrix

This matrix shows the complete pipeline trace: from raw Python computer vision data through the WebSocket JSON schema to the exact HTML element rendered in the dashboard.

| Pipeline Metric | Python Producer Variable | Telemetry JSON Path (`TelemetryMessage`) | Frontend Consumer (`dashboard.js`) | DOM Element Rendered (`index.html`) |
|---|---|---|---|---|
| **Risk State** | `RiskEngine.evaluate().state` | `msg.risk_state.state` | `state` in `updateRiskBanner()` | `#alert-state-name`, `#risk-alert-card` |
| **Global Risk Score** | `RiskEngine.evaluate().global_risk` | `msg.risk_state.global_risk` | `currentGlobalRisk` in `updateRiskBanner()` | `#val-risk-score`, `#gauge-circle-stroke` |
| **Est. Contact (TTC)** | `Prediction.ttc_s` | `msg.tracks[i].ttc_s` | `currentTTC` in `handleTelemetryMessage()` | `#val-ttc` |
| **Pass Margin (CPA)** | `Prediction.cpa` | `msg.tracks[i].cpa` | `currentCPA` in `handleTelemetryMessage()` | `#val-cpa` |
| **Left Corridor Risk** | `RiskState.corridor_risks["left"]` | `msg.risk_state.corridor_risks.left` | `leftRisk` in `updateCorridorBoxes()` | `#txt-corr-left`, `#box-corr-left` |
| **Center Corridor Risk** | `RiskState.corridor_risks["center"]` | `msg.risk_state.corridor_risks.center` | `centerRisk` in `updateCorridorBoxes()` | `#txt-corr-center`, `#box-corr-center` |
| **Right Corridor Risk** | `RiskState.corridor_risks["right"]` | `msg.risk_state.corridor_risks.right` | `rightRisk` in `updateCorridorBoxes()` | `#txt-corr-right`, `#box-corr-right` |
| **Ground Walkability (M14)** | `FreeSpaceResult.centre` | `msg.freespace.centre` | `updateCorridorBoxes()` | Visual corridor floor outline / Walkable indicator |
| **Ground Hazard (M13)** | `HazardDetection.label` | `msg.hazards` | `drawLiveOverlay()` | Ground hazard highlight box on canvas |
| **Action Directive (M15)** | `GuidanceDecision.action.value` | `msg.guidance.action` | `updateDirectionRecommendation()` | `#rec-dir-name`, `#rec-dir-sub`, `#rec-arrow-icon` |
| **Guidance Rationale (M15)** | `GuidanceDecision.reason` | `msg.guidance.reason` | `updateDirectionRecommendation()` | `#rec-dir-sub`, AI reasoning card |
| **Recommended Direction** | `CorridorPolicy.evaluate().direction` | `msg.haptic.direction` | `updateDirectionRecommendation()` | `#rec-dir-name`, `#rec-dir-sub`, `#rec-arrow-icon` |
| **Track Bounding Box** | `Track.bbox_xyxy` | `msg.tracks[i].bbox` | `drawLiveOverlay()` | Canvas rectangles on `#live-overlay-canvas` |
| **Trajectory Vector** | `Track.vx`, `Track.vy` | `msg.tracks[i].velocity` | `drawLiveOverlay()` | Directional arrows on `#live-overlay-canvas` |
| **Haptic Motor Output** | `HapticCommand.pattern_id` | `msg.haptic.pattern_id` | `updateHapticTelemetry()` | `#svg-motor-left`, `#svg-motor-right` |
| **Arduino Status** | `ArduinoInterface.get_status()` | `msg.pipeline_health.arduino` | `handleTelemetryMessage()` | `#dev-ard-status` |
| **Watchdog Timestamp** | `time.monotonic()` | `msg.t_sent` | `lastMessageTimestamp` | `#conn-status-badge`, `#staleness-banner` |

---

## 6. Hardware & Firmware Integration Guide

### 1. Arduino Pinout & Motor Mapping
The firmware (`firmware/haptic_controller/haptic_controller.ino`) maps directional haptic commands to 3× PWM-capable digital output pins:
- **Left Vibration Motor**: Pin `9` (PWM)
- **Center Vibration Motor**: Pin `10` (PWM)
- **Right Vibration Motor**: Pin `11` (PWM)
- **Serial Baud Rate**: `115200 bps`

### 2. Serial Wire Protocol
Commands sent from Python (`spatialvector/hmi/arduino_interface.py`) to Arduino:
```
CMD,<direction>,<urgency>,<pattern_id>,<duration_ms>\n
```
Examples:
- `CMD,LEFT,2,LEFT_SLOW,400\n` → Pulses Left motor at 40% PWM for 400ms.
- `CMD,RIGHT,4,RIGHT_FAST,200\n` → Pulses Right motor at 80% PWM for 200ms.
- `CMD,STOP,5,STOP_CRITICAL,150\n` → Pulses all three motors simultaneously at 100% PWM.

### 3. Running with Real Hardware vs Simulated Hardware
- **Without hardware (Synthetic / Laptop Demo)**:
  ```bash
  python run.py --synthetic --port 8081
  ```
  *(Uses `SimulatedArduinoInterface` automatically).*
- **With real Arduino hardware connected via USB**:
  ```bash
  # Windows:
  python run.py --arduino-port COM3 --port 8081

  # Linux / Raspberry Pi:
  python run.py --arduino-port /dev/ttyACM0 --port 8081
  ```

---

## 7. Hackathon Presentation & Demo Runbook

### Option 1: Live Presentation with Synthetic Simulation (Zero Hardware Needed)
1. Open PowerShell / Terminal in project directory:
   ```powershell
   python run.py --synthetic --port 8081
   ```
2. Open Chrome to [`http://localhost:8081/`](http://localhost:8081/).
3. In the bottom navigation, click **Test** (Tab 4).
4. Run the benchmark scenarios:
   - **Scenario S1: Walking Parallel to Wall/Object**: Demonstrates that proximity does *not* trigger false alerts because trajectories do not intersect.
   - **Scenario S2: Fast Approaching Scooter**: Shows high TTC urgency triggering `STOP_CRITICAL` and immediate right-corridor clearance.
   - **Scenario S3: Center Path Blocked**: Demonstrates the Corridor Triad computing Left (`0.12 SAFE`) vs Center (`0.85 RISKY`) and recommending `MOVE LEFT`.

### Option 2: Live Webcam Demonstration
1. Ensure your webcam is connected. Set the camera index in `camera_source.txt`:
   ```
   0
   ```
2. Start the master pipeline:
   ```powershell
   python run.py --source 0 --port 8081
   ```
3. Open [`http://localhost:8081/`](http://localhost:8081/) in your browser.
4. Walk towards the camera: observe the **AR HUD Overlay**, bounding boxes, **Risk Gauge** escalation from SAFE to CAUTION, and corridor scores updating in real-time.

### Option 3: Phone Camera Streaming over VDO.Ninja (Chest-Mounted Demo)
1. On your smartphone, navigate to [vdo.ninja/webcam](https://vdo.ninja/webcam).
2. Start streaming and copy the View Link (e.g. `https://vdo.ninja/?view=kshitizcam`).
3. Save the URL in `camera_source.txt` or pass it via CLI:
   ```powershell
   python run.py --source "https://vdo.ninja/?view=kshitizcam" --port 8081
   ```
4. Open the dashboard on your laptop or phone at `http://<your-laptop-ip>:8081/`.
5. The live phone camera stream displays in the dashboard with AR collision vector overlays.

---
*Created for Nirmaan 2026 Hackathon · SpatialVector-HMI Project Repository.*
