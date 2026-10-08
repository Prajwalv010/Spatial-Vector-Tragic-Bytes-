# SPATIALVECTOR-HMI — Engineering Blueprint v2

**Predictive Collision Intelligence for Assistive Mobility**
Council Review + Architecture + Module Specifications + 4-Person Implementation Roadmap

> **Nirmaan 2026 Hackathon · Healthcare & Biotech Track**

---

## Core Thesis

> The goal is not to detect everything. The goal is to decide whether the user and an obstacle are on a collision course, how soon the risk materializes, which direction is safer, and how to communicate that decision with minimal cognitive load.

Prepared as the engineering planning baseline for the Nirmaan 2026 Healthcare & Biotech prototype. The source proposal already defines a chest-mounted camera, YOLO, optical flow, FOE, TTC/CPA, directional HMI and a 24-hour proof-of-concept. This document restructures that proposal into a buildable, testable modular system.

---

## Revision Notes (v2 — Independent Pressure-Test Pass)

All original content is preserved. Seven corrections were made after an independent review:
1. Removed leftover citation-tool artifacts throughout the text
2. Revised the Feasibility rating in the Executive Council Verdict scorecard from **High** to **Medium-High**, since ego-motion/optical-flow from a walking chest-worn camera is harder than the original rating implied
3. Added two non-goals — ground-level hazards are out of scope, and no unvalidated testing on real visually-impaired users
4. Added a required **fallback caution threshold** to M05 so the team is not tuning it live under demo pressure
5. Added a **live numeric risk-score requirement** to the dashboard so judges see the system reasoning during silent/SAFE states, not just during alerts
6. Added a **named integration partner requirement** for the M07–M09 prediction chain, which was a single point of failure in the original plan
7. Added Section 14, a one-page **Quick Reference Cheat Sheet** for use during the build

---

## Document Map

1. [Executive Council Verdict](#1-executive-council-verdict)
2. [Facial Recognition Council Decision](#2-facial-recognition-council-decision)
3. [Updated Product Definition](#3-updated-product-definition)
4. [Updated System Architecture](#4-updated-system-architecture)
5. [Feature Priority Matrix](#5-feature-priority-matrix)
6. [Module-by-Module Engineering Specification](#6-module-by-module-engineering-specification)
7. [Dashboard + Local Server Design](#7-dashboard--local-server-design)
8. [Hardware Plan + Procurement](#8-hardware-plan--procurement)
9. [Four-Person Work Breakdown](#9-four-person-work-breakdown)
10. [Simulation, Replay and Integration Testing](#10-simulation-replay-and-integration-testing)
11. [24-Hour Build Roadmap](#11-24-hour-build-roadmap)
12. [Demo Script](#12-demo-script)
13. [Research Positioning + References](#13-research-positioning--references)
14. [Quick Reference Cheat Sheet](#14-quick-reference-cheat-sheet)

---

## 1. Executive Council Verdict

Council structure used: five independent lenses (Contrarian, First Principles, Expansionist, Outsider, Executor), followed by cross-critique and a chairman-style synthesis.

| Lens | Verdict | Engineering Implication |
|---|---|---|
| Contrarian | The idea can collapse into a conventional smart obstacle detector if prediction is not visibly demonstrated. | Make trajectory intersection, CPA, uncertainty and changing risk the centerpiece. |
| First Principles | The real problem is not proximity; it is predicted intersection of future occupied spaces. | Represent user and obstacle motion as trajectories and ask whether/when their safety corridors overlap. |
| Expansionist | There is a path from warning device to local collision-risk engine. | Add risk field + safe corridor + dynamic-obstacle prediction instead of unrelated AI features. |
| Outsider | A judge may see "YOLO + Arduino + vibration" unless the behavior difference is obvious. | Design the demo around same distance/different trajectory → different alert. |
| Executor | Most of the upgraded logic is feasible if modules are decoupled and laptop compute is retained for the hackathon. | Use clean data contracts, simulation/replay and milestone gates. |
| **Chairman** | **KEEP THE CORE; STRENGTHEN THE PREDICTION/DECISION LAYER.** | Do not broaden into navigation, voice assistant, cloud AI or generic smart-cane features yet. |

### What the Source Proposal Already Gets Right

The source PDF already has the essential conceptual chain:
`webcam → object detection + optical flow → FOE/TTC/CPA → hazard decision → directional HMI`

with an Arduino/3-motor physical path as the next hardware iteration. The upgrade here is to turn those concepts into explicit modules with tracking, ego-motion compensation, risk state, safe-corridor selection, telemetry and replay.

### Council Scorecard

| Dimension | Current Concept | Upgraded Target | Reason |
|---|---|---|---|
| Problem clarity | Strong | Strong | Proximity versus collision-course reasoning is a clear distinction. |
| Technical novelty | Medium | High for hackathon demo | The value comes from combining tracking + ego-motion + TTC/CPA + intersection + risk policy. |
| Demo clarity | Medium | Very high | Force judges to see silent-vs-warning behavior at the same distance. |
| Feasibility | High | **Medium-High** | Do not move compute to embedded hardware during the first build. Ego-motion/optical-flow from a walking, chest-worn camera is harder than vehicle-mounted setups. |
| Product breadth | Medium | Medium+ | Add only features that reinforce collision intelligence. |
| Safety credibility | Medium | Higher | Explicit uncertainty/degraded state + deterministic fallback + testing. |

---

## 2. Facial Recognition Council Decision

**Council question:** "Should we add facial recognition so the phone dashboard can show a person's face or identity?"

> **Chairman Decision: DO NOT put facial recognition in the safety-critical path.** It can exist as an optional Social Assist module in a later milestone, preferably on-device/local, opt-in and limited to enrolled contacts. The safety system should identify an object as `Person #ID` and use motion/risk attributes; it should not need identity to decide whether a collision is likely.

| Council Lens | Position | Why |
|---|---|---|
| Contrarian | Reject as a core feature | It adds compute, latency, false-match risk and privacy questions without improving TTC/CPA. |
| First Principles | Not required to solve collision | A person's identity has no role in deciding whether trajectories intersect. |
| Expansionist | Useful as a separate accessibility layer | Known-contact recognition could support social interaction after the safety engine is stable. |
| Outsider | Dashboard face display looks impressive but can look like surveillance | Judges may challenge privacy and consent; it can distract from the USP. |
| Executor | Feasible later, expensive in engineering attention | Requires face detection, embeddings, enrollment, thresholding, unknown handling and test data. |

External research reinforces this caution. Studies of camera-based assistive technologies report privacy and social-acceptance concerns for bystanders and users, including concerns about AI mischaracterization and facial-recognition errors.

### Recommended Boundary

- **Safety mode**: no face image required; dashboard shows `Person #4`, class, trajectory, TTC, CPA, risk and direction.
- **Social Assist mode (P2)**: detect faces, compute embeddings, compare only against an enrolled local contact list, return `Known / Unknown / Uncertain`.
- Do **not** upload raw frames or biometric embeddings to a general cloud service. Keep processing local.
- For a live demo, use consenting team members and a clearly labeled "Social Assist" panel separate from the safety panel.

---

## 3. Updated Product Definition

### 3.1 Problem

The source proposal identifies a gap between raw proximity and actual collision risk: a user moving parallel to a wall can be near the wall without being on a collision course, while a moving object can approach the user from farther away but become dangerous because trajectories converge.

**Target population**: People with moderate-to-severe visual impairment who already use a cane or guide dog but have gaps around head/chest-level hazards and crowded or parallel-obstacle environments.

### 3.2 Product Statement

SpatialVector-HMI is a chest-worn, vision-first local safety system that tracks nearby objects, estimates camera/user motion, predicts future trajectory intersection, computes collision risk, selects a safer corridor and communicates that decision through directional haptics. A phone dashboard observes system state and logs evidence; it is **not** the primary safety channel.

### 3.3 USP

> *"We do not ask only what is nearby. We estimate whether the user and the obstacle are on a collision course, how soon the risk will materialize, and which local corridor is safer."*

### 3.4 Non-Goals

- ❌ Certified medical mobility-aid status
- ❌ Measured reduction in falls/injuries unless evidence is collected
- ❌ Facial identity, voice assistant, GPS navigation, or cloud analytics as a prerequisite for the safety pipeline
- ❌ YOLO confidence treated as collision probability
- ❌ Single frame as enough evidence for motion prediction
- ⚠️ Ground-level hazards (curbs, stairs, potholes) — Formerly a non-goal, now actively supported via M13 (Ground Hazard Detector), M14 (Walkable-Ground Estimator), and M15 (Navigation Decision Engine). A white cane remains recommended as secondary tactile confirmation.
- ❌ Unvalidated build on actual visually-impaired end users — sighted volunteers under blindfold/simulated-impairment conditions only until safety gates pass

### 3.5 Success Criteria

| Metric | Prototype Acceptance Target | How to Verify |
|---|---|---|
| End-to-end loop | Stable real-time loop with no persistent queue growth | Measure camera timestamps, inference time, queue depth |
| Parallel-wall suppression | No sustained warning while moving parallel under a predefined safe test path | 10+ repeated passes; count false warning events |
| Approach escalation | Risk rises as user approaches a static obstacle | Repeated runs at controlled walking speeds |
| Crossing-object detection | A crossing pedestrian transitions from safe/caution to warning before corridor intersection | Replay video with known ground-truth crossing times |
| Direction | Chosen haptic side matches minimum-risk corridor | Compare predicted safe corridor to scripted path |
| Repeatability | Same replay produces equivalent risk sequence within tolerance | Replay identical recorded telemetry |
| Failure behavior | Low confidence / sensor disagreement produces explicit degraded state | Inject dropped frames / unstable flow / missing IMU |

---

## 4. Updated System Architecture

The source PDF's architecture is reorganized into 12 modules so each unit can be implemented and verified independently. The webcam and Arduino remain non-negotiable.

### 4.1 Information Flow

| Boundary | Producer | Consumer | Contract |
|---|---|---|---|
| Sensing | Camera / IMU | M01 | timestamped raw sample; monotonic clock |
| Perception | M02 detector | M03 tracker | detection list: class, bbox, confidence, timestamp |
| Motion | M03 + M04 + IMU | M05/M06 | track history, flow vectors, angular motion |
| Prediction | M06 + M03 | M07 | relative motion + geometric state per track |
| Decision | M07 | M08 | TTC, CPA, intersection, miss distance |
| HMI | M08/M09 | Arduino | direction + urgency + pattern ID |
| Telemetry | M07/M08/M09 | phone | JSON/WebSocket message; no safety dependence |
| Evidence | all core modules | M12 | timestamped structured log |

> **Integration Rule**: Every module must be usable with recorded/simulated inputs. A module is not "done" merely because its live output looks correct once. It is done when its input contract, output contract, failure states and replay test pass.

---

## 5. Feature Priority Matrix

| Priority | Feature | Why | Dependencies | Demo Value | Build Status Gate |
|---|---|---|---|---|---|
| P0 | Frame capture + timestamps | Everything depends on deterministic timing | Webcam | Medium | Frames arrive at target FPS with bounded queue |
| P0 | YOLO object detection | Base perception | M01 | Medium | Stable boxes + confidence |
| P0 | Multi-object tracking | Motion requires identity persistence | M02 | High | Stable IDs over 3–10s sequences |
| P0 | Optical flow | Motion field + FOE foundation | M01 | High | Flow field stable on scripted motion |
| P0 | IMU ingestion | Separate camera/body motion from scene motion | IMU hardware | High | Timestamp alignment + rotation signal |
| P0 | Ego-motion compensation | Reduces false motion from user/camera rotation | M04 + IMU | Very high | Wall-turn test improves consistency |
| P0 | TTC + CPA | Turns motion into collision timing and miss distance | M03/M06 | Very high | Known scripted trajectories match expected trend |
| P0 | Trajectory intersection | Answers the actual product question | M06/M07 | Very high | Same distance/different direction yields different risk |
| P0 | Risk state machine | Prevents noisy alerts | M07 | Very high | SAFE/CAUTION/WARNING/DEGRADED transitions are stable |
| P0 | Safe corridor | Converts prediction into action | M08 | Very high | Correct lowest-risk side selected |
| P0 | Arduino haptic output | Makes device physically meaningful | M09 | Very high | Direction + pulse patterns verified |
| P0 | Logger + replay | Makes module testing repeatable | All | High | Replay produces reproducible output |
| P1 | Phone dashboard | Makes internal state understandable to judges/team | Telemetry gateway | High | Phone receives live state < target latency |
| P1 | Dynamic obstacle prediction | Pedestrian/cycle crossing behavior | Tracking + motion | Very high | Crossing test passes |
| P1 | Uncertainty / sensor disagreement | Prevents overconfident behavior | All motion signals | High | Degraded cases are visible |
| P1 | Scenario modes | Tune thresholds for indoor/outdoor demo conditions | Risk engine | Medium | Mode changes parameters cleanly |
| P2 | Free-space segmentation | More explicit safe-area reasoning | Camera + model | High | Adds safe corridor evidence |
| P2 | Face detection | Enabler for social assist, not safety | YOLO/face model | Medium | Runs asynchronously without slowing safety |
| P2 | Enrolled-contact face recognition | Optional social accessibility feature | Face detector + local store | Medium | Known/Unknown/Uncertain correctly handled |
| P3 | Embedded deployment | Productization beyond hackathon | Stable software | High | Runs on edge board at target FPS |
| P3 | Voice/navigation/cloud | Broadens product but weakens immediate thesis | Many dependencies | Low for current demo | Defer |

### 5.1 Recommended Implementation Order

1. P0 perception → tracking → motion → prediction → risk → haptics
2. P0 replay/testing harness **in parallel**, not at the end
3. P1 phone telemetry only after the safety core is stable
4. P2 face recognition only after the entire safety path is demonstrably stable
5. P3 embedded deployment only after laptop prototype is repeatable

---

## 6. Module-by-Module Engineering Specification

Each module follows the same contract: purpose → inputs → outputs → requirements → implementation → failure modes → standalone tests → integration test.

> **Rule**: Threshold values should be stored in configuration files, not hard-coded across modules.

---

### M01 — Frame Acquisition & Timebase [P0]

| Item | Specification |
|---|---|
| **Inputs** | Webcam frames; Frame timestamps; Configuration: resolution/FPS |
| **Outputs** | frame_id; image; t_capture; fps_estimate |
| **Key requirements** | Use one monotonic clock for camera, IMU and outgoing HMI timestamps. Bound the frame queue; drop stale frames rather than creating unbounded latency. Expose camera disconnect/reconnect state. Record actual FPS, not nominal FPS. |
| **Implementation** | OpenCV VideoCapture; fixed capture thread; ring buffer |
| **Failure modes** | Camera disconnect; Timestamp drift; Queue buildup |
| **Standalone tests** | T01: 60s capture, verify monotonically increasing IDs/timestamps. T02: block camera for 2s, verify graceful recovery. T03: stress CPU and verify queue stays bounded. |
| **Integration rule** | Feed M01 output into M02, M04 and M12. Ensure identical timestamps appear in replay. |

---

### M02 — Object Detection [P0]

| Item | Specification |
|---|---|
| **Inputs** | M01 frames; Model weights; Confidence threshold |
| **Outputs** | detection_id; class_id/name; bbox_xyxy; confidence; timestamp |
| **Key requirements** | Run lightweight detector. Keep confidence and class metadata. Filter only by application-configured classes; do not discard unknown objects silently. Separate detection confidence from risk. |
| **Implementation** | Ultralytics YOLO nano or ONNX Runtime deployment |
| **Failure modes** | Low light; Motion blur; Occlusion; False positives |
| **Standalone tests** | T04: known test clip; inspect precision/recall qualitatively. T05: ensure low-confidence boxes can still help recover a track if tracker policy allows. |
| **Integration rule** | M02 must work on recorded frames without M03. |

---

### M03 — Multi-Object Tracking [P0]

| Item | Specification |
|---|---|
| **Inputs** | M02 detections; Frame timestamps; Optional camera motion compensation |
| **Outputs** | track_id; class; bbox_history; center_history; estimated_image_velocity; track_age; track_confidence |
| **Key requirements** | Stable IDs matter more than fancy appearance features for the first build. Start with ByteTrack for a baseline; BoT-SORT is a strong option for moving-camera scenes because it supports camera-motion compensation and optional ReID. Keep at least N frames of history for motion fitting. |
| **Implementation** | Ultralytics tracking API; evaluate ByteTrack vs BoT-SORT on moving-camera clips |
| **Failure modes** | ID switches; Track loss; Occlusion; Duplicate IDs |
| **Standalone tests** | T06: person walks across frame; ID persists. T07: camera rotates; track does not explode. T08: two people cross; record ID-switch count. |
| **Integration rule** | M03 output feeds M06 and M07; M03 is not allowed to send haptic commands. |

---

### M04 — Optical Flow & FOE [P0]

| Item | Specification |
|---|---|
| **Inputs** | M01 consecutive frames; ROI/mask; Optional feature points |
| **Outputs** | flow_vectors; flow_quality; FOE_x/y; FOE_confidence |
| **Key requirements** | Use Lucas-Kanade or equivalent sparse flow for the first build. Track flow quality and reject obviously unstable estimates. Compute FOE as a geometric signal, not a standalone collision decision. |
| **Implementation** | OpenCV; robust point filtering; median/RANSAC-style outlier rejection as needed |
| **Failure modes** | Low texture; Motion blur; Large camera rotation; Dynamic foreground objects |
| **Standalone tests** | T09: forward walking scene should show coherent expansion. T10: parallel wall should not automatically imply collision. T11: rotate camera and verify quality degrades predictably rather than generating extreme false risk. |
| **Integration rule** | M04 feeds M05/M06; do not directly trigger alerts. |

---

### M05 — IMU Ingestion + Ego-Motion Compensation [P0]

| Item | Specification |
|---|---|
| **Inputs** | Gyroscope/IMU stream; M04 flow; Timestamp synchronization |
| **Outputs** | angular_velocity; orientation_estimate (if available); ego_rotation; corrected_flow; motion_quality |
| **Key requirements** | Use IMU to explain camera/body rotation. Synchronize IMU and video timestamps. Keep a degraded mode when IMU is absent or invalid. Do not claim full metric odometry from a 6-axis IMU alone. **Fix a conservative fallback caution threshold before hour 18 — do not tune this live under demo pressure.** |
| **Implementation** | Arduino or microcontroller IMU reader; serial/BLE transport to laptop; complementary/filtering as needed |
| **Failure modes** | Timestamp mismatch; Bias drift; Cable/serial drops; Body vibration |
| **Standalone tests** | T12: rotate camera while scene is static; corrected flow should show lower rotational component. T13: disable IMU and verify system enters degraded state, not silent failure. |
| **Integration rule** | M05 output becomes a prerequisite for high-confidence motion prediction. |

---

### M06 — Motion & Geometry [P0]

| Item | Specification |
|---|---|
| **Inputs** | M03 track histories; M04 FOE/flow; M05 corrected motion; Camera calibration parameters |
| **Outputs** | bearing; relative_image_velocity; motion_vector; FOE_containment; geometry_confidence |
| **Key requirements** | Represent object bearing relative to image/camera center. Normalize coordinates to 0..1 so resolution changes do not alter logic. Use FOE as one cue for whether an object is aligned with future travel direction. |
| **Implementation** | NumPy; camera intrinsics optional for the first prototype; keep geometry functions unit-testable |
| **Failure modes** | Camera FOV changes; Lens distortion; Track jitter |
| **Standalone tests** | T14: synthetic trajectories yield expected bearings. T15: perturb boxes with noise and ensure output remains bounded. |
| **Integration rule** | M06 must have no direct hardware dependencies. |

---

### M07 — Collision Prediction: TTC + CPA + Intersection [P0]

> ⭐ **This is the project's core intelligence module.**

| Item | Specification |
|---|---|
| **Inputs** | Track velocity; User motion estimate; Object motion estimate; Geometry state |
| **Outputs** | ttc_s; cpa_m_or_normalized; intersection_flag; miss_distance; prediction_confidence |
| **Key requirements** | TTC answers timing under an assumed intersection; CPA answers closest approach; intersection logic decides whether the path actually crosses. Do not display TTC as collision probability. Handle receding/parallel motion as valid safe outcomes. Use a conservative finite horizon. |
| **Implementation** | Vectorized NumPy math; explicit test vectors |
| **Failure modes** | Near-zero relative velocity; Noisy velocity; Occlusion; Non-linear pedestrian motion |
| **Standalone tests** | T16: approaching obstacle gives decreasing TTC. T17: parallel wall gives no intersection. T18: crossing pedestrian gives finite TTC + small CPA. T19: receding object gives no active collision. |
| **Integration rule** | No YOLO class should directly map to "danger". |

---

### M08 — Risk Engine & State Machine [P0]

| Item | Specification |
|---|---|
| **Inputs** | TTC; CPA; intersection; prediction confidence; object class/priority; user speed |
| **Outputs** | risk_score_0_1; risk_state; risk_reason; confidence; recommended_horizon |
| **Key requirements** | Use weighted/monotonic rules first; do not train a black-box risk model during the first build. Risk should increase when TTC decreases, miss distance shrinks and intersection confidence rises. Use hysteresis/debounce so warning does not flicker. Expose why risk changed. |
| **Implementation** | Pure Python/NumPy rules; configurable weights/thresholds |
| **Failure modes** | Oscillation around threshold; Single-frame spikes; Low-confidence overreaction |
| **Standalone tests** | T20: feed synthetic sequences and assert state transitions. T21: same distance, different trajectory → different risk. T22: add noise and verify state hysteresis. |
| **Integration rule** | M08 is the single source of truth for warning state. |

---

### M09 — Safe-Corridor Selector + Haptic Policy [P0]

| Item | Specification |
|---|---|
| **Inputs** | Per-track risk; Spatial position; Free corridor estimates; Risk state |
| **Outputs** | left_risk; center_risk; right_risk; safe_direction; urgency_level; haptic_pattern |
| **Key requirements** | Compute risk by corridor rather than choosing the closest object. Direction = lowest credible risk corridor. Urgency = rising risk/TTC trend, not distance alone. If all corridors are unsafe, issue a distinct "stop/critical" pattern. |
| **Implementation** | Policy module; three-zone discretization for prototype |
| **Failure modes** | Conflicting objects; All corridors blocked; No confidence |
| **Standalone tests** | T23: obstacle left only → right preferred. T24: obstacle center → side preferred. T25: all high → critical pattern. |
| **Integration rule** | M09 is the only module that decides which motor pattern to command. |

---

### M10 — Arduino Haptic Interface [P0]

| Item | Specification |
|---|---|
| **Inputs** | Serial/BLE command; Pattern ID; Intensity/duration |
| **Outputs** | ACK; device_state; motor_test_result |
| **Key requirements** | Use compact commands such as `D,L,3` or JSON-line messages. Never block the safety loop on an ACK. Include watchdog timeout so stale alerts do not remain active. |
| **Implementation** | Arduino; transistor/MOSFET drivers or suitable motor driver stage; 3 motors |
| **Failure modes** | Motor current draw; Loose wire; Serial loss; Stuck motor |
| **Standalone tests** | T26: direct command test for all three motors. T27: command timeout returns motors to safe idle. T28: 1000 command sequence with no deadlock. |
| **Integration rule** | Keep Arduino protocol documented in one file so M09 and M10 never guess the interface. |

---

### M11 — Local Telemetry Gateway + Phone Dashboard [P1]

| Item | Specification |
|---|---|
| **Inputs** | Risk state; tracks; TTC/CPA; system health; optional preview |
| **Outputs** | Dashboard JSON/WebSocket stream; connection health; event log |
| **Key requirements** | Phone is observational/control UI, not the safety decision maker. Prefer laptop/local edge server over cloud for the prototype. Send metadata by default; raw video should be opt-in. Dashboard should show risk explanation, not only a red/green light. |
| **Implementation** | FastAPI/Flask/WebSocket or similar lightweight local service; browser-based phone UI is sufficient |
| **Failure modes** | Wi-Fi dropout; Latency; Phone disconnect; Privacy leakage |
| **Standalone tests** | T29: phone reconnects without crashing core. T30: measure telemetry latency. T31: server down → Arduino safety still works. |
| **Integration rule** | This module must be removable without changing M07–M10. |

---

### M12 — Logger + Replay + Evaluation Harness [P0]

| Item | Specification |
|---|---|
| **Inputs** | Raw frames metadata; detections; tracks; prediction; risk; HMI commands; IMU |
| **Outputs** | timestamped session file; replay stream; metrics report |
| **Key requirements** | Use a single session_id and synchronized timestamps. Store enough state to reproduce the decision. Replay should bypass webcam and inject identical module inputs. Keep a deterministic synthetic scenario generator. |
| **Implementation** | JSONL/CSV for first prototype; OpenCV video file for frames |
| **Failure modes** | Clock drift; Missing fields; Version mismatch |
| **Standalone tests** | T32: replay 10 times; output sequence should match tolerance. T33: corrupt one record; parser reports clear error. T34: compare baseline and upgraded risk engine on same clip. |
| **Integration rule** | M12 is the team's integration safety net. |

---

## 7. Dashboard + Local Server Design

> The laptop is the local server/edge compute node. The phone connects over the same local Wi-Fi/hotspot. This keeps latency and privacy under control while still giving a live mobile interface.

| Data | Send to Phone? | Default Treatment |
|---|---|---|
| Risk state | Yes | SAFE / CAUTION / WARNING / CRITICAL / DEGRADED |
| TTC / CPA | Yes | Display for judge/debug mode |
| Track metadata | Yes | Person #4, bicycle #2, position, velocity |
| Haptic command | Yes | Show current motor/pattern |
| Raw camera stream | Optional | Keep local unless needed for demo; can send low-rate preview |
| Face image / embedding | **No** by default | Only optional Social Assist, preferably local, consented |
| Logs | Optional | Upload/export after session, not continuously to cloud |

### Recommended Dashboard Screens

1. **Live Safety**: camera preview, bounding boxes, FOE, trajectory arrows, TTC, CPA, risk score, three corridor risk bars. **Show the numeric risk score live even in SAFE/silent states**, so judges can see the system actively reasoning when no haptic alert fires.
2. **Haptic State**: Left/Center/Right motor pattern, urgency, last command, Arduino health.
3. **Prediction Inspector**: selected track with position, velocity vector, predicted path, intersection point and uncertainty.
4. **Test/Replay**: scenario name, expected result, actual result, pass/fail, session ID.
5. **Social Assist (P2)**: separate page; Known/Unknown/Uncertain contact result. Do not mix identity into the safety score.

### Example Telemetry Message

```json
{
  "session_id": "demo_014",
  "ts": 123456.78,
  "track_id": 4,
  "class": "person",
  "risk": 0.84,
  "state": "WARNING",
  "ttc_s": 1.7,
  "cpa": 0.31,
  "intersection": true,
  "safe_direction": "LEFT",
  "haptic": { "pattern": "LEFT_FAST", "urgency": 3 },
  "confidence": 0.78
}
```

---

## 8. Hardware Plan + Procurement

| Item | Status / Owner | Purpose | Action |
|---|---|---|---|
| Webcam | Get from HOD | Chest-mounted vision input | Request webcam + compatible cable/adapter |
| Arduino | Get from HOD | Haptic controller / interface | Request Uno or equivalent already available in lab |
| Vibration motors ×3 | Buy (~₹150–₹200) | Left / Center / Right tactile output | Ask HOD for approval; buy 3 matched motors if possible |
| Power bank | Prajwal | Portable power | Verify USB output stability |
| Gyroscope / IMU | Ask HOD | Rotation / ego-motion compensation | Request MPU6050 or available IMU; prioritize stable cable/connector |
| Chest harness | Already OK | Stable camera placement | Fix camera rigidly; mark reference orientation |
| Motor driver stage | Check lab stock / buy | Protect/control motor current | Use appropriate transistor/MOSFET driver or motor driver board. Do not drive motors directly from Arduino GPIO if current exceeds GPIO limits. |
| Jumper wires / breadboard / resistors | Lab stock / buy | Prototype interconnect | Keep spare wires and connectors |
| USB cable / extension | Lab stock / buy | Arduino/laptop link | Short, secure cable preferred for wearable prototype |

### 8.1 Physical Layout

- **Camera**: centered on chest, rigid, with a repeatable vertical and horizontal reference.
- **IMU**: mounted close to the camera body/harness so rotational measurements approximate camera motion.
- **Motors**: physically separated and labeled LEFT/CENTER/RIGHT; keep the pattern distinguishable through touch.
- Arduino and laptop/phone connectivity must not create loose cables around the wearer.

### 8.2 What NOT to Buy Yet

- VL53L1X/ToF sensor — explicitly left out of this baseline
- LiDAR / depth camera — not needed for the thesis prototype
- GPS — irrelevant to local collision prediction
- Extra displays/audio hardware — haptics are the primary HMI
- Multiple redundant sensors before the prediction pipeline is validated

---

## 9. Four-Person Work Breakdown

> Use role ownership, but require every module to have an integration partner. Nobody should build an isolated "feature demo" that cannot feed the common schema.

| Person | Primary Ownership | Secondary | Deliverables |
|---|---|---|---|
| Person 1 — Perception | M01 Frame Acquisition + M02 YOLO + M03 Tracking | M12 replay integration | Detector/tracker package, timestamps, track schema, recorded test clips |
| Person 2 — Motion | M04 Optical Flow/FOE + M05 IMU/Ego Motion + M06 Geometry | Calibration tools | Motion package, flow quality score, IMU reader, synchronized states |
| Person 3 — Prediction | M07 TTC/CPA/Intersection + M08 Risk Engine + M09 Safe Corridor | Synthetic simulation + named integration partner from hour 0 | Core prediction library, test vectors, state machine, corridor logic |
| Person 4 — HMI/Product | M10 Arduino + M11 Phone Dashboard + integration UI | M12 logging | Haptic firmware/protocol, local server, mobile web UI, health/status |

### 9.1 Shared Integration Contract

| Shared Object | Mandatory Fields |
|---|---|
| Frame | session_id, frame_id, timestamp, image |
| Detection | frame_id, class, bbox, confidence |
| Track | track_id, class, bbox, timestamp, history, image_velocity, confidence |
| MotionState | timestamp, ego_rotation, FOE, flow_quality, motion_quality |
| Prediction | track_id, TTC, CPA/miss_distance, intersection, prediction_confidence |
| RiskState | timestamp, global_risk, state, reason_codes, corridor_risks |
| HapticCommand | timestamp, direction, urgency, pattern_id, duration_ms |
| Telemetry | timestamp, system health, selected track, risk state, haptic state |

> **Team Integration Rule**: Use dataclasses/Pydantic models or typed dictionaries. Do not pass ad-hoc tuples whose meaning changes from person to person.

### 9.2 Integration Gates

| Gate | Condition |
|---|---|
| Gate A | M01→M02→M03 works on recorded video |
| Gate B | M03 + M04 + M05 produce stable motion signals |
| Gate C | M07 passes synthetic trajectory tests with no camera/model dependencies |
| Gate D | M08/M09 consume synthetic prediction data and drive Arduino patterns |
| Gate E | M01→M09 runs live with logger |
| Gate F | M11 mirrors live state without becoming a dependency |
| Gate G | Full demo scenarios pass twice from clean startup |

---

## 10. Simulation, Replay and Integration Testing

> The team should not wait for the full wearable to test whether the collision logic works. Generate or replay known motion states first, then connect real sensors one module at a time.

### 10.1 Synthetic Trajectory Scenarios

| Scenario | User Vector | Obstacle Vector | Expected Outcome |
|---|---|---|---|
| S1 Parallel wall | → | → same direction / offset | SAFE |
| S2 Head-on | → | stationary | WARNING/CRITICAL as TTC falls |
| S3 Crossing pedestrian | → | ↓ across corridor | WARNING before intersection |
| S4 Safe pass | → | ↘ away from corridor | SAFE despite low current distance |
| S5 Receding object | → | → faster away | SAFE / no active collision |
| S6 All corridors blocked | → | multiple obstacles | CRITICAL / stop pattern |
| S7 Noisy measurement | → | random perturbation | Stable state due to hysteresis/confidence |
| S8 Sensor disagreement | → | prediction inconsistent with IMU/flow | DEGRADED, not silent false certainty |

### 10.2 Replay Test Flow

1. Record a session: webcam video + IMU + timestamps + intermediate outputs
2. Save a versioned session schema
3. Replay the session without opening the camera
4. Compare M07/M08/M09 outputs against the recorded baseline
5. Change one module (e.g., risk thresholds) and rerun the same session
6. Produce a simple before/after metric table

### 10.3 Fault-Injection Tests

| Injected Fault | Expected Behavior |
|---|---|
| Camera frame drop 10% | Keep bounded latency; lower confidence if motion becomes unreliable |
| IMU disconnect | Enter DEGRADED; continue only with conservative motion policy |
| Tracker loses object | Track confidence falls; no spurious high-risk command from stale state |
| Phone disconnect | Safety path unchanged; Arduino continues |
| Arduino disconnect | Dashboard shows HMI offline; core logs decision; no fake "haptic delivered" claim |
| Optical flow quality collapse | Prediction confidence reduced; avoid overconfident warning |

### 10.4 Module Test Evidence Sheet

| Module | Test ID | Input Fixture | Expected | Actual | Pass/Fail |
|---|---|---|---|---|---|
| M03 Tracking | T06 | Crossing pedestrian clip | Persistent ID | — | — |
| M05 IMU | T12 | Static scene + camera rotation | Reduced rotational flow | — | — |
| M07 Prediction | T18 | Crossing vectors | Finite TTC + low CPA | — | — |
| M08 Risk | T21 | Same distance / different trajectory | Different risk | — | — |
| M09 HMI | T24 | Center risk high | Side selected | — | — |
| M10 Arduino | T27 | No command for > timeout | Motors idle | — | — |
| M11 Dashboard | T31 | Server stopped | Safety still runs | — | — |
| M12 Replay | T32 | Same session 10× | Equivalent outputs | — | — |

---

## 11. 24-Hour Build Roadmap

> The goal is to have a working vertical slice early, then improve intelligence and physical HMI. **Do not spend the first half of the hackathon polishing the dashboard.**

| Time | Person 1 | Person 2 | Person 3 | Person 4 | Milestone |
|---|---|---|---|---|---|
| 0–2h | Camera + frame schema | IMU wiring + readout | Synthetic trajectory generator | Arduino motor test | All teams have executable smoke tests |
| 2–4h | YOLO baseline | Optical flow baseline | TTC/CPA unit tests | Serial haptic protocol | P0 foundations pass |
| 4–6h | Tracking | FOE + flow quality | Intersection logic | Risk state + haptic mapping | Prediction works on synthetic data |
| 6–8h | Tracking tuning | IMU sync | M07 integration | Arduino physical test | Synthetic → haptic end-to-end |
| 8–10h | Video + tracking recording | Ego-motion compensation | M08 risk engine | Phone local server skeleton | Live laptop demo core |
| 10–12h | Fix ID/box jitter | Motion edge cases | Safe corridor | Dashboard live cards | Scenario 1 + 2 pass |
| 12–15h | Optimization | Calibration | Dynamic crossing scenario | Haptic pattern tuning | Scenario 3 + 4 pass |
| 15–18h | Logging/replay integration | Fault injection | Threshold tuning | Dashboard polish | Repeatable replay |
| 18–21h | End-to-end regression | Physical harness test | Evaluation metrics | Demo recording | Full system stable |
| 21–24h | Bug fix only | Bug fix only | Bug fix only | Demo UI only | ⛔ No new core features |

### 11.1 Feature Freeze Policy

> **Hard Rule**: At T-3 hours, freeze new features. After that point, only bugs, calibration, reliability, documentation and demo polish are allowed. Face recognition is not allowed into the final 3-hour window unless every P0/P1 feature already passes.

---

## 12. Demo Script

> The demo should prove the thesis rather than recite the architecture.

| Scene | Physical Action | Screen Evidence | Haptic Behavior | Judge Takeaway |
|---|---|---|---|---|
| 1. Baseline | Walk in open space | Risk low; corridors green | Silent | System does not constantly alert |
| 2. Parallel wall | Walk beside wall at close but safe offset | Wall detected; trajectory parallel; TTC not active/intersection false | Silent or low caution | Nearby is not automatically dangerous |
| 3. Turn toward wall | Rotate body toward obstacle | FOE/bearing changes; TTC appears; risk rises | Center pulses faster | Risk changes because trajectory changes |
| 4. Crossing person | Person walks across future path | Track ID + velocity + predicted path + CPA/TTC | Side motor warns | Dynamic obstacles are predicted, not just detected |
| 5. Safe passing person | Person passes outside corridor | CPA remains safe | No warning | Same distance can produce different decisions |
| 6. Sensor degradation | Briefly interrupt IMU or create low-flow scene | Confidence drops; DEGRADED state | Distinct conservative pattern or silence | System exposes uncertainty rather than pretending certainty |

### 12.1 What NOT to Say to Judges

| ❌ Don't Say | ✅ Say Instead |
|---|---|
| "It prevents accidents." | "Prototype predicts collision risk; not certified." |
| "95% accurate." | Only say this with measured test data and a defined dataset. |
| "Facial recognition makes it smarter." | "It is a separate optional Social Assist capability." |
| "Everything is on the server." | "Local edge server/laptop during prototype; phone is a client." |
| "TTC means probability of collision." | "TTC combined with intersection/CPA/confidence determines risk." |

---

## 13. Research Positioning + References

The updated design should be presented as an engineering contribution built around a specific reasoning problem, not as a claim that obstacle detection itself is novel.

### 13.1 Relevant External Evidence

| Reference | Relevance |
|---|---|
| [**Pedestrian Detection with Wearable Cameras for the Blind: A Two-way Perspective**](https://pmc.ncbi.nlm.nih.gov/articles/PMC7423406/) — Lee, Sato, Asakawa et al., CHI 2020 | Wearable camera privacy and social acceptance; highlights accessibility value and privacy/social-acceptance tensions of always-on cameras. Directly informs our design decisions (no raw video to cloud, no identity in safety path). |
| [Shared Privacy Concerns — Microsoft Research](https://www.microsoft.com/en-us/research/?p=1121952) | A study of visually impaired wearers and sighted bystanders found shared concerns about AI fallibility and mischaracterization, including facial-recognition errors. |
| [Assistive IoT Device with Face Recognition](https://doi.org/10.1080/17483107.2025.2582033) | A recent assistive-device study combines real-time object detection, facial recognition and obstacle sensing, demonstrating that this feature combination already exists in research — reinforces our differentiation via prediction intelligence. |
| [SmartCane — IIT Delhi](https://assistech.iitd.ernet.in/smartcane.php) | IIT Delhi's SmartCane uses ultrasonic sensing to detect knee-to-head-height obstacles and communicates information through vibratory patterns — demonstrates that obstacle sensing + tactile HMI is an established assistive direction. |
| [WeWALK Obstacle Detection](https://support.wewalk.io/en/article/customizing-obstacle-detection-settings-through-the-smart-cane-2/) | WeWALK supports adjustable obstacle-detection ranges and configurable sound/vibration feedback, reinforcing that "sensor + vibration" alone is not sufficient differentiation. |
| [Ultralytics Tracking Docs](https://docs.ultralytics.com/modes/track) | Currently documents ByteTrack, BoT-SORT and other trackers; its guidance specifically recommends BoT-SORT for moving-camera footage because it supports camera-motion compensation and optional ReID. |
| [Raspberry Pi AI HAT+](https://www.raspberrypi.com/documentation/accessories/ai-hat-plus.html) | Raspberry Pi's AI HAT+ family provides 13/26 TOPS variants for hardware-accelerated local vision workloads — useful for a future edge-deployment phase. |

### 13.2 Competitive Positioning

| Existing Direction | What It Already Does | SpatialVector Differentiation |
|---|---|---|
| Traditional/smart cane | Obstacle/proximity detection + vibration | Predictive trajectory intersection and risk, not proximity only |
| Wearable AI vision | Object/person understanding | Prioritize temporal motion and collision geometry as the core safety decision |
| Multi-feature assistive devices | Object detection + face recognition + smart feedback | Do not use identity as the core; demonstrate predictive collision intelligence |
| LiDAR/dual-LiDAR wearables | Spatial awareness + hazard alerts | Vision-first, low-cost prototype; prediction logic remains the research/engineering story |

---

## 14. Quick Reference Cheat Sheet

> Print or pin this page during the build.

### Shared Data Contract Fields (full detail: Section 9.1)

- **Frame** — session_id, frame_id, timestamp, image
- **Detection** — frame_id, class, bbox, confidence
- **Track** — track_id, class, bbox, timestamp, history, image_velocity, confidence
- **MotionState** — timestamp, ego_rotation, FOE, flow_quality, motion_quality
- **Prediction** — track_id, TTC, CPA/miss_distance, intersection, prediction_confidence
- **RiskState** — timestamp, global_risk, state, reason_codes, corridor_risks
- **HapticCommand** — timestamp, direction, urgency, pattern_id, duration_ms
- **Telemetry** — timestamp, system health, selected track, risk state, haptic state

### Integration Gates (full detail: Section 9.2)

- **Gate A** — M01→M02→M03 works on recorded video
- **Gate B** — M03 + M04 + M05 produce stable motion signals
- **Gate C** — M07 passes synthetic trajectory tests, no camera/model dependency
- **Gate D** — M08/M09 consume synthetic prediction data and drive Arduino patterns
- **Gate E** — M01→M09 runs live with the logger attached
- **Gate F** — M11 mirrors live state without becoming a dependency
- **Gate G** — full demo scenarios pass twice from a clean startup

### Hour-by-Hour Milestones (full detail: Section 11)

- **0–2h** — executable smoke tests across all four roles
- **2–4h** — P0 foundations pass
- **4–6h** — prediction works on synthetic data
- **6–8h** — synthetic → physical haptic end-to-end
- **8–10h** — live laptop demo core running
- **10–12h** — demo scenarios 1 and 2 pass
- **12–15h** — demo scenarios 3 and 4 pass
- **15–18h** — replay is repeatable
- **18–21h** — full system stable end-to-end
- **21–24h** — ⛔ feature freeze: bug fixes and demo polish only

### If Something Breaks

- **M04/M05** (optical flow + ego-motion) is the module most likely to run over budget — a walking, chest-worn camera is noisier than a vehicle-mounted one. If it is not stable by hour 18, switch to the pre-agreed conservative fallback threshold rather than tuning it live.
- **M07–M09** (prediction, risk, corridor) is the module everything else depends on. It has a named integration partner from hour 0 — use them the moment the primary owner is blocked, do not wait.
- Any module disconnect (phone, Arduino, IMU) should enter DEGRADED and say so on the dashboard — verify this explicitly before the demo, not during it.
- Do not run an unvalidated build on an actual visually-impaired user — sighted, blindfolded volunteers only until the safety path clears its integration gates.

---

## 15. Ground Hazards, Freespace, Evaluation Harness & Monocular Limitations

### 15.1 Architecture Extension (M13, M14, M15)
To address the critical safety requirement that "WALK FORWARD" must never be commanded without positive physical evidence of walkable ground:
- **M13 (GroundHazardDetector)**: Detects potholes and road depressions. Operates on an honest-fallback principle: when a trained model (`pothole_yolov8.pt`) is loaded, it serves as the primary detector; when no model is found, the system displays `POTHOLE MODEL NOT LOADED`, disables pothole alerts, and only runs unverified anomaly detection behind an explicit flag. All candidates are strictly gated against the ground plane (`y >= roi_y_start`) and all detected YOLO objects.
- **M14 (FreeSpaceEstimator)**: Evaluates per-corridor ground validity through multiple independent cues:
  1. *Frame-level validity*: Laplacian variance (blur), mean intensity (under/over-exposure), and texture variance.
  2. *Gradient jump detector*: Catches single-row steps and windowed diagonal jumps (table edges, ledges, table corners).
  3. *Unbroken bottom-up continuity scan*: Verifies that ground begins at the user's feet (bottom of the frame) and extends unbroken upward.
  4. *Unexplained dark blob gate*: Shadows and dark objects that are not recognized ground are flagged as UNKNOWN.
  5. *YOLO bounding-box obstruction*: Any tracked or raw object overlapping the corridor immediately forces `BLOCKED`.
- **M15 (NavigationDecisionEngine)**: The single source of truth for navigation commands. Requires multi-cue agreement, consecutive confirmation frames, and clear path verification before issuing `WALK FORWARD`.

### 15.2 Ground-Truth Evaluation Harness (`scripts/evaluate.py`)
To prevent threshold tuning by eye and ensure safety invariants:
- **Test Fixtures (`tests/fixtures/real/`)**: 17 benchmark scenarios covering table edges, corners, desk close-ups, blank walls, stairs/ledges, dark rooms, covered lenses, clear corridors, outdoor paths, and potholes.
- **Automated Verification**: Evaluates false walk-forward rate (must be 0.00% across all negative fixtures), precision/recall, and latency.

### 15.3 Known Physical Limitations of Monocular Vision
A single 2D camera inherently loses depth information:
1. *Extreme shadows vs. holes*: Deep optical shadows under bright sun cannot always be distinguished from physical depressions without stereo disparity or active depth sensing.
2. *Drop-offs with matching textures*: Downward stairs made of identical material to the landing may lack edge contrast from certain vantage points.
3. *Body tilt and camera pitch*: Heavy torso swaying during rapid walking shifts the ground ROI unless compensated by IMU pitch.
For these reasons, the primary rule is **Fail Safe**: if ground cannot be confirmed with high confidence, the system commands `CAUTION` or `STOP`, never `WALK FORWARD`.

---

## Final Build Checklist

- [ ] Architecture reviewed and all P0 modules assigned
- [ ] Webcam + Arduino acquired from HOD
- [ ] Power bank assigned to Prajwal
- [ ] Vibration motors purchase/approval routed through HOD; budget target ₹150–₹200 noted
- [ ] Gyroscope/IMU requested from HOD
- [ ] Chest harness ready and camera mounting orientation marked
- [ ] No VL53L1X in baseline
- [ ] M03 tracking tested on moving-camera footage
- [ ] M05 IMU timestamps aligned with camera timestamps
- [ ] M07 synthetic TTC/CPA/intersection tests pass
- [ ] M08 same-distance/different-trajectory test passes
- [ ] M09 left/center/right safe corridor tested
- [ ] M10 haptic watchdog tested
- [ ] M12 replay can reproduce decisions
- [ ] M11 phone dashboard works locally and does not block safety
- [ ] Face recognition excluded from safety path; only P2 Social Assist after P0/P1
- [ ] Four scripted demo scenarios recorded and repeatable
- [ ] No unsupported claims about accuracy, safety or clinical outcomes

---

*Primary project source: NIRMAAN 2026 proposal, "Real-Time Predictive Mobility Assistance", pages 2–6.*

