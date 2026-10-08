# SpatialVector-HMI — Ground Hazard & Pothole Model Card

## Overview
This directory contains configuration, verification scripts, and model specifications for M13 Ground Hazard (Pothole & Surface Damage) Detection.

To keep the repository lightweight and adhere to edge-deployment best practices, large `.pt` model weights are not committed to git. Instead, weights are loaded locally from the user's workspace or downloaded using the verified download script.

## Supported Models
- **`pothole_yolov8.pt`**: Fine-tuned YOLOv8 nano detector trained specifically for road depression and pothole detection.
  - Architecture: YOLOv8n (nano)
  - Input resolution: 640x640
  - Precision: FP32 / FP16
  - Expected placement: `pothole_yolov8.pt` in repository root or `models/pothole_yolov8.pt`

## Honest Fallback Architecture (Safety-Critical Invariant)
1. **When a trained model is loaded**:
   - Primary detections come from `pothole_yolov8.pt`.
   - Bounding boxes are filtered by confidence (`conf >= 0.45`), ground ROI boundaries (`y >= roi_y_start`), and gated against all tracked objects (persons, backpacks, chairs, tables, laptops).
2. **When no model is loaded**:
   - The system displays `POTHOLE MODEL NOT LOADED` in HUD telemetry and overlay.
   - Pothole voice alerts are disabled by default.
   - The unverified heuristic can only run behind an explicit config flag (`allow_heuristic=True`), where it is labelled `possible surface anomaly (unverified)` and never drives high-urgency guidance decisions.

## How to Download / Verify
Run the automated verification script:
```bash
python models/download_verify_model.py
```
Or train a customized model using:
```bash
python scripts/train_pothole_detector.py --data path/to/dataset/data.yaml --epochs 30
```
