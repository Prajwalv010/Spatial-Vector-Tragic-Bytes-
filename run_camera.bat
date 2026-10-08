@echo off
title SpatialVector-HMI — Camera Prediction Engine
cd /d "%~dp0"
echo ======================================================================
echo    SpatialVector-HMI -- Camera Perception & Prediction Engine Viewer
echo ======================================================================
echo.
echo Starting direct camera prediction visualizer...
echo Controls:
echo   [SPACE]     Pause/Resume playback
echo   [H]         Toggle AR HUD overlays
echo   [T]         Toggle Technical Telemetry
echo   [1] - [5]   Switch to Benchmark Scenes S1-S5
echo   [C]         Switch back to Live Camera
echo   [Q] / [ESC] Quit
echo.
python run_camera_prediction_viewer.py %*
pause
