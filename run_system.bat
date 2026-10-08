@echo off
title SpatialVector-HMI Master Launcher
echo ======================================================================
echo           SpatialVector-HMI -- Activating Full System
echo ======================================================================
echo [*] Starting Web Dashboard, WebSocket Telemetry, and Decision Pipeline...
echo [*] Dashboard will open on http://localhost:8081/
echo.
python run.py %*
pause
