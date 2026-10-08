@echo off
REM SpatialVector-HMI — Gate G Cold-Start Verification Batch Runner
cd /d "%~dp0\.."
echo Running Gate G Cold-Start Verification Suite...
python demo\run_gate_g.py
pause
