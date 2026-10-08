"""SpatialVector-HMI — System Launcher (Backwards Compatibility Wrapper)

Canonical Entrypoint for Demo Day: python run.py (or run.bat / run_system.bat)
All execution logic is centralized in run.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from run import main, build_waiting_telemetry_message

__all__ = ["main", "build_waiting_telemetry_message"]

if __name__ == "__main__":
    main()
