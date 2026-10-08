"""Quick utility to set or view the active camera/stream source.

Usage:
  python scripts/set_camera_source.py "https://vdo.ninja/?view=myroom"
  python scripts/set_camera_source.py 0
  python scripts/set_camera_source.py                  # interactive prompt
"""

import sys
from pathlib import Path

# Add repository root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from spatialvector.perception.source_resolver import (
    resolve_camera_source,
    write_camera_source_file,
    read_camera_source_file,
    get_camera_source_file_path,
)


def main():
    if len(sys.argv) > 1:
        new_source = sys.argv[1].strip()
        path = write_camera_source_file(new_source)
        print(f"[OK] Camera source updated in {path}:")
        print(f"     -> {new_source}")
    else:
        current = read_camera_source_file()
        print("=" * 65)
        print("SpatialVector-HMI — Set Camera Source")
        print("=" * 65)
        if current:
            print(f"Current saved source: {current}")
        else:
            print("No saved source in camera_source.txt (currently using default config/webcam)")
        print("\nTip: In VDO.Ninja, use a fixed name on your phone to never change links again:")
        print("     Phone (broadcaster): https://vdo.ninja/?push=kshitizcam")
        print("     Desktop (viewer):    https://vdo.ninja/?view=kshitizcam")
        print("-" * 65)
        try:
            val = input("Enter new VDO.Ninja URL or device index (or press Enter to keep): ").strip()
            if val:
                path = write_camera_source_file(val)
                print(f"[OK] Saved to {path.name}: {val}")
            else:
                print("No change made.")
        except (KeyboardInterrupt, EOFError):
            print("\nAborted.")


if __name__ == "__main__":
    main()
