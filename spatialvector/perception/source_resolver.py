"""Camera source resolution utility for SpatialVector-HMI.

Resolves video/stream source with high flexibility so users don't have to
retype long or ephemeral VDO.Ninja links in CLI commands each time.

Resolution Priority:
1. Explicit CLI argument (`--source <url_or_idx>`) if provided and not "prompt"
2. Interactive prompt (`--prompt` or `--source prompt`)
3. `camera_source.txt` file in project root or `spatialvector/config/camera_source.txt`
4. Environment variable `SPATIALVECTOR_CAMERA_URL` or `CAMERA_URL`
5. Config file `spatialvector/config/default.yaml` (`camera.stream_url`)
6. Fallback: `camera.device_index` in config (default: 0)
"""

import os
from pathlib import Path
import sys
from typing import Tuple, Union
import yaml


def get_project_root() -> Path:
    """Finds the Nirman-Hackathon project root directory."""
    current = Path(__file__).resolve()
    for parent in [current] + list(current.parents):
        if (parent / "spatialvector").exists() and (parent / "scripts").exists():
            return parent
    return Path.cwd()


def get_camera_source_file_path() -> Path:
    """Returns the primary path to the camera source file in project root."""
    return get_project_root() / "camera_source.txt"


def read_camera_source_file() -> Union[str, None]:
    """Reads camera source from camera_source.txt if it exists."""
    root = get_project_root()
    candidate_paths = [
        root / "camera_source.txt",
        root / "spatialvector" / "config" / "camera_source.txt",
    ]
    for path in candidate_paths:
        if path.exists():
            try:
                content = path.read_text(encoding="utf-8").strip()
                lines = [
                    line.strip()
                    for line in content.splitlines()
                    if line.strip() and not line.strip().startswith("#")
                ]
                if lines:
                    return lines[0]
            except Exception:
                pass
    return None


def write_camera_source_file(url_or_index: Union[str, int]) -> Path:
    """Saves the camera source to camera_source.txt in project root."""
    path = get_camera_source_file_path()
    content = (
        "# SpatialVector-HMI Camera Source Configuration\n"
        "# Paste your VDO.Ninja URL, stream URL, or webcam index below.\n"
        "# Tip: To make VDO.Ninja permanent, use ?push=myroom on your phone and ?view=myroom here.\n"
        f"{str(url_or_index).strip()}\n"
    )
    path.write_text(content, encoding="utf-8")
    return path


def resolve_camera_source(
    cli_source: Union[str, int, None] = None,
    interactive: bool = False,
    config_path: Union[str, Path, None] = None,
) -> Tuple[Union[str, int], str]:
    """Resolves the video source and returns (source_value, source_origin_description).

    Returns:
        (source, description): where source is either an int (device index) or str (URL/file path).
    """
    root = get_project_root()

    # Load default.yaml if available
    config_stream_url = None
    config_device_index = 0
    cfg_file = Path(config_path) if config_path else root / "spatialvector" / "config" / "default.yaml"
    if cfg_file.exists():
        try:
            with open(cfg_file, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
                cam_cfg = cfg.get("camera", {})
                config_stream_url = cam_cfg.get("stream_url")
                config_device_index = cam_cfg.get("device_index", 0)
        except Exception:
            pass

    file_source = read_camera_source_file()
    env_source = os.environ.get("SPATIALVECTOR_CAMERA_URL") or os.environ.get("CAMERA_URL")

    # Interactive prompt mode
    if interactive or (isinstance(cli_source, str) and cli_source.strip().lower() == "prompt"):
        default_val = file_source or config_stream_url or env_source or str(config_device_index)
        print("\n" + "=" * 65)
        print("SpatialVector-HMI — Camera Source Configuration")
        print("=" * 65)
        print(f"Current default: {default_val}")
        print("Tip: If using VDO.Ninja, use ?push=NAME on phone and ?view=NAME here for a fixed link.")
        try:
            user_input = input(f"Enter new URL/index (press Enter to keep '{default_val}'): ").strip()
        except (EOFError, KeyboardInterrupt):
            user_input = ""

        chosen = user_input if user_input else default_val
        write_camera_source_file(chosen)
        val = int(chosen) if chosen.isdigit() else chosen
        return val, f"interactive prompt (saved to {get_camera_source_file_path().name})"

    # 1. Explicit CLI argument (not None and not empty and not the default "0" if other source configured)
    if cli_source is not None and str(cli_source).strip() != "":
        s = str(cli_source).strip()
        # If user passed explicit CLI source, use it directly
        val = int(s) if s.isdigit() else s
        return val, "command-line argument (--source)"

    # 2. camera_source.txt in project root
    if file_source:
        val = int(file_source) if file_source.isdigit() else file_source
        return val, f"camera_source.txt ({get_camera_source_file_path().name})"

    # 3. Environment variable
    if env_source:
        s = env_source.strip()
        val = int(s) if s.isdigit() else s
        return val, "environment variable (SPATIALVECTOR_CAMERA_URL)"

    # 4. default.yaml stream_url
    if config_stream_url:
        s = str(config_stream_url).strip()
        val = int(s) if s.isdigit() else s
        return val, "config (spatialvector/config/default.yaml: camera.stream_url)"

    # 5. Default webcam index
    return int(config_device_index), "default webcam index (camera.device_index)"
