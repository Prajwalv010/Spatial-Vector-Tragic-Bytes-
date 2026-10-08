"""M12 — Session Logger (Safety-Net Recording Harness).

Records complete, synchronized timelines for every module's output during
live or synthetic runs in structured JSONL format.

Rules:
1. One session_id, synchronized timestamps using time.monotonic().
2. Structured header record with schema_version: 1.
3. Records enough state to reproduce decisions offline deterministically.
4. Non-blocking buffered writes so disk I/O does not degrade frame rates.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
import json
import logging
from pathlib import Path
import queue
import threading
import time
from typing import Any, Dict, Optional, Union

from spatialvector.hmi.schemas import SessionRecord

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1


def _sanitize_payload(obj: Any) -> Any:
    """Recursively converts dataclasses, numpy types, and tuples into serializable types."""
    if is_dataclass(obj):
        return _sanitize_payload(asdict(obj))
    if isinstance(obj, dict):
        return {str(k): _sanitize_payload(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_payload(x) for x in obj]
    if hasattr(obj, "item"):  # numpy scalars
        return obj.item()
    if hasattr(obj, "tolist"):  # numpy arrays
        return obj.tolist()
    return obj


class SessionLogger:
    """Buffered, non-blocking session recorder writing JSONL records."""

    def __init__(
        self,
        session_id: Optional[str] = None,
        base_dir: Union[str, Path] = "sessions",
        config_snapshot: Optional[Dict[str, Any]] = None,
        buffer_size: int = 1000,
    ):
        self.session_id = session_id or f"session_{int(time.time())}"
        self.base_dir = Path(base_dir)
        self.session_dir = self.base_dir / self.session_id
        self.session_file = self.session_dir / "session.jsonl"
        self.schema_version = SCHEMA_VERSION

        self.session_dir.mkdir(parents=True, exist_ok=True)

        self._queue: queue.Queue[Optional[str]] = queue.Queue(maxsize=buffer_size)
        self._stop_event = threading.Event()
        self._writer_thread = threading.Thread(target=self._writer_loop, daemon=True, name="SessionWriter")
        self._writer_thread.start()

        # Write session header
        header_payload = {
            "schema_version": self.schema_version,
            "session_id": self.session_id,
            "created_at": time.time(),
            "created_monotonic": time.monotonic(),
            "config": _sanitize_payload(config_snapshot or {}),
        }
        self.log(SessionRecord(
            session_id=self.session_id,
            record_type="header",
            ts=time.monotonic(),
            frame_id=None,
            payload=header_payload,
        ))

        logger.info(f"[M12] SessionLogger recording to {self.session_file}")

    def log(self, record: SessionRecord) -> None:
        """Enqueues a SessionRecord for non-blocking write."""
        if self._stop_event.is_set():
            return
        payload_sanitized = _sanitize_payload(record.payload)
        record_dict = {
            "session_id": record.session_id,
            "record_type": record.record_type,
            "ts": record.ts,
            "frame_id": record.frame_id,
            "payload": payload_sanitized,
        }
        json_line = json.dumps(record_dict) + "\n"
        try:
            self._queue.put_nowait(json_line)
        except queue.Full:
            logger.warning("[M12] Logger buffer full — dropping record")

    def log_event(
        self,
        record_type: str,
        payload: Any,
        frame_id: Optional[int] = None,
        ts: Optional[float] = None,
    ) -> None:
        """Convenience method for logging arbitrary module outputs."""
        self.log(SessionRecord(
            session_id=self.session_id,
            record_type=record_type,
            ts=ts if ts is not None else time.monotonic(),
            frame_id=frame_id,
            payload=_sanitize_payload(payload),
        ))

    def close(self) -> None:
        """Flushes remaining records and closes file cleanly."""
        if self._stop_event.is_set():
            return
        self._stop_event.set()
        try:
            self._queue.put_nowait(None)  # Sentinel to terminate writer loop
        except queue.Full:
            pass
        if self._writer_thread and self._writer_thread.is_alive():
            self._writer_thread.join(timeout=2.0)
        logger.info(f"[M12] SessionLogger closed. File size: {self.session_file.stat().st_size if self.session_file.exists() else 0} bytes")

    def _writer_loop(self):
        with open(self.session_file, "a", encoding="utf-8") as f:
            while True:
                try:
                    item = self._queue.get(timeout=0.2)
                except queue.Empty:
                    if self._stop_event.is_set():
                        break
                    continue

                if item is None:
                    break

                f.write(item)
                f.flush()

            # Drain any remaining records before exiting
            while not self._queue.empty():
                try:
                    item = self._queue.get_nowait()
                    if item:
                        f.write(item)
                except queue.Empty:
                    break
            f.flush()


def list_sessions(base_dir: Union[str, Path] = "sessions") -> list[dict]:
    """Lists all available recorded sessions with metadata."""
    base = Path(base_dir)
    if not base.exists():
        return []

    sessions = []
    for item in base.iterdir():
        if item.is_dir():
            jsonl_file = item / "session.jsonl"
            if jsonl_file.exists():
                size = jsonl_file.stat().st_size
                created = jsonl_file.stat().st_mtime
                header = {}
                try:
                    with open(jsonl_file, "r", encoding="utf-8") as f:
                        first_line = f.readline()
                        if first_line:
                            header = json.loads(first_line)
                except Exception:
                    pass

                sessions.append({
                    "session_id": item.name,
                    "file_path": str(jsonl_file),
                    "size_bytes": size,
                    "modified_time": created,
                    "schema_version": header.get("schema_version", SCHEMA_VERSION),
                    "created_at": header.get("created_at"),
                })

    sessions.sort(key=lambda s: s["modified_time"], reverse=True)
    return sessions


def export_session_json(session_id: str, base_dir: Union[str, Path] = "sessions") -> list[dict]:
    """Reads a complete session JSONL file and returns it as a list of dictionaries."""
    session_file = Path(base_dir) / session_id / "session.jsonl"
    if not session_file.exists():
        raise FileNotFoundError(f"Session '{session_id}' not found at {session_file}")

    records = []
    with open(session_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def export_session_csv(session_id: str, base_dir: Union[str, Path] = "sessions") -> str:
    """Exports session timeline records into flattened CSV format.

    Flattening Decisions (Explicitly Documented):
    ---------------------------------------------
    1. Base columns: 'ts', 'frame_id', 'record_type'.
    2. 'risk' records flattened:
       - 'state': string (e.g. SAFE, CAUTION, WARNING, CRITICAL, DEGRADED)
       - 'global_risk': float (0..1)
       - 'corridor_left', 'corridor_center', 'corridor_right': individual float columns
       - 'confidence': float (0..1)
       - 'reason_codes': semicolon-delimited string
    3. 'haptic' records flattened:
       - 'direction': string (LEFT, CENTER, RIGHT, STOP)
       - 'urgency': integer (1..5)
       - 'pattern_id': string (e.g. LEFT_FAST, ALL_CLEAR)
       - 'duration_ms': integer
    4. 'predictions' records flattened:
       - 'num_predictions': integer
       - 'min_ttc_s': float or empty
       - 'any_intersection': boolean
    """
    import csv
    import io

    records = export_session_json(session_id, base_dir=base_dir)
    output = io.StringIO()
    writer = csv.writer(output)

    headers = [
        "ts",
        "frame_id",
        "record_type",
        "state",
        "global_risk",
        "corridor_left",
        "corridor_center",
        "corridor_right",
        "confidence",
        "reason_codes",
        "haptic_direction",
        "haptic_urgency",
        "haptic_pattern_id",
        "haptic_duration_ms",
        "num_predictions",
        "min_ttc_s",
        "any_intersection",
    ]
    writer.writerow(headers)

    for rec in records:
        rtype = rec.get("record_type")
        if rtype == "header":
            continue

        ts = rec.get("ts", 0.0)
        fid = rec.get("frame_id", 0)
        payload = rec.get("payload", {})

        row = {h: "" for h in headers}
        row["ts"] = f"{ts:.4f}"
        row["frame_id"] = str(fid)
        row["record_type"] = str(rtype)

        if rtype == "risk":
            row["state"] = payload.get("state", "")
            row["global_risk"] = f"{payload.get('global_risk', 0.0):.3f}"
            c_risks = payload.get("corridor_risks", {})
            row["corridor_left"] = f"{c_risks.get('left', 0.0):.3f}"
            row["corridor_center"] = f"{c_risks.get('center', 0.0):.3f}"
            row["corridor_right"] = f"{c_risks.get('right', 0.0):.3f}"
            row["confidence"] = f"{payload.get('confidence', 0.0):.3f}"
            reasons = payload.get("reason_codes", [])
            row["reason_codes"] = ";".join(reasons) if isinstance(reasons, list) else str(reasons)

        elif rtype == "haptic":
            row["haptic_direction"] = payload.get("direction", "")
            row["haptic_urgency"] = str(payload.get("urgency", 1))
            row["haptic_pattern_id"] = payload.get("pattern_id", "")
            row["haptic_duration_ms"] = str(payload.get("duration_ms", 0))

        elif rtype == "predictions":
            preds = payload if isinstance(payload, list) else []
            row["num_predictions"] = str(len(preds))
            ttcs = [p.get("ttc_s") for p in preds if p.get("ttc_s") is not None]
            row["min_ttc_s"] = f"{min(ttcs):.2f}" if ttcs else ""
            row["any_intersection"] = str(any(p.get("intersection_flag", False) for p in preds))

        writer.writerow([row[h] for h in headers])

    return output.getvalue()
