"""M11 — Local Telemetry Gateway & Server.

Streams pipeline state (TelemetryMessage) over WebSockets to mobile and desktop
dashboards in real-time. Serves the web dashboard as static files over HTTP.

Rules (from PROTOCOL.md & Engineering Blueprint):
1. Never block the safety loop — broadcasts are queued and dispatched asynchronously.
2. Disconnected, late-connecting, or slow clients are handled gracefully and dropped if necessary.
3. Completely removable — the safety pipeline operates normally even if TelemetryServer is not running.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
import threading
import time
from typing import Any, Dict, List, Optional, Set

from fastapi import FastAPI, HTTPException, Response, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn

from spatialvector.hmi.schemas import TelemetryMessage

logger = logging.getLogger(__name__)


class TelemetryServer:
    """FastAPI & WebSocket server for streaming telemetry to the phone dashboard."""

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8080,
        stale_threshold_s: float = 1.5,
    ):
        self.host = host
        self.port = port
        self.stale_threshold_s = stale_threshold_s

        self.app = FastAPI(title="SpatialVector-HMI Telemetry")
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        self._active_connections: Set[WebSocket] = set()
        self._connections_lock = threading.Lock()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._server_thread: Optional[threading.Thread] = None
        self._uvicorn_server: Optional[uvicorn.Server] = None
        self._stop_event = threading.Event()
        self._latest_message: Optional[dict] = None
        self._risk_engine: Optional[Any] = None

        # MJPEG video streaming — latest JPEG frame bytes pushed by pipeline
        self._latest_frame_jpeg: Optional[bytes] = None
        self._frame_lock = threading.Lock()
        self._frame_event = threading.Event()  # signals new frame availability

        self._setup_routes()

    def set_risk_engine(self, engine: Any) -> None:
        """Wires live M08 RiskEngine to telemetry server for runtime configuration."""
        self._risk_engine = engine

    def push_frame(self, frame_bgr) -> None:
        """Encodes a raw BGR OpenCV frame as JPEG and stores it for MJPEG streaming.

        Called from the main pipeline loop after each processed frame.
        Non-blocking: encoding is fast (<1ms for 720p JPEG at quality 70).
        """
        try:
            import cv2
            ret, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if ret:
                with self._frame_lock:
                    self._latest_frame_jpeg = buf.tobytes()
                self._frame_event.set()
        except Exception:
            pass

    def _setup_routes(self):
        @self.app.get("/api/video")
        async def video_feed():
            """MJPEG streaming endpoint — embed as <img src='/api/video'> in dashboard."""
            from fastapi.responses import StreamingResponse

            async def generate():
                boundary = b"--frame"
                while not self._stop_event.is_set():
                    # Wait up to 1s for a new frame
                    await asyncio.sleep(0.033)  # ~30 fps poll
                    with self._frame_lock:
                        jpeg = self._latest_frame_jpeg
                    if jpeg:
                        yield (
                            boundary
                            + b"\r\nContent-Type: image/jpeg\r\n"
                            + b"Content-Length: " + str(len(jpeg)).encode() + b"\r\n\r\n"
                            + jpeg
                            + b"\r\n"
                        )

            return StreamingResponse(
                generate(),
                media_type="multipart/x-mixed-replace; boundary=frame",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

        @self.app.get("/api/health")
        async def health():
            return {
                "status": "OK",
                "active_clients": len(self._active_connections),
                "timestamp": time.monotonic(),
            }

        @self.app.get("/api/source")
        async def get_source():
            try:
                from spatialvector.perception.source_resolver import resolve_camera_source, read_camera_source_file
                src, origin = resolve_camera_source()
                return {
                    "source": str(src),
                    "origin": origin,
                    "saved": read_camera_source_file(),
                }
            except Exception as e:
                return {"source": "0", "origin": "error", "error": str(e)}

        @self.app.post("/api/set-source")
        async def set_source(req: dict):
            try:
                from spatialvector.perception.source_resolver import write_camera_source_file
                val = req.get("source")
                if val:
                    p = write_camera_source_file(str(val).strip())
                    return {"status": "OK", "source": str(val).strip(), "path": str(p)}
                return {"status": "ERROR", "message": "Missing 'source'"}
            except Exception as e:
                return {"status": "ERROR", "error": str(e)}

        @self.app.get("/api/settings/risk-thresholds")
        async def get_risk_thresholds():
            """Returns active risk engine weights and state thresholds."""
            if self._risk_engine is not None:
                return {
                    "status": "OK",
                    "config": self._risk_engine.config.to_dict(),
                    "source": "live_engine",
                }
            from spatialvector.decision.risk_engine import RiskEngineConfig
            return {
                "status": "OK",
                "config": RiskEngineConfig().to_dict(),
                "source": "defaults",
            }

        @self.app.post("/api/settings/risk-thresholds")
        async def set_risk_thresholds(payload: dict):
            """Updates active risk engine weights and thresholds with validation."""
            from spatialvector.decision.risk_engine import RiskEngine, RiskEngineConfig
            try:
                cfg = RiskEngineConfig.from_dict(payload)
                RiskEngine.validate_config(cfg)

                if self._risk_engine is not None:
                    self._risk_engine.update_config(cfg)

                persisted = False
                if payload.get("persist", False):
                    self._persist_risk_config_to_yaml(cfg)
                    persisted = True

                return {
                    "status": "OK",
                    "config": cfg.to_dict(),
                    "persisted": persisted,
                    "message": "Risk thresholds updated successfully." + (" Persisted to default.yaml." if persisted else " (Session only)"),
                }
            except ValueError as ve:
                raise HTTPException(status_code=400, detail=str(ve))
            except Exception as exc:
                raise HTTPException(status_code=500, detail=f"Failed to update thresholds: {exc}")

        @self.app.get("/api/sessions")
        async def get_sessions():
            """Lists all recorded sessions with sizes and timestamps."""
            try:
                from spatialvector.hmi.logger import list_sessions
                return {"status": "OK", "sessions": list_sessions()}
            except Exception as e:
                return {"status": "ERROR", "error": str(e), "sessions": []}

        @self.app.get("/api/sessions/{session_id}/export")
        async def export_session(session_id: str, format: str = "json"):
            """Exports a recorded session as JSON or flattened CSV."""
            from spatialvector.hmi.logger import export_session_csv, export_session_json
            fmt = format.lower().strip()
            try:
                if fmt == "csv":
                    csv_data = export_session_csv(session_id)
                    return Response(
                        content=csv_data,
                        media_type="text/csv",
                        headers={"Content-Disposition": f'attachment; filename="{session_id}.csv"'},
                    )
                else:
                    json_data = export_session_json(session_id)
                    return Response(
                        content=json.dumps(json_data, indent=2),
                        media_type="application/json",
                        headers={"Content-Disposition": f'attachment; filename="{session_id}.json"'},
                    )
            except FileNotFoundError:
                raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Export failed: {e}")

        @self.app.websocket("/ws/telemetry")
        async def websocket_endpoint(websocket: WebSocket):
            await websocket.accept()
            with self._connections_lock:
                self._active_connections.add(websocket)
            logger.info(f"[M11] WebSocket client connected. Total clients: {len(self._active_connections)}")

            # Send immediate latest state if available
            if self._latest_message:
                try:
                    await websocket.send_text(json.dumps(self._latest_message))
                except Exception:
                    pass

            try:
                while not self._stop_event.is_set():
                    # Keep connection open; read ping/pong or client messages
                    try:
                        data = await asyncio.wait_for(websocket.receive_text(), timeout=1.0)
                    except asyncio.TimeoutError:
                        continue
            except WebSocketDisconnect:
                pass
            except Exception as exc:
                logger.debug(f"[M11] Client connection closed: {exc}")
            finally:
                with self._connections_lock:
                    self._active_connections.discard(websocket)
                logger.info(f"[M11] WebSocket client disconnected. Remaining: {len(self._active_connections)}")

        # Mount web static directories
        repo_root = Path(__file__).resolve().parent.parent.parent
        shared_dir = repo_root / "web" / "shared"
        app_simple_dir = repo_root / "web" / "app_simple"
        dashboard_dir = repo_root / "web" / "dashboard"

        if shared_dir.exists():
            self.app.mount("/shared", StaticFiles(directory=str(shared_dir)), name="shared")
        if app_simple_dir.exists():
            self.app.mount("/app_simple", StaticFiles(directory=str(app_simple_dir), html=True), name="app_simple")
        if dashboard_dir.exists():
            self.app.mount("/", StaticFiles(directory=str(dashboard_dir), html=True), name="dashboard")

    def _persist_risk_config_to_yaml(self, cfg: Any) -> None:
        """Persists updated RiskEngineConfig back to spatialvector/config/default.yaml."""
        import yaml
        yaml_path = Path(__file__).resolve().parent.parent / "config" / "default.yaml"
        if yaml_path.exists():
            with open(yaml_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            if "risk_engine" not in data:
                data["risk_engine"] = {}
            data["risk_engine"]["weight_ttc"] = cfg.weight_ttc
            data["risk_engine"]["weight_miss_distance"] = cfg.weight_miss_distance
            data["risk_engine"]["weight_intersection_confidence"] = cfg.weight_intersection_confidence
            data["risk_engine"]["state_thresholds"] = dict(cfg.state_thresholds)
            data["risk_engine"]["hysteresis_frames_up"] = cfg.hysteresis_frames_up
            data["risk_engine"]["hysteresis_frames_down"] = cfg.hysteresis_frames_down
            data["risk_engine"]["degraded_confidence_threshold"] = cfg.degraded_confidence_threshold
            with open(yaml_path, "w", encoding="utf-8") as f:
                yaml.dump(data, f, sort_keys=False)

    def start(self):
        """Starts uvicorn server in a daemon background thread."""
        self._stop_event.clear()
        self._server_thread = threading.Thread(target=self._run_server, daemon=True, name="TelemetryServer")
        self._server_thread.start()
        # Give server a brief moment to initialize
        time.sleep(0.15)
        logger.info(f"[M11] TelemetryServer running at http://{self.host}:{self.port}")

    def stop(self):
        """Stops uvicorn server and closes active websocket connections."""
        self._stop_event.set()
        if self._uvicorn_server:
            self._uvicorn_server.should_exit = True
        if self._server_thread and self._server_thread.is_alive():
            self._server_thread.join(timeout=2.0)
        with self._connections_lock:
            self._active_connections.clear()
        logger.info("[M11] TelemetryServer stopped.")

    def broadcast(self, message: TelemetryMessage) -> None:
        """Asynchronously dispatches TelemetryMessage to all connected WebSocket clients.

        Never blocks the caller.
        """
        payload = message.to_dict()
        self._latest_message = payload

        if not self._active_connections or not self._loop or not self._loop.is_running():
            return

        json_str = json.dumps(payload)

        # Schedule broadcast coroutine on the server's event loop
        try:
            self._loop.call_soon_threadsafe(self._async_broadcast, json_str)
        except RuntimeError:
            pass

    def _async_broadcast(self, json_str: str):
        with self._connections_lock:
            targets = list(self._active_connections)

        for ws in targets:
            try:
                asyncio.create_task(self._send_safe(ws, json_str))
            except Exception:
                pass

    async def _send_safe(self, ws: WebSocket, text: str):
        try:
            await ws.send_text(text)
        except Exception:
            with self._connections_lock:
                self._active_connections.discard(ws)

    def _run_server(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

        config = uvicorn.Config(
            app=self.app,
            host=self.host,
            port=self.port,
            log_level="warning",
            loop="asyncio",
        )
        self._uvicorn_server = uvicorn.Server(config)
        try:
            self._loop.run_until_complete(self._uvicorn_server.serve())
        except Exception:
            pass
        finally:
            try:
                self._loop.run_until_complete(self._loop.shutdown_asyncgens())
                self._loop.close()
            except Exception:
                pass

