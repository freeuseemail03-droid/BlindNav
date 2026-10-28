"""
server.py
===========================================================================
FastAPI web backend for the AI-Powered Blind Navigation Assistant.

Reuses the existing desktop pipeline modules UNCHANGED:
    detector.py -> ObjectDetector        (YOLO11n / ONNX Runtime)
    obstacle_manager.py -> ObstacleManager
    navigation.py -> NavigationEngine
    utils.py -> draw_detections, draw_hud

Architecture
------------
    Webcam --(thread)--> latest frame
        --> ObjectDetector.detect()          [every DETECT_EVERY_N_FRAMES]
        --> ObstacleManager.process_frame()  (distance/priority/position)
        --> NavigationEngine.get_announcement()
        --> visual guidance panel (state.instruction)
        -> JPEG + JSON state broadcast over WebSocket (/ws/stream)

Run:  .venv\\Scripts\\python.exe server.py   ->  http://127.0.0.1:8000
===========================================================================
"""

import base64
import asyncio
import io
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import psutil
from fastapi import FastAPI, File, Form, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import config
from detector import ObjectDetector
from environment_memory import EnvironmentMemory
from family_recognition import FaceModel, FaceRecognizer, validate_uploaded_image
from free_space import FreeSpaceAnalyzer
from history_store import HistoryStore
from navigation import NavigationEngine
from navigation_detector import NavigationObjectDetector
from obstacle_manager import ObstacleManager
from path_planner import PathPlanner
from scene_summary import SceneSummary
from sos import SOSManager
from utils import (
    FpsCounter, draw_detections, estimate_brightness, enhance_low_light,
    is_low_light,
)
from vehicle_awareness import VehicleAwareness

APP_VERSION = "2.1.0"
STATIC_DIR = Path(__file__).resolve().parent / "static"
FAMILY_PHOTO_DIR = Path(__file__).resolve().parent / config.FAMILY_DATA_DIR / "photos"

MAX_STREAM_FPS = 60          # cap on encoded frames sent to browsers (camera-limited in practice)
JPEG_QUALITY = 72


# ---------------------------------------------------------------------------
# Camera capture thread (latest-frame pattern: old frames are dropped so the
# processing loop always works on the freshest image).
# ---------------------------------------------------------------------------

class CameraThread:
    def __init__(self, index: int = config.CAMERA_INDEX):
        self.index = index
        self._cap: cv2.VideoCapture | None = None
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._frame_id = 0
        self._cond = threading.Condition(self._lock)
        self._alive = False
        self._thread: threading.Thread | None = None
        self.error: str | None = None

    @property
    def running(self) -> bool:
        return self._alive

    def start(self, width: int, height: int) -> bool:
        if self._alive:
            return True
        self.error = None
        # Prefer the DirectShow backend on Windows. The default Media
        # Foundation backend buffers several frames, which shows up as laggy /
        # frozen webcam video. Fall back to the default backend if DSHOW
        # cannot open the device (some virtual/driver-only cameras).
        cap = cv2.VideoCapture(self.index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(self.index)
        if not cap.isOpened():
            self.error = (
                f"Could not open camera (index {self.index}). "
                "Check that a webcam is connected and not used by another app."
            )
            return False
        # Keep only the newest frame so we never render a stale, buffered one.
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if config.CAMERA_FORCE_MJPG:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FPS, getattr(config, "CAMERA_FPS", 60))

        # Warm-up: discard exposure-adjusting first frames.
        for _ in range(5):
            cap.read()

        self._cap = cap
        self._alive = True
        self._thread = threading.Thread(target=self._loop, name="camera", daemon=True)
        self._thread.start()
        return True

    def _loop(self) -> None:
        empty = 0
        while self._alive:
            ok, frame = self._cap.read() if self._cap else (False, None)
            if not ok or frame is None:
                empty += 1
                if empty >= config.MAX_CONSECUTIVE_EMPTY_FRAMES:
                    self.error = "Camera stopped delivering frames."
                    break
                time.sleep(0.02)
                continue
            empty = 0
            with self._cond:
                self._frame = frame
                self._frame_id += 1
                self._cond.notify_all()

    def read_latest(self) -> np.ndarray | None:
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def wait_frame(self, last_id: int,
                   timeout: float = 0.1) -> tuple[int, np.ndarray | None]:
        """Block until a frame newer than ``last_id`` arrives (or timeout).

        Lets the pipeline run exactly one pass per real camera frame instead
        of spinning and re-encoding the same frame when the loop is faster
        than the camera.
        """
        with self._cond:
            if self._frame_id == last_id and self._alive:
                self._cond.wait(timeout)
            if self._frame is None:
                return self._frame_id, None
            return self._frame_id, self._frame.copy()

    def get_resolution(self) -> tuple[int, int]:
        if self._cap is None:
            return 0, 0
        return int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(
            self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        )

    def stop(self) -> None:
        self._alive = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        with self._lock:
            self._frame = None


# ---------------------------------------------------------------------------
# Processing worker: detection -> obstacles -> navigation -> guidance -> stream.
# ---------------------------------------------------------------------------

class NavigationWorker:
    RUNNING, PAUSED, STOPPED = "RUNNING", "PAUSED", "STOPPED"

    def __init__(self) -> None:
        self.detector = ObjectDetector(
            model_path=config.MODEL_PATH,
            target_classes=config.TARGET_CLASSES,
            confidence=config.CONFIDENCE_THRESHOLD,
            iou=config.IOU_THRESHOLD,
            img_size=config.IMG_SIZE,
        )
        self.model_error: str | None = None
        try:
            self.detector.load()
        except Exception as exc:
            self.model_error = str(exc)

        self.camera = CameraThread()

        # Optional Family Recognition layer. Never fatal: if it fails to
        # initialise, is_available() returns False and plain "person" logic
        # continues exactly as before.
        self.family = FaceRecognizer()
        self.family_enabled = config.FAMILY_RECOGNITION_ENABLED
        # Warm the face model at startup so the first Start doesn't stall on a
        # ~10s lazy load. Safe: if it fails, is_available() stays False and
        # the rest of navigation is unaffected.
        if self.family_enabled:
            try:
                self.family.model.is_ready()
            except Exception:
                pass

        self.obstacle_mgr = ObstacleManager()
        self.nav_engine = NavigationEngine()
        self.fps_counter = FpsCounter()

        # ---- detection / guidance modules (all optional / degradable) ----
        self.vehicles = VehicleAwareness()
        self.free_space = FreeSpaceAnalyzer()
        self.planner = PathPlanner()
        self.scene = SceneSummary()
        self.memory = EnvironmentMemory()
        self.sos_mgr = SOSManager()
        self.history = HistoryStore()
        self.nav_detector = NavigationObjectDetector()
        self.nav_detector.load()          # graceful: "MODEL NOT INSTALLED" if absent

        self.state = self.STOPPED
        self.detect_every = config.DETECT_EVERY_N_FRAMES
        self._frame_no = 0
        self._cached_detections: list = []

        # Low-light state (measured, not fabricated).
        self.low_light_mode = config.LOW_LIGHT_MODE       # auto|on|off
        self.low_light_active = False
        self.brightness = 255.0
        self.path_columns = {"LEFT": "CLEAR", "CENTER": "CLEAR", "RIGHT": "CLEAR"}
        self.path_guidance = ""

        # Live metrics (all measured, never fabricated).
        self.inference_ms = 0.0
        self.cpu_percent = 0.0
        self.started_at = time.time()
        self.last_instruction = ""
        self._seq = 0
        self._last_cam_seq = -1
        self._payload_lock = threading.Lock()
        self._payload: dict | None = None
        self._last_loop_error = 0.0

        self._thread = threading.Thread(target=self._loop_guarded,
                                        name="pipeline", daemon=True)
        self._thread.start()

    # ------------------------------------------------------------------
    # Control API (called from FastAPI endpoints)
    # ------------------------------------------------------------------

    def start(self, width: int | None = None, height: int | None = None,
              camera_index: int | None = None) -> tuple[bool, str]:
        if self.model_error:
            return False, f"AI model is not ready: {self.model_error}"
        if camera_index is not None and int(camera_index) != self.camera.index:
            if self.camera.running:
                self.camera.stop()
            self.camera = CameraThread(int(camera_index))
        w = width or config.CAMERA_WIDTH
        h = height or config.CAMERA_HEIGHT
        if not self.camera.running and not self.camera.start(w, h):
            return False, self.camera.error or "Camera unavailable."
        # Reset per-session state: announcement timers and obstacle tracking.
        self.nav_engine = NavigationEngine()
        self.obstacle_mgr = ObstacleManager()
        self.obstacle_mgr.set_frame_width(max(w, 1))
        self._cached_detections = []
        self.state = self.RUNNING
        return True, "Navigation started."

    def pause(self) -> None:
        if self.state == self.RUNNING:
            self.state = self.PAUSED

    def resume(self) -> None:
        if self.state == self.PAUSED:
            self.state = self.RUNNING

    def stop(self) -> None:
        self.state = self.STOPPED
        self._cached_detections = []
        self.camera.stop()

    def apply_settings(self, s: dict) -> dict:
        applied = {}
        if "confidence" in s:
            v = float(s["confidence"])
            self.detector.set_confidence(v)
            applied["confidence"] = v
        if "detect_every" in s:
            self.detect_every = max(1, min(int(s["detect_every"]), 10))
            applied["detect_every"] = self.detect_every
        if "family_enabled" in s:
            self.family_enabled = bool(s["family_enabled"])
            self.family.enabled = self.family_enabled
            applied["family_enabled"] = self.family_enabled
        if "family_announce_mode" in s:
            mode = str(s["family_announce_mode"])
            if mode in ("relationship", "name", "name+relation"):
                self.family.announce_mode = mode
                applied["family_announce_mode"] = mode
        if "family_threshold" in s:
            v = max(0.2, min(float(s["family_threshold"]), 0.9))
            self.family.threshold = v
            applied["family_threshold"] = v

        # ----- optional guidance features ------------------------------
        if "low_light_mode" in s:
            m = str(s["low_light_mode"]).strip().lower()
            if m in ("auto", "on", "off"):
                self.low_light_mode = m
                applied["low_light_mode"] = m
        if "vehicle_warning_enabled" in s:
            self.vehicles.enabled = bool(s["vehicle_warning_enabled"])
            applied["vehicle_warning_enabled"] = self.vehicles.enabled
        if "safe_path_enabled" in s:
            self.planner.enabled = bool(s["safe_path_enabled"])
            applied["safe_path_enabled"] = self.planner.enabled
        if "free_space_enabled" in s:
            self.free_space.enabled = bool(s["free_space_enabled"])
            applied["free_space_enabled"] = self.free_space.enabled
        if "scene_summary_enabled" in s:
            self.scene.enabled = bool(s["scene_summary_enabled"])
            applied["scene_summary_enabled"] = self.scene.enabled
        return applied

    # ------------------------------------------------------------------
    # On-demand helpers
    # ------------------------------------------------------------------

    def set_low_light(self, mode: str) -> str:
        mode = mode.strip().lower()
        if mode in ("auto", "on", "off"):
            self.low_light_mode = mode
        return self.low_light_mode

    def announce_scene(self) -> dict:
        """Fresh scene description from the current detections (on demand)."""
        frame = self.camera.read_latest()
        obstacles = self.obstacle_mgr.process_frame(self._cached_detections) \
            if self.state == self.RUNNING else []
        fs = self.free_space.analyze(frame.shape, obstacles) \
            if frame is not None else None
        summary = self.scene.build_fresh(obstacles, fs)
        if summary:
            self.history.add(
                {"type": "SCENE", "text": summary, "priority": "INFO"})
        return {"ok": bool(summary), "summary": summary}

    # ------------------------------------------------------------------
    # Status snapshot for REST clients
    # ------------------------------------------------------------------

    def status(self) -> dict:
        cam_w, cam_h = self.camera.get_resolution()
        return {
            "app_version": APP_VERSION,
            "status": self.state,
            "model": {
                "name": self.detector.model_name,
                "device": self.detector.get_device(),
                "img_size": config.IMG_SIZE,
                "ready": self.model_error is None,
                "error": self.model_error,
            },
            "camera": {
                "connected": self.camera.running,
                "index": self.camera.index,
                "width": cam_w,
                "height": cam_h,
                "error": self.camera.error,
            },
            "metrics": {
                "fps": round(self.fps_counter.fps, 1),
                "inference_ms": round(self.inference_ms, 1),
                "cpu_percent": round(self.cpu_percent, 1),
                "objects": len(self._cached_detections),
                "uptime_s": round(time.time() - self.started_at, 1),
                "gpu_percent": None,   # no CUDA on this machine - honest N/A
            },
            "settings": {
                "confidence": self.detector._confidence,
                "detect_every": self.detect_every,
            },
            "family": {
                "enabled": self.family_enabled,
                "available": self.family.is_available(),
                "model_ready": self.family.model.is_ready(),
                "members": len(self.family.store.list_members()),
                "announce_mode": self.family.announce_mode,
                "model_error": self.family.model.error,
            },
            "low_light": {
                "mode": self.low_light_mode,
                "active": self.low_light_active,
                "brightness": round(self.brightness, 1),
            },
            "vehicle_awareness": {"enabled": self.vehicles.enabled,
                                  "classes": config.VEHICLE_CLASSES},
            "navigation_model": self.nav_detector.status(),
            "free_space": {"enabled": self.free_space.enabled},
            "safe_path": {"enabled": self.planner.enabled},
            "scene_summary": {"enabled": self.scene.enabled},
            "depth": self._depth_status(),
            "sos": self.sos_mgr.status(),
        }

    def _depth_status(self) -> dict:
        try:
            from depth_provider import DepthProviderManager
            if not hasattr(self, "_depth"):
                self._depth = DepthProviderManager()
            return self._depth.status()
        except Exception as exc:
            return {"error": str(exc), "active": "monocular"}

    # ------------------------------------------------------------------
    # Payload access for WebSocket clients
    # ------------------------------------------------------------------

    def get_payload(self) -> tuple[int, dict | None]:
        with self._payload_lock:
            return self._seq, self._payload

    # ------------------------------------------------------------------
    # Main pipeline loop
    # ------------------------------------------------------------------

    def _loop_guarded(self) -> None:
        """Run the pipeline loop, restarting it if a frame ever raises.

        The streaming thread is the one thing that must never die: if it
        stops, the dashboard shows a frozen last frame with no recovery. This
        guard turns any unexpected per-frame error into a skipped frame
        (logged at most every few seconds) instead of a dead stream.
        """
        while True:
            try:
                self._loop()
            except Exception as exc:  # noqa: BLE001 - never let the stream die
                now = time.monotonic()
                if now - self._last_loop_error >= 5.0:
                    self._last_loop_error = now
                    print(f"[WARN] Pipeline error (frame skipped): {exc!r}")
                time.sleep(0.1)

    def _loop(self) -> None:
        min_frame_time = 1.0 / MAX_STREAM_FPS
        while True:
            loop_t0 = time.perf_counter()

            if psutil is not None:
                self.cpu_percent = psutil.cpu_percent(interval=None)

            if self.state == self.STOPPED:
                self._publish({}, b"")
                time.sleep(0.05)
                continue

            cam_seq, frame = self.camera.wait_frame(self._last_cam_seq)
            if frame is None:
                self._publish({"camera_connected": False}, b"")
                time.sleep(0.05)
                continue
            # One pass per real camera frame: the loop can spin faster than the
            # camera, and re-encoding an identical frame only burns CPU and
            # bandwidth without adding any smoothness.
            if cam_seq == self._last_cam_seq:
                continue
            self._last_cam_seq = cam_seq

            # --- low-light awareness (always measured, honest) ----------
            try:
                brightness = estimate_brightness(frame)
                self.brightness = brightness
                if self.low_light_mode == "on":
                    use_enhance = True
                elif self.low_light_mode == "off":
                    use_enhance = False
                else:  # auto
                    use_enhance = is_low_light(brightness)
                if use_enhance:
                    frame = enhance_low_light(frame)
                self.low_light_active = use_enhance
            except Exception:
                self.low_light_active = False

            self._frame_no += 1
            detecting = self.state == self.RUNNING and (
                self._frame_no % max(self.detect_every, 1) == 1
            )

            if detecting:
                t0 = time.perf_counter()
                detections = self.detector.detect(frame)
                # Optional custom navigation model (graceful when absent).
                if self.nav_detector._model is not None:
                    try:
                        detections += self.nav_detector.detect(frame)
                    except Exception:
                        pass
                self._cached_detections = detections
                self.inference_ms = (
                    0.9 * self.inference_ms + 0.1 * (time.perf_counter() - t0) * 1000
                    if self.inference_ms
                    else (time.perf_counter() - t0) * 1000
                )

            if self.state == self.RUNNING:
                obstacles = self.obstacle_mgr.process_frame(self._cached_detections)

                # --- optional family recognition (information layer) -------
                if self.family_enabled and self.family.is_available():
                    person_boxes = [
                        {"bbox": o["bbox"], "track_id": o["track_id"]}
                        for o in obstacles if o["label"] == "person"
                    ]
                    identities = self.family.recognize_people(frame, person_boxes)
                    for pbox, res in zip(person_boxes, identities):
                        tid = pbox["track_id"]
                        # Cadence-skipped ticks carry no fresh recognition
                        # (reason == "disabled"). Feeding them into the
                        # stability filter would reset the counter between
                        # real recognitions, so the identity could never
                        # become stable. Only genuine results update the
                        # filter; skipped ticks reuse what is already stable.
                        if res.get("reason") != "disabled":
                            self.family.smooth_track(tid, res["identity"])
                            similarity = res["similarity"]
                        else:
                            similarity = None
                        stable = self.family.stable_identity(tid)
                        for o in obstacles:
                            if o["label"] == "person" and o["track_id"] == tid:
                                o["family"] = stable
                                if similarity is not None:
                                    o["family_sim"] = similarity
                                if stable != "unknown":
                                    o["member_id"] = stable

                # --- guidance scene features (all cheap) ------------------
                vehicle_states = self.vehicles.update(obstacles)
                self._last_vehicle_states = vehicle_states
                path = self.free_space.analyze(frame.shape, obstacles)
                self.path_columns = path
                # Vehicle warnings include evasive advice, so they need the
                # walkable-path columns to avoid pointing into a blocked lane.
                vehicle_warnings = self.vehicles.warnings(vehicle_states, path)
                guidance = self.planner.guidance(path, obstacles)
                self.path_guidance = guidance
                self.memory.update(obstacles)

                # --- history logging (dedupe via merge window) -----------
                if obstacles:
                    top = obstacles[0]
                    if top["priority"] in ("CRITICAL", "HIGH"):
                        self.history.add({
                            "type": "OBSTACLE", "label": top["label"],
                            "distance": top["distance"],
                            "position": top["position"],
                            "priority": top["priority"],
                        })
                for v in vehicle_states:
                    if v["state"] == "approaching":
                        self.history.add({
                            "type": "VEHICLE", "label": v["label"],
                            "position": v["position"], "text": "approaching",
                            "priority": "HIGH",
                        })
                if obstacles and any(o.get("family") and o["family"] != "unknown"
                                     for o in obstacles):
                    fam = next(o for o in obstacles
                               if o.get("family") and o["family"] != "unknown")
                    name = self.family.store.get(fam["member_id"])
                    self.history.add({
                        "type": "FAMILY",
                        "label": (name["relationship"] if name else fam["family"]),
                        "distance": fam["distance"],
                        "position": fam["position"], "priority": "INFO",
                    })

                # --- priority-ordered guidance decision -------------------
                self._update_instruction(obstacles, vehicle_warnings, guidance)

            elif self.state == self.PAUSED:  # frozen last-known state
                obstacles = self.obstacle_mgr.process_frame(self._cached_detections)
            else:
                obstacles = []

            annotated = draw_detections(frame, obstacles)
            if self.family_enabled:
                self._draw_family_names(annotated, obstacles)
            self._draw_path_overlay(annotated)
            if self.low_light_active:
                cv2.putText(
                    annotated, "LOW LIGHT - enhanced", (10, 95),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 255), 2, cv2.LINE_AA,
                )
            fps = self.fps_counter.tick()
            if self.state == self.PAUSED:
                cv2.putText(
                    annotated, "PAUSED", (annotated.shape[1] // 2 - 90, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 215, 255), 3, cv2.LINE_AA,
                )

            ok, buf = cv2.imencode(
                ".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
            )

            objects_payload = []
            veh_payload = []
            for o in obstacles[:8]:
                fam_member = None
                fam_identity = o.get("family")
                if fam_identity and fam_identity != "unknown":
                    fam_member = self.family.store.get(fam_identity)
                objects_payload.append({
                    "label": o["label"],
                    "confidence": round(float(o["confidence"]), 2),
                    "distance": float(o["distance"]),
                    "position": o["position"],
                    "priority": o["priority"],
                    "family": (fam_member["name"] if fam_member else None),
                    "family_rel": (fam_member["relationship"] if fam_member else None),
                    "family_sim": round(float(o.get("family_sim", 0.0)), 2),
                })
            for v in getattr(self, "_last_vehicle_states", []) or []:
                veh_payload.append({
                    "label": v["label"], "state": v["state"],
                    "distance": v["distance"], "position": v["position"],
                })
            state = {
                "type": "frame",
                "seq": self._seq + 1,
                "status": self.state,
                "fps": round(fps, 1),
                "inference_ms": round(self.inference_ms, 1),
                "cpu_percent": round(self.cpu_percent, 1),
                "objects": objects_payload,
                "instruction": self.last_instruction,
                "top_priority": obstacles[0]["priority"] if obstacles else ("SAFE" if self.state == self.RUNNING else None),
                "camera_connected": True,
                "path": self.path_columns,
                "guidance": self.path_guidance,
                "low_light": {"active": self.low_light_active,
                              "mode": self.low_light_mode,
                              "brightness": round(self.brightness, 1)},
                "vehicles": veh_payload,
                "ts": time.time(),
            }
            self._publish(state, buf.tobytes() if ok else b"")

            # Throttle to MAX_STREAM_FPS.
            spent = time.perf_counter() - loop_t0
            if spent < min_frame_time:
                time.sleep(min_frame_time - spent)

    def _set_instruction(self, text: str) -> None:
        """Update the visual guidance text only when it actually changes."""
        if text and text != self.last_instruction:
            self.last_instruction = text

    def _update_instruction(self, obstacles: list[dict],
                            vehicle_warnings: list[str], guidance: str) -> None:
        """Pick the guidance shown on the dashboard (visual only).

        1 CRITICAL obstacle | 2 immediate path blockage | 3 approaching
        vehicle | 4 high/medium obstacle | 5 safe-path guidance | 6 info.
        """
        top = obstacles[0] if obstacles else None
        announcement = self.nav_engine.get_announcement(obstacles)

        # 1. Critical obstacle -> the most specific, immediate warning.
        if top and top["priority"] == "CRITICAL":
            if top.get("family") and top["family"] != "unknown":
                phrase = self.family.announce_phrase(
                    top["member_id"] or top["family"])
                self._set_instruction(self._person_with_identity(top, phrase))
            elif announcement:
                self._set_instruction(announcement)
            return

        # 2. Immediate path blockage (all three columns blocked).
        if guidance and guidance.startswith("Stop."):
            self._set_instruction(guidance)
            self.history.add(
                {"type": "PATH", "text": guidance, "priority": "CRITICAL"})
            return

        # 3. Approaching vehicle (temporal evidence required upstream).
        if vehicle_warnings:
            self._set_instruction(vehicle_warnings[0])
            return

        # 4. High / medium obstacle announcement.
        if announcement:
            if top and top.get("family") and top["family"] != "unknown":
                phrase = self.family.announce_phrase(
                    top["member_id"] or top["family"])
                self._set_instruction(self._person_with_identity(top, phrase))
            else:
                self._set_instruction(announcement)
            return

        # 5. Safe-path guidance (free space, throttled by the planner).
        if guidance:
            self._set_instruction(guidance)
            return

        # 6. Dashboard-only summary of the nearest obstacle.
        if obstacles:
            top = obstacles[0]
            if top.get("family") and top["family"] != "unknown":
                phrase = self.family.announce_phrase(
                    top["member_id"] or top["family"])
                self._set_instruction(self._person_with_identity(top, phrase))
            else:
                self._set_instruction(
                    f"{top['label'].capitalize()} {top['distance']:.1f} m "
                    f"{top['position'].lower()}"
                )
        else:
            self.last_instruction = ""

    @staticmethod
    def _person_with_identity(obs: dict, phrase: str) -> str:
        """Build a natural guidance instruction for a recognised person,
        rounding the distance naturally (2.13 -> '2 meters', 1.48 -> '1.5').
        """
        label = obs.get("label", "person")
        dist = obs["distance"]
        position = obs["position"]

        def dist_str(d: float) -> str:
            if d < 0.5:
                return "very close"
            r = round(d * 2) / 2  # nearest 0.5 m
            if r < 1.5:
                return f"{r:.1f}".rstrip("0").rstrip(".") + " meters"
            return f"{int(round(r) or 0)} meters" if r == int(r) else f"{r:.1f} meters"

        if dist < 0.6 or obs.get("priority") == "CRITICAL":
            return (f"Warning! {phrase} {dist_str(dist)} "
                    f"{NavigationWorker._pos_words(position)}. Please stop.")
        return (f"{phrase} is {dist_str(dist)} "
                f"{NavigationWorker._pos_words(position)}.")

    @staticmethod
    def _pos_words(position: str) -> str:
        return {"LEFT": "on your left", "RIGHT": "on your right"}.get(position, "ahead")

    def _draw_family_names(self, canvas: np.ndarray, obstacles: list[dict]) -> None:
        for o in obstacles:
            family = o.get("family")
            if o.get("label") != "person" or not family or family == "unknown":
                continue
            name = self.family.store.get(o["member_id"])
            display = (name.get("name") or name.get("relationship")) if name else family
            x1, y1, x2, y2 = o["bbox"]
            color = (60, 200, 255)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
            label = f"{display}"
            ff = cv2.FONT_HERSHEY_SIMPLEX
            (tw, th), _ = cv2.getTextSize(label, ff, 0.55, 2)
            cv2.rectangle(canvas, (x1, y1), (x1 + tw + 8, y1 - th - 10), color, -1)
            cv2.putText(canvas, label, (x1 + 4, y1 - 6), ff, 0.55, (0, 0, 0), 2, cv2.LINE_AA)
            if o.get("family_sim"):
                cv2.putText(canvas, f"{o['family_sim']*100:.0f}%", (x2 - 40, y2 - 8),
                            ff, 0.5, color, 2, cv2.LINE_AA)

    def _draw_path_overlay(self, canvas: np.ndarray) -> None:
        try:
            box_colors = {
                "CLEAR": (80, 220, 120),
                "CAUTION": (80, 170, 255),
                "BLOCKED": (70, 70, 255),
            }
            names = {"LEFT": "L", "CENTER": "C", "RIGHT": "R"}
            parts = []
            for col in ("LEFT", "CENTER", "RIGHT"):
                st = self.path_columns.get(col, "CLEAR")
                color = box_colors.get(st, (255, 255, 255))
                parts.append(f"{names[col]}:{st[:2]}")
                cv2.rectangle(canvas, (6 + 150 * list(
                    ("LEFT", "CENTER", "RIGHT")).index(col), 16),
                              (6 + 136 + 150 * list(
                               ("LEFT", "CENTER", "RIGHT")).index(col), 40),
                              color, -1)
            label = "  ".join(parts)
            cv2.putText(canvas, label, (6, 32), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (20, 20, 20), 2, cv2.LINE_AA)
        except Exception:
            pass

    def _publish(self, state: dict, jpeg: bytes) -> None:
        with self._payload_lock:
            self._seq += 1
            self._payload = {"json": state, "jpeg_b64": base64.b64encode(jpeg).decode("ascii") if jpeg else ""}


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

worker: NavigationWorker | None = None
app = FastAPI(title="BlindNav AI", version=APP_VERSION)


@app.middleware("http")
async def _revalidate_static_assets(request, call_next):
    """Stop browsers from serving stale CSS/JS/HTML after an update.

    Starlette's StaticFiles only sends ``ETag``/``Last-Modified``, so browsers
    apply heuristic caching and may keep showing an old UI without ever
    revalidating. Sending an explicit ``Cache-Control: no-cache`` makes the
    browser revalidate on every load; unchanged files still reply ``304`` via
    the existing ETag, so this costs nothing in the normal case.
    """
    response = await call_next(request)
    response.headers.setdefault("Cache-Control", "no-cache")
    return response


@app.on_event("startup")
def _startup() -> None:
    global worker
    worker = NavigationWorker()


@app.get("/api/status")
def api_status():
    if worker is None:
        return JSONResponse({"error": "starting up"}, status_code=503)
    return worker.status()


@app.post("/api/control")
def api_control(action: str, camera: int | None = None):
    if worker is None:
        return JSONResponse({"ok": False, "message": "Server starting up."}, status_code=503)
    action = action.lower().strip()
    if action == "start":
        ok, msg = worker.start(camera_index=camera)
        return {"ok": ok, "message": msg, "status": worker.state}
    if action == "pause":
        worker.pause()
        return {"ok": True, "message": "Paused.", "status": worker.state}
    if action == "resume":
        worker.resume()
        return {"ok": True, "message": "Resumed.", "status": worker.state}
    if action == "stop":
        worker.stop()
        return {"ok": True, "message": "Stopped.", "status": worker.state}
    return JSONResponse({"ok": False, "message": f"Unknown action '{action}'."}, status_code=400)


@app.post("/api/settings")
def api_settings(payload: dict):
    if worker is None:
        return JSONResponse({"ok": False, "message": "Server starting up."}, status_code=503)
    applied = worker.apply_settings(payload)
    return {"ok": True, "applied": applied}


@app.get("/api/cameras")
def api_cameras():
    """Probe camera indices 0..3 that are not currently owned by the app."""
    found = []
    active = worker.camera.index if worker else 0
    for idx in range(4):
        if worker and worker.camera.running and idx == active:
            found.append(idx)
            continue
        cap = cv2.VideoCapture(idx)
        opened = cap.isOpened()
        if opened:
            cap.release()
        if opened:
            found.append(idx)
        time.sleep(0.05)
    return {"cameras": found, "active": active}


# -------------------------------------------------------------------------
# Multimodal AI API  (history / path / depth / scene / SOS / low-light)
# -------------------------------------------------------------------------

class LowLightModel(BaseModel):
    mode: str


class SoscContactsModel(BaseModel):
    contacts: list[dict]


@app.get("/api/history")
def api_history(limit: int = 100):
    w = _require_worker()
    return {"entries": w.history.list(limit=limit)}


@app.delete("/api/history")
def api_history_clear():
    w = _require_worker()
    return {"ok": True, "removed": w.history.clear()}


@app.get("/api/navigation/path")
def api_navigation_path():
    w = _require_worker()
    return {"path": w.path_columns, "guidance": w.path_guidance,
            "free_space_enabled": w.free_space.enabled,
            "safe_path_enabled": w.planner.enabled}


@app.get("/api/depth/status")
def api_depth_status():
    w = _require_worker()
    return w._depth_status()


@app.post("/api/scene/summary")
def api_scene_summary():
    w = _require_worker()
    return w.announce_scene()


@app.post("/api/low-light")
def api_low_light(body: LowLightModel):
    w = _require_worker()
    mode = w.set_low_light(body.mode)
    return {"ok": True, "mode": mode, "active": w.low_light_active}


@app.get("/api/sos/status")
def api_sos_status():
    w = _require_worker()
    return w.sos_mgr.status()


@app.post("/api/sos/arm")
def api_sos_arm():
    w = _require_worker()
    return w.sos_mgr.arm()


@app.post("/api/sos/confirm")
def api_sos_confirm():
    w = _require_worker()
    return w.sos_mgr.confirm()


@app.post("/api/sos/cancel")
def api_sos_cancel():
    w = _require_worker()
    return w.sos_mgr.cancel()


@app.put("/api/sos/contacts")
def api_sos_contacts(body: SoscContactsModel):
    w = _require_worker()
    return {"ok": True, "contacts": w.sos_mgr.set_contacts(body.contacts)}


# -------------------------------------------------------------------------
# Family Recognition API
# -------------------------------------------------------------------------

class UpdateMemberModel(BaseModel):
    name: str | None = None
    relationship: str | None = None


def _require_worker():
    if worker is None:
        raise RuntimeError("Server is still starting up.")
    return worker


@app.get("/api/family/relationships")
def family_relationships():
    return {"relationships": config.FAMILY_RELATIONSHIPS,
            "announce_modes": ["relationship", "name", "name+relation"]}


@app.get("/api/family/members")
def family_members():
    w = _require_worker()
    return {"members": w.family.store.list_members(),
            "enabled": w.family_enabled,
            "available": w.family.is_available(),
            "model_ready": w.family.model.is_ready(),
            "model_error": w.family.model.error,
            "threshold": w.family.threshold,
            "announce_mode": w.family.announce_mode}


@app.get("/api/family/members/{member_id}")
def family_member(member_id: str):
    w = _require_worker()
    m = w.family.store.get(member_id)
    if not m:
        return JSONResponse({"ok": False, "error": "Family member not found."}, status_code=404)
    return m


@app.post("/api/family/register")
async def family_register(
    name: str = Form(...),
    relationship: str = Form(...),
    member_id: str = Form(""),
    files: list[UploadFile] = File(...),
):
    """Register (or add photos to) a family member from uploaded photos.

    Each uploaded file is validated: must decode, contain EXACTLY ONE usable
    face. Embeddings are generated and persisted only for valid photos.
    """
    w = _require_worker()
    if not w.family.model.is_ready():
        return JSONResponse({"ok": False, "error":
            w.family.model.error or "Face model is not available."}, status_code=503)
    if not files:
        return JSONResponse({"ok": False, "error": "No photos were uploaded."}, status_code=400)

    embeddings: list[np.ndarray] = []
    originals: list[tuple[np.ndarray, str]] = []
    results: list[dict] = []

    # Read all uploads first (async), then do the heavy face validation in a
    # worker thread so we never block the async event loop.
    raw_files = []
    for f in files:
        raw = await f.read()
        raw_files.append((f.filename or "pic.jpg", raw))

    import asyncio

    def _process(raw_files):
        emb: list[np.ndarray] = []
        orig: list[tuple[np.ndarray, str]] = []
        res: list[dict] = []
        for filename, raw in raw_files:
            ext = Path(filename).suffix.lower() or ".jpg"
            if ext not in (".jpg", ".jpeg", ".png", ".webp", ".bmp"):
                res.append({"filename": filename, "status": "error",
                            "message": f"Unsupported format '{ext}'. Use JPG/PNG."})
                continue
            img = _decode_image(raw)
            if img is None:
                res.append({"filename": filename, "status": "error",
                            "message": "Could not read image (corrupted or unsupported)."})
                continue
            validation = validate_uploaded_image(img, w.family.model)
            if not validation["ok"]:
                res.append({"filename": filename, "status": "error",
                            "message": validation["error"]})
                continue
            face = validation["faces"][0]
            emb.append(face["embedding"])
            orig.append((img, ext))
            res.append({"filename": filename, "status": "ok",
                        "faces": 1, "message": "Face detected."})
        return emb, orig, res

    embeddings, originals, results = await asyncio.get_running_loop().run_in_executor(
        None, _process, raw_files)

    # Only register if at least one valid face photo exists.
    if not embeddings:
        return {"ok": False, "results": results,
                "error": "None of the uploaded photos had a single valid face."}

    signature_member_id = member_id.strip() or None
    member = w.family.register(signature_member_id, name, relationship,
                               embeddings, originals)
    return {"ok": True, "member": member, "results": results}


@app.put("/api/family/members/{member_id}")
async def family_update(member_id: str, body: UpdateMemberModel):
    w = _require_worker()
    try:
        m = w.family.store.update(member_id, body.name, body.relationship)
    except Exception as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=404)
    return {"ok": True, "member": m}


@app.delete("/api/family/members/{member_id}")
async def family_delete(member_id: str):
    w = _require_worker()
    try:
        w.family.store.delete(member_id)
    except Exception as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=404)
    return {"ok": True}


@app.get("/api/family/settings")
def family_settings():
    w = _require_worker()
    return {"enabled": w.family_enabled,
            "announce_mode": w.family.announce_mode,
            "threshold": w.family.threshold}


def _decode_image(raw: bytes) -> np.ndarray | None:
    try:
        buf = np.frombuffer(raw, dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None


@app.websocket("/ws/stream")
async def ws_stream(ws: WebSocket):
    await ws.accept()
    last_seq = -1
    try:
        while True:
            seq, payload = worker.get_payload() if worker else (last_seq, None)
            if payload is not None and seq != last_seq:
                last_seq = seq
                await ws.send_json(payload)
            else:
                # Poll well below the frame interval so every published frame
                # is picked up. A 33 ms sleep here raced the ~33 ms producer
                # and silently dropped about a quarter of the frames.
                await asyncio.sleep(0.004)
    except WebSocketDisconnect:
        pass
    except Exception:
        pass


# Serve registered family photos locally (privacy: never sent off-device).
app.mount("/family-photos", StaticFiles(directory=str(FAMILY_PHOTO_DIR)), name="family-photos")

# Static frontend (landing/dashboard/settings/about) served at "/".
app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="BlindNav AI web server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    print("=" * 60)
    print(" BlindNav AI  -  web server")
    print(f" Open http://{args.host}:{args.port} in your browser")
    print("=" * 60)
    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
