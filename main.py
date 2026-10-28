"""
main.py
===========================================================================
Entry point: Camera -> YOLO26l -> Detection -> Distance -> Priority ->
             Navigation -> on-screen guidance (visual only)
===========================================================================
"""

import os
import sys


def _ensure_venv() -> None:
    """Pick the correct interpreter when critical packages are missing.

    Order of preference:
      1. Shared libraries on D: (PYTHONPATH set, e.g. D:\\program\\pylibs)
         -> system Python is used as-is; the project stays small.
      2. Project-local .venv (fallback, original behaviour).
    """
    try:
        from ultralytics import YOLO  # noqa: F401
        return
    except ImportError:
        pass

    if os.environ.get("PYTHONPATH", "").strip():
        print("[ERROR] PYTHONPATH is set but 'ultralytics' cannot be imported.")
        sys.exit(1)

    venv_python = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), ".venv", "Scripts", "python.exe"
    )
    if sys.executable.replace("\\", "/").lower() == venv_python.replace("\\", "/").lower():
        return
    if not os.path.isfile(venv_python):
        print(
            "[ERROR] No Python environment found.\n"
            "Either set PYTHONPATH to a shared library folder\n"
            "    python -m pip install --target D:\\program\\pylibs -r requirements.txt\n"
            "...or create a project .venv:\n"
            "    python -m venv .venv && python -m pip install -r requirements.txt"
        )
        sys.exit(1)

    print(f"[INFO] Relaunching with venv interpreter: {venv_python}")
    os.execv(venv_python, [venv_python] + sys.argv)


_ensure_venv()

try:
    import cv2
except ImportError:
    print(
        "[ERROR] Missing dependency 'opencv-python'.\n"
        "Install it with:  pip install -r requirements.txt"
    )
    sys.exit(1)

import config
from detector import ObjectDetector
from obstacle_manager import ObstacleManager
from navigation import NavigationEngine
from utils import FpsCounter, draw_detections, draw_hud


def _friendly_exit(message: str, code: int = 1) -> None:
    print(f"\n[ERROR] {message}")
    print("Hint: see README.md for setup and troubleshooting instructions.")
    sys.exit(code)


def main() -> None:
    print("=" * 60)
    print(" AI-Powered Blind Navigation Assistant")
    print("=" * 60)

    # --- 1. Object detector -------------------------------------------------
    detector = ObjectDetector(
        model_path=config.MODEL_PATH,
        target_classes=config.TARGET_CLASSES,
        confidence=config.CONFIDENCE_THRESHOLD,
        iou=config.IOU_THRESHOLD,
        img_size=config.IMG_SIZE,
    )
    try:
        detector.load()
    except Exception as exc:
        _friendly_exit(str(exc))

    device = detector.get_device()
    print(f"[INFO] Model ready: {config.MODEL_PATH.name} (device: {device}).")

    supported = detector.supported_classes()
    missing = set(config.TARGET_CLASSES) - supported
    if missing:
        print(f"[INFO] Not in model: {', '.join(sorted(missing))}")
    print(f"[INFO] Watching for {len(supported & set(config.TARGET_CLASSES))} classes.")

    # --- 3. Pipeline components ---------------------------------------------
    obstacle_mgr = ObstacleManager()
    nav_engine = NavigationEngine()
    fps_counter = FpsCounter()

    # --- 4. Camera ----------------------------------------------------------
    camera = cv2.VideoCapture(config.CAMERA_INDEX)
    if not camera.isOpened():
        _friendly_exit(
            f"Could not open camera (index {config.CAMERA_INDEX}). "
            "Check that a webcam is connected and not used by another app."
        )
    if config.CAMERA_SET_RESOLUTION:
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
    if config.CAMERA_FORCE_MJPG:
        camera.set(
            cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG")
        )
    camera.set(cv2.CAP_PROP_FPS, 30)

    frame_w = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
    obstacle_mgr.set_frame_width(frame_w)
    camera_fps = camera.get(cv2.CAP_PROP_FPS)
    print(f"[INFO] Camera opened (index {config.CAMERA_INDEX}, {frame_w}x{frame_h} @ {camera_fps:.0f}fps).")
    print(f"[INFO] Device: {device} | Press Q to quit.")
    print("-" * 60)

    # --- 5. Main loop -------------------------------------------------------
    # Discard the first frames: some webcams emit dark/exposure-adjusting
    # frames right after opening, which would trigger bogus detections.
    for _ in range(5):
        camera.read()

    empty_frames = 0
    frame_count = 0
    cached_detections = []

    try:
        while True:
            ok, frame = camera.read()
            if not ok or frame is None:
                empty_frames += 1
                if empty_frames >= config.MAX_CONSECUTIVE_EMPTY_FRAMES:
                    print("[ERROR] Camera stopped delivering frames - exiting.")
                    break
                continue
            empty_frames = 0
            frame_count += 1

            # 5.1 Detect (only every Nth frame; reuse cached results otherwise)
            if frame_count % config.DETECT_EVERY_N_FRAMES == 1:
                cached_detections = detector.detect(frame)
            detections = cached_detections

            # 5.2 Obstacle management (priority, position, tracking)
            obstacles = obstacle_mgr.process_frame(detections)

            # 5.3 Navigation decision (visual guidance)
            announcement = nav_engine.get_announcement(obstacles)

            # 5.4 Render
            annotated = draw_detections(frame, obstacles)
            fps = fps_counter.tick()
            annotated = draw_hud(annotated, fps, device, obstacles)
            if announcement:
                y = annotated.shape[0] - 20
                cv2.putText(annotated, announcement, (14, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 5, cv2.LINE_AA)
                cv2.putText(annotated, announcement, (14, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (120, 230, 255), 2, cv2.LINE_AA)
            cv2.imshow(config.WINDOW_NAME, annotated)

            # 5.5 Exit conditions
            key = cv2.waitKey(1) & 0xFF
            if key == config.EXIT_KEY_CODE:
                print("[INFO] Q pressed - quitting.")
                break

            try:
                if cv2.getWindowProperty(config.WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                    print("[INFO] Window closed - quitting.")
                    break
            except cv2.error:
                break

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted (Ctrl+C) - shutting down.")
    except Exception as exc:
        print(f"[ERROR] Unexpected failure in main loop: {exc}")
    finally:
        camera.release()
        cv2.destroyAllWindows()
        print("[INFO] Shutdown complete. Goodbye.")


if __name__ == "__main__":
    main()
