"""
navigation_detector.py
===========================================================================
Custom navigation-specific object detector (optional second YOLO model).

Detects navigation hazards that a general COCO model does not know:
door, stairs_up, stairs_down, curb, pothole, crosswalk, traffic_light,
zebra_crossing.

The model weights are NOT shipped with the project (they must be trained /
provided by the user). Until a weights file exists at
config.NAVIGATION_MODEL_PATH the module reports "MODEL NOT INSTALLED" and
the main pipeline continues with YOLO11n only. It never crashes the app and
it never fabricates detections for an absent model.
===========================================================================
"""

import numpy as np

import config
from detector import Detection


class NavigationObjectDetector:
    def __init__(self) -> None:
        self.path = config.NAVIGATION_MODEL_PATH
        self.enabled = config.NAVIGATION_MODEL_ENABLED
        self.confidence = config.NAVIGATION_MODEL_CONFIDENCE
        self.target_classes = config.NAVIGATION_MODEL_CLASSES
        self._model = None
        self._error: str | None = None
        self._state = "DISABLED" if not self.enabled else "MODEL NOT INSTALLED"
        self._device = "CPU"

    # ------------------------------------------------------------------ API
    def load(self) -> None:
        if not self.enabled:
            return
        if self._model is not None:
            return
        if not self.path.exists():
            self._state = "MODEL NOT INSTALLED"
            self._error = (
                "Navigation model weights not found at "
                f"'{self.path}'. Train/provide a model and place it there, "
                "or disable NAVIGATION_MODEL_ENABLED. Using YOLO11n only."
            )
            print(f"[WARN] {self._error}")
            return
        try:
            from ultralytics import YOLO
            self._model = YOLO(str(self.path), task="detect")
            self._state = "READY"
            try:
                names = getattr(self._model, "names", None)
                if isinstance(names, dict):
                    self.target_classes = [CLASS_ALIAS(name) for name in names.values()]
            except Exception:
                pass
        except Exception as exc:
            self._state = "ERROR"
            self._error = str(exc)

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Detect navigation objects in a frame (empty when model is absent)."""
        if self._model is None or frame is None or frame.size == 0:
            return []
        try:
            results = self._model.predict(
                source=frame, imgsz=config.IMG_SIZE,
                conf=self.confidence, verbose=False, device=self._device,
            )
        except Exception as exc:
            print(f"[WARN] Navigation model inference failed: {exc}")
            return []
        if not results or results[0].boxes is None:
            return []
        names = results[0].names
        out = []
        for box in results[0].boxes:
            try:
                class_id = int(box.cls[0])
                label = CLASS_ALIAS(names.get(class_id, f"class_{class_id}"))
                conf = float(box.conf[0])
                x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
                out.append(Detection(
                    label=label, confidence=conf, box=(x1, y1, x2, y2),
                    center=((x1 + x2) // 2, (y1 + y2) // 2),
                ))
            except (IndexError, TypeError, ValueError):
                continue
        return out

    def status(self) -> dict:
        return {
            "enabled": self.enabled,
            "available": self._model is not None,
            "status": self._state,
            "path": str(self.path),
            "error": self._error,
            "classes": self.target_classes,
        }


def CLASS_ALIAS(name: str) -> str:
    return {"traffic light": "traffic_light",
            "zebra crossing": "zebra_crossing"}.get(name, name)