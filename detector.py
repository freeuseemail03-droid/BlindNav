"""
detector.py
===========================================================================
Object detection module built on top of Ultralytics YOLO11n (pretrained on
the COCO dataset), served through ONNX Runtime for fast CPU inference.

Responsibilities
----------------
* Locate (and, when missing, download) the pretrained YOLO11n weights.
* Run inference on a single BGR frame coming from the webcam.
* Return a list of Detection objects with class, confidence, bounding box
  and center coordinates.
===========================================================================
"""

import shutil
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

import config

try:
    from ultralytics import YOLO
    from ultralytics.utils.downloads import attempt_download_asset
except ImportError as _import_error:
    raise ImportError(
        "The 'ultralytics' package is required.\n"
        "Install it with:  pip install -r requirements.txt"
    ) from _import_error

CLASS_ALIASES = {
    "dining table": "table",
    "potted plant": "plant",
    "sports ball": "ball",
    "hair drier": "hair dryer",
}


@dataclass
class Detection:
    """A single enriched object detection.

    Attributes
    ----------
    label:
        Human-readable class name.
    confidence:
        Model confidence score 0.0 .. 1.0.
    box:
        Bounding box (x1, y1, x2, y2) in pixels.
    center:
        Centre of the bounding box (cx, cy) in pixels.
    """

    label: str
    confidence: float
    box: tuple
    center: tuple = field(default=(0, 0))


class ObjectDetector:
    """Wraps YOLO11n: model loading, inference, filtering."""

    def __init__(
        self,
        model_path: Path,
        target_classes: list,
        confidence: float,
        iou: float,
        img_size: int,
    ) -> None:
        self._model_path = Path(model_path)
        self._target_classes = set(target_classes)
        self._confidence = confidence
        self._iou = iou
        self._img_size = img_size
        self._model = None
        self._device = "cpu"
        self._half = False
        self._half_kwargs: dict = {"half": True}

    def load(self) -> None:
        """Load the YOLO model, downloading weights if necessary."""
        self._model_path.parent.mkdir(parents=True, exist_ok=True)

        if not self._model_path.exists():
            self._download_weights()

        if not self._model_path.exists():
            raise RuntimeError(
                f"Model weights were not found at '{self._model_path}'. "
                "Check your internet connection and try again."
            )

        try:
            self._model = YOLO(str(self._model_path), task="detect")
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load the YOLO model from '{self._model_path}': {exc}\n"
                "The weight file may be corrupted or truncated - delete it and "
                "re-run so it is downloaded again."
            ) from exc

        # Warm up: the first inference builds internal state and can take
        # several seconds; do it here instead of on the first camera frame.
        try:
            self._model.predict(
                source=np.zeros((self._img_size, self._img_size, 3), dtype=np.uint8),
                imgsz=self._img_size,
                conf=self._confidence,
                iou=self._iou,
                verbose=False,
                device=self._device,
            )
        except Exception:
            pass

        try:
            import torch
            if torch.cuda.is_available():
                self._device = "cuda"
                self._half = True
        except Exception:
            pass
        if self._device == "cuda":
            try:
                self._model.to(self._device)
            except Exception:
                pass
        # Work out the FP16 predictor kwarg for this ultralytics build.
        self._half_kwargs = self._resolve_fp16_kwarg()

    def _resolve_fp16_kwarg(self) -> dict:
        """Return the FP16 predict kwarg supported by this ultralytics build.

        Modern builds use ``quantize=16``; older ones use the now-deprecated
        ``half=True``. The deprecated form prints a warning *on every frame*,
        which floods the console/log; we detect the supported form once here
        (with a throwaway probe) instead of paying that cost per frame.
        """
        probe = np.zeros((self._img_size, self._img_size, 3), dtype=np.uint8)
        try:
            self._model.predict(
                source=probe, imgsz=self._img_size, conf=self._confidence,
                verbose=False, device="cpu", quantize=16,
            )
            return {"quantize": 16}
        except Exception:
            return {"half": True}

    def detect(self, frame: np.ndarray) -> list:
        """Run inference on one frame and return filtered detections."""
        if self._model is None:
            raise RuntimeError("Detector is not loaded; call load() first.")
        if frame is None or frame.size == 0:
            return []

        try:
            kwargs = dict(
                source=frame,
                imgsz=self._img_size,
                conf=self._confidence,
                iou=self._iou,
                verbose=False,
                device=self._device,
            )
            if self._half:
                kwargs.update(self._half_kwargs)
            results = self._model.predict(**kwargs)
        except Exception as exc:
            print(f"[WARN] Inference failed, frame skipped: {exc}")
            return []

        if not results or results[0].boxes is None:
            return []

        return self._parse_results(results[0], frame)

    def supported_classes(self) -> set:
        """Return the set of class names the loaded model can detect."""
        if self._model is None:
            return set()
        names = getattr(self._model, "names", None)
        if not isinstance(names, dict):
            return set()
        return {CLASS_ALIASES.get(name, name) for name in names.values()}

    def get_device(self) -> str:
        """Return 'CUDA' or 'CPU' indicating which device is being used."""
        return "CUDA" if self._device == "cuda" else "CPU"

    def set_confidence(self, value: float) -> None:
        """Update the minimum confidence threshold used at inference time."""
        self._confidence = max(0.05, min(float(value), 0.95))

    @property
    def model_name(self) -> str:
        """File name of the loaded weights."""
        return self._model_path.name

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _download_weights(self) -> None:
        """Download the official YOLO11n weights into the models/ folder."""
        print("[INFO] YOLO11n weights not found - downloading (one-time)...")

        try:
            attempt_download_asset(str(self._model_path))
            return
        except Exception as exc:
            print(f"[WARN] Download attempt 1 failed: {exc}")

        try:
            attempt_download_asset(str(self._model_path), release="v8.4.0")
            return
        except Exception as exc:
            print(f"[WARN] Download attempt 2 failed: {exc}")

        try:
            YOLO(str(config.BASE_DIR / config.MODEL_DOWNLOAD_NAME))
            downloaded = config.BASE_DIR / config.MODEL_DOWNLOAD_NAME
            if downloaded.exists():
                shutil.move(str(downloaded), str(self._model_path))
        except Exception as exc:
            raise RuntimeError(
                "Could not download the YOLO11n weights. Check your internet "
                f"connection and try again.\nDetails: {exc}"
            ) from exc

    def _parse_results(self, result, frame: np.ndarray) -> list:
        """Convert one Ultralytics result object into filtered Detections."""
        names = result.names
        h, w = frame.shape[:2]
        detections = []

        for box in result.boxes:
            try:
                class_id = int(box.cls[0])
                raw_name = names.get(class_id, f"class_{class_id}")
                name = CLASS_ALIASES.get(raw_name, raw_name)
                if name not in self._target_classes:
                    continue
                confidence = float(box.conf[0])
                if confidence < self._confidence:
                    continue
                x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                detections.append(
                    Detection(
                        label=name,
                        confidence=confidence,
                        box=(x1, y1, x2, y2),
                        center=(cx, cy),
                    )
                )
            except (IndexError, TypeError, ValueError) as exc:
                print(f"[WARN] Skipped a malformed detection: {exc}")
                continue

        return detections
