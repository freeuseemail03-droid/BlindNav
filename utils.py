"""
utils.py
===========================================================================
Generic helper utilities: FPS counter, drawing helpers.
===========================================================================
"""

import time

import cv2
import numpy as np

import config
from detector import Detection

# ---------------------------------------------------------------------------
# Drawing constants
# ---------------------------------------------------------------------------

_PRIORITY_COLORS = {
    "CRITICAL": (0, 0, 255),    # red
    "HIGH":     (0, 128, 255),  # orange
    "MEDIUM":   (0, 255, 255),  # yellow
    "LOW":      (0, 255, 0),    # green
}

_PALETTE = [
    (70, 70, 220), (70, 220, 70), (220, 70, 70), (70, 200, 220),
    (200, 70, 220), (220, 200, 70), (200, 200, 200), (70, 140, 220),
]

_IRREGULAR_PLURALS = {
    "person": "people",
    "knife": "knives",
    "leaf": "leaves",
}


# ---------------------------------------------------------------------------
# FPS measurement
# ---------------------------------------------------------------------------

class FpsCounter:
    """Exponential moving average FPS counter."""

    def __init__(self, smoothing: float = 0.1) -> None:
        self._smoothing = smoothing
        self._ema = 0.0
        self._last_time = None

    def tick(self) -> float:
        now = time.monotonic()
        if self._last_time is not None:
            dt = now - self._last_time
            if dt > 0.0:
                instant_fps = 1.0 / dt
                if self._ema == 0.0:
                    self._ema = instant_fps
                else:
                    self._ema = (
                        self._smoothing * instant_fps
                        + (1.0 - self._smoothing) * self._ema
                    )
        self._last_time = now
        return self._ema

    @property
    def fps(self) -> float:
        return self._ema


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _color_for(label: str) -> tuple:
    seed = sum((i + 1) * ord(ch) for i, ch in enumerate(label))
    return _PALETTE[seed % len(_PALETTE)]


# ---------------------------------------------------------------------------
# Low-light analysis + enhancement
# ---------------------------------------------------------------------------

def estimate_brightness(frame: np.ndarray) -> float:
    """Mean grayscale intensity of the frame (0..255)."""
    if frame is None or frame.size == 0:
        return 255.0
    if frame.ndim == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame
    return float(np.mean(gray))


def is_low_light(brightness: float,
                 threshold: float = None) -> bool:
    threshold = threshold if threshold is not None else config.LOW_LIGHT_BRIGHTNESS_THRESHOLD
    return brightness < threshold


def enhance_low_light(frame: np.ndarray) -> np.ndarray:
    """Improve visibility without over-processing bright frames.

    Applies gamma correction (+ optional CLAHE) only to the luminance if the
    caller has already decided the scene is dark.  Returns a new image.
    """
    if frame is None or frame.size == 0:
        return frame
    frame = frame.copy()
    if frame.ndim == 3:
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = cv2.pow(l / 255.0, config.LOW_LIGHT_ENHANCE_GAMMA) * 255.0
        l = l.astype(np.uint8)
        if config.LOW_LIGHT_ENHANCE_CLAHE:
            clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
            l = clahe.apply(l)
        frame = cv2.merge((l, a, b))
        return cv2.cvtColor(frame, cv2.COLOR_LAB2BGR)
    l = cv2.pow(frame / 255.0, config.LOW_LIGHT_ENHANCE_GAMMA) * 255.0
    return l.astype(np.uint8)


def draw_detections(frame: np.ndarray, obstacles: list[dict]) -> np.ndarray:
    """Draw bounding boxes, labels, distance, position and priority.

    Draws in place on ``frame`` (and returns it) to avoid a costly
    full-image copy on every iteration of the real-time loop.
    """
    if frame is None or frame.size == 0:
        return frame

    canvas = frame
    h, w = canvas.shape[:2]

    for obs in obstacles:
        x1, y1, x2, y2 = obs["bbox"]
        label = obs["label"]
        conf = obs["confidence"]
        dist = obs["distance"]
        position = obs["position"]
        priority = obs["priority"]

        color = _PRIORITY_COLORS.get(priority, (200, 200, 200))

        # Bounding box
        thickness = 3 if priority in ("CRITICAL", "HIGH") else 2
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)

        # Multi-line label block
        lines = [
            f"{label.upper()}",
            f"{conf:.0%}",
            f"{dist:.1f}m",
            position,
            priority,
        ]
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.45
        line_thickness = 1
        line_height = 16
        padding = 4

        label_top = y1 - padding
        block_height = len(lines) * line_height + padding
        if label_top - block_height < 0:
            label_top = y1 + block_height + padding

        max_text_w = 0
        for line in lines:
            (tw, _), _ = cv2.getTextSize(line, font, font_scale, line_thickness)
            if tw > max_text_w:
                max_text_w = tw

        cv2.rectangle(
            canvas,
            (x1, label_top - block_height),
            (x1 + max_text_w + padding * 2, label_top),
            color,
            thickness=-1,
        )

        for i, line in enumerate(lines):
            ty = label_top - block_height + padding + (i + 1) * line_height - 3
            txt_color = (0, 0, 0) if priority in ("CRITICAL", "HIGH", "MEDIUM") else (255, 255, 255)
            cv2.putText(canvas, line, (x1 + padding, ty), font, font_scale, txt_color, line_thickness, cv2.LINE_AA)

    return canvas


def draw_hud(canvas: np.ndarray, fps: float, device: str, obstacles: list[dict]) -> np.ndarray:
    """Draw FPS, device info, and status overlay."""
    h, w = canvas.shape[:2]

    # Top-left: FPS + Device
    info_lines = [
        f"Device: {device}",
        f"FPS: {fps:.1f}",
        f"Objects: {len(obstacles)}",
    ]
    font = cv2.FONT_HERSHEY_SIMPLEX
    y_pos = 25
    for line in info_lines:
        cv2.putText(canvas, line, (10, y_pos), font, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
        y_pos += 25

    # Bottom: most urgent obstacle
    if obstacles:
        top = obstacles[0]
        status = f"{top['priority']}: {top['label']} {top['distance']:.1f}m {top['position']}"
        color = _PRIORITY_COLORS.get(top["priority"], (200, 200, 200))
        cv2.putText(canvas, status, (10, h - 15), font, 0.55, color, 2, cv2.LINE_AA)
    else:
        cv2.putText(canvas, "No obstacles detected", (10, h - 15), font, 0.55, (0, 255, 0), 2, cv2.LINE_AA)

    return canvas
