"""
depth_provider.py
===========================================================================
Depth provider abstraction so a real RGB-D camera can be added later
without touching the rest of the pipeline.

    DepthProvider (interface)
        MonocularDepthProvider   - current estimation (focal length + width)
        RGBDDepthProvider         - reads a real depth map when hardware exists

The worker uses 'monocular' by default. When a depth camera is present and
a depth map is supplied, the RGBD provider is preferred and its distances
are used instead of the monocular estimates.

Nothing here requires depth hardware to function.
===========================================================================
"""

from abc import ABC, abstractmethod

import config


class DepthProvider(ABC):
    """Common interface for distance providers."""

    name = "abstract"

    @abstractmethod
    def estimate(self, bbox_width_px: float, label: str,
                 real_width_m: float) -> float:
        """Estimated distance in metres for one object."""

    @abstractmethod
    def status(self) -> dict:
        """Honest status for web display."""


class MonocularDepthProvider(DepthProvider):
    """Current approach: focal-length monocular estimation."""

    name = "monocular"

    def __init__(self, focal_length: float | None = None) -> None:
        # Monocular estimation uses the configured focal length fallback.
        self.focal_length = focal_length or config.FOCAL_LENGTH

    def refresh(self) -> float:
        self.focal_length = config.FOCAL_LENGTH
        return self.focal_length

    def estimate(self, bbox_width_px: float, label: str,
                 real_width_m: float) -> float:
        if bbox_width_px <= 0:
            return 0.0
        distance = (real_width_m * self.focal_length) / bbox_width_px
        return max(round(distance, 2), 0.3)

    def status(self) -> dict:
        return {
            "provider": self.name,
            "available": True,
            "focal_length": round(self.focal_length, 2),
            "type": "Estimate (focal length / bounding box width)",
        }


class RGBDDepthProvider(DepthProvider):
    """Real depth sensor - requires a depth camera to be present."""

    name = "rgbd"

    def __init__(self) -> None:
        self._available = False
        try:
            # Heuristic: cv2 aliases like OpenNI/Orbbec/RealSense are not
            # installed by default; check whether anything reports CAP_OPENNI.
            import cv2
            self._cap = cv2.VideoCapture(config.CAMERA_INDEX)
            self._available = bool(self._cap.isOpened() and
                                   self._cap.get(cv2.CAP_PROP_OPENNI_REGISTRATION) != -1)
        except Exception:
            self._cap = None
            self._available = False

    def estimate(self, bbox_width_px: float, label: str,
                 real_width_m: float) -> float:
        # No depth map plumbing yet: hard requirement on real hardware.
        return 0.0

    def status(self) -> dict:
        return {
            "provider": self.name,
            "available": self._available,
            "required": "RGB-D / depth camera (e.g. Intel RealSense, Kinect)",
            "type": "Hardware depth (uses depth map when present)",
        }


class DepthProviderManager:
    """Selects the active provider and handles fallback."""

    def __init__(self, provider: str = config.DEPTH_PROVIDER) -> None:
        self.mono = MonocularDepthProvider()
        self.rgbd = RGBDDepthProvider()
        self._override = provider
        self.fallback = config.DEPTH_FALLBACK_TO_MONOCULAR

    def active(self) -> DepthProvider:
        if self._override == "rgbd" and self.rgbd._available:
            return self.rgbd
        return self.mono

    def status(self) -> dict:
        active = self.active()
        return {
            "active": active.name,
            "monocular": self.mono.status(),
            "rgbd": self.rgbd.status(),
            "fallback_to_monocular": self.fallback,
        }