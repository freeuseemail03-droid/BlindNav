"""
scene_summary.py
===========================================================================
Deterministic, rule-based "Describe surroundings" scene summaries.

No cloud LLM is required and no accuracy is claimed: the summary is built
from the objects currently detected plus the free-space analysis. It is a
deliberately conservative description of what the vision pipeline can see.
===========================================================================
"""

import time

import config

# label -> probable environment keyword
_SCENE_HINTS = {
    "bed": "bedroom", "toilet": "bathroom", "sink": "bathroom",
    "refrigerator": "kitchen", "oven": "kitchen", "microwave": "kitchen",
    "laptop": "office", "tv": "living room", "couch": "living room",
}


class SceneSummary:
    def __init__(self) -> None:
        self.enabled = config.SCENE_SUMMARY_ENABLED
        self._last_summary = ""
        self._last_at = 0.0
        self._last_objects: list[dict] = []

    def summarize(self, obstacles: list[dict], free_space: dict | None = None) -> str:
        """Build a concise summary from the current detections."""
        if not self.enabled:
            return ""
        now = time.monotonic()
        labels = [o["label"] for o in obstacles]
        if labels == self._last_objects:
            if now - self._last_at < config.SCENE_SUMMARY_INTERVAL_SECONDS:
                return ""
        self._last_objects = labels
        self._last_at = now

        return self.build_fresh(obstacles, free_space)

    def build_fresh(self, obstacles: list[dict], free_space: dict | None = None) -> str:
        if not obstacles:
            self._last_summary = "The area ahead appears clear."
            return self._last_summary

        # Environment guess (never stated as a certainty).
        hints = [k for k in _SCENE_HINTS if k in [o["label"] for o in obstacles]]
        env = _SCENE_HINTS.get(hints[0]) if hints else None

        parts = []
        seen = set()
        for o in obstacles[: config.SCENE_SUMMARY_MAX_OBJECTS]:
            if o["label"] in seen:
                continue
            seen.add(o["label"])
            if o["position"] == "CENTER":
                parts.append(f"{o['label']} ahead")
            else:
                parts.append(f"{o['label']} on the {o['position'].lower()}")

        if env:
            summary = f"You appear to be in a {env}. "
        else:
            summary = "You appear to be in an open area. "
        summary += ". ".join(parts[:3])
        if len(parts) > 3:
            summary += f", with {len(parts) - 3} other object(s) nearby."

        if free_space and free_space.get("CENTER") == "BLOCKED":
            summary += " The way ahead is blocked."
        self._last_summary = summary
        return summary