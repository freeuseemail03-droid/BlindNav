"""
navigation.py
===========================================================================
Navigation logic: decides what instruction to show the user.

Takes the ranked list of enriched obstacles and produces a single natural
sentence prioritising the most urgent situation. The text is shown on the
camera overlay and dashboard, and (when the user enables voice assistance)
spoken aloud by the browser — so the wording is written to be listened to:
"Bottle is 4 meters on your right."
===========================================================================
"""

import time

import config


def _distance_words(dist: float) -> str:
    """Convert a distance in metres to a natural phrase."""
    if dist < 0.5:
        return "very close"
    elif dist < 1.0:
        return f"{dist:.1f} meters"
    elif dist < 3.0:
        return f"{dist:.0f} meters"
    else:
        return f"{dist:.0f} meters"


def _position_words(position: str) -> str:
    """Convert a position tag to a natural phrase."""
    if position == "LEFT":
        return "on your left"
    elif position == "RIGHT":
        return "on your right"
    return "ahead"


def _build_single_obstacle_message(obstacle: dict) -> str:
    """Build a guidance sentence for a single obstacle.

    The wording is shared by the on-screen guidance and the spoken voice, so it
    is kept natural to listen to: "Bottle is 4 meters on your right."
    """
    label = obstacle["label"]
    dist = obstacle["distance"]
    position = obstacle["position"]
    priority = obstacle["priority"]
    where = _position_words(position)

    if priority == "CRITICAL":
        if dist < 0.5:
            return f"Warning! {label.capitalize()} is very close {where}. Please stop."
        return (f"Warning! {label.capitalize()} is {_distance_words(dist)} "
                f"{where}. Please stop.")

    if priority == "HIGH":
        return f"{label.capitalize()} is {_distance_words(dist)} {where}."

    return f"{label.capitalize()} is {_distance_words(dist)} {where}."


class NavigationEngine:
    """Decides which guidance message to surface from enriched obstacles."""

    def __init__(self) -> None:
        self._last_clear_time = 0.0
        self._last_announced_ids: set = set()

    def get_announcement(self, obstacles: list[dict]) -> str:
        """Return the best guidance message for the current obstacles, or "".

        Parameters
        ----------
        obstacles:
            Ranked list of enriched obstacle dicts from ObstacleManager.
        """
        now = time.monotonic()

        if not obstacles:
            # No obstacles - maybe surface a clear-path message
            if now - self._last_clear_time >= config.CLEAR_PATH_INTERVAL:
                self._last_clear_time = now
                return "Path ahead is clear."
            return ""

        # Pick the top-most urgent obstacle(s)
        top = obstacles[:config.MAX_ANNOUNCED_OBJECTS]

        # If the most urgent is CRITICAL, always surface it immediately
        if top[0]["priority"] == "CRITICAL":
            return _build_single_obstacle_message(top[0])

        # For HIGH: surface if different from what we last said
        if top[0]["priority"] == "HIGH":
            return _build_single_obstacle_message(top[0])

        # For MEDIUM/LOW: surface if there's something new
        if len(top) == 1:
            return _build_single_obstacle_message(top[0])

        # Multiple objects: combine briefly
        parts = []
        for obs in top:
            parts.append(_build_single_obstacle_message(obs))

        # Join with " Also, " for natural flow
        combined = parts[0]
        for part in parts[1:]:
            combined += f" Also, {part[0].lower()}{part[1:]}"
        return combined
