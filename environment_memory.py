"""
environment_memory.py
===========================================================================
Lightweight session-level environment memory (NOT full SLAM).

Remembers recently seen objects with their approximate position + direction
so the assistant can answer simple questions like "Where was the door?"
Honest by construction: this is a coarse local memory, not a map.
===========================================================================
"""

import time

import config


class EnvironmentMemory:
    def __init__(self) -> None:
        self.enabled = config.ENVIRONMENT_MEMORY_ENABLED
        self._items: list[dict] = []
        self.max_items = config.ENVIRONMENT_MEMORY_MAX_ITEMS
        self.ttl = config.ENVIRONMENT_MEMORY_TTL_SECONDS

    # ------------------------------------------------------------------ API
    def update(self, obstacles: list[dict], now: float | None = None) -> None:
        """Remember the current frame's objects."""
        if not self.enabled:
            return
        now = now or time.time()
        for o in obstacles:
            direction = o.get("position", "CENTER")
            item = {
                "label": o.get("label"),
                "distance": round(o.get("distance", 0.0), 1),
                "direction": direction.lower(),
                "track_id": o.get("track_id"),
                "t": now,
            }
            self._items.append(item)
        # prune TTL + cap
        self._items = [i for i in self._items if now - i["t"] <= self.ttl]
        if len(self._items) > self.max_items:
            self._items = self._items[-self.max_items:]

    def recall(self, label: str | None = None) -> list[dict]:
        """Return remembered objects (optionally filtered by label)."""
        q = (label or "").strip().lower()
        items = self._items
        if q:
            items = [i for i in items if q in i["label"].lower()]
        # one entry per (label, direction, distance bucket)
        seen, out = set(), []
        for i in reversed(items):
            key = (i["label"], i["direction"], round(i["distance"] * 2) / 2)
            if key in seen:
                continue
            seen.add(key)
            out.append(i)
        return out

    def count(self) -> int:
        return len(self._items)