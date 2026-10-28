"""
history_store.py
===========================================================================
In-memory ring buffer of navigation events for the History page.

Events are stored as plain dicts:
    {"ts": epoch, "type": "OBSTACLE|FAMILY|VEHICLE|PATH|SCENE|SOS",
     "label": "...", "distance": 0.7, "position": "CENTER", "priority": "CRITICAL",
     "text": "..."}    # text used for SOS / guidance

Duplicates within HISTORY_MERGE_SECONDS are collapsed so the list does not
become a wall of identical alarms.
===========================================================================
"""

import threading
import time
from collections import deque

import config


class HistoryStore:
    def __init__(self, max_entries: int = config.HISTORY_MAX_ENTRIES) -> None:
        self._lock = threading.Lock()
        self._entries: deque = deque(maxlen=max_entries)
        self._last: dict | None = None

    # ------------------------------------------------------------------ write
    def add(self, event: dict) -> dict:
        event = dict(event)
        event.setdefault("ts", time.time())
        with self._lock:
            now = time.time()
            key = (event.get("type"), event.get("label"),
                   event.get("priority"), event.get("text"))
            if self._last and key == self._last["_key"] and \
                    now - self._last["ts"] < config.HISTORY_MERGE_SECONDS:
                self._last["ts"] = now      # refresh the duplicate's time
                self._last["count"] = self._last.get("count", 1) + 1
                return self._last
            event["_key"] = key
            event["count"] = 1
            self._entries.append(event)
            self._last = event
            return event

    # ------------------------------------------------------------------- read
    def list(self, limit: int = 100) -> list[dict]:
        """Newest first."""
        with self._lock:
            out = [dict(e) for e in self._entries]
        out.reverse()
        for e in out:
            e.pop("_key", None)
        return out[:limit]

    def clear(self) -> int:
        with self._lock:
            n = len(self._entries)
            self._entries.clear()
            self._last = None
        return n