"""
vehicle_awareness.py
===========================================================================
Intelligent vehicle awareness.

Combines YOLO vehicle detections with temporal tracking to decide whether a
vehicle is stationary, approaching, moving away, or uncertain -- based on
SUFFICIENT temporal evidence only. The system never claims a vehicle is
moving from a single frame.

States
------
stationary   - enough samples, distance roughly constant
approaching  - enough samples, distance shrinking beyond the threshold
moving_away  - enough samples, distance growing
uncertain    - not enough temporal evidence yet

Wording is deliberately cautious: "Vehicle may be approaching" unless the
trend is very clear.
===========================================================================
"""

import time

import config


def _where_words(position: str) -> str:
    """'on your left' | 'on your right' | 'directly ahead'."""
    return {"LEFT": "on your left", "RIGHT": "on your right"}.get(
        position, "directly ahead")


def _side_words(position: str) -> str:
    """'the left' | 'the right' | 'ahead' — used after 'approaching from'."""
    return {"LEFT": "the left", "RIGHT": "the right"}.get(position, "ahead")


class VehicleAwareness:
    def __init__(self) -> None:
        self._history: dict[int, list[dict]] = {}   # track_id -> samples
        self._max_history = 30
        self.enabled = config.VEHICLE_WARNING_ENABLED
        self._last_warning = ""
        self._last_warning_at = 0.0

    # ------------------------------------------------------------------ API
    def update(self, obstacles: list[dict], now: float | None = None) -> list[dict]:
        """Feed the current frame's obstacles; returns vehicle status dicts."""
        now = now or time.time()
        if not self.enabled:
            return []
        vehicle_ids = {id(o) for o in obstacles
                       if o.get("label") in config.VEHICLE_CLASSES}

        # Retire tracks that disappeared.
        for tid in list(self._history):
            if tid not in vehicle_ids:
                if now - self._history[tid][-1]["t"] > 2.0:
                    self._history.pop(tid, None)

        out = []
        for o in obstacles:
            if o.get("label") not in config.VEHICLE_CLASSES:
                continue
            tid = o["track_id"]
            dist = o["distance"]
            pos = o["position"]
            self._history.setdefault(tid, []).append(
                {"t": now, "dist": dist, "pos": pos})
            h = self._history[tid]
            if len(h) > self._max_history:
                del h[:len(h) - self._max_history]

            state = self._state(h)
            out.append({
                "track_id": tid,
                "label": o["label"],
                "distance": dist,
                "position": pos,
                "state": state,           # stationary|approaching|moving_away|uncertain
                "confidence": self._confidence(h),
                "samples": len(h),
            })
        return out

    def warnings(self, vehicle_states: list[dict],
                 path_columns: dict | None = None) -> list[str]:
        """Produce warning phrases for the current vehicle states.

        Each phrase carries the distance and, for a real threat, the evasive
        advice ("Move to your right.") chosen from the walkable-path columns so
        the recommendation never points into a blocked lane.
        """
        phrases = []
        for v in vehicle_states:
            label = v["label"].capitalize()
            dist = v["distance"]
            pos = v["position"]
            state = v["state"]
            where = _where_words(pos)
            side = _side_words(pos)
            if dist <= config.VEHICLE_VERY_CLOSE_DISTANCE:
                phrases.append(
                    f"Warning! {label} very close {where}. "
                    f"{self._evasive_advice(pos, path_columns)}")
            elif state == "approaching" and dist <= config.VEHICLE_APPROACH_MAX_DISTANCE:
                if v["confidence"] >= 2.0:
                    phrases.append(
                        f"Warning! {label} approaching from {side}, "
                        f"{dist:.0f} meters away. "
                        f"{self._evasive_advice(pos, path_columns)}")
                else:
                    phrases.append(
                        f"{label} may be approaching from {side}, "
                        f"{dist:.0f} meters. "
                        f"{self._evasive_advice(pos, path_columns)}")
            elif state == "stationary":
                phrases.append(f"{label} {dist:.0f} meters {where}.")
            # moving_away / uncertain -> no alarm (do not distract)
        return phrases

    @staticmethod
    def _evasive_advice(position: str, path_columns: dict | None) -> str:
        """Tell the user which way to step.

        Never points into the vehicle's own lane, and prefers a walkable column
        that free-space analysis reports as CLEAR over one that is only CAUTION.
        """
        if position in ("LEFT", "RIGHT"):
            away = "RIGHT" if position == "LEFT" else "LEFT"
            order = [away, "CENTER"]          # step away, else straight ahead
        else:
            order = ["RIGHT", "LEFT", "CENTER"]   # vehicle ahead: pick a side

        def state_of(col: str) -> str:
            return (path_columns or {}).get(col, "CLEAR")

        for wanted in ("CLEAR", "CAUTION"):
            for col in order:
                if state_of(col) != wanted:
                    continue
                if col == "CENTER":
                    return ("Move forward carefully." if wanted == "CAUTION"
                            else "Continue forward.")
                return f"Move to your {col.lower()}."
        return "Stop."

    def top_warning(self, warnings: list[str]) -> str:
        """Merge vehicle warnings into a single concise phrase."""
        if not warnings:
            self._last_warning_at = 0.0
            self._last_warning = ""
            return ""
        phrase = warnings[0]
        self._last_warning = phrase
        self._last_warning_at = time.time()
        return phrase

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def _state(history: list[dict]) -> str:
        if len(history) < config.VEHICLE_MOTION_MIN_SAMPLES:
            return "uncertain"
        window = history[-(config.VEHICLE_MOTION_MIN_SAMPLES + 2):]
        d0 = window[0]["dist"]
        d1 = window[-1]["dist"]
        delta = d1 - d0
        span = window[-1]["t"] - window[0]["t"]
        if span <= 0:
            return "uncertain"
        rate = delta / max(span, 0.5)
        if rate < -config.VEHICLE_APPROACH_DISTANCE_DELTA / 3.0:
            return "approaching"
        if rate > config.VEHICLE_APPROACH_DISTANCE_DELTA / 3.0:
            return "moving_away"
        return "stationary"

    @staticmethod
    def _confidence(history: list[dict]) -> float:
        """0..3 how much temporal evidence we have (samples + consistency)."""
        if len(history) < config.VEHICLE_MOTION_MIN_SAMPLES:
            return 0.0
        dists = [h["dist"] for h in history[-8:]]
        spread = max(dists) - min(dists)
        if spread < 0.05:
            return 3.0   # very consistent stationary
        return min(2.5, 1.0 + len(history) / 10.0)