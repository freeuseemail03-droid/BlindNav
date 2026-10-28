"""
free_space.py
===========================================================================
Free-space / walkable-area analysis.

Determines whether the user can plausibly walk LEFT, CENTER or RIGHT based
on where obstacles are in the lower part of the frame (the ground plane near
the user's feet). Output for each column: CLEAR | CAUTION | BLOCKED.

IMPORTANT: this is a heuristic (monocular, no depth). It never overrides a
critical obstacle warning - see path_planner.py for the safety ordering.
===========================================================================
"""

import config


class FreeSpaceAnalyzer:
    def __init__(self) -> None:
        self.enabled = config.FREE_SPACE_ENABLED
        self._columns = config.FREE_SPACE_COLUMNS

    # ------------------------------------------------------------------ API
    def analyze(self, frame_shape: tuple[int, int], obstacles: list[dict]) -> dict:
        """Return {"LEFT": "CLEAR|CAUTION|BLOCKED", "CENTER": ..., "RIGHT": ...}."""
        if not self.enabled or frame_shape is None:
            return {c: "CLEAR" for c in self._columns}
        h, w = frame_shape[:2]
        if h == 0 or w == 0:
            return {c: "CLEAR" for c in self._columns}

        horizon = int(h * (1.0 - config.FREE_SPACE_HORIZON_FRACTION))
        column_w = w / len(self._columns)
        scores: dict[str, int] = {c: 0 for c in self._columns}  # 0=clear 1=caution 2=blocked

        for o in obstacles:
            x1, y1, x2, y2 = o["bbox"]
            if y2 <= horizon:               # below the walkable band -> ignore
                continue
            norm_height = (y2 - y1) / max(h, 1)
            if norm_height < config.FREE_SPACE_MIN_OBSTACLE_HEIGHT:
                continue
            if o["label"] not in config.FREE_SPACE_BLOCK_LABELS:
                continue
            cx = (x1 + x2) / 2.0
            col = min(len(self._columns) - 1, max(0, int(cx / column_w)))
            name = self._columns[col]
            if o["priority"] in ("CRITICAL", "HIGH"):
                scores[name] = 2
            else:
                scores[name] = max(scores[name], 1)

        out = {}
        for col in self._columns:
            out[col] = ("BLOCKED" if scores[col] == 2
                        else ("CAUTION" if scores[col] == 1 else "CLEAR"))
        return out