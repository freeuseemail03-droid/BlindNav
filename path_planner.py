"""
path_planner.py
===========================================================================
Safe-path guidance built on top of free-space analysis.

Priority rule (SAFETY FIRST - never override a critical obstruction):
    CRITICAL obstacle   -> "Stop. Path blocked." / explicit warning
    otherwise           -> column guidance:

        CENTER clear        -> "Continue straight."
        RIGHT clear         -> "Move right."
        LEFT clear          -> "Move left."
        all blocked         -> "Stop. No clear path."
        mixed CAUTION       -> prefer the clear column, otherwise advise caution.
===========================================================================
"""

import time

import config


class PathPlanner:
    def __init__(self) -> None:
        self.enabled = config.SAFE_PATH_ENABLED
        self._last_guidance = ""
        self._last_at = 0.0

    def guidance(self, free_space: dict, obstacles: list[dict],
                 force: bool = False) -> str:
        """Return a guidance phrase (or "")."""
        if not self.enabled:
            return ""
        now = time.monotonic()

        # 1) Critical obstruction ALWAYS wins.
        critical = [o for o in obstacles if o.get("priority") == "CRITICAL"]
        if critical:
            top = critical[0]
            self._last_guidance = (
                f"Stop. {top['label']} very close. No clear path.")
            self._last_at = now
            return self._last_guidance

        columns = free_space or {}
        if not columns:
            return ""

        # 2) Prefer center, then right, then left (camera faces forward).
        if columns.get("CENTER") == "CLEAR":
            guidance = "Continue straight."
        elif columns.get("RIGHT") == "CLEAR":
            guidance = "Move right."
        elif columns.get("LEFT") == "CLEAR":
            guidance = "Move left."
        elif all(v == "BLOCKED" for v in columns.values()):
            guidance = "Stop. No clear path."
        else:
            # Mixed caution - advise the least risky direction.
            if columns.get("CENTER") == "CAUTION":
                guidance = "Proceed with caution ahead."
            elif columns.get("RIGHT") == "CAUTION":
                guidance = "Caution. Slight right may help."
            elif columns.get("LEFT") == "CAUTION":
                guidance = "Caution. Slight left may help."
            else:
                guidance = "Proceed with caution."

        # 3) Cooldown so we do not nag continuously.
        if not force and now - self._last_at < config.SAFE_PATH_INTERVAL_SECONDS:
            if guidance == self._last_guidance:
                return ""
        self._last_guidance = guidance
        self._last_at = now
        return guidance