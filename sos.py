"""
sos.py
===========================================================================
Emergency SOS with explicit user confirmation and LOCAL delivery.

Safety rules baked in:
    * SOS is armed only by an explicit user action (the SOS button).
    * A confirmation screen is REQUIRED before anything is recorded.
    * The alert is never described as "sent" unless a delivery method really
      completed. Default methods:
        log       - appends a local sos_log.txt record (works everywhere)
        phone_tel - provides a tel: dial shortcut for a manual phone call
    * Any method that cannot run reports "not configured".
===========================================================================
"""

import json
import threading
import time
from pathlib import Path

import config


class SOSManager:
    def __init__(self, base_path: Path | None = None) -> None:
        self.base = base_path or Path(__file__).resolve().parent
        self.contacts_path = self.base / config.SOS_CONTACTS_FILE
        self.log_path = self.base / "sos_log.txt"
        self._lock = threading.Lock()
        self._contacts: list[dict] = []
        self.state = "idle"            # idle | pending | confirmed
        self._pending_at = 0.0
        self._location: dict | None = None
        self.last_alert: dict | None = None
        self._load()

    # ------------------------------------------------------------- contacts
    def _load(self) -> None:
        if self.contacts_path.exists():
            try:
                self._contacts = json.loads(self.contacts_path.read_text(encoding="utf-8"))
            except Exception:
                self._contacts = []
        if not isinstance(self._contacts, list):
            self._contacts = []

    def _save(self) -> None:
        self.contacts_path.write_text(
            json.dumps(self._contacts, indent=2, ensure_ascii=False), encoding="utf-8")

    def list_contacts(self) -> list[dict]:
        with self._lock:
            return list(self._contacts)

    def set_contacts(self, contacts: list[dict]) -> list[dict]:
        """Replace the contact list (name, phone, email)."""
        cleaned = []
        for c in contacts[:10]:
            name = (c.get("name") or "").strip()
            phone = (c.get("phone") or "").strip()
            email = (c.get("email") or "").strip()
            if name or phone or email:
                cleaned.append({"name": name, "phone": phone, "email": email})
        with self._lock:
            self._contacts = cleaned
            self._save()
        return list(self._contacts)

    # ------------------------------------------------------------- lifecycle
    def arm(self, location: dict | None = None) -> dict:
        """Begin the confirmation window (user must confirm afterwards)."""
        self._location = location
        self.state = "pending"
        self._pending_at = time.time()
        return {"state": self.state}

    def confirm(self) -> dict:
        """User confirmed: build + deliver the alert."""
        if self.state != "pending":
            return {"ok": False, "message": "No pending SOS to confirm.",
                    "state": self.state}
        alert = self._build_alert()
        deliveries = self._deliver(alert)
        alert["deliveries"] = deliveries
        alert["delivered"] = any(d.get("delivered") for d in deliveries)
        self.last_alert = alert
        self.state = "confirmed"
        return {"ok": True, "state": self.state, "alert": alert,
                "message": ("Emergency alert recorded locally."
                            if alert["delivered"] else
                            "No delivery method configured; nothing sent.")}

    def cancel(self) -> dict:
        was = self.state
        self.state = "idle"
        return {"ok": was == "pending", "state": self.state,
                "message": "SOS cancelled." if was == "pending" else "No pending SOS."}

    # ------------------------------------------------------------------ misc
    def status(self) -> dict:
        methods = []
        for method in config.SOS_DELIVERY_METHODS:
            methods.append({
                "method": method,
                "configured": method == "log",            # log always works
                "description": ("Local log file" if method == "log"
                                else "Manual tel: dial shortcut (requires phone)"),
            })
        return {
            "enabled": config.SOS_ENABLED,
            "state": self.state,
            "contacts_count": len(self._contacts),
            "contacts": self.list_contacts(),
            "delivery_methods": methods,
            "countdown": (config.SOS_CONFIRM_TIMEOUT_SECONDS
                          - (time.time() - self._pending_at)
                          if self.state == "pending" else None),
            "last_alert": self.last_alert,
        }

    # ---------------------------------------------------------- internals
    def _build_alert(self) -> dict:
        loc = self._location or {}
        msg = "EMERGENCY SOS from BlindNav AI - the user needs help."
        extra = []
        if loc.get("latitude"):
            extra.append(f"Location: {loc['latitude']:.5f},{loc['longitude']:.5f}"
                         f" (approx {loc.get('accuracy')} m)")
        elif loc.get("text"):
            extra.append(f"Location: {loc['text']}")
        if extra:
            msg += " " + " ".join(extra)
        return {"message": msg, "ts": time.time(), "location": loc,
                "contacts": self.list_contacts()}

    def _deliver(self, alert: dict) -> list[dict]:
        deliveries = []
        if "log" in config.SOS_DELIVERY_METHODS:
            try:
                line = (f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] "
                        f"{alert['message']}\n")
                with open(self.log_path, "a", encoding="utf-8") as f:
                    f.write(line)
                deliveries.append({"method": "log", "delivered": True,
                                   "detail": str(self.log_path)})
            except Exception as exc:
                deliveries.append({"method": "log", "delivered": False,
                                   "error": str(exc)})
        if "phone_tel" in config.SOS_DELIVERY_METHODS:
            # A tel: link cannot be dialled automatically; it is a manual step.
            phones = [c["phone"] for c in alert["contacts"] if c.get("phone")]
            deliveries.append({
                "method": "phone_tel",
                "delivered": False,       # honest: NOT sent automatically
                "manual_action": "true",
                "tel_links": [f"tel:{p}" for p in phones],
                "note": "Manual telephone call required - nothing dialled automatically.",
            })
        return deliveries