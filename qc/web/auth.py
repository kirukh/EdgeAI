"""Access control of the web UI.

* **Setup PIN** (``ui.setup_pin``): unlocks the setup area. The PIN is checked on the server;
  the page receives a random session token (``X-Setup-Token``) that expires after
  ``ui.setup_timeout_min`` minutes without setup activity. After 5 wrong PINs *from one
  device* input from that device is blocked for 60 s (a wrong PIN on one tablet does not
  lock out the technician on another).
* **CSRF protection**: every modifying request must carry the header ``X-QC-Client``. A
  foreign web page in the operator's browser cannot set custom headers on cross-site
  requests without a CORS preflight, which this server never allows (OWASP "custom request
  header" defence).
* **Optional network token** (environment variable ``QC_TOKEN``): all modifying requests
  must then also send ``X-QC-Token`` (open the page with ``?token=...``).
"""

from __future__ import annotations

import hmac
import os
import secrets
import threading
import time
from dataclasses import dataclass, field

from flask import Request

from ..config import UIConfig

MAX_FAILED = 5
LOCKOUT_S = 60
CLIENT_HEADER = "X-QC-Client"


@dataclass
class _Attempts:
    failed: int = 0
    blocked_until: float = 0.0


@dataclass
class SetupAuth:
    ui: UIConfig
    network_token: str = field(default_factory=lambda: os.environ.get("QC_TOKEN", ""))
    _sessions: dict[str, float] = field(default_factory=dict)  # setup token → last activity
    _attempts: dict[str, _Attempts] = field(default_factory=dict)  # client address → failed PINs
    _guard: threading.Lock = field(default_factory=threading.Lock)

    @property
    def pin_required(self) -> bool:
        return bool(self.ui.setup_pin)

    def check_request(self, request: Request) -> str | None:
        """None if the request may pass, otherwise the reason for refusing it (HTTP 403)."""
        if request.method not in ("POST", "PUT", "DELETE"):
            return None
        if request.headers.get(CLIENT_HEADER) != "1":
            return "Missing X-QC-Client header (cross-site request?)."
        if self.network_token and not hmac.compare_digest(request.headers.get("X-QC-Token", ""), self.network_token):
            return "Invalid network token."
        return None

    def _expire(self, now: float) -> None:
        timeout = self.ui.setup_timeout_min * 60
        for k in [k for k, t in self._sessions.items() if now - t > timeout]:
            del self._sessions[k]

    def any_unlocked(self) -> bool:
        """Is the setup area open on any device (sessions that timed out are closed here)?"""
        if not self.pin_required:
            return True
        with self._guard:
            self._expire(time.time())
            return bool(self._sessions)

    def unlocked(self, request: Request) -> bool:
        if not self.pin_required:
            return True
        sent = request.headers.get("X-Setup-Token", "")
        now = time.time()
        with self._guard:
            self._expire(now)
            if sent and sent in self._sessions:
                self._sessions[sent] = now
                return True
        return False

    def unlock(self, pin: str, client: str) -> tuple[str | None, str, int]:
        """→ (session token or None, error text, HTTP status). Wrong PINs are counted per client."""
        if not self.pin_required:
            return "", "", 200
        now = time.time()
        with self._guard:
            att = self._attempts.setdefault(client, _Attempts())
            if now < att.blocked_until:
                return None, f"Too many wrong PINs – wait {int(att.blocked_until - now) + 1} s.", 429
            if not hmac.compare_digest(pin.encode(), self.ui.setup_pin.encode()):
                att.failed += 1
                if att.failed >= MAX_FAILED:
                    att.failed, att.blocked_until = 0, now + LOCKOUT_S
                return None, "Wrong PIN.", 403
            att.failed = 0
            token = secrets.token_urlsafe(24)
            self._sessions[token] = now
            return token, "", 200

    def lock(self, token: str) -> None:
        with self._guard:
            self._sessions.pop(token, None)
