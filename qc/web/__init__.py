"""Web user interface.

Why a web UI instead of a local app: the Pi runs headless at the conveyor and is operated
from a tablet or laptop on the same network. The page has no external dependencies (works
offline). Two access levels, one blueprint each:

* ``operator_api`` – supervisor, no PIN: start/stop, select part, teach in, “Result wrong?”,
  finish batch.
* ``setup_api``    – setup technician, PIN: sensitivity, models, focus assistant, archives,
  simulator.

Access control and CSRF protection: see ``auth.py``.
"""

from __future__ import annotations

from pathlib import Path

from flask import Flask, Response, jsonify, request, send_from_directory
from flask.typing import ResponseReturnValue

from ..model import TrainingError
from ..system import QCSystem
from . import operator_api, setup_api
from .auth import SetupAuth

STATIC = Path(__file__).parent / "static"
SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    # only own scripts; inline style attributes are used by the UI templates
    "Content-Security-Policy": "default-src 'self'; img-src 'self' data: blob:; style-src 'self' 'unsafe-inline'; "
    "frame-ancestors 'none'",
}


def create_app(system: QCSystem) -> Flask:
    app = Flask(__name__, static_folder=None)
    auth = SetupAuth(system.cfg.ui)
    app.extensions["qc_system"] = system
    app.extensions["qc_auth"] = auth
    app.register_blueprint(operator_api.bp)
    app.register_blueprint(setup_api.bp)

    @app.before_request
    def _check_request() -> ResponseReturnValue | None:
        reason = auth.check_request(request)
        if reason:
            return jsonify({"ok": False, "error": reason}), 403
        return None

    @app.after_request
    def _headers(resp: Response) -> Response:
        resp.headers.update(SECURITY_HEADERS)
        if request.path.startswith("/api/"):
            resp.headers["Cache-Control"] = "no-store"
        return resp

    @app.errorhandler(ValueError)
    @app.errorhandler(TrainingError)
    def _user_error(e: Exception) -> ResponseReturnValue:
        return jsonify({"ok": False, "error": str(e)}), 400

    @app.get("/")
    def index() -> ResponseReturnValue:
        return send_from_directory(STATIC, "index.html")

    @app.get("/static/<path:name>")
    def static_file(name: str) -> ResponseReturnValue:
        return send_from_directory(STATIC, name, max_age=0)  # revalidate (ETag) – no stale UI after an update

    return app
