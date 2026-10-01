"""Web user interface for the supervisor.

Why a web UI instead of a local app: the Pi usually runs headless right at the
conveyor; it is operated from a tablet or laptop on the same network. The page
has no external dependencies (works offline).

Two access levels:

* **Supervisor** (no PIN): start/stop, select part, teach in a new part, report a
  wrong result, reference check, archive a batch.
* **Setup** (PIN from ``config.json → ui.setup_pin``): methods, sensitivity,
  calibration, lighting, GPIO, model management, simulator. The PIN is checked on
  the server; the page receives a session token that expires after
  ``ui.setup_timeout_min`` minutes without setup activity.

Optional network protection on top: set the environment variable ``QC_TOKEN`` –
all modifying requests must then send the token (open the page with ``?token=...``).
"""
from __future__ import annotations

import hmac
import os
import secrets
import threading
import time
from pathlib import Path

from flask import Flask, Response, abort, jsonify, request, send_file, send_from_directory

from . import archive as qa
from . import methods as M
from . import model as qm
from . import representations as reps
from .recipe import Recipe
from .system import QCSystem


def create_app(system: QCSystem) -> Flask:
    static_dir = Path(__file__).parent / "static"
    app = Flask(__name__, static_folder=None)
    token = os.environ.get("QC_TOKEN", "")
    ui = system.cfg.ui
    sessions: dict[str, float] = {}             # setup token → last activity
    failed = {"n": 0, "until": 0.0}
    guard = threading.Lock()

    # endpoints that need the setup PIN (POST/DELETE); everything else is supervisor level
    SETUP_ENDPOINTS = {"model_retrain", "model_learn_collected", "model_discard_collected", "model_sensitivity",
                       "model_delete", "settings", "io_test", "calib_start", "calib_capture", "calib_compute",
                       "calib_cancel", "calib_delete", "reset", "archive_delete", "sim", "focus"}

    def setup_unlocked() -> bool:
        if not ui.setup_pin:
            return True
        sent = request.headers.get("X-Setup-Token", "")
        now = time.time()
        with guard:
            for k in [k for k, t in sessions.items() if now - t > ui.setup_timeout_min * 60]:
                del sessions[k]
            if sent and sent in sessions:
                sessions[sent] = now
                return True
        return False

    @app.before_request
    def _auth():
        if token and request.method in ("POST", "DELETE"):
            sent = request.headers.get("X-QC-Token", "")
            if not hmac.compare_digest(sent, token):
                abort(403)
        if request.endpoint in SETUP_ENDPOINTS and request.method in ("POST", "DELETE") and not setup_unlocked():
            return jsonify({"ok": False, "error": "Setup area is locked – enter the setup PIN."}), 403

    @app.post("/api/setup/unlock")
    def setup_unlock():
        if not ui.setup_pin:
            return ok(token="", timeout_min=ui.setup_timeout_min)
        now = time.time()
        with guard:
            if now < failed["until"]:
                return jsonify({"ok": False, "error": f"Too many wrong PINs – wait {int(failed['until'] - now) + 1} s."}), 429
        pin = str((request.json or {}).get("pin", ""))
        if not hmac.compare_digest(pin.encode(), ui.setup_pin.encode()):
            with guard:
                failed["n"] += 1
                if failed["n"] >= 5:
                    failed["n"], failed["until"] = 0, now + 60
            system.log("warn", "Wrong setup PIN entered.")
            return jsonify({"ok": False, "error": "Wrong PIN."}), 403
        t = secrets.token_urlsafe(24)
        with guard:
            failed["n"] = 0
            sessions[t] = now
        system.log("info", "Setup area unlocked.")
        return ok(token=t, timeout_min=ui.setup_timeout_min)

    @app.post("/api/setup/lock")
    def setup_lock():
        with guard:
            sessions.pop(request.headers.get("X-Setup-Token", ""), None)
        return ok()

    @app.get("/api/setup/check")
    def setup_check():
        return jsonify({"unlocked": setup_unlocked(), "pin_required": bool(ui.setup_pin)})

    def ok(**extra):
        return jsonify({"ok": True, **extra})

    @app.errorhandler(ValueError)
    def _value_error(e):
        return jsonify({"ok": False, "error": str(e)}), 400

    @app.errorhandler(qm.TrainingError)
    def _train_error(e):
        return jsonify({"ok": False, "error": str(e)}), 400

    @app.get("/")
    def index():
        return send_from_directory(static_dir, "index.html")

    @app.get("/stream.mjpg")
    def stream():
        def gen():
            last = None
            while True:
                frame = system.preview_jpeg
                if frame is not None and frame is not last:
                    last = frame
                    yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
                time.sleep(0.05)

        return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.get("/snapshot.jpg")
    def snapshot():
        if system.preview_jpeg is None:
            abort(404)
        return Response(system.preview_jpeg, mimetype="image/jpeg")

    @app.get("/api/status")
    def status():
        return jsonify(system.status())

    @app.get("/api/options")
    def options():
        return jsonify({
            "methods": M.METHOD_LABELS,
            "representations": reps.labels(),
            "defaults": {"methods": system.cfg.method.methods,
                         "diff_representation": system.cfg.method.diff_representation,
                         "pca_representation": system.cfg.method.pca_representation},
            "target": system.cfg.target_reference_count,
            "decision": system.cfg.method.decision,
            "standalone": {"geometry": system.cfg.method.geometry_standalone,
                           "diff": system.cfg.method.diff_standalone, "pca": system.cfg.method.pca_standalone},
        })

    # --- learning mode ---------------------------------------------------
    @app.post("/api/learn/start")
    def learn_start():
        d = request.json or {}
        system.start_learning(d.get("name", ""), bool(d.get("auto_finish")))
        return ok()

    @app.post("/api/learn/finish")
    def learn_finish():
        # method choices only from the setup area – the supervisor always uses the defaults
        system.finish_learning(_overrides(request.json or {}) if setup_unlocked() else None)
        return ok()

    @app.post("/api/learn/cancel")
    def learn_cancel():
        system.cancel_learning()
        return ok()

    @app.post("/api/learn/undo")
    def learn_undo():
        idx = (request.json or {}).get("index")
        system.undo_reference(int(idx) if idx is not None else None)
        return ok()

    @app.get("/api/learn/thumb/<int:i>.jpg")
    def learn_thumb(i):
        try:
            return Response(system.learn_thumbs[i], mimetype="image/jpeg")
        except IndexError:
            abort(404)

    @app.post("/api/capture")
    def capture():
        system.manual_capture()
        return ok()

    # --- inspection mode -------------------------------------------------
    @app.post("/api/inspect/start")
    def inspect_start():
        system.start_inspection((request.json or {}).get("model"))
        return ok()

    @app.post("/api/inspect/stop")
    def inspect_stop():
        system.stop_inspection()
        return ok()

    # --- models ----------------------------------------------------------
    @app.get("/api/models")
    def models():
        return jsonify(system.list_models())

    @app.get("/api/models/<slug>")
    def model_detail(slug):
        r = Recipe.load(_model_path(slug))
        return jsonify({**r.summary(), "report": r.report(), "warnings": r.warnings()})

    @app.post("/api/models/<slug>/select")
    def model_select(slug):
        _model_path(slug)
        system.select_model(slug)
        return ok()

    @app.post("/api/models/<slug>/retrain")
    def model_retrain(slug):
        _model_path(slug)
        system.retrain_model(slug, _overrides(request.json or {}))
        return ok()

    @app.post("/api/models/<slug>/learn-collected")
    def model_learn_collected(slug):
        _model_path(slug)
        system.learn_collected(slug)
        return ok()

    @app.post("/api/models/<slug>/discard-collected")
    def model_discard_collected(slug):
        _model_path(slug)
        return ok(discarded=system.discard_collected(slug))

    @app.post("/api/models/<slug>/sensitivity")
    def model_sensitivity(slug):
        _model_path(slug)
        return ok(sensitivity=system.set_sensitivity(slug, request.json or {}))

    @app.delete("/api/models/<slug>")
    def model_delete(slug):
        _model_path(slug)
        system.delete_model(slug)
        return ok()

    # --- results ---------------------------------------------------------
    @app.get("/api/results")
    def results():
        return jsonify(list(system.history))

    @app.get("/api/results/<rid>/<kind>.jpg")
    @app.get("/api/results/<rid>/<int:idx>/<kind>.jpg")
    def result_image(rid, kind, idx=0):
        imgs = system.images.get(rid)
        if imgs is None or kind not in ("overlay", "detail") or idx >= len(imgs):
            abort(404)
        _, overlay, detail = imgs[idx]
        return Response(overlay if kind == "overlay" else detail, mimetype="image/jpeg")

    @app.post("/api/results/<rid>/feedback")
    def result_feedback(rid):
        d = request.json or {}
        return ok(**system.feedback(rid, str(d.get("verdict", "")), bool(d.get("add_reference"))))

    # --- self-test / settings / IO / calibration ------------------------------
    @app.post("/api/selftest/start")
    def selftest_start():
        system.start_selftest()
        return ok()

    @app.post("/api/settings")
    def settings():
        d = request.json or {}
        system.update_settings(d.get("shots_per_part"), d.get("lighting_mode"), d.get("self_learning"),
                               d.get("self_learning_auto"))
        return ok()

    @app.post("/api/focus")
    def focus():
        system.set_focus_assist(bool((request.json or {}).get("on")))
        return ok()

    @app.post("/api/io/test-reject")
    def io_test():
        system.io.test_reject()
        return ok()

    @app.post("/api/calibration/start")
    def calib_start():
        d = request.json or {}
        system.calib_start(d.get("cols"), d.get("rows"), d.get("square_mm"))
        return ok()

    @app.post("/api/calibration/capture")
    def calib_capture():
        return ok(**system.calib_capture())

    @app.post("/api/calibration/compute")
    def calib_compute():
        return ok(calibration=system.calib_compute())

    @app.post("/api/calibration/cancel")
    def calib_cancel():
        system.calib_session = None
        return ok()

    @app.delete("/api/calibration")
    def calib_delete():
        system.calib_delete()
        return ok()

    @app.get("/api/calibration/preview.jpg")
    def calib_preview():
        s = system.calib_session
        if not s or not s.get("preview"):
            abort(404)
        return Response(s["preview"], mimetype="image/jpeg")

    @app.post("/api/counters/reset")
    def reset():
        system.reset_counts()
        return ok()

    # --- results archive -------------------------------------------------
    @app.get("/api/archives")
    def archives():
        return jsonify({"archives": qa.list_archives(system.cfg.storage.archive_dir),
                        "pending": qa.summarize(system.cfg.storage.results_dir)})

    @app.post("/api/archives")
    def archive_create():
        info = system.archive_results((request.json or {}).get("label", ""))
        return ok(**info)

    @app.get("/api/archives/<name>")
    def archive_download(name):
        try:
            path = qa.archive_path(system.cfg.storage.archive_dir, name)
        except qa.ArchiveError:
            abort(404)
        return send_file(path, mimetype="application/zip", as_attachment=True, download_name=path.name)

    @app.delete("/api/archives/<name>")
    def archive_delete(name):
        try:
            qa.delete_archive(system.cfg.storage.archive_dir, name)
        except qa.ArchiveError:
            abort(404)
        system.log("info", f"Archive {name} deleted.")
        return ok()

    @app.post("/api/sim")
    def sim():
        d = request.json or {}
        rate = d.get("defect_rate")
        system.set_sim(d.get("lighting"), float(rate) if rate is not None else None, d.get("part_type"),
                       d.get("any_angle"), d.get("board"))
        return ok()

    def _model_path(slug: str) -> Path:
        base = Path(system.cfg.storage.models_dir).resolve()
        p = (base / slug).resolve()
        if p.parent != base or not ((p / "recipe.json").exists() or (p / "meta.json").exists()):   # no traversal
            abort(404)
        return p

    def _overrides(d: dict) -> dict:
        out = {}
        if isinstance(d.get("methods"), list):
            out["methods"] = [m for m in d["methods"] if m in M.METHODS]
            if not out["methods"]:
                raise ValueError("Select at least one inspection method.")
        for k in ("diff_representation", "pca_representation"):
            if d.get(k) in reps.REPRESENTATIONS:
                out[k] = d[k]
        return out

    return app
