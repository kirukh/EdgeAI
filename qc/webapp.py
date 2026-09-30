"""Web user interface for the supervisor.

Why a web UI instead of a local app: the Pi usually runs headless right at the
conveyor; it is operated from a tablet or laptop on the same network. The page
has no external dependencies (works offline).

Optional access protection: set the environment variable ``QC_TOKEN`` – all
modifying requests must then send the token (open the page with
``?token=...``).
"""
from __future__ import annotations

import hmac
import os
import time
from pathlib import Path

from flask import Flask, Response, abort, jsonify, request, send_file, send_from_directory

from . import archive as qa
from . import methods as M
from . import model as qm
from . import representations as reps
from .system import QCSystem


def create_app(system: QCSystem) -> Flask:
    static_dir = Path(__file__).parent / "static"
    app = Flask(__name__, static_folder=None)
    token = os.environ.get("QC_TOKEN", "")

    @app.before_request
    def _auth():
        if token and request.method in ("POST", "DELETE"):
            sent = request.headers.get("X-QC-Token", "")
            if not hmac.compare_digest(sent, token):
                abort(403)

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
        })

    # --- learning mode ---------------------------------------------------
    @app.post("/api/learn/start")
    def learn_start():
        system.start_learning((request.json or {}).get("name", ""))
        return ok()

    @app.post("/api/learn/finish")
    def learn_finish():
        system.finish_learning(_overrides(request.json or {}))
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
        return jsonify(qm.list_models(system.cfg.storage.models_dir))

    @app.get("/api/models/<slug>")
    def model_detail(slug):
        m = qm.QCModel.load(_model_path(slug))
        return jsonify({**m.summary(), "report": m.report})

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
    def result_image(rid, kind):
        imgs = system.images.get(rid)
        if imgs is None or kind not in ("overlay", "detail"):
            abort(404)
        return Response(imgs[0] if kind == "overlay" else imgs[1], mimetype="image/jpeg")

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
                       d.get("any_angle"))
        return ok()

    def _model_path(slug: str) -> Path:
        base = Path(system.cfg.storage.models_dir).resolve()
        p = (base / slug).resolve()
        if p.parent != base or not (p / "meta.json").exists():   # no path traversal
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
