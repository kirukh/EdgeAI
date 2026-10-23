"""Operator level (no PIN): live image, status, teach-in, inspection, results, finish batch."""

from __future__ import annotations

import time
from collections.abc import Iterator
from typing import Any, cast

from flask import Blueprint, Response, abort, current_app, jsonify, request, send_file
from flask.typing import ResponseReturnValue

from .. import archive as qa
from .. import methods as M
from ..system import QCSystem
from .auth import SetupAuth

bp = Blueprint("operator", __name__)


def system() -> QCSystem:
    """The running inspection system of this app."""
    return cast(QCSystem, current_app.extensions["qc_system"])


def auth() -> SetupAuth:
    """Setup PIN and request checks of this app."""
    return cast(SetupAuth, current_app.extensions["qc_auth"])


def ok(**extra: Any) -> Response:
    """Success answer of an action: ``{"ok": true, …}``."""
    return jsonify({"ok": True, **extra})


def body() -> dict[str, Any]:
    """JSON body of the request (empty if there is none)."""
    data = request.get_json(silent=True)
    return data if isinstance(data, dict) else {}


# ---------------------------------------------------------------- live image + status
@bp.get("/stream.mjpg")
def stream() -> ResponseReturnValue:
    sys_ = system()

    def frames() -> Iterator[bytes]:
        last = None
        while True:
            frame = sys_.preview_jpeg
            if frame is not None and frame is not last:
                last = frame
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
            time.sleep(0.05)

    return Response(frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@bp.get("/snapshot.jpg")
def snapshot() -> ResponseReturnValue:
    if system().preview_jpeg is None:
        abort(404)
    return Response(system().preview_jpeg, mimetype="image/jpeg")


def _end_focus_without_setup() -> None:
    """The focus assistant belongs to the setup area: it ends when no device has it open any more
    (locked or timed out) – the operator screen cannot switch it off."""
    if system().cam.focus is not None and not auth().any_unlocked():
        system().set_focus_assist(False)


@bp.get("/api/status")
def status() -> ResponseReturnValue:
    _end_focus_without_setup()
    return jsonify(system().status())


@bp.get("/api/options")
def options() -> ResponseReturnValue:
    cfg = system().cfg.method
    return jsonify(
        {
            "methods": M.METHOD_LABELS,
            "decision": cfg.decision,
            "standalone": {"geometry": cfg.geometry_standalone, "diff": cfg.diff_standalone, "pca": cfg.pca_standalone},
        }
    )


# ---------------------------------------------------------------- setup PIN
@bp.post("/api/setup/unlock")
def setup_unlock() -> ResponseReturnValue:
    token, error, code = auth().unlock(str(body().get("pin", "")), request.remote_addr or "?")
    if token is None:
        system().log("warn", "Wrong setup PIN entered." if code == 403 else error)
        return jsonify({"ok": False, "error": error}), code
    if token:
        system().log("info", "Setup area unlocked.")
    return ok(token=token, timeout_min=system().cfg.ui.setup_timeout_min)


@bp.post("/api/setup/lock")
def setup_lock() -> ResponseReturnValue:
    auth().lock(request.headers.get("X-Setup-Token", ""))
    _end_focus_without_setup()
    return ok()


@bp.get("/api/setup/check")
def setup_check() -> ResponseReturnValue:
    return jsonify({"unlocked": auth().unlocked(request), "pin_required": auth().pin_required})


# ---------------------------------------------------------------- teach-in
@bp.post("/api/camera-setup/start")
def camera_setup_start() -> ResponseReturnValue:
    system().start_camera_setup()
    return ok()


@bp.post("/api/camera-setup/cancel")
def camera_setup_cancel() -> ResponseReturnValue:
    system().cancel_camera_setup()
    return ok()


@bp.post("/api/learn/start")
def learn_start() -> ResponseReturnValue:
    d = body()
    system().start_learning(str(d.get("name", "")), bool(d.get("auto_finish")))
    return ok()


@bp.post("/api/learn/finish")
def learn_finish() -> ResponseReturnValue:
    system().finish_learning()
    return ok()


@bp.post("/api/learn/cancel")
def learn_cancel() -> ResponseReturnValue:
    system().cancel_learning()
    return ok()


@bp.post("/api/learn/undo")
def learn_undo() -> ResponseReturnValue:
    idx = body().get("index")
    system().undo_reference(int(idx) if idx is not None else None)
    return ok()


@bp.get("/api/learn/thumb/<int:i>.jpg")
def learn_thumb(i: int) -> ResponseReturnValue:
    thumbs = system().teach.thumbs
    if not 0 <= i < len(thumbs):
        abort(404)
    return Response(thumbs[i], mimetype="image/jpeg")


@bp.post("/api/capture")
def capture() -> ResponseReturnValue:
    system().manual_capture()
    return ok()


# ---------------------------------------------------------------- inspection
@bp.post("/api/inspect/start")
def inspect_start() -> ResponseReturnValue:
    system().start_inspection(body().get("model"))
    return ok()


@bp.post("/api/inspect/stop")
def inspect_stop() -> ResponseReturnValue:
    system().stop_inspection()
    return ok()


@bp.get("/api/models")
def models() -> ResponseReturnValue:
    return jsonify(system().models.list())


@bp.post("/api/models/<slug>/select")
def model_select(slug: str) -> ResponseReturnValue:
    system().select_model(slug)
    return ok()


# ---------------------------------------------------------------- results
@bp.get("/api/results")
def results() -> ResponseReturnValue:
    return jsonify(list(system().results.history))


@bp.get("/api/results/<rid>/<kind>.jpg")
def result_image(rid: str, kind: str) -> ResponseReturnValue:
    imgs = system().results.images.get(rid)
    if imgs is None or kind not in ("overlay", "detail"):
        return abort(404)
    return Response(imgs[0] if kind == "overlay" else imgs[1], mimetype="image/jpeg")


@bp.post("/api/results/<rid>/feedback")
def result_feedback(rid: str) -> ResponseReturnValue:
    return ok(**system().feedback(rid, str(body().get("verdict", ""))))


# ---------------------------------------------------------------- finish batch / archives
@bp.get("/api/archives")
def archives() -> ResponseReturnValue:
    st = system().cfg.storage
    return jsonify({"archives": qa.list_archives(st.archive_dir), "pending": qa.summarize(st.results_dir)})


@bp.post("/api/archives")
def archive_create() -> ResponseReturnValue:
    return ok(**system().archive_results(str(body().get("label", ""))))


@bp.get("/api/archives/<name>")
def archive_download(name: str) -> ResponseReturnValue:
    try:
        path = qa.archive_path(system().cfg.storage.archive_dir, name)
    except qa.ArchiveError:
        abort(404)
    return send_file(path, mimetype="application/zip", as_attachment=True, download_name=path.name)
