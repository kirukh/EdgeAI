"""Setup level (PIN): sensitivity, training report, model updates (continuous improvement), delete
models, focus assistant, counters, archives, simulator. Every route of this blueprint requires an
unlocked setup session."""

from __future__ import annotations

from flask import Blueprint, abort, jsonify, request, send_file
from flask.typing import ResponseReturnValue

from .. import archive as qa
from ..learning import PENDING, LearningStore
from ..model import QCModel
from .operator_api import auth, body, ok, system

bp = Blueprint("setup", __name__, url_prefix="/api")


@bp.before_request
def require_pin() -> ResponseReturnValue | None:
    if not auth().unlocked(request):
        return jsonify({"ok": False, "error": "Setup area is locked – enter the setup PIN."}), 403
    return None


@bp.get("/models/<slug>")
def model_detail(slug: str) -> ResponseReturnValue:
    path = system().models.find(slug)
    if path is None:
        return abort(404)
    m = QCModel.load(path)
    return jsonify({**m.summary(), "report": m.report, "warnings": m.warnings()})


@bp.post("/models/<slug>/sensitivity")
def model_sensitivity(slug: str) -> ResponseReturnValue:
    return ok(sensitivity=system().set_sensitivity(slug, body()))


@bp.post("/models/<slug>/update")
def model_update(slug: str) -> ResponseReturnValue:
    system().improve.update_now(slug)
    return ok()


@bp.post("/models/<slug>/undo")
def model_undo(slug: str) -> ResponseReturnValue:
    system().improve.undo(slug)
    return ok()


@bp.get("/models/<slug>/waiting")
def waiting(slug: str) -> ResponseReturnValue:
    return jsonify(system().improve.waiting(slug))


@bp.get("/models/<slug>/waiting/<name>.png")
def waiting_image(slug: str, name: str) -> ResponseReturnValue:
    path = system().models.find(slug)
    img = LearningStore(path).image_path(PENDING, name) if path is not None else None
    if img is None:
        return abort(404)
    return send_file(img, mimetype="image/png")


@bp.delete("/models/<slug>/waiting/<name>")
def waiting_discard(slug: str, name: str) -> ResponseReturnValue:
    system().improve.discard(slug, name)
    return ok()


@bp.delete("/models/<slug>")
def model_delete(slug: str) -> ResponseReturnValue:
    system().delete_model(slug)
    return ok()


@bp.post("/belt-axis")
def belt_axis() -> ResponseReturnValue:
    system().set_belt_axis(str(body().get("axis", "")))
    return ok()


@bp.post("/trigger-mode")
def trigger_mode() -> ResponseReturnValue:
    system().set_trigger_mode(str(body().get("mode", "")))
    return ok()


@bp.post("/focus")
def focus() -> ResponseReturnValue:
    system().set_focus_assist(bool(body().get("on")))
    return ok()


@bp.post("/counters/reset")
def reset() -> ResponseReturnValue:
    system().reset_counts()
    return ok()


@bp.delete("/archives/<name>")
def archive_delete(name: str) -> ResponseReturnValue:
    try:
        qa.delete_archive(system().cfg.storage.archive_dir, name)
    except qa.ArchiveError:
        abort(404)
    system().log("info", f"Archive {name} deleted.")
    return ok()


@bp.post("/sim")
def sim() -> ResponseReturnValue:
    d = body()
    rate = d.get("defect_rate")
    system().set_sim(
        d.get("lighting"), float(rate) if rate is not None else None, d.get("part_type"), d.get("any_angle")
    )
    return ok()
