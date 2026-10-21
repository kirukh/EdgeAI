"""Setup level (PIN): sensitivity, training report, model updates (continuous improvement), delete
models, focus assistant, counters, archives, simulator. Every route of this blueprint requires an
unlocked setup session."""

from __future__ import annotations

from flask import Blueprint, abort, jsonify, request, send_file

from .. import archive as qa
from ..learning import PENDING, LearningStore
from ..model import QCModel, model_path
from .operator_api import auth, body, ok, system

bp = Blueprint("setup", __name__, url_prefix="/api")


@bp.before_request
def require_pin():
    if not auth().unlocked(request):
        return jsonify({"ok": False, "error": "Setup area is locked – enter the setup PIN."}), 403
    return None


@bp.get("/models/<slug>")
def model_detail(slug: str):
    path = model_path(system().cfg.storage.models_dir, slug)
    if path is None:
        return abort(404)
    m = QCModel.load(path)
    return jsonify({**m.summary(), "report": m.report, "warnings": m.warnings()})


@bp.post("/models/<slug>/sensitivity")
def model_sensitivity(slug: str):
    return ok(sensitivity=system().set_sensitivity(slug, body()))


@bp.post("/models/<slug>/update")
def model_update(slug: str):
    system().improve.update_now(slug)
    return ok()


@bp.post("/models/<slug>/undo")
def model_undo(slug: str):
    system().improve.undo(slug)
    return ok()


@bp.get("/models/<slug>/waiting")
def waiting(slug: str):
    return jsonify(system().improve.waiting(slug))


@bp.get("/models/<slug>/waiting/<name>.png")
def waiting_image(slug: str, name: str):
    path = model_path(system().cfg.storage.models_dir, slug)
    img = LearningStore(path).image_path(PENDING, name) if path is not None else None
    if img is None:
        return abort(404)
    return send_file(img, mimetype="image/png")


@bp.delete("/models/<slug>/waiting/<name>")
def waiting_discard(slug: str, name: str):
    system().improve.discard(slug, name)
    return ok()


@bp.delete("/models/<slug>")
def model_delete(slug: str):
    system().delete_model(slug)
    return ok()


@bp.post("/belt-axis")
def belt_axis():
    system().set_belt_axis(str(body().get("axis", "")))
    return ok()


@bp.post("/focus")
def focus():
    system().set_focus_assist(bool(body().get("on")))
    return ok()


@bp.post("/counters/reset")
def reset():
    system().reset_counts()
    return ok()


@bp.delete("/archives/<name>")
def archive_delete(name: str):
    try:
        qa.delete_archive(system().cfg.storage.archive_dir, name)
    except qa.ArchiveError:
        abort(404)
    system().log("info", f"Archive {name} deleted.")
    return ok()


@bp.post("/sim")
def sim():
    d = body()
    rate = d.get("defect_rate")
    system().set_sim(
        d.get("lighting"), float(rate) if rate is not None else None, d.get("part_type"), d.get("any_angle")
    )
    return ok()
