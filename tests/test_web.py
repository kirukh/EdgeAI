"""Web API: model management, teach-in assistant, simulator settings (without a running camera loop)."""

from __future__ import annotations

import numpy as np
from conftest import shots, web_client

from qc.config import LocalizationConfig, MethodConfig
from qc.model import QCModel


def _unlocked(client, pin: str) -> dict:
    return {"X-Setup-Token": client.post("/api/setup/unlock", json={"pin": pin}).json["token"]}


def test_model_management(make_system):
    s = make_system()
    rng = np.random.default_rng(3)
    QCModel.train("Plate", shots("A", "front", 6, rng), LocalizationConfig(), MethodConfig()).save(
        s.cfg.storage.models_dir
    )
    c = web_client(s)
    h = _unlocked(c, s.cfg.ui.setup_pin)
    assert [m["slug"] for m in c.get("/api/models").json] == ["Plate"]
    assert c.get("/api/models/Plate").status_code == 403  # training report: setup only
    detail = c.get("/api/models/Plate", headers=h).json
    assert detail["report"]["references_used"] == 6 and isinstance(detail["warnings"], list)
    assert c.get("/api/models/nope", headers=h).status_code == 404
    assert c.post("/api/models/Plate/select").json["ok"]
    assert s.model is not None and s.model.slug == "Plate"
    r = c.post("/api/models/Plate/sensitivity", json={"pca": 1.5}, headers=h).json
    assert r["sensitivity"]["pca"] == 1.5 and s.model.sensitivity["pca"] == 1.5  # active model updated too
    assert QCModel.load(s.model.path).sensitivity["pca"] == 1.5  # and saved
    assert c.post("/api/inspect/start", json={}).json["ok"]
    assert c.delete("/api/models/Plate", headers=h).status_code == 400  # not while inspecting
    c.post("/api/inspect/stop")
    assert c.delete("/api/models/Plate", headers=h).json["ok"]
    assert s.model is None and c.get("/api/models").json == []
    assert c.post("/api/models/Plate/select").status_code == 400


def test_teach_in_assistant_via_api(make_system):
    s = make_system()
    c = web_client(s)
    assert c.post("/api/learn/start", json={"name": ""}).status_code == 400  # a name is required
    assert c.post("/api/learn/start", json={"name": "Plate"}).json["ok"]
    assert c.post("/api/learn/start", json={"name": "Other"}).status_code == 400  # already learning
    rng = np.random.default_rng(4)
    for f in shots("A", "front", 3, rng):
        s._handle_part([f])
    s._handle_part([np.zeros((480, 640, 3), np.uint8)])  # empty belt → discarded
    st = c.get("/api/status").json
    assert st["learning"]["count"] == 3 and st["mode"] == "learning"
    assert c.get("/api/learn/thumb/2.jpg").status_code == 200
    assert c.get("/api/learn/thumb/9.jpg").status_code == 404
    assert c.post("/api/learn/undo", json={"index": 0}).json["ok"]
    assert c.post("/api/learn/finish").status_code == 400  # 2 < minimum of 3
    assert c.post("/api/learn/cancel").json["ok"]
    assert c.get("/api/status").json["mode"] == "idle" and s.teach.count == 0
    assert c.post("/api/capture").status_code == 400  # only while teaching in / inspecting


def test_simulator_settings_and_counters(make_system):
    s = make_system()
    c = web_client(s)
    h = _unlocked(c, s.cfg.ui.setup_pin)
    assert c.post("/api/sim", json={"part_type": "C", "defect_rate": 2.0, "any_angle": True}, headers=h).json["ok"]
    assert s.camera.part_type == "C" and s.camera.defect_rate == 1.0 and s.camera.cfg.sim_any_angle
    assert c.post("/api/sim", json={"lighting": "back"}, headers=h).json["ok"]
    assert c.get("/api/status").json["sim"]["lighting"] == "back"
    assert c.post("/api/counters/reset", headers=h).json["ok"]
    assert c.post("/api/archives", json={}).status_code == 400  # nothing to archive yet


def test_evaluation_tool_produces_a_table():
    from qc import evaluate as E

    train, test = E.synthetic_sets("A", "front", n_train=6, n_good=4, n_per_defect=1, seed=1)
    grid = [g for g in E.default_grid(MethodConfig()) if g[0] == "geometry"]
    rows = E.evaluate(train, test, grid, LocalizationConfig())
    md = E.to_markdown(rows, "smoke")
    assert "| geometry |" in md and "False alarms" in md
    assert rows[0].per_defect["missing"] == 1.0  # a missing hole is always found


def test_operator_screen_endpoints(make_system):
    """Live image, options, PIN state, results with images and feedback – as the operator page uses them."""
    s = make_system()
    rng = np.random.default_rng(6)
    QCModel.train("Plate", shots("A", "front", 6, rng), LocalizationConfig(), MethodConfig()).save(
        s.cfg.storage.models_dir
    )
    c = web_client(s)
    assert c.get("/snapshot.jpg").status_code == 404  # no frame yet
    s._update_preview(s._read())
    assert c.get("/snapshot.jpg").data[:2] == b"\xff\xd8"  # JPEG
    stream = c.get("/stream.mjpg", buffered=False)
    assert next(iter(stream.response)).startswith(b"--frame\r\nContent-Type: image/jpeg")
    stream.close()
    opts = c.get("/api/options").json
    assert set(opts["methods"]) == {"geometry", "diff", "pca"} and opts["decision"] == "fusion"

    assert c.get("/api/setup/check").json == {"unlocked": False, "pin_required": True}
    h = _unlocked(c, s.cfg.ui.setup_pin)
    assert c.get("/api/setup/check", headers=h).json["unlocked"]
    assert c.post("/api/setup/lock", headers=h).json["ok"]
    assert not c.get("/api/setup/check", headers=h).json["unlocked"]  # token no longer valid

    assert c.post("/api/inspect/start", json={"model": "Plate"}).json["ok"]
    assert c.post("/api/camera-setup/start").status_code == 400  # only from idle
    for f in shots("A", "front", 1, rng, "missing"):
        s._handle_part([f])
    rid = c.get("/api/results").json[0]["image_id"]
    assert c.get(f"/api/results/{rid}/overlay.jpg").status_code == 200
    assert c.get(f"/api/results/{rid}/detail.jpg").status_code == 200
    assert c.get(f"/api/results/{rid}/other.jpg").status_code == 404
    assert c.get("/api/results/nope/overlay.jpg").status_code == 404
    r = c.post(f"/api/results/{rid}/feedback", json={"verdict": "nok"}).json
    assert r["ok"] and r["outcome"] == "confirmed_nok"
    assert c.post("/api/inspect/stop").json["ok"]
    assert c.post("/api/camera-setup/cancel").json["ok"] and s.mode == "idle"
    assert c.get("/api/archives/nope.zip").status_code == 404
