"""Pipeline tests with synthetic parts.  Run:  python -m pytest -q"""
from __future__ import annotations

import time

import numpy as np
import pytest

from qc import synthetic as S
from qc.alignment import localize_and_align
from qc.config import AppConfig, LocalizationConfig, MethodConfig
from qc.model import QCModel, TrainingError, list_models, retrain

GEOMETRY_EXPECT = {
    "missing": "MISSING_HOLE",
    "position": "HOLE_POSITION",
    "diameter": "HOLE_DIAMETER",
    "extra": "EXTRA_HOLE",
    "corner": "OUTLINE",
}


def frames(part_type, lighting, n, rng, defect=None):
    return [S.compose(S.make_part(part_type, defect, rng), S.random_pose(rng), lighting, rng=rng) for _ in range(n)]


@pytest.fixture(scope="module")
def front_model():
    rng = np.random.default_rng(42)
    return QCModel.train("A-front", frames("A", "front", 12, rng), LocalizationConfig(), MethodConfig())


def test_alignment_handles_rotation_and_flip():
    rng = np.random.default_rng(0)
    cfg = LocalizationConfig()
    base = S.compose(S.make_part("A", None, rng), (320, 240, 0), "front", rng=rng)
    ref = localize_and_align(base, cfg)
    for ang in (7, -8, 180, 186):
        img = S.compose(S.make_part("A", None, rng), (300, 250, ang), "front", rng=rng)
        a = localize_and_align(img, cfg, ref.canvas_size, ref.norm)
        diff = np.abs(np.clip(a.norm, 0, 1) - np.clip(ref.norm, 0, 1)).mean()
        assert diff < 0.03, f"Alignment at {ang}° inaccurate ({diff:.3f})"


def test_no_part_returns_status():
    rng = np.random.default_rng(1)
    model = QCModel.train("t", frames("A", "front", 5, rng), LocalizationConfig(), MethodConfig(methods=["geometry"]))
    empty = S.compose(None, (0, 0, 0), "front", rng=rng)
    assert model.inspect(empty).status == "NO_PART"


def test_good_parts_pass(front_model):
    rng = np.random.default_rng(7)
    results = [front_model.inspect(f) for f in frames("A", "front", 15, rng)]
    assert all(r.ok for r in results), [d.to_dict() for r in results for d in r.defects]


@pytest.mark.parametrize("defect,kind", sorted(GEOMETRY_EXPECT.items()))
def test_geometry_defects_classified(front_model, defect, kind):
    rng = np.random.default_rng(11)
    for f in frames("A", "front", 4, rng, defect):
        r = front_model.inspect(f)
        assert r.status == "NOK"
        assert kind in {d.kind for d in r.defects}


def test_blind_hole_frontlight(front_model):
    rng = np.random.default_rng(5)
    for f in frames("A", "front", 4, rng, "blind"):
        kinds = {d.kind for d in front_model.inspect(f).defects}
        assert "BLIND_HOLE" in kinds


def test_surface_defects_frontlight(front_model):
    rng = np.random.default_rng(9)
    assert all(front_model.inspect(f).status == "NOK" for f in frames("A", "front", 4, rng, "stain"))


def test_backlight_part_B():
    rng = np.random.default_rng(3)
    model = QCModel.train("B-back", frames("B", "back", 12, rng), LocalizationConfig(), MethodConfig())
    assert all(model.inspect(f).ok for f in frames("B", "back", 10, rng))
    for defect in ("missing", "diameter", "position"):
        assert all(model.inspect(f).status == "NOK" for f in frames("B", "back", 3, rng, defect))


def test_save_load_roundtrip(tmp_path, front_model):
    front_model.save(tmp_path)
    loaded = QCModel.load(tmp_path / "A-front")
    rng = np.random.default_rng(21)
    for f in frames("A", "front", 3, rng, "missing") + frames("A", "front", 3, rng):
        a, b = front_model.inspect(f, False), loaded.inspect(f, False)
        assert a.status == b.status
        assert [d.kind for d in a.defects] == [d.kind for d in b.defects]
    assert list_models(tmp_path)[0]["name"] == "A-front"
    m2 = retrain(tmp_path / "A-front", MethodConfig(methods=["diff"], diff_representation="edges"))
    assert list(m2.methods) == ["diff"]
    assert QCModel.load(tmp_path / "A-front").method_cfg.diff_representation == "edges"


def test_too_few_references():
    rng = np.random.default_rng(2)
    with pytest.raises(TrainingError):
        QCModel.train("x", frames("A", "front", 2, rng), LocalizationConfig(), MethodConfig())


def test_web_flow(tmp_path):
    """Complete workflow via the HTTP API with a simulated conveyor."""
    from qc.camera import SimulatorSource
    from qc.system import QCSystem
    from qc.webapp import create_app

    cfg = AppConfig()
    cfg.camera.fps = 400
    cfg.camera.width, cfg.camera.height = 640, 480
    cfg.camera.sim_speed_px = 30
    cfg.storage.models_dir = str(tmp_path / "models")
    cfg.storage.results_dir = str(tmp_path / "results")
    system = QCSystem(cfg, SimulatorSource(cfg.camera, seed=1))
    system.start()
    client = create_app(system).test_client()

    def wait(cond, timeout=60):
        t0 = time.time()
        while not cond():
            assert time.time() - t0 < timeout, "Timeout"
            time.sleep(0.05)

    try:
        assert client.get("/").status_code == 200
        assert client.post("/api/learn/start", json={"name": "Plate A"}).json["ok"]
        wait(lambda: client.get("/api/status").json["learning"]["count"] >= 6)
        # method choice is a setup function → unlock with the PIN first
        tok = client.post("/api/setup/unlock", json={"pin": cfg.ui.setup_pin}).json["token"]
        assert client.post("/api/learn/finish", json={"methods": ["geometry", "diff"]},
                           headers={"X-Setup-Token": tok}).json["ok"]
        wait(lambda: client.get("/api/status").json["mode"] == "idle")
        models = client.get("/api/models").json
        assert models and models[0]["methods"] == {"main": ["geometry", "diff"]}
        assert client.post("/api/inspect/start", json={"model": models[0]["slug"]}).json["ok"]
        wait(lambda: len(client.get("/api/results").json) >= 4)
        client.post("/api/inspect/stop")
        results = client.get("/api/results").json
        rid = results[0]["image_id"]
        assert client.get(f"/api/results/{rid}/overlay.jpg").status_code == 200
        # The simulator provides the expected class → the result must match it
        agree = [(r["status"] == "OK") == (r["ground_truth"] == "ok") for r in results]
        assert np.mean(agree) >= 0.75, results
        assert client.get("/api/models/../../etc").status_code == 404
    finally:
        system.stop()


@pytest.mark.parametrize("part_type", ["C", "D"])
def test_new_shapes_any_rotation(part_type):
    """Triangle and square parts arriving in any rotation: consistent alignment and detection."""
    rng = np.random.default_rng(8)

    def shots(n, defect=None):
        return [S.compose(S.make_part(part_type, defect, rng), S.random_pose(rng, any_angle=True), "front", rng=rng)
                for _ in range(n)]

    model = QCModel.train(part_type, shots(12), LocalizationConfig(), MethodConfig())
    geo = model.report["methods"]["geometry"]
    assert max(geo["position_std_px"]) < 1.5, geo      # references aligned consistently
    assert not geo["high_scatter_holes"]
    assert all(model.inspect(f, False).ok for f in shots(10))
    for defect, kind in (("missing", "MISSING_HOLE"), ("position", "HOLE_POSITION"), ("diameter", "HOLE_DIAMETER")):
        for f in shots(3, defect):
            r = model.inspect(f, False)
            assert r.status == "NOK" and kind in {d.kind for d in r.defects}, (defect, [d.to_dict() for d in r.defects])


def test_mirrored_part():
    """A part lying upside down (mirror image) is NOK by default and OK with allow_mirror."""
    import cv2

    rng = np.random.default_rng(12)
    frames_ = [S.compose(S.make_part("B", None, rng), S.random_pose(rng), "front", rng=rng) for _ in range(10)]
    mirrored = [cv2.flip(S.compose(S.make_part("B", None, rng), S.random_pose(rng), "front", rng=rng), 1) for _ in range(4)]
    strict = QCModel.train("B", frames_, LocalizationConfig(), MethodConfig(methods=["geometry"]))
    assert all(strict.inspect(f, False).status == "NOK" for f in mirrored)
    lenient = QCModel.train("B", frames_, LocalizationConfig(allow_mirror=True), MethodConfig(methods=["geometry"]))
    assert all(lenient.inspect(f, False).ok for f in mirrored)


def test_archive_results(tmp_path):
    """Batch finished: results are zipped with a summary, the results folder is emptied."""
    import json
    import zipfile

    from qc import archive as qa
    from qc.camera import FolderSource
    from qc.system import QCSystem
    from qc.webapp import create_app

    rng = np.random.default_rng(5)
    model = QCModel.train("Plate A", frames("A", "front", 6, rng), LocalizationConfig(), MethodConfig(methods=["geometry"]))
    cfg = AppConfig()
    cfg.storage.models_dir = str(tmp_path / "models")
    cfg.storage.results_dir = str(tmp_path / "results")
    cfg.storage.archive_dir = str(tmp_path / "archive")
    model.save(cfg.storage.models_dir)
    img_dir = tmp_path / "imgs"
    img_dir.mkdir()
    import cv2
    for i, f in enumerate(frames("A", "front", 3, rng) + frames("A", "front", 2, rng, "missing")):
        cv2.imwrite(str(img_dir / f"{i}.png"), f)
    from qc.recipe import Recipe

    system = QCSystem(cfg, FolderSource(str(img_dir), loop=False))
    system.model = Recipe.load(tmp_path / "models" / "Plate_A")      # old single-model folder → recipe
    system.mode = "inspecting"
    for _ in range(5):
        system._handle_part([{"main": system.camera.next_part()}], time.time())
    client = create_app(system).test_client()

    pending = client.get("/api/archives").json["pending"]
    assert pending["inspected"] == 5 and pending["nok"] == 2
    assert client.post("/api/archives", json={}).status_code == 400          # inspection still running
    system.mode = "idle"
    r = client.post("/api/archives", json={"label": "Batch 1"}).json
    assert r["ok"] and r["file"].startswith("Batch_1_")
    assert not qa.has_results(cfg.storage.results_dir)                       # results folder emptied
    assert system.counts["OK"] == 0 and not system.history                   # counters reset
    with zipfile.ZipFile(tmp_path / "archive" / r["file"]) as zf:
        names = zf.namelist()
        summary = json.loads(zf.read("summary.json"))
    assert "summary.txt" in names and any(n.endswith("inspection_log.csv") for n in names)
    assert any(n.endswith("_marked.jpg") for n in names)                     # NOK images included
    assert summary["ok"] == 3 and summary["nok"] == 2 and summary["defect_types"] == {"Hole missing": 2}
    assert client.get(f"/api/archives/{r['file']}").status_code == 200       # download
    assert client.get("/api/archives/..%2Fsecret.zip").status_code == 404
    assert client.post("/api/archives", json={}).status_code == 400          # nothing left to archive
    assert client.delete(f"/api/archives/{r['file']}").status_code == 403   # setup PIN required
    tok = client.post("/api/setup/unlock", json={"pin": cfg.ui.setup_pin}).json["token"]
    assert client.delete(f"/api/archives/{r['file']}", headers={"X-Setup-Token": tok}).json["ok"]
    assert client.get("/api/archives").json["archives"] == []
