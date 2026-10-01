"""Tests of the extended features: fusion, sensitivity, sub-pixel measurement, dual light,
multi-shot voting, feedback/retraining, drift, calibration, self-test, GPIO (simulated)."""
from __future__ import annotations

import time
from dataclasses import replace

import cv2
import numpy as np
import pytest

from qc import calibration as qcal
from qc import synthetic as S
from qc.alignment import align_part, detect_part, to_gray
from qc.camera import SimulatorSource
from qc.config import AppConfig, DriftConfig, IOConfig, LocalizationConfig, MethodConfig
from qc.drift import DriftMonitor
from qc.io_control import IOController
from qc.methods import MethodResult, find_holes
from qc.model import QCModel, decide
from qc.recipe import Recipe


def shots(part_type, lighting, n, rng, defect=None, **pose):
    return [S.compose(S.make_part(part_type, defect, rng), S.random_pose(rng, **pose), lighting, rng=rng)
            for _ in range(n)]


def pair(rng, defect=None):
    """Same part, same pose, once with backlight and once with front light (dual-light capture)."""
    part = S.make_part("A", defect, rng)
    pose = S.random_pose(rng)
    seed = int(rng.integers(1 << 30))
    return {"back": S.compose(part, pose, "back", rng=np.random.default_rng(seed)),
            "front": S.compose(part, pose, "front", rng=np.random.default_rng(seed + 1))}


# ------------------------------------------------------------------ decision
def test_fusion_rule():
    cfg = MethodConfig(decision="fusion", pca_standalone=1.5)
    r = lambda m, s: MethodResult(m, s < 1, s)            # noqa: E731
    assert decide([r("geometry", 0.2), r("diff", 0.3), r("pca", 1.2)], cfg)[0] is True     # ML alone, weak → OK
    assert decide([r("geometry", 0.2), r("diff", 0.3), r("pca", 1.6)], cfg)[0] is False    # ML alone, strong → NOK
    assert decide([r("geometry", 0.2), r("diff", 1.05), r("pca", 1.2)], cfg)[0] is False   # two agree → NOK
    assert decide([r("geometry", 1.01), r("diff", 0.0), r("pca", 0.0)], cfg)[0] is False   # geometry decides alone
    assert decide([r("geometry", 0.2), r("diff", 0.3), r("pca", 1.2)], replace(cfg, decision="or"))[0] is False


def test_sensitivity_changes_tolerance():
    rng = np.random.default_rng(1)
    model = QCModel.train("A", shots("A", "front", 10, rng), LocalizationConfig(), MethodConfig(methods=["geometry"]))
    part = S.make_part("A", None, rng)
    part.holes[0]["x"] += 4.5                  # 4.5 px offset – inside the default tolerance of 6 px
    img = S.compose(part, S.random_pose(rng), "front", rng=rng)
    assert model.inspect(img, False).ok
    strict = model.inspect(img, False, sensitivity={"geometry": 2.0})       # tolerance 3 px
    assert strict.status == "NOK" and strict.defects[0].kind == "HOLE_POSITION"


# ------------------------------------------------------- sub-pixel measurement
def test_subpixel_hole_measurement():
    """Known sub-pixel shifts must be measured with < 0.06 px error (binary masks: ~0.2 px)."""
    cfg = LocalizationConfig()
    part = S.make_part("A", None, np.random.default_rng(0))
    xs = []
    for dx in np.arange(0, 1.01, 0.1):
        f = S.compose(part, (320 + dx, 240, 0), "back", rng=np.random.default_rng(1))
        g = to_gray(f)
        det = detect_part(g, cfg)
        from qc.alignment import _levels, _render
        a = _render(f, g, det, np.array([[1, 0, -120], [0, 1, -100], [0, 0, 1]], float), (400, 280), *_levels(g, det))
        xs.append(sorted(find_holes(a, 40)[0], key=lambda h: h.cx)[0].cx)
    err = (np.array(xs) - xs[0]) - np.arange(0, 1.01, 0.1)
    assert np.sqrt(np.mean(err ** 2)) < 0.06


# ---------------------------------------------------------------- dual light
def test_dual_light_blind_hole_and_surface():
    rng = np.random.default_rng(4)
    loc = LocalizationConfig()
    mc = MethodConfig()
    recipe = Recipe.train("A dual", [pair(rng) for _ in range(12)], loc,
                          {"back": replace(mc, methods=["geometry"]), "front": mc})
    assert recipe.mode == "dual"
    for _ in range(5):
        assert recipe.inspect(pair(rng), False).ok
    for _ in range(3):
        kinds = {(d.kind, d.channel) for d in recipe.inspect(pair(rng, "blind"), False).defects}
        assert ("BLIND_HOLE", "front") in kinds and ("MISSING_HOLE", "back") not in kinds   # merged correctly
    for _ in range(3):
        r = recipe.inspect(pair(rng, "stain"), False)
        assert r.status == "NOK" and all(d.channel == "front" for d in r.defects)
    for _ in range(3):
        r = recipe.inspect(pair(rng, "diameter"), False)
        assert ("HOLE_DIAMETER", "back") in {(d.kind, d.channel) for d in r.defects}


def test_recipe_save_load_and_legacy(tmp_path):
    rng = np.random.default_rng(6)
    legacy = QCModel.train("Old", shots("A", "front", 6, rng), LocalizationConfig(), MethodConfig(methods=["geometry"]))
    legacy.save(tmp_path)                                    # old layout: meta.json directly in the folder
    r = Recipe.load(tmp_path / "Old")
    assert r.legacy and r.mode == "single"
    r.set_sensitivity({"geometry": 1.3})                     # converts to the recipe layout
    r2 = Recipe.load(tmp_path / "Old")
    assert not r2.legacy and r2.sensitivity["geometry"] == 1.3
    assert len(r2.channels["main"].reference_frames()) == 6  # references survived the conversion


# ------------------------------------------------------ feedback / retraining
def test_feedback_reference_and_retrain(tmp_path):
    rng = np.random.default_rng(7)
    recipe = Recipe.train("Plate", [{"main": f} for f in shots("A", "front", 6, rng)], LocalizationConfig(),
                          {"main": MethodConfig()})
    recipe.save(tmp_path)
    assert recipe.pending_count() == 0
    recipe.add_pending({"main": shots("A", "front", 1, rng)[0]})
    recipe.add_pending({"main": shots("A", "front", 1, rng)[0]})
    assert recipe.pending_count() == 2
    new = recipe.retrain()
    assert new.channels["main"].n_samples == 8 and new.pending_count() == 0
    assert Recipe.load(tmp_path / "Plate").channels["main"].n_samples == 8


# ------------------------------------------------------------------ drift
def test_drift_detects_darker_lighting():
    rng = np.random.default_rng(9)
    model = QCModel.train("A", shots("A", "front", 10, rng), LocalizationConfig(), MethodConfig(methods=["geometry"]))
    mon = DriftMonitor(DriftConfig(window=10))
    mon.reset({"main": model.baseline})
    for f in shots("A", "front", 10, rng):
        mon.update(model.inspect(f, False).measurements)
    assert not mon.state()["warnings"]
    for f in shots("A", "front", 10, rng):
        dark = (f.astype(np.float32) * 0.7).astype(np.uint8)           # LEDs lost 30 % brightness
        mon.update(model.inspect(dark, False).measurements)
    warnings = mon.state()["warnings"]
    assert any("Contrast" in w for w in warnings)


# ---------------------------------------------------------------- calibration
def test_calibration_recovers_scale_and_distortion():
    cols, rows, sq = 9, 6, 10.0
    K = np.array([[1100, 0, 640], [0, 1100, 480], [0, 0, 1]], float)
    D = np.array([-0.25, 0.08, 0, 0, 0], float)
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * sq
    rng = np.random.default_rng(0)
    sets = []
    for i in range(8):
        rvec = rng.uniform(-0.35, 0.35, 3) if i else np.zeros(3)
        tvec = np.array([rng.uniform(-60, 20), rng.uniform(-40, 10), 300.0 if i == 0 else rng.uniform(250, 380)])
        pts, _ = cv2.projectPoints(objp, rvec, tvec, K, D)
        sets.append((pts + rng.normal(0, 0.1, pts.shape)).astype(np.float32))
    cal = qcal.compute(sets, (1280, 960), cols, rows, sq)
    assert cal.camera_matrix is not None
    assert abs(cal.mm_per_px_full - 300 / 1100) / (300 / 1100) < 0.005
    assert abs(np.ravel(cal.dist_coeffs)[0] - D[0]) < 0.02


def test_calibration_rejects_degenerate_images():
    cam = SimulatorSource(AppConfig().camera, seed=2)
    sets = []
    for _ in range(6):                           # all boards flat → focal length/distortion not identifiable
        cam.board, cam._boards_shown = None, 0
        cam.show_board()
        f = cam.read()
        sets.append(qcal.find_corners(f, 9, 6))
    cal = qcal.compute(sets, (f.shape[1], f.shape[0]), 9, 6, 10.0)
    assert cal.camera_matrix is None and cal.note                    # distortion rejected, scale kept
    assert abs(cal.mm_per_px(640) - SimulatorSource.MM_PER_PX) < 0.003


def test_dimensions_in_mm():
    rng = np.random.default_rng(3)
    mm = 0.25
    model = QCModel.train("A", shots("A", "front", 8, rng), LocalizationConfig(),
                          replace(MethodConfig(methods=["geometry"]), pos_tol_mm=0.5), mm_per_px=mm)
    assert model.report["methods"]["geometry"]["hole_diameters_mm"][0] == pytest.approx(34.9 * mm, abs=0.4)
    r = model.inspect(shots("A", "front", 1, rng, "position")[0], False)
    d = next(x for x in r.defects if x.kind == "HOLE_POSITION")
    assert "mm" in d.detail and "tolerance 0.50 mm" in d.detail


# ---------------------------------------------------------------- rotation modes
@pytest.mark.parametrize("mode", ["flip", "off"])
def test_rotation_modes(mode):
    rng = np.random.default_rng(10)
    loc = replace(LocalizationConfig(), rotation_search=mode)
    kw = {"allow_flip": mode == "flip"}
    model = QCModel.train("A", shots("A", "front", 10, rng, **kw), loc, MethodConfig())
    assert all(model.inspect(f, False).ok for f in shots("A", "front", 8, rng, **kw))
    assert all(model.inspect(f, False).status == "NOK" for f in shots("A", "front", 3, rng, "missing", **kw))


# ---------------------------------------------------------------- GPIO (simulated)
def test_io_reject_timing():
    io = IOController(IOConfig(enabled=True, reject_pin=17, reject_delay_ms=150, reject_pulse_ms=50))
    assert io.backend in ("simulated", "gpio")
    t0 = time.time()
    io.signal_result("NOK", t_capture=t0 - 0.1)          # inspection took 100 ms → reject 50 ms later
    io.signal_result("OK", t_capture=t0)
    assert io.rejects == 1
    time.sleep(0.25)
    texts = [e["text"] for e in io.events]
    assert any("reject scheduled in 5" in t or "reject scheduled in 4" in t for t in texts), texts
    assert any("Reject (GPIO 17) ON" in t for t in texts)
    io.close()


# ------------------------------------------------------ system: shots, self-test
def _system(tmp_path, **cam):
    from qc.system import QCSystem

    cfg = AppConfig()
    cfg.camera.fps, cfg.camera.width, cfg.camera.height, cfg.camera.sim_speed_px = 400, 640, 480, 20
    for k, v in cam.items():
        setattr(cfg.camera, k, v)
    cfg.storage.models_dir = str(tmp_path / "models")
    cfg.storage.results_dir = str(tmp_path / "results")
    cfg.storage.archive_dir = str(tmp_path / "archive")
    cfg.calibration.file = str(tmp_path / "calibration.json")
    return QCSystem(cfg, SimulatorSource(cfg.camera, seed=5))


def _wait(cond, timeout=90):
    t0 = time.time()
    while not cond():
        assert time.time() - t0 < timeout, "Timeout"
        time.sleep(0.05)


def test_multishot_voting_and_selftest(tmp_path):
    s = _system(tmp_path, sim_defect_rate=0.5)
    s.start()
    try:
        s.start_learning("Plate")
        _wait(lambda: len(s.learn_captures) >= 7)
        s.finish_learning()
        _wait(lambda: s.mode == "idle" and s.model is not None)
        s.update_settings(shots_per_part=3)
        s.start_inspection()
        _wait(lambda: len(s.history) >= 6)
        s.stop_inspection()
        assert all(len(h["shots"]) >= 1 for h in s.history)
        assert any(len(h["shots"]) == 3 for h in s.history)
        agree = np.mean([(h["status"] == "OK") == (h["ground_truth"] == "ok") for h in s.history])
        assert agree >= 0.8
        s.start_selftest()
        _wait(lambda: s.mode == "idle", 120)
        assert s.last_selftest["passed"] is True
        assert [st["result"] for st in s.last_selftest["steps"]] == ["OK", "NOK"]
    finally:
        s.stop()


def test_system_feedback_flow(tmp_path):
    s = _system(tmp_path, sim_defect_rate=1.0)
    s.start()
    try:
        s.start_learning("Plate")
        _wait(lambda: len(s.learn_captures) >= 6)
        s.finish_learning()
        _wait(lambda: s.mode == "idle" and s.model is not None)
        s.start_inspection()
        _wait(lambda: any(h["status"] == "NOK" for h in s.history))
        s.stop_inspection()
        rid = next(h["image_id"] for h in s.history if h["status"] == "NOK")
        out = s.feedback(rid, "ok", add_reference=True)
        assert out["outcome"] == "false_alarm" and out["pending"] == 1
        assert s.feedback_counts["false_alarm"] == 1
        s.feedback(rid, "nok")                                  # supervisor changes his mind
        assert s.feedback_counts == {"false_alarm": 0, "missed": 0, "confirmed": 1}
        from qc import archive as qa
        summary = qa.summarize(s.cfg.storage.results_dir)["supervisor_feedback"]
        assert summary["reviewed"] == 1 and summary["confirmed"] == 1          # last word counts
    finally:
        s.stop()


def test_defect_priority_independent_of_method_order():
    """Regression: the UI sends methods alphabetically (diff before geometry) – the specific
    geometry finding must still win over the generic surface deviation at the same spot."""
    rng = np.random.default_rng(11)
    mc = MethodConfig(methods=["diff", "geometry", "pca"])
    recipe = Recipe.train("A", [pair(rng) for _ in range(10)], LocalizationConfig(),
                          {"back": replace(mc, methods=["geometry"]), "front": mc})
    kinds = [d.kind for d in recipe.inspect(pair(rng, "blind"), False).defects]
    assert kinds == ["BLIND_HOLE"], kinds


# ------------------------------------------------------------------ self-learning
def _subtle_offset(rng, d=8.0):
    import math
    p = S.make_part("A", None, rng)
    a = rng.uniform(0, 2 * math.pi)
    p.holes[0]["x"] += d * math.cos(a)
    p.holes[0]["y"] += d * math.sin(a)
    return {"main": S.compose(p, S.random_pose(rng), "front", rng=rng)}


def _plate_recipe(tmp_path, rng, n=15):
    r = Recipe.train("Plate", [{"main": f} for f in shots("A", "front", n, rng)], LocalizationConfig(),
                     {"main": MethodConfig()})
    r.save(tmp_path)
    r.add_known_bad({"main": shots("A", "front", 1, rng, "scratch")[0]}, "selftest")
    r.add_known_bad({"main": shots("A", "front", 1, rng, "position")[0]}, "confirmed")
    return r


def test_self_learning_clean_pool_accepted(tmp_path):
    rng = np.random.default_rng(21)
    r = _plate_recipe(tmp_path, rng)
    nominal = r.channels["main"].methods["geometry"].exp_xy.copy()
    thr0 = r.channels["main"].methods["pca"].threshold
    for f in shots("A", "front", 40, rng):
        r.add_collected({"main": f}, max_pool=100)
    new, rep = r.learn_collected(max_references=80)
    assert rep["accepted"] and new is not None, rep["reasons"]
    ch = new.channels["main"]
    assert ch.n_anchor == 15 and len(ch.learned_frames()) == 40
    assert np.allclose(ch.methods["geometry"].exp_xy, nominal)          # geometry anchored to the taught-in parts
    assert ch.methods["pca"].threshold < thr0                              # ML became stricter
    assert rep["known_bad"] == {"available": 2, "checked": 2, "still_detected": 2}
    loaded = Recipe.load(tmp_path / "Plate")
    assert loaded.collected_count() == 0 and len(loaded.channels["main"].learned_frames()) == 40
    again = loaded.retrain()                                               # normal retraining keeps learned parts
    assert len(again.channels["main"].learned_frames()) == 40 and again.channels["main"].n_anchor == 15


def test_self_learning_contaminated_pool_rejected(tmp_path):
    rng = np.random.default_rng(22)
    r = _plate_recipe(tmp_path, rng)
    for f in shots("A", "front", 37, rng):
        r.add_collected({"main": f}, max_pool=100)
    for _ in range(3):                                   # parts with an 8 px hole offset slipped into the pool
        r.add_collected(_subtle_offset(rng), max_pool=100)
    new, rep = r.learn_collected(max_references=80)
    assert new is None and not rep["accepted"] and rep["reasons"]
    loaded = Recipe.load(tmp_path / "Plate")
    assert len(loaded.channels["main"].learned_frames()) == 0              # model unchanged
    assert loaded.collected_count() == 40 and loaded.self_learning_log[-1]["accepted"] is False


def test_pool_cap_reservoir(tmp_path):
    rng = np.random.default_rng(23)
    r = _plate_recipe(tmp_path, rng, n=5)
    frame = {"main": shots("A", "front", 1, rng)[0]}
    for _ in range(30):
        r.add_collected(frame, max_pool=8, rng=rng)
    assert r.collected_count() == 8 and r.collected_seen == 30


def test_system_collects_only_clear_good_parts_and_missed_feedback(tmp_path):
    s = _system(tmp_path, sim_defect_rate=0.4)
    s.cfg.self_learning.max_pool = 50
    s.cfg.self_learning.auto = False           # manual button (setup area) in this test
    s.start()
    try:
        s.start_learning("Plate")
        _wait(lambda: len(s.learn_captures) >= 8)
        s.finish_learning()
        _wait(lambda: s.mode == "idle" and s.model is not None)
        s.start_inspection()
        _wait(lambda: sum(1 for h in s.history if h.get("collected")) >= 3 and
                      any(h["status"] == "NOK" for h in s.history), 120)
        s.stop_inspection()
        collected = [h for h in s.history if h.get("collected")]
        assert all(h["status"] == "OK" and h["ground_truth"] == "ok" for h in collected)
        assert all(max(m["score"] for m in h["methods"]) < s.cfg.self_learning.margin for h in collected)
        assert s.model.collected_count() == len(collected)
        # the supervisor says one of them was actually defective → removed from the pool, kept as known bad
        s.feedback(collected[0]["image_id"], "nok")
        assert s.model.collected_count() == len(collected) - 1 and s.model.known_bad_count() == 1
        s.learn_collected(s.model.slug)
        _wait(lambda: s.mode == "idle" and s.last_self_learning is not None, 120)
        assert s.last_self_learning["accepted"], s.last_self_learning["reasons"]
        assert len(s.model.channels["main"].learned_frames()) == len(collected) - 1
    finally:
        s.stop()


def test_simple_teach_in_and_automatic_self_learning(tmp_path):
    """Supervisor flow: teach-in finishes by itself at the target count; during inspection the
    collected good parts are trained in automatically in the background – inspection keeps running."""
    s = _system(tmp_path, sim_defect_rate=0.2)
    s.cfg.target_reference_count = 8
    s.cfg.self_learning.auto_every = 6
    s.start()
    try:
        s.start_learning("Plate", auto_finish=True)
        _wait(lambda: s.mode == "idle" and s.model is not None, 120)          # no "finish" click needed
        assert s.model.channels["main"].n_anchor == 8
        s.start_inspection()
        _wait(lambda: s.last_self_learning is not None, 180)
        assert s.mode == "inspecting"                                         # never interrupted
        rep = s.last_self_learning
        assert rep["automatic"] and rep["accepted"], rep["reasons"]
        _wait(lambda: not s.status()["self_learning_running"])
        assert len(s.model.channels["main"].learned_frames()) >= 6
        assert s.model.collected_count() < 6                                  # pool was emptied
        s.stop_inspection()
        st = s.status()
        assert isinstance(st["notices"], list) and st["learning"]["minimum"] >= 1
    finally:
        s.stop()


def test_setup_pin_protects_expert_functions(tmp_path):
    from qc.webapp import create_app

    s = _system(tmp_path)
    s.cfg.ui.setup_pin = "4711"
    c = create_app(s).test_client()
    # supervisor functions work without PIN
    assert c.post("/api/counters/reset").status_code == 403                   # setup function
    assert c.post("/api/settings", json={"shots_per_part": 3}).status_code == 403
    assert s.cfg.inspection.shots_per_part == 1
    assert c.post("/api/setup/unlock", json={"pin": "0000"}).status_code == 403
    tok = c.post("/api/setup/unlock", json={"pin": "4711"}).json["token"]
    h = {"X-Setup-Token": tok}
    assert c.get("/api/setup/check", headers=h).json["unlocked"]
    assert c.post("/api/settings", json={"shots_per_part": 3}, headers=h).json["ok"]
    assert s.cfg.inspection.shots_per_part == 3
    c.post("/api/setup/lock", headers=h)
    assert c.post("/api/settings", json={"shots_per_part": 1}, headers=h).status_code == 403
    for _ in range(5):                                                        # brute force → lockout
        c.post("/api/setup/unlock", json={"pin": "1"})
    assert c.post("/api/setup/unlock", json={"pin": "4711"}).status_code == 429
    s.cfg.ui.setup_pin = ""                                                   # no PIN configured → open
    assert c.post("/api/counters/reset").json["ok"]
