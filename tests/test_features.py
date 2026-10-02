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
from qc.alignment import detect_part, to_gray
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
        assert s.shots_per_part == 3                           # always on, no setting needed
        s.cfg.inspection.shots_per_part = 1
        assert s.shots_per_part == 3                           # cannot be switched off
        s.cfg.inspection.confirm_margin = 0.0                  # evaluate every image in this test
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
    assert c.post("/api/settings", json={"self_learning": False}).status_code == 403
    assert s.cfg.self_learning.enabled is True
    assert c.post("/api/setup/unlock", json={"pin": "0000"}).status_code == 403
    tok = c.post("/api/setup/unlock", json={"pin": "4711"}).json["token"]
    h = {"X-Setup-Token": tok}
    assert c.get("/api/setup/check", headers=h).json["unlocked"]
    assert c.post("/api/settings", json={"self_learning": False}, headers=h).json["ok"]
    assert s.cfg.self_learning.enabled is False
    c.post("/api/setup/lock", headers=h)
    assert c.post("/api/settings", json={"self_learning": True}, headers=h).status_code == 403
    for _ in range(5):                                                        # brute force → lockout
        c.post("/api/setup/unlock", json={"pin": "1"})
    assert c.post("/api/setup/unlock", json={"pin": "4711"}).status_code == 429
    s.cfg.ui.setup_pin = ""                                                   # no PIN configured → open
    assert c.post("/api/counters/reset").json["ok"]


def test_focus_assistant_measures_sharpness(tmp_path):
    """Sharp image → higher value than a blurred one; the overlay runs inside the preview."""
    from qc.system import QCSystem

    rng = np.random.default_rng(31)
    sharp = cv2.cvtColor(shots("A", "front", 1, rng)[0], cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(sharp, (0, 0), 2.5)
    assert QCSystem.sharpness(sharp) > 2 * QCSystem.sharpness(blurred)
    s = _system(tmp_path)
    s.set_focus_assist(True)
    frame = s._read()
    s.trigger.update(frame)
    s._update_preview(frame)
    st = s.status()
    assert st["focus_assist"]["value"] > 0 and st["focus_assist"]["best"] >= st["focus_assist"]["value"]
    s.set_focus_assist(False)
    assert s.status()["focus_assist"] is None


def test_undistortion_only_for_inspected_frames(tmp_path):
    """The loop works on raw frames; only learned/inspected captures are undistorted."""
    s = _system(tmp_path)
    calls = []
    s.undistort = lambda f: (calls.append(1), f)[1]
    frame = s._read()
    assert not calls                                   # trigger/preview frame: no remap
    out = s._undistort_shots([{"main": frame}, {"main": frame}])
    assert len(calls) == 2 and out[0]["main"] is frame


# ------------------------------------------------------------ automatic camera setup
def test_autosetup_building_blocks():
    from qc import autosetup as AS
    from qc.config import AutoSetupConfig

    ac = AutoSetupConfig()
    e, g, done = AS.next_exposure(4000, 1.0, 255, ac)              # clipped → halve
    assert e == 2000 and g == 1.0 and not done
    e, g, done = AS.next_exposure(4000, 1.0, 112.5, ac)            # too dark → exposure ×2
    assert e == 8000 and not done
    e, g = AS.split_exposure(16000, 4000, ac)                      # above the limit → gain
    assert e == 4000 and g == 4.0
    crop, zoom = AS.crop_for_path((0.5, 0.5), (0.2, 0.267), "x", ac)
    assert abs(crop[2] - crop[3]) < 1e-9 and 1.0 < zoom <= ac.max_zoom          # square pixels, zoomed in
    assert crop[0] >= 0 and crop[0] + crop[2] <= 1 and crop[1] + crop[3] <= 1
    crop, zoom = AS.crop_for_path((0.9, 0.1), (0.05, 0.067), "y", ac)          # clamped to sensor and max zoom
    assert zoom == ac.max_zoom and crop[0] + crop[2] <= 1.0 + 1e-9 and crop[1] >= 0
    track = [(i * 0.1, 0.1 + i * 0.05, 0.5 + 0.001 * i) for i in range(10)]
    path = AS.path_from_track(track)
    assert path["axis"] == "x" and abs(path["across"] - 0.5) < 0.01 and abs(path["speed"] - 0.5) < 0.05
    assert AS.path_from_track([(0, 0.5, 0.5), (1, 0.5, 0.5), (2, 0.51, 0.5)]) is None      # did not move


def test_edge_width_detects_blur():
    from qc import autosetup as AS

    rng = np.random.default_rng(41)
    frame = shots("A", "front", 1, rng)[0]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    det = detect_part(gray, LocalizationConfig())
    sharp = AS.edge_width(gray, det.contour)
    blurry = AS.edge_width(cv2.GaussianBlur(gray, (0, 0), 2.5), det.contour)
    assert sharp < 3.0 < blurry, (sharp, blurry)


def test_automatic_camera_setup_end_to_end(tmp_path):
    """Badly over-exposed camera + small parts: the setup fixes exposure, measures belt direction
    and speed, zooms in, the profile is stored with the model and re-applied when it is selected."""
    s = _system(tmp_path, sim_defect_rate=0.0, exposure_us=9000, sim_part_scale=0.5, fps=60, sim_speed_px=12)
    s.cfg.target_reference_count = 6
    s.start()
    try:
        s.start_camera_setup()
        _wait(lambda: s.mode == "idle", 60)
        assert s.cam_setup["error"] is None, s.cam_setup
        prof = s.pending_profile
        assert prof["exposure_us"] < 9000 and prof["crop"] == [0.0, 0.0, 1.0, 1.0]
        s.start_learning("Small plate", auto_finish=True)
        _wait(lambda: s.learn_path is None, 60)                 # first part: belt direction + zoom
        assert s.learn_profile["axis"] == "x" and s.learn_profile["zoom"] > 1.3
        assert s.learn_profile["motion_blur_px"] <= s.cfg.autosetup.max_blur_px + 1e-6
        _wait(lambda: s.mode == "idle" and s.model is not None, 180)
        cp = s.model.camera_profile
        assert cp["auto"] and cp["zoom"] > 1.3 and cp["sharp"]
        assert abs(s.camera.crop[2] - 1 / cp["zoom"]) < 0.01
        # no model → base settings; selecting the part again restores its profile
        s._apply_camera(None)
        assert s.camera.crop == (0.0, 0.0, 1.0, 1.0)
        s.select_model(s.model.slug)
        assert abs(s.camera.crop[2] - 1 / cp["zoom"]) < 0.01
        s.start_inspection()
        _wait(lambda: len(s.history) >= 4, 120)
        s.stop_inspection()
        assert all(h["status"] == "OK" for h in s.history), [h["defects"] for h in s.history]
    finally:
        s.stop()


def test_automatic_focus_on_cameras_with_focus_motor(tmp_path):
    s = _system(tmp_path, sim_autofocus=True, sim_defocus=2.5, fps=60)
    s.start()
    try:
        s.start_camera_setup()
        _wait(lambda: s.mode == "idle", 60)
        r = s.cam_setup["result"]
        assert r["autofocus"] and r["lens_position"] == 4.0 and s.camera.defocus == 0
        assert r["edge_width_px"] < 3.0
    finally:
        s.stop()


# ------------------------------------------------------------ robustness / refactoring
def test_overloaded_worker_rejects_uninspected_parts(tmp_path):
    """Queue full (Pi cannot keep up): the part is rejected without inspection (fail-safe) and reported."""
    s = _system(tmp_path)
    s.cfg.io.reject_pin = 17                       # simulated GPIO
    s.mode = "inspecting"
    frame = {"main": np.zeros((48, 64, 3), np.uint8)}
    for _ in range(s.QUEUE_PARTS):                 # worker not started → the queue fills up
        s._submit([frame], time.time(), False)
    assert s.skipped == 0
    s._submit([frame], time.time(), False)
    assert s.skipped == 1 and s.io.rejects == 1
    assert any("without inspection" in n["text"] for n in s.supervisor_notices())
    s.io.close()


def test_config_validation_reports_typos_and_bad_values():
    from qc.config import ConfigError

    for f in ("config.example.json", "config.pi3_imx219.json"):
        AppConfig.load(f)                                                   # shipped files are valid
    with pytest.raises(ConfigError, match="exposure"):
        AppConfig.from_dict({"camera": {"exposure": 3000}})                # typo of exposure_us
    with pytest.raises(ConfigError, match="trigger.axis"):
        AppConfig.from_dict({"trigger": {"axis": "z"}})
    with pytest.raises(ConfigError, match="shots_per_part"):
        AppConfig.from_dict({"inspection": {"shots_per_part": 1}})


def test_settings_from_the_setup_area_survive_a_restart(tmp_path):
    s = _system(tmp_path)
    s.update_settings(self_learning_auto=False)
    s.set_method_defaults({"methods": ["geometry", "diff"], "pca_representation": "norm"})
    s.io.close()
    s2 = _system(tmp_path)                                                  # "restart"
    assert s2.cfg.self_learning.auto is False
    assert s2.cfg.method.methods == ["geometry", "diff"]
    s2.io.close()


def test_model_update_can_be_undone(tmp_path):
    rng = np.random.default_rng(51)
    r = Recipe.train("Plate", [{"main": f} for f in shots("A", "front", 8, rng)], LocalizationConfig(),
                     {"main": MethodConfig(methods=["geometry"])})
    r.save(tmp_path)
    assert not Recipe.has_previous(tmp_path / "Plate")
    r.add_collected({"main": shots("A", "front", 1, rng)[0]}, max_pool=10)
    r2 = Recipe.train("Plate", [{"main": f} for f in shots("A", "front", 6, rng)], LocalizationConfig(),
                      {"main": MethodConfig(methods=["geometry", "diff"])})
    r2.save(tmp_path)                                                       # same name → old version kept
    assert Recipe.has_previous(tmp_path / "Plate")
    restored = Recipe.restore_previous(tmp_path / "Plate")
    assert list(restored.channels["main"].methods) == ["geometry"]
    assert restored.collected_count() == 1                                   # side folders stay with the part
    again = Recipe.restore_previous(tmp_path / "Plate")                     # the undo can be undone
    assert list(again.channels["main"].methods) == ["geometry", "diff"]


def test_pi_camera_source_against_fake_picamera2(monkeypatch):
    """The Pi camera code (normally only runnable on a Pi) against a stand-in for picamera2."""
    import sys
    import types

    class FakeRequest:
        def make_array(self, name):
            return np.zeros((480, 640, 3), np.uint8)

        def get_metadata(self):
            return {"SensorTimestamp": time.monotonic_ns() - 20_000_000, "ColourGains": (1.5, 1.7)}

        def release(self):
            pass

    class FakePicamera2:
        camera_controls = {"ScalerCrop": ((0, 0, 64, 64), (0, 0, 3280, 2464), (0, 0, 3280, 2464)), "ExposureTime": (1, 1, 1)}
        camera_properties = {"Model": "imx219", "PixelArraySize": (3280, 2464)}

        def __init__(self):
            self.controls, self.video_kwargs = {}, None

        def create_video_configuration(self, **kw):
            self.video_kwargs = kw
            return kw

        def configure(self, c): pass
        def start(self): pass
        def stop(self): pass
        def close(self): pass

        def set_controls(self, d):
            self.controls.update(d)

        def capture_request(self):
            return FakeRequest()

        def capture_metadata(self):
            return {"ScalerCrop": (0, 0, 3280, 2464)}

        def camera_configuration(self):
            return {"sensor": {"output_size": (1640, 1232)}}

    monkeypatch.setitem(sys.modules, "picamera2", types.SimpleNamespace(Picamera2=FakePicamera2))
    monkeypatch.setattr(time, "sleep", lambda s: None)
    from qc.camera import PiCameraSource
    from qc.config import CameraConfig

    cfg = CameraConfig(source="pi", width=640, height=480, sensor_mode=[1640, 1232], exposure_us=2000)
    cam = PiCameraSource(cfg)
    assert cam.cam.video_kwargs["sensor"] == {"output_size": (1640, 1232)} and cam.cam.video_kwargs["queue"] is False
    assert not cam.has_autofocus and "fixed-focus" in cam.info and "readout 1640×1232" in cam.info
    assert cam.cam.controls["AeEnable"] is False and cam.cam.controls["ExposureTime"] == 2000
    frame = cam.read()
    assert frame.shape == (480, 640, 3) and 0.01 < cam.last_age_s < 0.5            # age from the sensor timestamp
    assert cam.awb_result() == (1.5, 1.7)
    cam.set_crop((0.25, 0.25, 0.5, 0.5))
    assert cam.cam.controls["ScalerCrop"] == (820, 616, 1640, 1232) and cam.crop == (0.25, 0.25, 0.5, 0.5)
    cam.set_exposure(800, 2.0)
    assert cam.cam.controls["ExposureTime"] == 800 and cam.cam.controls["AnalogueGain"] == 2.0
