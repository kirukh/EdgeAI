"""Tests of the features around the core pipeline: decision rule, sensitivity, sub-pixel measurement,
drift, mm dimensions, multi-shot voting, supervisor feedback, camera setup, web security, config."""

from __future__ import annotations

import json
import time
from dataclasses import replace
from typing import ClassVar

import cv2
import numpy as np
import pytest
from conftest import shots, wait, web_client

from qc import synthetic as S
from qc.alignment import detect_part, levels, render, to_gray
from qc.config import AppConfig, ConfigError, DriftConfig, LocalizationConfig, MethodConfig
from qc.drift import DriftMonitor
from qc.inspection import vote
from qc.methods import MethodResult, find_holes
from qc.model import InspectionResult, QCModel, decide


# ------------------------------------------------------------------ decision
def test_fusion_rule():
    cfg = MethodConfig(decision="fusion", pca_standalone=1.5)

    def r(m, s):
        return MethodResult(m, s < 1, s)

    assert decide([r("geometry", 0.2), r("diff", 0.3), r("pca", 1.2)], cfg)[0] is True  # ML alone, weak → OK
    assert decide([r("geometry", 0.2), r("diff", 0.3), r("pca", 1.6)], cfg)[0] is False  # ML alone, strong → NOK
    assert decide([r("geometry", 0.2), r("diff", 1.05), r("pca", 1.2)], cfg)[0] is False  # two agree → NOK
    assert decide([r("geometry", 1.01), r("diff", 0.0), r("pca", 0.0)], cfg)[0] is False  # geometry decides alone
    assert decide([r("geometry", 0.2), r("diff", 0.3), r("pca", 1.2)], replace(cfg, decision="or"))[0] is False


def test_sensitivity_changes_tolerance():
    rng = np.random.default_rng(1)
    model = QCModel.train("A", shots("A", "front", 10, rng), LocalizationConfig(), MethodConfig(methods=["geometry"]))
    part = S.make_part("A", None, rng)
    part.holes[0]["x"] += 4.5  # 4.5 px offset – inside the default tolerance of 6 px
    img = S.compose(part, S.random_pose(rng), "front", rng=rng)
    assert model.inspect(img, False).ok
    strict = model.inspect(img, False, sensitivity={"geometry": 2.0})  # tolerance 3 px
    assert strict.status == "NOK" and strict.defects[0].kind == "HOLE_POSITION"
    model.set_sensitivity({"geometry": 2.0})  # stored in the model → used by default
    assert model.inspect(img, False).status == "NOK"


def test_subpixel_hole_measurement():
    """Known sub-pixel shifts must be measured with < 0.06 px error (binary masks: ~0.2 px)."""
    cfg = LocalizationConfig()
    part = S.make_part("A", None, np.random.default_rng(0))
    xs = []
    for dx in np.arange(0, 1.01, 0.1):
        f = S.compose(part, (320 + dx, 240, 0), "back", rng=np.random.default_rng(1))
        g = to_gray(f)
        det = detect_part(g, cfg)
        a = render(f, g, det, np.array([[1, 0, -120], [0, 1, -100], [0, 0, 1]], float), (400, 280), *levels(g, det))
        xs.append(sorted(find_holes(a, 40)[0], key=lambda h: h.cx)[0].cx)
    err = (np.array(xs) - xs[0]) - np.arange(0, 1.01, 0.1)
    assert np.sqrt(np.mean(err**2)) < 0.06


def test_drift_detects_darker_lighting():
    rng = np.random.default_rng(9)
    model = QCModel.train("A", shots("A", "front", 10, rng), LocalizationConfig(), MethodConfig(methods=["geometry"]))
    mon = DriftMonitor(DriftConfig(window=10))
    mon.reset(model.baseline)
    for f in shots("A", "front", 10, rng):
        mon.update(model.inspect(f, False).measurements)
    assert not mon.state()["warnings"]
    for f in shots("A", "front", 10, rng):
        dark = (f.astype(np.float32) * 0.7).astype(np.uint8)  # LEDs lost 30 % brightness
        mon.update(model.inspect(dark, False).measurements)
    assert any("Contrast" in w for w in mon.state()["warnings"])


def test_dimensions_in_mm():
    rng = np.random.default_rng(3)
    mm = 0.25
    model = QCModel.train(
        "A",
        shots("A", "front", 8, rng),
        LocalizationConfig(),
        replace(MethodConfig(methods=["geometry"]), pos_tol_mm=0.5),
        mm_per_px=mm,
    )
    assert model.report["methods"]["geometry"]["hole_diameters_mm"][0] == pytest.approx(34.9 * mm, abs=0.4)
    r = model.inspect(shots("A", "front", 1, rng, "position")[0], False)
    d = next(x for x in r.defects if x.kind == "HOLE_POSITION")
    assert "mm" in d.detail and "tolerance 0.50 mm" in d.detail


@pytest.mark.parametrize("mode", ["flip", "off"])
def test_rotation_modes(mode):
    rng = np.random.default_rng(10)
    loc = replace(LocalizationConfig(), rotation_search=mode)
    kw = {"allow_flip": mode == "flip"}
    model = QCModel.train("A", shots("A", "front", 10, rng, **kw), loc, MethodConfig())
    assert all(model.inspect(f, False).ok for f in shots("A", "front", 8, rng, **kw))
    assert all(model.inspect(f, False).status == "NOK" for f in shots("A", "front", 3, rng, "missing", **kw))


# ------------------------------------------------------------------ majority vote
def _res(status, worst=0.0):
    return InspectionResult(status, "m", "t", 10.0, worst=worst)


def test_vote():
    assert vote([_res("OK"), _res("NOK", 2), _res("OK")]).status == "OK"
    nok = vote([_res("NOK", 1.2), _res("NOK", 3.0), _res("OK")])
    assert nok.status == "NOK" and nok.worst == 3.0  # the clearest NOK shot is shown
    assert nok.shots == ["NOK", "NOK", "OK"] and nok.time_ms == 30.0
    assert vote([_res("NO_PART"), _res("OK"), _res("NOK")]).status == "NOK"  # tie → safe side
    assert vote([_res("NO_PART"), _res("OK"), _res("NOK")], tie_is_nok=False).status == "OK"
    assert vote([_res("NO_PART"), _res("NO_PART")]).status == "NO_PART"


# ------------------------------------------------------------------ system
def test_multishot_voting(make_system):
    s = make_system(sim_defect_rate=0.5)
    s.start()
    s.start_learning("Plate")
    wait(lambda: s.teach.count >= 7)
    s.finish_learning()
    wait(lambda: s.mode == "idle" and s.model is not None)
    assert s.shots_per_part == 3  # always on, no setting needed
    s.cfg.inspection.shots_per_part = 1
    assert s.shots_per_part == 3  # cannot be switched off
    s.cfg.inspection.confirm_margin = 0.0  # evaluate every image in this test
    s.start_inspection()
    wait(lambda: len(s.results.history) >= 6)
    s.stop_inspection()
    hist = list(s.results.history)
    assert any(len(h["shots"]) == 3 for h in hist)
    agree = np.mean([(h["status"] == "OK") == (h["ground_truth"] == "ok") for h in hist])
    assert agree >= 0.8


def test_supervisor_feedback_flow(make_system):
    s = make_system(sim_defect_rate=1.0)
    s.start()
    s.start_learning("Plate")
    wait(lambda: s.teach.count >= 6)
    s.finish_learning()
    wait(lambda: s.mode == "idle" and s.model is not None)
    s.start_inspection()
    wait(lambda: any(h["status"] == "NOK" for h in s.results.history))
    s.stop_inspection()
    rid = next(h["image_id"] for h in s.results.history if h["status"] == "NOK")
    from qc.learning import KNOWN_BAD, PENDING, LearningStore

    store = LearningStore(s.model.path)
    out = s.feedback(rid, "ok")  # "within our tolerance" → waits for the next model update
    assert out["outcome"] == "false_alarm" and out["waiting"] == 1 and store.count(PENDING) == 1
    assert s.results.feedback_counts["false_alarm"] == 1
    s.feedback(rid, "nok")  # supervisor changes their mind – the last word counts
    assert s.results.feedback_counts == {"false_alarm": 0, "missed": 0, "confirmed": 1}
    assert store.count(PENDING) == 0 and store.count(KNOWN_BAD) == 1  # now a test part instead
    from qc import archive as qa

    summary = qa.summarize(s.cfg.storage.results_dir)["supervisor_feedback"]
    assert summary["reviewed"] == 1 and summary["confirmed"] == 1
    with pytest.raises(ValueError):
        s.feedback(rid, "maybe")


def test_teach_in_finishes_by_itself(make_system):
    s = make_system(sim_defect_rate=0.2)
    s.cfg.target_reference_count = 6
    s.teach.target = 6
    s.start()
    s.start_learning("Plate", auto_finish=True)
    wait(lambda: s.mode == "idle" and s.model is not None, 120)  # no "finish" click needed
    assert s.model.n_samples == 6 and s.model.path is not None
    st = s.status()
    assert isinstance(st["notices"], list) and st["learning"]["minimum"] >= 1


def test_overloaded_worker_logs_uninspected_parts(make_system):
    """Queue full (Pi cannot keep up): the part is logged as not inspected and reported."""
    s = make_system()
    s.mode = "inspecting"
    frame = np.zeros((48, 64, 3), np.uint8)
    for _ in range(3):  # worker not started → the queue fills up
        s._submit([frame], False)
    assert s.results.skipped == 0
    s._submit([frame], False)
    assert s.results.skipped == 1
    assert any("could not be inspected" in n["text"] for n in s.supervisor_notices())
    log = next((s.results.day_dir()).glob("inspection_log.csv")).read_text(encoding="utf-8")
    assert "NOT_INSPECTED" in log


def test_focus_assistant_measures_sharpness(make_system):
    """Sharp image → higher value than a blurred one; the overlay runs inside the preview."""
    from qc.camera_setup import CameraSetup

    rng = np.random.default_rng(31)
    sharp = cv2.cvtColor(shots("A", "front", 1, rng)[0], cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(sharp, (0, 0), 2.5)
    assert CameraSetup.sharpness(sharp) > 2 * CameraSetup.sharpness(blurred)
    s = make_system()
    s.set_focus_assist(True)
    frame = s._read()
    s.trigger.update(frame)
    s._update_preview(frame)
    st = s.status()
    assert st["focus_assist"]["value"] > 0 and st["focus_assist"]["best"] >= st["focus_assist"]["value"]
    s.set_focus_assist(False)
    assert s.status()["focus_assist"] is None


# ------------------------------------------------------------ automatic camera setup
def test_autosetup_building_blocks():
    from qc import autosetup as AS
    from qc.config import AutoSetupConfig

    ac = AutoSetupConfig()
    e, g, done = AS.next_exposure(4000, 1.0, 255, ac)  # clipped → halve
    assert e == 2000 and g == 1.0 and not done
    e, g, done = AS.next_exposure(4000, 1.0, 112.5, ac)  # too dark → exposure ×2
    assert e == 8000 and not done
    e, g = AS.split_exposure(16000, 4000, ac)  # above the limit → gain
    assert e == 4000 and g == 4.0
    crop, zoom = AS.crop_for_path((0.5, 0.5), (0.2, 0.267), "x", ac)
    assert abs(crop[2] - crop[3]) < 1e-9 and 1.0 < zoom <= ac.max_zoom  # square pixels, zoomed in
    assert crop[0] >= 0 and crop[0] + crop[2] <= 1 and crop[1] + crop[3] <= 1
    crop, zoom = AS.crop_for_path((0.9, 0.1), (0.05, 0.067), "y", ac)  # clamped to sensor and max zoom
    assert zoom == ac.max_zoom and crop[0] + crop[2] <= 1.0 + 1e-9 and crop[1] >= 0
    track = [(i * 0.1, 0.1 + i * 0.05, 0.5 + 0.001 * i) for i in range(10)]
    path = AS.path_from_track(track)
    assert path["axis"] == "x" and abs(path["across"] - 0.5) < 0.01 and abs(path["speed"] - 0.5) < 0.05
    assert AS.path_from_track([(0, 0.5, 0.5), (1, 0.5, 0.5), (2, 0.51, 0.5)]) is None  # did not move


def test_edge_width_detects_blur():
    from qc import autosetup as AS

    rng = np.random.default_rng(41)
    gray = cv2.cvtColor(shots("A", "front", 1, rng)[0], cv2.COLOR_BGR2GRAY)
    det = detect_part(gray, LocalizationConfig())
    sharp = AS.edge_width(gray, det.contour)
    blurry = AS.edge_width(cv2.GaussianBlur(gray, (0, 0), 2.5), det.contour)
    assert sharp < 3.0 < blurry, (sharp, blurry)


def test_automatic_camera_setup_end_to_end(make_system):
    """Badly over-exposed camera + small parts: the setup fixes exposure, measures belt direction
    and speed, zooms in, the profile is stored with the model and re-applied when it is selected."""
    s = make_system(sim_defect_rate=0.0, exposure_us=9000, sim_part_scale=0.5, fps=60, sim_speed_px=12)
    s.cfg.target_reference_count = 6
    s.teach.target = 6
    s.start()
    s.start_camera_setup()
    wait(lambda: s.mode == "idle", 60)
    assert s.cam.status["error"] is None, s.cam.status
    prof = s.cam.pending_profile
    assert prof["exposure_us"] < 9000 and prof["crop"] == [0.0, 0.0, 1.0, 1.0]
    s.start_learning("Small plate", auto_finish=True)
    wait(lambda: s.cam.path is None, 60)  # first part: belt direction + zoom
    assert s.cam.profile["axis"] == "x" and s.cam.profile["zoom"] > 1.3
    assert s.cam.profile["motion_blur_px"] <= s.cfg.autosetup.max_blur_px + 1e-6
    wait(lambda: s.mode == "idle" and s.model is not None, 180)
    cp = s.model.camera_profile
    assert cp["auto"] and cp["zoom"] > 1.3 and cp["sharp"]
    assert abs(s.camera.crop[2] - 1 / cp["zoom"]) < 0.01
    s.cam.apply(None)  # no model → base settings; selecting the part again restores its profile
    assert s.camera.crop == (0.0, 0.0, 1.0, 1.0)
    s.select_model(s.model.slug)
    assert abs(s.camera.crop[2] - 1 / cp["zoom"]) < 0.01
    s.start_inspection()
    wait(lambda: len(s.results.history) >= 4, 120)
    s.stop_inspection()
    assert all(h["status"] == "OK" for h in s.results.history), [h["defects"] for h in s.results.history]


def test_automatic_focus_on_cameras_with_focus_motor(make_system):
    s = make_system(sim_autofocus=True, sim_defocus=2.5, fps=60)
    s.start()
    s.start_camera_setup()
    wait(lambda: s.mode == "idle", 60)
    r = s.cam.status["result"]
    assert r["autofocus"] and r["lens_position"] == 4.0 and s.camera.defocus == 0
    assert r["edge_width_px"] < 3.0


# ------------------------------------------------------------------ web security
def test_setup_pin_protects_expert_functions(make_system):
    s = make_system()
    s.cfg.ui.setup_pin = "4711"
    c = web_client(s)
    assert c.post("/api/counters/reset").status_code == 403  # setup function
    assert c.post("/api/focus", json={"on": True}).status_code == 403
    assert c.post("/api/setup/unlock", json={"pin": "0000"}).status_code == 403
    tok = c.post("/api/setup/unlock", json={"pin": "4711"}).json["token"]
    h = {"X-Setup-Token": tok}
    assert c.get("/api/setup/check", headers=h).json["unlocked"]
    assert c.post("/api/focus", json={"on": True}, headers=h).json["ok"]
    assert s.cam.focus is not None
    c.post("/api/setup/lock", headers=h)
    assert c.post("/api/focus", json={"on": False}, headers=h).status_code == 403
    s.cfg.ui.setup_pin = ""  # no PIN configured → open
    assert c.post("/api/counters/reset").json["ok"]


def test_pin_lockout_is_per_device(make_system):
    """5 wrong PINs block only the device that typed them (no lock-out of the technician)."""
    s = make_system()
    s.cfg.ui.setup_pin = "4711"
    c = web_client(s)
    attacker = {"REMOTE_ADDR": "10.0.0.66"}
    for _ in range(5):
        c.post("/api/setup/unlock", json={"pin": "1"}, environ_base=attacker)
    assert c.post("/api/setup/unlock", json={"pin": "4711"}, environ_base=attacker).status_code == 429
    assert c.post("/api/setup/unlock", json={"pin": "4711"}, environ_base={"REMOTE_ADDR": "10.0.0.5"}).json["ok"]


def test_csrf_and_security_headers(make_system):
    from qc.web import create_app

    s = make_system()
    bare = create_app(s).test_client()  # like a form on a foreign web page: no custom header
    r = bare.post("/api/inspect/stop")
    assert r.status_code == 403 and "X-QC-Client" in r.json["error"]
    assert bare.get("/api/status").status_code == 200  # reading needs no header
    resp = web_client(s).get("/")
    assert resp.headers["X-Frame-Options"] == "DENY" and "default-src 'self'" in resp.headers["Content-Security-Policy"]
    assert web_client(s).get("/api/status").headers["Cache-Control"] == "no-store"


# ------------------------------------------------------------------ config + compatibility
def test_config_validation_reports_typos_and_bad_values():
    for f in ("config.example.json", "config.pi3_imx219.json"):
        AppConfig.load(f)  # shipped files are valid
    with pytest.raises(ConfigError, match="exposure"):
        AppConfig.from_dict({"camera": {"exposure": 3000}})  # typo of exposure_us
    with pytest.raises(ConfigError, match=r"trigger\.axis"):
        AppConfig.from_dict({"trigger": {"axis": "z"}})
    with pytest.raises(ConfigError, match="shots_per_part"):
        AppConfig.from_dict({"inspection": {"shots_per_part": 1}})


def test_config_of_the_full_version_still_loads(make_system):
    """config.json from the full version: removed sections are ignored and reported, not fatal."""
    cfg = AppConfig.from_dict(
        {"self_learning": {"max_references": 40}, "io": {"enabled": False}, "ui": {"preview_fps": 5}}
    )
    assert cfg.ignored == ["io", "self_learning"] and cfg.ui.preview_fps == 5
    assert "ignored" not in cfg.to_dict()


def test_model_of_the_full_version_still_loads(tmp_path):
    """Single-light model saved by the full version (recipe.json + channels/main) → loads, and the
    first change converts the folder to the current layout."""
    rng = np.random.default_rng(6)
    m = QCModel.train("Old", shots("A", "front", 6, rng), LocalizationConfig(), MethodConfig(methods=["geometry"]))
    m.save(tmp_path / "x")
    old = tmp_path / "models" / "Old"
    (old / "channels").mkdir(parents=True)
    (tmp_path / "x" / "Old").rename(old / "channels" / "main")
    (old / "recipe.json").write_text(
        json.dumps(
            {
                "name": "Old",
                "channels": ["main"],
                "format": 3,
                "sensitivity": {"geometry": 1.3},
                "camera_profile": {"zoom": 2.0},
            }
        )
    )
    loaded = QCModel.load(old)
    assert loaded.sensitivity["geometry"] == 1.3 and loaded.camera_profile == {"zoom": 2.0}
    assert len(loaded.reference_frames()) == 6
    loaded.set_sensitivity({"diff": 0.8})
    assert (old / "meta.json").exists() and not (old / "recipe.json").exists()
    again = QCModel.load(old)
    assert again.sensitivity == {"geometry": 1.3, "diff": 0.8, "pca": 1.0} and len(again.reference_frames()) == 6
    img = shots("A", "front", 1, rng, "missing")[0]
    assert again.inspect(img, False).status == "NOK"


# ------------------------------------------------------------------ camera sources
class FakeRequest:
    """Stand-ins for picamera2 – the Pi camera code is normally only runnable on a Pi."""

    def make_array(self, name):
        return np.zeros((480, 640, 3), np.uint8)

    def get_metadata(self):
        return {"SensorTimestamp": time.monotonic_ns() - 20_000_000, "ColourGains": (1.5, 1.7)}

    def release(self):
        pass


class FakePicamera2:
    camera_controls: ClassVar[dict] = {
        "ScalerCrop": ((0, 0, 64, 64), (0, 0, 3280, 2464), (0, 0, 3280, 2464)),
        "ExposureTime": (1, 1, 1),
    }
    camera_properties: ClassVar[dict] = {"Model": "imx219", "PixelArraySize": (3280, 2464)}

    def __init__(self):
        self.controls, self.video_kwargs = {}, None

    def create_video_configuration(self, **kw):
        self.video_kwargs = kw
        return kw

    def configure(self, c):
        pass

    def start(self):
        pass

    def stop(self):
        pass

    def close(self):
        pass

    def set_controls(self, d):
        self.controls.update(d)

    def capture_request(self):
        return FakeRequest()

    def capture_metadata(self):
        return {"ScalerCrop": (0, 0, 3280, 2464), "LensPosition": self.controls.get("LensPosition", 2.0)}

    def autofocus_cycle(self):
        self.controls["AfCycle"] = True

    def camera_configuration(self):
        return {"sensor": {"output_size": (1640, 1232)}}


class FakeAutofocusPicamera2(FakePicamera2):
    """Camera Module 3: with focus motor."""

    camera_controls: ClassVar[dict] = {**FakePicamera2.camera_controls, "AfMode": (0, 2, 0)}


def test_pi_camera_source_against_fake_picamera2(monkeypatch):
    """The Pi camera code (normally only runnable on a Pi) against a stand-in for picamera2."""
    import sys
    import types

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
    assert frame.shape == (480, 640, 3) and 0.01 < cam.last_age_s < 0.5  # age from the sensor timestamp
    assert cam.awb_result() == (1.5, 1.7)
    cam.set_crop((0.25, 0.25, 0.5, 0.5))
    assert cam.cam.controls["ScalerCrop"] == (820, 616, 1640, 1232) and cam.crop == (0.25, 0.25, 0.5, 0.5)
    cam.set_exposure(800, 2.0)
    assert cam.cam.controls["ExposureTime"] == 800 and cam.cam.controls["AnalogueGain"] == 2.0


def test_pi_camera_with_focus_motor(monkeypatch):
    """Camera Module 3: one autofocus run at start (or a fixed lens position), autofocus on the part."""
    import sys
    import types

    monkeypatch.setitem(sys.modules, "picamera2", types.SimpleNamespace(Picamera2=FakeAutofocusPicamera2))
    monkeypatch.setattr(time, "sleep", lambda s: None)
    from qc.camera import PiCameraSource
    from qc.config import CameraConfig

    cam = PiCameraSource(CameraConfig(source="pi", exposure_us=0))
    assert cam.has_autofocus and cam.cam.controls["AfCycle"] and "focus fixed at 2.00 dioptres (≈ 50 cm)" in cam.info
    assert "AeEnable" not in cam.cam.controls  # exposure_us = 0 → automatic exposure
    cam = PiCameraSource(CameraConfig(source="pi", lens_position=4.0))
    assert cam.cam.controls["AfMode"] == 0 and "4.00 dioptres" in cam.info  # fixed lens position
    cam.set_lens(5.0)
    assert cam.cam.controls["LensPosition"] == 5.0
    cam.set_colour_gains((1.2, 1.4))
    assert cam.cam.controls["ColourGains"] == (1.2, 1.4) and cam.cam.controls["AwbEnable"] is False
    cam.awb_start()
    assert cam.cam.controls["AwbEnable"] is True
    cam.set_crop((0.5, 0.5, 0.5, 0.5))
    cam.af_start((0.0, 0.0, 0.5, 0.5))  # window on a quarter of the zoomed image
    assert cam.cam.controls["AfWindows"] == [(1640, 1232, 820, 616)] and cam.cam.controls["AfTrigger"] == 0
    assert cam.af_state() == ("scanning", None)
    cam.last_metadata = {"AfState": 2, "LensPosition": 3.5}
    assert cam.af_state() == ("focused", 3.5)
    cam.last_metadata = {"AfState": 3}
    assert cam.af_state() == ("failed", None)
    cam.close()


def test_camera_sources_from_the_config(tmp_path):
    from qc.camera import FolderSource, SimulatorSource, open_camera
    from qc.config import CameraConfig

    assert isinstance(open_camera(CameraConfig(source="sim")), SimulatorSource)
    with pytest.raises(RuntimeError, match="No images"):
        open_camera(CameraConfig(source=f"folder:{tmp_path}"))
    for i in range(2):
        cv2.imwrite(str(tmp_path / f"{i}.png"), np.full((48, 64, 3), i, np.uint8))
    src = open_camera(CameraConfig(source=f"folder:{tmp_path}"))
    assert isinstance(src, FolderSource) and not src.continuous and src.current_file.name == "0.png"
    assert src.next_part()[0, 0, 0] == 0 and src.next_part()[0, 0, 0] == 1 and src.current_file.name == "1.png"
    assert src.next_part()[0, 0, 0] == 0  # loops by default
    assert FolderSource(str(tmp_path), loop=False).next_part() is not None
    with pytest.raises(ValueError, match="Unknown camera source"):
        open_camera(CameraConfig(source="ftp:camera"))
    with pytest.raises(RuntimeError, match="Could not open camera"):
        open_camera(CameraConfig(source="usb:/dev/does-not-exist"))
