"""Process control: camera loop, conveyor trigger, learning, inspection, self-test.

The ``QCSystem`` class is independent of the user interface; the web UI
(``webapp.py``) only calls its methods. This makes it easy to add a local UI
(e.g. a touch display) or a PLC connection later on.

Per part the loop does::

    trigger fires ─┬─ single light: 1 … N shots of the moving part (majority vote)
                   └─ dual light:   image with idle light + image with the other light
                → recipe.inspect() → decision → UI / log / lamps / reject / drift monitor
"""
from __future__ import annotations

import csv
import itertools
import json
import threading
import time
from collections import deque
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from . import archive as qa
from . import autosetup as AS
from . import calibration as qcal
from . import model as qm
from .alignment import _levels, detect_part, resize_to_width, to_gray
from .camera import CameraSource, FolderSource, SimulatorSource
from .config import AppConfig, LocalizationConfig, TriggerConfig
from .drift import DriftMonitor
from .io_control import IOController
from .recipe import Recipe, list_recipes


class PartTrigger:
    """Fires exactly one capture per part – when it crosses the image centre
    or rests inside the centre band (conveyor stopped)."""

    def __init__(self, cfg: TriggerConfig, loc: LocalizationConfig):
        self.cfg = cfg
        self.loc = replace(loc, border_margin_px=2)
        self.armed = True
        self.prev_pos: float | None = None
        self.misses = 0
        self.last_det = None
        self.scale = 1.0

    def update(self, frame: np.ndarray) -> bool:
        small = resize_to_width(frame, self.cfg.detect_width)
        self.scale = frame.shape[1] / small.shape[1]
        det = detect_part(to_gray(small), self.loc)
        self.last_det = det
        axis = 0 if self.cfg.axis == "x" else 1
        size = small.shape[1 - axis]
        pos = det.centroid[axis] / size - 0.5 if det is not None else None
        in_band = pos is not None and abs(pos) < self.cfg.center_band_ratio

        fire = False
        if in_band and det.complete:
            self.misses = 0
            if self.armed and self.prev_pos is not None:
                crossed = np.sign(pos) != np.sign(self.prev_pos) or abs(pos) < 0.01
                still = abs(pos - self.prev_pos) < 0.002
                if crossed or still:
                    fire = True
                    self.armed = False
        else:
            self.misses += 1
            if self.misses >= self.cfg.rearm_frames:
                self.armed = True
        self.prev_pos = pos
        return fire

    @property
    def part_complete(self) -> bool:
        return self.last_det is not None and self.last_det.complete

    def draw(self, img: np.ndarray) -> None:
        h, w = img.shape[:2]
        b = self.cfg.center_band_ratio
        color = (0, 200, 255) if self.armed else (120, 120, 120)
        if self.cfg.axis == "x":
            for x in (int(w * (0.5 - b)), int(w * (0.5 + b))):
                cv2.line(img, (x, 0), (x, h), color, 1, cv2.LINE_AA)
        else:
            for y in (int(h * (0.5 - b)), int(h * (0.5 + b))):
                cv2.line(img, (0, y), (w, y), color, 1, cv2.LINE_AA)
        if self.last_det is not None:
            s = img.shape[1] / (self.cfg.detect_width)
            c = (self.last_det.contour.astype(np.float32) * s).astype(np.int32)
            cv2.drawContours(img, [c], -1, (255, 200, 0) if self.last_det.complete else (100, 100, 100), 1, cv2.LINE_AA)


def _jpg(img: np.ndarray, q: int = 85) -> bytes:
    return cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])[1].tobytes()


class QCSystem:
    PREVIEW_WIDTH = 640
    FRAMES_KEPT = 30          # results whose images are kept in memory for feedback
    MODES = ("idle", "camera_setup", "learning", "training", "inspecting", "selftest")

    def __init__(self, cfg: AppConfig, camera: CameraSource):
        self.cfg = cfg
        self.camera = camera
        self.trigger = PartTrigger(cfg.trigger, cfg.localization)
        self.lock = threading.RLock()
        self.mode = "idle"
        self.model: Recipe | None = None
        self.learn_name = ""
        self.learn_captures: list[dict[str, np.ndarray]] = []
        self.learn_thumbs: list[bytes] = []
        self.history: deque[dict] = deque(maxlen=cfg.storage.history_size)
        self.images: dict[str, list[tuple[str, bytes, bytes]]] = {}     # rid → [(channel, overlay, detail)]
        self.frames: dict[str, dict[str, np.ndarray]] = {}               # rid → captures (for feedback)
        self.counts = {"OK": 0, "NOK": 0, "NO_PART": 0}
        self.feedback_counts = {"false_alarm": 0, "missed": 0, "confirmed": 0}
        self.messages: deque[dict] = deque(maxlen=30)
        self.last_result: dict | None = None
        self.preview_jpeg: bytes | None = None
        self.last_frame: np.ndarray | None = None       # as delivered by the camera (trigger, preview, calibration)
        self.last_raw: np.ndarray | None = None
        self.focus_assist: dict | None = None            # {"until", "best", "value"} while the assistant is on
        # automatic camera setup (qc/autosetup.py)
        self._frame_seq = 0
        self._frame_cv = threading.Condition()
        self.cam_setup: dict | None = None               # status of the running/last camera setup
        self.pending_profile: dict | None = None         # result of the static step, waiting for teach-in
        self.learn_profile: dict | None = None           # camera profile of the part being taught in
        self.learn_path: dict | None = None              # first moving part: belt direction + zoom
        self.learn_edges: list[float] = []               # sharpness (edge width) of the references
        self._applied: dict | None = None                # camera settings currently applied
        self._base_axis = cfg.trigger.axis
        self.fps = 0.0
        self._ids = itertools.count(1)
        self._stop = threading.Event()
        self._manual = threading.Event()
        self._manual_fired = False
        self._thread: threading.Thread | None = None
        self._last_folder_step = 0.0
        self._burst: dict | None = None
        self._last_captures: dict | None = None
        self.folder_interval = 1.0
        self.selftest: dict | None = None
        self.last_selftest: dict | None = None
        self.last_self_learning: dict | None = None
        self.calib_session: dict | None = None
        self._bg_learning: str | None = None     # slug of the model being self-learned in the background
        self._bg_tainted = False                 # a collected part was reported as defective meanwhile
        self.learn_auto_finish = False
        Path(cfg.storage.models_dir).mkdir(parents=True, exist_ok=True)
        Path(cfg.storage.results_dir).mkdir(parents=True, exist_ok=True)

        self.drift = DriftMonitor(cfg.drift)
        self.io = IOController(cfg.io, light_hook=self._sim_light_hook)
        self.calibration = qcal.Calibration.load(cfg.calibration.file)
        self.undistort = qcal.Undistorter(self.calibration) if self.calibration else None
        self._selftest_file = Path(cfg.storage.models_dir).parent / "selftest_last.json"
        if self._selftest_file.exists():
            try:
                self.last_selftest = json.loads(self._selftest_file.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                pass
        if self.is_folder and cfg.lighting.mode == "dual":
            cfg.lighting.mode = "single"
            self.log("warn", "Dual-light mode needs a live camera – image folder runs in single-light mode.")
        self._apply_idle_light()
        if getattr(camera, "info", ""):
            self.log("info", f"Camera: {camera.info}")
        self.log("info", f"System started – camera: {camera.name}"
                         + (f", calibrated ({self.mm_per_px:.4f} mm/px)" if self.mm_per_px else ""))

    # --------------------------------------------------------------- helpers
    def log(self, level: str, text: str) -> None:
        self.messages.appendleft({"t": datetime.now().strftime("%H:%M:%S"), "level": level, "text": text})

    @property
    def is_sim(self) -> bool:
        return isinstance(self.camera, SimulatorSource)

    @property
    def is_folder(self) -> bool:
        return isinstance(self.camera, FolderSource)

    @property
    def dual(self) -> bool:
        return self.cfg.lighting.mode == "dual"

    @property
    def mm_per_px(self) -> float | None:
        # the calibration was made at the full field of view; with the digital zoom a pixel covers
        # only the crop's fraction of it
        if not self.calibration:
            return None
        return self.calibration.mm_per_px(self.cfg.localization.work_width) * self.camera.crop[2]

    @property
    def zoomed(self) -> bool:
        return self.camera.crop[2] < 0.999

    def _sim_light_hook(self, light: str) -> None:
        if self.is_sim and self.dual:
            self.camera.switch_light(light)

    def _apply_idle_light(self) -> None:
        if self.dual:
            self.io.set_light(self.cfg.lighting.idle_light)
        else:
            self.io.set_light(self.cfg.lighting.idle_light)
            if self.is_sim:
                self.camera.light = self.camera.cfg.sim_lighting

    def _channels(self) -> list[str]:
        return ["back", "front"] if self.dual else ["main"]

    def _read(self) -> np.ndarray | None:
        """Raw camera frame. Lens distortion is only corrected for the frames that are actually
        learned/inspected (``_undistort_shots``) – a full-frame remap at camera rate is too
        expensive on a small Pi, and the trigger only needs the rough part position."""
        raw = self.camera.read()
        if raw is not None:
            self.last_raw = raw
        return raw

    def _undistort_shots(self, shots: list[dict]) -> list[dict]:
        if not self.undistort or self.zoomed:      # maps are for the full field of view; a crop of the
            return shots                           # image centre has little distortion anyway
        return [{ch: (self.undistort(f) if f is not None else None) for ch, f in c.items()} for c in shots]

    # ------------------------------------------------------------------ loop
    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, name="qc-loop", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3)
        self.camera.close()
        self.io.close()

    def _loop(self) -> None:
        t_last, n = time.time(), 0
        last_preview = 0.0
        while not self._stop.is_set():
            frame = self._read()
            if frame is None:
                time.sleep(0.05)
                continue
            t_capture = time.time() - self.camera.last_age_s      # when the sensor saw it, not when we got it
            self.last_frame = frame
            with self._frame_cv:
                self._frame_seq += 1
                self._frame_cv.notify_all()
            active = self.mode in ("learning", "inspecting", "selftest")
            fire = False
            if self.camera.continuous:
                fire = self.trigger.update(frame) and active
                if self.learn_path is not None and self.mode == "learning":
                    self._track_path(frame, t_capture)
                if self._burst is not None:
                    self._continue_burst(frame)
            elif self.mode in ("inspecting", "selftest") and time.time() - self._last_folder_step > self.folder_interval:
                # Image folder: in inspection mode automatically one image per interval
                frame = self.camera.next_part()
                self._last_folder_step = time.time()
                fire = frame is not None
            self._manual_fired = self._manual.is_set()
            if self._manual.is_set():
                self._manual.clear()
                if self.is_folder and self.mode == "learning":
                    frame = self.camera.next_part()
                fire = frame is not None and active
            if fire and self._burst is None:
                self._on_part(frame, t_capture)

            n += 1
            if time.time() - t_last >= 1.0:
                self.fps = n / (time.time() - t_last)
                t_last, n = time.time(), 0
            if time.time() - last_preview > 1.0 / max(0.5, self.cfg.ui.preview_fps):
                self._update_preview(frame)
                last_preview = time.time()

    def _on_part(self, frame: np.ndarray, t_capture: float) -> None:
        """A part was triggered: collect the images of this part."""
        if self.dual and self.camera.continuous:
            captures = self._capture_dual(frame)
            self._handle_part([captures], t_capture)
            return
        shots = self.shots_per_part
        if self.mode in ("inspecting", "selftest") and self.camera.continuous:
            self._burst = {"shots": [{"main": frame}], "n": shots, "t0": t_capture}
            return
        self._handle_part([{"main": frame}], t_capture)

    def _continue_burst(self, frame: np.ndarray) -> None:
        b = self._burst
        if self.trigger.part_complete and len(b["shots"]) < b["n"]:
            b["shots"].append({"main": frame})
        if len(b["shots"]) >= b["n"] or not self.trigger.part_complete:
            self._burst = None
            self._handle_part(b["shots"], b["t0"])

    def _capture_dual(self, first: np.ndarray) -> dict[str, np.ndarray]:
        """Image with the idle light is already there; switch to the other light for a second image."""
        idle = self.cfg.lighting.idle_light
        other = "back" if idle == "front" else "front"
        self.io.set_light(other)
        for _ in range(max(0, int(self.cfg.lighting.settle_frames))):
            self._read()
        second = self._read()
        self.io.set_light(idle)
        return {idle: first, other: second}

    def _update_preview(self, frame: np.ndarray) -> None:
        img = resize_to_width(frame, self.PREVIEW_WIDTH).copy()
        fa = self.focus_assist
        if fa is not None:
            if time.time() > fa["until"]:
                self.focus_assist = None
            else:
                self._draw_focus(img, frame, fa)
        elif self.camera.continuous:
            self.trigger.draw(img)
        label = {"idle": "READY", "camera_setup": "ADJUSTING CAMERA", "learning": f"LEARNING {len(self.learn_captures)}/{self.cfg.target_reference_count}",
                 "training": "TRAINING ...", "inspecting": "INSPECTION ACTIVE",
                 "selftest": "SELF-TEST: " + (self.selftest or {}).get("step", "").upper() + " PART"}[self.mode]
        cv2.rectangle(img, (0, img.shape[0] - 26), (img.shape[1], img.shape[0]), (0, 0, 0), cv2.FILLED)
        cv2.putText(img, f"{label}   {self.fps:4.1f} fps", (8, img.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(self.cfg.ui.preview_quality)])
        if ok:
            self.preview_jpeg = buf.tobytes()

    # ------------------------------------------------------- focus assistant
    FOCUS_MINUTES = 10

    def set_focus_assist(self, on: bool) -> None:
        """Live sharpness value in the camera image – for lenses that are focused by hand
        (Camera Module v2/IMX219, HQ camera, Global Shutter camera)."""
        self.focus_assist = {"until": time.time() + self.FOCUS_MINUTES * 60, "best": 0.0, "value": 0.0} if on else None
        self.log("info", "Focus assistant " + ("on – turn the lens slowly until the value is at its maximum."
                                               if on else "off."))

    @staticmethod
    def sharpness(gray: np.ndarray) -> float:
        """Focus measure: variance of the Laplacian (higher = sharper). Robust against noise
        through a light blur; only comparable for the same scene and lighting."""
        g = cv2.GaussianBlur(gray, (3, 3), 0)
        return float(cv2.Laplacian(g, cv2.CV_32F).var())

    def _draw_focus(self, img: np.ndarray, frame: np.ndarray, fa: dict) -> None:
        h, w = frame.shape[:2]
        # measure in full resolution on the central 50 % (or on the part, if one is visible)
        det = self.trigger.last_det
        on_part = det is not None and det.complete
        if on_part:
            x, y, bw, bh = cv2.boundingRect((det.contour.astype(np.float32) * self.trigger.scale).astype(np.int32))
            x0, y0, x1, y1 = max(0, x), max(0, y), min(w, x + bw), min(h, y + bh)
        else:
            x0, y0, x1, y1 = w // 4, h // 4, 3 * w // 4, 3 * h // 4
        roi = frame[y0:y1, x0:x1]
        if roi.size == 0:
            return
        v = self.sharpness(to_gray(roi))
        fa["value"] = 0.7 * fa["value"] + 0.3 * v if fa["value"] else v       # smooth sensor noise
        fa["best"] = max(fa["best"], fa["value"])
        s = img.shape[1] / w
        cv2.rectangle(img, (int(x0 * s), int(y0 * s)), (int(x1 * s), int(y1 * s)), (0, 200, 255), 1)
        rel = fa["value"] / fa["best"] if fa["best"] else 0.0
        col = (60, 200, 60) if rel > 0.95 else (0, 200, 255) if rel > 0.8 else (60, 60, 230)
        cv2.rectangle(img, (0, 0), (img.shape[1], 44), (0, 0, 0), cv2.FILLED)
        cv2.rectangle(img, (8, 30), (8 + int((img.shape[1] - 16) * min(1.0, rel)), 40), col, cv2.FILLED)
        cv2.putText(img, f"FOCUS {fa['value']:.0f}   best {fa['best']:.0f}   ({rel * 100:.0f} %)", (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 1, cv2.LINE_AA)
        if not on_part:
            cv2.putText(img, "no part in view - lay a part flat under the camera", (8, img.shape[0] - 36),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1, cv2.LINE_AA)

    def _handle_part(self, shots: list[dict[str, np.ndarray]], t_capture: float) -> None:
        shots = self._undistort_shots(shots)
        with self.lock:
            if self.mode == "learning" and self.learn_path is not None:
                if not self.camera.continuous or self._manual_fired:
                    # stopped belt / image folder: no belt direction to measure → zoom from the still part
                    self._finish_path(None)
                    self.log("info", "Camera zoomed in on the part – capture the part again.")
                return                                   # the path part is never a reference
            if self.mode == "learning":
                self._add_reference(shots[0])
            elif self.mode in ("inspecting", "selftest") and self.model is not None:
                self._inspect(shots, t_capture)

    # --------------------------------------------------------- learning mode
    def start_learning(self, name: str, auto_finish: bool = False) -> None:
        """``auto_finish``: create the model automatically once the target number of parts is reached."""
        name = name.strip()
        if not name:
            raise ValueError("Please enter a name for the part type.")
        with self.lock:
            if self.mode != "idle":
                raise ValueError("Stop the current mode first.")
            self.mode = "learning"
            self.learn_name = name
            self.learn_auto_finish = bool(auto_finish)
            self.learn_captures, self.learn_thumbs, self.learn_edges = [], [], []
            if self.pending_profile is not None:
                # automatic camera setup done: the first moving part sets belt direction and zoom
                self.learn_profile, self.pending_profile = self.pending_profile, None
                self.learn_path = {"track": [], "seen": False}
                if not self.camera.continuous:
                    self._finish_path(None)
            else:
                self.learn_profile, self.learn_path = None, None
                self._apply_camera(None)                # never teach in with another part's zoom
            if self.is_sim:
                self.camera.force_good = True    # only good parts on the conveyor while learning
            self.log("info", f"Learning mode started for “{name}”"
                             + (" (dual light)" if self.dual else "")
                             + f" – please feed {self.cfg.target_reference_count} good parts.")

    def _add_reference(self, captures: dict[str, np.ndarray]) -> None:
        for ch, frame in captures.items():
            if frame is None:
                self.log("warn", "Reference discarded: second image missing.")
                return
            work = resize_to_width(frame, self.cfg.localization.work_width)
            det = detect_part(to_gray(work), self.cfg.localization)
            if det is None or not det.complete:
                self.log("warn", "Reference image discarded: no complete part detected"
                                 + (f" ({ch} light)." if ch != "main" else "."))
                return
        self.learn_captures.append({ch: f.copy() for ch, f in captures.items()})
        thumb_src = captures.get("front", next(iter(captures.values())))
        ew = AS.edge_width(to_gray(work), det.contour)        # sharpness of this reference (last channel)
        if ew is not None:
            self.learn_edges.append(ew)
        self.learn_thumbs.append(_jpg(resize_to_width(thumb_src, 200), 70))
        self.log("info", f"Reference image {len(self.learn_captures)} captured.")
        if self.learn_auto_finish and len(self.learn_captures) >= self.cfg.target_reference_count:
            self.finish_learning()

    def undo_reference(self, index: int | None = None) -> None:
        with self.lock:
            if not self.learn_captures:
                return
            i = len(self.learn_captures) - 1 if index is None else index
            self.learn_captures.pop(i)
            self.learn_thumbs.pop(i)
            self.log("info", f"Reference image {i + 1} removed.")

    def cancel_learning(self) -> None:
        with self.lock:
            self.mode = "idle"
            self.learn_captures, self.learn_thumbs, self.learn_edges = [], [], []
            self.learn_profile = self.learn_path = None
            self._apply_camera(self.model.camera_profile if self.model else None)
            if self.is_sim:
                self.camera.force_good = False
            self.log("info", "Learning mode cancelled.")

    def finish_learning(self, method_overrides: dict | None = None) -> None:
        with self.lock:
            if self.mode != "learning":
                raise ValueError("Learning mode is not active.")
            if len(self.learn_captures) < qm.MIN_REFERENCES:
                raise ValueError(f"At least {qm.MIN_REFERENCES} reference images required "
                                 f"(currently {len(self.learn_captures)}).")
            captures, name = list(self.learn_captures), self.learn_name
            mcfgs = self._method_cfgs(method_overrides)
            self.mode = "training"
            if self.is_sim:
                self.camera.force_good = False
        threading.Thread(target=self._train, args=(name, captures, mcfgs), daemon=True).start()

    def _method_cfg(self, overrides: dict | None):
        mcfg = replace(self.cfg.method)
        for k, v in (overrides or {}).items():
            if hasattr(mcfg, k) and v not in (None, ""):
                setattr(mcfg, k, v)
        if not mcfg.methods:
            raise ValueError("Select at least one inspection method.")
        return mcfg

    def _method_cfgs(self, overrides: dict | None) -> dict:
        main = self._method_cfg(overrides)
        if not self.dual:
            return {"main": main}
        lc = self.cfg.lighting
        front = replace(main, methods=list(lc.front_methods)) if lc.front_methods else main
        return {"back": replace(main, methods=list(lc.back_methods)), "front": front}

    def _train(self, name, captures, mcfgs) -> None:
        try:
            cal = self.calibration
            recipe = Recipe.train(name, captures, self.cfg.localization, mcfgs,
                                  self.mm_per_px, cal.id if cal else None)
            prof = dict(self.learn_profile or {"auto": False, "crop": list(self.camera.crop)})
            if self.learn_edges:
                prof["edge_width_px"] = round(float(np.median(self.learn_edges)), 2)
                prof["sharp"] = prof["edge_width_px"] <= self.cfg.autosetup.edge_width_max_px
            recipe.camera_profile = prof
            recipe.save(self.cfg.storage.models_dir)
            with self.lock:
                self._set_model(recipe)
                self.learn_captures, self.learn_thumbs = [], []
                self.mode = "idle"
            rep = recipe.channels[recipe.primary].report
            msg = f"Model “{name}” created ({rep['references_used']} references, {rep['train_time_s']} s)."
            if rep["references_skipped"]:
                msg += f" {len(rep['references_skipped'])} image(s) without a complete part skipped."
            self.log("ok", msg)
            for w in recipe.warnings():
                self.log("warn", w)
            if prof.get("sharp") is False:
                self.log("warn", f"The image is not sharp (edges {prof['edge_width_px']} px wide, should be ≤ "
                                 f"{self.cfg.autosetup.edge_width_max_px}). Focus the lens once (focus assistant), "
                                 "then teach in again – detection is less precise this way.")
            self.learn_profile = None
        except Exception as e:  # noqa: BLE001 - show the error to the supervisor
            with self.lock:
                self.mode = "learning"
            self.log("error", f"Training failed: {e}")

    # ------------------------------------------------------- model handling
    def _set_model(self, recipe: Recipe | None) -> None:
        self.model = recipe
        if recipe is not None:
            self.drift.reset({ch: m.baseline for ch, m in recipe.channels.items()})
            self._apply_camera(recipe.camera_profile)

    # ------------------------------------------------- camera settings / auto setup
    def _apply_camera(self, profile: dict | None) -> None:
        """Applies a part type's camera profile (or the base settings from config.json)."""
        cam, base, p = self.camera, self.cfg.camera, profile or {}
        want = {
            "exposure": (p["exposure_us"], p.get("gain", 1.0)) if "exposure_us" in p else
                        ((base.exposure_us, base.analogue_gain) if base.exposure_us > 0 else None),
            "gains": tuple(p["colour_gains"]) if p.get("colour_gains") else
                     (tuple(base.colour_gains) if base.lock_white_balance else None),
            "lens": p.get("lens_position", base.lens_position),
            "crop": tuple(p.get("crop") or AS.FULL),
            "axis": p.get("axis") or self._base_axis,
        }
        if want == self._applied:
            return
        if cam.can_control:
            try:
                if want["exposure"]:
                    cam.set_exposure(*want["exposure"])
                if want["gains"]:
                    cam.set_colour_gains(want["gains"])
                if want["lens"] is not None:
                    cam.set_lens(want["lens"])
                cam.set_crop(want["crop"])
            except Exception as e:  # noqa: BLE001 - a camera problem must not stop the system
                self.log("error", f"Camera settings could not be applied: {e}")
        self.cfg.trigger.axis = want["axis"]
        self.trigger.armed, self.trigger.prev_pos = True, None
        self._applied = want

    def _wait_frames(self, n: int, timeout: float = 5.0) -> None:
        with self._frame_cv:
            target = self._frame_seq + n
            self._frame_cv.wait_for(lambda: self._frame_seq >= target, timeout)

    def _measure_part(self):
        """Latest frame → (frame, work image, detection or None)."""
        frame = self.last_raw
        if frame is None:
            raise ValueError("No camera image.")
        work = resize_to_width(frame, self.cfg.localization.work_width)
        det = detect_part(to_gray(work), self.cfg.localization)
        return frame, work, (det if det is not None and det.complete else None)

    def start_camera_setup(self) -> None:
        """Step 1 of the automatic camera setup: ONE good part lies still under the camera."""
        with self.lock:
            if self.mode != "idle":
                raise ValueError("Stop the current mode first.")
            if not self.camera.can_control:
                raise ValueError("This camera source cannot be adjusted automatically (Pi camera or simulator only).")
            self.mode = "camera_setup"
            self.pending_profile = None
            self._applied = None                 # camera state changes now – re-apply everything afterwards
            self.cam_setup = {"running": True, "step": "starting", "error": None, "result": None,
                              "started": datetime.now().isoformat(timespec="seconds")}
        threading.Thread(target=self._camera_setup_run, name="qc-camera-setup", daemon=True).start()

    def cancel_camera_setup(self) -> None:
        with self.lock:
            self.pending_profile = None
            self.cam_setup = None
            if self.mode == "camera_setup":
                self.mode = "idle"
            self._applied = None
            self._apply_camera(self.model.camera_profile if self.model else None)

    def _camera_setup_run(self) -> None:
        cam, ac, st = self.camera, self.cfg.autosetup, self.cam_setup
        if self.is_sim:
            cam.hold_part(True)
        try:
            st["step"] = "white balance"
            cam.set_crop(AS.FULL)
            e, g = float(self.cfg.camera.exposure_us or 4000), 1.0
            cam.set_exposure(e, g)
            cam.awb_start()
            self._wait_frames(15)
            gains = cam.awb_result()
            if gains:
                cam.set_colour_gains(gains)

            st["step"] = "exposure"
            det = None
            for _ in range(12):
                self._wait_frames(4)
                levels = []
                for light in (self._lights() or [None]):
                    if light:
                        self.io.set_light(light)
                        self._wait_frames(max(2, self.cfg.lighting.settle_frames + 1))
                    frame, work, det = self._measure_part()
                    levels.append(AS.bright_level(frame, det.contour if det else None, frame.shape[1] / work.shape[1]))
                if self.dual:
                    self.io.set_light(self.cfg.lighting.idle_light)
                e, g, done = AS.next_exposure(e, g, max(levels), ac)
                cam.set_exposure(e, g)
                if done and det is not None:
                    break
            if det is None:
                raise ValueError("No complete part found. Lay ONE good part flat under the camera, in the middle "
                                 "of the image, and try again.")

            lens = None
            if cam.has_autofocus:
                st["step"] = "focus"
                x, y, w, h = cv2.boundingRect(det.contour.astype(np.int32))
                W, H = work.shape[1], work.shape[0]
                cam.af_start((x / W, y / H, w / W, h / H))
                for _ in range(120):
                    self._wait_frames(1)
                    state, pos = cam.af_state()
                    if state != "scanning":
                        break
                if state == "focused" and pos:
                    lens = round(pos, 3)
                    cam.set_lens(lens)
                else:
                    self.log("warn", "Autofocus did not find a sharp image – check that the part is in the middle.")

            st["step"] = "measuring"
            self._wait_frames(4)
            frame, work, det = self._measure_part()
            if det is None:
                raise ValueError("The part is no longer visible – keep it still under the camera.")
            gray = to_gray(work)
            part_level, bg_level = _levels(gray, det)
            edge = AS.edge_width(gray, det.contour)
            W, H = work.shape[1], work.shape[0]
            cx, cy, D = AS.part_extent(det.contour)
            center = AS.to_sensor(cx, cy, (W, H), AS.FULL)
            diameter = (D / W, D / H)
            prof = {"auto": True, "created": datetime.now().isoformat(timespec="seconds"),
                    "exposure_us": int(e), "gain": float(g), "colour_gains": list(gains) if gains else None,
                    "lens_position": lens, "crop": list(AS.FULL), "zoom": 1.0, "axis": self._base_axis,
                    "center": [round(center[0], 4), round(center[1], 4)],
                    "diameter": [round(diameter[0], 4), round(diameter[1], 4)],
                    "contrast": round(abs(part_level - bg_level), 1),
                    "edge_width_full_px": round(edge, 2) if edge else None}
            notes = []
            if prof["contrast"] < 40:
                notes.append(f"Low contrast between part and belt ({prof['contrast']:.0f} grey levels) – "
                             "a darker/brighter belt or backlight would make detection more reliable.")
            if g >= ac.max_gain - 1e-3:
                notes.append("Little light: the image needs the maximum gain (more noise). Brighter lighting helps.")
            if min(diameter) * 1.35 > 1.0:
                notes.append("The part almost fills the image – parts may be cut off at the edge. Mount the camera "
                             "a little higher (no zoom is possible either way).")
            if not cam.has_autofocus and edge and edge > ac.edge_width_max_px:
                notes.append(f"The image is blurry (edges {edge:.1f} px). This camera has no focus motor: "
                             "turn the lens once by hand (focus assistant), then run the setup again.")
            prof["notes"] = notes
            with self.lock:
                self.pending_profile = prof
                st.update(running=False, step="done", result={
                    "exposure_ms": round(e / 1000, 2), "gain": g, "white_balance": bool(gains),
                    "autofocus": cam.has_autofocus, "lens_position": lens, "contrast": prof["contrast"],
                    "edge_width_px": prof["edge_width_full_px"], "notes": notes})
            self.log("ok", f"Camera adjusted: exposure {e / 1000:.2f} ms, gain {g:.2f}"
                     + (f", focus {lens:.2f} dpt" if lens else "") + ". Next: let the parts run – the first one "
                     "measures the belt direction and sets the zoom.")
            for n in notes:
                self.log("warn", n)
        except Exception as ex:  # noqa: BLE001
            st.update(running=False, step="failed", error=str(ex))
            self.log("error", f"Automatic camera setup failed: {ex}")
            self._applied = None
            self._apply_camera(self.model.camera_profile if self.model else None)
        finally:
            if self.is_sim:
                cam.hold_part(False)
            with self.lock:
                if self.mode == "camera_setup":
                    self.mode = "idle"

    def _lights(self) -> list[str]:
        return ["back", "front"] if self.dual else []

    def _track_path(self, frame: np.ndarray, t: float) -> None:
        """Follows the first part that runs through the image (full field of view)."""
        lp = self.learn_path
        det = self.trigger.last_det
        H, W = frame.shape[:2]
        if det is not None and det.complete:
            s = self.trigger.scale
            lp["track"].append((t,) + AS.to_sensor(det.centroid[0] * s, det.centroid[1] * s, (W, H), self.camera.crop))
            lp["seen"] = True
        elif lp["seen"] and det is None:                 # the part has left the image
            path = AS.path_from_track(lp["track"])
            if path is None:                             # it did not move (taken away by hand) → wait for the next
                lp["track"], lp["seen"] = [], False
                return
            with self.lock:
                if self.learn_path is not None:
                    self._finish_path(path, (W, H))

    def _finish_path(self, path: dict | None, out_size: tuple[int, int] | None = None) -> None:
        """Belt direction known → digital zoom along the path and exposure limit for the belt speed."""
        ac, prof = self.cfg.autosetup, self.learn_profile
        self.learn_path = None
        if prof is None:
            return
        axis = path["axis"] if path else (prof.get("axis") or self._base_axis)
        cx, cy = prof["center"]
        if path:
            if axis == "x":
                cy = path["across"]
            else:
                cx = path["across"]
        crop, zoom = AS.crop_for_path((cx, cy), tuple(prof["diameter"]), axis, ac)
        prof.update(crop=list(crop), zoom=zoom, axis=axis)
        if path:
            W, H = out_size or (self.last_raw.shape[1], self.last_raw.shape[0])
            v_out = path["speed"] * (W if axis == "x" else H) / crop[2]       # output px per second
            e_max = ac.max_blur_px / max(v_out, 1e-6) * 1e6
            product = prof["exposure_us"] * prof["gain"]
            e, g = AS.split_exposure(product, min(ac.max_exposure_us, e_max), ac)
            prof.update(exposure_us=e, gain=g, belt_speed_px_s=round(v_out, 1),
                        motion_blur_px=round(v_out * e / 1e6, 2))
            if product / e > ac.max_gain + 1e-6:
                msg = (f"Not enough light for this belt speed: motion blur {prof['motion_blur_px']:.1f} px "
                       f"(goal ≤ {ac.max_blur_px}). Brighter light or a slower belt improves accuracy.")
                prof.setdefault("notes", []).append(msg)
                self.log("warn", msg)
        self._applied = None
        self._apply_camera(prof)
        self.trigger.armed, self.trigger.prev_pos = False, None     # ignore the part still in the image
        self.log("ok", f"Belt direction {axis}, digital zoom {zoom:.2f}×"
                       + (f", exposure {prof['exposure_us'] / 1000:.2f} ms (motion blur {prof['motion_blur_px']} px)"
                          if path else "") + " – now capturing the good parts.")

    def model_warnings(self) -> list[str]:
        r = self.model
        if r is None:
            return []
        out = []
        cur = self.calibration.id if self.calibration else None
        if r.calibration_id != cur:
            out.append("Model was trained with a different camera calibration – retrain it (dimensions in mm "
                       "and the image geometry do not match)." if r.calibration_id or cur else "")
        if set(r.channels) != set(self._channels()):
            out.append(f"Model was trained in {r.mode}-light mode, but the system runs in "
                       f"{self.cfg.lighting.mode}-light mode.")
        return [w for w in out if w]

    def select_model(self, slug: str) -> None:
        with self.lock:
            self._set_model(Recipe.load(Path(self.cfg.storage.models_dir) / slug))
            self.log("info", f"Model “{self.model.name}” loaded.")
            for w in self.model_warnings():
                self.log("warn", w)

    def start_inspection(self, slug: str | None = None) -> None:
        if slug:
            self.select_model(slug)
        with self.lock:
            if self.model is None:
                raise ValueError("No model selected.")
            if set(self.model.channels) != set(self._channels()):
                raise ValueError(f"The model was trained in {self.model.mode}-light mode – switch the lighting "
                                 "mode in Setup or teach in again.")
            if self.mode != "idle":
                raise ValueError("Stop the current mode first.")
            self.mode = "inspecting"
            self.log("info", f"Inspection started with model “{self.model.name}”.")

    def stop_inspection(self) -> None:
        with self.lock:
            self._burst = None
            if self.mode in ("inspecting", "selftest"):
                if self.mode == "selftest":
                    self.selftest = None
                self.mode = "idle"
                self.log("info", "Inspection stopped.")

    def manual_capture(self) -> None:
        if self.mode not in ("learning", "inspecting", "selftest"):
            raise ValueError("Manual capture is only available in learning, inspection or self-test mode.")
        self._manual.set()

    # --------------------------------------------------------- inspection
    @property
    def shots_per_part(self) -> int:
        """Multi-shot is always on: at least 3 images per part (majority vote)."""
        return max(3, int(self.cfg.inspection.shots_per_part))

    def _inspect(self, shots: list[dict[str, np.ndarray]], t_capture: float) -> None:
        # Adaptive majority vote: a CLEARLY good first image (every method far inside its
        # tolerance) is final – the other images would only confirm it. Anything else (NOK,
        # borderline, no part) is decided by the majority of all images. Keeps the Pi fast.
        first = self.model.inspect(shots[0])
        clear = first.status == "OK" and all(r.score < self.cfg.inspection.confirm_margin
                                             for r in first.method_results)
        results = [first] if clear or len(shots) == 1 else [first] + [self.model.inspect(c) for c in shots[1:]]
        res = self._vote(results)
        idx = min(range(len(results)), key=lambda i: 0 if results[i] is res else 1)
        captures = shots[idx]
        self._last_captures = captures
        if self.is_sim:
            res.ground_truth = self.camera.current_defect or "ok"
        if self.is_folder:
            res.ground_truth = str(self.camera.current_file.name)
        rid = f"{next(self._ids):06d}"
        res.image_id = rid
        d = res.to_dict()
        d["feedback"] = None
        if self.mode == "selftest":
            self._selftest_step(res, d)
        else:
            self.counts[res.status] = self.counts.get(res.status, 0) + 1
            self.io.signal_result(res.status, t_capture)
            if res.status != "NO_PART":
                self.drift.update(res.measurements)
        self.last_result = d
        self.history.appendleft(d)
        if res.images:
            self.images[rid] = [(im["channel"], _jpg(im["overlay"]), _jpg(im["detail"])) for im in res.images]
        if res.status != "NO_PART":    # keep the images for supervisor feedback (bounded)
            self.frames[rid] = {ch: resize_to_width(f, self.cfg.localization.work_width).copy()
                                for ch, f in captures.items() if f is not None}
            for k in list(self.frames)[:-self.FRAMES_KEPT]:
                del self.frames[k]
        if self.mode == "inspecting" and res.status == "OK":
            stem = self._maybe_collect(res, captures)
            if stem:
                d["collected"] = stem
        keep = {h["image_id"] for h in self.history}
        for store in (self.images, self.frames):
            for k in [k for k in store if k not in keep]:
                del store[k]
        if self.mode != "selftest":
            self._persist(res, captures)

    def _maybe_collect(self, res: qm.InspectionResult, captures: dict[str, np.ndarray]) -> str | None:
        """Self-learning: offer clearly good parts to the model's pool.

        Only parts far inside all tolerances qualify (every method score < margin), and
        nothing is collected while the drift monitor warns (the image is not as learned)."""
        sl = self.cfg.self_learning
        if not sl.enabled or self.model is None or self._bg_learning or self.drift.state()["warnings"]:
            return None
        if any(r.score >= sl.margin for r in res.method_results):
            return None
        if any(s != "OK" for s in res.shots):
            return None
        try:
            stem = self.model.add_collected(captures, sl.max_pool)
        except OSError as e:
            self.log("warn", f"Could not store collected part: {e}")
            return None
        if sl.auto and self.model.collected_count() >= max(1, sl.auto_every):
            self._start_auto_learning()
        return stem

    def _start_auto_learning(self) -> None:
        """Automatic self-learning in the background – inspection keeps running with the old model.
        Called under ``self.lock``. The new model only replaces the old one if all safety checks pass."""
        slug = self.model.slug
        self._bg_learning, self._bg_tainted = slug, False
        sl = self.cfg.self_learning

        def run():
            fresh = None
            try:
                fresh = Recipe.load(Path(self.cfg.storage.models_dir) / slug)
                new, rep = fresh.learn_collected(sl.max_references, sl.growth_limit, commit=False)
                rep = {**rep, "model": fresh.name, "automatic": True}
                with self.lock:
                    if new is not None and self._bg_tainted:
                        # a part of the pool was reported as a missed defect during training
                        new = None
                        rep["accepted"] = False
                        rep["reasons"] = ["A collected part was reported as defective during training."]
                        fresh.reject_learned(rep)
                if new is not None:
                    fresh.commit_learned(new, rep)          # collection is paused, inspection keeps running
                else:
                    fresh.discard_collected()              # do not retry the same (suspicious) pool
                with self.lock:
                    if new is not None and self.model is not None and self.model.slug == slug:
                        new.sensitivity = self.model.sensitivity
                        self._set_model(new)
                    self.last_self_learning = rep
                if new is not None:
                    refs = next(iter(rep["references"].values()))
                    self.log("ok", f"Automatic self-learning “{fresh.name}”: {rep['collected']} good part(s) trained in "
                                   f"({refs['anchor']} anchor + {refs['learned_after']} learned references). "
                                   "Safety checks passed.")
                else:
                    self.log("warn", f"Automatic self-learning “{fresh.name}” rejected – model unchanged, collected "
                                     "parts discarded. " + " ".join(rep["reasons"]))
            except Exception as e:  # noqa: BLE001
                self.log("error", f"Automatic self-learning failed: {e}")
            finally:
                with self.lock:
                    self._bg_learning = None

        threading.Thread(target=run, name="qc-self-learning", daemon=True).start()

    def _vote(self, results: list[qm.InspectionResult]) -> qm.InspectionResult:
        """Majority vote over several shots of the same part."""
        if len(results) == 1:
            results[0].shots = [results[0].status]
            return results[0]
        valid = [r for r in results if r.status != "NO_PART"]
        statuses = [r.status for r in results]
        if not valid:
            rep = results[0]
        else:
            n_nok = sum(r.status == "NOK" for r in valid)
            nok = n_nok * 2 > len(valid) or (n_nok * 2 == len(valid) and self.cfg.inspection.tie_is_nok)
            if nok:
                rep = max((r for r in valid if r.status == "NOK"), key=lambda r: r.worst)
            else:
                rep = next(r for r in valid if r.status == "OK")
        rep.shots = statuses
        rep.time_ms = sum(r.time_ms for r in results)
        return rep

    def _day_dir(self) -> Path:
        day = Path(self.cfg.storage.results_dir) / datetime.now().strftime("%Y-%m-%d")
        day.mkdir(parents=True, exist_ok=True)
        return day

    def _persist(self, res: qm.InspectionResult, captures: dict[str, np.ndarray]) -> None:
        st = self.cfg.storage
        day = self._day_dir()
        stem = f"{datetime.now().strftime('%H%M%S_%f')[:-3]}_{res.status}"
        save_img = (res.status == "NOK" and st.save_nok_images) or (res.status == "OK" and st.save_ok_images)
        if save_img and res.images:
            for im in res.images:
                suffix = "" if im["channel"] == "main" else f"_{im['channel']}"
                cv2.imwrite(str(day / f"{stem}{suffix}_marked.jpg"), im["overlay"])
                if captures.get(im["channel"]) is not None:
                    cv2.imwrite(str(day / f"{stem}{suffix}_original.png"), captures[im["channel"]])
        log = day / "inspection_log.csv"
        new = not log.exists()
        with log.open("a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["time", "model", "result", "defects", "duration_ms", "image", "ground_truth",
                            "image_id", "shots"])
            w.writerow([res.timestamp, res.model, res.status,
                        " | ".join(f"{x.label}: {x.detail}" for x in res.defects),
                        f"{res.time_ms:.0f}", stem if save_img else "", res.ground_truth or "",
                        res.image_id, "/".join(res.shots)])

    # ----------------------------------------------------------- feedback
    def feedback(self, rid: str, verdict: str, add_reference: bool = False) -> dict:
        """Supervisor corrects/confirms a result: verdict "ok" or "nok"."""
        if verdict not in ("ok", "nok"):
            raise ValueError("verdict must be 'ok' or 'nok'.")
        with self.lock:
            entry = next((h for h in self.history if h["image_id"] == rid), None)
            if entry is None or entry.get("selftest"):
                raise ValueError("Result not found (only the recent history can be corrected).")
            if entry["status"] == "NO_PART":
                raise ValueError("Nothing to correct for 'no part'.")
            if entry.get("feedback"):
                self.feedback_counts[_outcome_key(entry["feedback"])] -= 1
            is_nok = entry["status"] == "NOK"
            outcome = ("false_alarm" if verdict == "ok" else "confirmed_nok") if is_nok else \
                      ("missed" if verdict == "nok" else "confirmed_ok")
            entry["feedback"] = outcome
            self.feedback_counts[_outcome_key(outcome)] += 1
            added = 0
            recipe = self.model if self.model and self.model.name == entry["model"] else None
            captures = self.frames.get(rid)
            if outcome == "missed" and recipe is not None:
                if entry.get("collected"):          # a defective part must never be learned as good
                    recipe.remove_collected(entry.pop("collected"))
                    if self._bg_learning == recipe.slug:
                        self._bg_tainted = True
                if captures:
                    recipe.add_known_bad(captures, "missed")
            if outcome == "confirmed_nok" and recipe is not None and captures:
                recipe.add_known_bad(captures, "confirmed")
            if add_reference and outcome == "false_alarm":
                if not captures:
                    raise ValueError("The image of this part is no longer available.")
                if recipe is None:
                    raise ValueError("The model of this result is not loaded.")
                added = recipe.add_pending(captures)
                entry["reference_added"] = True
            day = self._day_dir()
            log = day / "feedback_log.csv"
            new = not log.exists()
            with log.open("a", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                if new:
                    w.writerow(["time", "image_id", "inspected_at", "model", "result", "supervisor", "outcome",
                                "added_as_reference"])
                w.writerow([datetime.now().isoformat(timespec="seconds"), rid, entry["timestamp"], entry["model"],
                            entry["status"], verdict.upper(), outcome, "yes" if entry.get("reference_added") else ""])
        text = {"false_alarm": "false alarm", "missed": "missed defect", "confirmed_ok": "confirmed OK",
                "confirmed_nok": "confirmed NOK"}[outcome]
        self.log("warn" if outcome in ("false_alarm", "missed") else "info",
                 f"Supervisor feedback for #{rid}: {text}"
                 + (f" – queued as reference ({added} pending, retrain the model to use them)" if added else ""))
        return {"outcome": outcome, "pending": added}

    # ----------------------------------------------------------- self-test
    def start_selftest(self) -> None:
        """Reference part check (e.g. at shift start): one known good and one known bad part."""
        with self.lock:
            if self.model is None:
                raise ValueError("Select a model first.")
            if self.mode != "idle":
                raise ValueError("Stop the current mode first.")
            if set(self.model.channels) != set(self._channels()):
                raise ValueError("The model does not match the lighting mode.")
            self.selftest = {"step": "good", "steps": [], "started": datetime.now().isoformat(timespec="seconds")}
            self.mode = "selftest"
            if self.is_sim:
                self.camera.force_good = True
            self.log("info", "Self-test started – feed the known GOOD reference part.")

    def _selftest_step(self, res: qm.InspectionResult, d: dict) -> None:
        st = self.selftest
        expected = "OK" if st["step"] == "good" else "NOK"
        if res.status == "NO_PART":
            self.log("warn", "Self-test: no part recognised – feed the part again.")
            return
        passed = res.status == expected
        d["selftest"] = st["step"]
        if st["step"] == "bad" and passed and self._last_captures is not None:
            self.model.add_known_bad(self._last_captures, "selftest")
        st["steps"].append({"part": st["step"], "expected": expected, "result": res.status, "passed": passed,
                            "defects": [x.label for x in res.defects]})
        self.log("ok" if passed else "error",
                 f"Self-test {st['step']} part: expected {expected}, got {res.status} → {'PASS' if passed else 'FAIL'}")
        if st["step"] == "good":
            st["step"] = "bad"
            if self.is_sim:
                self.camera.force_good = False
                self.camera.force_defect = True
            self.log("info", "Self-test – now feed the known DEFECTIVE reference part.")
            return
        ok = all(s["passed"] for s in st["steps"])
        summary = {"time": datetime.now().isoformat(timespec="seconds"), "model": self.model.name,
                   "passed": ok, "steps": st["steps"]}
        self.last_selftest = summary
        try:
            self._selftest_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        except OSError:
            pass
        log = self._day_dir() / "selftest_log.csv"
        new = not log.exists()
        with log.open("a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["time", "model", "passed", "good_part_result", "bad_part_result"])
            w.writerow([summary["time"], summary["model"], "PASS" if ok else "FAIL",
                        st["steps"][0]["result"], st["steps"][1]["result"]])
        self.log("ok" if ok else "error", f"Self-test {'PASSED' if ok else 'FAILED'}"
                 + ("" if ok else " – do not start production; check lighting, camera and model."))
        self.selftest = None
        self.mode = "idle"
        if self.is_sim:
            self.camera.force_good = False
            self.camera.force_defect = False

    # ----------------------------------------------------------- calibration
    def calib_start(self, cols: int | None = None, rows: int | None = None, square_mm: float | None = None) -> None:
        self._apply_camera(None)            # calibrate at the full field of view (zoom is accounted for later)
        c = self.cfg.calibration
        self.calib_session = {"cols": int(cols or c.board_cols), "rows": int(rows or c.board_rows),
                              "square_mm": float(square_mm or c.square_mm), "corners": [], "size": None,
                              "preview": None}

    def calib_capture(self) -> dict:
        if self.mode != "idle":
            raise ValueError("Stop learning/inspection before calibrating.")
        if self.calib_session is None:
            self.calib_start()
        s = self.calib_session
        frame = self.last_raw
        if frame is None:
            raise ValueError("No camera image yet.")
        corners = qcal.find_corners(frame, s["cols"], s["rows"])
        s["preview"] = _jpg(resize_to_width(qcal.draw_corners(frame, corners, s["cols"], s["rows"]), 640), 80)
        if corners is None:
            raise ValueError(f"Checkerboard with {s['cols']} × {s['rows']} inner corners not found – "
                             "check the corner count, lighting and that the whole board is visible.")
        size = (frame.shape[1], frame.shape[0])
        if s["size"] and s["size"] != size:
            raise ValueError("Camera resolution changed during the calibration – start again.")
        s["size"] = size
        s["corners"].append(corners)
        n = len(s["corners"])
        self.log("info", f"Calibration image {n} captured" + (" (scale reference – board flat on the belt)." if n == 1 else "."))
        return {"count": n}

    def calib_compute(self) -> dict:
        s = self.calib_session
        if not s or not s["corners"]:
            raise ValueError("Capture at least one checkerboard image first.")
        cal = qcal.compute(s["corners"], s["size"], s["cols"], s["rows"], s["square_mm"])
        cal.save(self.cfg.calibration.file)
        with self.lock:
            self.calibration = cal
            self.undistort = qcal.Undistorter(cal)
        self.calib_session = None
        self._apply_camera(self.model.camera_profile if self.model else None)
        msg = f"Calibration saved: {self.mm_per_px:.4f} mm/px"
        if cal.camera_matrix:
            msg += f", lens distortion corrected (RMS {cal.rms_px:.2f} px)"
        elif not cal.note:
            msg += f" (scale only – capture ≥ {qcal.MIN_IMAGES_DISTORTION} images to also correct lens distortion)"
        self.log("ok", msg + ". Retrain existing models.")
        if cal.note:
            self.log("warn", cal.note)
        return self.calibration_status()

    def calib_cancel(self) -> None:
        self.calib_session = None
        self._apply_camera(self.model.camera_profile if self.model else None)

    def calib_delete(self) -> None:
        Path(self.cfg.calibration.file).unlink(missing_ok=True)
        with self.lock:
            self.calibration, self.undistort = None, None
        self.log("info", "Calibration deleted – measuring in pixels.")

    def calibration_status(self) -> dict:
        c = self.calibration
        s = self.calib_session
        return {
            "calibrated": c is not None,
            "id": c.id if c else None,
            "mm_per_px": round(self.mm_per_px, 5) if c else None,
            "distortion": bool(c and c.camera_matrix),
            "rms_px": round(c.rms_px, 3) if c and c.rms_px is not None else None,
            "n_images": c.n_images if c else 0,
            "board": c.board if c else None,
            "note": c.note if c else "",
            "session": {"count": len(s["corners"]), "cols": s["cols"], "rows": s["rows"],
                        "square_mm": s["square_mm"], "has_preview": s["preview"] is not None} if s else None,
            "defaults": {"cols": self.cfg.calibration.board_cols, "rows": self.cfg.calibration.board_rows,
                         "square_mm": self.cfg.calibration.square_mm},
        }

    # ------------------------------------------------------------ settings
    def update_settings(self, lighting_mode: str | None = None,
                        self_learning: bool | None = None, self_learning_auto: bool | None = None) -> None:
        with self.lock:
            if self_learning_auto is not None:
                self.cfg.self_learning.auto = bool(self_learning_auto)
                self.log("info", f"Automatic self-learning: {'on' if self_learning_auto else 'off'}")
            if self_learning is not None:
                self.cfg.self_learning.enabled = bool(self_learning)
                self.log("info", f"Collecting good parts for self-learning: {'on' if self_learning else 'off'}")
            if lighting_mode is not None and lighting_mode != self.cfg.lighting.mode:
                if self.mode != "idle":
                    raise ValueError("Stop learning/inspection before changing the lighting mode.")
                if lighting_mode not in ("single", "dual"):
                    raise ValueError("Lighting mode must be 'single' or 'dual'.")
                if lighting_mode == "dual" and not self.camera.continuous:
                    raise ValueError("Dual-light mode needs a live camera.")
                self.cfg.lighting.mode = lighting_mode
                self._apply_idle_light()
                self.log("info", f"Lighting mode: {lighting_mode}. Models must be taught in for this mode.")

    # ------------------------------------------------------------ management
    def delete_model(self, slug: str) -> None:
        import shutil

        path = Path(self.cfg.storage.models_dir) / slug
        if not ((path / "recipe.json").exists() or (path / "meta.json").exists()):
            raise ValueError("Model not found.")
        with self.lock:
            if self._bg_learning == slug:
                raise ValueError("The model is being updated right now – try again in a moment.")
            if self.model is not None and self.model.slug == slug:
                if self.mode in ("inspecting", "selftest"):
                    raise ValueError("The active model cannot be deleted while inspection is running.")
                self._set_model(None)
            shutil.rmtree(path)
            self.log("info", f"Model “{slug}” deleted.")

    def retrain_model(self, slug: str, overrides: dict) -> None:
        path = Path(self.cfg.storage.models_dir) / slug
        with self.lock:
            if self.mode != "idle":
                raise ValueError("Please stop the inspection first.")
            if self._bg_learning:
                raise ValueError("Automatic self-learning is running – try again in a moment.")
            if overrides:
                self._method_cfg(overrides)       # validate
            self.mode = "training"

        def run():
            try:
                old = Recipe.load(path)
                n_pending = old.pending_count()
                new = old.retrain(overrides or None)
                with self.lock:
                    self._set_model(new)
                self.log("ok", f"Model “{new.name}” retrained"
                               + (f" including {n_pending} feedback reference(s)" if n_pending else "") + ".")
                for w in new.warnings():
                    self.log("warn", w)
            except Exception as e:  # noqa: BLE001
                self.log("error", f"Retraining failed: {e}")
            finally:
                with self.lock:
                    self.mode = "idle"

        threading.Thread(target=run, daemon=True).start()

    def learn_collected(self, slug: str) -> None:
        """Button “Learn collected good parts”: train, check, and only then replace the model."""
        path = Path(self.cfg.storage.models_dir) / slug
        with self.lock:
            if self.mode != "idle":
                raise ValueError("Please stop the inspection first.")
            if self._bg_learning:
                raise ValueError("Automatic self-learning is running – try again in a moment.")
            recipe = Recipe.load(path)
            if recipe.collected_count() == 0:
                raise ValueError("No collected good parts yet.")
            self.mode = "training"
        sl = self.cfg.self_learning

        def run():
            try:
                new, rep = recipe.learn_collected(sl.max_references, sl.growth_limit)
                self.last_self_learning = {**rep, "model": recipe.name}
                if new is not None:
                    with self.lock:
                        if self.model is None or self.model.slug == slug:
                            self._set_model(new)
                    thr = ", ".join(f"{t['what']} {t['change_percent']:+.0f} %" for t in rep["thresholds"]
                                    if t["method"] == "pca")
                    refs = next(iter(rep["references"].values()))
                    self.log("ok", f"Self-learning “{recipe.name}”: {rep['collected']} collected part(s) trained in "
                                   f"({refs['anchor']} anchor + {refs['learned_after']} learned references"
                                   + (f"; {thr}" if thr else "") + "). Safety checks passed.")
                else:
                    self.log("error", f"Self-learning “{recipe.name}” REJECTED – model unchanged. "
                             + " ".join(rep["reasons"]) + " Check/discard the collected parts.")
            except Exception as e:  # noqa: BLE001
                self.log("error", f"Self-learning failed: {e}")
            finally:
                with self.lock:
                    self.mode = "idle"

        threading.Thread(target=run, daemon=True).start()

    def discard_collected(self, slug: str) -> int:
        with self.lock:
            recipe = self.model if self.model and self.model.slug == slug else \
                Recipe.load(Path(self.cfg.storage.models_dir) / slug)
            n = recipe.discard_collected()
        self.log("info", f"{n} collected part(s) of “{recipe.name}” discarded.")
        return n

    def set_sensitivity(self, slug: str, values: dict) -> dict:
        with self.lock:
            recipe = self.model if self.model and self.model.slug == slug else \
                Recipe.load(Path(self.cfg.storage.models_dir) / slug)
            recipe.set_sensitivity(values)
            if self.model and self.model.slug == slug:
                self.model.sensitivity = recipe.sensitivity
        self.log("info", f"Sensitivity of “{recipe.name}”: "
                         + ", ".join(f"{k} ×{v:g}" for k, v in recipe.sensitivity.items()))
        return recipe.sensitivity

    def set_sim(self, lighting: str | None = None, defect_rate: float | None = None, part_type: str | None = None,
                any_angle: bool | None = None, board: bool | None = None) -> None:
        if not self.is_sim:
            raise ValueError("Only available in the simulator.")
        from .synthetic import PART_TYPES

        with self.lock:
            if board is not None:
                self.camera.show_board(self.cfg.calibration.board_cols, self.cfg.calibration.board_rows,
                                       self.cfg.calibration.square_mm) if board else self.camera.hide_board()
                return
            if defect_rate is not None:
                self.camera.defect_rate = float(np.clip(defect_rate, 0, 1))
            if part_type in PART_TYPES:
                self.camera.part_type = part_type
            if any_angle is not None:
                self.camera.cfg.sim_any_angle = bool(any_angle)
            if lighting in ("front", "back"):
                self.camera.set_lighting(lighting)
                self._apply_idle_light()
            else:
                self.camera._spawn()

    def archive_results(self, label: str = "") -> dict:
        """Batch finished: pack all results into a ZIP, empty the results folder, reset counters."""
        with self.lock:   # inspections also write under this lock → no half-written files
            if self.mode in ("inspecting", "selftest"):
                raise ValueError("Please stop the inspection first.")
            if not label.strip() and self.model is not None:
                label = self.model.name
            info = qa.create_archive(self.cfg.storage.results_dir, self.cfg.storage.archive_dir, label)
            self.reset_counts()
        s = info["summary"]
        self.log("ok", f"Results archived as {info['file']} ({s['inspected']} parts, {s['nok']} NOK) – "
                       "results folder cleared.")
        return info

    def reset_counts(self) -> None:
        with self.lock:
            self.counts = {"OK": 0, "NOK": 0, "NO_PART": 0}
            self.feedback_counts = {"false_alarm": 0, "missed": 0, "confirmed": 0}
            self.history.clear()
            self.images.clear()
            self.frames.clear()
            self.last_result = None

    def list_models(self) -> list[dict]:
        return list_recipes(self.cfg.storage.models_dir)

    DRIFT_HINTS = {
        "contrast": "The image contrast has changed. Check the lighting (lamp off, dirty or ageing).",
        "background": "The belt / background brightness has changed. Check ambient light, clean the belt or backlight.",
        "area": "Parts look bigger or smaller than when taught. Has the camera been moved?",
        "sharpness": "The image is getting blurry. Clean the camera lens and check the focus.",
    }

    def supervisor_notices(self, drift: dict | None = None) -> list[dict]:
        """Short plain-language hints for the supervisor screen (no technical details)."""
        out: list[dict] = []
        if self.camera.continuous and self.fps == 0 and self.last_frame is None:
            out.append({"level": "error", "text": "No camera image. Check the camera cable and restart the device."})
        if self.model is None and self.mode == "idle":
            out.append({"level": "info", "text": "Select a part, or teach in a new part."})
        if self.model_warnings():
            out.append({"level": "error", "text": "This part has to be taught in again (the camera setup was "
                                                  "changed). Call the setup technician or teach it in again."})
        if self.model is not None:
            drift = drift or self.drift.state()
            keys = {r["key"] for r in drift.get("rows", []) if r.get("warning")}
            out += [{"level": "warn", "text": self.DRIFT_HINTS[k]} for k in ("sharpness", "contrast", "background", "area")
                    if k in keys]
        prof = (self.model.camera_profile or {}) if self.model else {}
        if prof.get("sharp") is False:
            out.append({"level": "warn", "text": "The camera image is not sharp – the lens has to be focused once "
                                                 "(setup technician). Detection is less precise until then."})
        st = self.last_selftest
        if st and not st.get("passed") and self.model is not None and st.get("model") == self.model.name:
            out.append({"level": "error", "text": "The last reference part check FAILED. Do not start production – "
                                                  "call the setup technician."})
        ls = self.last_self_learning
        if ls and ls.get("automatic") and not ls.get("accepted"):
            out.append({"level": "info", "text": "An automatic model update was rejected (safety check). "
                                                 "The model is unchanged – please inform the setup technician."})
        return out

    def status(self) -> dict:
        m = self.model
        drift = self.drift.state() if m else {"rows": [], "warnings": []}
        return {
            "mode": self.mode, "camera": self.camera.name, "fps": round(self.fps, 1),
            "armed": self.trigger.armed, "continuous": self.camera.continuous,
            "model": m.summary() if m else None,
            "model_warnings": self.model_warnings(),
            "learning": {"name": self.learn_name, "count": len(self.learn_captures),
                         "target": self.cfg.target_reference_count, "minimum": qm.MIN_REFERENCES,
                         "auto_finish": self.learn_auto_finish, "path_pending": self.learn_path is not None},
            "camera_setup": self.cam_setup,
            "camera_controls": bool(self.camera.can_control and self.cfg.autosetup.enabled),
            "camera_state": {"exposure_us": (self._applied or {}).get("exposure", (None,))[0] if (self._applied or {}).get("exposure") else None,
                             "gain": (self._applied or {}).get("exposure", (None, None))[1] if (self._applied or {}).get("exposure") else None,
                             "zoom": round(1.0 / self.camera.crop[2], 2), "axis": self.cfg.trigger.axis,
                             "auto": bool(m and (m.camera_profile or {}).get("auto"))},
            "self_learning_running": self._bg_learning is not None,
            "notices": self.supervisor_notices(drift),
            "setup_pin_required": bool(self.cfg.ui.setup_pin),
            "focus_assist": {"value": round(self.focus_assist["value"], 1), "best": round(self.focus_assist["best"], 1)}
            if self.focus_assist else None,
            "camera_info": getattr(self.camera, "info", ""),
            "counts": self.counts, "feedback": self.feedback_counts, "last_result": self.last_result,
            "messages": list(self.messages)[:12],
            "settings": {"shots_per_part": self.shots_per_part, "lighting_mode": self.cfg.lighting.mode,
                         "idle_light": self.cfg.lighting.idle_light,
                         "rotation_search": self.cfg.localization.rotation_search,
                         "self_learning": self.cfg.self_learning.enabled,
                         "self_learning_auto": self.cfg.self_learning.auto,
                         "self_learning_every": self.cfg.self_learning.auto_every,
                         "self_learning_margin": self.cfg.self_learning.margin,
                         "self_learning_max_pool": self.cfg.self_learning.max_pool},
            "last_self_learning": self.last_self_learning,
            "io": self.io.status(),
            "drift": drift,
            "calibration": self.calibration_status(),
            "selftest": {"active": self.selftest, "last": self.last_selftest},
            "sim": {"lighting": self.camera.cfg.sim_lighting, "defect_rate": self.camera.defect_rate,
                    "part_type": self.camera.part_type, "any_angle": self.camera.cfg.sim_any_angle,
                    "board": self.camera.board is not None}
            if self.is_sim else None,
        }


def _outcome_key(outcome: str) -> str:
    return "confirmed" if outcome.startswith("confirmed") else outcome
