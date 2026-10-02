"""Process control: camera loop, conveyor trigger, teach-in, inspection.

The ``QCSystem`` class is independent of the user interface; the web UI
(``webapp.py``) only calls its methods. It is split into the core (this file) and
three mixins: camera (``system_camera.py``), models (``system_models.py``) and
status (``system_status.py``).

Two threads do the work, so the live image and the trigger never wait for an inspection::

    camera thread:  frame → trigger → 3 shots (or dual-light pair) ──┐ queue (max. 3 parts)
    worker thread:  ◄─────────────────────────────────────────────────┘
                    recipe.inspect() → vote → UI / log / lamps / reject / drift monitor

If the worker cannot keep up (queue full), the part is rejected without inspection
(fail-safe) and the supervisor gets a notice.
"""
from __future__ import annotations

import csv
import itertools
import json
import logging
import queue
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
from .alignment import detect_part, resize_to_width, to_gray
from .camera import CameraSource, FolderSource, SimulatorSource
from .config import AppConfig
from .drift import DriftMonitor
from .io_control import IOController
from .recipe import Recipe
from .system_camera import CameraMixin
from .system_models import ModelsMixin
from .system_status import StatusMixin
from .trigger import PartTrigger
from .visualize import encode_jpg as _jpg


class QCSystem(CameraMixin, ModelsMixin, StatusMixin):
    PREVIEW_WIDTH = 640
    FRAMES_KEPT = 30          # results whose images are kept in memory for feedback
    QUEUE_PARTS = 3           # parts waiting for inspection before new ones are rejected uninspected

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
        self._thread: threading.Thread | None = None
        self._worker_thread: threading.Thread | None = None
        self._jobs: queue.Queue = queue.Queue(maxsize=self.QUEUE_PARTS)
        self.skipped = 0                                 # parts rejected uninspected (Pi overloaded)
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

        self._settings_file = Path(cfg.storage.models_dir).parent / "settings.json"
        self._load_settings()
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

    def log(self, level: str, text: str) -> None:
        """Message for the UI (last 30) and the log file (``data/logs/qc.log``, see main.py)."""
        self.messages.appendleft({"t": datetime.now().strftime("%H:%M:%S"), "level": level, "text": text})
        logging.getLogger("qc").log({"error": logging.ERROR, "warn": logging.WARNING}.get(level, logging.INFO), text)

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
        self.io.set_light(self.cfg.lighting.idle_light)
        if self.is_sim and not self.dual:
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

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, name="qc-camera", daemon=True)
        self._worker_thread = threading.Thread(target=self._worker, name="qc-inspect", daemon=True)
        self._thread.start()
        self._worker_thread.start()

    def stop(self) -> None:
        self._stop.set()
        for t in (self._thread, self._worker_thread):
            if t:
                t.join(timeout=3)
        self.camera.close()
        self.io.close()

    # ------------------------------------------------------------ worker
    def _submit(self, shots: list[dict[str, np.ndarray]], t_capture: float, manual: bool) -> None:
        """Hands a captured part to the inspection thread."""
        try:
            self._jobs.put_nowait((shots, t_capture, manual))
        except queue.Full:
            self.skipped += 1
            if self.mode == "inspecting":
                self.io.signal_result("NOK", t_capture)      # fail-safe: never let an uninspected part pass
            self.log("warn", "Inspection cannot keep up – part rejected without inspection. Slow the belt down "
                             "or increase the distance between the parts.")

    def _worker(self) -> None:
        while not self._stop.is_set():
            try:
                shots, t_capture, manual = self._jobs.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                self._handle_part(shots, t_capture, manual)
            except Exception as e:  # noqa: BLE001 - one bad part must not stop the inspection
                logging.getLogger("qc").exception("Inspection failed")
                self.log("error", f"Inspection of a part failed: {e}")

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
            manual = self._manual.is_set()
            if manual:
                self._manual.clear()
                if self.is_folder and self.mode == "learning":
                    frame = self.camera.next_part()
                fire = frame is not None and active
            if fire and self._burst is None:
                self._on_part(frame, t_capture, manual)

            n += 1
            if time.time() - t_last >= 1.0:
                self.fps = n / (time.time() - t_last)
                t_last, n = time.time(), 0
            if time.time() - last_preview > 1.0 / max(0.5, self.cfg.ui.preview_fps):
                self._update_preview(frame)
                last_preview = time.time()

    def _on_part(self, frame: np.ndarray, t_capture: float, manual: bool = False) -> None:
        """A part was triggered: collect the images of this part (camera thread)."""
        if self.dual and self.camera.continuous:
            self._submit([self._capture_dual(frame)], t_capture, manual)
        elif self.mode in ("inspecting", "selftest") and self.camera.continuous:
            self._burst = {"shots": [{"main": frame}], "n": self.shots_per_part, "t0": t_capture}
        else:
            self._submit([{"main": frame}], t_capture, manual)

    def _continue_burst(self, frame: np.ndarray) -> None:
        b = self._burst
        if self.trigger.part_complete and len(b["shots"]) < b["n"]:
            b["shots"].append({"main": frame})
        if len(b["shots"]) >= b["n"] or not self.trigger.part_complete:
            self._burst = None
            self._submit(b["shots"], b["t0"], False)

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

    def _handle_part(self, shots: list[dict[str, np.ndarray]], t_capture: float, manual: bool = False) -> None:
        """Teach-in or inspection of one captured part (worker thread)."""
        shots = self._undistort_shots(shots)
        with self.lock:
            if self.mode == "learning" and self.learn_path is not None:
                if not self.camera.continuous or manual:
                    # stopped belt / image folder: no belt direction to measure → zoom from the still part
                    self._finish_path(None)
                    self.log("info", "Camera zoomed in on the part – capture the part again.")
                return                                   # the path part is never a reference
            if self.mode == "learning":
                self._add_reference(shots[0])
            elif self.mode in ("inspecting", "selftest") and self.model is not None:
                self._inspect(shots, t_capture)

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

    def _set_model(self, recipe: Recipe | None) -> None:
        self.model = recipe
        if recipe is not None:
            self.drift.reset({ch: m.baseline for ch, m in recipe.channels.items()})
            self._apply_camera(recipe.camera_profile)

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

    # ------------------------------------------------------------ settings
    # Settings changed in the setup area are stored in data/settings.json and applied on top of
    # config.json at every start (config.json itself is never rewritten by the program).
    SETTINGS_KEYS = {"lighting_mode": ("lighting", "mode"), "self_learning": ("self_learning", "enabled"),
                     "self_learning_auto": ("self_learning", "auto"), "methods": ("method", "methods"),
                     "diff_representation": ("method", "diff_representation"),
                     "pca_representation": ("method", "pca_representation")}

    def _load_settings(self) -> None:
        try:
            saved = json.loads(self._settings_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        for key, (section, attr) in self.SETTINGS_KEYS.items():
            if key in saved:
                setattr(getattr(self.cfg, section), attr, saved[key])
        try:
            self.cfg.validate()
        except ValueError as e:                    # a broken settings file must not stop the system
            self.log("warn", f"Ignoring saved settings ({e}).")

    def _save_settings(self) -> None:
        data = {key: getattr(getattr(self.cfg, section), attr) for key, (section, attr) in self.SETTINGS_KEYS.items()}
        tmp = self._settings_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        tmp.replace(self._settings_file)

    def set_method_defaults(self, values: dict) -> dict:
        """Methods and image representations used for new models (setup area → Models)."""
        from .methods import METHODS
        from .representations import REPRESENTATIONS

        with self.lock:
            m = self.cfg.method
            if "methods" in values:
                methods = [x for x in METHODS if x in values["methods"]]        # fixed order
                if not methods:
                    raise ValueError("Select at least one inspection method.")
                m.methods = methods
            for k in ("diff_representation", "pca_representation"):
                if values.get(k) in REPRESENTATIONS:
                    setattr(m, k, values[k])
            self._save_settings()
        self.log("info", f"Method settings for new models: {', '.join(m.methods)} "
                         f"(comparison: {m.diff_representation}, ML: {m.pca_representation})")
        return {"methods": m.methods, "diff_representation": m.diff_representation,
                "pca_representation": m.pca_representation}

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
            self._save_settings()

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
            self.skipped = 0
            self.history.clear()
            self.images.clear()
            self.frames.clear()
            self.last_result = None

