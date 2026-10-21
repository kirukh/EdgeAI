"""Process control: camera loop, conveyor trigger, teach-in and inspection.

``QCSystem`` coordinates the components and owns the operating mode; the work is done by them:

    CameraSetup      (camera_setup.py)  camera profiles, automatic camera setup, focus assistant
    TeachIn          (teach.py)         reference images of a new part type
    QCModel          (model.py)         training and inspection of one part type
    inspect_part     (inspection.py)    several images per part, majority vote
    ResultStore      (results.py)       history, counters, feedback, CSV logs
    ModelImprovement (improvement.py)   continuous improvement from the operator's decisions
    DriftMonitor     (drift.py)         image quality compared with the learning phase

Two threads do the work, so the live image and the trigger never wait for an inspection::

    camera thread:  frame → trigger → 3 shots ──┐ queue (max. 3 parts)
    worker thread:  ◄───────────────────────────┘ inspect_part() → results / drift monitor

If the worker cannot keep up (queue full), the part is logged as not inspected and the
supervisor gets a notice. The web UI (``qc/web``) only calls the public methods of the system
and its components.
"""

from __future__ import annotations

import logging
import queue
import shutil
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from . import archive as qa
from .alignment import resize_to_width
from .camera import CameraSource, FolderSource, SimulatorSource
from .camera_setup import CameraSetup
from .config import AppConfig
from .drift import DriftMonitor
from .improvement import ModelImprovement
from .inspection import inspect_part
from .learning import KNOWN_BAD, PENDING, LearningStore
from .model import MIN_REFERENCES, QCModel, has_previous, list_models, model_path
from .results import OUTCOME_TEXT, ResultStore
from .synthetic import PART_TYPES
from .teach import TeachIn
from .trigger import PartTrigger

MODES = ("idle", "camera_setup", "learning", "training", "inspecting")
PREVIEW_WIDTH = 640
QUEUE_PARTS = 3  # parts waiting for inspection before new ones are not inspected
MESSAGES_KEPT = 30
DRIFT_HINTS = {
    "sharpness": "The image is getting blurry. Clean the camera lens and check the focus.",
    "contrast": "The image contrast has changed. Check the lighting (lamp off, dirty or ageing).",
    "background": "The belt / background brightness has changed. Check ambient light, clean the belt or backlight.",
    "area": "Parts look bigger or smaller than when taught. Has the camera been moved?",
}
PREVIEW_LABEL = {
    "idle": "READY",
    "camera_setup": "ADJUSTING CAMERA",
    "learning": "LEARNING",
    "training": "TRAINING ...",
    "inspecting": "INSPECTION ACTIVE",
}


class QCSystem:
    def __init__(self, cfg: AppConfig, camera: CameraSource):
        self.cfg = cfg
        self.camera = camera
        self.lock = threading.RLock()
        self.mode = "idle"
        self.model: QCModel | None = None
        self.messages: deque[dict] = deque(maxlen=MESSAGES_KEPT)
        self.trigger = PartTrigger(cfg.trigger, cfg.localization)
        self.teach = TeachIn(cfg.target_reference_count, cfg.localization)
        self.results = ResultStore(cfg.storage, cfg.localization.work_width)
        self.drift = DriftMonitor(cfg.drift)
        self.cam = CameraSetup(camera, cfg, self.trigger, self.log, self._wait_frames, lambda: self.last_raw)
        self.preview_jpeg: bytes | None = None
        self.last_frame: np.ndarray | None = None
        self.last_raw: np.ndarray | None = None
        self.fps = 0.0
        self._frame_seq = 0
        self._frame_cv = threading.Condition()
        self._stop = threading.Event()
        self._manual = threading.Event()
        self._threads: list[threading.Thread] = []
        self._jobs: queue.Queue = queue.Queue(maxsize=QUEUE_PARTS)
        self._burst: dict | None = None
        self._last_folder_step = 0.0
        self.folder_interval = 1.0
        self.improve = ModelImprovement(cfg, self.lock, lambda: self.model, self._set_model, self.log)
        Path(cfg.storage.models_dir).mkdir(parents=True, exist_ok=True)
        for section in cfg.ignored:
            self.log("warn", f"config.json: section “{section}” belongs to the full version and is ignored.")
        if getattr(camera, "info", ""):
            self.log("info", f"Camera: {camera.info}")
        self.log(
            "info",
            f"System started – camera: {camera.name}"
            + (f", {cfg.camera.mm_per_px} mm/px" if cfg.camera.mm_per_px else ""),
        )

    def log(self, level: str, text: str) -> None:
        """Message for the UI (last 30) and the log file (``data/logs/qc.log``, see cli.py)."""
        self.messages.appendleft({"t": datetime.now().strftime("%H:%M:%S"), "level": level, "text": text})
        logging.getLogger("qc").log({"error": logging.ERROR, "warn": logging.WARNING}.get(level, logging.INFO), text)

    # ------------------------------------------------------------------ properties
    @property
    def mm_per_px(self) -> float | None:
        """Scale of the working image; with digital zoom a pixel covers only the crop's fraction."""
        base = self.cfg.camera.mm_per_px
        return base * self.camera.crop[2] if base else None

    @property
    def shots_per_part(self) -> int:
        """Multi-shot is always on: at least 3 images per part (majority vote)."""
        return max(3, int(self.cfg.inspection.shots_per_part))

    # ------------------------------------------------------------------ threads
    def start(self) -> None:
        self._threads = [
            threading.Thread(target=self._camera_loop, name="qc-camera", daemon=True),
            threading.Thread(target=self._worker, name="qc-inspect", daemon=True),
        ]
        for t in self._threads:
            t.start()

    def stop(self) -> None:
        self._stop.set()
        for t in self._threads:
            t.join(timeout=3)
        self.camera.close()

    def _read(self) -> np.ndarray | None:
        raw = self.camera.read()
        if raw is not None:
            self.last_raw = raw
        return raw

    def _wait_frames(self, n: int, timeout: float = 5.0) -> None:
        with self._frame_cv:
            target = self._frame_seq + n
            self._frame_cv.wait_for(lambda: self._frame_seq >= target, timeout)

    def _camera_loop(self) -> None:
        t_fps, n, last_preview = time.time(), 0, 0.0
        while not self._stop.is_set():
            frame = self._read()
            if frame is None:
                time.sleep(0.05)
                continue
            t_capture = time.time() - self.camera.last_age_s  # when the sensor saw it (belt speed)
            self.last_frame = frame
            with self._frame_cv:
                self._frame_seq += 1
                self._frame_cv.notify_all()
            part, manual = self._poll_trigger(frame, t_capture)
            if part is not None and self._burst is None:
                self._on_part(part, manual)
            n += 1
            if time.time() - t_fps >= 1.0:
                self.fps, t_fps, n = n / (time.time() - t_fps), time.time(), 0
            if time.time() - last_preview > 1.0 / max(0.5, self.cfg.ui.preview_fps):
                self._update_preview(frame)
                last_preview = time.time()

    def _poll_trigger(self, frame: np.ndarray, t_capture: float) -> tuple[np.ndarray | None, bool]:
        """Was a part captured in this frame? → (image of the part or None, manual capture?)"""
        active = self.mode in ("learning", "inspecting")
        part = None
        if self.camera.continuous:
            if self.trigger.update(frame) and active:
                part = frame
            if self.cam.path is not None and self.mode == "learning":
                with self.lock:
                    self.cam.track(frame, t_capture)
            if self._burst is not None:
                self._continue_burst(frame)
        elif (
            isinstance(self.camera, FolderSource)
            and self.mode == "inspecting"
            and time.time() - self._last_folder_step > self.folder_interval
        ):
            part = self.camera.next_part()  # image folder: one image per interval
            self._last_folder_step = time.time()
        manual = self._manual.is_set()
        if manual:
            self._manual.clear()
            if isinstance(self.camera, FolderSource) and self.mode == "learning":
                frame = self.camera.next_part()
            part = frame if active else None
        return part, manual

    def _on_part(self, frame: np.ndarray, manual: bool) -> None:
        """A part was triggered: during inspection collect further shots while it passes."""
        if self.mode == "inspecting" and self.camera.continuous:
            self._burst = {"shots": [frame], "n": self.shots_per_part}
        else:
            self._submit([frame], manual)

    def _continue_burst(self, frame: np.ndarray) -> None:
        b = self._burst
        if b is None:
            return
        if self.trigger.part_complete and len(b["shots"]) < b["n"]:
            b["shots"].append(frame)
        if len(b["shots"]) >= b["n"] or not self.trigger.part_complete:
            self._burst = None
            self._submit(b["shots"], False)

    def _submit(self, shots: list[np.ndarray], manual: bool) -> None:
        """Hands a captured part to the inspection thread."""
        try:
            self._jobs.put_nowait((shots, manual))
        except queue.Full:
            if self.mode == "inspecting":
                self.results.add_skipped(self.model.name if self.model else "")
            self.log(
                "warn",
                "Inspection cannot keep up – part not inspected. Slow the belt down or increase "
                "the distance between the parts.",
            )

    def _worker(self) -> None:
        while not self._stop.is_set():
            try:
                shots, manual = self._jobs.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                self._handle_part(shots, manual)
            except Exception as e:
                logging.getLogger("qc").exception("Inspection failed")
                self.log("error", f"Inspection of a part failed: {e}")

    def _handle_part(self, shots: list[np.ndarray], manual: bool = False) -> None:
        """Teach-in or inspection of one captured part (worker thread)."""
        with self.lock:
            if self.mode == "learning":
                if self.cam.path is not None:  # the path part is never a reference
                    if not self.camera.continuous or manual:  # stopped belt: zoom from the still part
                        self.cam.finish_path(None)
                        self.log("info", "Camera zoomed in on the part – capture the part again.")
                    return
                self._add_reference(shots[0])
                return
            model = self.model if self.mode == "inspecting" else None
        if model is not None:
            self._inspect(model, shots)

    def manual_capture(self) -> None:
        if self.mode not in ("learning", "inspecting"):
            raise ValueError("Manual capture is only available while teaching in or inspecting.")
        self._manual.set()

    # ------------------------------------------------------------------ preview
    def _update_preview(self, frame: np.ndarray) -> None:
        img = resize_to_width(frame, PREVIEW_WIDTH).copy()
        if not self.cam.draw_focus(img, frame) and self.camera.continuous:
            self.trigger.draw(img)
        label = PREVIEW_LABEL[self.mode]
        if self.mode == "learning":
            label += f" {self.teach.count}/{self.cfg.target_reference_count}"
        cv2.rectangle(img, (0, img.shape[0] - 26), (img.shape[1], img.shape[0]), (0, 0, 0), cv2.FILLED)
        cv2.putText(
            img,
            f"{label}   {self.fps:4.1f} fps",
            (8, img.shape[0] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(self.cfg.ui.preview_quality)])
        if ok:
            self.preview_jpeg = buf.tobytes()

    # ------------------------------------------------------------------ camera setup
    def start_camera_setup(self) -> None:
        """Step 1 of the automatic camera setup: ONE good part lies still under the camera."""
        with self.lock:
            if self.mode != "idle":
                raise ValueError("Stop the current mode first.")
            if not self.camera.can_control:
                raise ValueError("This camera source cannot be adjusted automatically (Pi camera or simulator only).")
            self.mode = "camera_setup"
        threading.Thread(target=self._run_camera_setup, name="qc-camera-setup", daemon=True).start()

    def _run_camera_setup(self) -> None:
        if isinstance(self.camera, SimulatorSource):
            self.camera.hold_part(True)
        try:
            prof = self.cam.run()
            self.log(
                "ok",
                f"Camera adjusted: exposure {prof['exposure_us'] / 1000:.2f} ms, gain {prof['gain']:.2f}"
                + (f", focus {prof['lens_position']:.2f} dpt" if prof["lens_position"] else "")
                + ". Next: let the parts run – the first one measures the belt direction and sets the zoom.",
            )
            for note in prof["notes"]:
                self.log("warn", note)
        except Exception as ex:
            self.log("error", f"Automatic camera setup failed: {ex}")
            self.cam.apply(self.model.camera_profile if self.model else None, force=True)
        finally:
            if isinstance(self.camera, SimulatorSource):
                self.camera.hold_part(False)
            with self.lock:
                if self.mode == "camera_setup":
                    self.mode = "idle"

    def cancel_camera_setup(self) -> None:
        with self.lock:
            self.cam.pending_profile, self.cam.status = None, None
            if self.mode == "camera_setup":
                self.mode = "idle"
            self.cam.apply(self.model.camera_profile if self.model else None, force=True)

    def set_focus_assist(self, on: bool) -> None:
        self.cam.set_focus_assist(on)

    def set_belt_axis(self, axis: str) -> None:
        """Belt direction in the image – how the camera is mounted: "x" = left ↔ right (trigger bars
        vertical), "y" = top ↕ bottom (camera turned by 90°, bars horizontal). Saved in config.json."""
        if axis not in ("x", "y"):
            raise ValueError("Belt direction must be x (left ↔ right) or y (top ↕ bottom).")
        with self.lock:
            if self.mode != "idle":
                raise ValueError("Stop the current mode first.")
            self.cam.base_axis = axis
            model = self.model
            prof = model.camera_profile if model is not None else None
            if model is not None and prof and prof.get("axis") not in (None, axis):
                model.camera_profile = {**prof, "axis": axis}
                model.save_meta()
            self.cam.apply(model.camera_profile if model else None, force=True)
        text = "left ↔ right (bars vertical)" if axis == "x" else "top ↕ bottom (bars horizontal)"
        self._save_trigger_setting("axis", axis, f"Belt direction in the image: {text}")
        if model is not None:
            self.log("warn", f"If the camera was turned: teach “{model.name}” in again (references, zoom).")

    def set_trigger_mode(self, mode: str) -> None:
        """When the image is taken: "pass" = as the part crosses the image centre (continuous belt),
        "still" = once it rests in the centre band (stop-and-go belt). Saved in config.json."""
        if mode not in ("pass", "still"):
            raise ValueError("Capture mode must be pass (part passes through) or still (part stops in the centre).")
        with self.lock:
            if self.mode != "idle":
                raise ValueError("Stop the current mode first.")
            self.cfg.trigger.mode = mode
            self.trigger.armed, self.trigger.prev_pos, self.trigger.moving_captures = True, None, 0
            self.trigger.track.clear()
        text = "while the part passes the centre" if mode == "pass" else "when the part rests in the centre"
        self._save_trigger_setting("mode", mode, f"Capture {text}")

    def _save_trigger_setting(self, key: str, value: str, text: str) -> None:
        try:
            saved = self.cfg.persist("trigger", key, value)
        except OSError as e:
            saved = False
            self.log("warn", f"Setting could not be saved in the configuration: {e}")
        self.log("ok", text + (" – saved." if saved else "."))

    # ------------------------------------------------------------------ teach-in
    def start_learning(self, name: str, auto_finish: bool = False) -> None:
        """``auto_finish``: create the model automatically once the target number of parts is reached."""
        with self.lock:
            if self.mode != "idle":
                raise ValueError("Stop the current mode first.")
            self.teach.start(name, auto_finish)
            self.mode = "learning"
            self.cam.begin_teach_in()
            if self.cam.path is not None and not self.camera.continuous:
                self.cam.finish_path(None)
            if isinstance(self.camera, SimulatorSource):
                self.camera.force_good = True  # only good parts on the conveyor while learning
        self.log(
            "info",
            f"Learning mode started for “{self.teach.name}” – please feed "
            f"{self.cfg.target_reference_count} good parts.",
        )

    def _add_reference(self, frame: np.ndarray) -> None:
        if not self.teach.add(frame):
            self.log("warn", "Reference image discarded: no complete part detected.")
            return
        self.log("info", f"Reference image {self.teach.count} captured.")
        if self.teach.complete:
            self.finish_learning()

    def undo_reference(self, index: int | None = None) -> None:
        with self.lock:
            i = self.teach.remove(index)
        self.log("info", f"Reference image {i + 1} removed.")

    def cancel_learning(self) -> None:
        with self.lock:
            self.mode = "idle"
            self.teach.clear()
            self.cam.end_teach_in()
            self.cam.apply(self.model.camera_profile if self.model else None)
            if isinstance(self.camera, SimulatorSource):
                self.camera.force_good = False
        self.log("info", "Learning mode cancelled.")

    def finish_learning(self) -> None:
        with self.lock:
            if self.mode != "learning":
                raise ValueError("Learning mode is not active.")
            if self.teach.count < MIN_REFERENCES:
                raise ValueError(f"At least {MIN_REFERENCES} reference images required (currently {self.teach.count}).")
            name, frames = self.teach.name, list(self.teach.captures)
            self.mode = "training"
            if isinstance(self.camera, SimulatorSource):
                self.camera.force_good = False
        threading.Thread(target=self._train, args=(name, frames), name="qc-training", daemon=True).start()

    def _train(self, name: str, frames: list[np.ndarray]) -> None:
        try:
            model = QCModel.train(name, frames, self.cfg.localization, self.cfg.method, self.mm_per_px)
            prof = dict(self.cam.profile or {"auto": False, "crop": list(self.camera.crop)})
            edge = self.teach.edge_width()
            if edge is not None:
                prof["edge_width_px"] = round(edge, 2)
                prof["sharp"] = edge <= self.cfg.autosetup.edge_width_max_px
            model.camera_profile = prof
            # a new teach-in starts fresh (no accepted parts of the old version); the old one can be restored
            model.save(self.cfg.storage.models_dir, keep_previous=True, keep_side_data=False)
        except Exception as e:
            with self.lock:
                self.mode = "learning"
            self.log("error", f"Training failed: {e}")
            return
        with self.lock:
            self._set_model(model)
            self.teach.clear()
            self.cam.end_teach_in()
            self.mode = "idle"
        rep = model.report
        msg = f"Model “{name}” created ({rep['references_used']} references, {rep['train_time_s']} s)."
        if rep["references_skipped"]:
            msg += f" {len(rep['references_skipped'])} image(s) without a complete part skipped."
        self.log("ok", msg)
        for w in model.warnings():
            self.log("warn", w)
        if prof.get("sharp") is False:
            self.log(
                "warn",
                f"The image is not sharp (edges {prof['edge_width_px']} px wide, should be ≤ "
                f"{self.cfg.autosetup.edge_width_max_px}). Focus the lens once (focus assistant), "
                "then teach in again – detection is less precise this way.",
            )

    # ------------------------------------------------------------------ inspection
    def start_inspection(self, slug: str | None = None) -> None:
        if slug:
            self.select_model(slug)
        with self.lock:
            if self.model is None:
                raise ValueError("No model selected.")
            if self.mode != "idle":
                raise ValueError("Stop the current mode first.")
            self.mode = "inspecting"
        self.log("info", f"Inspection started with model “{self.model.name}”.")

    def stop_inspection(self) -> None:
        with self.lock:
            self._burst = None
            if self.mode == "inspecting":
                self.mode = "idle"
                self.log("info", "Inspection stopped.")

    def _inspect(self, model: QCModel, shots: list[np.ndarray]) -> dict:
        """Inspects one part (outside the lock – the UI stays responsive) and records the result."""
        ic = self.cfg.inspection
        res, idx = inspect_part(model, shots, ic.confirm_margin, ic.tie_is_nok)
        if isinstance(self.camera, SimulatorSource):
            res.ground_truth = self.camera.current_defect or "ok"
        elif isinstance(self.camera, FolderSource):
            res.ground_truth = self.camera.current_file.name
        with self.lock:
            entry = self.results.add(res, shots[idx])
            if res.status != "NO_PART":
                self.drift.update(res.measurements)
        return entry

    # ------------------------------------------------------------------ models
    def _set_model(self, model: QCModel | None) -> None:
        self.model = model
        self.drift.reset(model.baseline if model else None)
        if model is not None:
            self.cam.apply(model.camera_profile)

    def select_model(self, slug: str) -> None:
        path = model_path(self.cfg.storage.models_dir, slug)
        if path is None:
            raise ValueError("Model not found.")
        model = QCModel.load(path)
        with self.lock:
            if self.mode != "idle":
                raise ValueError("Stop the current mode first.")
            self._set_model(model)
        self.log("info", f"Model “{model.name}” loaded.")

    def delete_model(self, slug: str) -> None:
        path = model_path(self.cfg.storage.models_dir, slug)
        if path is None:
            raise ValueError("Model not found.")
        with self.lock:
            if self.improve.running:
                raise ValueError("A model update is running – try again in a moment.")
            if self.model is not None and self.model.slug == slug:
                if self.mode == "inspecting":
                    raise ValueError("The active model cannot be deleted while inspection is running.")
                self._set_model(None)
            shutil.rmtree(path)
        self.log("info", f"Model “{slug}” deleted.")

    def set_sensitivity(self, slug: str, values: dict[str, float]) -> dict[str, float]:
        path = model_path(self.cfg.storage.models_dir, slug)
        if path is None:
            raise ValueError("Model not found.")
        with self.lock:
            model = self.model if self.model is not None and self.model.slug == slug else QCModel.load(path)
            model.set_sensitivity(values)
        self.log(
            "info", f"Sensitivity of “{model.name}”: " + ", ".join(f"{k} ×{v:g}" for k, v in model.sensitivity.items())
        )
        return model.sensitivity

    def list_models(self) -> list[dict]:
        models = list_models(self.cfg.storage.models_dir)
        for m in models:
            path = Path(self.cfg.storage.models_dir) / m["slug"]
            store = LearningStore(path)
            m.update(waiting=store.count(PENDING), known_bad=store.count(KNOWN_BAD), has_previous=has_previous(path))
        return models

    # ------------------------------------------------------------------ results
    def feedback(self, rid: str, verdict: str) -> dict:
        """Supervisor confirms/corrects a result. "OK" on a NOK result = the part is within the
        company's tolerance → it is accepted and learned with the next model update."""
        with self.lock:
            outcome = self.results.feedback(rid, verdict)
            entry = self.results.entry(rid)
            learning = self.improve.on_feedback(self.model, entry, self.results.frames.get(entry["image_id"]), outcome)
        text = "accepted as good (within tolerance)" if "waiting" in learning and outcome == "false_alarm" else None
        self.log(
            "warn" if outcome in ("false_alarm", "missed") else "info",
            f"Supervisor feedback for #{rid}: {text or OUTCOME_TEXT[outcome]}",
        )
        if learning.pop("start_update", False):
            self.improve.start(None, manual=False)
        return {"outcome": outcome, **learning}

    def archive_results(self, label: str = "") -> dict:
        """Batch finished: pack all results into a ZIP, empty the results folder, reset counters."""
        with self.lock:  # results are written under this lock too → no half-written files
            if self.mode == "inspecting":
                raise ValueError("Please stop the inspection first.")
            if not label.strip() and self.model is not None:
                label = self.model.name
            info = qa.create_archive(self.cfg.storage.results_dir, self.cfg.storage.archive_dir, label)
            self.results.reset()
            self.trigger.moving_captures = 0
        s = info["summary"]
        self.log(
            "ok",
            f"Results archived as {info['file']} ({s['inspected']} parts, {s['nok']} NOK) – results folder cleared.",
        )
        return info

    def reset_counts(self) -> None:
        with self.lock:
            self.results.reset()
            self.trigger.moving_captures = 0

    def set_sim(
        self,
        lighting: str | None = None,
        defect_rate: float | None = None,
        part_type: str | None = None,
        any_angle: bool | None = None,
    ) -> None:
        cam = self.camera
        if not isinstance(cam, SimulatorSource):
            raise ValueError("Only available in the simulator.")
        with self.lock:
            if defect_rate is not None:
                cam.defect_rate = float(np.clip(defect_rate, 0, 1))
            if part_type in PART_TYPES:
                cam.part_type = part_type
            if any_angle is not None:
                cam.cfg.sim_any_angle = bool(any_angle)
            if lighting in ("front", "back"):
                cam.set_lighting(lighting)
            else:
                cam.respawn()

    # ------------------------------------------------------------------ status
    def supervisor_notices(self) -> list[dict]:
        """Short plain-language hints for the operator screen (no technical details)."""
        out: list[dict] = []
        if self.camera.continuous and self.fps == 0 and self.last_frame is None:
            out.append({"level": "error", "text": "No camera image. Check the camera cable and restart the device."})
        if self.model is None and self.mode == "idle":
            out.append({"level": "info", "text": "Select a part, or teach in a new part."})
        if self.model is not None:
            keys = {r["key"] for r in self.drift.state()["rows"] if r["warning"]}
            out += [{"level": "warn", "text": text} for k, text in DRIFT_HINTS.items() if k in keys]
            if (self.model.camera_profile or {}).get("sharp") is False:
                out.append(
                    {
                        "level": "warn",
                        "text": "The camera image is not sharp – the lens has to be focused "
                        "once (setup technician). Detection is less precise until then.",
                    }
                )
        if self.cfg.trigger.mode == "still" and self.trigger.moving_captures:
            out.append(
                {
                    "level": "warn",
                    "text": f"{self.trigger.moving_captures} part(s) did not stop in the image centre and were "
                    "photographed while moving. Check where the belt stops, or switch the capture mode to "
                    "“part passes through” (setup area).",
                }
            )
        out += self.improve.notices(self.model)
        if self.results.skipped:
            out.append(
                {
                    "level": "error",
                    "text": f"{self.results.skipped} part(s) could not be inspected because the "
                    "system could not keep up. Slow the belt down.",
                }
            )
        return out

    def status(self) -> dict:
        m, fa, r = self.model, self.cam.focus, self.results
        return {
            "mode": self.mode,
            "camera": self.camera.name,
            "camera_info": getattr(self.camera, "info", ""),
            "fps": round(self.fps, 1),
            "armed": self.trigger.armed,
            "trigger_mode": self.cfg.trigger.mode,
            "continuous": self.camera.continuous,
            "model": m.summary() if m else None,
            "learning": {
                "name": self.teach.name,
                "count": self.teach.count,
                "target": self.teach.target,
                "minimum": MIN_REFERENCES,
                "auto_finish": self.teach.auto_finish,
                "path_pending": self.cam.path is not None,
            },
            "camera_setup": self.cam.status,
            "camera_controls": self.cam.controllable,
            "camera_state": {**self.cam.state(), "auto": bool(m and (m.camera_profile or {}).get("auto"))},
            "mm_per_px": self.mm_per_px,
            "queue": self._jobs.qsize(),
            "skipped": r.skipped,
            "notices": self.supervisor_notices(),
            "setup_pin_required": bool(self.cfg.ui.setup_pin),
            "focus_assist": {"value": round(fa["value"], 1), "best": round(fa["best"], 1)} if fa else None,
            "counts": r.counts,
            "feedback": r.feedback_counts,
            "last_result": r.last_result,
            "messages": list(self.messages)[:12],
            "improvement": self.improve.state(),
            "settings": {
                "shots_per_part": self.shots_per_part,
                "rotation_search": self.cfg.localization.rotation_search,
            },
            "drift": self.drift.state(),
            "sim": {
                "lighting": self.camera.cfg.sim_lighting,
                "defect_rate": self.camera.defect_rate,
                "part_type": self.camera.part_type,
                "any_angle": self.camera.cfg.sim_any_angle,
            }
            if isinstance(self.camera, SimulatorSource)
            else None,
        }
