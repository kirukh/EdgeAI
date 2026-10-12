"""Process control: camera loop, conveyor trigger, teach-in and inspection.

``QCSystem`` coordinates the components and owns the operating mode; the work is done by them:

    CameraSetup (camera_setup.py)  camera profiles, automatic camera setup, focus assistant
    TeachIn     (teach.py)         reference images of a new part type
    QCModel     (model.py)         training and inspection of one part type
    inspect_part (inspection.py)   several images per part, majority vote
    ResultStore (results.py)       history, counters, feedback, CSV logs
    learning.py                    continuous improvement from the operator's decisions
    DriftMonitor (drift.py)        image quality compared with the learning phase

Two threads do the work, so the live image and the trigger never wait for an inspection::

    camera thread:  frame → trigger → 3 shots ──┐ queue (max. 3 parts)
    worker thread:  ◄───────────────────────────┘ inspect_part() → results / drift monitor

If the worker cannot keep up (queue full), the part is logged as not inspected and the
supervisor gets a notice. The web UI (``qc/web``) only calls the public methods.
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
from .inspection import inspect_part
from .learning import KNOWN_BAD, PENDING, LearningStore, release, update_model
from .model import MIN_REFERENCES, QCModel, has_previous, list_models, model_path, restore_previous
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
        self._update_running = False  # a model update (learning.py) is being trained
        self.last_update: dict | None = None
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
            if self._update_running:
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
            learning = self._learn_from_feedback(self.results.entry(rid), outcome)
        text = "accepted as good (within tolerance)" if "waiting" in learning and outcome == "false_alarm" else None
        self.log(
            "warn" if outcome in ("false_alarm", "missed") else "info",
            f"Supervisor feedback for #{rid}: {text or OUTCOME_TEXT[outcome]}",
        )
        if learning.pop("start_update", False):
            self._start_update(None, manual=False)
        return {"outcome": outcome, **learning}

    def _learn_from_feedback(self, entry: dict, outcome: str) -> dict:
        """Accepted parts wait for the next update; confirmed defective parts become test parts
        for its release check. Called under the lock."""
        lc, model = self.cfg.learning, self.model
        if not lc.enabled or model is None or model.path is None or model.name != entry["model"]:
            return {}
        store = LearningStore(model.path)
        for kind in (PENDING, KNOWN_BAD):  # the supervisor changed their mind – drop the earlier decision
            stem = entry.pop(f"{kind}_stem", None)
            if stem:
                store.remove(kind, [stem])
        frame = self.results.frames.get(entry["image_id"])
        if frame is None:
            return {}
        info = {
            "image_id": entry["image_id"],
            "inspected_at": entry["timestamp"],
            "result": entry["status"],
            "defects": sorted({d["label"] for d in entry["defects"]}),
            "model_version": entry["model_version"],
        }
        if outcome == "false_alarm":
            entry[f"{PENDING}_stem"] = store.add(PENDING, frame, info, lc.max_waiting)
        elif outcome in ("missed", "confirmed_nok"):
            entry[f"{KNOWN_BAD}_stem"] = store.add(KNOWN_BAD, frame, info, lc.max_known_bad)
        waiting = store.count(PENDING)
        start = outcome == "false_alarm" and 0 < lc.update_after <= waiting and not self._update_running
        return {"waiting": waiting, "update_after": lc.update_after, "start_update": start}

    # ------------------------------------------------------------------ continuous improvement
    def update_model_now(self, slug: str) -> None:
        """Button in the setup area: learn the waiting accepted parts now."""
        path = model_path(self.cfg.storage.models_dir, slug)
        if path is None:
            raise ValueError("Model not found.")
        if LearningStore(path).count(PENDING) == 0:
            raise ValueError("No accepted parts are waiting for this model.")
        self._start_update(slug, manual=True)

    def _start_update(self, slug: str | None, manual: bool) -> None:
        with self.lock:
            if self._update_running:
                if manual:
                    raise ValueError("A model update is already running.")
                return
            model = self.model
            if slug is not None and (model is None or model.slug != slug):
                path = model_path(self.cfg.storage.models_dir, slug)
                model = QCModel.load(path) if path is not None else None
            if model is None or model.path is None:
                raise ValueError("Model not found.")
            self._update_running = True
        threading.Thread(target=self._run_update, args=(model,), name="qc-model-update", daemon=True).start()

    def _run_update(self, model: QCModel) -> None:
        """Trains and checks a new version in the background – inspection keeps running."""
        assert model.path is not None
        store = LearningStore(model.path)
        try:
            new, report = update_model(model, store)
            if new is not None:
                with self.lock:
                    active = self.model is not None and self.model.slug == model.slug
                    if active and self.model is not None:  # sensitivity may have been changed meanwhile
                        new.sensitivity = dict(self.model.sensitivity)
                    release(new, store, report)
                    if active:
                        self._set_model(new)
                self.log(
                    "ok",
                    f"Model “{model.name}” updated to version {new.version}: {report.accepted_parts} "
                    f"accepted part(s) learned, {report.known_bad_still_detected}/{report.known_bad_checked} "
                    "known defective parts still detected.",
                )
                if report.still_nok:
                    self.log(
                        "warn",
                        f"{len(report.still_nok)} accepted part(s) would still be rated NOK "
                        f"({'; '.join(report.still_nok[:3])}). Missing, extra and blind holes are always "
                        "defects; for other deviations lower the sensitivity of that method in the setup area "
                        "if such parts are acceptable.",
                    )
            else:
                self.log(
                    "error",
                    f"Model update of “{model.name}” rejected – the model is unchanged. " + " ".join(report.reasons),
                )
            self.last_update = report.to_dict()
        except Exception as e:  # a failed update must never stop the inspection
            logging.getLogger("qc").exception("Model update failed")
            self.log("error", f"Model update failed: {e}")
        finally:
            with self.lock:
                self._update_running = False

    def undo_update(self, slug: str) -> None:
        """Back to the version before the last update (can itself be undone the same way)."""
        path = model_path(self.cfg.storage.models_dir, slug)
        if path is None:
            raise ValueError("Model not found.")
        with self.lock:
            if self._update_running:
                raise ValueError("A model update is running – try again in a moment.")
            restored = restore_previous(path)
            if self.model is not None and self.model.slug == slug:
                restored.sensitivity = dict(self.model.sensitivity)
                self._set_model(restored)
        self.log("ok", f"Model “{restored.name}” restored to version {restored.version}.")

    def waiting_parts(self, slug: str) -> list[dict]:
        path = model_path(self.cfg.storage.models_dir, slug)
        if path is None:
            raise ValueError("Model not found.")
        return [{"name": stem, **info} for stem, _, info in LearningStore(path).items(PENDING)]

    def discard_waiting(self, slug: str, name: str) -> None:
        """A part was accepted by mistake – it is not learned."""
        path = model_path(self.cfg.storage.models_dir, slug)
        if path is None or LearningStore(path).image_path(PENDING, name) is None:
            raise ValueError("Accepted part not found.")
        LearningStore(path).remove(PENDING, [name])
        self.log("info", f"Accepted part {name} of “{slug}” discarded – it will not be learned.")

    def archive_results(self, label: str = "") -> dict:
        """Batch finished: pack all results into a ZIP, empty the results folder, reset counters."""
        with self.lock:  # results are written under this lock too → no half-written files
            if self.mode == "inspecting":
                raise ValueError("Please stop the inspection first.")
            if not label.strip() and self.model is not None:
                label = self.model.name
            info = qa.create_archive(self.cfg.storage.results_dir, self.cfg.storage.archive_dir, label)
            self.results.reset()
        s = info["summary"]
        self.log(
            "ok",
            f"Results archived as {info['file']} ({s['inspected']} parts, {s['nok']} NOK) – results folder cleared.",
        )
        return info

    def reset_counts(self) -> None:
        with self.lock:
            self.results.reset()

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
        out += self._improvement_notices()
        if self.results.skipped:
            out.append(
                {
                    "level": "error",
                    "text": f"{self.results.skipped} part(s) could not be inspected because the "
                    "system could not keep up. Slow the belt down.",
                }
            )
        return out

    def _improvement_notices(self) -> list[dict]:
        if self.model is None or self.model.path is None or not self.cfg.learning.enabled:
            return []
        if self._update_running:
            return [
                {
                    "level": "info",
                    "text": "The model is being updated with the accepted parts – inspection "
                    "continues with the current version.",
                }
            ]
        last = self.last_update
        if last and not last["released"] and last["model"] == self.model.name:
            return [
                {
                    "level": "warn",
                    "text": "The last model update was rejected (it contradicted parts confirmed "
                    "as defective). The model is unchanged – inform the setup technician.",
                }
            ]
        waiting, after = LearningStore(self.model.path).count(PENDING), self.cfg.learning.update_after
        if waiting:
            when = f"after {after}" if after else "by the setup technician"
            return [{"level": "info", "text": f"{waiting} accepted part(s) waiting – the model learns them {when}."}]
        return []

    def status(self) -> dict:
        m, fa, r = self.model, self.cam.focus, self.results
        return {
            "mode": self.mode,
            "camera": self.camera.name,
            "camera_info": getattr(self.camera, "info", ""),
            "fps": round(self.fps, 1),
            "armed": self.trigger.armed,
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
            "improvement": {
                "enabled": self.cfg.learning.enabled,
                "update_after": self.cfg.learning.update_after,
                "running": self._update_running,
                "last": self.last_update,
            },
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
