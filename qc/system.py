"""Process control: camera loop, conveyor trigger, learning and inspection mode.

The ``QCSystem`` class is independent of the user interface; the web UI
(``webapp.py``) only calls its methods. This makes it easy to add a local UI
(e.g. a touch display) or a PLC connection later on.
"""
from __future__ import annotations

import csv
import itertools
import threading
import time
from collections import deque
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from . import archive as qa
from . import model as qm
from .alignment import detect_part, resize_to_width, to_gray
from .camera import CameraSource, FolderSource, SimulatorSource
from .config import AppConfig, LocalizationConfig, TriggerConfig


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


class QCSystem:
    PREVIEW_WIDTH = 640

    def __init__(self, cfg: AppConfig, camera: CameraSource):
        self.cfg = cfg
        self.camera = camera
        self.trigger = PartTrigger(cfg.trigger, cfg.localization)
        self.lock = threading.RLock()
        self.mode = "idle"                       # idle | learning | training | inspecting
        self.model: qm.QCModel | None = None
        self.learn_name = ""
        self.learn_frames: list[np.ndarray] = []
        self.learn_thumbs: list[bytes] = []
        self.history: deque[dict] = deque(maxlen=cfg.storage.history_size)
        self.images: dict[str, tuple[bytes, bytes]] = {}
        self.counts = {"OK": 0, "NOK": 0, "NO_PART": 0}
        self.messages: deque[dict] = deque(maxlen=30)
        self.last_result: dict | None = None
        self.preview_jpeg: bytes | None = None
        self.last_frame: np.ndarray | None = None
        self.fps = 0.0
        self._ids = itertools.count(1)
        self._stop = threading.Event()
        self._manual = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_folder_step = 0.0
        self.folder_interval = 1.0
        Path(cfg.storage.models_dir).mkdir(parents=True, exist_ok=True)
        Path(cfg.storage.results_dir).mkdir(parents=True, exist_ok=True)
        self.log("info", f"System started – camera: {camera.name}")

    # --------------------------------------------------------------- helpers
    def log(self, level: str, text: str) -> None:
        self.messages.appendleft({"t": datetime.now().strftime("%H:%M:%S"), "level": level, "text": text})

    @property
    def is_sim(self) -> bool:
        return isinstance(self.camera, SimulatorSource)

    @property
    def is_folder(self) -> bool:
        return isinstance(self.camera, FolderSource)

    # ------------------------------------------------------------------ loop
    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, name="qc-loop", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3)
        self.camera.close()

    def _loop(self) -> None:
        t_last, n = time.time(), 0
        last_preview = 0.0
        while not self._stop.is_set():
            frame = self.camera.read()
            if frame is None:
                time.sleep(0.05)
                continue
            self.last_frame = frame
            fire = False
            if self.camera.continuous:
                fire = self.trigger.update(frame) and self.mode in ("learning", "inspecting")
            elif self.mode == "inspecting" and time.time() - self._last_folder_step > self.folder_interval:
                # Image folder: in inspection mode, automatically one image per interval
                frame = self.camera.next_part()
                self._last_folder_step = time.time()
                fire = frame is not None
            if self._manual.is_set():
                self._manual.clear()
                if self.is_folder and self.mode == "learning":
                    frame = self.camera.next_part()
                fire = frame is not None
            if fire:
                self._handle_part(frame)

            n += 1
            if time.time() - t_last >= 1.0:
                self.fps = n / (time.time() - t_last)
                t_last, n = time.time(), 0
            if time.time() - last_preview > 0.08:
                self._update_preview(frame)
                last_preview = time.time()

    def _update_preview(self, frame: np.ndarray) -> None:
        img = resize_to_width(frame, self.PREVIEW_WIDTH).copy()
        if self.camera.continuous:
            self.trigger.draw(img)
        label = {"idle": "READY", "learning": f"LEARNING {len(self.learn_frames)}/{self.cfg.target_reference_count}",
                 "training": "TRAINING ...", "inspecting": "INSPECTION ACTIVE"}[self.mode]
        cv2.rectangle(img, (0, img.shape[0] - 26), (img.shape[1], img.shape[0]), (0, 0, 0), cv2.FILLED)
        cv2.putText(img, f"{label}   {self.fps:4.1f} fps", (8, img.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 75])
        if ok:
            self.preview_jpeg = buf.tobytes()

    def _handle_part(self, frame: np.ndarray) -> None:
        with self.lock:
            if self.mode == "learning":
                self._add_reference(frame)
            elif self.mode == "inspecting" and self.model is not None:
                self._inspect(frame)

    # --------------------------------------------------------- learning mode
    def start_learning(self, name: str) -> None:
        name = name.strip()
        if not name:
            raise ValueError("Please enter a name for the part type.")
        with self.lock:
            self.mode = "learning"
            self.learn_name = name
            self.learn_frames, self.learn_thumbs = [], []
            if self.is_sim:
                self.camera.force_good = True    # only good parts on the conveyor while learning
            self.log("info", f"Learning mode started for “{name}” – please feed "
                             f"{self.cfg.target_reference_count} good parts.")

    def _add_reference(self, frame: np.ndarray) -> None:
        work = resize_to_width(frame, self.cfg.localization.work_width)
        det = detect_part(to_gray(work), self.cfg.localization)
        if det is None or not det.complete:
            self.log("warn", "Reference image discarded: no complete part detected.")
            return
        self.learn_frames.append(frame.copy())
        thumb = resize_to_width(frame, 200)
        self.learn_thumbs.append(cv2.imencode(".jpg", thumb, [cv2.IMWRITE_JPEG_QUALITY, 70])[1].tobytes())
        self.log("info", f"Reference image {len(self.learn_frames)} captured.")

    def undo_reference(self, index: int | None = None) -> None:
        with self.lock:
            if not self.learn_frames:
                return
            i = len(self.learn_frames) - 1 if index is None else index
            self.learn_frames.pop(i)
            self.learn_thumbs.pop(i)
            self.log("info", f"Reference image {i + 1} removed.")

    def cancel_learning(self) -> None:
        with self.lock:
            self.mode = "idle"
            self.learn_frames, self.learn_thumbs = [], []
            if self.is_sim:
                self.camera.force_good = False
            self.log("info", "Learning mode cancelled.")

    def finish_learning(self, method_overrides: dict | None = None) -> None:
        with self.lock:
            if self.mode != "learning":
                raise ValueError("Learning mode is not active.")
            if len(self.learn_frames) < qm.MIN_REFERENCES:
                raise ValueError(f"At least {qm.MIN_REFERENCES} reference images required "
                                 f"(currently {len(self.learn_frames)}).")
            frames, name = list(self.learn_frames), self.learn_name
            self.mode = "training"
            if self.is_sim:
                self.camera.force_good = False
        mcfg = self._method_cfg(method_overrides)
        threading.Thread(target=self._train, args=(name, frames, mcfg), daemon=True).start()

    def _method_cfg(self, overrides: dict | None):
        mcfg = replace(self.cfg.method)
        for k, v in (overrides or {}).items():
            if hasattr(mcfg, k) and v not in (None, ""):
                setattr(mcfg, k, v)
        if not mcfg.methods:
            raise ValueError("Select at least one inspection method.")
        return mcfg

    def _train(self, name, frames, mcfg) -> None:
        try:
            model = qm.QCModel.train(name, frames, self.cfg.localization, mcfg)
            model.save(self.cfg.storage.models_dir)
            with self.lock:
                self.model = model
                self.learn_frames, self.learn_thumbs = [], []
                self.mode = "idle"
            rep = model.report
            msg = f"Model “{name}” created ({rep['references_used']} references, {rep['train_time_s']} s)."
            if rep["references_skipped"]:
                msg += f" {len(rep['references_skipped'])} image(s) without a complete part skipped."
            self.log("ok", msg)
            if rep["references_flagged_nok"]:
                self.log("warn", "Warning: reference image(s) "
                         + ", ".join(str(i + 1) for i in rep["references_flagged_nok"])
                         + " deviate strongly – was a defective part taught in by mistake?")
            if rep.get("references_poorly_aligned"):
                self.log("warn", "Reference image(s) "
                         + ", ".join(str(i + 1) for i in rep["references_poorly_aligned"])
                         + " could not be aligned consistently – remove them and teach in again.")
            scatter = rep["methods"].get("geometry", {}).get("high_scatter_holes", [])
            if scatter:
                self.log("warn", f"{len(scatter)} hole(s) scatter strongly between the references – the model "
                                 "will be insensitive to position errors. Check the lighting/contrast and teach in again.")
            incons = rep["methods"].get("geometry", {}).get("inconsistent_references", [])
            if incons:
                self.log("warn", "Inconsistent hole count in reference(s) " + ", ".join(str(i + 1) for i in incons) + ".")
        except Exception as e:  # noqa: BLE001 - show the error to the supervisor
            with self.lock:
                self.mode = "learning"
            self.log("error", f"Training failed: {e}")

    # ------------------------------------------------------- inspection mode
    def select_model(self, slug: str) -> None:
        with self.lock:
            self.model = qm.QCModel.load(Path(self.cfg.storage.models_dir) / slug)
            self.log("info", f"Model “{self.model.name}” loaded.")

    def start_inspection(self, slug: str | None = None) -> None:
        if slug:
            self.select_model(slug)
        with self.lock:
            if self.model is None:
                raise ValueError("No model selected.")
            self.mode = "inspecting"
            self.log("info", f"Inspection started with model “{self.model.name}”.")

    def stop_inspection(self) -> None:
        with self.lock:
            if self.mode == "inspecting":
                self.mode = "idle"
                self.log("info", "Inspection stopped.")

    def manual_capture(self) -> None:
        if self.mode not in ("learning", "inspecting"):
            raise ValueError("Manual capture is only available in learning or inspection mode.")
        self._manual.set()

    def _inspect(self, frame: np.ndarray) -> None:
        res = self.model.inspect(frame)
        if self.is_sim:
            res.ground_truth = self.camera.current_defect or "ok"
        if self.is_folder:
            res.ground_truth = str(self.camera.current_file.name)
        rid = f"{next(self._ids):06d}"
        res.image_id = rid
        self.counts[res.status] = self.counts.get(res.status, 0) + 1
        d = res.to_dict()
        self.last_result = d
        self.history.appendleft(d)
        if res.overlay is not None:
            ov = cv2.imencode(".jpg", res.overlay, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()
            de = cv2.imencode(".jpg", res.detail, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()
            self.images[rid] = (ov, de)
            keep = {h["image_id"] for h in self.history}
            for k in [k for k in self.images if k not in keep]:
                del self.images[k]
        self._persist(res, frame)

    def _persist(self, res: qm.InspectionResult, frame: np.ndarray) -> None:
        st = self.cfg.storage
        day = Path(st.results_dir) / datetime.now().strftime("%Y-%m-%d")
        day.mkdir(parents=True, exist_ok=True)
        stem = f"{datetime.now().strftime('%H%M%S_%f')[:-3]}_{res.status}"
        save_img = (res.status == "NOK" and st.save_nok_images) or (res.status == "OK" and st.save_ok_images)
        if save_img and res.overlay is not None:
            cv2.imwrite(str(day / f"{stem}_marked.jpg"), res.overlay)
            cv2.imwrite(str(day / f"{stem}_original.png"), frame)
        log = day / "inspection_log.csv"
        new = not log.exists()
        with log.open("a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["time", "model", "result", "defects", "duration_ms", "image", "ground_truth"])
            w.writerow([res.timestamp, res.model, res.status,
                        " | ".join(f"{x.label}: {x.detail}" for x in res.defects),
                        f"{res.time_ms:.0f}", stem if save_img else "", res.ground_truth or ""])

    # ------------------------------------------------------------ management
    def delete_model(self, slug: str) -> None:
        import shutil

        path = Path(self.cfg.storage.models_dir) / slug
        if not (path / "meta.json").exists():
            raise ValueError("Model not found.")
        with self.lock:
            if self.model is not None and self.model.path and self.model.path.name == slug:
                if self.mode == "inspecting":
                    raise ValueError("The active model cannot be deleted while inspection is running.")
                self.model = None
            shutil.rmtree(path)
            self.log("info", f"Model “{slug}” deleted.")

    def retrain_model(self, slug: str, overrides: dict) -> None:
        mcfg = self._method_cfg(overrides)
        path = Path(self.cfg.storage.models_dir) / slug
        with self.lock:
            if self.mode in ("inspecting", "training"):
                raise ValueError("Please stop the inspection first.")
            prev = self.mode
            self.mode = "training"

        def run():
            try:
                m = qm.retrain(path, mcfg)
                with self.lock:
                    self.model = m
                self.log("ok", f"Model “{m.name}” retrained with {', '.join(mcfg.methods)}.")
            except Exception as e:  # noqa: BLE001
                self.log("error", f"Retraining failed: {e}")
            finally:
                with self.lock:
                    self.mode = prev

        threading.Thread(target=run, daemon=True).start()

    def set_sim(self, lighting: str | None = None, defect_rate: float | None = None, part_type: str | None = None,
                any_angle: bool | None = None) -> None:
        if not self.is_sim:
            raise ValueError("Only available in the simulator.")
        from .synthetic import PART_TYPES

        with self.lock:
            if defect_rate is not None:
                self.camera.defect_rate = float(np.clip(defect_rate, 0, 1))
            if part_type in PART_TYPES:
                self.camera.part_type = part_type
            if any_angle is not None:
                self.camera.cfg.sim_any_angle = bool(any_angle)
            if lighting in ("front", "back"):
                self.camera.set_lighting(lighting)
            else:
                self.camera._spawn()

    def archive_results(self, label: str = "") -> dict:
        """Batch finished: pack all results into a ZIP, empty the results folder, reset counters."""
        with self.lock:   # inspections also write under this lock → no half-written files
            if self.mode == "inspecting":
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
            self.history.clear()
            self.images.clear()
            self.last_result = None

    def status(self) -> dict:
        m = self.model
        return {
            "mode": self.mode, "camera": self.camera.name, "fps": round(self.fps, 1),
            "armed": self.trigger.armed, "continuous": self.camera.continuous,
            "model": m.summary() if m else None,
            "learning": {"name": self.learn_name, "count": len(self.learn_frames),
                         "target": self.cfg.target_reference_count},
            "counts": self.counts, "last_result": self.last_result,
            "messages": list(self.messages)[:12],
            "sim": {"lighting": self.camera.cfg.sim_lighting, "defect_rate": self.camera.defect_rate,
                    "part_type": self.camera.part_type, "any_angle": self.camera.cfg.sim_any_angle}
            if self.is_sim else None,
        }
