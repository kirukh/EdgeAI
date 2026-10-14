"""Continuous improvement in operation: from the operator's decisions to new model versions.

``ModelImprovement`` is the component of ``QCSystem`` for it (the calculations are in learning.py):

    "within our tolerance" on a NOK result  → the part waits for the next update (accepted part)
    confirmed defective part                → test part for the release check of the next update
    after ``learning.update_after`` parts   → a new version is trained in the background (inspection
    (or "Update now")                         continues) and only released if the release check passes
    "Undo last update"                      → swaps back to the previous version (can itself be undone)
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from pathlib import Path

import numpy as np

from .config import AppConfig
from .learning import KNOWN_BAD, PENDING, LearningStore, release, update_model
from .model import QCModel, model_path, restore_previous


class ModelImprovement:
    def __init__(
        self,
        cfg: AppConfig,
        lock: threading.RLock,
        active_model: Callable[[], QCModel | None],
        set_model: Callable[[QCModel], None],
        log: Callable[[str, str], None],
    ):
        self.cfg, self.lock, self.log = cfg, lock, log
        self.active_model, self.set_model = active_model, set_model
        self.running = False  # a model update is being trained
        self.last: dict | None = None  # report of the last update (released or rejected)

    def _path(self, slug: str) -> Path:
        path = model_path(self.cfg.storage.models_dir, slug)
        if path is None:
            raise ValueError("Model not found.")
        return path

    # ------------------------------------------------------------------ operator decisions
    def on_feedback(self, model: QCModel | None, entry: dict, frame: np.ndarray | None, outcome: str) -> dict:
        """Accepted parts wait for the next update; confirmed defective parts become test parts
        for its release check. Called under the lock."""
        lc = self.cfg.learning
        if not lc.enabled or model is None or model.path is None or model.name != entry["model"]:
            return {}
        store = LearningStore(model.path)
        for kind in (PENDING, KNOWN_BAD):  # the supervisor changed their mind – drop the earlier decision
            stem = entry.pop(f"{kind}_stem", None)
            if stem:
                store.remove(kind, [stem])
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
        start = outcome == "false_alarm" and 0 < lc.update_after <= waiting and not self.running
        return {"waiting": waiting, "update_after": lc.update_after, "start_update": start}

    # ------------------------------------------------------------------ model updates
    def update_now(self, slug: str) -> None:
        """Button in the setup area: learn the waiting accepted parts now."""
        if LearningStore(self._path(slug)).count(PENDING) == 0:
            raise ValueError("No accepted parts are waiting for this model.")
        self.start(slug, manual=True)

    def start(self, slug: str | None, manual: bool) -> None:
        """Starts an update of the model ``slug`` (None = the active model) in the background."""
        with self.lock:
            if self.running:
                if manual:
                    raise ValueError("A model update is already running.")
                return
            model = self.active_model()
            if slug is not None and (model is None or model.slug != slug):
                model = QCModel.load(self._path(slug))
            if model is None or model.path is None:
                raise ValueError("Model not found.")
            self.running = True
        threading.Thread(target=self._run, args=(model,), name="qc-model-update", daemon=True).start()

    def _run(self, model: QCModel) -> None:
        """Trains and checks a new version in the background – inspection keeps running."""
        assert model.path is not None
        store = LearningStore(model.path)
        try:
            new, report = update_model(model, store)
            if new is not None:
                with self.lock:
                    active = self.active_model()
                    if active is not None and active.slug == model.slug:
                        new.sensitivity = dict(active.sensitivity)  # may have been changed meanwhile
                    release(new, store, report)
                    if active is not None and active.slug == model.slug:
                        self.set_model(new)
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
            self.last = report.to_dict()
        except Exception as e:  # a failed update must never stop the inspection
            logging.getLogger("qc").exception("Model update failed")
            self.log("error", f"Model update failed: {e}")
        finally:
            with self.lock:
                self.running = False

    def undo(self, slug: str) -> None:
        """Back to the version before the last update (can itself be undone the same way)."""
        path = self._path(slug)
        with self.lock:
            if self.running:
                raise ValueError("A model update is running – try again in a moment.")
            restored = restore_previous(path)
            active = self.active_model()
            if active is not None and active.slug == slug:
                restored.sensitivity = dict(active.sensitivity)
                self.set_model(restored)
        self.log("ok", f"Model “{restored.name}” restored to version {restored.version}.")

    # ------------------------------------------------------------------ accepted parts
    def waiting(self, slug: str) -> list[dict]:
        return [{"name": stem, **info} for stem, _, info in LearningStore(self._path(slug)).items(PENDING)]

    def discard(self, slug: str, name: str) -> None:
        """A part was accepted by mistake – it is not learned."""
        path = model_path(self.cfg.storage.models_dir, slug)
        if path is None or LearningStore(path).image_path(PENDING, name) is None:
            raise ValueError("Accepted part not found.")
        LearningStore(path).remove(PENDING, [name])
        self.log("info", f"Accepted part {name} of “{slug}” discarded – it will not be learned.")

    # ------------------------------------------------------------------ status
    def notices(self, model: QCModel | None) -> list[dict]:
        """Hint for the operator screen about accepted parts and updates of the active model."""
        if model is None or model.path is None or not self.cfg.learning.enabled:
            return []
        if self.running:
            return [
                {
                    "level": "info",
                    "text": "The model is being updated with the accepted parts – inspection "
                    "continues with the current version.",
                }
            ]
        last = self.last
        if last and not last["released"] and last["model"] == model.name:
            return [
                {
                    "level": "warn",
                    "text": "The last model update was rejected (it contradicted parts confirmed "
                    "as defective). The model is unchanged – inform the setup technician.",
                }
            ]
        waiting, after = LearningStore(model.path).count(PENDING), self.cfg.learning.update_after
        if waiting:
            when = f"after {after}" if after else "by the setup technician"
            return [{"level": "info", "text": f"{waiting} accepted part(s) waiting – the model learns them {when}."}]
        return []

    def state(self) -> dict:
        lc = self.cfg.learning
        return {"enabled": lc.enabled, "update_after": lc.update_after, "running": self.running, "last": self.last}
