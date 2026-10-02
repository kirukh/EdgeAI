"""Model side of the QC system: model management, self-learning, supervisor feedback, reference part check."""
from __future__ import annotations

import csv
import json
import threading
from datetime import datetime
from pathlib import Path

import numpy as np

from . import model as qm
from .recipe import Recipe, list_recipes




def _outcome_key(outcome: str) -> str:
    return "confirmed" if outcome.startswith("confirmed") else outcome



class ModelsMixin:
    """Mixin of :class:`qc.system.QCSystem` (uses its state: camera, cfg, lock, model, …)."""

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

    def restore_model(self, slug: str) -> None:
        """Undo the last update of a model (new teach-in, retraining or self-learning)."""
        path = Path(self.cfg.storage.models_dir) / slug
        with self.lock:
            if self._bg_learning == slug or self.mode == "training":
                raise ValueError("The model is being updated right now – try again in a moment.")
            if self.mode in ("inspecting", "selftest") and self.model is not None and self.model.slug == slug:
                raise ValueError("Stop the inspection first.")
            recipe = Recipe.restore_previous(path)
            if self.model is not None and self.model.slug == slug:
                self._set_model(recipe)
        self.log("ok", f"Model “{recipe.name}” restored to its previous version (the undone version can be "
                       "restored the same way).")

    def list_models(self) -> list[dict]:
        return list_recipes(self.cfg.storage.models_dir)
