"""Recipe = everything the system knows about one part type.

A recipe consists of one or more *channels*, one per lighting:

* single-light mode: one channel ``main``
* dual-light mode:   ``back`` (backlight – exact silhouette, holes, outline) and
                     ``front`` (front light – surface, blind holes)

Each channel is an independent :class:`~qc.model.QCModel` (own alignment reference
and methods). The recipe combines their method results into one decision, merges
the defects of both channels and stores per-recipe settings such as the
sensitivity per method and references queued by supervisor feedback.

Folder layout::

    data/models/<name>/
        recipe.json                  name, channels, sensitivity
        channels/<channel>/          one QCModel (meta.json, arrays.npz, refs/, pending/)

Old single-model folders (``meta.json`` directly in the folder) are loaded as a
recipe with the channel ``main``.
"""
from __future__ import annotations

import json
import os
import shutil
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from . import methods as M
from .alignment import resize_to_width
from .config import LocalizationConfig, MethodConfig
from .model import InspectionResult, QCModel, TrainingError, decide, merge_defects, slugify
from .visualize import annotate

CHANNEL_LABELS = {"main": "", "front": "Front light", "back": "Backlight"}
GEOMETRY_KINDS = {"MISSING_HOLE", "BLIND_HOLE", "HOLE_POSITION", "HOLE_DIAMETER", "HOLE_SHAPE", "EXTRA_HOLE", "OUTLINE"}
METHOD_NAMES = ("geometry", "diff", "pca")


class Recipe:
    def __init__(self, name: str, channels: dict[str, QCModel], sensitivity: dict | None = None,
                 created: str | None = None):
        self.name = name
        self.channels = channels
        self.sensitivity = {m: 1.0 for m in METHOD_NAMES}
        self.sensitivity.update(sensitivity or {})
        self.created = created or datetime.now().isoformat(timespec="seconds")
        self.path: Path | None = None
        self.legacy = False
        self.collected_seen = 0                  # parts offered to the pool (reservoir sampling)
        self.self_learning_log: list[dict] = []
        # Camera settings found by the automatic camera setup for this part type
        # (exposure, gain, white balance, focus, zoom/crop, belt axis) – applied when the model is selected.
        self.camera_profile: dict | None = None

    # ------------------------------------------------------------ properties
    @property
    def mode(self) -> str:
        return "single" if list(self.channels) == ["main"] else "dual"

    @property
    def primary(self) -> str:
        """Channel shown first (thumbnails, first result image)."""
        return "front" if "front" in self.channels else next(iter(self.channels))

    @property
    def mm_per_px(self) -> float | None:
        return self.channels[self.primary].mm_per_px

    @property
    def calibration_id(self) -> str | None:
        return self.channels[self.primary].calibration_id

    @property
    def slug(self) -> str:
        return self.path.name if self.path else slugify(self.name)

    # --------------------------------------------------------------- training
    @classmethod
    def train(cls, name: str, captures: list[dict[str, np.ndarray]], loc_cfg: LocalizationConfig,
              method_cfgs: dict[str, MethodConfig], mm_per_px: float | None = None,
              calibration_id: str | None = None, sensitivity: dict | None = None) -> "Recipe":
        if not captures:
            raise TrainingError("No reference images.")
        channels = {}
        for ch in captures[0]:
            frames = [c[ch] for c in captures if c.get(ch) is not None]
            channels[ch] = QCModel.train(name, frames, loc_cfg, method_cfgs[ch], mm_per_px, calibration_id)
        return cls(name, channels, sensitivity)

    # ------------------------------------------------------------- inspection
    def inspect(self, captures: dict[str, np.ndarray], visualize: bool = True) -> InspectionResult:
        t0 = time.perf_counter()
        ts = datetime.now().isoformat(timespec="milliseconds")
        analyses = {}
        for ch, model in self.channels.items():
            frame = captures.get(ch)
            an = model.analyze(frame, self.sensitivity) if frame is not None else None
            if an is None:
                return InspectionResult("NO_PART", self.name, ts, (time.perf_counter() - t0) * 1000)
            for r in an.results:
                r.channel = ch if self.mode == "dual" else ""
            analyses[ch] = an

        all_results = [r for an in analyses.values() for r in an.results]
        ok, decisive = decide(all_results, self.channels[self.primary].method_cfg)

        per_channel: dict[str, list[M.Defect]] = {}
        for ch in analyses:
            defects = merge_defects([r for r in decisive if r.channel == (ch if self.mode == "dual" else "")])
            for d in defects:
                d.channel = ch if self.mode == "dual" else ""
            per_channel[ch] = defects
        if self.mode == "dual":
            per_channel = _merge_dual(per_channel, self.channels)
        shown = [d for ch in self._display_order() for d in per_channel.get(ch, [])]

        res = InspectionResult("OK" if ok else "NOK", self.name, ts, 0.0, shown, all_results,
                               measurements={ch: an.measurements for ch, an in analyses.items()},
                               worst=max((r.score for r in decisive), default=0.0))
        if visualize:
            for ch in self._display_order():
                an = analyses[ch]
                heats = [r.heatmap for r in decisive if r.heatmap is not None and r.channel == an.results[0].channel]
                heat = np.max(np.stack(heats), axis=0) if heats else None
                overlay, detail = annotate(an.aligned, ok, per_channel.get(ch, []), heat)
                res.images.append({"channel": ch if self.mode == "dual" else "main", "overlay": overlay, "detail": detail})
            res.overlay, res.detail = res.images[0]["overlay"], res.images[0]["detail"]
        res.time_ms = (time.perf_counter() - t0) * 1000
        return res

    def _display_order(self) -> list[str]:
        return [self.primary] + [c for c in self.channels if c != self.primary]

    # ------------------------------------------------------------ persistence
    def save(self, models_dir: str | Path) -> Path:
        path = Path(models_dir) / slugify(self.name) if self.path is None else self.path
        path.mkdir(parents=True, exist_ok=True)
        for ch, model in self.channels.items():
            model.save_to(path / "channels" / ch)
        if self.legacy:          # convert an old single-model folder to the recipe layout
            for n in ("meta.json", "arrays.npz"):
                (path / n).unlink(missing_ok=True)
            for n in ("refs", "learned"):
                if (path / n).exists():
                    shutil.rmtree(path / n)
            for n in ("pending", "collected", "known_bad"):
                target = path / "channels" / "main" / n
                if (path / n).exists() and not target.exists():
                    shutil.move(str(path / n), str(target))
            self.legacy = False
        self.path = path
        self.save_meta()
        return path

    def save_meta(self) -> None:
        meta = {"name": self.name, "created": self.created, "channels": list(self.channels),
                "sensitivity": self.sensitivity, "format": 3, "collected_seen": self.collected_seen,
                "self_learning_log": self.self_learning_log, "camera_profile": self.camera_profile}
        tmp = self.path / "recipe.json.tmp"
        tmp.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, self.path / "recipe.json")

    @classmethod
    def load(cls, path: str | Path) -> "Recipe":
        path = Path(path)
        if (path / "recipe.json").exists():
            meta = json.loads((path / "recipe.json").read_text(encoding="utf-8"))
            channels = {ch: QCModel.load(path / "channels" / ch) for ch in meta["channels"]}
            r = cls(meta["name"], channels, meta.get("sensitivity"), meta.get("created"))
            r.collected_seen = meta.get("collected_seen", 0)
            r.self_learning_log = meta.get("self_learning_log", [])
            r.camera_profile = meta.get("camera_profile")
        elif (path / "meta.json").exists():
            model = QCModel.load(path)
            r = cls(model.name, {"main": model}, None, model.created)
            r.legacy = True
        else:
            raise FileNotFoundError(f"No model in {path}")
        r.path = path
        return r

    def set_sensitivity(self, values: dict) -> None:
        for k, v in values.items():
            if k in METHOD_NAMES:
                self.sensitivity[k] = float(min(3.0, max(0.25, float(v))))
        if self.path is not None:
            if self.legacy:
                self.save(self.path.parent)
            else:
                self.save_meta()

    # ------------------------------------------------ feedback / retraining
    def _dir(self, ch: str, name: str) -> Path:
        model = self.channels[ch]
        return (model.path if model.path else self.path / "channels" / ch) / name

    def _pending_dir(self, ch: str) -> Path:
        return self._dir(ch, "pending")

    def _write(self, folder: str, captures: dict[str, np.ndarray], stem: str) -> None:
        for ch, model in self.channels.items():
            if captures.get(ch) is None:
                continue
            d = self._dir(ch, folder)
            d.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(d / f"{stem}.png"), resize_to_width(captures[ch], model.loc_cfg.work_width))

    def _count(self, folder: str) -> int:
        if self.path is None:
            return 0
        counts = [len(list(self._dir(ch, folder).glob("*.png"))) for ch in self.channels]
        return min(counts) if counts else 0

    def _read(self, folder: str) -> list[dict[str, np.ndarray]]:
        """All images of a side folder as captures {channel: image} (matched by file name)."""
        if self.path is None:
            return []
        stems = None
        for ch in self.channels:
            names = {p.stem for p in self._dir(ch, folder).glob("*.png")}
            stems = names if stems is None else stems & names
        return [{ch: cv2.imread(str(self._dir(ch, folder) / f"{st}.png")) for ch in self.channels}
                for st in sorted(stems or [])]

    def add_pending(self, captures: dict[str, np.ndarray]) -> int:
        """Queues a part rated NOK by mistake as an additional good (anchor) reference."""
        self._write("pending", captures, "fb_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
        return self.pending_count()

    def pending_count(self) -> int:
        return self._count("pending")

    # -------------------------------------------- self-learning (collected good parts)
    def add_collected(self, captures: dict[str, np.ndarray], max_pool: int, rng=None) -> str | None:
        """Offers a clearly good part to the pool. Reservoir sampling keeps a uniform sample
        over the whole production period instead of only the most recent parts."""
        rng = rng or np.random.default_rng()
        self.collected_seen += 1
        stem = "c_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        n = self._count("collected")
        if n >= max_pool:
            j = int(rng.integers(self.collected_seen))
            if j >= max_pool:
                self.save_meta()
                return None
            victims = sorted(self._dir(self.primary, "collected").glob("*.png"))
            self.remove_collected(victims[int(rng.integers(len(victims)))].stem)
        self._write("collected", captures, stem)
        self.save_meta()
        return stem

    def remove_collected(self, stem: str) -> bool:
        found = False
        for ch in self.channels:
            p = self._dir(ch, "collected") / f"{stem}.png"
            if p.exists():
                p.unlink()
                found = True
        return found

    def collected_count(self) -> int:
        return self._count("collected")

    def discard_collected(self) -> int:
        n = self.collected_count()
        for ch in self.channels:
            d = self._dir(ch, "collected")
            if d.exists():
                shutil.rmtree(d)
        return n

    def add_known_bad(self, captures: dict[str, np.ndarray], label: str, max_keep: int = 60) -> None:
        """Parts known to be defective (supervisor-confirmed NOK, self-test bad part). They are
        the safety check for self-learning: a new model must still detect them."""
        self._write("known_bad", captures, f"{label}_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
        for ch in self.channels:
            files = sorted(self._dir(ch, "known_bad").glob("*.png"), key=lambda p: p.stat().st_mtime)
            for p in files[:-max_keep]:
                p.unlink()

    def known_bad_count(self) -> int:
        return self._count("known_bad")

    def learn_collected(self, max_references: int = 80, growth_limit: float = 0.3,
                        seed: int = 0, commit: bool = True) -> tuple["Recipe | None", dict]:
        """Trains the collected good parts in – only if the new model passes the safety checks.

        1. Anchor references (+ queued feedback references) stay the basis of geometry,
           alignment and drift baseline; learned + collected parts refine diff/pca.
        2. Thresholds of the statistical methods must not loosen by more than
           ``growth_limit`` (more good references normally make them *tighter*;
           a clear increase means unusual/defective parts slipped into the pool).
        3. Every known defective part that the current model detects must still be detected.
        Returns (new recipe or None, report). ``commit=False``: an accepted candidate is not
        saved yet – call :meth:`commit_learned` (lets the caller do a last check first).
        """
        t0 = time.time()
        rng = np.random.default_rng(seed)
        collected = self._read("collected")
        report = {"time": datetime.now().isoformat(timespec="seconds"), "collected": len(collected),
                  "accepted": False, "reasons": [], "thresholds": [], "known_bad": None}
        if not collected:
            report["reasons"].append("No collected good parts.")
            return None, report
        pending = self._read("pending")
        new_channels = {}
        for ch, model in self.channels.items():
            anchors = model.reference_frames() + [c[ch] for c in pending]
            extra = model.learned_frames() + [c[ch] for c in collected]
            room = max(0, max_references - len(anchors))
            if len(extra) > room:
                idx = np.random.default_rng(seed).choice(len(extra), room, replace=False)
                extra = [extra[i] for i in sorted(idx)]
            new = QCModel.train(self.name, anchors + extra, model.loc_cfg, model.method_cfg,
                                model.mm_per_px, model.calibration_id, n_anchor=len(anchors))
            new.created = model.created
            new_channels[ch] = new
            report.setdefault("references", {})[ch] = {"anchor": len(anchors), "learned_before": model.learned_count(),
                                                       "learned_after": len(extra)}
            # --- check 2: thresholds
            for mname, (old_v, new_v, label) in _thresholds(model, new).items():
                change = (new_v - old_v) / old_v if old_v else 0.0
                report["thresholds"].append({"channel": ch, "method": mname, "what": label,
                                             "before": round(old_v, 4), "after": round(new_v, 4),
                                             "change_percent": round(change * 100, 1)})
                if change > growth_limit:
                    report["reasons"].append(f"{label} would loosen by {change * 100:.0f} % (limit "
                                             f"{growth_limit * 100:.0f} %) – the pool probably contains unusual parts.")
        cand = Recipe(self.name, new_channels, self.sensitivity, self.created)
        cand.camera_profile = self.camera_profile
        # --- check 3: known defective parts
        bad = self._read("known_bad")
        checked = still = 0
        for c in bad:
            if self.inspect(c, False).status == "NOK":
                checked += 1
                if cand.inspect(c, False).status == "NOK":
                    still += 1
        report["known_bad"] = {"available": len(bad), "checked": checked, "still_detected": still}
        if still < checked:
            report["reasons"].append(f"Only {still} of {checked} known defective parts would still be detected.")
        report["train_time_s"] = round(time.time() - t0, 1)
        report["accepted"] = not report["reasons"]
        if report["accepted"]:
            cand.path, cand.legacy = self.path, self.legacy
            cand.collected_seen = 0
            if commit:
                self.commit_learned(cand, report)
            return cand, report
        self.reject_learned(report)
        return None, report

    def commit_learned(self, cand: "Recipe", report: dict) -> None:
        """Saves an accepted self-learning candidate in place of this model."""
        cand.self_learning_log = (self.self_learning_log + [report])[-10:]
        if self.path:
            self.discard_collected()
            for ch in self.channels:                 # queued feedback references are now anchors
                d = self._pending_dir(ch)
                if d.exists():
                    shutil.rmtree(d)
            cand.save(self.path.parent)

    def reject_learned(self, report: dict) -> None:
        self.self_learning_log = (self.self_learning_log + [report])[-10:]
        if self.path:
            self.save_meta()

    def retrain(self, method_overrides: dict | None = None) -> "Recipe":
        """Retrains all channels: anchor references + queued feedback references (become anchors)
        + already learned good parts. Optionally with new method settings."""
        pending = self._read("pending")
        new_channels = {}
        for ch, model in self.channels.items():
            anchors = model.reference_frames() + [c[ch] for c in pending]
            cfg = model.method_cfg
            if method_overrides and not (self.mode == "dual" and ch == "back"):
                from dataclasses import replace
                cfg = replace(cfg, **method_overrides)
            new = QCModel.train(self.name, anchors + model.learned_frames(), model.loc_cfg, cfg,
                                model.mm_per_px, model.calibration_id, n_anchor=len(anchors))
            new.created = model.created
            new_channels[ch] = new
        r = Recipe(self.name, new_channels, self.sensitivity, self.created)
        r.camera_profile = self.camera_profile
        r.path, r.legacy = self.path, self.legacy
        r.collected_seen, r.self_learning_log = self.collected_seen, self.self_learning_log
        if self.path:
            for ch in self.channels:
                d = self._pending_dir(ch)
                if d.exists():
                    shutil.rmtree(d)
            r.save(self.path.parent)
        return r

    # ------------------------------------------------------------------ info
    def report(self) -> dict:
        return {ch: m.report for ch, m in self.channels.items()}

    def warnings(self) -> list[str]:
        out = []
        for ch, m in self.channels.items():
            rep = m.report
            prefix = f"[{CHANNEL_LABELS.get(ch) or ch}] " if self.mode == "dual" else ""
            if rep.get("references_flagged_nok"):
                out.append(prefix + "Reference image(s) " + ", ".join(str(i + 1) for i in rep["references_flagged_nok"])
                           + " deviate strongly – was a defective part taught in by mistake?")
            if rep.get("references_poorly_aligned"):
                out.append(prefix + "Reference image(s) " + ", ".join(str(i + 1) for i in rep["references_poorly_aligned"])
                           + " could not be aligned consistently – remove them and teach in again.")
            geo = rep.get("methods", {}).get("geometry", {})
            if geo.get("high_scatter_holes"):
                out.append(prefix + f"{len(geo['high_scatter_holes'])} hole(s) scatter strongly between the references – "
                           "the model will be insensitive to position errors. Check lighting/contrast and teach in again.")
            if geo.get("inconsistent_references"):
                out.append(prefix + "Inconsistent hole count in reference(s) "
                           + ", ".join(str(i + 1) for i in geo["inconsistent_references"]) + ".")
        return out

    def summary(self) -> dict:
        p = self.channels[self.primary]
        return {
            "name": self.name, "slug": self.slug, "created": self.created, "mode": self.mode,
            "channels": list(self.channels),
            "n_samples": p.n_samples,
            "methods": {ch: list(m.methods) for ch, m in self.channels.items()},
            "diff_representation": p.method_cfg.diff_representation,
            "pca_representation": p.method_cfg.pca_representation,
            "decision": p.method_cfg.decision,
            "sensitivity": self.sensitivity,
            "pending": self.pending_count(),
            "collected": self.collected_count(),
            "learned": p.learned_count(),
            "anchor": p.n_anchor,
            "known_bad": self.known_bad_count(),
            "last_self_learning": self.self_learning_log[-1] if self.self_learning_log else None,
            "mm_per_px": self.mm_per_px,
            "calibration_id": self.calibration_id,
            "camera_profile": self.camera_profile,
            "flagged": p.report.get("references_flagged_nok", []),
            "skipped": p.report.get("references_skipped", []),
        }


def _thresholds(old: QCModel, new: QCModel) -> dict[str, tuple[float, float, str]]:
    """Comparable thresholds of the statistical methods (higher = more tolerant)."""
    out = {}
    if "pca" in old.methods and "pca" in new.methods:
        out["pca"] = (old.methods["pca"].threshold, new.methods["pca"].threshold, "ML anomaly threshold")
    if "diff" in old.methods and "diff" in new.methods:
        o, n = old.methods["diff"], new.methods["diff"]
        out["diff"] = (float(np.median(o.std[o.region > 0])), float(np.median(n.std[n.region > 0])),
                       "reference comparison spread")
        out["diff_area"] = (o.min_area, n.min_area, "reference comparison min. area")
    return out


def _merge_dual(per_channel: dict[str, list[M.Defect]], channels: dict[str, QCModel]) -> dict[str, list[M.Defect]]:
    """Merges the defects of back- and front-light channel.

    * Geometry (holes, outline) is measured more exactly in backlight → geometry
      defects of the front channel are dropped if the back channel measures geometry.
    * Exception: "not drilled through" can only be seen in front light. Each such
      report replaces one "hole missing" of the back channel (backlight shows no
      opening for a blind hole).
    * Surface defects come only from the front channel anyway.
    """
    back, front = per_channel.get("back", []), per_channel.get("front", [])
    if "back" not in channels or "geometry" not in channels["back"].methods:
        return per_channel
    blind = [d for d in front if d.kind == "BLIND_HOLE"]
    back = list(back)
    for _ in blind:
        idx = next((i for i, d in enumerate(back) if d.kind == "MISSING_HOLE"), None)
        if idx is not None:
            back.pop(idx)
    front = [d for d in front if d.kind not in GEOMETRY_KINDS or d.kind == "BLIND_HOLE"]
    return {"back": back, "front": front}


def list_recipes(models_dir: str | Path) -> list[dict]:
    out = []
    for p in sorted(Path(models_dir).glob("*")):
        if not ((p / "recipe.json").exists() or (p / "meta.json").exists()):
            continue
        try:
            out.append(Recipe.load(p).summary())
        except Exception:  # noqa: BLE001 - e.g. a model being written right now; skip it this time
            continue
    return out
