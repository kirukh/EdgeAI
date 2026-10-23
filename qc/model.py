"""Model = one taught-in part type: learning phase (``train``) and run phase (``inspect``).

A model contains the alignment reference, the learned state of all enabled inspection
methods, the sensitivity per method and the camera profile found by the automatic camera
setup. It is stored as a folder and can be selected again later::

    data/models/<name>/
        meta.json      metadata, settings, sensitivity, camera profile, training report
        arrays.npz     reference images/statistics of the methods
        refs/*.png     the reference images (working resolution)
"""

from __future__ import annotations

import json
import os
import re
import shutil
import time
from collections.abc import Collection
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from . import methods as M
from .alignment import (
    AlignedPart,
    PartDetection,
    align_part,
    default_canvas,
    detect_part,
    ncc,
    resize_to_width,
    to_gray,
)
from .config import LocalizationConfig, MethodConfig, _merge
from .visualize import annotate

MIN_REFERENCES = 3
METHOD_NAMES = ("geometry", "diff", "pca")
POOR_ALIGNMENT = 0.9  # normalised cross-correlation of a reference with the mean image
FORMAT = 4
# Data of the part type kept next to the model and carried over to new versions (learning.py)
SIDE_FOLDERS = ("pending", "known_bad")
PREVIOUS = "previous"  # the version before the last update (one-step undo)


class TrainingError(RuntimeError):
    """Training not possible (too few usable references, unsupported model) – the message is shown to the user."""

    pass


@dataclass
class InspectionResult:
    """Inspection result of one image (or the majority vote of several) with defects, scores and marked images."""

    status: str  # "OK" | "NOK" | "NO_PART"
    model: str
    timestamp: str
    time_ms: float
    defects: list[M.Defect] = field(default_factory=list[Any])
    method_results: list[M.MethodResult] = field(default_factory=list[Any])
    overlay: np.ndarray | None = None  # working image with markings
    detail: np.ndarray | None = None  # aligned part with heatmap
    image_id: str = ""
    model_version: int = 1
    ground_truth: str | None = None  # simulator/image folder only
    measurements: dict[str, Any] = field(default_factory=dict[str, Any])  # image quality (drift monitor)
    shots: list[str] = field(default_factory=list[Any])  # status of each shot (majority vote)
    worst: float = 0.0  # highest decisive score (picks the representative shot)

    @property
    def ok(self) -> bool:
        """True if the part is OK."""
        return self.status == "OK"

    def to_dict(self) -> dict[str, Any]:
        """Result as a plain dict (history, log, UI) – without the images."""
        return {
            "status": self.status,
            "model": self.model,
            "timestamp": self.timestamp,
            "time_ms": round(self.time_ms, 1),
            "image_id": self.image_id,
            "defects": [d.to_dict() for d in self.defects],
            "methods": [r.to_dict() for r in self.method_results],
            "ground_truth": self.ground_truth,
            "model_version": self.model_version,
            "has_image": self.overlay is not None,
            "shots": self.shots,
        }


def standalone_levels(cfg: MethodConfig) -> dict[str, float]:
    """Score above which a method decides NOK on its own (fusion rule)."""
    return {"geometry": cfg.geometry_standalone, "diff": cfg.diff_standalone, "pca": cfg.pca_standalone}


def decide(results: list[M.MethodResult], cfg: MethodConfig) -> tuple[bool, list[M.MethodResult]]:
    """Combines the method results → (ok, decisive results).

    "or":     every method with score ≥ 1 is decisive.
    "fusion": a method is decisive alone only above its standalone level; if at least two
              methods are ≥ 1 at the same time (they agree), all of them are decisive.
    """
    over = [r for r in results if r.score >= 1.0]
    levels = standalone_levels(cfg)
    if cfg.decision == "or" or len(over) >= 2:
        decisive = over
    else:
        decisive = [r for r in over if r.score >= levels.get(r.method, 1.0)]
    for r in results:
        r.standalone = 1.0 if cfg.decision == "or" else levels.get(r.method, 1.0)
        r.decisive = any(r is d for d in decisive)
    return not decisive, decisive


def merge_defects(results: list[M.MethodResult]) -> list[M.Defect]:
    """Defects of the decisive methods; reports of the same location are merged
    (geometry first – it gives the most specific description)."""
    shown: list[M.Defect] = []
    priority = {"geometry": 0, "diff": 1, "pca": 2}
    for r in sorted(results, key=lambda r: priority.get(r.method, 9)):
        for d in r.defects:
            cx, cy = d.bbox[0] + d.bbox[2] / 2, d.bbox[1] + d.bbox[3] / 2
            if any(_near(k.bbox, cx, cy) for k in shown):
                continue
            shown.append(d)
    return shown


def _near(bbox: tuple[int, int, int, int], cx: float, cy: float, pad: int = 8) -> bool:
    x, y, w, h = bbox
    return x - pad <= cx <= x + w + pad and y - pad <= cy <= y + h + pad


def measure(a: AlignedPart) -> dict[str, float]:
    """Image-quality measurements for drift monitoring (lighting, distance, focus)."""
    band = cv2.morphologyEx(a.filled_mask, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
    n = np.clip(a.norm, 0, 1.5)
    g = cv2.magnitude(cv2.Sobel(n, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(n, cv2.CV_32F, 0, 1, ksize=3))
    return {
        "contrast": float(abs(a.part_level - a.bg_level)),  # grey levels between part and belt
        "background": float(a.bg_level),
        "area": float(np.count_nonzero(a.filled_mask)),  # changes with camera distance/zoom
        "sharpness": float(np.percentile(g[band], 90)) if band.any() else 0.0,  # focus/motion blur
    }


def slugify(name: str) -> str:
    """File-system safe name of a part type (folder name of the model)."""
    s = re.sub(r"[^A-Za-z0-9_\-]+", "_", name.strip()).strip("_")
    return s or "model"


def _prepare(frame: np.ndarray, loc: LocalizationConfig) -> tuple[np.ndarray, PartDetection | None]:
    work = resize_to_width(frame, loc.work_width)
    return work, detect_part(to_gray(work), loc)


@dataclass
class FitResult:
    """One training pass over the prepared references (see ``QCModel._fit``)."""

    model: QCModel
    aligned: list[AlignedPart]
    consistency: list[float]  # similarity of every reference with the mean image
    suspects: dict[int, list[str]]  # reference index → why it stands out (leave-one-out)
    excludable: set[int]  # suspects found by a method that may decide alone (geometry, diff)
    self_nok: list[int]  # references the finished model itself rates NOK


def _prepare_all(
    frames: list[np.ndarray], loc: LocalizationConfig
) -> tuple[list[tuple[np.ndarray, PartDetection]], list[int], list[int]]:
    """→ (prepared images with a complete part, their frame indices, skipped frame indices)."""
    prepared, orig, skipped = [], [], []
    for i, f in enumerate(frames):
        work, det = _prepare(f, loc)
        if det is None or not det.complete:
            skipped.append(i)
        else:
            prepared.append((work, det))
            orig.append(i)
    return prepared, orig, skipped


def _screen(fit: FitResult, cfg: MethodConfig, accepted: set[int]) -> list[int] | None:
    """Reference screening → indices of the references to keep, or None (keep all).

    Too many suspects means the references as a whole are inconsistent (lighting, focus, wrong
    part type …): then nothing is excluded – the suspects are only reported. Parts the operator
    accepted are never excluded: they deviate on purpose.
    """
    n = len(fit.aligned)
    drop = fit.excludable - accepted
    if not cfg.screen_references or not drop:
        return None
    cap = max(1, int(n * cfg.screen_max_ratio))
    if len(drop) > cap or n - len(drop) < MIN_REFERENCES:
        return None
    return [k for k in range(n) if k not in drop]


def _training_report(
    fit: FitResult,
    orig: list[int],
    skipped: list[int],
    excluded: dict[int, list[str]],
    accepted: set[int],
    mm_per_px: float | None,
    seconds: float,
) -> dict[str, Any]:
    """Training report. All reference indices are indices into the taught-in frames (image number − 1)."""
    model = fit.model
    report: dict[str, Any] = {
        "references_used": len(fit.aligned),
        "references_skipped": skipped,
        "canvas": list(model.canvas),
        "polarity": model.polarity,
        "train_time_s": round(seconds, 2),
        "references_excluded": [{"index": i, "reasons": r} for i, r in sorted(excluded.items())],
        "references_suspect": [
            {"index": orig[k], "reasons": r} for k, r in sorted(fit.suspects.items()) if k not in accepted
        ],
        "references_accepted": [orig[k] for k in sorted(accepted)],
        "references_flagged_nok": [orig[k] for k in fit.self_nok if k not in accepted],
        "alignment_consistency_min": round(float(min(fit.consistency)), 3),
        "references_poorly_aligned": [orig[k] for k, c in enumerate(fit.consistency) if c < POOR_ALIGNMENT],
        "methods": {k: m.report for k, m in model.methods.items()},
        "baseline": model.baseline,
        "mm_per_px": mm_per_px,
    }
    geo = report["methods"].get("geometry")
    if geo is not None and mm_per_px:
        geo["hole_diameters_mm"] = [round(d * mm_per_px, 3) for d in geo["hole_diameters_px"]]
    return report


@dataclass
class ChannelAnalysis:
    """Aligned part and the results of all methods for one image (before the decision)."""

    aligned: AlignedPart
    results: list[M.MethodResult]
    measurements: dict[str, float]


class QCModel:
    """A taught part type: alignment reference, inspection methods, sensitivity, camera profile, versions."""

    def __init__(self, name: str, loc_cfg: LocalizationConfig, method_cfg: MethodConfig):
        self.name = name
        self.loc_cfg = loc_cfg
        self.method_cfg = method_cfg
        self.canvas: tuple[int, int] = (0, 0)
        self.ref_norm: np.ndarray | None = None
        self.methods: dict[str, M.Method] = {}
        self.created = datetime.now().isoformat(timespec="seconds")
        self.n_samples = 0
        self.polarity = "auto"
        self.report: dict[str, Any] = {}
        self.path: Path | None = None
        self.mm_per_px: float | None = None  # scale at training time (incl. zoom)
        self.baseline: dict[str, float] = {}  # drift monitoring: measurements of the references
        self.sensitivity: dict[str, float] = dict[str, Any].fromkeys(METHOD_NAMES, 1.0)
        # Camera settings found by the automatic camera setup (exposure, gain, white balance,
        # focus, zoom/crop, belt axis) – applied when the model is selected.
        self.camera_profile: dict[str, Any] | None = None
        self._ref_frames: list[np.ndarray] | None = None
        self._refs_dir: Path | None = None
        self._legacy = False  # loaded from the full version's folder layout
        # Continuous improvement (learning.py): reference indices accepted by the operator,
        # model version and the history of updates
        self.accepted: list[int] = []
        self.version = 1
        self.history: list[dict[str, Any]] = []

    @property
    def slug(self) -> str:
        """Folder name of the model."""
        return self.path.name if self.path else slugify(self.name)

    # ------------------------------------------------------------------ learning
    @classmethod
    def train(
        cls,
        name: str,
        frames: list[np.ndarray],
        loc_cfg: LocalizationConfig,
        method_cfg: MethodConfig,
        mm_per_px: float | None = None,
        accepted: Collection[int] = (),
    ) -> QCModel:
        """Learns a part type from images of good parts.

        ``accepted`` = indices of ``frames`` that the operator accepted as good although they
        deviate ("within our tolerance"); the methods tolerate their deviation locally.

        Reference screening (``method_cfg.screen_references``): after a first training pass every
        reference is compared with the *other* references (leave-one-out). References that stand
        out clearly – usually a defective part taught in by mistake – are left out and the model
        is trained again without them. Only geometry and the image comparison may exclude; an ML
        finding alone is only reported (ML scatters more on good parts).
        """
        t0 = time.time()
        prepared, orig, skipped = _prepare_all(frames, loc_cfg)
        acc = {k for k, o in enumerate(orig) if o in accepted}
        fit = cls._fit(name, prepared, loc_cfg, method_cfg, acc)
        excluded: dict[int, list[str]] = {}
        keep = _screen(fit, method_cfg, acc)
        if keep is not None:  # leave the conspicuous references out and train again
            excluded = {orig[k]: fit.suspects[k] for k in range(len(prepared)) if k not in keep}
            prepared, orig = [prepared[k] for k in keep], [orig[k] for k in keep]
            acc = {k for k, o in enumerate(orig) if o in accepted}
            fit = cls._fit(name, prepared, loc_cfg, method_cfg, acc)

        model = fit.model
        model.mm_per_px = mm_per_px
        model.accepted = sorted(acc)
        meas = [measure(a) for a in fit.aligned]
        model.baseline = {k: float(np.median([m[k] for m in meas])) for k in meas[0]}
        model.report = _training_report(fit, orig, skipped, excluded, acc, mm_per_px, time.time() - t0)
        model._ref_frames = [w for w, _ in prepared]
        return model

    @classmethod
    def _fit(
        cls,
        name: str,
        prepared: list[tuple[np.ndarray, PartDetection]],
        loc_cfg: LocalizationConfig,
        method_cfg: MethodConfig,
        accepted: set[int] | None = None,
    ) -> FitResult:
        """Alignment + all methods on the prepared references (one training pass)."""
        if len(prepared) < MIN_REFERENCES:
            raise TrainingError(
                f"Too few usable reference images ({len(prepared)}, at least {MIN_REFERENCES} required)."
            )
        model = cls(name, loc_cfg, method_cfg)
        sizes = np.array([default_canvas(d, loc_cfg.canvas_margin_ratio) for _, d in prepared])
        model.canvas = (int(sizes[:, 0].max()), int(sizes[:, 1].max()))

        # Iterative alignment: 1st image → mean image → refined mean image
        ref = align_part(*prepared[0], loc_cfg, model.canvas).norm
        for _ in range(2):
            aligned = [align_part(w, d, loc_cfg, model.canvas, ref) for w, d in prepared]
            ref = np.mean([a.norm for a in aligned], axis=0).astype(np.float32)
        aligned = [align_part(w, d, loc_cfg, model.canvas, ref) for w, d in prepared]
        model.ref_norm = ref
        model.polarity = aligned[0].polarity
        model.n_samples = len(aligned)
        # A low similarity with the mean image means a reference could not be aligned properly
        # (e.g. a symmetric outline with an asymmetric hole pattern that is hard to see).
        consistency = [ncc(np.clip(a.norm, 0, 1), np.clip(ref, 0, 1)) for a in aligned]

        suspects: dict[int, list[str]] = {}
        excludable: set[int] = set()
        order = {"geometry": 0, "diff": 1, "pca": 2}
        for mname in sorted(method_cfg.methods, key=lambda m: order.get(m, 9)):
            m = M.create(mname, method_cfg)
            m.fit(aligned, accepted or set())
            model.methods[mname] = m
            for i, why in m.suspects.items():
                suspects.setdefault(i, []).extend(f"{M.METHOD_LABELS.get(mname, mname)}: {w}" for w in why)
                if mname != "pca":
                    excludable.add(i)

        def rated_nok(a: AlignedPart) -> bool:
            return not decide([m.score(a) for m in model.methods.values()], method_cfg)[0]

        self_nok = [k for k, a in enumerate(aligned) if rated_nok(a)]
        return FitResult(model, aligned, consistency, suspects, excludable, self_nok)

    # ------------------------------------------------------------------ inspection
    def align(self, frame: np.ndarray) -> AlignedPart | None:
        """Finds the part in a frame and aligns it to the reference (None if no complete part is visible)."""
        work, det = _prepare(frame, self.loc_cfg)
        if det is None or not det.complete:
            return None
        return align_part(work, det, self.loc_cfg, self.canvas, self.ref_norm)

    def analyze(self, frame: np.ndarray, sensitivity: dict[str, float] | None = None) -> ChannelAnalysis | None:
        """Aligns the part and runs all methods (no decision yet)."""
        aligned = self.align(frame)
        if aligned is None:
            return None
        sens = self.sensitivity if sensitivity is None else sensitivity
        results = [m.score(aligned, sens.get(k, 1.0), self.mm_per_px) for k, m in self.methods.items()]
        return ChannelAnalysis(aligned, results, measure(aligned))

    def inspect(
        self, frame: np.ndarray, visualize: bool = True, sensitivity: dict[str, float] | None = None
    ) -> InspectionResult:
        """One image → OK / NOK / NO_PART. ``sensitivity`` overrides the model's own values."""
        t0 = time.perf_counter()
        ts = datetime.now().isoformat(timespec="milliseconds")
        an = self.analyze(frame, sensitivity)
        if an is None:
            return InspectionResult(
                "NO_PART", self.name, ts, (time.perf_counter() - t0) * 1000, model_version=self.version
            )
        ok, decisive = decide(an.results, self.method_cfg)
        shown = merge_defects(decisive)
        res = InspectionResult(
            "OK" if ok else "NOK",
            self.name,
            ts,
            0.0,
            shown,
            an.results,
            measurements=an.measurements,
            worst=max((r.score for r in decisive), default=0.0),
            model_version=self.version,
        )
        if visualize:
            heats = [r.heatmap for r in decisive if r.heatmap is not None]
            heat = np.max(np.stack(heats), axis=0) if heats else None
            res.overlay, res.detail = annotate(an.aligned, ok, shown, heat)
        res.time_ms = (time.perf_counter() - t0) * 1000
        return res

    # ------------------------------------------------------------------ settings
    def set_sensitivity(self, values: dict[str, float]) -> None:
        """> 1 = stricter (tolerances/thresholds divided by the value). Saved immediately."""
        for k, v in values.items():
            if k in METHOD_NAMES:
                self.sensitivity[k] = float(min(3.0, max(0.25, float(v))))
        self.save_meta()

    def save_meta(self) -> None:
        """Writes the settings (sensitivity, camera profile …) of a saved model."""
        if self.path is None:
            return
        if self._legacy:
            self.save_to(self.path)  # converts the folder to the current layout
        else:
            self._write_meta(self.path)

    # ------------------------------------------------------------------ persistence
    def save(self, models_dir: str | Path, keep_previous: bool = False, keep_side_data: bool = True) -> Path:
        """Saves to ``models_dir/<slug>``; an existing model with the same name is replaced."""
        return self.save_to(Path(models_dir) / slugify(self.name), keep_previous, keep_side_data)

    def save_to(self, path: Path, keep_previous: bool = False, keep_side_data: bool = True) -> Path:
        """Writes the model atomically. ``keep_previous``: the replaced version is kept in
        ``previous/`` (undo). ``keep_side_data``: accepted/known defective parts of the part type
        (``SIDE_FOLDERS``) are carried over – a new teach-in starts without them."""
        refs = self.reference_frames()  # read before the folder is replaced
        tmp = path.with_name(path.name + ".tmp")
        shutil.rmtree(tmp, ignore_errors=True)
        (tmp / "refs").mkdir(parents=True)
        arrays: dict[str, Any] = {"ref_norm": self.ref_norm}
        for k, m in self.methods.items():
            _, arr = m.state()
            arrays.update({f"{k}__{a}": v for a, v in arr.items()})
        np.savez_compressed(tmp / "arrays.npz", **arrays)
        for i, f in enumerate(refs):
            cv2.imwrite(str(tmp / "refs" / f"ref_{i:03d}.png"), f)
        self._write_meta(tmp)
        # swap in one step: readers (model list in the UI) never see a half-written model
        old = path.with_name(path.name + ".old")
        shutil.rmtree(old, ignore_errors=True)
        if path.exists():
            os.replace(path, old)
        os.replace(tmp, path)
        if old.exists():
            if keep_side_data:
                for side in SIDE_FOLDERS:
                    if (old / side).exists():
                        os.replace(old / side, path / side)
            shutil.rmtree(old / PREVIOUS, ignore_errors=True)  # one step of undo is kept, not a chain
            if keep_previous and (old / "meta.json").exists():
                os.replace(old, path / PREVIOUS)
            shutil.rmtree(old, ignore_errors=True)
        self._ref_frames, self.path, self._refs_dir, self._legacy = refs, path, path / "refs", False
        return path

    def _write_meta(self, folder: Path) -> None:
        meta = {
            "format": FORMAT,
            "name": self.name,
            "created": self.created,
            "canvas": list(self.canvas),
            "n_samples": self.n_samples,
            "polarity": self.polarity,
            "localization": asdict(self.loc_cfg),
            "method_config": asdict(self.method_cfg),
            "methods": {k: m.state()[0] for k, m in self.methods.items()},
            "report": self.report,
            "mm_per_px": self.mm_per_px,
            "baseline": self.baseline,
            "sensitivity": self.sensitivity,
            "camera_profile": self.camera_profile,
            "accepted": self.accepted,
            "version": self.version,
            "history": self.history,
        }
        tmp = folder / "meta.json.tmp"
        tmp.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, folder / "meta.json")

    @classmethod
    def load(cls, path: str | Path) -> QCModel:
        """Loads a saved model (also models of version 1 and 2.0)."""
        path = Path(path)
        extra: dict[str, Any] = {}
        if (path / "recipe.json").exists():  # saved by the full version (format 3)
            extra = json.loads((path / "recipe.json").read_text(encoding="utf-8"))
            if extra.get("channels") != ["main"]:
                raise TrainingError(f"“{extra.get('name')}” is a dual-light model – teach the part in again.")
            folder = path / "channels" / "main"
        elif (path / "meta.json").exists():
            folder = path
        else:
            raise FileNotFoundError(f"No model in {path}")
        meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
        loc = _merge(LocalizationConfig(), meta["localization"], strict=False)
        mcfg = _merge(MethodConfig(), meta["method_config"], strict=False)
        model = cls(meta["name"], loc, mcfg)
        model.created = meta["created"]
        model.canvas = (int(meta["canvas"][0]), int(meta["canvas"][1]))
        model.n_samples = meta["n_samples"]
        model.polarity = meta["polarity"]
        model.report = meta.get("report", {})
        model.mm_per_px = meta.get("mm_per_px")
        model.baseline = meta.get("baseline", {})
        model.sensitivity.update(extra.get("sensitivity") or meta.get("sensitivity") or {})
        model.camera_profile = extra.get("camera_profile", meta.get("camera_profile"))
        model.accepted = list(meta.get("accepted", []))
        model.version = int(meta.get("version", 1))
        model.history = list(meta.get("history", []))
        with np.load(folder / "arrays.npz") as npz:
            arrays = {k: npz[k] for k in npz.files}
        model.ref_norm = arrays["ref_norm"]
        for k, mmeta in meta["methods"].items():
            m = M.create(k, mcfg)
            prefix = f"{k}__"
            m.load_state(mmeta, {a[len(prefix) :]: v for a, v in arrays.items() if a.startswith(prefix)})
            model.methods[k] = m
        model.path, model._refs_dir, model._legacy = path, folder / "refs", folder != path
        return model

    def reference_frames(self) -> list[np.ndarray]:
        """The reference images the model was trained with."""
        if self._ref_frames is not None:
            return list(self._ref_frames)
        if self._refs_dir is None or not self._refs_dir.exists():
            return []
        images = (cv2.imread(str(p)) for p in sorted(self._refs_dir.glob("*.png")))
        return [img for img in images if img is not None]

    # ------------------------------------------------------------------ info
    def warnings(self) -> list[str]:
        """Plain-language hints after training (shown to the setup technician)."""
        rep = self.report
        out = []

        def numbers(indices: list[int]) -> str:
            return ", ".join(str(i + 1) for i in indices)

        excl = rep.get("references_excluded", [])
        if excl:
            out.append(
                f"Reference image(s) {numbers([e['index'] for e in excl])} left out – clearly different from the "
                f"other good parts, probably defective ({'; '.join(e['reasons'][0] for e in excl[:3])}). "
                "Check these parts."
            )
        sus = rep.get("references_suspect", [])
        if sus:
            out.append(
                f"Reference image(s) {numbers([e['index'] for e in sus])} differ from the other good parts but "
                f"were kept ({'; '.join(e['reasons'][0] for e in sus[:3])}). If several images are affected, "
                "check lighting and focus, then teach in again."
            )
        if rep.get("references_flagged_nok"):
            out.append(
                f"Reference image(s) {numbers(rep['references_flagged_nok'])} deviate strongly – "
                "was a defective part taught in by mistake?"
            )
        if rep.get("references_poorly_aligned"):
            out.append(
                f"Reference image(s) {numbers(rep['references_poorly_aligned'])} could not be aligned "
                "consistently – remove them and teach in again."
            )
        geo = rep.get("methods", {}).get("geometry", {})
        if geo.get("high_scatter_holes"):
            out.append(
                f"{len(geo['high_scatter_holes'])} hole(s) scatter strongly between the references – the model "
                "will be insensitive to position errors. Check lighting/contrast and teach in again."
            )
        if geo.get("inconsistent_references"):
            out.append(f"Inconsistent hole count in reference(s) {numbers(geo['inconsistent_references'])}.")
        return out

    def summary(self) -> dict[str, Any]:
        """Short description for the UI (name, references, methods, sensitivity, version)."""
        return {
            "name": self.name,
            "slug": self.slug,
            "created": self.created,
            "n_samples": self.n_samples,
            "methods": list(self.methods),
            "diff_representation": self.method_cfg.diff_representation,
            "pca_representation": self.method_cfg.pca_representation,
            "decision": self.method_cfg.decision,
            "sensitivity": self.sensitivity,
            "mm_per_px": self.mm_per_px,
            "camera_profile": self.camera_profile,
            "excluded": [e["index"] for e in self.report.get("references_excluded", [])],
            "accepted": len(self.accepted),
            "version": self.version,
            "history": self.history,
        }


def has_previous(path: Path) -> bool:
    """Has the model a previous version (undo of the last update possible)?"""
    return (path / PREVIOUS / "meta.json").exists()


def restore_previous(path: Path) -> QCModel:
    """Undo of the last update: swaps the current and the previous version (so the undo can itself
    be undone). Accepted and known defective parts stay with the part type."""
    if not has_previous(path):
        raise ValueError("There is no previous version of this model.")
    swap, old = path.with_name(path.name + ".swap"), path.with_name(path.name + ".old")
    shutil.rmtree(swap, ignore_errors=True)
    shutil.rmtree(old, ignore_errors=True)
    os.replace(path / PREVIOUS, swap)
    for side in SIDE_FOLDERS:
        if (path / side).exists():
            os.replace(path / side, swap / side)
    os.replace(path, old)
    os.replace(swap, path)
    os.replace(old, path / PREVIOUS)
    return QCModel.load(path)


def model_path(models_dir: str | Path, slug: str) -> Path | None:
    """Folder of a stored model, or None (also for names that try to leave ``models_dir``)."""
    base = Path(models_dir).resolve()
    p = (base / slug).resolve()
    if p.parent != base or not ((p / "meta.json").exists() or (p / "recipe.json").exists()):
        return None
    return p


def list_models(models_dir: str | Path) -> list[dict[str, Any]]:
    """Summaries of all models in ``models_dir`` (unreadable folders are skipped)."""
    out = []
    for p in sorted(Path(models_dir).glob("*")):
        if p.suffix in (".tmp", ".old", ".swap") or model_path(models_dir, p.name) is None:
            continue
        try:
            out.append(QCModel.load(p).summary())
        except (OSError, KeyError, ValueError, TrainingError, json.JSONDecodeError):
            continue  # e.g. a dual-light model of the full version
    return out
