"""Model = a learned part type (learning phase) + inspection (run phase).

A model contains the alignment reference, the parameters and the learned
state of all enabled inspection methods. It is stored as a folder and can
later be selected again or retrained with different settings (the reference
images are stored alongside)::

    data/models/<name>/
        meta.json      metadata, settings, training report
        arrays.npz     reference images/statistics of the methods
        refs/*.png     the captured reference images (working resolution)
"""
from __future__ import annotations

import json
import os
import re
import shutil
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from . import methods as M
from .alignment import AlignedPart, PartDetection, align_part, detect_part, resize_to_width, to_gray
from .config import LocalizationConfig, MethodConfig, _merge
from .visualize import annotate

MIN_REFERENCES = 3


class TrainingError(RuntimeError):
    pass


@dataclass
class InspectionResult:
    status: str                          # "OK" | "NOK" | "NO_PART"
    model: str
    timestamp: str
    time_ms: float
    defects: list[M.Defect] = field(default_factory=list)
    method_results: list[M.MethodResult] = field(default_factory=list)
    overlay: np.ndarray | None = None     # working image with markings (first channel)
    detail: np.ndarray | None = None      # aligned part with heatmap (first channel)
    image_id: str = ""
    ground_truth: str | None = None       # simulator/evaluation only
    images: list[dict] = field(default_factory=list)          # per channel: {"channel", "overlay", "detail"}
    measurements: dict = field(default_factory=dict)          # per channel: drift measurements
    shots: list[str] = field(default_factory=list)            # status of each shot (multi-shot voting)
    worst: float = 0.0                                        # highest decisive score (for voting)

    @property
    def ok(self) -> bool:
        return self.status == "OK"

    def to_dict(self) -> dict:
        return {
            "status": self.status, "model": self.model, "timestamp": self.timestamp,
            "time_ms": round(self.time_ms, 1), "image_id": self.image_id,
            "defects": [d.to_dict() for d in self.defects],
            "methods": [r.to_dict() for r in self.method_results],
            "ground_truth": self.ground_truth,
            "channels": [im["channel"] for im in self.images],
            "shots": self.shots,
        }


def standalone_levels(cfg: MethodConfig) -> dict[str, float]:
    return {"geometry": cfg.geometry_standalone, "diff": cfg.diff_standalone, "pca": cfg.pca_standalone}


def decide(results: list[M.MethodResult], cfg: MethodConfig) -> tuple[bool, list[M.MethodResult]]:
    """Combines the method results → (ok, decisive results).

    "or":     every method with score ≥ 1 is decisive.
    "fusion": a method is decisive alone only above its standalone level; if at least two
              methods are ≥ 1 at the same time (they agree), all of them are decisive.
    """
    over = [r for r in results if r.score >= 1.0]
    if cfg.decision == "or":
        decisive = over
    else:
        levels = standalone_levels(cfg)
        decisive = [r for r in over if r.score >= levels.get(r.method, 1.0)]
        if len(over) >= 2:
            decisive = over
    levels = standalone_levels(cfg)
    for r in results:
        r.standalone = 1.0 if cfg.decision == "or" else levels.get(r.method, 1.0)
        r.decisive = any(r is d for d in decisive)
    return not decisive, decisive


def merge_defects(results: list[M.MethodResult]) -> list[M.Defect]:
    """Defects of the decisive methods; reports of the same location are merged
    (earlier methods – geometry first – take precedence)."""
    shown: list[M.Defect] = []
    priority = {"geometry": 0, "diff": 1, "pca": 2}          # most specific defect description first
    for r in sorted(results, key=lambda r: priority.get(r.method, 9)):
        for d in r.defects:
            cx, cy = d.bbox[0] + d.bbox[2] / 2, d.bbox[1] + d.bbox[3] / 2
            if any(k.bbox[0] - 8 <= cx <= k.bbox[0] + k.bbox[2] + 8 and
                   k.bbox[1] - 8 <= cy <= k.bbox[1] + k.bbox[3] + 8 for k in shown):
                continue
            shown.append(d)
    return shown


def measure(a: AlignedPart) -> dict:
    """Image-quality measurements for drift monitoring (lighting, distance, focus)."""
    band = cv2.morphologyEx(a.filled_mask, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
    n = np.clip(a.norm, 0, 1.5)
    g = cv2.magnitude(cv2.Sobel(n, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(n, cv2.CV_32F, 0, 1, ksize=3))
    return {
        "contrast": float(abs(a.part_level - a.bg_level)),      # grey levels between part and belt
        "background": float(a.bg_level),
        "area": float(np.count_nonzero(a.filled_mask)),          # changes with camera distance/zoom
        "sharpness": float(np.percentile(g[band], 90)) if band.any() else 0.0,   # edge steepness (focus/motion blur)
    }


@dataclass
class ChannelAnalysis:
    aligned: AlignedPart
    results: list[M.MethodResult]
    measurements: dict


def slugify(name: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_\-]+", "_", name.strip()).strip("_")
    return s or "model"


def _prepare(frame: np.ndarray, loc: LocalizationConfig) -> tuple[np.ndarray, PartDetection | None]:
    work = resize_to_width(frame, loc.work_width)
    det = detect_part(to_gray(work), loc)
    return work, det


class QCModel:
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
        self.report: dict = {}
        self.path: Path | None = None
        self.mm_per_px: float | None = None       # scale at training time (camera calibration)
        self.calibration_id: str | None = None
        self.baseline: dict = {}                  # drift monitoring: measurements of the references
        self.n_anchor = 0

    # ------------------------------------------------------------------ learning
    @classmethod
    def train(cls, name: str, frames: list[np.ndarray], loc_cfg: LocalizationConfig,
              method_cfg: MethodConfig, mm_per_px: float | None = None,
              calibration_id: str | None = None, n_anchor: int | None = None) -> "QCModel":
        """``frames[:n_anchor]`` are the *anchor* references (taught in or confirmed by the
        supervisor). Alignment reference, geometry (nominal dimensions and tolerances) and
        the drift baseline are learned from them only. Additional frames (good parts
        collected during operation) only refine the statistical methods (diff, pca), so
        slow process drift such as tool wear cannot be learned away."""
        t0 = time.time()
        model = cls(name, loc_cfg, method_cfg)
        model.mm_per_px, model.calibration_id = mm_per_px, calibration_id
        n_anchor = len(frames) if n_anchor is None else n_anchor
        prepared, skipped, is_anchor = [], [], []
        for i, f in enumerate(frames):
            work, det = _prepare(f, loc_cfg)
            if det is None or not det.complete:
                skipped.append(i)
            else:
                prepared.append((work, det))
                is_anchor.append(i < n_anchor)
        if sum(is_anchor) < MIN_REFERENCES:
            raise TrainingError(f"Too few usable reference images ({sum(is_anchor)}, at least {MIN_REFERENCES} required).")

        from .alignment import default_canvas
        sizes = np.array([default_canvas(d, loc_cfg.canvas_margin_ratio) for _, d in prepared])
        model.canvas = (int(sizes[:, 0].max()), int(sizes[:, 1].max()))

        # Iterative alignment: 1st image → mean image → refined mean image (from the anchors)
        anchors = [p for p, a in zip(prepared, is_anchor) if a]
        aligned = [align_part(w, d, loc_cfg, model.canvas) for w, d in anchors[:1]]
        ref = aligned[0].norm
        for _ in range(2):
            aligned_anchor = [align_part(w, d, loc_cfg, model.canvas, ref) for w, d in anchors]
            ref = np.mean([a.norm for a in aligned_anchor], axis=0).astype(np.float32)
        aligned = [align_part(w, d, loc_cfg, model.canvas, ref) for w, d in prepared]
        aligned_anchor = [a for a, an in zip(aligned, is_anchor) if an]
        model.ref_norm = ref
        model.n_anchor = len(aligned_anchor)
        # Consistency check: every reference must look like the mean image.
        # A low value means a reference could not be aligned properly
        # (e.g. a symmetric outline with an asymmetric hole pattern that is hard to see).
        from .alignment import _ncc
        consistency = [_ncc(np.clip(a.norm, 0, 1), np.clip(ref, 0, 1)) for a in aligned]
        poorly_aligned = [i for i, c in enumerate(consistency) if c < 0.9]
        model.polarity = aligned[0].polarity
        model.n_samples = len(aligned)

        order = {"geometry": 0, "diff": 1, "pca": 2}
        for mname in sorted(method_cfg.methods, key=lambda m: order.get(m, 9)):
            m = M.create(mname, method_cfg)
            m.fit(aligned_anchor if mname == "geometry" else aligned)
            model.methods[mname] = m

        # Self-test: how many references would be rated NOK? (hint for the supervisor)
        self_nok = [i for i, a in enumerate(aligned)
                    if not decide([m.score(a) for m in model.methods.values()], method_cfg)[0]]
        meas = [measure(a) for a in aligned_anchor]
        model.baseline = {k: float(np.median([m[k] for m in meas])) for k in meas[0]}
        model.report = {
            "references_used": len(aligned), "references_skipped": skipped,
            "anchor_references": len(aligned_anchor), "learned_references": len(aligned) - len(aligned_anchor),
            "canvas": list(model.canvas), "polarity": model.polarity,
            "train_time_s": round(time.time() - t0, 2),
            "references_flagged_nok": self_nok,
            "alignment_consistency_min": round(float(min(consistency)), 3),
            "references_poorly_aligned": poorly_aligned,
            "methods": {k: m.report for k, m in model.methods.items()},
            "baseline": model.baseline,
            "mm_per_px": mm_per_px,
        }
        geo = model.report["methods"].get("geometry")
        if geo is not None and mm_per_px:
            geo["hole_diameters_mm"] = [round(d * mm_per_px, 3) for d in geo["hole_diameters_px"]]
        model._ref_frames = [w for (w, _), a in zip(prepared, is_anchor) if a]
        model._learned_frames = [w for (w, _), a in zip(prepared, is_anchor) if not a]
        return model

    # ------------------------------------------------------------------ inspection
    def align(self, frame: np.ndarray) -> AlignedPart | None:
        work, det = _prepare(frame, self.loc_cfg)
        if det is None or not det.complete:
            return None
        return align_part(work, det, self.loc_cfg, self.canvas, self.ref_norm)

    def analyze(self, frame: np.ndarray, sensitivity: dict | None = None) -> ChannelAnalysis | None:
        """Aligns the part and runs all methods (no decision yet)."""
        aligned = self.align(frame)
        if aligned is None:
            return None
        sens = sensitivity or {}
        results = [m.score(aligned, sens.get(k, 1.0), self.mm_per_px) for k, m in self.methods.items()]
        return ChannelAnalysis(aligned, results, measure(aligned))

    def inspect(self, frame: np.ndarray, visualize: bool = True, sensitivity: dict | None = None) -> InspectionResult:
        t0 = time.perf_counter()
        ts = datetime.now().isoformat(timespec="milliseconds")
        an = self.analyze(frame, sensitivity)
        if an is None:
            return InspectionResult("NO_PART", self.name, ts, (time.perf_counter() - t0) * 1000)
        ok, decisive = decide(an.results, self.method_cfg)
        shown = merge_defects(decisive)
        res = InspectionResult("OK" if ok else "NOK", self.name, ts, 0.0, shown, an.results,
                               measurements={"main": an.measurements},
                               worst=max((r.score for r in decisive), default=0.0))
        if visualize:
            heats = [r.heatmap for r in decisive if r.heatmap is not None]
            heat = np.max(np.stack(heats), axis=0) if heats else None
            res.overlay, res.detail = annotate(an.aligned, ok, shown, heat)
            res.images = [{"channel": "main", "overlay": res.overlay, "detail": res.detail}]
        res.time_ms = (time.perf_counter() - t0) * 1000
        return res

    # ------------------------------------------------------------- persistence
    def save(self, models_dir: str | Path) -> Path:
        return self.save_to(Path(models_dir) / slugify(self.name))

    def save_to(self, path: str | Path) -> Path:
        path = Path(path)
        ref_frames = self.reference_frames()             # read before deleting
        learned_frames = self.learned_frames()
        # model files are replaced; side folders (pending/, collected/, known_bad/) are kept
        for n in ("meta.json", "arrays.npz"):
            (path / n).unlink(missing_ok=True)
        for n in ("refs", "learned"):
            if (path / n).exists():
                shutil.rmtree(path / n)
        (path / "refs").mkdir(parents=True)
        arrays = {"ref_norm": self.ref_norm}
        meta_methods = {}
        for k, m in self.methods.items():
            meta, arr = m.state()
            meta_methods[k] = meta
            arrays.update({f"{k}__{a}": v for a, v in arr.items()})
        meta = {
            "name": self.name, "created": self.created, "canvas": list(self.canvas),
            "n_samples": self.n_samples, "polarity": self.polarity,
            "localization": asdict(self.loc_cfg), "method_config": asdict(self.method_cfg),
            "methods": meta_methods, "report": self.report, "format": 2,
            "mm_per_px": self.mm_per_px, "calibration_id": self.calibration_id, "baseline": self.baseline,
            "n_anchor": self.n_anchor,
        }
        # atomic writes: readers (e.g. the model list in the UI) never see half-written files
        tmp = path / "arrays.tmp.npz"
        np.savez_compressed(tmp, **arrays)
        os.replace(tmp, path / "arrays.npz")
        tmp = path / "meta.json.tmp"
        tmp.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, path / "meta.json")
        for i, f in enumerate(ref_frames):
            cv2.imwrite(str(path / "refs" / f"ref_{i:03d}.png"), f)
        if learned_frames:
            (path / "learned").mkdir()
            for i, f in enumerate(learned_frames):
                cv2.imwrite(str(path / "learned" / f"learned_{i:03d}.png"), f)
        self._ref_frames, self._learned_frames = ref_frames, learned_frames
        self.path = path
        return path

    @classmethod
    def load(cls, path: str | Path) -> "QCModel":
        path = Path(path)
        meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
        loc = _merge(LocalizationConfig(), meta["localization"], strict=False)
        mcfg = _merge(MethodConfig(), meta["method_config"], strict=False)
        model = cls(meta["name"], loc, mcfg)
        model.created = meta["created"]
        model.canvas = tuple(meta["canvas"])
        model.n_samples = meta["n_samples"]
        model.polarity = meta["polarity"]
        model.report = meta.get("report", {})
        model.mm_per_px = meta.get("mm_per_px")
        model.calibration_id = meta.get("calibration_id")
        model.baseline = meta.get("baseline", {})
        model.n_anchor = meta.get("n_anchor", meta["n_samples"])
        with np.load(path / "arrays.npz") as npz:
            arrays = {k: npz[k] for k in npz.files}
        model.ref_norm = arrays["ref_norm"]
        for k, mmeta in meta["methods"].items():
            m = M.create(k, mcfg)
            prefix = f"{k}__"
            m.load_state(mmeta, {a[len(prefix):]: v for a, v in arrays.items() if a.startswith(prefix)})
            model.methods[k] = m
        model.path = path
        return model

    def reference_frames(self) -> list[np.ndarray]:
        """Anchor references (taught in / confirmed by the supervisor)."""
        if getattr(self, "_ref_frames", None):
            return list(self._ref_frames)
        if self.path is None or not (self.path / "refs").exists():
            return []
        return [cv2.imread(str(p)) for p in sorted((self.path / "refs").glob("*.png"))]

    def learned_count(self) -> int:
        if getattr(self, "_learned_frames", None) is not None:
            return len(self._learned_frames)
        if self.path is None or not (self.path / "learned").exists():
            return 0
        return len(list((self.path / "learned").glob("*.png")))

    def learned_frames(self) -> list[np.ndarray]:
        """Good parts collected during operation that were already trained in."""
        if getattr(self, "_learned_frames", None) is not None:
            return list(self._learned_frames)
        if self.path is None or not (self.path / "learned").exists():
            return []
        return [cv2.imread(str(p)) for p in sorted((self.path / "learned").glob("*.png"))]

    def summary(self) -> dict:
        return {
            "name": self.name, "created": self.created, "n_samples": self.n_samples,
            "methods": list(self.methods), "polarity": self.polarity,
            "diff_representation": self.method_cfg.diff_representation,
            "pca_representation": self.method_cfg.pca_representation,
            "flagged": self.report.get("references_flagged_nok", []),
            "skipped": self.report.get("references_skipped", []),
            "slug": self.path.name if self.path else slugify(self.name),
        }


def list_models(models_dir: str | Path) -> list[dict]:
    out = []
    for p in sorted(Path(models_dir).glob("*/meta.json")):
        try:
            meta = json.loads(p.read_text(encoding="utf-8"))
            out.append({"slug": p.parent.name, "name": meta["name"], "created": meta["created"],
                        "n_samples": meta["n_samples"], "methods": list(meta["methods"]),
                        "diff_representation": meta["method_config"].get("diff_representation"),
                        "pca_representation": meta["method_config"].get("pca_representation")})
        except (OSError, KeyError, json.JSONDecodeError):
            continue
    return out


def retrain(path: str | Path, method_cfg: MethodConfig, loc_cfg: LocalizationConfig | None = None) -> QCModel:
    """Retrains a stored model with new method settings."""
    old = QCModel.load(path)
    frames = old.reference_frames()
    model = QCModel.train(old.name, frames, loc_cfg or old.loc_cfg, method_cfg)
    model.created = old.created
    model.save(Path(path).parent)
    return model
