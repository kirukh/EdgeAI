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
    overlay: np.ndarray | None = None     # working image with markings
    detail: np.ndarray | None = None      # aligned part with heatmap
    image_id: str = ""
    ground_truth: str | None = None       # simulator/evaluation only

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
        }


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

    # ------------------------------------------------------------------ learning
    @classmethod
    def train(cls, name: str, frames: list[np.ndarray], loc_cfg: LocalizationConfig,
              method_cfg: MethodConfig) -> "QCModel":
        t0 = time.time()
        model = cls(name, loc_cfg, method_cfg)
        prepared, skipped = [], []
        for i, f in enumerate(frames):
            work, det = _prepare(f, loc_cfg)
            if det is None or not det.complete:
                skipped.append(i)
            else:
                prepared.append((work, det))
        if len(prepared) < MIN_REFERENCES:
            raise TrainingError(f"Too few usable reference images ({len(prepared)}, at least {MIN_REFERENCES} required).")

        from .alignment import default_canvas
        sizes = np.array([default_canvas(d, loc_cfg.canvas_margin_ratio) for _, d in prepared])
        model.canvas = (int(sizes[:, 0].max()), int(sizes[:, 1].max()))

        # Iterative alignment: 1st image → mean image → refined mean image
        aligned = [align_part(w, d, loc_cfg, model.canvas) for w, d in prepared]
        ref = aligned[0].norm
        for _ in range(2):
            aligned = [align_part(w, d, loc_cfg, model.canvas, ref) for w, d in prepared]
            ref = np.mean([a.norm for a in aligned], axis=0).astype(np.float32)
        model.ref_norm = ref
        # Consistency check: every reference must look like the mean image.
        # A low value means a reference could not be aligned properly
        # (e.g. a symmetric outline with an asymmetric hole pattern that is hard to see).
        from .alignment import _ncc
        consistency = [_ncc(np.clip(a.norm, 0, 1), np.clip(ref, 0, 1)) for a in aligned]
        poorly_aligned = [i for i, c in enumerate(consistency) if c < 0.9]
        model.polarity = aligned[0].polarity
        model.n_samples = len(aligned)

        for mname in method_cfg.methods:
            m = M.create(mname, method_cfg)
            m.fit(aligned)
            model.methods[mname] = m

        # Self-test: how many references would be rated NOK? (hint for the supervisor)
        self_nok = [i for i, a in enumerate(aligned) if not all(m.score(a).ok for m in model.methods.values())]
        model.report = {
            "references_used": len(aligned), "references_skipped": skipped,
            "canvas": list(model.canvas), "polarity": model.polarity,
            "train_time_s": round(time.time() - t0, 2),
            "references_flagged_nok": self_nok,
            "alignment_consistency_min": round(float(min(consistency)), 3),
            "references_poorly_aligned": poorly_aligned,
            "methods": {k: m.report for k, m in model.methods.items()},
        }
        model._ref_frames = [w for w, _ in prepared]
        return model

    # ------------------------------------------------------------------ inspection
    def align(self, frame: np.ndarray) -> AlignedPart | None:
        work, det = _prepare(frame, self.loc_cfg)
        if det is None or not det.complete:
            return None
        return align_part(work, det, self.loc_cfg, self.canvas, self.ref_norm)

    def inspect(self, frame: np.ndarray, visualize: bool = True) -> InspectionResult:
        t0 = time.perf_counter()
        ts = datetime.now().isoformat(timespec="milliseconds")
        aligned = self.align(frame)
        if aligned is None:
            return InspectionResult("NO_PART", self.name, ts, (time.perf_counter() - t0) * 1000)

        results = [m.score(aligned) for m in self.methods.values()]
        ok = all(r.ok for r in results)

        # Merge duplicate reports of the same location (geometry takes precedence)
        shown: list[M.Defect] = []
        for r in results:
            for d in r.defects:
                cx, cy = d.bbox[0] + d.bbox[2] / 2, d.bbox[1] + d.bbox[3] / 2
                if any(k.bbox[0] - 8 <= cx <= k.bbox[0] + k.bbox[2] + 8 and
                       k.bbox[1] - 8 <= cy <= k.bbox[1] + k.bbox[3] + 8 for k in shown):
                    continue
                shown.append(d)

        res = InspectionResult("OK" if ok else "NOK", self.name, ts, 0.0, shown, results)
        if visualize:
            heats = [r.heatmap for r in results if r.heatmap is not None and not r.ok]
            heat = np.max(np.stack(heats), axis=0) if heats else None
            res.overlay, res.detail = annotate(aligned, ok, shown, heat)
        res.time_ms = (time.perf_counter() - t0) * 1000
        return res

    # ------------------------------------------------------------- persistence
    def save(self, models_dir: str | Path) -> Path:
        path = Path(models_dir) / slugify(self.name)
        if path.exists():
            shutil.rmtree(path)
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
            "methods": meta_methods, "report": self.report, "format": 1,
        }
        (path / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        np.savez_compressed(path / "arrays.npz", **arrays)
        for i, f in enumerate(getattr(self, "_ref_frames", [])):
            cv2.imwrite(str(path / "refs" / f"ref_{i:03d}.png"), f)
        self.path = path
        return path

    @classmethod
    def load(cls, path: str | Path) -> "QCModel":
        path = Path(path)
        meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
        loc = _merge(LocalizationConfig(), meta["localization"])
        mcfg = _merge(MethodConfig(), meta["method_config"])
        model = cls(meta["name"], loc, mcfg)
        model.created = meta["created"]
        model.canvas = tuple(meta["canvas"])
        model.n_samples = meta["n_samples"]
        model.polarity = meta["polarity"]
        model.report = meta.get("report", {})
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
        if self.path is None:
            return list(getattr(self, "_ref_frames", []))
        return [cv2.imread(str(p)) for p in sorted((self.path / "refs").glob("*.png"))]

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
