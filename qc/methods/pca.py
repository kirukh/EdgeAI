"""``pca`` – machine learning: PCA subspace of the good parts, the reconstruction error is the anomaly score."""

from __future__ import annotations

from collections.abc import Collection
from typing import Any

import cv2
import numpy as np

from .. import representations as reps
from ..alignment import AlignedPart
from .base import (
    Defect,
    Method,
    MethodResult,
    bbox_from_mask,
    components,
    robust_outliers,
    split_accepted,
    tolerance_map,
)


class PCAMethod(Method):
    name = "pca"

    def _vec(self, s: AlignedPart) -> tuple[np.ndarray, tuple[int, int]]:
        r = reps.compute(s, self.cfg.pca_representation)
        h, w = r.shape
        size = (self.cfg.pca_size, max(8, int(round(self.cfg.pca_size * h / w))))
        return cv2.resize(r, size, interpolation=cv2.INTER_AREA), size

    @staticmethod
    def _basis(data: np.ndarray, variance: float) -> tuple[np.ndarray, np.ndarray]:
        mean = data.mean(0)
        centred = data - mean
        if len(data) < 2:
            return mean, np.zeros((0, data.shape[1]), np.float32)
        _, sv, vt = np.linalg.svd(centred, full_matrices=False)
        ev = sv**2
        if ev.sum() <= 0:
            return mean, np.zeros((0, data.shape[1]), np.float32)
        k = int(np.searchsorted(np.cumsum(ev) / ev.sum(), variance) + 1)
        k = min(k, max(1, len(data) - 2))
        return mean, vt[:k].astype(np.float32)

    def _residual(self, v: np.ndarray, mean: np.ndarray, comps: np.ndarray, size: tuple[int, int]) -> np.ndarray:
        c = v - mean
        r = c - (c @ comps.T) @ comps if len(comps) else c
        rmap = np.abs(r.reshape(size[1], size[0]))
        out: np.ndarray = cv2.GaussianBlur(rmap, (3, 3), 0)
        return out

    accept_res: np.ndarray | None = None  # local tolerance from parts the operator accepted

    def _effective(self, rmap: np.ndarray) -> np.ndarray:
        return rmap if self.accept_res is None else np.maximum(rmap - self.accept_res, 0)

    def fit(self, samples: list[AlignedPart], accepted: Collection[int] = ()) -> None:
        normal, acc = split_accepted(len(samples), accepted)
        vecs = [self._vec(s) for s in samples]
        self.size = vecs[0][1]
        all_x = np.stack([v[0].ravel() for v in vecs]).astype(np.float32)
        X = all_x[normal]
        self.mean, self.comps = self._basis(X, self.cfg.pca_variance)
        self.accept_res = None
        loo = []
        if len(X) >= 4:
            for i in range(len(X)):
                m, c = self._basis(np.delete(X, i, axis=0), self.cfg.pca_variance)
                loo.append(float(self._residual(X[i], m, c, self.size).max()))
        self.suspects = {}
        if loo:
            # ML alone is less certain than geometry/comparison → same stand-alone level as in decide()
            for i in robust_outliers(
                loo, self.cfg.pca_margin * float(np.median(loo)), self.cfg.screen_k, self.cfg.pca_standalone
            ):
                self._flag(normal[i], f"ML anomaly compared with the other references (residual {loo[i]:.3f})")
            self.threshold = max(self.cfg.pca_margin * float(max(loo)), float(np.mean(loo) + 4 * np.std(loo)))
        else:
            self.threshold = self.cfg.pca_residual_threshold
        if acc:  # accepted deviations are tolerated where they are – the global threshold stays
            res = [self._residual(all_x[i], self.mean, self.comps, self.size) for i in sorted(acc)]
            self.accept_res = tolerance_map(res, [r > self.threshold / 2 for r in res], 3)
        self.report = {
            "representation": self.cfg.pca_representation,
            "components": len(self.comps),
            "loo_scores": [round(v, 4) for v in loo],
            "threshold": round(self.threshold, 4),
        }

    def score(self, s: AlignedPart, sens: float = 1.0, mm_per_px: float | None = None) -> MethodResult:
        v, _ = self._vec(s)
        rmap = self._effective(self._residual(v.ravel().astype(np.float32), self.mean, self.comps, self.size))
        thr = self.threshold / max(float(sens), 1e-3)
        score = float(rmap.max()) / thr
        h, w = s.gray.shape
        big = cv2.resize(rmap, (w, h), interpolation=cv2.INTER_LINEAR)
        defects = []
        if score >= 1.0:
            cell = (w / self.size[0]) * (h / self.size[1])
            for _area, bbox, m in components(big > thr, cell)[0]:
                defects.append(
                    Defect(
                        "ANOMALY",
                        self.name,
                        bbox,
                        f"residual {float(np.max(big[m])):.3f} (threshold {thr:.3f})",
                        float(np.max(big[m])) / thr,
                    )
                )
            if not defects:
                defects.append(Defect("ANOMALY", self.name, bbox_from_mask(big >= np.max(big) * 0.9), "", score))
        heat = np.clip(big / (2 * thr), 0, 1).astype(np.float32)
        return MethodResult(self.name, bool(score < 1.0), float(score), defects, heat)

    def state(self) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
        return (
            {"threshold": self.threshold, "size": list(self.size), "report": self.report},
            {"mean": self.mean, "comps": self.comps}
            | ({"accept_res": self.accept_res} if self.accept_res is not None else {}),
        )

    def load_state(self, meta: dict[str, Any], arrays: dict[str, np.ndarray]) -> None:
        self.threshold = meta["threshold"]
        self.size = tuple(meta["size"])
        self.report = meta.get("report", {})
        self.mean, self.comps = arrays["mean"], arrays["comps"]
        self.accept_res = arrays.get("accept_res")
