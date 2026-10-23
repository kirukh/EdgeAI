"""``diff`` – reference image comparison: z-score map against the mean and spread image of the good parts."""

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
    components,
    max_component_area,
    robust_outliers,
    split_accepted,
    tolerance_map,
)


class DiffMethod(Method):
    name = "diff"

    def _rep(self, s: AlignedPart) -> np.ndarray:
        r = reps.compute(s, self.cfg.diff_representation)
        b = self.cfg.diff_blur | 1
        return cv2.GaussianBlur(r, (b, b), 0) if b > 1 else r

    accept_dev: np.ndarray | None = None  # local tolerance from parts the operator accepted

    def _zmap(self, x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
        d = np.abs(x - mean)
        if self.accept_dev is not None:
            d = np.maximum(d - self.accept_dev, 0)
        z: np.ndarray = d / std * self.region
        return z

    def fit(self, samples: list[AlignedPart], accepted: Collection[int] = ()) -> None:
        normal, acc = split_accepted(len(samples), accepted)
        all_x = np.stack([self._rep(s) for s in samples])
        X = all_x[normal]
        filled = (np.mean([samples[i].filled_mask > 0 for i in normal], axis=0) > 0.5).astype(np.uint8)
        self.region = (cv2.dilate(filled, np.ones((9, 9), np.uint8)) > 0).astype(np.float32)
        floor = self.cfg.diff_sigma_floor
        self.mean = X.mean(0)
        self.std = np.maximum(X.std(0, ddof=1) if len(X) > 1 else np.zeros_like(self.mean), floor)
        self.accept_dev = None

        # Leave-one-out: how large do deviation areas get for good parts?
        loo = []
        if len(X) >= 3:
            for i in range(len(X)):
                rest = np.delete(X, i, axis=0)
                z = self._zmap(X[i], rest.mean(0), np.maximum(rest.std(0, ddof=1), floor))
                loo.append(max_component_area(self._clean(z > self.cfg.diff_z_threshold)))
        self.min_area = float(max(self.cfg.diff_min_area_px, 1.5 * max(loo, default=0)))
        self.suspects = {}
        for i in robust_outliers(loo, self.cfg.diff_min_area_px, self.cfg.screen_k):
            self._flag(normal[i], f"differs from the other references in the image comparison ({loo[i]:.0f} px)")
        if acc:  # accepted deviations are tolerated where they are – the global threshold stays
            devs = [np.abs(all_x[i] - self.mean) for i in sorted(acc)]
            marked = [d / self.std * self.region > self.cfg.diff_z_threshold / 2 for d in devs]
            self.accept_dev = tolerance_map(devs, marked, 7)
        self.report = {
            "representation": self.cfg.diff_representation,
            "loo_max_area_px": [round(v, 1) for v in loo],
            "min_area_px": round(self.min_area, 1),
        }

    @staticmethod
    def _clean(b: np.ndarray) -> np.ndarray:
        return cv2.morphologyEx(b.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    def score(self, s: AlignedPart, sens: float = 1.0, mm_per_px: float | None = None) -> MethodResult:
        sens = max(float(sens), 1e-3)
        z = self._zmap(self._rep(s), self.mean, self.std)
        zt = self.cfg.diff_z_threshold / sens
        min_area = self.min_area / sens
        comps, largest = components(self._clean(z > zt), min_area)

        def area_text(px: float) -> str:
            return f"{px * mm_per_px**2:.1f} mm²" if mm_per_px else f"{px:.0f} px"

        defects = [
            Defect("SURFACE", self.name, bbox, f"{area_text(area)}, max z = {float(z[m].max()):.1f}", area / min_area)
            for area, bbox, m in comps
        ]
        score = largest / min_area  # the largest region – defect or not
        heat = np.clip(z / (2 * zt), 0, 1).astype(np.float32)
        return MethodResult(self.name, not defects, float(score), defects, heat)

    def state(self) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
        return (
            {"min_area": self.min_area, "report": self.report},
            {"mean": self.mean, "std": self.std, "region": self.region}
            | ({"accept_dev": self.accept_dev} if self.accept_dev is not None else {}),
        )

    def load_state(self, meta: dict[str, Any], arrays: dict[str, np.ndarray]) -> None:
        self.min_area = meta["min_area"]
        self.report = meta.get("report", {})
        self.mean, self.std, self.region = arrays["mean"], arrays["std"], arrays["region"]
        self.accept_dev = arrays.get("accept_dev")
