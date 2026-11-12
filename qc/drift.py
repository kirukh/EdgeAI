"""Drift monitoring: is the image still as good as during the learning phase?

For every inspected part four image-quality values are measured (see
``model.measure``) and compared with the values of the reference parts:

* contrast part ↔ belt  – LED ageing, changed exposure, dirty lens
* background brightness – ambient light, dirty belt/backlight panel
* part area             – camera distance or zoom changed (camera bumped)
* edge sharpness        – focus drifted, motion blur (belt faster, exposure longer)

The median over a rolling window is used, so single outliers do not trigger.
A warning does not stop the inspection – it tells the supervisor to check the
setup before the detection quality silently degrades.
"""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np

from .config import DriftConfig

LABELS = {
    "contrast": "Contrast",
    "background": "Background brightness",
    "area": "Part size",
    "sharpness": "Edge sharpness",
}


class DriftMonitor:
    """Image quality (contrast, background, part size, sharpness) of the last parts compared with the learning phase."""

    def __init__(self, cfg: DriftConfig):
        self.cfg = cfg
        self.baseline: dict[str, float] = {}
        self.values: dict[str, deque[float]] = {}

    def reset(self, baseline: dict[str, float] | None) -> None:
        """New model: ``baseline`` = {measurement: value of the reference parts}."""
        self.baseline = dict(baseline or {})
        self.values = {k: deque(maxlen=self.cfg.window) for k in self.baseline}

    def update(self, measurements: dict[str, float]) -> None:
        """Adds the measurements of one inspected part."""
        for k, v in measurements.items():
            if k in self.values:
                self.values[k].append(v)

    def _deviation(self, key: str, ref: float, cur: float) -> tuple[str, bool]:
        """(text, out of tolerance?) for one measurement."""
        c = self.cfg
        if key == "background":
            dev = cur - ref
            return f"{dev:+.0f} grey levels", abs(dev) > c.background_tol
        rel = (cur - ref) / ref if ref else 0.0
        limit = {"contrast": c.contrast_tol, "area": c.area_tol, "sharpness": c.sharpness_tol}[key]
        bad = rel < -limit if key == "sharpness" else abs(rel) > limit  # sharper is never a problem
        return f"{rel * 100:+.1f} %", bad

    def state(self) -> dict[str, Any]:
        """Current deviations and warnings (only once at least half the window is filled)."""
        rows, warnings = [], []
        for k, ref in self.baseline.items():
            vals = self.values[k]
            if not vals:
                continue
            cur = float(np.median(vals))
            txt, bad = self._deviation(k, ref, cur)
            warn = bad and len(vals) >= max(3, self.cfg.window // 2)
            rows.append(
                {
                    "key": k,
                    "label": LABELS[k],
                    "reference": round(ref, 2),
                    "current": round(cur, 2),
                    "deviation": txt,
                    "warning": warn,
                    "samples": len(vals),
                }
            )
            if warn:
                warnings.append(f"{LABELS[k]} {txt}")
        return {"rows": rows, "warnings": warnings}
