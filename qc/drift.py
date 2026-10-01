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

import numpy as np

from .config import DriftConfig

LABELS = {"contrast": "Contrast", "background": "Background brightness", "area": "Part size",
          "sharpness": "Edge sharpness"}


class DriftMonitor:
    def __init__(self, cfg: DriftConfig):
        self.cfg = cfg
        self.baselines: dict[str, dict] = {}
        self.values: dict[str, dict[str, deque]] = {}

    def reset(self, baselines: dict[str, dict]) -> None:
        """New model: ``baselines`` = {channel: {measurement: reference value}}."""
        self.baselines = {ch: b for ch, b in baselines.items() if b}
        self.values = {ch: {k: deque(maxlen=self.cfg.window) for k in b} for ch, b in self.baselines.items()}

    def update(self, measurements: dict[str, dict]) -> None:
        for ch, meas in measurements.items():
            for k, v in meas.items():
                if ch in self.values and k in self.values[ch]:
                    self.values[ch][k].append(v)

    def state(self) -> dict:
        """Current deviations and warnings (only once at least half the window is filled)."""
        c = self.cfg
        rows, warnings = [], []
        for ch, base in self.baselines.items():
            for k, ref in base.items():
                vals = self.values[ch][k]
                if not vals:
                    continue
                cur = float(np.median(vals))
                if k == "background":
                    dev, lim = cur - ref, c.background_tol
                    txt = f"{dev:+.0f} grey levels"
                    bad = abs(dev) > lim
                else:
                    dev = (cur - ref) / ref if ref else 0.0
                    lim = {"contrast": c.contrast_tol, "area": c.area_tol, "sharpness": c.sharpness_tol}[k]
                    txt = f"{dev * 100:+.1f} %"
                    bad = (dev < -lim) if k == "sharpness" else abs(dev) > lim
                ready = len(vals) >= max(3, c.window // 2)
                label = LABELS[k] + (f" ({ch})" if ch != "main" else "")
                rows.append({"channel": ch, "key": k, "label": label, "reference": round(ref, 2),
                             "current": round(cur, 2), "deviation": txt, "warning": bool(bad and ready),
                             "samples": len(vals)})
                if bad and ready:
                    warnings.append(f"{label} {txt}")
        return {"rows": rows, "warnings": warnings}
