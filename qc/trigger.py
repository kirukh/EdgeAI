"""Conveyor trigger: fires exactly one capture per part."""
from __future__ import annotations

from dataclasses import replace

import cv2
import numpy as np

from .alignment import detect_part, resize_to_width, to_gray
from .config import LocalizationConfig, TriggerConfig


class PartTrigger:
    """Fires exactly one capture per part – when it crosses the image centre
    or rests inside the centre band (conveyor stopped)."""

    def __init__(self, cfg: TriggerConfig, loc: LocalizationConfig):
        self.cfg = cfg
        self.loc = replace(loc, border_margin_px=2)
        self.armed = True
        self.prev_pos: float | None = None
        self.misses = 0
        self.last_det = None
        self.scale = 1.0

    def update(self, frame: np.ndarray) -> bool:
        small = resize_to_width(frame, self.cfg.detect_width)
        self.scale = frame.shape[1] / small.shape[1]
        det = detect_part(to_gray(small), self.loc)
        self.last_det = det
        axis = 0 if self.cfg.axis == "x" else 1
        size = small.shape[1 - axis]
        pos = det.centroid[axis] / size - 0.5 if det is not None else None
        in_band = pos is not None and abs(pos) < self.cfg.center_band_ratio

        fire = False
        if in_band and det.complete:
            self.misses = 0
            if self.armed and self.prev_pos is not None:
                crossed = np.sign(pos) != np.sign(self.prev_pos) or abs(pos) < 0.01
                still = abs(pos - self.prev_pos) < 0.002
                if crossed or still:
                    fire = True
                    self.armed = False
        else:
            self.misses += 1
            if self.misses >= self.cfg.rearm_frames:
                self.armed = True
        self.prev_pos = pos
        return fire

    @property
    def part_complete(self) -> bool:
        return self.last_det is not None and self.last_det.complete

    def draw(self, img: np.ndarray) -> None:
        h, w = img.shape[:2]
        b = self.cfg.center_band_ratio
        color = (0, 200, 255) if self.armed else (120, 120, 120)
        if self.cfg.axis == "x":
            for x in (int(w * (0.5 - b)), int(w * (0.5 + b))):
                cv2.line(img, (x, 0), (x, h), color, 1, cv2.LINE_AA)
        else:
            for y in (int(h * (0.5 - b)), int(h * (0.5 + b))):
                cv2.line(img, (0, y), (w, y), color, 1, cv2.LINE_AA)
        if self.last_det is not None:
            s = img.shape[1] / (self.cfg.detect_width)
            c = (self.last_det.contour.astype(np.float32) * s).astype(np.int32)
            cv2.drawContours(img, [c], -1, (255, 200, 0) if self.last_det.complete else (100, 100, 100), 1, cv2.LINE_AA)
