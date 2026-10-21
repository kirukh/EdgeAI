"""Conveyor trigger: fires exactly one capture per part.

Two modes (``trigger.mode``):

* ``pass``  – continuous belt: capture when the part crosses the image centre, or when it rests
  inside the centre band.
* ``still`` – stop-and-go belt: capture only once the part has rested inside the centre band for
  ``still_frames`` frames, so the image is never taken while the part moves or brakes. A part that
  leaves the band without stopping is still captured (in motion) and counted in ``moving_captures``.
"""

from __future__ import annotations

from collections import deque
from dataclasses import replace

import cv2
import numpy as np

from .alignment import PartDetection, detect_part, resize_to_width, to_gray
from .config import LocalizationConfig, TriggerConfig

STILL_TOLERANCE = 0.004  # "still": centroid movement over still_frames frames (fraction of the image size)


class PartTrigger:
    """Fires exactly one capture per part – when it crosses the image centre
    or rests inside the centre band (conveyor stopped)."""

    def __init__(self, cfg: TriggerConfig, loc: LocalizationConfig):
        self.cfg = cfg
        self.loc = replace(loc, border_margin_px=2)
        self.armed = True
        self.prev_pos: float | None = None
        self.misses = 0
        self.last_det: PartDetection | None = None
        self.scale = 1.0
        self.track: deque[float] = deque(maxlen=max(2, cfg.still_frames))  # positions in the centre band
        self.moving_captures = 0  # "still": parts that did not stop and were captured in motion

    def update(self, frame: np.ndarray) -> bool:
        small = resize_to_width(frame, self.cfg.detect_width)
        self.scale = frame.shape[1] / small.shape[1]
        det = detect_part(to_gray(small), self.loc)
        self.last_det = det
        axis = 0 if self.cfg.axis == "x" else 1
        size = small.shape[1 - axis]
        pos = det.centroid[axis] / size - 0.5 if det is not None else None

        fire = False
        if det is not None and pos is not None and abs(pos) < self.cfg.center_band_ratio and det.complete:
            self.misses = 0
            self.track.append(pos)
            if self.armed:
                fire = self._still() if self.cfg.mode == "still" else self._crossed_or_still(pos)
        else:
            if self.cfg.mode == "still" and self.armed and self.track and det is not None and det.complete:
                fire = True  # left the centre band without stopping – capture it anyway, in motion
                self.moving_captures += 1
            self.track.clear()
            self.misses += 1
            if self.misses >= self.cfg.rearm_frames:
                self.armed = True
        if fire:
            self.armed = False
        self.prev_pos = pos
        return fire

    def _crossed_or_still(self, pos: float) -> bool:
        prev = self.prev_pos
        if prev is None:
            return False
        return bool(np.sign(pos) != np.sign(prev) or abs(pos) < 0.01 or abs(pos - prev) < 0.002)

    def _still(self) -> bool:
        if self.track.maxlen != self.cfg.still_frames:  # setting changed at run time
            self.track = deque(self.track, maxlen=max(2, self.cfg.still_frames))
        return len(self.track) == self.track.maxlen and max(self.track) - min(self.track) < STILL_TOLERANCE

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
