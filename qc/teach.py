"""Teach-in session: collects the reference images of good parts for a new model."""

from __future__ import annotations

import numpy as np

from . import autosetup as AS
from .alignment import detect_part, resize_to_width, to_gray
from .config import LocalizationConfig
from .visualize import encode_jpg

THUMB_WIDTH = 200


class TeachIn:
    def __init__(self, target: int, loc_cfg: LocalizationConfig):
        self.target = target
        self.loc_cfg = loc_cfg
        self.name = ""
        self.auto_finish = False  # create the model by itself once ``target`` parts are captured
        self.captures: list[np.ndarray] = []
        self.thumbs: list[bytes] = []
        self.edges: list[float] = []  # sharpness (10–90 % edge width) of every reference

    def start(self, name: str, auto_finish: bool = False) -> None:
        name = name.strip()
        if not name:
            raise ValueError("Please enter a name for the part type.")
        self.clear()
        self.name, self.auto_finish = name, auto_finish

    def clear(self) -> None:
        self.captures, self.thumbs, self.edges = [], [], []

    @property
    def count(self) -> int:
        return len(self.captures)

    @property
    def complete(self) -> bool:
        return self.auto_finish and self.count >= self.target

    def add(self, frame: np.ndarray) -> bool:
        """Adds a reference image. False (image discarded) if no complete part is visible."""
        work = resize_to_width(frame, self.loc_cfg.work_width)
        gray = to_gray(work)
        det = detect_part(gray, self.loc_cfg)
        if det is None or not det.complete:
            return False
        self.captures.append(frame.copy())
        self.thumbs.append(encode_jpg(resize_to_width(frame, THUMB_WIDTH), 70))
        edge = AS.edge_width(gray, det.contour)
        if edge is not None:
            self.edges.append(edge)
        return True

    def remove(self, index: int | None = None) -> int:
        """Removes a reference (default: the last one). Returns its position."""
        if not self.captures:
            raise ValueError("No reference image to remove.")
        i = len(self.captures) - 1 if index is None else index
        if not 0 <= i < len(self.captures):
            raise ValueError("Reference image not found.")
        self.captures.pop(i)
        self.thumbs.pop(i)
        return i

    def edge_width(self) -> float | None:
        """Median edge width of the references (image sharpness), None if unknown."""
        return float(np.median(self.edges)) if self.edges else None
