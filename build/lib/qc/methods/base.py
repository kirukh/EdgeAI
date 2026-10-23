"""Common parts of the inspection methods: defects, results, the ``Method`` interface and helpers
shared by geometry, diff and pca."""

from __future__ import annotations

from collections.abc import Collection, Sequence
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from ..alignment import AlignedPart
from ..config import MethodConfig

DEFECT_LABELS = {
    "MISSING_HOLE": "Hole missing",
    "BLIND_HOLE": "Hole not drilled through",
    "HOLE_POSITION": "Hole mispositioned",
    "HOLE_DIAMETER": "Wrong hole diameter",
    "HOLE_SHAPE": "Wrong cut-out shape",
    "EXTRA_HOLE": "Extra hole/cut-out",
    "OUTLINE": "Outline deviation",
    "SURFACE": "Deviation from reference image",
    "ANOMALY": "ML anomaly",
}

METHOD_LABELS = {
    "geometry": "Geometry (holes + outline)",
    "diff": "Reference image comparison",
    "pca": "ML anomaly detection (PCA)",
}


@dataclass
class Defect:
    """A defect found by a method, with its location on the canvas for the marking."""

    kind: str
    method: str
    bbox: tuple[int, int, int, int]  # x, y, w, h in canvas coordinates
    detail: str = ""
    severity: float = 1.0  # deviation relative to the tolerance
    channel: str = ""  # lighting channel ("front"/"back") in dual-light mode

    @property
    def label(self) -> str:
        """Plain-language name of the defect type."""
        return DEFECT_LABELS.get(self.kind, self.kind)

    def to_dict(self) -> dict[str, Any]:
        """Defect as a plain dict (log, UI)."""
        return {
            "kind": self.kind,
            "label": self.label,
            "method": self.method,
            "bbox": [int(v) for v in self.bbox],
            "detail": self.detail,
            "severity": round(float(self.severity), 3),
            "channel": self.channel,
        }


@dataclass
class MethodResult:
    """Result of one method for one image: OK/NOK, score (≥ 1 = NOK), defects and an optional heat map."""

    method: str
    ok: bool
    score: float  # ≥ 1.0 means NOK (normalised to the threshold)
    defects: list[Defect] = field(default_factory=list[Any])
    heatmap: np.ndarray | None = None  # float32 [0,1] on the canvas
    channel: str = ""
    standalone: float = 1.0  # score this method needs to decide alone (fusion rule)
    decisive: bool = False  # did this method contribute to the NOK decision?

    def to_dict(self) -> dict[str, Any]:
        """Result as a plain dict (log, UI)."""
        return {
            "method": self.method,
            "label": METHOD_LABELS.get(self.method, self.method),
            "ok": bool(self.ok),
            "score": round(float(self.score), 3),
            "defects": len(self.defects),
            "channel": self.channel,
            "standalone": float(self.standalone),
            "decisive": bool(self.decisive),
        }


def bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int]:
    """Bounding box (x, y, w, h) of the True pixels of a mask."""
    ys, xs = np.nonzero(mask)
    return int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)


Component = tuple[float, tuple[int, int, int, int], np.ndarray]  # area, bounding box, mask


def components(binary: np.ndarray, min_area: float) -> tuple[list[Component], float]:
    """Connected regions ≥ min_area → (list of (area, bbox, mask), area of the largest region of any size).
    One pass for both: the score of a method is the largest region, the defects are the ones ≥ min_area."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary.astype(np.uint8), connectivity=8)
    out: list[Component] = []
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            x, y, w, h = stats[i, :4]
            out.append((float(area), (int(x), int(y), int(w), int(h)), labels == i))
    return out, float(np.max(stats[1:, cv2.CC_STAT_AREA])) if n > 1 else 0.0


def robust_outliers(values: Sequence[float], floor: float, k: float, factor: float = 1.0) -> list[int]:
    """Indices of values that stand out clearly from the others (leave-one-out screening).

    Outlier = value ≥ ``floor`` (it would count as a defect at all) AND above
    median + k · MAD·1.4826 (robust spread, not pulled up by the outliers themselves,
    so even two or three bad references among 15 are found). ``factor`` raises the limit
    for less certain methods (ML: its stand-alone level, like in the decision rule).
    Needs at least 4 values.
    """
    v = np.asarray(values, np.float64)
    if len(v) < 4:
        return []
    med = float(np.median(v))
    spread = 1.4826 * float(np.median(np.abs(v - med)))
    return [i for i, x in enumerate(v) if x >= floor and x > factor * (med + k * spread)]


def max_component_area(binary: np.ndarray) -> float:
    """Area of the largest connected region (0 if there is none)."""
    n, _, stats, _ = cv2.connectedComponentsWithStats(binary.astype(np.uint8), connectivity=8)
    return float(np.max(stats[1:, cv2.CC_STAT_AREA])) if n > 1 else 0.0


class Method:
    """Interface of an inspection method: learn from good parts (``fit``), rate a part (``score``), save/load."""

    name = "base"

    def __init__(self, cfg: MethodConfig):
        self.cfg = cfg
        self.report: dict[str, Any] = {}
        # set by fit(): sample index → reasons why this reference differs clearly from the
        # other references (probably a defective part taught in by mistake)
        self.suspects: dict[int, list[str]] = {}

    def _flag(self, i: int, why: str) -> None:
        self.suspects.setdefault(int(i), []).append(why)

    def fit(self, samples: list[AlignedPart], accepted: Collection[int] = ()) -> None:
        """Learns from good parts. ``accepted`` = indices of parts the operator accepted as good
        although they deviate ("within our tolerance"): their deviation is tolerated *locally*
        (at that hole, that place of the image) – the global thresholds stay as they are, so
        accepting e.g. a shifted hole does not make the method blind for scratches."""
        raise NotImplementedError

    def score(self, sample: AlignedPart, sens: float = 1.0, mm_per_px: float | None = None) -> MethodResult:
        """``sens`` > 1 = more sensitive (tolerances/thresholds divided by sens).
        ``mm_per_px`` (from the camera calibration) switches the reported dimensions to mm."""
        raise NotImplementedError

    def state(self) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
        """What is saved with the model: (JSON-able settings, NumPy arrays)."""
        raise NotImplementedError

    def load_state(self, meta: dict[str, Any], arrays: dict[str, np.ndarray]) -> None:
        """Restores the method from ``state()`` of a saved model."""
        raise NotImplementedError


ACCEPT_MARGIN = 1.1  # a deviation the operator accepted is tolerated with 10 % margin


def split_accepted(n: int, accepted: Collection[int]) -> tuple[list[int], set[int]]:
    """→ (indices of the normal good parts, accepted indices). The nominal state (mean image,
    nominal geometry, thresholds) is learned from the normal parts only; without any normal part
    all parts count as normal."""
    acc = {i for i in accepted if 0 <= i < n}
    normal = [i for i in range(n) if i not in acc]
    return (normal, acc) if normal else (list(range(n)), set())


def tolerance_map(deviations: list[np.ndarray], conspicuous: list[np.ndarray], size: int) -> np.ndarray:
    """Local tolerance from the deviations of accepted parts.

    Only where an accepted part was really conspicuous (``conspicuous`` masks) – elsewhere its
    deviation is ordinary noise/texture, and tolerating it would make the method blind
    everywhere. There the largest deviation is spread over ``size`` px (position scatter of the
    next similar parts) and gets a 10 % margin; everywhere else the tolerance is 0."""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    parts = []
    for dev, mask in zip(deviations, conspicuous, strict=True):
        region = cv2.dilate(mask.astype(np.uint8), k) > 0
        parts.append(np.where(region, dev, 0).astype(np.float32))
    return cv2.dilate(np.max(np.stack(parts), axis=0), k) * ACCEPT_MARGIN
