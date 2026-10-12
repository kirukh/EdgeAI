"""Inspection methods.

Three methods that can be used individually or combined:

``geometry`` – classical image processing
    Holes/cut-outs are segmented as background pixels inside the part
    outline and matched against the learned nominal geometry (Hungarian
    matching). Detects: missing holes, holes not drilled through (edge ring
    without see-through), mispositioned, too large/small, wrongly shaped and
    extra holes, as well as outline deviations.

``diff`` – reference image comparison / difference map
    Per-pixel mean and spread image from the good parts; a new part is
    compared as a z-score map. Finds surface defects (scratches,
    discoloration) and anything else that deviates from the reference.

``pca`` – machine learning (unsupervised anomaly detection)
    PCA subspace of the good parts; the reconstruction error of a new part is
    the anomaly score. The threshold is calibrated via leave-one-out on the
    good parts – no defect examples are required.

All methods operate on the aligned part (``AlignedPart``) and return
``Defect`` objects with a location (canvas coordinates) for visualisation.
"""

from __future__ import annotations

import math
from collections.abc import Collection
from dataclasses import dataclass, field
from typing import NamedTuple

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from . import representations as reps
from .alignment import AlignedPart
from .config import MethodConfig

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
    kind: str
    method: str
    bbox: tuple[int, int, int, int]  # x, y, w, h in canvas coordinates
    detail: str = ""
    severity: float = 1.0  # deviation relative to the tolerance
    channel: str = ""  # lighting channel ("front"/"back") in dual-light mode

    @property
    def label(self) -> str:
        return DEFECT_LABELS.get(self.kind, self.kind)

    def to_dict(self) -> dict:
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
    method: str
    ok: bool
    score: float  # ≥ 1.0 means NOK (normalised to the threshold)
    defects: list[Defect] = field(default_factory=list)
    heatmap: np.ndarray | None = None  # float32 [0,1] on the canvas
    channel: str = ""
    standalone: float = 1.0  # score this method needs to decide alone (fusion rule)
    decisive: bool = False  # did this method contribute to the NOK decision?

    def to_dict(self) -> dict:
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


def _bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    return int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)


def _components(binary: np.ndarray, min_area: float):
    """Connected regions ≥ min_area → list of (area, bbox, mask)."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary.astype(np.uint8), connectivity=8)
    out = []
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            x, y, w, h = stats[i, :4]
            out.append((float(area), (int(x), int(y), int(w), int(h)), labels == i))
    return out


def robust_outliers(values, floor: float, k: float, factor: float = 1.0) -> list[int]:
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


def _max_component_area(binary: np.ndarray) -> float:
    n, _, stats, _ = cv2.connectedComponentsWithStats(binary.astype(np.uint8), connectivity=8)
    return float(stats[1:, cv2.CC_STAT_AREA].max()) if n > 1 else 0.0


class Method:
    name = "base"

    def __init__(self, cfg: MethodConfig):
        self.cfg = cfg
        self.report: dict = {}
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

    def state(self) -> tuple[dict, dict[str, np.ndarray]]:
        raise NotImplementedError

    def load_state(self, meta: dict, arrays: dict[str, np.ndarray]) -> None:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------
@dataclass
class Hole:
    cx: float
    cy: float
    area: float
    eq_d: float
    aspect: float
    bbox: tuple[int, int, int, int]
    mask: np.ndarray


def find_holes(a: AlignedPart, min_area: float) -> tuple[list[Hole], np.ndarray]:
    """Segments through-openings: background brightness inside the outer contour."""
    k7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    inner = cv2.erode(a.filled_mask, k7)
    material = cv2.erode(a.filled_mask, np.ones((3, 3), np.uint8)) > 0  # inside the outline
    cand = ((a.norm < 0.5) & (inner > 0)).astype(np.uint8) * 255
    cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(cand, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    holes = []
    for c in contours:
        filled = np.zeros_like(cand)
        cv2.drawContours(filled, [c], -1, 255, cv2.FILLED)
        filled_area = float(np.count_nonzero(filled))
        if filled_area < min_area:
            continue
        # Ring-shaped regions (only a shadow edge, no see-through) are not openings
        if np.count_nonzero(cand[filled > 0]) / filled_area < 0.6:
            continue
        (_, _), (rw, rh), _ = cv2.minAreaRect(c)
        cx, cy, area = _subpixel_moments(a, filled, material)
        holes.append(
            Hole(
                cx=cx,
                cy=cy,
                area=area,
                eq_d=math.sqrt(4 * area / math.pi),
                aspect=max(rw, rh) / max(1.0, min(rw, rh)),
                bbox=cv2.boundingRect(c),
                mask=filled > 0,
            )
        )
    return holes, cand


def _subpixel_moments(a: AlignedPart, filled: np.ndarray, material: np.ndarray) -> tuple[float, float, float]:
    """Sub-pixel area and centre of an opening.

    The region around the opening is upsampled 4× (bilinear) and cut exactly at
    the 50 % level between material and background. The moments of this
    iso-contour give area and centre with a resolution of a fraction of a pixel,
    instead of the ±0.5 px steps of a binary pixel mask.
    """
    up = 4
    x, y, w, h = cv2.boundingRect(filled)
    pad = 3
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(filled.shape[1], x + w + pad), min(filled.shape[0], y + h + pad)
    roi = cv2.dilate(filled[y0:y1, x0:x1], _K_EDGE) > 0
    patch = np.where(roi & material[y0:y1, x0:x1], a.norm[y0:y1, x0:x1], 1.0).astype(np.float32)
    big = cv2.resize(patch, ((x1 - x0) * up, (y1 - y0) * up), interpolation=cv2.INTER_LINEAR)
    m = cv2.moments((big < 0.5).astype(np.uint8), binaryImage=True)
    if m["m00"] <= 0:
        m = cv2.moments(filled, binaryImage=True)
        return m["m10"] / m["m00"], m["m01"] / m["m00"], float(np.count_nonzero(filled))
    # upsampled pixel centre (u + 0.5) / up − 0.5 → original pixel coordinates
    cx = x0 + (m["m10"] / m["m00"] + 0.5) / up - 0.5
    cy = y0 + (m["m01"] / m["m00"] + 0.5) / up - 0.5
    return cx, cy, m["m00"] / (up * up)


_K_EDGE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))


def _match(expected_xy: np.ndarray, expected_d: np.ndarray, holes: list[Hole]):
    """Assigns detected holes to the nominal holes (Hungarian matching)."""
    if len(expected_xy) == 0 or len(holes) == 0:
        return [], list(range(len(expected_xy))), list(range(len(holes)))
    big = 1e6
    cost = np.full((len(expected_xy), len(holes)), big)
    for i, (x, y) in enumerate(expected_xy):
        gate = max(3.0 * expected_d[i], 40.0)
        for j, h in enumerate(holes):
            d = math.hypot(h.cx - x, h.cy - y)
            ratio = h.eq_d / max(expected_d[i], 1.0)
            if d < gate and 0.4 < ratio < 2.5:
                cost[i, j] = d / expected_d[i] + abs(math.log(ratio))
    rows, cols = linear_sum_assignment(cost)
    pairs = [(r, c) for r, c in zip(rows, cols, strict=True) if cost[r, c] < big]
    un_e = [i for i in range(len(expected_xy)) if i not in {p[0] for p in pairs}]
    un_d = [j for j in range(len(holes)) if j not in {p[1] for p in pairs}]
    return pairs, un_e, un_d


class HoleMeasurement(NamedTuple):
    """One hole of one reference, matched to a nominal hole."""

    x: float
    y: float
    d: float  # equivalent diameter
    aspect: float


def _match_references(
    detected: list[list[Hole]], ref: list[Hole], shape: tuple[int, ...], mask_from: Collection[int]
) -> tuple[list[dict[int, HoleMeasurement]], list[np.ndarray], list[int]]:
    """Assigns the holes of every reference to the holes of ``ref``.

    → (per nominal hole: {reference index: measurement}, hole masks summed over the references
    in ``mask_from``, references whose hole count does not match)."""
    exp_xy = np.array([[h.cx, h.cy] for h in ref], np.float64).reshape(-1, 2)
    exp_d = np.array([h.eq_d for h in ref], np.float64)
    per_hole: list[dict[int, HoleMeasurement]] = [{} for _ in ref]
    masks = [np.zeros(shape, np.float32) for _ in ref]
    inconsistent = []
    for idx, holes in enumerate(detected):
        pairs, un_e, un_d = _match(exp_xy, exp_d, holes)
        if un_e or un_d:
            inconsistent.append(idx)
        for e, j in pairs:
            h = holes[j]
            per_hole[e][idx] = HoleMeasurement(h.cx, h.cy, h.eq_d, h.aspect)
            if idx in mask_from:
                masks[e] += h.mask
    return per_hole, masks, inconsistent


ACCEPT_MARGIN = 1.1  # a deviation the operator accepted is tolerated with 10 % margin


def _split(n: int, accepted: Collection[int]) -> tuple[list[int], set[int]]:
    """→ (indices of the normal good parts, accepted indices). The nominal state (mean image,
    nominal geometry, thresholds) is learned from the normal parts only; without any normal part
    all parts count as normal."""
    acc = {i for i in accepted if 0 <= i < n}
    normal = [i for i in range(n) if i not in acc]
    return (normal, acc) if normal else (list(range(n)), set())


def _tolerance_map(deviations: list[np.ndarray], conspicuous: list[np.ndarray], size: int) -> np.ndarray:
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


def _length(px: float, mm_per_px: float | None) -> str:
    """Length for the defect text: mm (if the scale is known) or px."""
    return f"{px * mm_per_px:.2f} mm" if mm_per_px else f"{px:.1f} px"


def _box_around(x: float, y: float, d: float) -> tuple[int, int, int, int]:
    r = int(d * 0.75) + 4
    return (int(x - r), int(y - r), 2 * r, 2 * r)


class GeometryMethod(Method):
    name = "geometry"
    outline_tolerated: np.ndarray | None = None  # outline deviations accepted by the operator (mask)

    def fit(self, samples, accepted=()):
        normal, acc = _split(len(samples), accepted)
        detected = [find_holes(s, self.cfg.hole_min_area_px)[0] for s in samples]
        counts = [len(detected[i]) for i in normal]
        ref = detected[normal[counts.index(max(set(counts), key=counts.count))]]  # the usual hole count
        shape = samples[0].gray.shape
        per_hole, masks, inconsistent = _match_references(detected, ref, shape, set(normal))
        # nominal geometry from the normal good parts; accepted deviations widen the tolerance
        self._learn_nominal([{i: v for i, v in ph.items() if i not in acc} for ph in per_hole], masks, shape)
        self._learn_accepted(per_hole, acc)
        self.ref_filled = (np.mean([samples[i].filled_mask > 0 for i in normal], axis=0) > 0.5).astype(np.uint8)
        self.outline_tolerated = None
        tolerated = [self._outline_diff(samples[i]) for i in sorted(acc)]
        if tolerated:  # outline deviations of accepted parts are tolerated at that place
            mask = np.any(tolerated, axis=0).astype(np.uint8)
            self.outline_tolerated = cv2.dilate(mask, np.ones((7, 7), np.uint8))

        # Outline deviation: the largest deviation among the normal good parts is the baseline
        base = [_max_component_area(self._outline_diff(s)) for s in samples]
        normal_base = [base[i] for i in normal]
        self.outline_min_area = float(max(self.cfg.outline_min_area_px, 1.5 * max(normal_base, default=0)))
        self._screen(per_hole, inconsistent, base)

        n_e = len(ref)
        self.report = {
            "holes": n_e,
            "hole_diameters_px": [round(float(v), 1) for v in self.exp_d],
            "position_std_px": [round(float(v), 2) for v in self.pos_std],
            "inconsistent_references": inconsistent,
            # Holes whose position scatters strongly across the references → alignment
            # was inconsistent or the good parts themselves vary a lot.
            "high_scatter_holes": [i for i in range(n_e) if self.pos_std[i] > max(2.0, 0.1 * self.exp_d[i])],
            "outline_min_area_px": round(self.outline_min_area, 1),
            "n_samples": len(samples),
        }

    def _learn_nominal(self, per_hole: list[dict[int, HoleMeasurement]], masks: list[np.ndarray], shape) -> None:
        """Nominal position, diameter, shape and mask of every hole = mean over the references."""
        xs = [[(v.x, v.y) for v in ph.values()] for ph in per_hole]
        ds = [[v.d for v in ph.values()] for ph in per_hole]
        self.exp_xy = np.array([np.mean(x, axis=0) for x in xs]).reshape(-1, 2)
        self.pos_std = np.array(
            [np.sqrt(np.mean(np.sum((np.array(x) - np.mean(x, 0)) ** 2, 1))) if len(x) > 1 else 0.0 for x in xs]
        )
        self.exp_d = np.array([np.mean(d) for d in ds])
        self.d_std = np.array([np.std(d, ddof=1) if len(d) > 1 else 0.0 for d in ds])
        self.exp_aspect = np.array([np.mean([v.aspect for v in ph.values()]) for ph in per_hole])
        self.hole_masks = (
            np.stack([(m / max(1, len(per_hole[i])) > 0.5).astype(np.uint8) for i, m in enumerate(masks)])
            if per_hole
            else np.zeros((0, *shape), np.uint8)
        )

    def _screen(self, per_hole: list[dict[int, HoleMeasurement]], inconsistent, outline_areas) -> None:
        """Leave-one-out: is a reference a geometry defect compared with the OTHER references?"""
        c = self.cfg
        self.suspects = {}
        for i in inconsistent:
            self._flag(i, "hole count differs from the other references")
        for e, entries in enumerate(per_hole):
            for i, (x, y, d, _) in entries.items():
                others = np.array([v for k, v in entries.items() if k != i], np.float64)
                if len(others) < 2:
                    continue
                mxy = others[:, :2].mean(0)
                pos_sd = math.sqrt(float(np.mean(np.sum((others[:, :2] - mxy) ** 2, 1))))
                pos_tol = max(c.pos_tol_px, c.tol_sigma * pos_sd)
                md = float(others[:, 2].mean())
                d_tol = max(c.dia_tol_ratio * md, c.tol_sigma * float(others[:, 2].std(ddof=1)))
                pos_err = math.hypot(x - mxy[0], y - mxy[1])
                if pos_err > pos_tol:
                    self._flag(i, f"hole {e + 1} is {pos_err:.1f} px away from its position on the other references")
                elif abs(d - md) > d_tol:
                    self._flag(i, f"hole {e + 1} diameter {d:.1f} px instead of {md:.1f} px on the other references")
        for i in robust_outliers(outline_areas, c.outline_min_area_px, c.screen_k):
            self._flag(i, f"outline deviates ({outline_areas[i]:.0f} px)")

    def _learn_accepted(self, per_hole: list[dict[int, HoleMeasurement]], accepted: set[int]) -> None:
        """Deviations the operator accepted become the minimum tolerance of that hole: the
        accepted deviation + the normal measuring scatter (k·σ, as for the normal tolerance) –
        the next part with "the same" deviation measures a little differently."""
        n_e, k = len(per_hole), self.cfg.tol_sigma
        self.pos_accept, self.d_accept, self.aspect_accept = np.zeros(n_e), np.zeros(n_e), np.zeros(n_e)
        for e, entries in enumerate(per_hole):
            for i in accepted & entries.keys():
                v = entries[i]
                pos = math.hypot(v.x - self.exp_xy[e, 0], v.y - self.exp_xy[e, 1])
                self.pos_accept[e] = max(self.pos_accept[e], pos + k * self.pos_std[e])
                self.d_accept[e] = max(self.d_accept[e], abs(v.d - self.exp_d[e]) + k * self.d_std[e])
                rel = abs(v.aspect - self.exp_aspect[e]) / self.exp_aspect[e]
                self.aspect_accept[e] = max(self.aspect_accept[e], ACCEPT_MARGIN * rel)

    def _outline_diff(self, s: AlignedPart) -> np.ndarray:
        xor = cv2.bitwise_xor(self.ref_filled * 255, s.filled_mask)
        if self.outline_tolerated is not None:
            xor[self.outline_tolerated > 0] = 0
        return cv2.morphologyEx(xor, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))) > 0

    def _rim_edge(self, s: AlignedPart, edges: np.ndarray, i: int) -> float:
        tmpl = self.hole_masks[i]
        line = cv2.morphologyEx(tmpl, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
        band = cv2.dilate(line.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        return float(np.count_nonzero(edges[band])) / max(1, np.count_nonzero(line) / 2)

    def _pos_tol(self, e: int, mm_per_px: float | None) -> float:
        c = self.cfg
        if c.pos_tol_mm and mm_per_px:
            tol = c.pos_tol_mm / mm_per_px  # absolute tolerance from the drawing
        else:
            tol = max(c.pos_tol_px, c.tol_sigma * self.pos_std[e])
        return max(tol, self.pos_accept[e])  # never stricter than what the operator accepted

    def _dia_tol(self, e: int, mm_per_px: float | None) -> float:
        c = self.cfg
        if c.dia_tol_mm and mm_per_px:
            tol = c.dia_tol_mm / mm_per_px
        else:
            tol = max(c.dia_tol_ratio * self.exp_d[e], c.tol_sigma * self.d_std[e])
        return max(tol, self.d_accept[e])

    def score(self, s, sens=1.0, mm_per_px=None):
        sens = max(float(sens), 1e-3)
        holes, _ = find_holes(s, self.cfg.hole_min_area_px)
        pairs, un_e, un_d = _match(self.exp_xy, self.exp_d, holes)
        defects, worst = self._check_matched(holes, pairs, sens, mm_per_px)
        relocated, un_e, un_d = self._check_relocated(holes, un_e, un_d, sens, mm_per_px)
        defects += relocated + self._check_missing(s, un_e) + self._check_extra(holes, un_d, mm_per_px)
        outline_defects, outline_worst = self._check_outline(s, sens, mm_per_px)
        defects += outline_defects
        # missing/extra/relocated holes are always NOK, independent of the sensitivity
        presence = 2.0 if relocated or un_e or un_d else 0.0
        score = max(worst, outline_worst, presence)
        return MethodResult(self.name, bool(score < 1.0 and not defects), float(score), defects)

    def _check_matched(self, holes, pairs, sens, mm_per_px) -> tuple[list[Defect], float]:
        """Holes found at their nominal place: position, diameter, shape (graded by the sensitivity)."""
        defects: list[Defect] = []
        worst = 0.0
        for e, j in pairs:
            h = holes[j]
            box = _box_around(h.cx, h.cy, h.eq_d)
            pos_err = math.hypot(h.cx - self.exp_xy[e, 0], h.cy - self.exp_xy[e, 1])
            pos_tol = self._pos_tol(e, mm_per_px) / sens
            d_dev = h.eq_d - self.exp_d[e]
            d_tol = self._dia_tol(e, mm_per_px) / sens
            a_err = abs(h.aspect - self.exp_aspect[e]) / self.exp_aspect[e]
            a_tol = max(self.cfg.shape_tol_ratio, self.aspect_accept[e]) / sens
            worst = max(worst, pos_err / pos_tol, abs(d_dev) / d_tol, a_err / a_tol)
            if pos_err > pos_tol:
                text = f"offset {_length(pos_err, mm_per_px)} (tolerance {_length(pos_tol, mm_per_px)})"
                defects.append(Defect("HOLE_POSITION", self.name, box, text, pos_err / pos_tol))
            if abs(d_dev) > d_tol:
                text = (
                    f"Ø {_length(h.eq_d, mm_per_px)} instead of {_length(self.exp_d[e], mm_per_px)} "
                    f"({d_dev / self.exp_d[e] * 100:+.1f} %, tolerance ±{_length(d_tol, mm_per_px)})"
                )
                defects.append(Defect("HOLE_DIAMETER", self.name, box, text, abs(d_dev) / d_tol))
            elif a_err > a_tol:
                text = f"aspect ratio {h.aspect:.2f} instead of {self.exp_aspect[e]:.2f}"
                defects.append(Defect("HOLE_SHAPE", self.name, box, text, a_err / a_tol))
        return defects, worst

    def _check_relocated(self, holes, un_e, un_d, sens, mm_per_px) -> tuple[list[Defect], list[int], list[int]]:
        """A nominal hole without a match + an unmatched hole of the same size far away = position defect.
        Returns (defects, still missing nominal holes, still unexplained detected holes)."""
        defects: list[Defect] = []
        missing, used = list(un_e), set()
        for e in un_e:
            candidates = [
                (math.hypot(holes[j].cx - self.exp_xy[e, 0], holes[j].cy - self.exp_xy[e, 1]), j)
                for j in un_d
                if j not in used and 0.7 < holes[j].eq_d / self.exp_d[e] < 1.4
            ]
            if not candidates:
                continue
            dist, j = min(candidates)
            used.add(j)
            missing.remove(e)
            h, tol = holes[j], self._pos_tol(e, mm_per_px) / sens
            text = f"offset {_length(dist, mm_per_px)} (tolerance {_length(tol, mm_per_px)})"
            defects.append(
                Defect("HOLE_POSITION", self.name, _box_around(h.cx, h.cy, h.eq_d), text, max(dist / tol, 2.0))
            )
        return defects, missing, [j for j in un_d if j not in used]

    def _check_missing(self, s, missing) -> list[Defect]:
        """Nominal holes without see-through: an edge ring at the place = not drilled through."""
        if not missing:
            return []
        edges = cv2.Canny(cv2.GaussianBlur(s.gray, (5, 5), 0), 40, 120) > 0
        defects = []
        for e in missing:
            x, y = self.exp_xy[e]
            box = _box_around(x, y, self.exp_d[e])
            rim = self._rim_edge(s, edges, e)
            if rim >= self.cfg.rim_edge_threshold:
                defects.append(
                    Defect("BLIND_HOLE", self.name, box, f"edge present (rim ratio {rim:.2f}) but no see-through", 2.0)
                )
            else:
                defects.append(Defect("MISSING_HOLE", self.name, box, f"no hole visible (rim ratio {rim:.2f})", 2.0))
        return defects

    def _check_extra(self, holes, extra, mm_per_px) -> list[Defect]:
        return [
            Defect(
                "EXTRA_HOLE",
                self.name,
                _box_around(holes[j].cx, holes[j].cy, holes[j].eq_d),
                f"Ø {_length(holes[j].eq_d, mm_per_px)} at ({holes[j].cx:.0f}, {holes[j].cy:.0f})",
                2.0,
            )
            for j in extra
        ]

    def _check_outline(self, s, sens, mm_per_px) -> tuple[list[Defect], float]:
        outline = self._outline_diff(s)
        min_area = self.outline_min_area / sens
        defects = []
        for area, bbox, _ in _components(outline, min_area):
            text = f"{area * mm_per_px**2:.1f} mm²" if mm_per_px else f"{area:.0f} px"
            defects.append(Defect("OUTLINE", self.name, bbox, f"{text} deviation", area / min_area))
        return defects, _max_component_area(outline) / min_area

    def state(self):
        meta = {"outline_min_area": self.outline_min_area, "report": self.report}
        arrays = {
            "exp_xy": self.exp_xy,
            "pos_std": self.pos_std,
            "exp_d": self.exp_d,
            "d_std": self.d_std,
            "exp_aspect": self.exp_aspect,
            "hole_masks": self.hole_masks,
            "ref_filled": self.ref_filled,
            "pos_accept": self.pos_accept,
            "d_accept": self.d_accept,
            "aspect_accept": self.aspect_accept,
        }
        if self.outline_tolerated is not None:
            arrays["outline_tolerated"] = self.outline_tolerated
        return meta, arrays

    def load_state(self, meta, arrays):
        self.outline_min_area = meta["outline_min_area"]
        self.report = meta.get("report", {})
        for k in ("exp_xy", "pos_std", "exp_d", "d_std", "exp_aspect", "hole_masks", "ref_filled"):
            setattr(self, k, arrays[k])
        n_e = len(self.exp_d)  # models saved before operator acceptance existed: no accepted deviations
        for k in ("pos_accept", "d_accept", "aspect_accept"):
            setattr(self, k, arrays.get(k, np.zeros(n_e)))
        self.outline_tolerated = arrays.get("outline_tolerated")


# ---------------------------------------------------------------------------
# Reference image comparison (difference map)
# ---------------------------------------------------------------------------
class DiffMethod(Method):
    name = "diff"

    def _rep(self, s: AlignedPart) -> np.ndarray:
        r = reps.compute(s, self.cfg.diff_representation)
        b = self.cfg.diff_blur | 1
        return cv2.GaussianBlur(r, (b, b), 0) if b > 1 else r

    accept_dev: np.ndarray | None = None  # local tolerance from parts the operator accepted

    def _zmap(self, x, mean, std):
        d = np.abs(x - mean)
        if self.accept_dev is not None:
            d = np.maximum(d - self.accept_dev, 0)
        return d / std * self.region

    def fit(self, samples, accepted=()):
        normal, acc = _split(len(samples), accepted)
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
                loo.append(_max_component_area(self._clean(z > self.cfg.diff_z_threshold)))
        self.min_area = float(max(self.cfg.diff_min_area_px, 1.5 * max(loo, default=0)))
        self.suspects = {}
        for i in robust_outliers(loo, self.cfg.diff_min_area_px, self.cfg.screen_k):
            self._flag(normal[i], f"differs from the other references in the image comparison ({loo[i]:.0f} px)")
        if acc:  # accepted deviations are tolerated where they are – the global threshold stays
            devs = [np.abs(all_x[i] - self.mean) for i in sorted(acc)]
            marked = [d / self.std * self.region > self.cfg.diff_z_threshold / 2 for d in devs]
            self.accept_dev = _tolerance_map(devs, marked, 7)
        self.report = {
            "representation": self.cfg.diff_representation,
            "loo_max_area_px": [round(v, 1) for v in loo],
            "min_area_px": round(self.min_area, 1),
        }

    @staticmethod
    def _clean(b):
        return cv2.morphologyEx(b.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    def score(self, s, sens=1.0, mm_per_px=None):
        sens = max(float(sens), 1e-3)
        z = self._zmap(self._rep(s), self.mean, self.std)
        zt = self.cfg.diff_z_threshold / sens
        min_area = self.min_area / sens
        comps = _components(self._clean(z > zt), min_area)

        def area_text(px):
            return f"{px * mm_per_px**2:.1f} mm²" if mm_per_px else f"{px:.0f} px"

        defects = [
            Defect("SURFACE", self.name, bbox, f"{area_text(area)}, max z = {float(z[m].max()):.1f}", area / min_area)
            for area, bbox, m in comps
        ]
        score = max([area / min_area for area, _, _ in comps], default=0.0)
        if not comps:
            score = _max_component_area(self._clean(z > zt)) / min_area
        heat = np.clip(z / (2 * zt), 0, 1).astype(np.float32)
        return MethodResult(self.name, not defects, float(score), defects, heat)

    def state(self):
        return (
            {"min_area": self.min_area, "report": self.report},
            {"mean": self.mean, "std": self.std, "region": self.region}
            | ({"accept_dev": self.accept_dev} if self.accept_dev is not None else {}),
        )

    def load_state(self, meta, arrays):
        self.min_area = meta["min_area"]
        self.report = meta.get("report", {})
        self.mean, self.std, self.region = arrays["mean"], arrays["std"], arrays["region"]
        self.accept_dev = arrays.get("accept_dev")


# ---------------------------------------------------------------------------
# ML: PCA anomaly detection
# ---------------------------------------------------------------------------
class PCAMethod(Method):
    name = "pca"

    def _vec(self, s: AlignedPart):
        r = reps.compute(s, self.cfg.pca_representation)
        h, w = r.shape
        size = (self.cfg.pca_size, max(8, int(round(self.cfg.pca_size * h / w))))
        return cv2.resize(r, size, interpolation=cv2.INTER_AREA), size

    @staticmethod
    def _basis(data: np.ndarray, variance: float):
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

    def _residual(self, v, mean, comps, size):
        c = v - mean
        r = c - (c @ comps.T) @ comps if len(comps) else c
        rmap = np.abs(r.reshape(size[1], size[0]))
        return cv2.GaussianBlur(rmap, (3, 3), 0)

    accept_res: np.ndarray | None = None  # local tolerance from parts the operator accepted

    def _effective(self, rmap: np.ndarray) -> np.ndarray:
        return rmap if self.accept_res is None else np.maximum(rmap - self.accept_res, 0)

    def fit(self, samples, accepted=()):
        normal, acc = _split(len(samples), accepted)
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
            self.threshold = float(max(self.cfg.pca_margin * max(loo), np.mean(loo) + 4 * np.std(loo)))
        else:
            self.threshold = self.cfg.pca_residual_threshold
        if acc:  # accepted deviations are tolerated where they are – the global threshold stays
            res = [self._residual(all_x[i], self.mean, self.comps, self.size) for i in sorted(acc)]
            self.accept_res = _tolerance_map(res, [r > self.threshold / 2 for r in res], 3)
        self.report = {
            "representation": self.cfg.pca_representation,
            "components": len(self.comps),
            "loo_scores": [round(v, 4) for v in loo],
            "threshold": round(self.threshold, 4),
        }

    def score(self, s, sens=1.0, mm_per_px=None):
        v, _ = self._vec(s)
        rmap = self._effective(self._residual(v.ravel().astype(np.float32), self.mean, self.comps, self.size))
        thr = self.threshold / max(float(sens), 1e-3)
        score = float(rmap.max()) / thr
        h, w = s.gray.shape
        big = cv2.resize(rmap, (w, h), interpolation=cv2.INTER_LINEAR)
        defects = []
        if score >= 1.0:
            cell = (w / self.size[0]) * (h / self.size[1])
            for _area, bbox, m in _components(big > thr, cell):
                defects.append(
                    Defect(
                        "ANOMALY",
                        self.name,
                        bbox,
                        f"residual {float(big[m].max()):.3f} (threshold {thr:.3f})",
                        float(big[m].max()) / thr,
                    )
                )
            if not defects:
                defects.append(Defect("ANOMALY", self.name, _bbox_from_mask(big >= big.max() * 0.9), "", score))
        heat = np.clip(big / (2 * thr), 0, 1).astype(np.float32)
        return MethodResult(self.name, bool(score < 1.0), float(score), defects, heat)

    def state(self):
        return (
            {"threshold": self.threshold, "size": list(self.size), "report": self.report},
            {"mean": self.mean, "comps": self.comps}
            | ({"accept_res": self.accept_res} if self.accept_res is not None else {}),
        )

    def load_state(self, meta, arrays):
        self.threshold = meta["threshold"]
        self.size = tuple(meta["size"])
        self.report = meta.get("report", {})
        self.mean, self.comps = arrays["mean"], arrays["comps"]
        self.accept_res = arrays.get("accept_res")


METHODS: dict[str, type[Method]] = {"geometry": GeometryMethod, "diff": DiffMethod, "pca": PCAMethod}


def create(name: str, cfg: MethodConfig) -> Method:
    if name not in METHODS:
        raise ValueError(f"Unknown method '{name}'. Available: {', '.join(METHODS)}")
    return METHODS[name](cfg)
