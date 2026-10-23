"""``geometry`` – holes and outline against the nominal geometry learned from the good parts
(sub-pixel hole measurement, Hungarian matching, tolerances from the spread of the good parts)."""

from __future__ import annotations

import math
from collections.abc import Collection
from dataclasses import dataclass
from typing import NamedTuple

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from ..alignment import AlignedPart
from .base import (
    ACCEPT_MARGIN,
    Defect,
    Method,
    MethodResult,
    components,
    max_component_area,
    robust_outliers,
    split_accepted,
)


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
    cand: np.ndarray = ((a.norm < 0.5) & (inner > 0)).astype(np.uint8) * 255
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
        bx, by, bw, bh = cv2.boundingRect(c)
        cx, cy, area = _subpixel_moments(a, filled, material)
        holes.append(
            Hole(
                cx=cx,
                cy=cy,
                area=area,
                eq_d=math.sqrt(4 * area / math.pi),
                aspect=max(rw, rh) / max(1.0, min(rw, rh)),
                bbox=(bx, by, bw, bh),
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
        normal, acc = split_accepted(len(samples), accepted)
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
        base = [max_component_area(self._outline_diff(s)) for s in samples]
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

    def _rim_edge(self, edges: np.ndarray, i: int) -> float:
        tmpl = self.hole_masks[i]
        line = cv2.morphologyEx(tmpl, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
        band = cv2.dilate(line.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        return float(np.count_nonzero(edges[band])) / max(1.0, float(np.count_nonzero(line)) / 2)

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
            rim = self._rim_edge(edges, e)
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
        comps, largest = components(outline, min_area)
        for area, bbox, _ in comps:
            text = f"{area * mm_per_px**2:.1f} mm²" if mm_per_px else f"{area:.0f} px"
            defects.append(Defect("OUTLINE", self.name, bbox, f"{text} deviation", area / min_area))
        return defects, largest / min_area

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
