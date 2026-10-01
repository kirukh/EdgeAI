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
from dataclasses import dataclass, field

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
    bbox: tuple[int, int, int, int]      # x, y, w, h in canvas coordinates
    detail: str = ""
    severity: float = 1.0                 # deviation relative to the tolerance
    channel: str = ""                     # lighting channel ("front"/"back") in dual-light mode

    @property
    def label(self) -> str:
        return DEFECT_LABELS.get(self.kind, self.kind)

    def to_dict(self) -> dict:
        return {"kind": self.kind, "label": self.label, "method": self.method,
                "bbox": [int(v) for v in self.bbox], "detail": self.detail,
                "severity": round(float(self.severity), 3), "channel": self.channel}


@dataclass
class MethodResult:
    method: str
    ok: bool
    score: float          # ≥ 1.0 means NOK (normalised to the threshold)
    defects: list[Defect] = field(default_factory=list)
    heatmap: np.ndarray | None = None     # float32 [0,1] on the canvas
    channel: str = ""
    standalone: float = 1.0               # score this method needs to decide alone (fusion rule)
    decisive: bool = False                # did this method contribute to the NOK decision?

    def to_dict(self) -> dict:
        return {"method": self.method, "label": METHOD_LABELS.get(self.method, self.method),
                "ok": bool(self.ok), "score": round(float(self.score), 3), "defects": len(self.defects),
                "channel": self.channel, "standalone": float(self.standalone), "decisive": bool(self.decisive)}


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


def _max_component_area(binary: np.ndarray) -> float:
    n, _, stats, _ = cv2.connectedComponentsWithStats(binary.astype(np.uint8), connectivity=8)
    return float(stats[1:, cv2.CC_STAT_AREA].max()) if n > 1 else 0.0


class Method:
    name = "base"

    def __init__(self, cfg: MethodConfig):
        self.cfg = cfg
        self.report: dict = {}

    def fit(self, samples: list[AlignedPart]) -> None:
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
    material = cv2.erode(a.filled_mask, np.ones((3, 3), np.uint8)) > 0   # inside the outline
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
        holes.append(Hole(
            cx=cx, cy=cy, area=area,
            eq_d=math.sqrt(4 * area / math.pi),
            aspect=max(rw, rh) / max(1.0, min(rw, rh)),
            bbox=cv2.boundingRect(c), mask=filled > 0,
        ))
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
    pairs = [(r, c) for r, c in zip(rows, cols) if cost[r, c] < big]
    un_e = [i for i in range(len(expected_xy)) if i not in {p[0] for p in pairs}]
    un_d = [j for j in range(len(holes)) if j not in {p[1] for p in pairs}]
    return pairs, un_e, un_d


class GeometryMethod(Method):
    name = "geometry"

    def fit(self, samples):
        min_area = self.cfg.hole_min_area_px
        detected = [find_holes(s, min_area)[0] for s in samples]
        counts = [len(d) for d in detected]
        mode = max(set(counts), key=counts.count)
        ref = detected[counts.index(mode)]
        exp_xy = np.array([[h.cx, h.cy] for h in ref], np.float64).reshape(-1, 2)
        exp_d = np.array([h.eq_d for h in ref], np.float64)

        n_e = len(ref)
        xs, ds, asp = [[] for _ in range(n_e)], [[] for _ in range(n_e)], [[] for _ in range(n_e)]
        masks = [np.zeros(samples[0].gray.shape, np.float32) for _ in range(n_e)]
        inconsistent = []
        for idx, holes in enumerate(detected):
            pairs, un_e, un_d = _match(exp_xy, exp_d, holes)
            if un_e or un_d:
                inconsistent.append(idx)
            for e, j in pairs:
                xs[e].append((holes[j].cx, holes[j].cy))
                ds[e].append(holes[j].eq_d)
                asp[e].append(holes[j].aspect)
                masks[e] += holes[j].mask
        n = len(samples)
        self.exp_xy = np.array([np.mean(x, axis=0) for x in xs]).reshape(-1, 2)
        self.pos_std = np.array([np.sqrt(np.mean(np.sum((np.array(x) - np.mean(x, 0)) ** 2, 1))) if len(x) > 1 else 0.0 for x in xs])
        self.exp_d = np.array([np.mean(d) for d in ds])
        self.d_std = np.array([np.std(d, ddof=1) if len(d) > 1 else 0.0 for d in ds])
        self.exp_aspect = np.array([np.mean(a) for a in asp])
        self.hole_masks = np.stack([(m / max(1, len(xs[i])) > 0.5).astype(np.uint8) for i, m in enumerate(masks)]) \
            if n_e else np.zeros((0,) + samples[0].gray.shape, np.uint8)
        self.ref_filled = (np.mean([s.filled_mask > 0 for s in samples], axis=0) > 0.5).astype(np.uint8)

        # Outline deviation: the largest deviation among the good parts is the baseline
        base = [_max_component_area(self._outline_diff(s)) for s in samples]
        self.outline_min_area = float(max(self.cfg.outline_min_area_px, 1.5 * max(base, default=0)))

        self.report = {
            "holes": n_e,
            "hole_diameters_px": [round(float(v), 1) for v in self.exp_d],
            "position_std_px": [round(float(v), 2) for v in self.pos_std],
            "inconsistent_references": inconsistent,
            # Holes whose position scatters strongly across the references → alignment
            # was inconsistent or the good parts themselves vary a lot.
            "high_scatter_holes": [i for i in range(n_e) if self.pos_std[i] > max(2.0, 0.1 * self.exp_d[i])],
            "outline_min_area_px": round(self.outline_min_area, 1),
            "n_samples": n,
        }

    def _outline_diff(self, s: AlignedPart) -> np.ndarray:
        xor = cv2.bitwise_xor(self.ref_filled * 255, s.filled_mask)
        return cv2.morphologyEx(xor, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))) > 0

    def _rim_edge(self, s: AlignedPart, edges: np.ndarray, i: int) -> float:
        tmpl = self.hole_masks[i]
        line = cv2.morphologyEx(tmpl, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
        band = cv2.dilate(line.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        return float(np.count_nonzero(edges[band])) / max(1, np.count_nonzero(line) / 2)

    def _pos_tol(self, e: int, mm_per_px: float | None) -> float:
        c = self.cfg
        if c.pos_tol_mm and mm_per_px:
            return c.pos_tol_mm / mm_per_px                 # absolute tolerance from the drawing
        return max(c.pos_tol_px, c.tol_sigma * self.pos_std[e])

    def _dia_tol(self, e: int, mm_per_px: float | None) -> float:
        c = self.cfg
        if c.dia_tol_mm and mm_per_px:
            return c.dia_tol_mm / mm_per_px
        return max(c.dia_tol_ratio * self.exp_d[e], c.tol_sigma * self.d_std[e])

    def score(self, s, sens=1.0, mm_per_px=None):
        c = self.cfg
        sens = max(float(sens), 1e-3)
        holes, _ = find_holes(s, c.hole_min_area_px)
        pairs, un_e, un_d = _match(self.exp_xy, self.exp_d, holes)
        defects: list[Defect] = []
        worst = 0.0          # graded deviations (scaled by the sensitivity)
        presence = 0.0       # missing/extra/blind holes are always NOK, independent of the sensitivity

        def L(px):           # length in mm (if calibrated) or px
            return f"{px * mm_per_px:.2f} mm" if mm_per_px else f"{px:.1f} px"

        def box_around(x, y, d):
            r = int(d * 0.75) + 4
            return (int(x - r), int(y - r), 2 * r, 2 * r)

        for e, j in pairs:
            h = holes[j]
            pos_err = math.hypot(h.cx - self.exp_xy[e, 0], h.cy - self.exp_xy[e, 1])
            pos_tol = self._pos_tol(e, mm_per_px) / sens
            d_dev = h.eq_d - self.exp_d[e]
            d_tol = self._dia_tol(e, mm_per_px) / sens
            a_err = abs(h.aspect - self.exp_aspect[e]) / self.exp_aspect[e]
            a_tol = c.shape_tol_ratio / sens
            worst = max(worst, pos_err / pos_tol, abs(d_dev) / d_tol, a_err / a_tol)
            if pos_err > pos_tol:
                defects.append(Defect("HOLE_POSITION", self.name, box_around(h.cx, h.cy, h.eq_d),
                                      f"offset {L(pos_err)} (tolerance {L(pos_tol)})", pos_err / pos_tol))
            if abs(d_dev) > d_tol:
                defects.append(Defect("HOLE_DIAMETER", self.name, box_around(h.cx, h.cy, h.eq_d),
                                      f"Ø {L(h.eq_d)} instead of {L(self.exp_d[e])} "
                                      f"({d_dev / self.exp_d[e] * 100:+.1f} %, tolerance ±{L(d_tol)})", abs(d_dev) / d_tol))
            elif a_err > a_tol:
                defects.append(Defect("HOLE_SHAPE", self.name, box_around(h.cx, h.cy, h.eq_d),
                                      f"aspect ratio {h.aspect:.2f} instead of {self.exp_aspect[e]:.2f}", a_err / a_tol))

        # Unmatched: a far-offset hole of the same size → position defect
        used_d = set()
        for e in list(un_e):
            best = None
            for j in un_d:
                if j in used_d:
                    continue
                ratio = holes[j].eq_d / self.exp_d[e]
                if 0.7 < ratio < 1.4:
                    dist = math.hypot(holes[j].cx - self.exp_xy[e, 0], holes[j].cy - self.exp_xy[e, 1])
                    if best is None or dist < best[1]:
                        best = (j, dist)
            if best is not None:
                j, dist = best
                used_d.add(j)
                un_e.remove(e)
                h = holes[j]
                tol = self._pos_tol(e, mm_per_px) / sens
                defects.append(Defect("HOLE_POSITION", self.name, box_around(h.cx, h.cy, h.eq_d),
                                      f"offset {L(dist)} (tolerance {L(tol)})", max(dist / tol, 2.0)))
                presence = max(presence, 2.0)

        edges = cv2.Canny(cv2.GaussianBlur(s.gray, (5, 5), 0), 40, 120) > 0 if un_e else None
        for e in un_e:
            x, y = self.exp_xy[e]
            rim = self._rim_edge(s, edges, e)
            if rim >= c.rim_edge_threshold:
                defects.append(Defect("BLIND_HOLE", self.name, box_around(x, y, self.exp_d[e]),
                                      f"edge present (rim ratio {rim:.2f}) but no see-through", 2.0))
            else:
                defects.append(Defect("MISSING_HOLE", self.name, box_around(x, y, self.exp_d[e]),
                                      f"no hole visible (rim ratio {rim:.2f})", 2.0))
            presence = max(presence, 2.0)

        for j in un_d:
            if j in used_d:
                continue
            h = holes[j]
            defects.append(Defect("EXTRA_HOLE", self.name, box_around(h.cx, h.cy, h.eq_d),
                                  f"Ø {L(h.eq_d)} at ({h.cx:.0f}, {h.cy:.0f})", 2.0))
            presence = max(presence, 2.0)

        outline = self._outline_diff(s)
        min_area = self.outline_min_area / sens
        comps = _components(outline, min_area)
        for area, bbox, _ in comps:
            a_txt = f"{area * mm_per_px ** 2:.1f} mm²" if mm_per_px else f"{area:.0f} px"
            defects.append(Defect("OUTLINE", self.name, bbox, f"{a_txt} deviation", area / min_area))
        worst = max(worst, _max_component_area(outline) / min_area)

        score = max(worst, presence)
        return MethodResult(self.name, bool(score < 1.0 and not defects), float(score), defects)

    def state(self):
        meta = {"outline_min_area": self.outline_min_area, "report": self.report}
        arrays = {"exp_xy": self.exp_xy, "pos_std": self.pos_std, "exp_d": self.exp_d, "d_std": self.d_std,
                  "exp_aspect": self.exp_aspect, "hole_masks": self.hole_masks, "ref_filled": self.ref_filled}
        return meta, arrays

    def load_state(self, meta, arrays):
        self.outline_min_area = meta["outline_min_area"]
        self.report = meta.get("report", {})
        for k in ("exp_xy", "pos_std", "exp_d", "d_std", "exp_aspect", "hole_masks", "ref_filled"):
            setattr(self, k, arrays[k])


# ---------------------------------------------------------------------------
# Reference image comparison (difference map)
# ---------------------------------------------------------------------------
class DiffMethod(Method):
    name = "diff"

    def _rep(self, s: AlignedPart) -> np.ndarray:
        r = reps.compute(s, self.cfg.diff_representation)
        b = self.cfg.diff_blur | 1
        return cv2.GaussianBlur(r, (b, b), 0) if b > 1 else r

    def _zmap(self, x, mean, std):
        return np.abs(x - mean) / std * self.region

    def fit(self, samples):
        X = np.stack([self._rep(s) for s in samples])
        filled = (np.mean([s.filled_mask > 0 for s in samples], axis=0) > 0.5).astype(np.uint8)
        self.region = (cv2.dilate(filled, np.ones((9, 9), np.uint8)) > 0).astype(np.float32)
        floor = self.cfg.diff_sigma_floor
        self.mean = X.mean(0)
        self.std = np.maximum(X.std(0, ddof=1) if len(X) > 1 else np.zeros_like(self.mean), floor)

        # Leave-one-out: how large do deviation areas get for good parts?
        loo = []
        if len(X) >= 3:
            for i in range(len(X)):
                rest = np.delete(X, i, axis=0)
                z = self._zmap(X[i], rest.mean(0), np.maximum(rest.std(0, ddof=1), floor))
                loo.append(_max_component_area(self._clean(z > self.cfg.diff_z_threshold)))
        self.min_area = float(max(self.cfg.diff_min_area_px, 1.5 * max(loo, default=0)))
        self.report = {"representation": self.cfg.diff_representation, "loo_max_area_px": [round(v, 1) for v in loo],
                       "min_area_px": round(self.min_area, 1)}

    @staticmethod
    def _clean(b):
        return cv2.morphologyEx(b.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    def score(self, s, sens=1.0, mm_per_px=None):
        sens = max(float(sens), 1e-3)
        z = self._zmap(self._rep(s), self.mean, self.std)
        zt = self.cfg.diff_z_threshold / sens
        min_area = self.min_area / sens
        comps = _components(self._clean(z > zt), min_area)

        def A(px):
            return f"{px * mm_per_px ** 2:.1f} mm²" if mm_per_px else f"{px:.0f} px"

        defects = [Defect("SURFACE", self.name, bbox, f"{A(area)}, max z = {float(z[m].max()):.1f}",
                          area / min_area) for area, bbox, m in comps]
        score = max([area / min_area for area, _, _ in comps], default=0.0)
        if not comps:
            score = _max_component_area(self._clean(z > zt)) / min_area
        heat = np.clip(z / (2 * zt), 0, 1).astype(np.float32)
        return MethodResult(self.name, not defects, float(score), defects, heat)

    def state(self):
        return ({"min_area": self.min_area, "report": self.report},
                {"mean": self.mean, "std": self.std, "region": self.region})

    def load_state(self, meta, arrays):
        self.min_area = meta["min_area"]
        self.report = meta.get("report", {})
        self.mean, self.std, self.region = arrays["mean"], arrays["std"], arrays["region"]


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
    def _basis(X: np.ndarray, variance: float):
        mean = X.mean(0)
        Xc = X - mean
        if len(X) < 2:
            return mean, np.zeros((0, X.shape[1]), np.float32)
        _, sv, vt = np.linalg.svd(Xc, full_matrices=False)
        ev = sv ** 2
        if ev.sum() <= 0:
            return mean, np.zeros((0, X.shape[1]), np.float32)
        k = int(np.searchsorted(np.cumsum(ev) / ev.sum(), variance) + 1)
        k = min(k, max(1, len(X) - 2))
        return mean, vt[:k].astype(np.float32)

    def _residual(self, v, mean, comps, size):
        c = v - mean
        r = c - (c @ comps.T) @ comps if len(comps) else c
        rmap = np.abs(r.reshape(size[1], size[0]))
        return cv2.GaussianBlur(rmap, (3, 3), 0)

    def fit(self, samples):
        vecs = [self._vec(s) for s in samples]
        self.size = vecs[0][1]
        X = np.stack([v[0].ravel() for v in vecs]).astype(np.float32)
        self.mean, self.comps = self._basis(X, self.cfg.pca_variance)
        loo = []
        if len(X) >= 4:
            for i in range(len(X)):
                m, c = self._basis(np.delete(X, i, axis=0), self.cfg.pca_variance)
                loo.append(float(self._residual(X[i], m, c, self.size).max()))
        if loo:
            self.threshold = float(max(self.cfg.pca_margin * max(loo), np.mean(loo) + 4 * np.std(loo)))
        else:
            self.threshold = self.cfg.pca_residual_threshold
        self.report = {"representation": self.cfg.pca_representation, "components": int(len(self.comps)),
                       "loo_scores": [round(v, 4) for v in loo], "threshold": round(self.threshold, 4)}

    def score(self, s, sens=1.0, mm_per_px=None):
        v, _ = self._vec(s)
        rmap = self._residual(v.ravel().astype(np.float32), self.mean, self.comps, self.size)
        thr = self.threshold / max(float(sens), 1e-3)
        score = float(rmap.max()) / thr
        h, w = s.gray.shape
        big = cv2.resize(rmap, (w, h), interpolation=cv2.INTER_LINEAR)
        defects = []
        if score >= 1.0:
            cell = (w / self.size[0]) * (h / self.size[1])
            for area, bbox, m in _components(big > thr, cell):
                defects.append(Defect("ANOMALY", self.name, bbox, f"residual {float(big[m].max()):.3f} (threshold {thr:.3f})",
                                      float(big[m].max()) / thr))
            if not defects:
                defects.append(Defect("ANOMALY", self.name, _bbox_from_mask(big >= big.max() * 0.9), "", score))
        heat = np.clip(big / (2 * thr), 0, 1).astype(np.float32)
        return MethodResult(self.name, bool(score < 1.0), float(score), defects, heat)

    def state(self):
        return ({"threshold": self.threshold, "size": list(self.size), "report": self.report},
                {"mean": self.mean, "comps": self.comps})

    def load_state(self, meta, arrays):
        self.threshold = meta["threshold"]
        self.size = tuple(meta["size"])
        self.report = meta.get("report", {})
        self.mean, self.comps = arrays["mean"], arrays["comps"]


METHODS: dict[str, type[Method]] = {"geometry": GeometryMethod, "diff": DiffMethod, "pca": PCAMethod}


def create(name: str, cfg: MethodConfig) -> Method:
    if name not in METHODS:
        raise ValueError(f"Unknown method '{name}'. Available: {', '.join(METHODS)}")
    return METHODS[name](cfg)
