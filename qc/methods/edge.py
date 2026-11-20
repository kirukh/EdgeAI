"""``edge`` – the edges of the part, measured along the nominal contour (outline and cut-outs).

For milled and deburred parts the defects sit ON the edge: a wrong corner radius, an unclean
(chattered) milled edge, a burr, a chamfer that is too wide because the part was deburred too
deep. The image comparison is blind there (every sub-pixel of alignment error shows up at an
edge), and the outline comparison of ``geometry`` only sees deviations several pixels wide.

This method walks along the nominal contour and reads a short brightness profile across the edge
at every 2 px (inside → outside). From each profile it takes

* the **edge position** (sub-pixel, where the profile crosses the 50 % level) relative to the
  nominal contour → contour deviation: burrs, nicks, and – at corners – a different radius;
* the **roughness**: how much the edge position jitters from point to point → unclean milled edge;
* the **band** just inside the edge, measured from the edge that was found (so alignment errors
  cancel out) and relative to the flat surface behind it (so brightness differences between parts
  cancel out) → chamfer/rounding of the edge looks different: deburred too deep or not at all.

What a good edge looks like, and how much it scatters, is learned per contour point from the
good parts. A remaining small shift/rotation of the whole part is fitted and removed first.
"""

from __future__ import annotations

from collections.abc import Collection
from typing import Any

import cv2
import numpy as np
from scipy.ndimage import maximum_filter1d, minimum_filter1d, uniform_filter1d

from ..alignment import AlignedPart
from .base import ACCEPT_MARGIN, Defect, Method, MethodResult, robust_outliers, split_accepted

STEP = 2.0  # distance of the measuring points along the contour (px at 640 px working width)
PROFILE_STEP = 0.5  # sampling distance across the edge
REACH_IN, REACH_OUT = 14.0, 7.0  # profile length into the material / out of it
BAND_FROM, BAND_TO = 1.5, 8.0  # band inside the edge that is compared (distance from the edge)
SURFACE_FROM, SURFACE_TO = 9.0, 12.0  # flat surface behind the band: its brightness is the reference
BAND_WINDOW = 3.0  # a band deviation must be this deep (across the edge) – single bright/dark lines are texture
SMOOTH = 5  # measuring points averaged along the contour (noise)
BAND_SMOOTH = 9  # … for the band: a chamfer changes over longer stretches than a nick
ROUGH_WINDOW = 15  # measuring points over which the roughness is taken
FEATURES = ("shape", "rough", "band")
# The worst good part (leave-one-out) stays below 1/margin of the tolerance. The band is a brightness
# measurement in a textured surface – it scatters more from part to part than the edge position.
GOOD_MARGIN = {"shape": 1.5, "rough": 1.5, "band": 1.8}
KINDS = {"shape": "EDGE_SHAPE", "rough": "EDGE_ROUGH", "band": "EDGE_BAND"}


def _resample(contour: np.ndarray, step: float) -> np.ndarray:
    """Closed contour → points every ``step`` px of arc length."""
    c = contour.reshape(-1, 2).astype(np.float64)
    seg = np.linalg.norm(np.roll(c, -1, axis=0) - c, axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    n = max(8, int(s[-1] / step))
    at = np.arange(n) * s[-1] / n
    closed = np.vstack([c, c[:1]])
    return np.stack([np.interp(at, s, closed[:, 0]), np.interp(at, s, closed[:, 1])], axis=1)


def _normals(pts: np.ndarray, image: np.ndarray, probe: float) -> np.ndarray:
    """Unit normals of a closed point chain, pointing from the material (bright) to the background."""
    tangent = np.roll(pts, -3, axis=0) - np.roll(pts, 3, axis=0)
    tangent /= np.linalg.norm(tangent, axis=1, keepdims=True) + 1e-9
    normal = np.stack([tangent[:, 1], -tangent[:, 0]], axis=1)

    def level(sign: float) -> float:
        p = (pts + sign * probe * normal).astype(np.float32)
        return float(cv2.remap(image, p[:, 0], p[:, 1], cv2.INTER_LINEAR).mean())

    return normal if level(1.0) < level(-1.0) else -normal


class EdgeMethod(Method):
    """Edge inspection along the nominal contour: contour deviation, roughness and edge band."""

    name = "edge"

    # ------------------------------------------------------------------ measuring
    def _setup(self, mean_norm: np.ndarray) -> None:
        """Measuring points on all contours of the nominal part (outline and cut-outs)."""
        binary = (np.clip(mean_norm, 0, 1) > 0.5).astype(np.uint8)
        contours, _ = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        image = np.clip(mean_norm, 0, 1).astype(np.float32)
        largest = max((cv2.contourArea(c) for c in contours), default=0.0)
        pts, normals, outer = [], [], []
        seg: list[np.ndarray] = []
        for c in contours:
            if cv2.arcLength(c, True) < self.px(40):  # specks
                continue
            p = _resample(c, self.px(STEP))
            pts.append(p)
            normals.append(_normals(p, image, self.px(3)))
            seg.append(np.full(len(p), len(seg)))
            outer.append(np.full(len(p), cv2.contourArea(c) >= largest))
        if not pts:
            raise ValueError("No contour found on the reference parts.")
        self.pts, self.normals = np.vstack(pts), np.vstack(normals)
        self.seg, self.outer = np.concatenate(seg), np.concatenate(outer)
        self.centre = np.array([mean_norm.shape[1] / 2.0, mean_norm.shape[0] / 2.0])
        self._maps()

    def _maps(self) -> None:
        """Sampling grid: for every measuring point a profile from inside to outside."""
        self.t = np.arange(-self.px(REACH_IN), self.px(REACH_OUT) + 1e-6, self.px(PROFILE_STEP))
        grid = self.pts[:, None, :] + self.t[None, :, None] * self.normals[:, None, :]
        self.map_x, self.map_y = grid[..., 0].astype(np.float32), grid[..., 1].astype(np.float32)
        self.depths = np.arange(self.px(BAND_FROM), self.px(SURFACE_TO) + 1e-6, self.px(PROFILE_STEP))
        self.n_band = int(np.searchsorted(self.depths, self.px(BAND_TO), side="right"))
        self.n_surface = int(np.searchsorted(self.depths, self.px(SURFACE_FROM)))
        # effect of a small shift / rotation of the whole part on the edge position of every point
        r = self.pts - self.centre
        self.rigid = np.stack(
            [self.normals[:, 0], self.normals[:, 1], self.normals[:, 0] * -r[:, 1] + self.normals[:, 1] * r[:, 0]],
            axis=1,
        )

    def _along(self, values: np.ndarray, fn: Any, size: int) -> np.ndarray:
        """Applies a 1-D filter along every (closed) contour separately."""
        out = np.empty_like(values, dtype=np.float64)
        for k in range(int(self.seg.max()) + 1):
            sel = self.seg == k
            out[sel] = fn(values[sel].astype(np.float64), size=min(size, int(sel.sum())), mode="wrap", axis=0)
        return out

    def _measure(self, s: AlignedPart) -> tuple[np.ndarray, np.ndarray]:
        """→ (edge position per point relative to the nominal contour, band profile per point).
        Points without an edge within reach get the position NaN.

        Cut-outs are measured where they actually are: WHERE a hole sits is checked by ``geometry``
        with its own tolerance – here only the shape of its edge counts. A cut-out that sits
        off its nominal place is therefore measured a second time around its real position."""
        # not clipped at 1: a chamfer usually reflects MORE light than the flat surface
        image = np.clip(s.norm, 0, 1.5).astype(np.float32)
        pos, band = self._read(image, self.map_x, self.map_y)
        for k in np.unique(self.seg[~self.outer]):
            rows = self.seg == k
            ok = rows & np.isfinite(pos)
            if ok.sum() < 8:
                continue
            shift, *_ = np.linalg.lstsq(self.normals[ok], pos[ok], rcond=None)
            if np.hypot(*shift) > self.px(0.5):
                dx, dy = np.float32(shift[0]), np.float32(shift[1])
                pos[rows], band[rows] = self._read(image, self.map_x[rows] + dx, self.map_y[rows] + dy)
        return pos, band

    def _read(self, image: np.ndarray, map_x: np.ndarray, map_y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Profiles across the edge → (sub-pixel edge position, band profile behind the edge)."""
        prof = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR)
        a, b = prof[:, :-1], prof[:, 1:]
        crossing = (a >= 0.5) & (b < 0.5)
        frac = (a - 0.5) / np.maximum(a - b, 1e-6)
        step = self.t[1] - self.t[0]
        t_cross = np.where(crossing, self.t[:-1][None, :] + frac * step, np.inf)
        idx = np.argmin(np.abs(t_cross), axis=1)
        pos = t_cross[np.arange(len(idx)), idx]
        found = np.isfinite(pos)
        pos = np.where(found, pos, np.nan)
        # band: profile values at fixed distances INSIDE the measured edge, relative to the surface behind
        at = (np.where(found, pos, 0.0)[:, None] - self.depths[None, :] - self.t[0]) / step
        lo = np.clip(np.floor(at).astype(np.int64), 0, prof.shape[1] - 2)
        w = np.clip(at - lo, 0.0, 1.0)
        rows = np.arange(len(pos))[:, None]
        inner = prof[rows, lo] * (1 - w) + prof[rows, lo + 1] * w
        band = inner[:, : self.n_band] - inner[:, self.n_surface :].mean(axis=1, keepdims=True)
        return pos, band.astype(np.float32)

    def _remove_pose(self, pos: np.ndarray) -> np.ndarray:
        """Removes a remaining shift/rotation of the whole part (fitted on the outline, robustly) and
        the remaining shift of every cut-out (see ``_measure``)."""
        use = self.outer & np.isfinite(pos)
        out = pos.copy()
        for _ in range(2):
            if use.sum() < 12:
                break
            x, *_ = np.linalg.lstsq(self.rigid[use], pos[use], rcond=None)
            out = pos - self.rigid @ x
            limit = 3.0 * max(float(np.nanmedian(np.abs(out[use]))), self.px(0.2))
            use = self.outer & np.isfinite(pos) & (np.abs(out) < limit)
        for k in np.unique(self.seg[~self.outer]):
            sel = (self.seg == k) & np.isfinite(out)
            if sel.sum() >= 8:
                shift, *_ = np.linalg.lstsq(self.normals[sel], out[sel], rcond=None)
                out[self.seg == k] -= self.normals[self.seg == k] @ shift
        return out

    def _raw(self, s: AlignedPart) -> tuple[dict[str, np.ndarray], np.ndarray]:
        """→ (deviation of every measuring point for the three checks, signed contour deviation)."""
        pos, band = self._measure(s)
        dev = self._remove_pose(pos) - self.pos_mean
        missing = ~np.isfinite(dev)
        dev = np.where(missing, self.px(REACH_OUT), dev)  # no edge within reach = far off
        smooth = self._along(dev, uniform_filter1d, SMOOTH)
        fine = dev - self._along(dev, uniform_filter1d, 2 * SMOOTH - 1)
        rough = np.sqrt(np.maximum(self._along(np.where(missing, 0.0, fine) ** 2, uniform_filter1d, ROUGH_WINDOW), 0))
        # band: signed deviation, averaged along the edge and over a window across it
        across = max(2, int(round(BAND_WINDOW / PROFILE_STEP)))
        diff = self._along(band - self.band_mean, uniform_filter1d, BAND_SMOOTH)
        band_dev = np.abs(uniform_filter1d(diff, size=across, axis=1, mode="nearest")).max(axis=1)
        band_dev = np.where(self._along(missing.astype(np.float64), maximum_filter1d, SMOOTH) > 0, 1.0, band_dev)
        return {"shape": np.abs(smooth), "rough": rough, "band": band_dev}, smooth

    # ------------------------------------------------------------------ learning
    def fit(self, samples: list[AlignedPart], accepted: Collection[int] = ()) -> None:
        """Nominal edge position and band per point, tolerances from the scatter of the good parts."""
        c = self.cfg
        normal, acc = split_accepted(len(samples), accepted)
        self._setup(np.mean([samples[i].norm for i in normal], axis=0))
        measured = [self._measure(s) for s in samples]
        pos = np.stack([self._remove_pose(m[0]) for m in measured])
        self.pos_mean = np.nan_to_num(np.nanmean(pos[normal], axis=0))
        self.band_mean = np.mean([measured[i][1] for i in normal], axis=0).astype(np.float32)
        n = len(self.pts)
        self.accept = {f: np.zeros(n) for f in FEATURES}
        raw = [self._raw(s)[0] for s in samples]
        floor = {"shape": self.px(c.edge_tol_px), "rough": self.px(c.edge_rough_px), "band": c.edge_band_tol}
        self.tol = {}
        for f in FEATURES:
            v = np.stack([raw[i][f] for i in normal])
            tol = self._local_tolerance(v, floor[f])
            # … and every good part as a whole must stay clearly inside the tolerance learned from
            # the OTHER good parts (leave-one-out). This bounds the largest deviation anywhere on
            # the contour, which the per-point statistics of a few parts cannot.
            loo = [
                self._worst(v[k] / self._local_tolerance(np.delete(v, k, axis=0), floor[f]), f) for k in range(len(v))
            ]
            if len(v) < 4:
                loo = [0.0]
            self.tol[f] = tol * max(1.0, GOOD_MARGIN[f] * max(loo), float(np.mean(loo) + 6 * np.std(loo)))
        runs = {f: [self._worst(r[f] / self.tol[f], f) for r in raw] for f in FEATURES}
        self.suspects = {}
        for f, text in (("shape", "contour"), ("rough", "edge roughness"), ("band", "edge band (chamfer)")):
            for i in robust_outliers([runs[f][k] for k in normal], 1.0, c.screen_k):
                self._flag(normal[i], f"{text} differs from the other references ({runs[f][normal[i]]:.1f}× tolerance)")
        for f in FEATURES:  # deviations the operator accepted are tolerated where they are
            for i in sorted(acc):
                over = np.where(raw[i][f] > self.tol[f] / 2, raw[i][f] * ACCEPT_MARGIN, 0.0)
                self.accept[f] = np.maximum(self.accept[f], self._along(over, maximum_filter1d, 2 * SMOOTH + 1))
        self.report = {
            "measuring_points": n,
            "contours": int(self.seg.max()) + 1,
            "tolerance_px": round(float(np.median(self.tol["shape"])), 2),
            "roughness_px": round(float(np.median(self.tol["rough"])), 2),
            "band_tolerance": round(float(np.median(self.tol["band"])), 3),
        }

    def _local_tolerance(self, values: np.ndarray, floor: float) -> np.ndarray:
        """Per measuring point: mean + k·σ of the good parts, at least ``floor`` – wider where the good
        parts themselves vary more. A point never gets a tighter tolerance than its neighbours."""
        spread = values.std(axis=0, ddof=1) if len(values) > 1 else np.zeros(values.shape[1])
        level = self._along(values.mean(axis=0) + self.cfg.tol_sigma * spread, maximum_filter1d, SMOOTH)
        tol: np.ndarray = np.maximum(floor, level)
        return tol

    def _run(self, f: str) -> int:
        """Measuring points a deviation must cover in a row to count."""
        length = self.cfg.edge_band_run_px if f == "band" else self.cfg.edge_min_run_px
        return max(2, int(round(length / STEP)))

    def _worst(self, ratio: np.ndarray, f: str) -> float:
        """Largest ratio that is kept up over the minimum run length."""
        return float(self._along(ratio, minimum_filter1d, self._run(f)).max())

    # ------------------------------------------------------------------ inspection
    def score(self, s: AlignedPart, sens: float = 1.0, mm_per_px: float | None = None) -> MethodResult:
        """Runs of measuring points outside their tolerance are edge defects."""
        raw, signed = self._raw(s)
        sens = max(float(sens), 1e-3)
        ratios = {f: np.maximum(raw[f] - self.accept[f], 0) / (self.tol[f] / sens) for f in FEATURES}
        defects: list[Defect] = []
        score = 0.0
        for f in FEATURES:
            score = max(score, self._worst(ratios[f], f))
            # every point of a run that stays above the tolerance over the minimum length
            core = self._along(ratios[f], minimum_filter1d, self._run(f)) > 1.0
            hit = self._along(core.astype(np.float64), maximum_filter1d, self._run(f)) > 0
            defects += self._defects(f, ratios[f], hit, signed, mm_per_px)
        defects.sort(key=lambda d: -d.severity)
        return MethodResult(self.name, not defects, score, defects)

    def _defects(
        self, f: str, ratio: np.ndarray, hit: np.ndarray, signed: np.ndarray, mm_per_px: float | None
    ) -> list[Defect]:
        out = []
        for k in range(int(self.seg.max()) + 1):
            index = np.flatnonzero(self.seg == k)
            h = hit[index]
            if not h.any():
                continue
            if h.all():
                runs = [index]
            else:
                start = int(np.flatnonzero(~h)[0])  # begin outside a run, so runs do not wrap around
                order = np.roll(index, -start)
                hh = np.roll(h, -start)
                edges = np.flatnonzero(np.diff(np.concatenate([[0], hh.astype(np.int8), [0]])))
                runs = [order[a:b] for a, b in zip(edges[::2], edges[1::2], strict=True)]
            for run in runs:
                p = self.pts[run]
                pad = self.px(6)
                x0, y0 = p.min(axis=0) - pad
                x1, y1 = p.max(axis=0) + pad
                length = len(run) * self.px(STEP)
                out.append(
                    Defect(
                        KINDS[f],
                        self.name,
                        (int(x0), int(y0), int(x1 - x0), int(y1 - y0)),
                        self._text(f, run, length, signed, mm_per_px),
                        float(ratio[run].max()),
                    )
                )
        return out

    def _text(self, f: str, run: np.ndarray, length: float, signed: np.ndarray, mm_per_px: float | None) -> str:
        def size(px: float) -> str:
            return f"{px * mm_per_px:.2f} mm" if mm_per_px else f"{px:.1f} px"

        where = "outline" if self.outer[run[0]] else "cut-out"
        if f == "shape":
            worst = signed[run][np.argmax(np.abs(signed[run]))]
            turn = self.normals[run[0]] @ self.normals[run[-1]]
            what = "radius/corner" if turn < 0.9 else "edge"
            side = "material missing" if worst < 0 else "material standing proud"
            return f"{where}: {what} off by {size(abs(float(worst)))} over {size(length)} ({side})"
        if f == "rough":
            return f"{where}: edge not clean over {size(length)}"
        return f"{where}: chamfer/rounding of the edge differs over {size(length)}"

    # ------------------------------------------------------------------ persistence
    def state(self) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
        """Measuring points, nominal values, tolerances and accepted deviations."""
        arrays = {
            "pts": self.pts,
            "normals": self.normals,
            "seg": self.seg,
            "outer": self.outer,
            "centre": self.centre,
            "pos_mean": self.pos_mean,
            "band_mean": self.band_mean,
        }
        for f in FEATURES:
            arrays[f"tol_{f}"] = self.tol[f]
            arrays[f"accept_{f}"] = self.accept[f]
        return {"report": self.report}, arrays

    def load_state(self, meta: dict[str, Any], arrays: dict[str, np.ndarray]) -> None:
        """Restores the measuring points and tolerances."""
        self.report = meta.get("report", {})
        for k in ("pts", "normals", "seg", "outer", "centre", "pos_mean", "band_mean"):
            setattr(self, k, arrays[k])
        self.tol = {f: arrays[f"tol_{f}"] for f in FEATURES}
        self.accept = {f: arrays[f"accept_{f}"] for f in FEATURES}
        self._maps()
