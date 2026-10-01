"""Automatic camera setup – "lay a good part under the camera, the camera adjusts itself".

Runs once per part type when it is taught in; the result (a *camera profile*) is
stored with the model and applied whenever the model is selected:

1. **Static part** (belt stopped, one good part under the camera, full field of view)
   * white balance once, then locked;
   * exposure: the brightest relevant pixels (part + its surroundings, 99.5th
     percentile) are brought to ``target_level`` – bright, but nothing clipped;
     gain stays 1.0 (least noise) unless the exposure limit is reached;
   * autofocus on the part and lock (cameras with a focus motor, e.g. Camera Module 3);
   * size and position of the part → later zoom.
2. **First moving part** of the teach-in (still full field of view, not used as a reference)
   * belt direction (x or y in the image) and the line the parts travel on;
   * belt speed → longest exposure that keeps the motion blur below ``max_blur_px``
     (exposure shortened, gain raised to keep the brightness);
   * **digital zoom** (sensor crop): the image is cut to the part's path with
     margins, so the part covers many more pixels → finer measurement. The crop keeps
     the 4:3 aspect ratio, so pixels stay square.
3. **Teach-in parts**: sharpness is measured on every reference (edge width of the
   outline, 10–90 % rise). A blurry image is reported – cameras without a focus motor
   (Camera Module v2, HQ) must then be focused once by hand.

What cannot be automated: turning a lens by hand, moving the camera, the light itself.
"""
from __future__ import annotations

import math
import cv2
import numpy as np

from .config import AutoSetupConfig

FULL = (0.0, 0.0, 1.0, 1.0)          # crop as fractions of the sensor: x, y, w, h


# ------------------------------------------------------------------ exposure
def bright_level(frame: np.ndarray, contour: np.ndarray | None, scale: float = 1.0) -> float:
    """99.5th percentile of the brightest colour channel on the part and around it.
    Using the region (not the whole image) keeps belt reflections elsewhere out of it;
    including the surroundings makes it work for backlight (bright background) too."""
    img = frame.max(axis=2) if frame.ndim == 3 else frame
    if contour is not None:
        x, y, w, h = cv2.boundingRect((contour.astype(np.float32) * scale).astype(np.int32))
        mx, my = int(0.15 * w) + 4, int(0.15 * h) + 4
        img = img[max(0, y - my):y + h + my, max(0, x - mx):x + w + mx]
    if img.size == 0:
        return 0.0
    return float(np.percentile(img, 99.5))


def next_exposure(exposure_us: float, gain: float, level: float, cfg: AutoSetupConfig) -> tuple[int, float, bool]:
    """One step of the exposure control (the sensor is linear: brightness ∝ exposure × gain).
    Returns (exposure, gain, converged)."""
    if level >= 254.0:                               # clipped → the true level is unknown, halve
        product = exposure_us * gain * 0.5
        done = False
    else:
        factor = cfg.target_level / max(level, 1.0)
        product = exposure_us * gain * factor
        done = abs(factor - 1.0) < 0.04
    return split_exposure(product, cfg.max_exposure_us, cfg) + (done,)


def split_exposure(product: float, max_exposure_us: float, cfg: AutoSetupConfig) -> tuple[int, float]:
    """Exposure × gain product → (exposure, gain): exposure first (no noise), gain only above the limit."""
    e = float(np.clip(product, cfg.min_exposure_us, max_exposure_us))
    g = float(np.clip(product / e, 1.0, cfg.max_gain))
    return int(round(e)), round(g, 3)


# ------------------------------------------------------------------ sharpness
def edge_width(gray: np.ndarray, contour: np.ndarray, n: int = 72, half: int = 7) -> float | None:
    """Median 10–90 % rise distance (px) across the part outline – an absolute sharpness
    measure that does not depend on the scene (≈ 1–1.5 px for a sharp image, ≥ 3 px blurry)."""
    c = contour.reshape(-1, 2).astype(np.float32)
    if len(c) < 20:
        return None
    g = gray.astype(np.float32)
    idx = np.linspace(0, len(c), n, endpoint=False).astype(int)
    widths = []
    t = np.arange(-half, half + 0.01, 0.25, dtype=np.float32)
    for i in idx:
        p, a, b = c[i], c[(i - 3) % len(c)], c[(i + 3) % len(c)]
        tan = b - a
        norm = np.hypot(*tan)
        if norm < 1e-3:
            continue
        nx, ny = -tan[1] / norm, tan[0] / norm
        xs = (p[0] + t * nx).reshape(1, -1)
        ys = (p[1] + t * ny).reshape(1, -1)
        if xs.min() < 1 or ys.min() < 1 or xs.max() > g.shape[1] - 2 or ys.max() > g.shape[0] - 2:
            continue
        prof = cv2.remap(g, xs, ys, cv2.INTER_LINEAR).ravel()
        lo, hi = np.median(prof[:8]), np.median(prof[-8:])
        if abs(hi - lo) < 25:                        # too little contrast here (e.g. a hole touches)
            continue
        if hi < lo:
            prof, lo, hi = prof[::-1], hi, lo
        v = (prof - lo) / (hi - lo)
        # first crossing of 10 % and of 90 % from the dark side (monotone after smoothing)
        v = np.maximum.accumulate(v)
        i10 = np.searchsorted(v, 0.1)
        i90 = np.searchsorted(v, 0.9)
        if 0 < i10 < len(v) and 0 < i90 < len(v):
            widths.append((t[i90] - t[i10]))
    return float(np.median(widths)) if len(widths) >= 8 else None


# ------------------------------------------------------------------ crop / zoom
def part_extent(contour: np.ndarray) -> tuple[float, float, float]:
    """(cx, cy, diameter) of the smallest circle around the part, in pixels – rotation-invariant
    size, because the part may arrive in any rotation."""
    (cx, cy), r = cv2.minEnclosingCircle(contour.reshape(-1, 1, 2).astype(np.float32))
    return float(cx), float(cy), 2.0 * float(r)


def to_sensor(u: float, v: float, out_size: tuple[int, int], crop) -> tuple[float, float]:
    """Pixel in the output image → position as a fraction of the full sensor."""
    x, y, w, h = crop
    return x + u / out_size[0] * w, y + v / out_size[1] * h


def crop_for_path(center: tuple[float, float], diameter: tuple[float, float], axis: str,
                  cfg: AutoSetupConfig) -> tuple[tuple[float, float, float, float], float]:
    """Sensor crop (x, y, w, h as fractions) for a part of the given diameter (as fraction of the
    sensor width, height) travelling along ``axis`` through ``center``. Same aspect ratio as the
    sensor (w == h as fractions) → square pixels. Returns (crop, zoom)."""
    dw, dh = diameter
    along, across = cfg.margin_along, cfg.margin_across
    f = max(along * dw, across * dh) if axis == "x" else max(across * dw, along * dh)
    f = float(np.clip(f, 1.0 / cfg.max_zoom, 1.0))
    x = float(np.clip(center[0] - f / 2, 0.0, 1.0 - f))
    y = float(np.clip(center[1] - f / 2, 0.0, 1.0 - f))
    return (round(x, 5), round(y, 5), round(f, 5), round(f, 5)), round(1.0 / f, 3)


def fits(diameter: tuple[float, float], axis: str, cfg: AutoSetupConfig) -> bool:
    """Does the part with its margins fit into the full field of view at all?"""
    dw, dh = diameter
    return (cfg.margin_along * dw <= 1.0 and cfg.margin_across * dh <= 1.0) if axis == "x" else \
        (cfg.margin_across * dw <= 1.0 and cfg.margin_along * dh <= 1.0)


def path_from_track(points: list[tuple[float, float, float]]) -> dict | None:
    """Belt direction and speed from the centroid track of one passing part.
    ``points`` = (time s, x, y) in sensor fractions. Returns None if the part did not move."""
    if len(points) < 3:
        return None
    p = np.asarray(points, np.float64)
    dt = p[-1, 0] - p[0, 0]
    dx, dy = p[-1, 1] - p[0, 1], p[-1, 2] - p[0, 2]
    dist = math.hypot(dx, dy)
    if dist < 0.05 or dt <= 0:                      # moved less than 5 % of the image: belt stopped
        return None
    axis = "x" if abs(dx) >= abs(dy) else "y"
    across = float(np.median(p[:, 2] if axis == "x" else p[:, 1]))
    return {"axis": axis, "across": across, "speed": dist / dt,          # sensor fractions per second
            "direction": float(math.degrees(math.atan2(dy, dx)))}
