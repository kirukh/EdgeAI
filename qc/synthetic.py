"""Synthetic part images for development, tests and the simulator.

Generates metal plates with holes on a conveyor belt – either with front
light ("front") or backlight ("back", light source below the belt) –
including the typical defects from the project description.

The images do not replace real photos, but they make it possible to test the
complete pipeline (learn → inspect → visualise) without hardware and to
compare the methods systematically.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field

import cv2
import numpy as np

# Dimensions in pixels relative to a 640 px wide image
PART_TYPES: dict[str, dict] = {
    "A": {  # metal plate with two holes
        "size": (300, 170),
        "holes": [
            {"shape": "circle", "x": -90, "y": 0, "r": 17},
            {"shape": "circle", "x": 80, "y": 28, "r": 17},
        ],
    },
    "B": {  # metal plate with two different cut-outs
        "size": (300, 170),
        "holes": [
            {"shape": "circle", "x": -85, "y": 18, "r": 22},
            {"shape": "rect", "x": 72, "y": -28, "w": 58, "h": 22},
        ],
    },
    "C": {  # equilateral triangle with three holes (one smaller → orientation is unique)
        "outline": [(-135, 78), (135, 78), (0, -156)],
        "holes": [
            {"shape": "circle", "x": -72, "y": 42, "r": 15},
            {"shape": "circle", "x": 72, "y": 42, "r": 15},
            {"shape": "circle", "x": 0, "y": -80, "r": 10},
        ],
    },
    "D": {  # square plate with four holes (one hole offset → orientation is unique)
        "size": (220, 220),
        "holes": [
            {"shape": "circle", "x": -65, "y": -65, "r": 14},
            {"shape": "circle", "x": 65, "y": -65, "r": 14},
            {"shape": "circle", "x": 65, "y": 65, "r": 14},
            {"shape": "circle", "x": -40, "y": 65, "r": 14},
        ],
    },
}

PART_LABELS = {
    "A": "plate with two holes",
    "B": "hole + slot",
    "C": "triangle with three holes",
    "D": "square with four holes",
}


def outline_of(spec: dict) -> np.ndarray:
    """Outline polygon of a part type in local coordinates (centre = 0,0)."""
    if "outline" in spec:
        return np.array(spec["outline"], np.float64)
    pw, ph = spec["size"]
    return np.array([[-pw / 2, -ph / 2], [pw / 2, -ph / 2], [pw / 2, ph / 2], [-pw / 2, ph / 2]], np.float64)


def _random_inside(rng, poly: np.ndarray, margin: float, extra_ok=lambda x, y: True):
    """Random point inside the polygon with at least ``margin`` px distance to the edge."""
    cnt = poly.astype(np.float32).reshape(-1, 1, 2)
    (x0, y0), (x1, y1) = poly.min(0), poly.max(0)
    for _ in range(500):
        x, y = rng.uniform(x0, x1), rng.uniform(y0, y1)
        if cv2.pointPolygonTest(cnt, (float(x), float(y)), True) >= margin and extra_ok(x, y):
            return x, y
    return float(poly[:, 0].mean()), float(poly[:, 1].mean())

DEFECTS = ["missing", "blind", "position", "diameter", "extra", "scratch", "stain", "corner"]

DEFECT_LABELS = {
    "missing": "Hole missing",
    "blind": "Hole not drilled through",
    "position": "Hole mispositioned",
    "diameter": "Wrong hole diameter",
    "extra": "Extra hole",
    "scratch": "Scratch",
    "stain": "Discoloration/rust",
    "corner": "Damaged corner",
}


@dataclass
class PartInstance:
    type_name: str
    outline: np.ndarray                    # outline polygon (local coordinates)
    holes: list[dict]                      # through holes
    blind_holes: list[dict] = field(default_factory=list)
    scratches: list[dict] = field(default_factory=list)
    stains: list[dict] = field(default_factory=list)
    corner_cut: float = 0.0
    corner_idx: int = 0
    defect: str | None = None
    texture_seed: int = 0


def make_part(type_name: str = "A", defect: str | None = None, rng: np.random.Generator | None = None) -> PartInstance:
    rng = rng or np.random.default_rng()
    spec = PART_TYPES[type_name]
    holes = copy.deepcopy(spec["holes"])
    # manufacturing scatter of good parts (sub-pixel up to ~1 px)
    for h in holes:
        h["x"] += rng.normal(0, 0.5)
        h["y"] += rng.normal(0, 0.5)
    poly = outline_of(spec)
    part = PartInstance(type_name, poly, holes, defect=defect, texture_seed=int(rng.integers(1 << 30)))

    if defect is None:
        return part
    idx = int(rng.integers(len(holes)))
    if defect == "missing":
        holes.pop(idx)
    elif defect == "blind":
        part.blind_holes.append(holes.pop(idx))
    elif defect == "position":
        # shift the hole 12–22 px, but keep it inside the part (≥ 6 px wall to the edge)
        h = holes[idx]
        r = h.get("r", max(h.get("w", 0), h.get("h", 0)) / 2)
        cnt = poly.astype(np.float32).reshape(-1, 1, 2)
        for _ in range(200):
            ang, d = rng.uniform(0, 2 * math.pi), rng.uniform(12, 22)
            x, y = h["x"] + d * math.cos(ang), h["y"] + d * math.sin(ang)
            if cv2.pointPolygonTest(cnt, (float(x), float(y)), True) >= r + 6:
                break
        h["x"], h["y"] = x, y
    elif defect == "diameter":
        f = rng.choice([0.7, 1.35])
        h = holes[idx]
        if h["shape"] == "circle":
            h["r"] *= f
        else:
            h["w"] *= f
            h["h"] *= f
    elif defect == "extra":
        x, y = _random_inside(rng, poly, 22, lambda x, y: all(math.hypot(x - h["x"], y - h["y"]) > 50 for h in holes))
        holes.append({"shape": "circle", "x": x, "y": y, "r": 10})
    elif defect == "scratch":
        x, y = _random_inside(rng, poly, 25)
        ang = rng.uniform(0, math.pi)
        L = rng.uniform(60, 120)
        part.scratches.append({"x": x, "y": y, "angle": ang, "length": L, "delta": rng.choice([-70, 60])})
    elif defect == "stain":
        x, y = _random_inside(rng, poly, 25)
        part.stains.append({"x": x, "y": y, "rx": rng.uniform(14, 24), "ry": rng.uniform(9, 16)})
    elif defect == "corner":
        part.corner_cut = rng.uniform(26, 40)
        part.corner_idx = int(rng.integers(len(poly)))
    else:
        raise ValueError(f"Unknown defect: {defect}")
    return part


def _draw_hole(img, h, cx, cy, value, thickness=cv2.FILLED):
    if h["shape"] == "circle":
        cv2.circle(img, (int(round((cx + h["x"]) * 8)), int(round((cy + h["y"]) * 8))),
                   int(round(h["r"] * 8)), value, thickness, cv2.LINE_AA, shift=3)
    else:
        x0, y0 = cx + h["x"] - h["w"] / 2, cy + h["y"] - h["h"] / 2
        pts = np.array([[x0, y0], [x0 + h["w"], y0], [x0 + h["w"], y0 + h["h"]], [x0, y0 + h["h"]]]) * 8
        pts = pts.round().astype(np.int32)
        if thickness == cv2.FILLED:
            cv2.fillPoly(img, [pts], value, cv2.LINE_AA, shift=3)
        else:
            cv2.polylines(img, [pts], True, value, thickness, cv2.LINE_AA, shift=3)


def render_layers(part: PartInstance, lighting: str, scale: float = 1.0):
    """Renders the part in local coordinates → (colour layer, alpha, centre)."""
    poly = part.outline
    pad = 50
    ext = np.abs(poly).max(0)
    W, H = int(2 * ext[0] + 2 * pad), int(2 * ext[1] + 2 * pad)
    cx, cy = W / 2, H / 2
    rng = np.random.default_rng(part.texture_seed)

    alpha = np.zeros((H, W), np.float32)
    pts = poly + [cx, cy]
    cv2.fillPoly(alpha, [(pts * 8).round().astype(np.int32)], 1.0, cv2.LINE_AA, shift=3)
    if part.corner_cut:
        # chip off the corner: triangle between the vertex and points on both adjacent edges
        c, i, n = part.corner_cut, part.corner_idx, len(pts)
        v, prv, nxt = pts[i], pts[i - 1], pts[(i + 1) % n]
        a = v + c * (prv - v) / np.linalg.norm(prv - v)
        b = v + c * (nxt - v) / np.linalg.norm(nxt - v)
        out = v + 3 * (v - (a + b) / 2) / max(1e-6, np.linalg.norm(v - (a + b) / 2))
        cv2.fillPoly(alpha, [np.array([a, out, b]).round().astype(np.int32)], 0.0)
    for h in part.holes:
        _draw_hole(alpha, h, cx, cy, 0.0)

    if lighting == "front":
        base = np.array([182, 178, 172], np.float32)   # BGR, slightly bluish metal
        streaks = rng.normal(0, 7, (H, 1)).astype(np.float32)
        streaks = cv2.GaussianBlur(np.repeat(streaks, W, axis=1), (1, 3), 0)
        grad = np.linspace(-10, 10, W, dtype=np.float32)[None, :]
        tex = streaks + grad + rng.normal(0, 3, (H, W)).astype(np.float32)
    else:
        base = np.array([58, 56, 54], np.float32)
        tex = rng.normal(0, 3, (H, W)).astype(np.float32)
    layer = base[None, None, :] + tex[:, :, None]

    for h in part.blind_holes:
        m = np.zeros((H, W), np.float32)
        _draw_hole(m, h, cx, cy, 1.0)
        rim = np.zeros((H, W), np.float32)
        _draw_hole(rim, h, cx, cy, 1.0, thickness=3)
        if lighting == "front":
            layer = layer * (1 - 0.30 * m[:, :, None])      # bottom of the blind hole slightly darker
            layer = layer * (1 - 0.45 * rim[:, :, None])    # shadow ring at the hole edge
        else:
            layer = layer + 6 * m[:, :, None]

    for s in part.scratches:
        dx, dy = math.cos(s["angle"]) * s["length"] / 2, math.sin(s["angle"]) * s["length"] / 2
        m = np.zeros((H, W), np.float32)
        p1 = (int(cx + s["x"] - dx), int(cy + s["y"] - dy))
        p2 = (int(cx + s["x"] + dx), int(cy + s["y"] + dy))
        cv2.line(m, p1, p2, 1.0, 2, cv2.LINE_AA)
        d = s["delta"] if lighting == "front" else s["delta"] * 0.15
        layer = layer + d * m[:, :, None]

    for s in part.stains:
        m = np.zeros((H, W), np.float32)
        cv2.ellipse(m, (int(cx + s["x"]), int(cy + s["y"])), (int(s["rx"]), int(s["ry"])), 20, 0, 360, 1.0, cv2.FILLED)
        m = cv2.GaussianBlur(m, (0, 0), 4)
        rust = np.array([40, 80, 150], np.float32) if lighting == "front" else np.array([45, 50, 60], np.float32)
        layer = layer * (1 - 0.8 * m[:, :, None]) + rust[None, None, :] * 0.8 * m[:, :, None]

    layer = np.clip(layer, 0, 255)
    if scale != 1.0:
        W2, H2 = int(W * scale), int(H * scale)
        layer = cv2.resize(layer, (W2, H2), interpolation=cv2.INTER_LINEAR)
        alpha = cv2.resize(alpha, (W2, H2), interpolation=cv2.INTER_LINEAR)
        cx, cy = cx * scale, cy * scale
    return layer, alpha, (cx, cy)


def belt_background(size: tuple[int, int], lighting: str, rng: np.random.Generator) -> np.ndarray:
    w, h = size
    if lighting == "front":
        base = np.array([44, 46, 48], np.float32)
        noise = cv2.GaussianBlur(rng.normal(0, 6, (h, w)).astype(np.float32), (0, 0), 1.2)
    else:
        base = np.array([236, 236, 234], np.float32)
        noise = rng.normal(0, 2, (h, w)).astype(np.float32)
    return np.clip(base[None, None, :] + noise[:, :, None], 0, 255)


def compose(
    part: PartInstance | None,
    pose: tuple[float, float, float],
    lighting: str = "front",
    frame_size: tuple[int, int] = (640, 480),
    rng: np.random.Generator | None = None,
    layers=None,
    background: np.ndarray | None = None,
) -> np.ndarray:
    """Places the part (pose: cx, cy, angle in degrees) on the belt and simulates camera effects."""
    rng = rng or np.random.default_rng()
    w, h = frame_size
    scale = w / 640.0
    frame = background.copy() if background is not None else belt_background(frame_size, lighting, rng)
    if part is not None:
        layer, alpha, (lcx, lcy) = layers if layers is not None else render_layers(part, lighting, scale)
        cx, cy, ang = pose
        M = cv2.getRotationMatrix2D((lcx, lcy), ang, 1.0)
        M[0, 2] += cx * scale - lcx
        M[1, 2] += cy * scale - lcy
        lw = cv2.warpAffine(layer, M, (w, h), flags=cv2.INTER_LINEAR)
        aw = cv2.warpAffine(alpha, M, (w, h), flags=cv2.INTER_LINEAR)[:, :, None]
        frame = frame * (1 - aw) + lw * aw

    gain = rng.uniform(0.95, 1.05)                       # small lighting fluctuation
    frame = frame * gain + rng.normal(0, 2.5, frame.shape).astype(np.float32)
    frame = cv2.GaussianBlur(np.clip(frame, 0, 255), (0, 0), 0.6)
    return frame.astype(np.uint8)


def random_pose(rng: np.random.Generator, allow_flip: bool = True, jitter: float = 30, max_angle: float = 8,
                any_angle: bool = False):
    """Random position/rotation on the belt. ``any_angle`` = parts arrive in any rotation (0–360°)."""
    if any_angle:
        ang = rng.uniform(0, 360)
    else:
        ang = rng.uniform(-max_angle, max_angle) + (180 if allow_flip and rng.random() < 0.5 else 0)
    return (320 + rng.uniform(-jitter, jitter), 240 + rng.uniform(-jitter * 0.7, jitter * 0.7), ang)


def generate_dataset(out_dir, part_type="A", lighting="front", n_train=15, n_test_good=20, n_per_defect=5, seed=0,
                     any_angle=False):
    """Writes train/ (good parts only) and test/ok, test/nok/<defect> as PNG."""
    from pathlib import Path

    rng = np.random.default_rng(seed)
    out = Path(out_dir)
    (out / "train").mkdir(parents=True, exist_ok=True)
    (out / "test" / "ok").mkdir(parents=True, exist_ok=True)
    for i in range(n_train):
        cv2.imwrite(str(out / "train" / f"good_{i:03d}.png"), compose(make_part(part_type, None, rng), random_pose(rng, any_angle=any_angle), lighting, rng=rng))
    for i in range(n_test_good):
        cv2.imwrite(str(out / "test" / "ok" / f"good_{i:03d}.png"), compose(make_part(part_type, None, rng), random_pose(rng, any_angle=any_angle), lighting, rng=rng))
    for d in DEFECTS:
        (out / "test" / "nok" / d).mkdir(parents=True, exist_ok=True)
        for i in range(n_per_defect):
            img = compose(make_part(part_type, d, rng), random_pose(rng, any_angle=any_angle), lighting, rng=rng)
            cv2.imwrite(str(out / "test" / "nok" / d / f"{d}_{i:03d}.png"), img)
    return out
