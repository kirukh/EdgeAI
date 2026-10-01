"""Find the part and bring it into a normalised pose.

Steps per image:
1. Grayscale + Otsu threshold → separate the part from the conveyor
   (automatic polarity: front light = bright part on a dark belt,
   backlight = dark part in front of a bright background).
2. Largest outer contour = part; ``minAreaRect`` gives position and angle.
3. Coarse affine transform: long side of the bounding rectangle horizontal,
   part centroid in the centre of a fixed canvas.
4. Rotation search: the part is compared with the reference at all angles
   (0–360°, optionally also mirrored). This makes the alignment independent
   of the part shape – rectangles, squares, triangles, discs, … – and of the
   rotation in which the part arrives on the conveyor.
5. Optional ECC fine alignment to the mean reference image (sub-pixel).
6. Photometric normalisation: background → 0, part → 1.
   This cancels out small brightness fluctuations of the lighting.

All transforms are stored as a 3×3 matrix ``to_orig`` so that defect locations
can later be projected back exactly into the original image.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from .config import LocalizationConfig


@dataclass
class PartDetection:
    contour: np.ndarray
    rect: tuple                 # cv2.minAreaRect
    filled_mask: np.ndarray     # outer contour filled (including holes)
    part_mask: np.ndarray       # actual material (without holes)
    polarity: str               # "bright" | "dark" (part relative to background)
    complete: bool              # part lies completely inside the image
    area_ratio: float
    centroid: tuple[float, float]


@dataclass
class AlignedPart:
    frame: np.ndarray           # working image (BGR), basis for the visualisation
    bgr: np.ndarray             # aligned part (BGR)
    gray: np.ndarray            # aligned, uint8
    norm: np.ndarray            # aligned, float32, background≈0, part≈1
    filled_mask: np.ndarray     # aligned, uint8 0/255
    part_mask: np.ndarray       # aligned, uint8 0/255
    to_orig: np.ndarray         # 3×3: canvas coordinates → working image coordinates
    rect: tuple
    polarity: str
    part_level: float
    bg_level: float

    @property
    def canvas_size(self) -> tuple[int, int]:
        h, w = self.gray.shape[:2]
        return w, h


def to_gray(img: np.ndarray) -> np.ndarray:
    return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def resize_to_width(img: np.ndarray, width: int) -> np.ndarray:
    h, w = img.shape[:2]
    if w == width:
        return img
    scale = width / w
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    return cv2.resize(img, (width, int(round(h * scale))), interpolation=interp)


def detect_part(gray: np.ndarray, cfg: LocalizationConfig) -> PartDetection | None:
    """Finds the largest part in the image (or ``None``)."""
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    t, _ = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    polarity = cfg.polarity
    if polarity == "auto":
        h, w = gray.shape
        b = max(2, int(0.04 * min(h, w)))
        border = np.concatenate([blur[:b].ravel(), blur[-b:].ravel(), blur[:, :b].ravel(), blur[:, -b:].ravel()])
        polarity = "dark" if np.median(border) > t else "bright"

    if polarity == "bright":
        binary = (blur > t).astype(np.uint8) * 255
    else:
        binary = (blur <= t).astype(np.uint8) * 255

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(contour)
    area_ratio = area / float(gray.shape[0] * gray.shape[1])
    if area_ratio < cfg.min_part_area_ratio:
        return None

    x, y, bw, bh = cv2.boundingRect(contour)
    m = cfg.border_margin_px
    complete = x > m and y > m and x + bw < gray.shape[1] - m and y + bh < gray.shape[0] - m

    filled = np.zeros_like(gray)
    cv2.drawContours(filled, [contour], -1, 255, thickness=cv2.FILLED)
    part_mask = cv2.bitwise_and(binary, filled)

    mom = cv2.moments(contour)
    centroid = (mom["m10"] / mom["m00"], mom["m01"] / mom["m00"]) if mom["m00"] else (x + bw / 2, y + bh / 2)

    return PartDetection(
        contour=contour,
        rect=cv2.minAreaRect(contour),
        filled_mask=filled,
        part_mask=part_mask,
        polarity=polarity,
        complete=complete,
        area_ratio=area_ratio,
        centroid=centroid,
    )


def _coarse_angle(rect: tuple) -> float:
    box = cv2.boxPoints(rect)
    e1, e2 = box[1] - box[0], box[2] - box[1]
    long_edge = e1 if np.linalg.norm(e1) >= np.linalg.norm(e2) else e2
    angle = math.degrees(math.atan2(long_edge[1], long_edge[0]))
    # normalise to (-90, 90] → minimal rotation; the reference settles the 180° question later
    while angle > 90:
        angle -= 180
    while angle <= -90:
        angle += 180
    return angle


def default_canvas(det: "PartDetection", margin_ratio: float) -> tuple[int, int]:
    """Canvas size so that the part fits around its centroid (in coarse orientation)."""
    cx, cy = det.centroid
    a = math.radians(_coarse_angle(det.rect))
    pts = det.contour.reshape(-1, 2).astype(np.float64) - (cx, cy)
    # same rotation as cv2.getRotationMatrix2D(angle): x' = cos·x + sin·y, y' = −sin·x + cos·y
    rx = np.cos(a) * pts[:, 0] + np.sin(a) * pts[:, 1]
    ry = -np.sin(a) * pts[:, 0] + np.cos(a) * pts[:, 1]
    ex, ey = np.abs(rx).max(), np.abs(ry).max()
    pad = margin_ratio * 2 * max(ex, ey)
    cw = int(math.ceil((2 * ex + 2 * pad) / 2) * 2)
    ch = int(math.ceil((2 * ey + 2 * pad) / 2) * 2)
    return cw, ch


def _coarse_matrix(det: "PartDetection", canvas: tuple[int, int]) -> np.ndarray:
    """3×3 matrix working image → canvas (long side horizontal, centroid in the centre)."""
    cx, cy = det.centroid
    m = cv2.getRotationMatrix2D((cx, cy), _coarse_angle(det.rect), 1.0)
    m[0, 2] += canvas[0] / 2.0 - cx
    m[1, 2] += canvas[1] / 2.0 - cy
    return np.vstack([m, [0, 0, 1]])


def _canvas_rotation(canvas: tuple[int, int], angle: float, mirror: bool = False) -> np.ndarray:
    """3×3 matrix: rotate (and optionally mirror) the canvas about its centre."""
    w, h = canvas
    r = np.vstack([cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0), [0, 0, 1]])
    if mirror:
        r = r @ np.array([[-1, 0, w - 1], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    return r


def _warp(img: np.ndarray, to_canvas: np.ndarray, canvas: tuple[int, int], nearest=False, border_value=0):
    interp = cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR
    return cv2.warpAffine(
        img, to_canvas[:2], canvas, flags=interp,
        borderMode=cv2.BORDER_CONSTANT, borderValue=border_value,
    )


def _normalize(gray: np.ndarray, bg: float, part: float) -> np.ndarray:
    denom = part - bg
    if abs(denom) < 1e-3:
        denom = 1e-3 if denom >= 0 else -1e-3
    return np.clip((gray.astype(np.float32) - bg) / denom, -0.5, 1.5).astype(np.float32)


def _levels(gray: np.ndarray, det: PartDetection) -> tuple[float, float]:
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    inner = cv2.erode(det.part_mask, k)
    outer = cv2.dilate(det.filled_mask, k, iterations=2)
    part_px = gray[inner > 0]
    bg_px = gray[outer == 0]
    part_level = float(np.median(part_px)) if part_px.size else float(np.median(gray[det.part_mask > 0]))
    bg_level = float(np.median(bg_px)) if bg_px.size else float(255 - part_level)
    return part_level, bg_level


def _render(frame, gray, det, to_canvas, canvas, part_level, bg_level) -> AlignedPart:
    gray_c = _warp(gray, to_canvas, canvas, border_value=int(bg_level))
    bgr_c = _warp(frame, to_canvas, canvas, border_value=(int(bg_level),) * 3)
    filled_c = _warp(det.filled_mask, to_canvas, canvas, nearest=True)
    part_c = _warp(det.part_mask, to_canvas, canvas, nearest=True)
    return AlignedPart(
        frame=frame, bgr=bgr_c, gray=gray_c, norm=_normalize(gray_c, bg_level, part_level),
        filled_mask=filled_c, part_mask=part_c, to_orig=np.linalg.inv(to_canvas),
        rect=det.rect, polarity=det.polarity, part_level=part_level, bg_level=bg_level,
    )


def _search_patch(norm: np.ndarray, size: int = 96) -> np.ndarray:
    """Downscaled image on a square canvas; the canvas centre (= part centroid) maps
    exactly onto the patch centre, so rotations about the patch centre are exact."""
    h, w = norm.shape
    s = size / max(h, w)
    side = int(math.ceil(math.hypot(w, h) * s)) + 2
    src = cv2.GaussianBlur(np.clip(norm, 0, 1).astype(np.float32), (0, 0), max(0.5, 0.5 / s))   # anti-aliasing
    m = np.array([[s, 0, side / 2.0 - s * w / 2.0], [0, s, side / 2.0 - s * h / 2.0]], dtype=np.float64)
    return cv2.warpAffine(src, m, (side, side), flags=cv2.INTER_LINEAR)


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    a = a.ravel() - a.mean()
    b = b.ravel() - b.mean()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def _inner_region(reference_norm: np.ndarray) -> np.ndarray:
    """Interior of the reference part (outline filled, slightly eroded) – where the hole pattern is."""
    binary = (reference_norm > 0.5).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(binary)
    if contours:
        cv2.drawContours(filled, [max(contours, key=cv2.contourArea)], -1, 255, cv2.FILLED)
    return cv2.erode(filled, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))) > 0


def _rotate_canvas_image(img: np.ndarray, angle: float, mirror: bool) -> np.ndarray:
    h, w = img.shape[:2]
    return cv2.warpAffine(img, _canvas_rotation((w, h), angle, mirror)[:2], (w, h), flags=cv2.INTER_LINEAR)


def find_rotation(norm: np.ndarray, reference_norm: np.ndarray, allow_mirror: bool = False,
                  step: float = 4.0, mode: str = "full") -> tuple[float, bool, float]:
    """Angle (and mirroring) at which the part best matches the reference.

    1. Brute-force search over 0–360° on a small image (a few ms).
    2. All angles that score almost as well as the best one are candidates –
       e.g. 0°/90°/180°/270° for a square, 0°/120°/240° for a triangle.
    3. The candidates are compared in full resolution *inside* the part only,
       i.e. by their hole pattern, because the outline alone cannot tell them apart.
    Returns (angle, mirrored, similarity).
    """
    ref = _search_patch(reference_norm, 72)
    img = _search_patch(norm, 72)
    if img.shape != ref.shape:
        img = cv2.resize(img, ref.shape[::-1], interpolation=cv2.INTER_AREA)
    side = ref.shape[0]
    c = (side / 2.0, side / 2.0)
    ref_c = (ref - ref.mean()).ravel()
    ref_n = float(np.linalg.norm(ref_c)) + 1e-9
    imgs = {False: img, True: cv2.flip(img, 1)}

    def coarse_score(angle, mirror):
        rot = cv2.warpAffine(imgs[mirror], cv2.getRotationMatrix2D(c, angle, 1.0), (side, side), flags=cv2.INTER_LINEAR)
        v = rot.ravel()
        v = v - v.mean()
        return float(v @ ref_c) / (float(np.linalg.norm(v)) * ref_n + 1e-9)

    angles = np.arange(0, 360, step) if mode == "full" else np.array([0.0, 180.0])
    candidates = []
    for mirror in ((False, True) if allow_mirror else (False,)):
        scores = np.array([coarse_score(a, mirror) for a in angles])
        for i, sc in enumerate(scores):
            if mode != "full" or (sc >= scores[i - 1] and sc >= scores[(i + 1) % len(scores)]):   # local max
                candidates.append((sc, float(angles[i]), mirror))
    best_coarse = max(sc for sc, _, _ in candidates)
    candidates = sorted([cd for cd in candidates if cd[0] >= best_coarse - 0.15], reverse=True)[:8]

    region = _inner_region(reference_norm)
    ref_full = np.clip(reference_norm, 0, 1)
    best = None
    for _, a0, mirror in candidates:
        # 1° refinement on the small image
        a_ref = max(np.arange(a0 - step + 1, a0 + step, 1.0), key=lambda a: coarse_score(a, mirror))
        cand = _rotate_canvas_image(np.clip(norm, 0, 1), float(a_ref), mirror)
        sc = _ncc(cand[region], ref_full[region]) if region.any() else _ncc(cand, ref_full)
        if best is None or sc > best[2]:
            best = (float(a_ref) % 360, mirror, sc)
    return best


def align_part(
    frame: np.ndarray,
    det: PartDetection,
    cfg: LocalizationConfig,
    canvas: tuple[int, int] | None = None,
    reference_norm: np.ndarray | None = None,
) -> AlignedPart:
    """Aligns the detected part (see module docstring)."""
    gray = to_gray(frame)
    if canvas is None:
        canvas = default_canvas(det, cfg.canvas_margin_ratio)
    part_level, bg_level = _levels(gray, det)

    to_canvas = _coarse_matrix(det, canvas)
    aligned = _render(frame, gray, det, to_canvas, canvas, part_level, bg_level)

    if reference_norm is None:
        return aligned

    # In which rotation (0–360°, optionally mirrored) does the part match the reference?
    angle, mirror = 0.0, False
    if cfg.rotation_search in ("full", "flip") or cfg.allow_mirror:
        mode = cfg.rotation_search if cfg.rotation_search in ("full", "flip") else "flip"
        angle, mirror, _ = find_rotation(aligned.norm, reference_norm, cfg.allow_mirror, mode=mode)
        if cfg.rotation_search == "off" and not mirror:
            angle = 0.0
    if angle % 360 != 0 or mirror:
        to_canvas = _canvas_rotation(canvas, angle, mirror) @ to_canvas
        aligned = _render(frame, gray, det, to_canvas, canvas, part_level, bg_level)

    if cfg.use_ecc_refine:
        refined = _ecc_refine(frame, gray, det, aligned, to_canvas, canvas, reference_norm, part_level, bg_level,
                              _use_outline_datum(cfg, reference_norm))
        if refined is not None:
            aligned = refined
    return aligned


def _use_outline_datum(cfg: LocalizationConfig, reference_norm: np.ndarray) -> bool:
    """Align on the outer contour only (datum edges) – except for round outlines,
    which carry no rotation information; then the holes must be used as well."""
    if cfg.ecc_datum == "outline":
        return True
    if cfg.ecc_datum == "all":
        return False
    binary = (reference_norm > 0.5).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return False
    c = max(contours, key=cv2.contourArea)
    circularity = 4 * math.pi * cv2.contourArea(c) / max(1.0, cv2.arcLength(c, True) ** 2)
    return circularity < 0.9


def _ecc_refine(frame, gray, det, aligned, to_canvas, canvas, ref, part_level, bg_level, outline_datum=False):
    # Registration on the smoothed shape only. Surface texture such as brushed metal
    # rotates with the part and would otherwise bias the result.
    # outline_datum=True: only the filled outer contour is used (like datum edges on a
    # drawing), so a displaced hole cannot pull the alignment and hide its own offset.
    # The grey values at the edge are kept (sub-pixel information); only the interior,
    # at least 2 px away from any edge, is flattened to 1 to remove texture.
    def shape(img):
        img = np.clip(img, 0, 1).astype(np.float32)
        binary = (img > 0.5).astype(np.uint8)
        if outline_datum:
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if contours:
                binary = np.zeros_like(binary)
                cv2.drawContours(binary, [max(contours, key=cv2.contourArea)], -1, 1, cv2.FILLED)
        interior = cv2.erode(binary, np.ones((5, 5), np.uint8)) > 0
        out = np.where(interior, 1.0, img if not outline_datum else np.where(binary > 0, np.maximum(img, 0.5 * binary), img))
        return cv2.GaussianBlur(out.astype(np.float32), (0, 0), 1.0)

    # ECC on the half-resolution image (4× fewer pixels). The smoothed edge image keeps
    # sub-pixel information – a second pass at full resolution was measured to bring no
    # accuracy gain (0.155 vs 0.156 px) but doubled the time.
    tmpl, img = shape(ref), shape(aligned.norm)
    half = lambda x: cv2.resize(x, (x.shape[1] // 2, x.shape[0] // 2), interpolation=cv2.INTER_AREA)
    warp = np.eye(2, 3, dtype=np.float32)
    try:
        _, warp = cv2.findTransformECC(half(tmpl), half(img), warp, cv2.MOTION_EUCLIDEAN,
                                       (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 80, 1e-5), None, 3)
        warp[:, 2] *= 2.0
    except cv2.error:
        return None
    # Plausibility check: only accept small corrections
    shift = float(np.hypot(warp[0, 2], warp[1, 2]))
    angle = abs(math.degrees(math.atan2(warp[1, 0], warp[0, 0])))
    if shift > 0.1 * canvas[0] or angle > 5:
        return None
    w3 = np.vstack([warp.astype(np.float64), [0, 0, 1]])
    # ECC: aligned(W·x) ≈ ref(x)  →  canvas→original = inv(to_canvas) · W
    to_orig = np.linalg.inv(to_canvas) @ w3
    return _render(frame, gray, det, np.linalg.inv(to_orig), canvas, part_level, bg_level)


def localize_and_align(
    frame: np.ndarray,
    cfg: LocalizationConfig,
    canvas: tuple[int, int] | None = None,
    reference_norm: np.ndarray | None = None,
    require_complete: bool = True,
) -> AlignedPart | None:
    """Convenience function: working resolution → detection → alignment."""
    frame = resize_to_width(frame, cfg.work_width)
    det = detect_part(to_gray(frame), cfg)
    if det is None or (require_complete and not det.complete):
        return None
    return align_part(frame, det, cfg, canvas, reference_norm)
