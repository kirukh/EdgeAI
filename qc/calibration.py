"""Camera calibration with a printed checkerboard.

Two things are determined:

1. **Lens distortion** (optional, needs ≥ 5 images of the board in different
   positions/tilts): straight edges become straight, and a part at the image
   border is measured like one in the centre.
2. **Scale in mm/px**: from the FIRST image, in which the board must lie flat on
   the conveyor at the height of the parts (same plane as the part surface).

Procedure (UI card "Camera calibration" or ``main.py calibrate``):

    1. Print a checkerboard (e.g. 10 × 7 squares → 9 × 6 inner corners), measure the
       square size exactly with a calliper, glue it flat onto a rigid plate.
    2. Image 1: board flat on the belt under the camera.
    3. Optional images 2 … 15: board moved/tilted over the whole image area.
    4. "Compute calibration".

After a new calibration all models must be retrained (the images are undistorted
differently); the UI warns about models trained with another calibration.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

MIN_IMAGES_DISTORTION = 5


class CalibrationError(ValueError):
    pass


def find_corners(frame: np.ndarray, cols: int, rows: int) -> np.ndarray | None:
    gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    ok, corners = cv2.findChessboardCorners(gray, (cols, rows), flags)
    if not ok:
        return None
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 40, 1e-3)
    return cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), crit)


def draw_corners(frame: np.ndarray, corners: np.ndarray | None, cols: int, rows: int) -> np.ndarray:
    img = frame.copy()
    cv2.drawChessboardCorners(img, (cols, rows), corners, corners is not None)
    return img


@dataclass
class Calibration:
    id: str
    image_size: tuple[int, int]          # (w, h) of the camera frames
    camera_matrix: list | None           # 3×3 (None = no distortion correction)
    dist_coeffs: list | None
    mm_per_px_full: float                # scale in the undistorted full-resolution frame
    rms_px: float | None
    n_images: int
    board: dict
    note: str = ""

    def mm_per_px(self, work_width: int) -> float:
        """Scale at the working resolution used by the models."""
        return self.mm_per_px_full * self.image_size[0] / work_width

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "Calibration | None":
        p = Path(path)
        if not p.exists():
            return None
        d = json.loads(p.read_text(encoding="utf-8"))
        d["image_size"] = tuple(d["image_size"])
        return cls(**d)


class Undistorter:
    """Fast undistortion with precomputed maps (a few ms per frame)."""

    def __init__(self, calib: Calibration):
        self.calib = calib
        self.maps = None
        if calib.camera_matrix is not None:
            K = np.array(calib.camera_matrix, np.float64)
            D = np.array(calib.dist_coeffs, np.float64)
            self.maps = cv2.initUndistortRectifyMap(K, D, None, K, calib.image_size, cv2.CV_16SC2)

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        if self.maps is None or (frame.shape[1], frame.shape[0]) != self.calib.image_size:
            return frame
        return cv2.remap(frame, self.maps[0], self.maps[1], cv2.INTER_LINEAR)


def compute(corner_sets: list[np.ndarray], image_size: tuple[int, int], cols: int, rows: int,
            square_mm: float) -> Calibration:
    """``corner_sets[0]`` = board flat on the belt (defines the scale)."""
    if not corner_sets:
        raise CalibrationError("No checkerboard images captured.")
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_mm
    K = D = None
    rms = None
    note = ""
    if len(corner_sets) >= MIN_IMAGES_DISTORTION:
        flags = cv2.CALIB_FIX_K3                       # robust with few images
        rms, K, D, _, _, std_int, _, _ = cv2.calibrateCameraExtended(
            [objp] * len(corner_sets), corner_sets, image_size, None, None, flags=flags)
        std_int = std_int.ravel()
        # Plausibility: with too similar board poses (little tilt) focal length and distortion
        # cannot be separated – the estimate is then unreliable and would corrupt the scale.
        f_rel = std_int[0] / K[0, 0] if K[0, 0] else 1.0
        if f_rel > 0.03 or std_int[4] > 0.05 or rms > 1.5:
            note = (f"Lens distortion not reliable (focal length ±{f_rel * 100:.1f} %, k1 ±{std_int[4]:.3f}, "
                    f"RMS {rms:.2f} px) – tilt the board more (20–40°) and cover the whole image. "
                    "Only the scale was calibrated.")
            K = D = None
            rms = None
    # scale from the first image (after undistortion of the corner positions)
    pts = corner_sets[0].reshape(-1, 2).astype(np.float64)
    if K is not None:
        pts = cv2.undistortPoints(pts.reshape(-1, 1, 2), K, D, P=K).reshape(-1, 2)
    grid = pts.reshape(rows, cols, 2)
    d = np.concatenate([np.linalg.norm(np.diff(grid, axis=1), axis=2).ravel(),
                        np.linalg.norm(np.diff(grid, axis=0), axis=2).ravel()])
    mm_per_px = square_mm / float(np.median(d))
    return Calibration(
        id=datetime.now().strftime("%Y%m%d-%H%M%S"), image_size=tuple(image_size),
        camera_matrix=K.tolist() if K is not None else None, dist_coeffs=D.tolist() if D is not None else None,
        mm_per_px_full=mm_per_px, rms_px=float(rms) if rms is not None else None, n_images=len(corner_sets),
        board={"cols": cols, "rows": rows, "square_mm": square_mm}, note=note,
    )


def board_pdf(path: str, cols: int = 9, rows: int = 6, square_mm: float = 10.0, dpi: int = 600) -> dict:
    """Printable checkerboard at exact scale (A4). ``cols``/``rows`` = INNER corners,
    so the board has (cols+1) × (rows+1) squares. Print at 100 % / "actual size"."""
    from PIL import Image, ImageDraw

    a4 = (210.0, 297.0)
    nx, ny = cols + 1, rows + 1
    w_mm, h_mm = nx * square_mm, ny * square_mm
    landscape = w_mm > a4[0] - 20
    page = (a4[1], a4[0]) if landscape else a4
    if w_mm > page[0] - 20 or h_mm > page[1] - 40:
        raise ValueError(f"Board {w_mm:.0f} × {h_mm:.0f} mm does not fit on A4 – use smaller squares.")
    px = lambda mm: int(round(mm / 25.4 * dpi))          # noqa: E731
    img = Image.new("L", (px(page[0]), px(page[1])), 255)
    d = ImageDraw.Draw(img)
    x0, y0 = (page[0] - w_mm) / 2, 25.0
    for j in range(ny):
        for i in range(nx):
            if (i + j) % 2 == 0:
                d.rectangle([px(x0 + i * square_mm), px(y0 + j * square_mm),
                             px(x0 + (i + 1) * square_mm) - 1, px(y0 + (j + 1) * square_mm) - 1], fill=0)
    # 100 mm check line: measure it after printing
    ly = y0 + h_mm + 15
    d.rectangle([px(x0), px(ly), px(x0 + 100), px(ly) + max(2, px(0.4))], fill=0)
    for t in (0, 100):
        d.rectangle([px(x0 + t) - 2, px(ly - 3), px(x0 + t) + 2, px(ly + 3)], fill=0)
    try:
        from PIL import ImageFont
        font = ImageFont.load_default(size=px(3.2))
    except Exception:  # noqa: BLE001
        font = None
    d.text((px(x0), px(ly + 6)), f"Checkerboard {cols} x {rows} inner corners, square {square_mm:g} mm. "
           f"Print at 100 % (actual size). The line above must measure exactly 100 mm.", fill=0, font=font)
    img.save(path, "PDF", resolution=dpi)
    return {"cols": cols, "rows": rows, "square_mm": square_mm, "board_mm": (w_mm, h_mm), "landscape": landscape}
