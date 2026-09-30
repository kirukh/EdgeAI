"""Image representations for feature extraction.

Each representation returns a float32 image roughly in [0, 1] on the aligned
canvas. ``main.py evaluate`` compares which representation works best for
which inspection method.
"""
from __future__ import annotations

from typing import Callable

import cv2
import numpy as np

from .alignment import AlignedPart


def _gray(a: AlignedPart) -> np.ndarray:
    return a.gray.astype(np.float32) / 255.0


def _norm(a: AlignedPart) -> np.ndarray:
    # Photometrically normalised: background 0, part 1 → robust against brightness changes.
    # Not clipped at 1, otherwise bright scratches/specular spots would disappear.
    return np.clip(a.norm, 0.0, 1.5)


def _clahe(a: AlignedPart) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(a.gray).astype(np.float32) / 255.0


def _edges(a: AlignedPart) -> np.ndarray:
    # Sobel gradient magnitude on the normalised image (smooth, well suited for differences)
    n = cv2.GaussianBlur(np.clip(a.norm, 0, 1.5), (5, 5), 0)
    gx = cv2.Sobel(n, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(n, cv2.CV_32F, 0, 1, ksize=3)
    return np.clip(cv2.magnitude(gx, gy) / 2.0, 0, 1)


def _canny(a: AlignedPart) -> np.ndarray:
    e = cv2.Canny(cv2.GaussianBlur(a.gray, (5, 5), 0), 40, 120)
    e = cv2.dilate(e, np.ones((3, 3), np.uint8))
    return cv2.GaussianBlur(e.astype(np.float32) / 255.0, (5, 5), 0)


def _binary(a: AlignedPart) -> np.ndarray:
    # Threshold halfway between background and part brightness
    return (a.norm > 0.5).astype(np.float32)


def _adaptive(a: AlignedPart) -> np.ndarray:
    b = cv2.adaptiveThreshold(a.gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 5)
    return b.astype(np.float32) / 255.0


def _lab_l(a: AlignedPart) -> np.ndarray:
    return cv2.cvtColor(a.bgr, cv2.COLOR_BGR2LAB)[:, :, 0].astype(np.float32) / 255.0


def _lab_ab(a: AlignedPart) -> np.ndarray:
    # Colour deviation (rust, discoloration): distance of the a/b channels from neutral grey
    lab = cv2.cvtColor(a.bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    chroma = np.hypot(lab[:, :, 1] - 128, lab[:, :, 2] - 128)
    return np.clip(chroma / 40.0, 0, 1)


def _hsv_s(a: AlignedPart) -> np.ndarray:
    return cv2.cvtColor(a.bgr, cv2.COLOR_BGR2HSV)[:, :, 1].astype(np.float32) / 255.0


def _hsv_v(a: AlignedPart) -> np.ndarray:
    return cv2.cvtColor(a.bgr, cv2.COLOR_BGR2HSV)[:, :, 2].astype(np.float32) / 255.0


REPRESENTATIONS: dict[str, tuple[str, Callable[[AlignedPart], np.ndarray]]] = {
    "gray": ("Grayscale", _gray),
    "norm": ("Grayscale, photometrically normalised", _norm),
    "clahe": ("Contrast enhancement (CLAHE)", _clahe),
    "edges": ("Edges (Sobel magnitude)", _edges),
    "canny": ("Edges (Canny)", _canny),
    "binary": ("Threshold (global)", _binary),
    "adaptive": ("Threshold (adaptive)", _adaptive),
    "lab_l": ("LAB – lightness L", _lab_l),
    "lab_ab": ("LAB – chroma a/b", _lab_ab),
    "hsv_s": ("HSV – saturation", _hsv_s),
    "hsv_v": ("HSV – value", _hsv_v),
}


def compute(aligned: AlignedPart, name: str) -> np.ndarray:
    if name not in REPRESENTATIONS:
        raise ValueError(f"Unknown representation '{name}'. Available: {', '.join(REPRESENTATIONS)}")
    return REPRESENTATIONS[name][1](aligned).astype(np.float32)


def labels() -> dict[str, str]:
    return {k: v[0] for k, v in REPRESENTATIONS.items()}
