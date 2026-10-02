"""Visualisation of inspection results (highlight the defect location in colour)."""
from __future__ import annotations

import cv2
import numpy as np

from .alignment import AlignedPart

GREEN = (60, 190, 60)
RED = (40, 40, 230)


def _to_orig(pts: np.ndarray, to_orig: np.ndarray) -> np.ndarray:
    p = np.hstack([pts, np.ones((len(pts), 1))]) @ to_orig.T
    return p[:, :2]


def _box_pts(bbox) -> np.ndarray:
    x, y, w, h = bbox
    return np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], np.float64)


def _put_label(img, text, org, color, scale=0.5):
    x, y = int(org[0]), int(org[1])
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    x = max(2, min(x, img.shape[1] - tw - 4))
    y = max(th + 4, min(y, img.shape[0] - 4))
    cv2.rectangle(img, (x - 2, y - th - 4), (x + tw + 2, y + 3), (0, 0, 0), cv2.FILLED)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def _ascii(text: str) -> str:
    # OpenCV Hershey fonts only support ASCII
    return text.replace("Ø", "D").replace("–", "-").encode("ascii", "replace").decode()


def heat_overlay(img: np.ndarray, heat: np.ndarray, strength: float = 0.6) -> np.ndarray:
    heat = np.clip(heat, 0, 1)
    color = cv2.applyColorMap((heat * 255).astype(np.uint8), cv2.COLORMAP_JET)
    alpha = (np.clip((heat - 0.35) / 0.65, 0, 1) * strength)[:, :, None]
    return (img * (1 - alpha) + color * alpha).astype(np.uint8)


def annotate(aligned: AlignedPart, ok: bool, defects: list, heat: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    """Returns (full image with markings, detail view of the aligned part)."""
    frame = aligned.frame.copy()
    h, w = frame.shape[:2]

    if heat is not None and not ok:
        heat_back = cv2.warpAffine(heat, aligned.to_orig[:2], (w, h), flags=cv2.INTER_LINEAR)
        frame = heat_overlay(frame, heat_back, 0.5)

    # Part outline
    contours, _ = cv2.findContours(aligned.filled_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    color = GREEN if ok else RED
    for c in contours:
        pts = _to_orig(c.reshape(-1, 2).astype(np.float64), aligned.to_orig)
        cv2.polylines(frame, [pts.round().astype(np.int32)], True, color, 2, cv2.LINE_AA)

    detail = aligned.bgr.copy()
    if heat is not None and not ok:
        detail = heat_overlay(detail, heat, 0.55)

    for i, d in enumerate(defects, 1):
        pts = _to_orig(_box_pts(d.bbox), aligned.to_orig).round().astype(np.int32)
        cv2.polylines(frame, [pts], True, RED, 2, cv2.LINE_AA)
        _put_label(frame, f"{i}", (pts[:, 0].min(), pts[:, 1].min() - 4), (255, 255, 255))
        x, y, bw, bh = d.bbox
        cv2.rectangle(detail, (x, y), (x + bw, y + bh), RED, 2, cv2.LINE_AA)
        _put_label(detail, f"{i}: {_ascii(d.label)}", (x, y - 4), (255, 255, 255), 0.42)

    banner = "OK" if ok else "NOK"
    cv2.rectangle(frame, (0, 0), (110, 46), GREEN if ok else RED, cv2.FILLED)
    cv2.putText(frame, banner, (12, 36), cv2.FONT_HERSHEY_DUPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
    return frame, detail


def encode_jpg(img: np.ndarray, quality: int = 85) -> bytes:
    """JPEG bytes for the web UI."""
    return cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])[1].tobytes()
