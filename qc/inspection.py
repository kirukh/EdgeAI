"""Inspection of one part from several images (multi-shot majority vote).

Several images of the same part are taken while it passes the camera. Random false alarms
(dust, a reflection, sensor noise) affect only single images and are outvoted. To save time
on the Pi, a CLEARLY good first image (every method far inside its tolerance) is final.

With a stop-and-go belt (trigger mode ``still``) the shots show the resting part – they differ
only by sensor noise, lamp flicker and compression artefacts. A vote gains nothing there; instead
the shots are averaged into ONE image with less noise (``merge_still``), for teach-in and inspection.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .model import InspectionResult, QCModel


def merge_still(
    shots: list[np.ndarray], positions: Sequence[tuple[float, float]], tolerance: float
) -> list[np.ndarray]:
    """Shots of a resting part → one averaged image (noise ÷ √n). ``positions`` = part centre in
    every shot as a fraction of the image size; if the part moved more than ``tolerance`` between
    the shots they are returned unchanged (they are then voted on as usual)."""
    if len(shots) < 2 or len(positions) != len(shots):
        return shots
    p = np.asarray(positions, np.float64)
    if float(np.max(p.max(axis=0) - p.min(axis=0))) >= tolerance:
        return shots
    mean = np.mean(np.stack(shots).astype(np.float32), axis=0)
    return [np.clip(np.rint(mean), 0, 255).astype(np.uint8)]


def vote(results: list[InspectionResult], tie_is_nok: bool = True) -> InspectionResult:
    """Majority vote over several shots of the same part (NO_PART shots do not count)."""
    statuses = [r.status for r in results]
    valid = [r for r in results if r.status != "NO_PART"]
    if not valid:
        rep = results[0]
    else:
        n_nok = sum(r.status == "NOK" for r in valid)
        nok = n_nok * 2 > len(valid) or (n_nok * 2 == len(valid) and tie_is_nok)
        if nok:  # the most clearly defective shot is shown
            rep = max((r for r in valid if r.status == "NOK"), key=lambda r: r.worst)
        else:
            rep = next(r for r in valid if r.status == "OK")
    rep.shots = statuses
    rep.time_ms = sum(r.time_ms for r in results)
    return rep


def inspect_part(
    model: QCModel, shots: list[np.ndarray], confirm_margin: float = 0.5, tie_is_nok: bool = True
) -> tuple[InspectionResult, int]:
    """→ (result, index of the shot it shows). The other shots are only evaluated when the first
    one is not clearly good (every method score below ``confirm_margin``)."""
    first = model.inspect(shots[0])
    clear = first.ok and all(r.score < confirm_margin for r in first.method_results)
    results = [first] if clear or len(shots) == 1 else [first, *(model.inspect(s) for s in shots[1:])]
    res = vote(results, tie_is_nok)
    return res, next(i for i, r in enumerate(results) if r is res)
