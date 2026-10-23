"""Inspection of one part from several images (multi-shot majority vote).

Several images of the same part are taken while it passes the camera. Random false alarms
(dust, a reflection, sensor noise) affect only single images and are outvoted. To save time
on the Pi, a CLEARLY good first image (every method far inside its tolerance) is final.
"""

from __future__ import annotations

import numpy as np

from .model import InspectionResult, QCModel


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
