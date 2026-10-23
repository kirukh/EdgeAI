"""Continuous improvement: the operator adapts the model to the real conditions in operation.

The model is taught in with a few good parts under one condition. In operation the operator
sometimes sees a NOK result for a part that is still within the company's tolerance – a slightly
offset hole, a mark on the surface. With "Result wrong? → within our tolerance" this part is
*accepted*; the next model version learns its deviation as acceptable::

    NOK result ──► operator accepts ──► waiting (pending/) ──► after N parts or by button:
                                                               candidate model, trained in the
    OK result  ──► operator: "defective" ─► known_bad/          background (inspection continues)
    NOK result ──► operator confirms ────► known_bad/                      │
                                                                   release check:
                                              every known defective part is still detected?
                                                    yes → new version (previous kept for undo)
                                                    no  → model unchanged, reasons reported

How a deviation is learned (methods.py): only *locally* – the tolerance of that hole, the
tolerance map of that image region. The global thresholds stay, so accepting a shifted hole does
not make the model blind for scratches. Parts the operator accepted are never removed by the
reference screening; the model only learns from confirmed decisions, never by itself.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .model import QCModel

PENDING, KNOWN_BAD = "pending", "known_bad"


class LearningStore:
    """Accepted (``pending``) and confirmed defective (``known_bad``) parts of one part type,
    stored next to the model: ``<model>/<kind>/<stem>.png`` + ``<stem>.json``."""

    def __init__(self, model_dir: Path):
        self.dir = Path(model_dir)

    def add(self, kind: str, frame: np.ndarray, info: dict[str, Any], limit: int) -> str:
        """Stores a part; the oldest ones are dropped above ``limit``. Returns its name."""
        folder = self.dir / kind
        folder.mkdir(parents=True, exist_ok=True)
        stem = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        cv2.imwrite(str(folder / f"{stem}.png"), frame)
        tmp = folder / f"{stem}.json.tmp"
        tmp.write_text(json.dumps(info, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, folder / f"{stem}.json")
        stems = self.stems(kind)
        self.remove(kind, stems[: max(0, len(stems) - limit)])
        return stem

    def stems(self, kind: str) -> list[str]:
        folder = self.dir / kind
        return sorted(p.stem for p in folder.glob("*.json")) if folder.exists() else []

    def count(self, kind: str) -> int:
        return len(self.stems(kind))

    def items(self, kind: str) -> list[tuple[str, np.ndarray, dict[str, Any]]]:
        out = []
        for stem in self.stems(kind):
            img = cv2.imread(str(self.dir / kind / f"{stem}.png"))
            if img is not None:
                out.append((stem, img, json.loads((self.dir / kind / f"{stem}.json").read_text(encoding="utf-8"))))
        return out

    def image_path(self, kind: str, stem: str) -> Path | None:
        """PNG of a stored part, or None (also for names that try to leave the folder)."""
        folder = (self.dir / kind).resolve()
        p = (folder / f"{stem}.png").resolve()
        return p if p.parent == folder and p.is_file() else None

    def remove(self, kind: str, stems: list[str]) -> None:
        for stem in stems:
            for ext in (".png", ".json"):
                (self.dir / kind / f"{stem}{ext}").unlink(missing_ok=True)


@dataclass
class UpdateReport:
    """Result of one model update – shown in the setup area and kept in the model history."""

    time: str
    model: str
    version_before: int
    version_after: int | None = None  # None = rejected
    accepted_parts: int = 0  # parts the operator accepted that this update learns
    references: int = 0  # references of the new version
    known_bad_checked: int = 0  # known defective parts the current model detects …
    known_bad_still_detected: int = 0  # … and how many of them the candidate still detects
    still_nok: list[str] = field(default_factory=list[Any])  # accepted parts the candidate still rates NOK
    reasons: list[str] = field(default_factory=list[Any])  # why the update was rejected
    used: list[str] = field(default_factory=list[Any])  # names of the learned pending parts

    @property
    def released(self) -> bool:
        return self.version_after is not None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("used")
        d["released"] = self.released
        return d


def update_model(model: QCModel, store: LearningStore) -> tuple[QCModel | None, UpdateReport]:
    """Trains a candidate from the current references + all accepted parts and checks it.

    → (new version or None, report). Nothing is saved here; see :func:`release`.
    """
    report = UpdateReport(datetime.now().isoformat(timespec="seconds"), model.name, model.version)
    pending = store.items(PENDING)
    if not pending:
        report.reasons.append("No accepted parts waiting.")
        return None, report
    refs = model.reference_frames()
    frames = refs + [f for _, f, _ in pending]
    accepted = [*model.accepted, *range(len(refs), len(frames))]
    cand = QCModel.train(model.name, frames, model.loc_cfg, model.method_cfg, model.mm_per_px, accepted=accepted)
    cand.created, cand.sensitivity, cand.camera_profile = model.created, dict(model.sensitivity), model.camera_profile
    report.accepted_parts, report.references, report.used = len(pending), cand.n_samples, [s for s, _, _ in pending]

    # Release check: a defective part that the current model detects must still be detected.
    # This catches an acceptance by mistake ("this is OK") that contradicts an earlier "defective".
    for _, frame, _info in store.items(KNOWN_BAD):
        if model.inspect(frame, False).status == "NOK":
            report.known_bad_checked += 1
            if cand.inspect(frame, False).status == "NOK":
                report.known_bad_still_detected += 1
    lost = report.known_bad_checked - report.known_bad_still_detected
    if lost:
        report.reasons.append(
            f"The new version would no longer detect {lost} of {report.known_bad_checked} part(s) that were "
            "confirmed as defective – an accepted part probably contradicts an earlier decision. Check the "
            "accepted parts (setup area → Models), discard the wrong ones and update again."
        )
    for _, frame, _info in pending:  # information: what the update could not learn
        res = cand.inspect(frame, False)
        if res.status == "NOK":
            report.still_nok.append(", ".join(sorted({d.label for d in res.defects})) or "NOK")
    if report.reasons:
        return None, report
    cand.version = model.version + 1
    report.version_after = cand.version
    cand.history = [*model.history, report.to_dict()]
    return cand, report


def release(model: QCModel, store: LearningStore, report: UpdateReport) -> Path:
    """Saves a released candidate as the new version (the old one is kept for undo) and removes
    the learned parts from the waiting list."""
    path = model.save_to(store.dir, keep_previous=True, keep_side_data=True)
    store.remove(PENDING, report.used)
    return path
