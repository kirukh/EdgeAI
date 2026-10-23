"""Inspection results: recent history for the UI, counters, supervisor feedback and the logs.

Everything that leaves the program goes to ``storage.results_dir/<date>/``::

    inspection_log.csv     one row per part (also parts that could not be inspected)
    feedback_log.csv       supervisor corrections ("Result wrong?")
    *_marked.jpg / *_original.png   images of NOK parts (OK parts optionally)

"Finish batch" packs this folder into a ZIP (see ``archive.py``).
"""

from __future__ import annotations

import csv
import itertools
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .alignment import resize_to_width
from .config import StorageConfig
from .model import InspectionResult
from .visualize import encode_jpg

# new columns are appended at the end, so logs written by older versions stay readable
LOG_HEADER = [
    "time", "model", "result", "defects", "duration_ms", "image", "ground_truth", "image_id", "shots", "model_version",
]  # fmt: skip
FEEDBACK_HEADER = ["time", "image_id", "inspected_at", "model", "result", "supervisor", "outcome", "model_version"]
OUTCOME_TEXT = {
    "false_alarm": "false alarm",
    "missed": "missed defect",
    "confirmed_ok": "confirmed OK",
    "confirmed_nok": "confirmed NOK",
}
NOT_INSPECTED = "NOT_INSPECTED"


def _counter_key(outcome: str) -> str:
    return "confirmed" if outcome.startswith("confirmed") else outcome


def _append_csv(path: Path, header: list[str], row: list[Any]) -> None:
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(header)
        w.writerow(row)


class ResultStore:
    """Inspection history, counters, supervisor feedback and the CSV logs of the day."""

    def __init__(self, cfg: StorageConfig, work_width: int = 640):
        self.cfg = cfg
        self.work_width = work_width
        self.history: deque[dict[str, Any]] = deque(maxlen=cfg.history_size)
        self.images: dict[str, tuple[bytes, bytes]] = {}  # image id → (overlay, detail) as JPEG
        self.frames: dict[str, np.ndarray] = {}  # image id → inspected image (for "Result wrong?")
        self.counts = {"OK": 0, "NOK": 0, "NO_PART": 0}
        self.feedback_counts = {"false_alarm": 0, "missed": 0, "confirmed": 0}
        self.skipped = 0  # parts that could not be inspected (Pi overloaded)
        self.last_result: dict[str, Any] | None = None
        self._ids = itertools.count(1)
        Path(cfg.results_dir).mkdir(parents=True, exist_ok=True)

    def day_dir(self) -> Path:
        """Results folder of today (created if needed)."""
        day = Path(self.cfg.results_dir) / datetime.now().strftime("%Y-%m-%d")
        day.mkdir(parents=True, exist_ok=True)
        return day

    def add(self, res: InspectionResult, frame: np.ndarray | None) -> dict[str, Any]:
        """Counts, keeps and logs one inspected part. Returns the entry shown in the UI."""
        res.image_id = f"{next(self._ids):06d}"
        entry = res.to_dict()
        entry["feedback"] = None
        self.counts[res.status] = self.counts.get(res.status, 0) + 1
        self.last_result = entry
        self.history.appendleft(entry)
        if res.overlay is not None and res.detail is not None:
            self.images[res.image_id] = (encode_jpg(res.overlay), encode_jpg(res.detail))
        if frame is not None and res.status != "NO_PART":
            self.frames[res.image_id] = resize_to_width(frame, self.work_width).copy()
        keep = {h["image_id"] for h in self.history}
        for store in (self.images, self.frames):
            for k in [k for k in store if k not in keep]:
                del store[k]
        self._log(res, frame)
        return entry

    def add_skipped(self, model_name: str) -> None:
        """A part that could not be inspected – it is logged, never silently lost."""
        self.skipped += 1
        ts = datetime.now().isoformat(timespec="milliseconds")
        _append_csv(
            self.day_dir() / "inspection_log.csv",
            LOG_HEADER,
            [ts, model_name, NOT_INSPECTED, "Pi could not keep up", "", "", "", "", "", ""],
        )

    def _log(self, res: InspectionResult, frame: np.ndarray | None) -> None:
        day = self.day_dir()
        stem = f"{datetime.now().strftime('%H%M%S_%f')[:-3]}_{res.status}"
        save = (res.status == "NOK" and self.cfg.save_nok_images) or (res.status == "OK" and self.cfg.save_ok_images)
        overlay = res.overlay if save else None
        if overlay is not None:
            cv2.imwrite(str(day / f"{stem}_marked.jpg"), overlay)
            if frame is not None:
                cv2.imwrite(str(day / f"{stem}_original.png"), frame)
        _append_csv(
            day / "inspection_log.csv",
            LOG_HEADER,
            [
                res.timestamp,
                res.model,
                res.status,
                " | ".join(f"{x.label}: {x.detail}" for x in res.defects),
                f"{res.time_ms:.0f}",
                stem if overlay is not None else "",
                res.ground_truth or "",
                res.image_id,
                "/".join(res.shots),
                res.model_version,
            ],
        )

    def entry(self, rid: str) -> dict[str, Any]:
        """History entry of a result id (KeyError if it is no longer in the history)."""
        entry = next((h for h in self.history if h["image_id"] == rid), None)
        if entry is None:
            raise ValueError("Result not found (only the recent history can be corrected).")
        return entry

    def feedback(self, rid: str, verdict: str) -> str:
        """Supervisor confirms/corrects a recent result (verdict "ok" or "nok") → outcome."""
        if verdict not in ("ok", "nok"):
            raise ValueError("verdict must be 'ok' or 'nok'.")
        entry = self.entry(rid)
        if entry["status"] == "NO_PART":
            raise ValueError("Nothing to correct for 'no part'.")
        if entry.get("feedback"):  # the supervisor changed their mind – the last word counts
            self.feedback_counts[_counter_key(entry["feedback"])] -= 1
        if entry["status"] == "NOK":
            outcome = "false_alarm" if verdict == "ok" else "confirmed_nok"
        else:
            outcome = "missed" if verdict == "nok" else "confirmed_ok"
        entry["feedback"] = outcome
        self.feedback_counts[_counter_key(outcome)] += 1
        _append_csv(
            self.day_dir() / "feedback_log.csv",
            FEEDBACK_HEADER,
            [
                datetime.now().isoformat(timespec="seconds"),
                rid,
                entry["timestamp"],
                entry["model"],
                entry["status"],
                verdict.upper(),
                outcome,
                entry.get("model_version", ""),
            ],
        )
        return outcome

    def reset(self) -> None:
        """New batch: counters and history are cleared."""
        self.counts = {"OK": 0, "NOK": 0, "NO_PART": 0}
        self.feedback_counts = {"false_alarm": 0, "missed": 0, "confirmed": 0}
        self.skipped = 0
        self.history.clear()
        self.images.clear()
        self.frames.clear()
        self.last_result = None
