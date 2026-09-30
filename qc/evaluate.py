"""Systematic comparison of methods and image representations.

Answers the project question "Which combination of preprocessing, method and
lighting detects which defects most reliably?"

For every configuration the model is trained on good parts and measured on a
labelled test set:

* false alarm rate (good part → NOK, *false positive rate*)
* detection rate per defect type (defective part → NOK, *recall*)
* mean inspection time
"""
from __future__ import annotations

import csv
import time
from dataclasses import dataclass, field, replace
from pathlib import Path

import cv2
import numpy as np

from .camera import IMAGE_EXT
from .config import LocalizationConfig, MethodConfig
from .model import QCModel, TrainingError


@dataclass
class EvalRow:
    config: str
    fpr: float
    recall: float
    per_defect: dict[str, float] = field(default_factory=dict)
    time_ms: float = 0.0
    note: str = ""


def default_grid(base: MethodConfig) -> list[tuple[str, MethodConfig]]:
    grid = [("geometry", replace(base, methods=["geometry"]))]
    for r in ["norm", "gray", "clahe", "edges", "canny", "binary", "lab_ab", "hsv_s"]:
        grid.append((f"diff[{r}]", replace(base, methods=["diff"], diff_representation=r)))
    for r in ["norm", "edges", "canny", "binary"]:
        grid.append((f"pca[{r}]", replace(base, methods=["pca"], pca_representation=r)))
    grid.append(("geometry + diff[norm]", replace(base, methods=["geometry", "diff"], diff_representation="norm")))
    grid.append(("geometry + diff[norm] + pca[edges]",
                 replace(base, methods=["geometry", "diff", "pca"], diff_representation="norm", pca_representation="edges")))
    return grid


def load_labeled_folder(test_dir: str | Path) -> list[tuple[np.ndarray, str]]:
    """Expects ``ok/`` and ``nok/`` (optionally with one subfolder per defect type)."""
    items = []
    root = Path(test_dir)
    for p in sorted(root.rglob("*")):
        if p.suffix.lower() not in IMAGE_EXT:
            continue
        rel = p.relative_to(root).parts
        if rel[0].lower() == "ok":
            label = "ok"
        elif rel[0].lower() == "nok":
            label = rel[1] if len(rel) > 2 else "nok"
        else:
            continue
        items.append((cv2.imread(str(p)), label))
    return items


def load_folder(folder: str | Path) -> list[np.ndarray]:
    return [cv2.imread(str(p)) for p in sorted(Path(folder).rglob("*")) if p.suffix.lower() in IMAGE_EXT]


def evaluate(train: list[np.ndarray], test: list[tuple[np.ndarray, str]],
             grid: list[tuple[str, MethodConfig]], loc: LocalizationConfig | None = None,
             progress=print) -> list[EvalRow]:
    loc = loc or LocalizationConfig()
    labels = sorted({lbl for _, lbl in test if lbl != "ok"})
    rows = []
    for name, mcfg in grid:
        try:
            model = QCModel.train(name, train, loc, mcfg)
        except TrainingError as e:
            rows.append(EvalRow(name, float("nan"), float("nan"), note=str(e)))
            continue
        hits: dict[str, list[bool]] = {lbl: [] for lbl in labels}
        fp, n_ok, times = 0, 0, []
        for frame, lbl in test:
            t0 = time.perf_counter()
            res = model.inspect(frame, visualize=False)
            times.append((time.perf_counter() - t0) * 1000)
            nok = res.status != "OK"
            if lbl == "ok":
                n_ok += 1
                fp += nok
            else:
                hits[lbl].append(nok)
        per = {lbl: float(np.mean(v)) if v else float("nan") for lbl, v in hits.items()}
        all_hits = [h for v in hits.values() for h in v]
        row = EvalRow(name, fp / max(1, n_ok), float(np.mean(all_hits)) if all_hits else float("nan"),
                      per, float(np.mean(times)))
        rows.append(row)
        if progress:
            progress(f"  {name:38s} false alarms {row.fpr * 100:5.1f} %  detection {row.recall * 100:5.1f} %")
    return rows


def to_markdown(rows: list[EvalRow], title: str = "") -> str:
    labels = sorted({k for r in rows for k in r.per_defect})
    head = ["Configuration", "False alarms", "Detection (all)"] + labels + ["ms/part"]
    lines = [f"### {title}" if title else "", "", "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]

    def pct(v):
        return "–" if v != v else f"{v * 100:.0f} %"

    for r in rows:
        cells = [r.config, pct(r.fpr), pct(r.recall)] + [pct(r.per_defect.get(k, float("nan"))) for k in labels] + [f"{r.time_ms:.0f}"]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines).strip() + "\n"


def to_csv(rows: list[EvalRow], path: str | Path, extra: dict | None = None) -> None:
    labels = sorted({k for r in rows for k in r.per_defect})
    path = Path(path)
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        extra = extra or {}
        if new:
            w.writerow(list(extra) + ["config", "fpr", "recall"] + labels + ["time_ms"])
        for r in rows:
            w.writerow(list(extra.values()) + [r.config, r.fpr, r.recall] + [r.per_defect.get(k, "") for k in labels] + [r.time_ms])


def synthetic_sets(part_type: str, lighting: str, n_train=15, n_good=40, n_per_defect=10, seed=0, any_angle=False):
    from . import synthetic as S

    rng = np.random.default_rng(seed)
    train = [S.compose(S.make_part(part_type, None, rng), S.random_pose(rng, any_angle=any_angle), lighting, rng=rng) for _ in range(n_train)]
    test = [(S.compose(S.make_part(part_type, None, rng), S.random_pose(rng, any_angle=any_angle), lighting, rng=rng), "ok") for _ in range(n_good)]
    for d in S.DEFECTS:
        test += [(S.compose(S.make_part(part_type, d, rng), S.random_pose(rng, any_angle=any_angle), lighting, rng=rng), d) for _ in range(n_per_defect)]
    return train, test
