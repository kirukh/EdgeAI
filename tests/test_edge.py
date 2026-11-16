"""Edge inspection (``edge`` method): corner radius, unclean milled edge, chamfer/deburring.

These defects sit on the edge itself, where the image comparison is blind and the outline
comparison of ``geometry`` only sees deviations several pixels wide.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from qc import synthetic as S
from qc.config import LocalizationConfig, MethodConfig
from qc.model import QCModel

KIND = {"radius": "EDGE_SHAPE", "rough_edge": "EDGE_ROUGH", "chamfer": "EDGE_BAND"}


def frame(rng: np.random.Generator, defect: str | None = None, part_type: str = "E") -> np.ndarray:
    return S.compose(S.make_part(part_type, defect, rng), S.random_pose(rng), "front", rng=rng)


@pytest.fixture(scope="module")
def model() -> QCModel:
    rng = np.random.default_rng(11)
    return QCModel.train("Milled", [frame(rng) for _ in range(15)], LocalizationConfig(), MethodConfig())


def test_good_parts_pass_and_edge_defects_are_named(model):
    rng = np.random.default_rng(5)
    assert all(model.inspect(frame(rng), False).ok for _ in range(25))
    for defect, kind in KIND.items():
        results = [model.inspect(frame(rng, defect), False) for _ in range(12)]
        found = [r for r in results if not r.ok]
        # the chamfer is the hardest: a brightness band in a brushed surface, weakest along the brush marks
        assert len(found) >= (7 if defect == "chamfer" else 11), defect
        named = sum(any(d.kind == kind for d in r.defects) for r in found)
        assert named >= len(found) - 2, (defect, [[d.kind for d in r.defects] for r in found])


def test_the_other_methods_miss_these_defects():
    """Why the method exists: without it a chamfer that is too wide is not found at all."""
    rng = np.random.default_rng(11)
    without = MethodConfig(methods=["geometry", "diff", "pca"])
    m = QCModel.train("Milled", [frame(rng) for _ in range(15)], LocalizationConfig(), without)
    assert sum(not m.inspect(frame(rng, "chamfer"), False).ok for _ in range(12)) <= 2


def test_edge_text_tells_where_and_how_much(model):
    rng = np.random.default_rng(8)
    res = next(r for r in (model.inspect(frame(rng, "radius"), False) for _ in range(5)) if not r.ok)
    d = next(d for d in res.defects if d.kind == "EDGE_SHAPE")
    assert d.detail.startswith("outline: radius/corner off by ") and " px over " in d.detail
    assert d.label == "Edge contour / radius deviates"


def test_edge_model_survives_save_and_load(model, tmp_path):
    rng = np.random.default_rng(2)
    model.save(tmp_path)
    loaded = QCModel.load(tmp_path / "Milled")
    for defect in (None, "chamfer", "rough_edge"):
        f = frame(rng, defect)
        a = next(r for r in model.inspect(f, False).method_results if r.method == "edge")
        b = next(r for r in loaded.inspect(f, False).method_results if r.method == "edge")
        assert a.score == pytest.approx(b.score, rel=1e-5) and a.ok == b.ok


def test_hole_position_is_left_to_geometry(model):
    """A hole inside its position tolerance must not show up as an edge defect of the cut-out."""
    rng = np.random.default_rng(4)
    for _ in range(8):
        p = S.make_part("E", None, rng)
        p.holes[0]["x"] += 4.0  # geometry tolerance: 6 px
        res = model.inspect(S.compose(p, S.random_pose(rng), "front", rng=rng), False)
        edge = next(r for r in res.method_results if r.method == "edge")
        assert edge.ok and edge.score < 0.9, [(d.kind, d.detail) for d in edge.defects]


def test_higher_working_resolution_gives_the_same_decisions():
    """All pixel settings scale with ``localization.work_width``."""

    def big(rng: np.random.Generator, defect: str | None = None) -> np.ndarray:
        return cv2.resize(frame(rng, defect), None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)

    rng = np.random.default_rng(11)
    m = QCModel.train("Milled", [big(rng) for _ in range(12)], LocalizationConfig(work_width=960), MethodConfig())
    assert m.methods["edge"].scale == 1.5
    assert all(m.inspect(big(rng), False).ok for _ in range(8))
    for defect in ("radius", "rough_edge", "missing", "position", "stain"):
        assert sum(not m.inspect(big(rng, defect), False).ok for _ in range(4)) >= 3, defect


def test_accepted_chamfer_is_tolerated_only_there():
    """The operator accepts a part that was deburred deeper at one place: tolerated there, not elsewhere."""
    rng = np.random.default_rng(3)
    good = [frame(rng) for _ in range(15)]

    def deep(edge: int) -> np.ndarray:
        p = S.make_part("E", None, rng)
        mid = (p.outline[edge] + p.outline[(edge + 1) % 4]) / 2
        p.chamfer_defect = {"x": float(mid[0]), "y": float(mid[1]), "radius": 55.0, "width": 11.0}
        return S.compose(p, (320, 240, float(rng.uniform(-6, 6))), "front", rng=rng)

    def rate(m: QCModel, edge: int) -> float:
        return float(np.mean([not m.inspect(deep(edge), False).ok for _ in range(8)]))

    before = QCModel.train("Milled", good, LocalizationConfig(), MethodConfig())
    after = QCModel.train("Milled", [*good, deep(1)], LocalizationConfig(), MethodConfig(), accepted=[15])
    assert rate(before, 1) >= 0.85 and rate(after, 1) <= 0.15
    assert rate(after, 3) >= 0.85  # the opposite edge is still checked
