"""Reference screening: a defective part taught in by mistake must be found and left out.

Without screening, ONE defective reference among 15 loosened the thresholds of the
statistical methods many times over (e.g. image comparison 30 → 858 px) and the model
missed almost all scratches afterwards – without any warning.
"""

from __future__ import annotations

import numpy as np
import pytest

from qc import synthetic as S
from qc.config import LocalizationConfig, MethodConfig
from qc.model import QCModel


def shots(rng, n, defect=None, part_type="A", lighting="front", any_angle=False):
    return [
        S.compose(S.make_part(part_type, defect, rng), S.random_pose(rng, any_angle=any_angle), lighting, rng=rng)
        for _ in range(n)
    ]


def excluded(model: QCModel) -> list[int]:
    return [e["index"] for e in model.report["references_excluded"]]


@pytest.fixture(scope="module")
def test_parts():
    rng = np.random.default_rng(99)
    return {"ok": shots(rng, 15), "scratch": shots(rng, 10, "scratch"), "position": shots(rng, 10, "position")}


@pytest.mark.parametrize("bad", ["scratch", "position", "stain", "missing", "corner"])
def test_defective_reference_is_left_out(bad, test_parts):
    rng = np.random.default_rng(7)
    model = QCModel.train("t", shots(rng, 14) + shots(rng, 1, bad), LocalizationConfig(), MethodConfig())
    assert excluded(model) == [14], model.report["references_excluded"]
    assert model.report["references_used"] == 14
    assert len(model.reference_frames()) == 14  # the bad part is not stored as reference
    # thresholds stay tight → the model still finds the defects
    assert model.methods["diff"].min_area < 60
    for kind in ("scratch", "position"):
        nok = sum(model.inspect(f, visualize=False).status == "NOK" for f in test_parts[kind])
        assert nok >= 9, f"{kind}: only {nok}/10 detected"
    assert all(model.inspect(f, visualize=False).ok for f in test_parts["ok"])


def test_without_screening_the_model_goes_blind(test_parts):
    """Documents the problem the screening solves (same data as above, screening off)."""
    rng = np.random.default_rng(7)
    train = shots(rng, 14) + shots(rng, 1, "position")
    model = QCModel.train("t", train, LocalizationConfig(), MethodConfig(screen_references=False))
    assert excluded(model) == []
    assert model.methods["diff"].min_area > 300
    assert sum(model.inspect(f, visualize=False).status == "NOK" for f in test_parts["scratch"]) <= 2


def test_two_defective_references_are_both_found():
    rng = np.random.default_rng(8)
    train = shots(rng, 2) + shots(rng, 1, "scratch") + shots(rng, 11) + shots(rng, 1, "position")
    model = QCModel.train("t", train, LocalizationConfig(), MethodConfig())
    assert excluded(model) == [2, 14]


@pytest.mark.parametrize(
    "part_type,lighting,any_angle",
    [
        ("A", "front", False),
        ("B", "front", False),
        ("C", "front", True),
        ("D", "front", True),
        ("A", "back", False),
        ("B", "back", False),
    ],
)
def test_clean_teach_in_excludes_nothing(part_type, lighting, any_angle):
    for seed in range(2):
        rng = np.random.default_rng(1000 * seed + ord(part_type))
        model = QCModel.train(
            "t", shots(rng, 15, None, part_type, lighting, any_angle), LocalizationConfig(), MethodConfig()
        )
        assert excluded(model) == [], model.report["references_excluded"]


def test_too_many_odd_references_are_reported_not_excluded():
    """If many references stand out, the cause is the setup (light, focus …), not single parts:
    nothing is excluded, the supervisor gets a warning."""
    rng = np.random.default_rng(9)
    model = QCModel.train("t", shots(rng, 10) + shots(rng, 5, "position"), LocalizationConfig(), MethodConfig())
    assert model.report["references_excluded"] == []
    assert {e["index"] for e in model.report["references_suspect"]} >= {10, 11, 12, 13, 14}
    assert any("differ from the other good parts" in w for w in model.warnings())


def test_warning_names_the_excluded_reference():
    rng = np.random.default_rng(7)
    model = QCModel.train("t", shots(rng, 14) + shots(rng, 1, "missing"), LocalizationConfig(), MethodConfig())
    w = [x for x in model.warnings() if "left out" in x]
    assert w and "Reference image(s) 15 " in w[0]
