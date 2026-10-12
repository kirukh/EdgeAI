"""Continuous improvement: the operator accepts deviations that are within the company's tolerance.

The key property: an accepted deviation is tolerated *locally* (that hole, that image region).
The model must not become blind for other defects – which is what happens if a deviating part is
simply added as a normal reference (see test_screening.py).
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import wait, web_client

from qc import synthetic as S
from qc.config import LocalizationConfig, MethodConfig
from qc.learning import KNOWN_BAD, PENDING, LearningStore, release, update_model
from qc.methods import GeometryMethod
from qc.model import QCModel, has_previous, restore_previous

LOC, MC = LocalizationConfig(), MethodConfig()


def part(rng, defect=None, offset=0.0, hole=0, stain_at=None) -> np.ndarray:
    """Plate A; ``offset`` shifts one hole along x (a systematic deviation, e.g. a jig set slightly off)."""
    p = S.make_part("A", defect, rng)
    p.holes[hole]["x"] += offset
    if stain_at is not None:
        p.stains.append({"x": stain_at[0] + rng.normal(0, 2), "y": stain_at[1] + rng.normal(0, 2), "rx": 16, "ry": 11})
    return S.compose(p, S.random_pose(rng), "front", rng=rng)


def nok_rate(model: QCModel, frames: list[np.ndarray]) -> float:
    return float(np.mean([model.inspect(f, False).status == "NOK" for f in frames]))


@pytest.fixture(scope="module")
def taught():
    rng = np.random.default_rng(9)
    return [part(rng) for _ in range(19)], part(rng, offset=9), part(rng, stain_at=(20, -40))


@pytest.fixture(scope="module")
def tests():
    rng = np.random.default_rng(5)
    return {
        "good": [part(rng) for _ in range(20)],
        "like_accepted": [part(rng, offset=o) for o in (8, 9) for _ in range(8)],
        "much_more": [part(rng, offset=18) for _ in range(8)],
        "other_hole": [part(rng, offset=9, hole=1) for _ in range(8)],
        "stain_there": [part(rng, stain_at=(20, -40)) for _ in range(8)],
        "stain_elsewhere": [part(rng, "stain") for _ in range(10)],
        "scratch": [part(rng, "scratch") for _ in range(10)],
    }


# ------------------------------------------------------------------ learning a deviation
def test_accepted_hole_offset_is_tolerated_only_there(taught, tests):
    good, offset_part, _ = taught
    before = QCModel.train("p", good, LOC, MC)
    after = QCModel.train("p", [*good, offset_part], LOC, MC, accepted=[19])
    assert after.accepted == [19] and after.report["references_excluded"] == []
    assert nok_rate(before, tests["like_accepted"]) == 1.0
    assert nok_rate(after, tests["like_accepted"]) == 0.0  # the operator's decision is applied
    assert nok_rate(after, tests["much_more"]) == 1.0  # a larger offset is still a defect
    assert nok_rate(after, tests["other_hole"]) == 1.0  # … and so is the same offset at another hole
    assert nok_rate(after, tests["good"]) == 0.0
    # not blind for other defects (adding it as a normal reference would cost ~ all scratches)
    assert nok_rate(after, tests["scratch"]) == nok_rate(before, tests["scratch"])
    assert nok_rate(after, tests["stain_elsewhere"]) >= 0.9


def test_accepted_surface_mark_is_tolerated_only_at_its_place(taught, tests):
    good, _, stain_part = taught
    before = QCModel.train("p", good, LOC, MC)
    after = QCModel.train("p", [*good, stain_part], LOC, MC, accepted=[19])
    assert nok_rate(before, tests["stain_there"]) == 1.0
    assert nok_rate(after, tests["stain_there"]) == 0.0
    assert nok_rate(after, tests["stain_elsewhere"]) == 1.0
    assert nok_rate(after, tests["scratch"]) == nok_rate(before, tests["scratch"])


def test_accepted_parts_are_not_removed_by_the_screening(taught):
    good, offset_part, _ = taught
    plain = QCModel.train("p", [*good, offset_part], LOC, MC)  # not accepted → looks like a mistake
    assert [e["index"] for e in plain.report["references_excluded"]] == [19]
    accepted = QCModel.train("p", [*good, offset_part], LOC, MC, accepted=[19])
    assert accepted.report["references_accepted"] == [19]
    assert not any("image(s) 20 " in w for w in accepted.warnings())  # an accepted part is no warning


def test_model_without_learned_tolerances_loads():
    """Models saved before this feature have no tolerance arrays – they load with none."""
    rng = np.random.default_rng(1)
    m = QCModel.train("p", [part(rng) for _ in range(6)], LOC, MethodConfig(methods=["geometry"]))
    meta, arrays = m.methods["geometry"].state()
    for k in ("pos_accept", "d_accept", "aspect_accept"):
        arrays.pop(k)
    g = GeometryMethod(MethodConfig())
    g.load_state(meta, arrays)
    assert not g.pos_accept.any() and g.outline_tolerated is None


# ------------------------------------------------------------------ versions, release check, undo
def _saved(tmp_path, taught) -> QCModel:
    good, _, _ = taught
    m = QCModel.train("Plate", good[:12], LOC, MC)
    m.save(tmp_path)
    return QCModel.load(tmp_path / "Plate")


def test_update_release_and_undo(tmp_path, taught, tests):
    model = _saved(tmp_path, taught)
    store = LearningStore(model.path)
    rng = np.random.default_rng(3)
    for _ in range(2):
        store.add(PENDING, part(rng, offset=9), {"defects": ["Hole mispositioned"]}, limit=10)
    store.add(KNOWN_BAD, part(rng, "missing"), {}, limit=10)
    new, report = update_model(model, store)
    assert new is not None and report.released and report.version_after == 2
    assert report.known_bad_checked == report.known_bad_still_detected == 1
    release(new, store, report)
    assert store.count(PENDING) == 0 and store.count(KNOWN_BAD) == 1  # test parts stay with the part type
    v2 = QCModel.load(model.path)
    assert v2.version == 2 and len(v2.accepted) == 2 and v2.history[-1]["accepted_parts"] == 2
    assert nok_rate(v2, tests["like_accepted"]) == 0.0
    assert has_previous(model.path)
    v1 = restore_previous(model.path)  # undo …
    assert v1.version == 1 and LearningStore(model.path).count(KNOWN_BAD) == 1
    assert restore_previous(model.path).version == 2  # … and the undo can be undone


def test_release_check_rejects_a_contradicting_acceptance(tmp_path, taught):
    """Earlier the operator said "defective" for this deviation – accepting it now would make the
    model miss that part. The update is rejected and the model stays unchanged."""
    model = _saved(tmp_path, taught)
    store = LearningStore(model.path)
    rng = np.random.default_rng(4)
    store.add(KNOWN_BAD, part(rng, offset=9), {}, limit=10)
    store.add(PENDING, part(rng, offset=9), {}, limit=10)
    new, report = update_model(model, store)
    assert new is None and not report.released
    assert report.known_bad_checked == 1 and report.known_bad_still_detected == 0
    assert "no longer detect 1 of 1" in report.reasons[0]
    assert QCModel.load(model.path).version == 1 and store.count(PENDING) == 1


def test_new_teach_in_starts_fresh_but_can_be_undone(tmp_path, taught):
    model = _saved(tmp_path, taught)
    store = LearningStore(model.path)
    store.add(PENDING, taught[1], {}, limit=10)
    good = taught[0]
    QCModel.train("Plate", good[5:], LOC, MC).save(tmp_path, keep_previous=True, keep_side_data=False)
    assert store.count(PENDING) == 0 and has_previous(model.path)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["Plate"]


def test_store_keeps_the_newest_parts_and_rejects_foreign_names(tmp_path, taught):
    store = LearningStore(tmp_path)
    names = [store.add(PENDING, taught[0][0], {"n": i}, limit=3) for i in range(5)]
    assert store.stems(PENDING) == names[-3:]
    assert store.image_path(PENDING, names[-1]) is not None
    assert store.image_path(PENDING, "../../etc/passwd") is None


# ------------------------------------------------------------------ in operation
def test_operator_acceptance_updates_the_model_during_inspection(make_system):
    s = make_system(sim_defect_rate=0.6)
    s.cfg.learning.update_after = 2
    s.start()
    s.start_learning("Plate")
    wait(lambda: s.teach.count >= 8)
    s.finish_learning()
    wait(lambda: s.mode == "idle" and s.model is not None)
    s.start_inspection()
    wait(lambda: sum(h["status"] == "NOK" for h in s.results.history) >= 2, 120)
    noks = [h["image_id"] for h in s.results.history if h["status"] == "NOK"][:2]
    assert s.feedback(noks[0], "ok")["waiting"] == 1
    assert s.feedback(noks[1], "ok")["waiting"] == 2  # → update starts in the background
    wait(lambda: s.last_update is not None and not s._update_running, 120)
    assert s.last_update["released"], s.last_update
    assert s.mode == "inspecting" and s.model.version == 2  # inspection was never interrupted
    n = len(s.results.history)
    wait(lambda: len(s.results.history) > n, 60)
    assert s.results.history[0]["model_version"] == 2
    s.stop_inspection()
    assert next(m for m in s.list_models() if m["slug"] == "Plate")["has_previous"]
    s.undo_update("Plate")
    assert s.model.version == 1


def test_update_endpoints_need_the_setup_pin(make_system, taught):
    s = make_system()
    model = QCModel.train("Plate", taught[0][:8], LOC, MC)
    model.save(s.cfg.storage.models_dir)
    stem = LearningStore(model.path).add(PENDING, taught[1], {"defects": ["Hole mispositioned"]}, limit=10)
    c = web_client(s)
    assert c.post("/api/models/Plate/update").status_code == 403
    assert c.get("/api/models/Plate/waiting").status_code == 403
    h = {"X-Setup-Token": c.post("/api/setup/unlock", json={"pin": s.cfg.ui.setup_pin}).json["token"]}
    waiting = c.get("/api/models/Plate/waiting", headers=h).json
    assert [w["name"] for w in waiting] == [stem] and waiting[0]["defects"] == ["Hole mispositioned"]
    assert c.get(f"/api/models/Plate/waiting/{stem}.png", headers=h).status_code == 200
    assert c.delete(f"/api/models/Plate/waiting/{stem}", headers=h).json["ok"]
    assert c.post("/api/models/Plate/update", headers=h).status_code == 400  # nothing waiting any more
    assert c.post("/api/models/Plate/undo", headers=h).status_code == 400  # no previous version yet


def test_batch_summary_compares_model_versions(tmp_path):
    from qc import archive as qa

    day = tmp_path / "2026-10-12"
    day.mkdir()
    (day / "inspection_log.csv").write_text(
        "time,model,result,defects,duration_ms,image,ground_truth,image_id,shots,model_version\n"
        + "".join(f"t,Plate,{r},,1,,,{i:06d},OK,1\n" for i, r in enumerate(["NOK", "NOK", "OK", "OK"]))
        + "".join(f"t,Plate,OK,,1,,,{i:06d},OK,2\n" for i in range(4, 8)),
        encoding="utf-8",
    )
    (day / "feedback_log.csv").write_text(
        "time,image_id,inspected_at,model,result,supervisor,outcome,model_version\n"
        "t,000000,t,Plate,NOK,OK,false_alarm,1\nt,000001,t,Plate,NOK,OK,false_alarm,1\n",
        encoding="utf-8",
    )
    v = qa.summarize(tmp_path)["model_versions"]
    assert v["Plate v1"]["false_alarm_rate_percent"] == 50.0 and v["Plate v2"]["false_alarms"] == 0
    assert "Model versions" in qa._summary_text("batch", qa.summarize(tmp_path))
