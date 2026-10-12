"""Shared test helpers: synthetic parts, a simulated system and a web client."""

from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np
import pytest

from qc import synthetic as S
from qc.camera import SimulatorSource
from qc.config import AppConfig


def shots(
    part_type: str, lighting: str, n: int, rng: np.random.Generator, defect: str | None = None, **pose
) -> list[np.ndarray]:
    """``n`` images of synthetic parts (each a new part in a random pose)."""
    return [
        S.compose(S.make_part(part_type, defect, rng), S.random_pose(rng, **pose), lighting, rng=rng) for _ in range(n)
    ]


def wait(cond: Callable[[], bool], timeout: float = 90) -> None:
    t0 = time.time()
    while not cond():
        assert time.time() - t0 < timeout, "Timeout"
        time.sleep(0.05)


def make_config(tmp_path) -> AppConfig:
    cfg = AppConfig()
    cfg.camera.fps, cfg.camera.width, cfg.camera.height, cfg.camera.sim_speed_px = 400, 640, 480, 20
    cfg.storage.models_dir = str(tmp_path / "models")
    cfg.storage.results_dir = str(tmp_path / "results")
    cfg.storage.archive_dir = str(tmp_path / "archive")
    return cfg


@pytest.fixture
def make_system(tmp_path):
    """Factory for a QCSystem with a simulated conveyor; started systems are stopped afterwards."""
    from qc.system import QCSystem

    created = []

    def factory(seed: int = 5, **camera) -> QCSystem:
        cfg = make_config(tmp_path)
        for k, v in camera.items():
            setattr(cfg.camera, k, v)
        s = QCSystem(cfg, SimulatorSource(cfg.camera, seed=seed))
        created.append(s)
        return s

    yield factory
    for s in created:
        if s._threads:
            s.stop()


def web_client(system):
    """Flask test client that sends the CSRF header like the real page does."""
    from qc.web import create_app

    client = create_app(system).test_client()
    client.environ_base["HTTP_X_QC_CLIENT"] = "1"
    return client
