"""Stored part types (models) on disk: find, load, list, save and delete.

``ModelStore`` is the component of ``QCSystem`` for the model folders (``storage.models_dir``). Which model
is *active* stays with the system. Names are resolved safely – a name that tries to leave the folder is
"not found" (``model_path``).
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from .learning import KNOWN_BAD, PENDING, LearningStore
from .model import QCModel, has_previous, list_models, model_path


class ModelStore:
    """Model folders on disk: find, load, list, save and delete part types."""

    def __init__(self, models_dir: str | Path):
        self.dir = Path(models_dir)
        self.dir.mkdir(parents=True, exist_ok=True)

    def find(self, slug: str) -> Path | None:
        """Folder of the model, or None."""
        return model_path(self.dir, slug)

    def path(self, slug: str) -> Path:
        """Folder of the model; ValueError (shown to the user) if there is none."""
        path = self.find(slug)
        if path is None:
            raise ValueError("Model not found.")
        return path

    def load(self, slug: str) -> QCModel:
        """Loads a model by name."""
        return QCModel.load(self.path(slug))

    def save(self, model: QCModel, keep_previous: bool = False, keep_side_data: bool = True) -> Path:
        """Saves a model into its folder (``keep_previous``: the old version stays restorable)."""
        return model.save(self.dir, keep_previous=keep_previous, keep_side_data=keep_side_data)

    def delete(self, slug: str) -> None:
        """Deletes a model folder."""
        shutil.rmtree(self.path(slug))

    def list(self) -> list[dict[str, Any]]:
        """Summary of every model, with its accepted parts, test parts and whether it can be undone."""
        models = list_models(self.dir)
        for m in models:
            path = self.dir / m["slug"]
            store = LearningStore(path)
            m.update(waiting=store.count(PENDING), known_bad=store.count(KNOWN_BAD), has_previous=has_previous(path))
        return models
