from __future__ import annotations

import json
from pathlib import Path

import joblib

from app.core.config import MODELS_DIR


class ModelLoader:
    """Read the flat, immutable artifact release shipped with this repository."""

    def __init__(self, models_dir: Path = MODELS_DIR):
        self.models_dir = models_dir

    def _path(self, module: str, name: str) -> Path:
        root = self.models_dir.resolve()
        path = (root / module / name).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise FileNotFoundError(f"artifact missing or invalid: {module}/{name}")
        return path

    def load_joblib(self, module: str, name: str):
        return joblib.load(self._path(module, name))

    def load_json(self, module: str, name: str) -> dict:
        return json.loads(self._path(module, name).read_text())
