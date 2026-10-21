from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from app.models.loader import ModelLoader

HFCE_SARIMA_MODULE = "hfce_sarima_v4"
BUDGET_MODULE = "budget"


@dataclass
class ModuleModel:
    module: str
    model: Any
    evaluation: dict
    feature_columns: list[str]
    metadata: dict = field(default_factory=dict)

    @property
    def model_id(self) -> str:
        return self.metadata.get("model_id") or self.module


class ModelRegistry:
    """Load only the pinned serving release; fail startup on corrupt artifacts."""

    def __init__(self, loader: ModelLoader | None = None):
        self.loader = loader or ModelLoader()
        self.hfce_sarima: ModuleModel | None = None
        self.budget: ModuleModel | None = None

    def _verify_manifest(self) -> None:
        root = self.loader.models_dir.resolve()
        manifest = json.loads((root / "manifest.json").read_text())
        expected = {
            "hfce_sarima_v4/metadata.json",
            "hfce_sarima_v4/evaluation.json",
            "budget/metadata.json",
            "budget/evaluation.json",
            "budget/budget_config.json",
        }
        metadata = self.loader.load_json(HFCE_SARIMA_MODULE, "metadata.json")
        expected.update(f"{HFCE_SARIMA_MODULE}/{name}" for name in metadata["artifacts"])
        if set(manifest["sha256"]) != expected:
            raise ValueError("model manifest does not match the serving release")
        for relative, checksum in manifest["sha256"].items():
            path = (root / relative).resolve()
            if not path.is_relative_to(root) or not path.is_file():
                raise ValueError(f"invalid model release path: {relative}")
            if hashlib.sha256(path.read_bytes()).hexdigest() != checksum:
                raise ValueError(f"model release checksum mismatch: {relative}")

    def load_all(self) -> None:
        self._verify_manifest()
        evaluation = self.loader.load_json(HFCE_SARIMA_MODULE, "evaluation.json")
        metadata = self.loader.load_json(HFCE_SARIMA_MODULE, "metadata.json")
        categories = evaluation["categories"]
        if len(categories) != 12:
            raise ValueError("the HFCE serving release must contain all 12 categories")
        artifacts = {}
        for category, details in categories.items():
            name = details["artifact"]
            if name != f"{category}.joblib" or name not in metadata["artifacts"]:
                raise ValueError("invalid HFCE category artifact")
            artifact = self.loader.load_joblib(HFCE_SARIMA_MODULE, name)
            if artifact.get("family") != "sarima" or artifact.get("category") != category:
                raise ValueError("HFCE artifact does not match its category or model family")
            if artifact.get("transform") not in {"identity", "log", "log1p"}:
                raise ValueError("unsupported HFCE transformation")
            artifacts[category] = artifact
        hfce = ModuleModel(HFCE_SARIMA_MODULE, artifacts, evaluation, [], metadata)
        budget = ModuleModel(
            BUDGET_MODULE,
            self.loader.load_json(BUDGET_MODULE, "budget_config.json"),
            self.loader.load_json(BUDGET_MODULE, "evaluation.json"),
            [],
            self.loader.load_json(BUDGET_MODULE, "metadata.json"),
        )
        self.hfce_sarima, self.budget = hfce, budget

    @property
    def loaded_modules(self) -> list[str]:
        return [
            name
            for name, model in (
                (HFCE_SARIMA_MODULE, self.hfce_sarima),
                (BUDGET_MODULE, self.budget),
            )
            if model is not None
        ]

    @property
    def is_ready(self) -> bool:
        return (
            self.hfce_sarima is not None
            and len(self.hfce_sarima.model) == 12
            and self.budget is not None
        )
