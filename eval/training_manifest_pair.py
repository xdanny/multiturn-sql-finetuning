"""Shared validation helpers for paired training-manifest inputs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TrainingManifestSpec:
    label: str
    stage: str
    benchmark: str
    evaluation_mode: str


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def validate_training_manifest_path(
    *,
    manifest_path: Path,
    spec: TrainingManifestSpec,
) -> tuple[dict[str, Any], Path]:
    manifest = _load_json(manifest_path)
    if manifest.get("stage") != spec.stage:
        raise ValueError(f"{spec.label} training manifest must use stage={spec.stage}")
    if manifest.get("benchmark") != spec.benchmark:
        raise ValueError(f"{spec.label} training manifest must use benchmark={spec.benchmark}")
    if manifest.get("evaluation_mode") != spec.evaluation_mode:
        raise ValueError(
            f"{spec.label} training manifest must use evaluation_mode={spec.evaluation_mode}"
        )
    train_data_path = manifest.get("train_data_path")
    if not train_data_path:
        raise ValueError(f"{spec.label} training manifest is missing train_data_path")
    return manifest, Path(str(train_data_path))
