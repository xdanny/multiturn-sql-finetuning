from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval.training_manifest_pair import TrainingManifestSpec, validate_training_manifest_path


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n")


def test_validate_training_manifest_path_accepts_expected_stage_benchmark_and_mode(
    tmp_path: Path,
) -> None:
    train_path = tmp_path / "train.jsonl"
    manifest_path = tmp_path / "training.manifest.json"
    _write_json(
        manifest_path,
        {
            "stage": "metric_dsl",
            "benchmark": "synthetic_metric_dsl_bootstrap",
            "evaluation_mode": "metric_dsl",
            "train_data_path": str(train_path),
        },
    )

    manifest, resolved_train_path = validate_training_manifest_path(
        manifest_path=manifest_path,
        spec=TrainingManifestSpec(
            label="metric DSL",
            stage="metric_dsl",
            benchmark="synthetic_metric_dsl_bootstrap",
            evaluation_mode="metric_dsl",
        ),
    )

    assert manifest["stage"] == "metric_dsl"
    assert resolved_train_path == train_path


def test_validate_training_manifest_path_rejects_wrong_stage(tmp_path: Path) -> None:
    manifest_path = tmp_path / "training.manifest.json"
    _write_json(
        manifest_path,
        {
            "stage": "semantic_prompt",
            "benchmark": "synthetic_semantic_layer",
            "evaluation_mode": "non_oracle_generation",
            "train_data_path": str(tmp_path / "train.jsonl"),
        },
    )

    with pytest.raises(ValueError, match="semantic_layer"):
        validate_training_manifest_path(
            manifest_path=manifest_path,
            spec=TrainingManifestSpec(
                label="semantic_layer",
                stage="semantic_layer",
                benchmark="synthetic_semantic_layer",
                evaluation_mode="non_oracle_generation",
            ),
        )


def test_validate_training_manifest_path_requires_train_data_path(tmp_path: Path) -> None:
    manifest_path = tmp_path / "training.manifest.json"
    _write_json(
        manifest_path,
        {
            "stage": "direct_sql_control",
            "benchmark": "prepared",
            "evaluation_mode": "non_oracle_generation",
        },
    )

    with pytest.raises(ValueError, match="train_data_path"):
        validate_training_manifest_path(
            manifest_path=manifest_path,
            spec=TrainingManifestSpec(
                label="direct SQL",
                stage="direct_sql_control",
                benchmark="prepared",
                evaluation_mode="non_oracle_generation",
            ),
        )
