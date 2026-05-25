from __future__ import annotations

import json

import pytest

from eval.compare_rollout_history import (
    compare_rollout_manifest_files,
    compare_rollout_manifests,
)


def _teacher_manifest() -> dict:
    return {
        "schema_version": 1,
        "run_id": "teacher",
        "benchmark": "prepared",
        "model_name": "local-9b",
        "endpoint": "local",
        "evaluation_mode": "non_oracle_generation",
        "oracle_allowed": False,
        "input_path": "data/eval.jsonl",
        "input_sha256": "abc123",
        "output_path": "results/teacher.jsonl",
        "output_sha256": "teacher-output",
        "row_count": 2,
        "metrics": {
            "history_policy": "gold_sql_teacher_forced",
            "value_execution_accuracy": 0.5,
            "strict_execution_accuracy": 0.25,
            "dialog_count": 1,
        },
        "command": ["run-teacher"],
    }


def _rollout_manifest() -> dict:
    return {
        "schema_version": 1,
        "run_id": "rollout",
        "benchmark": "prepared_rollout",
        "model_name": "local-9b",
        "endpoint": "local",
        "evaluation_mode": "non_oracle_generation",
        "oracle_allowed": False,
        "input_path": "data/eval.jsonl",
        "input_sha256": "abc123",
        "output_path": "results/rollout.jsonl",
        "output_sha256": "rollout-output",
        "row_count": 2,
        "metrics": {
            "history_policy": "model_generated_sql_rollout",
            "value_execution_accuracy": 0.75,
            "strict_execution_accuracy": 0.5,
            "dialog_count": 1,
        },
        "command": ["run-rollout"],
    }


def test_compare_rollout_manifests_adds_same_model_same_input_delta() -> None:
    compared = compare_rollout_manifests(
        rollout_manifest=_rollout_manifest(),
        teacher_forced_manifest=_teacher_manifest(),
    )

    assert compared["run_id"] == "rollout"
    assert compared["metrics"]["teacher_forced_comparison_run_id"] == "teacher"
    assert compared["metrics"]["teacher_forced_model_name"] == "local-9b"
    assert compared["metrics"]["teacher_forced_input_sha256"] == "abc123"
    assert compared["metrics"]["teacher_forced_value_execution_accuracy"] == 0.5
    assert compared["metrics"]["teacher_forced_strict_execution_accuracy"] == 0.25
    assert compared["metrics"]["rollout_value_delta_vs_teacher_forced"] == pytest.approx(0.25)
    assert compared["metrics"]["rollout_strict_delta_vs_teacher_forced"] == pytest.approx(0.25)


def test_compare_rollout_manifests_rejects_model_mismatch() -> None:
    rollout = _rollout_manifest()
    teacher = _teacher_manifest()
    teacher["model_name"] = "other-model"

    with pytest.raises(ValueError, match="same model"):
        compare_rollout_manifests(
            rollout_manifest=rollout,
            teacher_forced_manifest=teacher,
        )


def test_compare_rollout_manifests_rejects_input_mismatch() -> None:
    rollout = _rollout_manifest()
    teacher = _teacher_manifest()
    teacher["input_sha256"] = "different"

    with pytest.raises(ValueError, match="same input"):
        compare_rollout_manifests(
            rollout_manifest=rollout,
            teacher_forced_manifest=teacher,
        )


def test_compare_rollout_manifests_rejects_oracle_rollout() -> None:
    rollout = _rollout_manifest()
    rollout["oracle_allowed"] = True
    rollout["evaluation_mode"] = "oracle_planner_diagnostic"

    with pytest.raises(ValueError, match="non-oracle"):
        compare_rollout_manifests(
            rollout_manifest=rollout,
            teacher_forced_manifest=_teacher_manifest(),
        )


def test_compare_rollout_manifests_rejects_oracle_teacher_forced_manifest() -> None:
    teacher = _teacher_manifest()
    teacher["oracle_allowed"] = True
    teacher["evaluation_mode"] = "oracle_planner_diagnostic"

    with pytest.raises(ValueError, match="non-oracle"):
        compare_rollout_manifests(
            rollout_manifest=_rollout_manifest(),
            teacher_forced_manifest=teacher,
        )


def test_compare_rollout_manifests_requires_teacher_forced_history_policy() -> None:
    teacher = _teacher_manifest()
    teacher["metrics"].pop("history_policy")

    with pytest.raises(ValueError, match="gold_sql_teacher_forced"):
        compare_rollout_manifests(
            rollout_manifest=_rollout_manifest(),
            teacher_forced_manifest=teacher,
        )


def test_compare_rollout_manifest_files_writes_augmented_manifest(tmp_path) -> None:
    rollout_path = tmp_path / "rollout.manifest.json"
    teacher_path = tmp_path / "teacher.manifest.json"
    output_path = tmp_path / "comparison.manifest.json"
    rollout_path.write_text(json.dumps(_rollout_manifest()) + "\n")
    teacher_path.write_text(json.dumps(_teacher_manifest()) + "\n")

    compared = compare_rollout_manifest_files(
        rollout_manifest_path=rollout_path,
        teacher_forced_manifest_path=teacher_path,
        output_path=output_path,
    )

    written = json.loads(output_path.read_text())
    assert written == compared
    assert written["metrics"]["teacher_forced_comparison_run_id"] == "teacher"
