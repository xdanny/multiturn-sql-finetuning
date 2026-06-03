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


def _rows(history_policy: str) -> list[dict]:
    evaluation_mode = "non_oracle_generation"
    return [
        {
            "id": "dialog-a:0",
            "dialog_id": "dialog-a",
            "turn_index": 0,
            "database_id": "store",
            "reference_sql": "SELECT COUNT(*) FROM orders;",
            "evaluation_mode": evaluation_mode,
            "history_policy": history_policy,
        },
        {
            "id": "dialog-a:1",
            "dialog_id": "dialog-a",
            "turn_index": 1,
            "database_id": "store",
            "reference_sql": "SELECT SUM(amount) FROM orders;",
            "evaluation_mode": evaluation_mode,
            "history_policy": history_policy,
        },
    ]


def test_compare_rollout_manifests_adds_same_model_same_input_delta() -> None:
    compared = compare_rollout_manifests(
        rollout_manifest=_rollout_manifest(),
        teacher_forced_manifest=_teacher_manifest(),
        rollout_rows=_rows("model_generated_sql_rollout"),
        teacher_forced_rows=_rows("gold_sql_teacher_forced"),
    )

    assert compared["run_id"] == "rollout"
    assert compared["metrics"]["teacher_forced_comparison_run_id"] == "teacher"
    assert compared["metrics"]["teacher_forced_model_name"] == "local-9b"
    assert compared["metrics"]["teacher_forced_input_sha256"] == "abc123"
    assert compared["metrics"]["teacher_forced_value_execution_accuracy"] == 0.5
    assert compared["metrics"]["teacher_forced_strict_execution_accuracy"] == 0.25
    assert compared["metrics"]["rollout_value_delta_vs_teacher_forced"] == pytest.approx(0.25)
    assert compared["metrics"]["rollout_strict_delta_vs_teacher_forced"] == pytest.approx(0.25)
    assert compared["metrics"]["teacher_forced_comparable_row_count"] == 2


def test_compare_rollout_manifests_requires_output_rows() -> None:
    with pytest.raises(ValueError, match="output rows"):
        compare_rollout_manifests(
            rollout_manifest=_rollout_manifest(),
            teacher_forced_manifest=_teacher_manifest(),
        )


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


def test_compare_rollout_manifests_rejects_row_count_mismatch() -> None:
    rollout = _rollout_manifest()
    teacher = _teacher_manifest()
    teacher["row_count"] = 100

    with pytest.raises(ValueError, match="row_count"):
        compare_rollout_manifests(
            rollout_manifest=rollout,
            teacher_forced_manifest=teacher,
        )


def test_compare_rollout_manifests_rejects_oracle_rollout() -> None:
    rollout = _rollout_manifest()
    rollout["oracle_allowed"] = True

    with pytest.raises(ValueError, match="non-oracle"):
        compare_rollout_manifests(
            rollout_manifest=rollout,
            teacher_forced_manifest=_teacher_manifest(),
        )


def test_compare_rollout_manifests_rejects_oracle_teacher_forced_manifest() -> None:
    teacher = _teacher_manifest()
    teacher["oracle_allowed"] = True

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
    rollout_output = tmp_path / "results" / "rollout.jsonl"
    teacher_output = tmp_path / "results" / "teacher.jsonl"
    rollout_output.parent.mkdir(parents=True)
    rollout_output.write_text(
        "\n".join(
            json.dumps(row) for row in _rows("model_generated_sql_rollout")
        )
        + "\n"
    )
    teacher_output.write_text(
        "\n".join(json.dumps(row) for row in _rows("gold_sql_teacher_forced")) + "\n"
    )
    rollout_manifest = _rollout_manifest()
    teacher_manifest = _teacher_manifest()
    rollout_manifest["output_path"] = str(rollout_output.relative_to(tmp_path))
    teacher_manifest["output_path"] = str(teacher_output.relative_to(tmp_path))
    rollout_path.write_text(json.dumps(rollout_manifest) + "\n")
    teacher_path.write_text(json.dumps(teacher_manifest) + "\n")

    compared = compare_rollout_manifest_files(
        rollout_manifest_path=rollout_path,
        teacher_forced_manifest_path=teacher_path,
        output_path=output_path,
        repo_root=tmp_path,
    )

    written = json.loads(output_path.read_text())
    assert written == compared
    assert written["metrics"]["teacher_forced_comparison_run_id"] == "teacher"
    assert written["metrics"]["teacher_forced_comparable_row_count"] == 2


def test_compare_rollout_manifest_files_rejects_row_identity_mismatch(tmp_path) -> None:
    rollout_path = tmp_path / "rollout.manifest.json"
    teacher_path = tmp_path / "teacher.manifest.json"
    output_path = tmp_path / "comparison.manifest.json"
    rollout_output = tmp_path / "results" / "rollout.jsonl"
    teacher_output = tmp_path / "results" / "teacher.jsonl"
    rollout_output.parent.mkdir(parents=True)
    rollout_output.write_text(
        "\n".join(
            json.dumps(row) for row in _rows("model_generated_sql_rollout")
        )
        + "\n"
    )
    teacher_rows = _rows("gold_sql_teacher_forced")
    teacher_rows[1]["reference_sql"] = "SELECT MAX(amount) FROM orders;"
    teacher_output.write_text("\n".join(json.dumps(row) for row in teacher_rows) + "\n")
    rollout_manifest = _rollout_manifest()
    teacher_manifest = _teacher_manifest()
    rollout_manifest["output_path"] = str(rollout_output.relative_to(tmp_path))
    teacher_manifest["output_path"] = str(teacher_output.relative_to(tmp_path))
    rollout_path.write_text(json.dumps(rollout_manifest) + "\n")
    teacher_path.write_text(json.dumps(teacher_manifest) + "\n")

    with pytest.raises(ValueError, match="row identity"):
        compare_rollout_manifest_files(
            rollout_manifest_path=rollout_path,
            teacher_forced_manifest_path=teacher_path,
            output_path=output_path,
            repo_root=tmp_path,
        )
