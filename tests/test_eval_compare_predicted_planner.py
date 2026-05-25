from __future__ import annotations

import json

import pytest

from eval.compare_predicted_planner import (
    compare_predicted_planner_manifest_files,
    compare_predicted_planner_manifests,
)


def _direct_manifest() -> dict:
    return {
        "schema_version": 1,
        "run_id": "direct",
        "benchmark": "prepared",
        "model_name": "local-9b",
        "endpoint": "local",
        "evaluation_mode": "non_oracle_generation",
        "oracle_allowed": False,
        "input_path": "data/direct.jsonl",
        "input_sha256": "direct-input",
        "output_path": "results/direct.jsonl",
        "output_sha256": "direct-output",
        "row_count": 2,
        "metrics": {
            "value_execution_accuracy": 0.5,
            "strict_execution_accuracy": 0.25,
            "dialog_count": 1,
        },
        "command": ["run-direct"],
    }


def _predicted_manifest() -> dict:
    return {
        "schema_version": 1,
        "run_id": "predicted",
        "benchmark": "prepared",
        "model_name": "local-9b",
        "endpoint": "local",
        "evaluation_mode": "predicted_planner",
        "oracle_allowed": False,
        "input_path": "data/predicted.jsonl",
        "input_sha256": "predicted-input",
        "output_path": "results/predicted.jsonl",
        "output_sha256": "predicted-output",
        "row_count": 2,
        "metrics": {
            "value_execution_accuracy": 0.75,
            "strict_execution_accuracy": 0.5,
            "dialog_count": 1,
        },
        "command": ["run-predicted"],
    }


def _rows(mode: str) -> list[dict]:
    return [
        {
            "id": "dialog-a:0",
            "dialog_id": "dialog-a",
            "turn_index": 0,
            "database_id": "store",
            "reference_sql": "SELECT COUNT(*) FROM orders;",
            "evaluation_mode": mode,
        },
        {
            "id": "dialog-a:1",
            "dialog_id": "dialog-a",
            "turn_index": 1,
            "database_id": "store",
            "reference_sql": "SELECT SUM(amount) FROM orders;",
            "evaluation_mode": mode,
        },
    ]


def test_compare_predicted_planner_manifests_adds_direct_sql_delta() -> None:
    compared = compare_predicted_planner_manifests(
        predicted_manifest=_predicted_manifest(),
        direct_manifest=_direct_manifest(),
        predicted_rows=_rows("predicted_planner"),
        direct_rows=_rows("non_oracle_generation"),
    )

    assert compared["run_id"] == "predicted"
    assert compared["metrics"]["direct_sql_comparison_run_id"] == "direct"
    assert compared["metrics"]["direct_sql_model_name"] == "local-9b"
    assert compared["metrics"]["direct_sql_value_execution_accuracy"] == 0.5
    assert compared["metrics"]["direct_sql_strict_execution_accuracy"] == 0.25
    assert compared["metrics"]["predicted_planner_value_delta_vs_direct_sql"] == pytest.approx(
        0.25
    )
    assert compared["metrics"]["predicted_planner_strict_delta_vs_direct_sql"] == pytest.approx(
        0.25
    )


def test_compare_predicted_planner_manifests_rejects_model_mismatch() -> None:
    direct = _direct_manifest()
    direct["model_name"] = "other-model"

    with pytest.raises(ValueError, match="same model"):
        compare_predicted_planner_manifests(
            predicted_manifest=_predicted_manifest(),
            direct_manifest=direct,
            predicted_rows=_rows("predicted_planner"),
            direct_rows=_rows("non_oracle_generation"),
        )


def test_compare_predicted_planner_manifests_rejects_oracle_predicted_manifest() -> None:
    predicted = _predicted_manifest()
    predicted["oracle_allowed"] = True
    predicted["evaluation_mode"] = "oracle_planner_diagnostic"

    with pytest.raises(ValueError, match="non-oracle"):
        compare_predicted_planner_manifests(
            predicted_manifest=predicted,
            direct_manifest=_direct_manifest(),
            predicted_rows=_rows("predicted_planner"),
            direct_rows=_rows("non_oracle_generation"),
        )


def test_compare_predicted_planner_manifests_rejects_oracle_output_rows() -> None:
    predicted_rows = _rows("predicted_planner")
    predicted_rows[0]["uses_oracle_planning_hints"] = True

    with pytest.raises(ValueError, match="oracle"):
        compare_predicted_planner_manifests(
            predicted_manifest=_predicted_manifest(),
            direct_manifest=_direct_manifest(),
            predicted_rows=predicted_rows,
            direct_rows=_rows("non_oracle_generation"),
        )

    direct_rows = _rows("non_oracle_generation")
    direct_rows[0]["semantic_context_pruned_by_oracle_labels"] = True

    with pytest.raises(ValueError, match="oracle"):
        compare_predicted_planner_manifests(
            predicted_manifest=_predicted_manifest(),
            direct_manifest=_direct_manifest(),
            predicted_rows=_rows("predicted_planner"),
            direct_rows=direct_rows,
        )


def test_compare_predicted_planner_manifests_rejects_wrong_direct_mode() -> None:
    direct = _direct_manifest()
    direct["evaluation_mode"] = "predicted_planner"

    with pytest.raises(ValueError, match="non_oracle_generation"):
        compare_predicted_planner_manifests(
            predicted_manifest=_predicted_manifest(),
            direct_manifest=direct,
            predicted_rows=_rows("predicted_planner"),
            direct_rows=_rows("non_oracle_generation"),
        )


def test_compare_predicted_planner_manifests_rejects_mismatched_rows() -> None:
    direct_rows = _rows("non_oracle_generation")
    direct_rows[1]["reference_sql"] = "SELECT AVG(amount) FROM orders;"

    with pytest.raises(ValueError, match="row identity"):
        compare_predicted_planner_manifests(
            predicted_manifest=_predicted_manifest(),
            direct_manifest=_direct_manifest(),
            predicted_rows=_rows("predicted_planner"),
            direct_rows=direct_rows,
        )


def test_compare_predicted_planner_manifests_requires_database_id_identity() -> None:
    predicted_rows = _rows("predicted_planner")
    predicted_rows[0].pop("database_id")

    with pytest.raises(ValueError, match="database_id"):
        compare_predicted_planner_manifests(
            predicted_manifest=_predicted_manifest(),
            direct_manifest=_direct_manifest(),
            predicted_rows=predicted_rows,
            direct_rows=_rows("non_oracle_generation"),
        )


def test_compare_predicted_planner_manifests_checks_manifest_row_count() -> None:
    predicted = _predicted_manifest()
    predicted["row_count"] = 3

    with pytest.raises(ValueError, match="row_count"):
        compare_predicted_planner_manifests(
            predicted_manifest=predicted,
            direct_manifest=_direct_manifest(),
            predicted_rows=_rows("predicted_planner"),
            direct_rows=_rows("non_oracle_generation"),
        )


def test_compare_predicted_planner_manifests_requires_predicted_output_mode() -> None:
    with pytest.raises(ValueError, match="predicted_planner output rows"):
        compare_predicted_planner_manifests(
            predicted_manifest=_predicted_manifest(),
            direct_manifest=_direct_manifest(),
            predicted_rows=_rows("non_oracle_generation"),
            direct_rows=_rows("non_oracle_generation"),
        )


def test_compare_predicted_planner_manifest_files_writes_augmented_manifest(tmp_path) -> None:
    predicted_output = tmp_path / "results" / "predicted.jsonl"
    direct_output = tmp_path / "results" / "direct.jsonl"
    predicted_output.parent.mkdir(parents=True)
    predicted_output.write_text("\n".join(json.dumps(row) for row in _rows("predicted_planner")) + "\n")
    direct_output.write_text(
        "\n".join(json.dumps(row) for row in _rows("non_oracle_generation")) + "\n"
    )
    predicted_manifest = _predicted_manifest()
    direct_manifest = _direct_manifest()
    predicted_manifest["output_path"] = str(predicted_output.relative_to(tmp_path))
    direct_manifest["output_path"] = str(direct_output.relative_to(tmp_path))
    predicted_manifest_path = tmp_path / "predicted.manifest.json"
    direct_manifest_path = tmp_path / "direct.manifest.json"
    output_path = tmp_path / "predicted.compared.manifest.json"
    predicted_manifest_path.write_text(json.dumps(predicted_manifest) + "\n")
    direct_manifest_path.write_text(json.dumps(direct_manifest) + "\n")

    compared = compare_predicted_planner_manifest_files(
        predicted_manifest_path=predicted_manifest_path,
        direct_manifest_path=direct_manifest_path,
        output_path=output_path,
        repo_root=tmp_path,
    )

    assert json.loads(output_path.read_text()) == compared
    assert compared["metrics"]["direct_sql_comparison_run_id"] == "direct"
