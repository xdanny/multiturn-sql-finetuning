from __future__ import annotations

import json

import pytest

from eval.compare_semantic_layer_direct_sql import (
    compare_semantic_layer_direct_sql_manifest_files,
    compare_semantic_layer_direct_sql_manifests,
)


def _semantic_manifest() -> dict:
    return {
        "schema_version": 1,
        "run_id": "semantic_layer",
        "benchmark": "synthetic_semantic_layer",
        "model_name": "semantic-model",
        "endpoint": "offline",
        "evaluation_mode": "non_oracle_generation",
        "oracle_allowed": False,
        "input_path": "data/semantic.jsonl",
        "input_sha256": "semantic-input",
        "output_path": "results/semantic.jsonl",
        "output_sha256": "semantic-output",
        "row_count": 2,
        "metrics": {
            "value_execution_accuracy": 0.75,
            "strict_execution_accuracy": 0.5,
            "execution_evaluated_rows": 2,
        },
        "command": ["run-semantic"],
    }


def _direct_manifest() -> dict:
    return {
        "schema_version": 1,
        "run_id": "direct_sql",
        "benchmark": "synthetic_semantic_layer_direct_sql",
        "model_name": "direct-model",
        "endpoint": "offline",
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
            "execution_evaluated_rows": 2,
        },
        "command": ["run-direct"],
    }


def _semantic_rows() -> list[dict]:
    return [
        {
            "id": "semantic-1",
            "fixture_id": "value_normalization_france",
            "evaluation_mode": "non_oracle_generation",
            "reference_sql": "SELECT SUM(amount) FROM orders;",
            "database_path": "store.sqlite",
            "value_execution_score": 1.0,
            "strict_execution_score": 1.0,
            "semantic_model_oracle_derived": False,
        },
        {
            "id": "semantic-2",
            "fixture_id": "grain_fanout_bridge",
            "evaluation_mode": "non_oracle_generation",
            "reference_sql": "SELECT COUNT(*) FROM orders;",
            "database_path": "store.sqlite",
            "value_execution_score": 0.5,
            "strict_execution_score": 0.0,
            "semantic_model_oracle_derived": False,
        },
    ]


def _direct_rows() -> list[dict]:
    return [
        {
            "id": "semantic-1",
            "fixture_id": "value_normalization_france",
            "evaluation_mode": "non_oracle_generation",
            "reference_sql": "SELECT SUM(amount) FROM orders;",
            "database_path": "store.sqlite",
            "value_execution_score": 1.0,
            "strict_execution_score": 1.0,
        },
        {
            "id": "semantic-2",
            "fixture_id": "grain_fanout_bridge",
            "evaluation_mode": "non_oracle_generation",
            "reference_sql": "SELECT COUNT(*) FROM orders;",
            "database_path": "store.sqlite",
            "value_execution_score": 0.0,
            "strict_execution_score": 0.0,
        },
    ]


def test_compare_semantic_layer_direct_sql_adds_delta() -> None:
    compared = compare_semantic_layer_direct_sql_manifests(
        semantic_manifest=_semantic_manifest(),
        direct_manifest=_direct_manifest(),
        semantic_rows=_semantic_rows(),
        direct_rows=_direct_rows(),
    )

    metrics = compared["metrics"]
    assert compared["run_id"] == "semantic_layer"
    assert metrics["direct_sql_comparison_run_id"] == "direct_sql"
    assert metrics["direct_sql_model_name"] == "direct-model"
    assert metrics["semantic_layer_value_delta_vs_direct_sql"] == pytest.approx(0.25)
    assert metrics["semantic_layer_strict_delta_vs_direct_sql"] == pytest.approx(0.25)
    assert metrics["semantic_layer_comparable_row_count"] == 2


def test_compare_semantic_layer_direct_sql_rejects_oracle_semantic_rows() -> None:
    semantic_rows = _semantic_rows()
    semantic_rows[0]["messages"] = [
        {"role": "user", "content": "SQL planning hints:\nRelevant tables: orders"}
    ]

    with pytest.raises(ValueError, match="oracle"):
        compare_semantic_layer_direct_sql_manifests(
            semantic_manifest=_semantic_manifest(),
            direct_manifest=_direct_manifest(),
            semantic_rows=semantic_rows,
            direct_rows=_direct_rows(),
        )


def test_compare_semantic_layer_direct_sql_rejects_wrong_benchmark() -> None:
    semantic_manifest = _semantic_manifest()
    semantic_manifest["benchmark"] = "prepared"

    with pytest.raises(ValueError, match="synthetic_semantic_layer"):
        compare_semantic_layer_direct_sql_manifests(
            semantic_manifest=semantic_manifest,
            direct_manifest=_direct_manifest(),
            semantic_rows=_semantic_rows(),
            direct_rows=_direct_rows(),
        )


def test_compare_semantic_layer_direct_sql_rejects_row_identity_mismatch() -> None:
    direct_rows = _direct_rows()
    direct_rows[1]["reference_sql"] = "SELECT AVG(amount) FROM orders;"

    with pytest.raises(ValueError, match="row identity"):
        compare_semantic_layer_direct_sql_manifests(
            semantic_manifest=_semantic_manifest(),
            direct_manifest=_direct_manifest(),
            semantic_rows=_semantic_rows(),
            direct_rows=direct_rows,
        )


def test_compare_semantic_layer_direct_sql_manifest_files_writes_augmented_manifest(
    tmp_path,
) -> None:
    semantic_output = tmp_path / "results" / "semantic.jsonl"
    direct_output = tmp_path / "results" / "direct.jsonl"
    semantic_output.parent.mkdir(parents=True)
    semantic_output.write_text("\n".join(json.dumps(row) for row in _semantic_rows()) + "\n")
    direct_output.write_text("\n".join(json.dumps(row) for row in _direct_rows()) + "\n")
    semantic_manifest = _semantic_manifest()
    direct_manifest = _direct_manifest()
    semantic_manifest["output_path"] = str(semantic_output.relative_to(tmp_path))
    direct_manifest["output_path"] = str(direct_output.relative_to(tmp_path))
    semantic_manifest_path = tmp_path / "semantic.manifest.json"
    direct_manifest_path = tmp_path / "direct.manifest.json"
    output_path = tmp_path / "semantic.compared.manifest.json"
    semantic_manifest_path.write_text(json.dumps(semantic_manifest) + "\n")
    direct_manifest_path.write_text(json.dumps(direct_manifest) + "\n")

    compared = compare_semantic_layer_direct_sql_manifest_files(
        semantic_manifest_path=semantic_manifest_path,
        direct_manifest_path=direct_manifest_path,
        output_path=output_path,
        repo_root=tmp_path,
    )

    assert compared["metrics"]["semantic_layer_value_delta_vs_direct_sql"] == pytest.approx(0.25)
    assert json.loads(output_path.read_text())["metrics"]["semantic_layer_comparable_row_count"] == 2
