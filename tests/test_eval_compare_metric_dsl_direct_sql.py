from __future__ import annotations

import json

import pytest

from eval.compare_metric_dsl_direct_sql import (
    compare_metric_dsl_direct_sql_manifest_files,
    compare_metric_dsl_direct_sql_manifests,
)


def _metric_manifest() -> dict:
    return {
        "schema_version": 1,
        "run_id": "metric_dsl",
        "benchmark": "metric_dsl",
        "model_name": "metric-dsl-model",
        "endpoint": "offline",
        "evaluation_mode": "metric_dsl",
        "oracle_allowed": False,
        "input_path": "data/metric_dsl.jsonl",
        "input_sha256": "metric-input",
        "output_path": "results/metric_dsl.jsonl",
        "output_sha256": "metric-output",
        "row_count": 2,
        "metrics": {
            "metric_dsl_parse_rate": 1.0,
            "metric_dsl_compile_rate": 1.0,
            "compiled_sql_execution_evaluated_rows": 2,
            "measure_preservation": 1.0,
            "value_execution_accuracy": 0.75,
            "strict_execution_accuracy": 0.5,
        },
        "command": ["run-metric-dsl"],
    }


def _direct_manifest() -> dict:
    return {
        "schema_version": 1,
        "run_id": "direct_sql",
        "benchmark": "metric_dsl_direct_sql",
        "model_name": "direct-sql-model",
        "endpoint": "local",
        "evaluation_mode": "non_oracle_generation",
        "oracle_allowed": False,
        "input_path": "data/direct_sql.jsonl",
        "input_sha256": "direct-input",
        "output_path": "results/direct_sql.jsonl",
        "output_sha256": "direct-output",
        "row_count": 2,
        "metrics": {
            "value_execution_accuracy": 0.5,
            "strict_execution_accuracy": 0.25,
        },
        "command": ["run-direct-sql"],
    }


def _metric_rows() -> list[dict]:
    return [
        {
            "id": "metric-1",
            "evaluation_mode": "metric_dsl",
            "reference_sql": "SELECT SUM(amount) FROM orders;",
            "database_path": "store.sqlite",
            "metric_dsl_parse_success": True,
            "compile_success": True,
            "sql_execution_attempted": True,
            "value_execution_score": 1.0,
        },
        {
            "id": "metric-2",
            "evaluation_mode": "metric_dsl",
            "reference_sql": "SELECT COUNT(*) FROM orders;",
            "database_path": "store.sqlite",
            "metric_dsl_parse_success": True,
            "compile_success": True,
            "sql_execution_attempted": True,
            "value_execution_score": 0.5,
        },
    ]


def _direct_rows() -> list[dict]:
    return [
        {
            "id": "metric-1",
            "evaluation_mode": "non_oracle_generation",
            "reference_sql": "SELECT SUM(amount) FROM orders;",
            "database_path": "store.sqlite",
            "value_execution_score": 1.0,
            "strict_execution_score": 1.0,
        },
        {
            "id": "metric-2",
            "evaluation_mode": "non_oracle_generation",
            "reference_sql": "SELECT COUNT(*) FROM orders;",
            "database_path": "store.sqlite",
            "value_execution_score": 0.0,
            "strict_execution_score": 0.0,
        },
    ]


def test_compare_metric_dsl_direct_sql_allows_different_models_and_adds_delta() -> None:
    compared = compare_metric_dsl_direct_sql_manifests(
        metric_dsl_manifest=_metric_manifest(),
        direct_sql_manifest=_direct_manifest(),
        metric_dsl_rows=_metric_rows(),
        direct_sql_rows=_direct_rows(),
    )

    metrics = compared["metrics"]
    assert compared["run_id"] == "metric_dsl"
    assert metrics["direct_sql_comparison_run_id"] == "direct_sql"
    assert metrics["direct_sql_model_name"] == "direct-sql-model"
    assert metrics["direct_sql_value_execution_accuracy"] == 0.5
    assert metrics["metric_dsl_value_delta_vs_direct_sql"] == pytest.approx(0.25)
    assert metrics["metric_dsl_strict_delta_vs_direct_sql"] == pytest.approx(0.25)
    assert metrics["metric_dsl_measure_preservation"] == 1.0
    assert metrics["metric_dsl_comparable_row_count"] == 2


def test_compare_metric_dsl_direct_sql_rejects_oracle_metric_manifest() -> None:
    metric_manifest = _metric_manifest()
    metric_manifest["oracle_allowed"] = True

    with pytest.raises(ValueError, match="non-oracle"):
        compare_metric_dsl_direct_sql_manifests(
            metric_dsl_manifest=metric_manifest,
            direct_sql_manifest=_direct_manifest(),
            metric_dsl_rows=_metric_rows(),
            direct_sql_rows=_direct_rows(),
        )


def test_compare_metric_dsl_direct_sql_rejects_non_metric_manifest() -> None:
    metric_manifest = _metric_manifest()
    metric_manifest["evaluation_mode"] = "non_oracle_generation"

    with pytest.raises(ValueError, match="metric_dsl"):
        compare_metric_dsl_direct_sql_manifests(
            metric_dsl_manifest=metric_manifest,
            direct_sql_manifest=_direct_manifest(),
            metric_dsl_rows=_metric_rows(),
            direct_sql_rows=_direct_rows(),
        )


def test_compare_metric_dsl_direct_sql_rejects_unrelated_direct_benchmark() -> None:
    direct_manifest = _direct_manifest()
    direct_manifest["benchmark"] = "prepared"

    with pytest.raises(ValueError, match="metric_dsl_direct_sql"):
        compare_metric_dsl_direct_sql_manifests(
            metric_dsl_manifest=_metric_manifest(),
            direct_sql_manifest=direct_manifest,
            metric_dsl_rows=_metric_rows(),
            direct_sql_rows=_direct_rows(),
        )


def test_compare_metric_dsl_direct_sql_requires_full_database_execution() -> None:
    metric_manifest = _metric_manifest()
    metric_manifest["metrics"]["compiled_sql_execution_evaluated_rows"] = 1

    with pytest.raises(ValueError, match="database-backed"):
        compare_metric_dsl_direct_sql_manifests(
            metric_dsl_manifest=metric_manifest,
            direct_sql_manifest=_direct_manifest(),
            metric_dsl_rows=_metric_rows(),
            direct_sql_rows=_direct_rows(),
        )


def test_compare_metric_dsl_direct_sql_rejects_metric_rows_with_wrong_mode() -> None:
    metric_rows = _metric_rows()
    metric_rows[0]["evaluation_mode"] = "non_oracle_generation"

    with pytest.raises(ValueError, match="metric_dsl mode"):
        compare_metric_dsl_direct_sql_manifests(
            metric_dsl_manifest=_metric_manifest(),
            direct_sql_manifest=_direct_manifest(),
            metric_dsl_rows=metric_rows,
            direct_sql_rows=_direct_rows(),
        )


def test_compare_metric_dsl_direct_sql_requires_direct_execution_scores() -> None:
    direct_rows = _direct_rows()
    direct_rows[0].pop("strict_execution_score")

    with pytest.raises(ValueError, match="direct SQL output rows must include execution scores"):
        compare_metric_dsl_direct_sql_manifests(
            metric_dsl_manifest=_metric_manifest(),
            direct_sql_manifest=_direct_manifest(),
            metric_dsl_rows=_metric_rows(),
            direct_sql_rows=direct_rows,
        )


def test_compare_metric_dsl_direct_sql_rejects_oracle_messages_in_direct_rows() -> None:
    direct_rows = _direct_rows()
    direct_rows[0]["messages"] = [
        {"role": "user", "content": "SQL planning hints:\nRelevant tables: orders"}
    ]

    with pytest.raises(ValueError, match="oracle"):
        compare_metric_dsl_direct_sql_manifests(
            metric_dsl_manifest=_metric_manifest(),
            direct_sql_manifest=_direct_manifest(),
            metric_dsl_rows=_metric_rows(),
            direct_sql_rows=direct_rows,
        )


def test_compare_metric_dsl_direct_sql_rejects_row_identity_mismatch() -> None:
    direct_rows = _direct_rows()
    direct_rows[1]["reference_sql"] = "SELECT AVG(amount) FROM orders;"

    with pytest.raises(ValueError, match="row identity"):
        compare_metric_dsl_direct_sql_manifests(
            metric_dsl_manifest=_metric_manifest(),
            direct_sql_manifest=_direct_manifest(),
            metric_dsl_rows=_metric_rows(),
            direct_sql_rows=direct_rows,
        )


def test_compare_metric_dsl_direct_sql_rejects_oracle_output_rows() -> None:
    metric_rows = _metric_rows()
    metric_rows[0]["semantic_model_oracle_derived"] = True

    with pytest.raises(ValueError, match="oracle"):
        compare_metric_dsl_direct_sql_manifests(
            metric_dsl_manifest=_metric_manifest(),
            direct_sql_manifest=_direct_manifest(),
            metric_dsl_rows=metric_rows,
            direct_sql_rows=_direct_rows(),
        )


def test_compare_metric_dsl_direct_sql_manifest_files_writes_augmented_manifest(tmp_path) -> None:
    metric_output = tmp_path / "results" / "metric_dsl.jsonl"
    direct_output = tmp_path / "results" / "direct_sql.jsonl"
    metric_output.parent.mkdir(parents=True)
    metric_output.write_text("\n".join(json.dumps(row) for row in _metric_rows()) + "\n")
    direct_output.write_text("\n".join(json.dumps(row) for row in _direct_rows()) + "\n")
    metric_manifest = _metric_manifest()
    direct_manifest = _direct_manifest()
    metric_manifest["output_path"] = str(metric_output.relative_to(tmp_path))
    direct_manifest["output_path"] = str(direct_output.relative_to(tmp_path))
    metric_manifest_path = tmp_path / "metric.manifest.json"
    direct_manifest_path = tmp_path / "direct.manifest.json"
    output_path = tmp_path / "metric.compared.manifest.json"
    metric_manifest_path.write_text(json.dumps(metric_manifest) + "\n")
    direct_manifest_path.write_text(json.dumps(direct_manifest) + "\n")

    compared = compare_metric_dsl_direct_sql_manifest_files(
        metric_dsl_manifest_path=metric_manifest_path,
        direct_sql_manifest_path=direct_manifest_path,
        output_path=output_path,
        repo_root=tmp_path,
    )

    assert json.loads(output_path.read_text()) == compared
    assert compared["metrics"]["direct_sql_comparison_run_id"] == "direct_sql"
