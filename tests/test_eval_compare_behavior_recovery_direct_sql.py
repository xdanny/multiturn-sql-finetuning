from __future__ import annotations

import json

import pytest

from eval.compare_behavior_recovery_direct_sql import (
    compare_behavior_recovery_direct_sql_manifest_files,
    compare_behavior_recovery_direct_sql_manifests,
)


def _recovery_manifest() -> dict:
    return {
        "schema_version": 1,
        "run_id": "behavior_recovery",
        "benchmark": "behavior_recovery",
        "model_name": "recovery-model",
        "endpoint": "offline",
        "evaluation_mode": "non_oracle_generation",
        "oracle_allowed": False,
        "input_path": "data/recovery.jsonl",
        "input_sha256": "recovery-input",
        "output_path": "results/recovery.jsonl",
        "output_sha256": "recovery-output",
        "row_count": 1,
        "metrics": {
            "value_execution_accuracy": 1.0,
            "strict_execution_accuracy": 1.0,
            "recovery_evaluated_rows": 1,
            "recovery_success_rate": 1.0,
        },
        "command": ["run-recovery"],
    }


def _direct_manifest() -> dict:
    return {
        "schema_version": 1,
        "run_id": "direct_sql",
        "benchmark": "behavior_recovery_direct_sql",
        "model_name": "direct-model",
        "endpoint": "offline",
        "evaluation_mode": "non_oracle_generation",
        "oracle_allowed": False,
        "input_path": "data/direct.jsonl",
        "input_sha256": "direct-input",
        "output_path": "results/direct.jsonl",
        "output_sha256": "direct-output",
        "row_count": 1,
        "metrics": {
            "value_execution_accuracy": 0.0,
            "strict_execution_accuracy": 0.0,
            "recovery_evaluated_rows": 1,
            "recovery_success_rate": 0.0,
        },
        "command": ["run-direct"],
    }


def _rows(success: bool) -> list[dict]:
    return [
        {
            "id": "recovery-1",
            "fixture_id": "recovery_empty_result",
            "evaluation_mode": "non_oracle_generation",
            "reference_sql": (
                "SELECT customers.name, SUM(orders.amount) AS revenue "
                "FROM orders JOIN customers ON orders.customer_id = customers.id "
                "WHERE customers.country_code = 'FR' GROUP BY customers.name "
                "ORDER BY revenue DESC LIMIT 1"
            ),
            "database_path": "recovery.sqlite",
            "value_execution_score": 1.0 if success else 0.0,
            "strict_execution_score": 1.0 if success else 0.0,
            "recovery_required": True,
            "recovery_success": success,
        }
    ]


def test_compare_behavior_recovery_direct_sql_adds_recovery_delta() -> None:
    compared = compare_behavior_recovery_direct_sql_manifests(
        behavior_recovery_manifest=_recovery_manifest(),
        direct_sql_manifest=_direct_manifest(),
        behavior_recovery_rows=_rows(True),
        direct_sql_rows=_rows(False),
    )

    metrics = compared["metrics"]
    assert metrics["direct_sql_comparison_run_id"] == "direct_sql"
    assert metrics["behavior_recovery_value_delta_vs_direct_sql"] == pytest.approx(1.0)
    assert metrics["behavior_recovery_strict_delta_vs_direct_sql"] == pytest.approx(1.0)
    assert metrics["behavior_recovery_success_delta_vs_direct_sql"] == pytest.approx(1.0)
    assert metrics["behavior_recovery_comparable_row_count"] == 1


def test_compare_behavior_recovery_direct_sql_requires_recovery_metrics() -> None:
    recovery_manifest = _recovery_manifest()
    recovery_manifest["metrics"].pop("recovery_success_rate")

    with pytest.raises(ValueError, match="recovery_success_rate"):
        compare_behavior_recovery_direct_sql_manifests(
            behavior_recovery_manifest=recovery_manifest,
            direct_sql_manifest=_direct_manifest(),
            behavior_recovery_rows=_rows(True),
            direct_sql_rows=_rows(False),
        )


def test_compare_behavior_recovery_direct_sql_rejects_row_identity_mismatch() -> None:
    direct_rows = _rows(False)
    direct_rows[0]["reference_sql"] = "SELECT COUNT(*) FROM orders;"

    with pytest.raises(ValueError, match="row identity"):
        compare_behavior_recovery_direct_sql_manifests(
            behavior_recovery_manifest=_recovery_manifest(),
            direct_sql_manifest=_direct_manifest(),
            behavior_recovery_rows=_rows(True),
            direct_sql_rows=direct_rows,
        )


def test_compare_behavior_recovery_direct_sql_manifest_files_write_augmented_manifest(
    tmp_path,
) -> None:
    recovery_output = tmp_path / "results" / "recovery.jsonl"
    direct_output = tmp_path / "results" / "direct.jsonl"
    recovery_output.parent.mkdir(parents=True)
    recovery_output.write_text("\n".join(json.dumps(row) for row in _rows(True)) + "\n")
    direct_output.write_text("\n".join(json.dumps(row) for row in _rows(False)) + "\n")
    recovery_manifest = _recovery_manifest()
    direct_manifest = _direct_manifest()
    recovery_manifest["output_path"] = str(recovery_output.relative_to(tmp_path))
    direct_manifest["output_path"] = str(direct_output.relative_to(tmp_path))
    recovery_manifest_path = tmp_path / "recovery.manifest.json"
    direct_manifest_path = tmp_path / "direct.manifest.json"
    output_path = tmp_path / "recovery.compared.manifest.json"
    recovery_manifest_path.write_text(json.dumps(recovery_manifest) + "\n")
    direct_manifest_path.write_text(json.dumps(direct_manifest) + "\n")

    compared = compare_behavior_recovery_direct_sql_manifest_files(
        behavior_recovery_manifest_path=recovery_manifest_path,
        direct_sql_manifest_path=direct_manifest_path,
        output_path=output_path,
        repo_root=tmp_path,
    )

    written = json.loads(output_path.read_text())
    assert written == compared
    assert written["metrics"]["behavior_recovery_success_delta_vs_direct_sql"] == 1.0
