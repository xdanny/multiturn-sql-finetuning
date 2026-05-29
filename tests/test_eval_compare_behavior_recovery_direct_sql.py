from __future__ import annotations

import json

import pytest

from eval.compare_behavior_recovery_direct_sql import (
    compare_behavior_recovery_direct_sql_manifest_files,
    compare_behavior_recovery_direct_sql_manifests,
)


def _manifest(run_id: str, *, value: float, strict: float, model: str = "model") -> dict:
    return {
        "schema_version": 1,
        "run_id": run_id,
        "benchmark": "prepared_rollout",
        "model_name": model,
        "endpoint": "local",
        "evaluation_mode": "non_oracle_generation",
        "oracle_allowed": False,
        "prompt_variant": None,
        "input_path": "inputs.jsonl",
        "input_sha256": "input-sha",
        "output_path": f"{run_id}.jsonl",
        "output_sha256": f"{run_id}-output-sha",
        "row_count": 1,
        "metrics": {
            "value_execution_accuracy": value,
            "strict_execution_accuracy": strict,
            "history_policy": "model_generated_sql_rollout",
        },
        "command": ["run"],
    }


def _rows() -> list[dict]:
    return [
        {
            "id": "dialog:1",
            "dialog_id": "dialog",
            "turn_index": 1,
            "database_id": "synthetic_revenue_schema_v1",
            "reference_sql": "SELECT 1;",
            "history_policy": "model_generated_sql_rollout",
            "evaluation_mode": "non_oracle_generation",
            "value_execution_score": 1.0,
            "strict_execution_score": 1.0,
        }
    ]


def test_compare_behavior_recovery_direct_sql_adds_direct_control_delta() -> None:
    compared = compare_behavior_recovery_direct_sql_manifests(
        recovery_manifest=_manifest("recovery", value=1.0, strict=1.0, model="recovery"),
        direct_sql_manifest=_manifest("direct", value=0.0, strict=0.0, model="direct"),
        recovery_rows=_rows(),
        direct_sql_rows=_rows(),
    )

    metrics = compared["metrics"]
    assert metrics["direct_sql_comparison_run_id"] == "direct"
    assert metrics["direct_sql_model_name"] == "direct"
    assert metrics["direct_sql_value_execution_accuracy"] == 0.0
    assert metrics["direct_sql_comparable_row_count"] == 1
    assert metrics["behavior_recovery_value_delta_vs_direct_sql"] == 1.0
    assert metrics["rollout_value_accuracy_delta_vs_direct_sql"] == 1.0
    assert compared["command"][-2:] == ["# compared-with-direct-sql", "direct"]


def test_compare_behavior_recovery_direct_sql_rejects_identity_mismatch() -> None:
    direct_rows = _rows()
    direct_rows[0]["reference_sql"] = "SELECT 2;"

    with pytest.raises(ValueError, match="row identity"):
        compare_behavior_recovery_direct_sql_manifests(
            recovery_manifest=_manifest("recovery", value=1.0, strict=1.0),
            direct_sql_manifest=_manifest("direct", value=0.0, strict=0.0),
            recovery_rows=_rows(),
            direct_sql_rows=direct_rows,
        )


def test_compare_behavior_recovery_direct_sql_manifest_files_write_output(tmp_path) -> None:
    input_path = tmp_path / "inputs.jsonl"
    input_path.write_text("{}\n")
    recovery_output = tmp_path / "recovery.jsonl"
    direct_output = tmp_path / "direct.jsonl"
    recovery_output.write_text(json.dumps(_rows()[0]) + "\n")
    direct_output.write_text(json.dumps(_rows()[0]) + "\n")
    recovery_manifest = _manifest("recovery", value=1.0, strict=1.0, model="recovery")
    direct_manifest = _manifest("direct", value=0.0, strict=0.0, model="direct")
    recovery_manifest["input_path"] = str(input_path.relative_to(tmp_path))
    direct_manifest["input_path"] = str(input_path.relative_to(tmp_path))
    recovery_manifest["output_path"] = str(recovery_output.relative_to(tmp_path))
    direct_manifest["output_path"] = str(direct_output.relative_to(tmp_path))
    recovery_manifest_path = tmp_path / "recovery.manifest.json"
    direct_manifest_path = tmp_path / "direct.manifest.json"
    output_path = tmp_path / "compared.manifest.json"
    recovery_manifest_path.write_text(json.dumps(recovery_manifest) + "\n")
    direct_manifest_path.write_text(json.dumps(direct_manifest) + "\n")

    compared = compare_behavior_recovery_direct_sql_manifest_files(
        recovery_manifest_path=recovery_manifest_path,
        direct_sql_manifest_path=direct_manifest_path,
        output_path=output_path,
        repo_root=tmp_path,
    )

    assert json.loads(output_path.read_text()) == compared
    assert compared["metrics"]["behavior_recovery_value_delta_vs_direct_sql"] == 1.0
