from __future__ import annotations

import json

from data.value_schema_repair_rollout_inputs import (
    build_value_schema_repair_rollout_inputs,
    summarize_value_schema_repair_rollout_inputs,
    write_value_schema_repair_rollout_input_artifacts,
)


def test_value_schema_repair_rollout_input_exposes_non_oracle_repair_context() -> None:
    [row] = build_value_schema_repair_rollout_inputs()
    prompt = "\n".join(message["content"] for message in row["messages"][:-1])

    assert row["artifact_type"] == "value_schema_repair_rollout_inputs"
    assert row["repair_reference_sql_visible_to_model"] is False
    assert row["seed_failure_sql_visible_to_model"] is True
    assert row["value_index_source"] == "database_contents"
    assert row["schema_guardrails_visible_to_model"] is True
    assert "France" in prompt
    assert '"storage_value": "FR"' in prompt
    assert "customers.customer_id" in prompt
    assert row["repair_reference_sql"] not in prompt
    assert "future" not in row["leakage_policy"]


def test_value_schema_repair_summary_records_artifact_boundaries() -> None:
    summary = summarize_value_schema_repair_rollout_inputs(
        build_value_schema_repair_rollout_inputs()
    )

    assert summary["rollout_input_count"] == 1
    assert summary["failure_mode_counts"]["value_normalization"] == 1
    assert summary["failure_mode_counts"]["recovery"] == 1
    assert summary["value_index_source"] == "database_contents"
    assert summary["schema_validation_guardrails_visible_to_model"] is True


def test_write_value_schema_repair_rollout_input_artifacts(tmp_path) -> None:
    output_path = tmp_path / "value_schema_repair_rollout_inputs.jsonl"
    summary_path = tmp_path / "value_schema_repair_rollout_inputs_summary.json"
    manifest_path = tmp_path / "value_schema_repair_rollout_inputs.manifest.json"

    manifest = write_value_schema_repair_rollout_input_artifacts(
        output_path=output_path,
        summary_path=summary_path,
        manifest_path=manifest_path,
    )

    rows = [json.loads(line) for line in output_path.read_text().splitlines()]
    summary = json.loads(summary_path.read_text())
    written_manifest = json.loads(manifest_path.read_text())

    assert len(rows) == 1
    assert summary["rollout_input_count"] == 1
    assert manifest == written_manifest
    assert manifest["artifact_type"] == "value_schema_repair_rollout_inputs"
    assert manifest["output_sha256"]
    assert manifest["summary_sha256"]
