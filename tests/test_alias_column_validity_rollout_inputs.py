from __future__ import annotations

import json

from data.alias_column_validity_rollout_inputs import (
    build_alias_column_validity_rollout_inputs,
    summarize_alias_column_validity_rollout_inputs,
    write_alias_column_validity_rollout_input_artifacts,
)


def test_alias_column_validity_input_exposes_constraints_without_reference_sql() -> None:
    [row] = build_alias_column_validity_rollout_inputs()
    prompt = "\n".join(message["content"] for message in row["messages"][:-1])

    assert row["artifact_type"] == "alias_column_validity_rollout_inputs"
    assert row["repair_reference_sql_visible_to_model"] is False
    assert row["column_role_constraints_visible_to_model"] is True
    assert "customers.customer_id" in prompt
    assert '"join_keys": [\n    "orders.customer_id = customers.id"\n  ]' in prompt
    assert row["expected_column_validity"]["visible_to_model"] is False
    assert row["repair_reference_sql"] not in prompt


def test_alias_column_validity_summary_records_scoring_gate() -> None:
    summary = summarize_alias_column_validity_rollout_inputs(
        build_alias_column_validity_rollout_inputs()
    )

    assert summary["rollout_input_count"] == 1
    assert summary["column_validity_scoring_command"] == "eval.alias_column_validity"
    assert summary["column_role_constraints_source"] == "schema_introspection_plus_visible_failure"


def test_write_alias_column_validity_artifacts(tmp_path) -> None:
    output_path = tmp_path / "alias_column_validity_rollout_inputs.jsonl"
    summary_path = tmp_path / "alias_column_validity_rollout_inputs_summary.json"
    manifest_path = tmp_path / "alias_column_validity_rollout_inputs.manifest.json"

    manifest = write_alias_column_validity_rollout_input_artifacts(
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
    assert manifest["artifact_type"] == "alias_column_validity_rollout_inputs"
    assert manifest["output_sha256"]
