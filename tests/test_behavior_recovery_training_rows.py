from __future__ import annotations

import json

from data.behavior_recovery_training_rows import (
    build_behavior_recovery_finetuning_rows,
    summarize_behavior_recovery_finetuning_rows,
    write_behavior_recovery_finetuning_artifacts,
)


def test_behavior_recovery_rows_include_generated_history_without_reference_sql_leak() -> None:
    recovery_rows, control_rows = build_behavior_recovery_finetuning_rows()

    assert [row["fixture_id"] for row in recovery_rows] == ["recovery_empty_result"]
    assert [row["fixture_id"] for row in control_rows] == ["recovery_empty_result"]
    assert recovery_rows[0]["training_target"] == "behavior_recovery"
    assert control_rows[0]["training_target"] == "direct_sql_control"
    assert recovery_rows[0]["history_policy"] == "generated_history_trace"

    prompt = recovery_rows[0]["messages"][1]["content"]
    assert "Previous generated SQL:" in prompt
    assert "Observed previous rows: []" in prompt
    assert "country_code = 'France'" in prompt
    assert "country_code = 'FR'" not in prompt
    assert "reference_sql" not in prompt
    assert "Alice" not in prompt


def test_write_behavior_recovery_finetuning_artifacts(tmp_path) -> None:
    recovery_output = tmp_path / "behavior_recovery_training_rows.jsonl"
    control_output = tmp_path / "behavior_recovery_direct_sql_training_rows.jsonl"
    summary_output = tmp_path / "behavior_recovery_training_rows_summary.json"
    manifest_output = tmp_path / "behavior_recovery_training_rows.manifest.json"

    manifest = write_behavior_recovery_finetuning_artifacts(
        recovery_output_path=recovery_output,
        control_output_path=control_output,
        summary_path=summary_output,
        manifest_path=manifest_output,
    )

    recovery_rows = [json.loads(line) for line in recovery_output.read_text().splitlines()]
    control_rows = [json.loads(line) for line in control_output.read_text().splitlines()]
    summary = summarize_behavior_recovery_finetuning_rows(recovery_rows, control_rows)

    assert manifest["behavior_recovery_row_count"] == 1
    assert manifest["direct_sql_control_row_count"] == 1
    assert manifest["evaluation_gate"] == "generated_history_rollout"
    assert manifest["recovery_output_sha256"]
    assert manifest["control_output_sha256"]
    assert json.loads(summary_output.read_text()) == summary
    assert json.loads(manifest_output.read_text()) == manifest
