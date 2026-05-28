from __future__ import annotations

import json

from data.behavior_recovery_rollout_inputs import (
    build_behavior_recovery_rollout_inputs,
    summarize_behavior_recovery_rollout_inputs,
    write_behavior_recovery_rollout_input_artifacts,
)


def test_behavior_recovery_rollout_inputs_seed_generated_failure_dialog() -> None:
    [record] = build_behavior_recovery_rollout_inputs()

    assert record["id"] == "recovery_empty_result"
    assert record["history_policy"] == "seeded_generated_failure_then_rollout"
    assert record["evaluation_mode"] == "non_oracle_generation"
    assert record["uses_oracle_planning_hints"] is False
    assert record["semantic_context_pruned_by_oracle_labels"] is False
    assert record["seeded_failure_turn_index"] == 0
    assert record["messages"][1]["role"] == "user"
    assert record["messages"][2]["role"] == "assistant"
    assert "France" in record["messages"][2]["content"]
    assert "Observed previous rows: []" in record["messages"][3]["content"]
    assert record["messages"][4]["content"] == record["repair_reference_sql"]
    assert record["repair_reference_sql_visible_to_model"] is False
    assert record["seed_failure_sql_visible_to_model"] is True
    assert record["messages"][4]["content"] not in record["messages"][3]["content"]


def test_behavior_recovery_rollout_summary_names_rollout_gate() -> None:
    rows = build_behavior_recovery_rollout_inputs()
    summary = summarize_behavior_recovery_rollout_inputs(rows)

    assert summary["rollout_input_count"] == 1
    assert summary["fixture_ids"] == ["recovery_empty_result"]
    assert summary["history_policy_counts"] == {"seeded_generated_failure_then_rollout": 1}
    assert "eval.rollout_eval" in summary["evaluation_gate"]


def test_write_behavior_recovery_rollout_input_artifacts(tmp_path) -> None:
    output_path = tmp_path / "behavior_recovery_rollout_inputs.jsonl"
    summary_path = tmp_path / "behavior_recovery_rollout_inputs_summary.json"
    manifest_path = tmp_path / "behavior_recovery_rollout_inputs.manifest.json"

    manifest = write_behavior_recovery_rollout_input_artifacts(
        output_path=output_path,
        summary_path=summary_path,
        manifest_path=manifest_path,
    )

    records = [json.loads(line) for line in output_path.read_text().splitlines()]
    summary = json.loads(summary_path.read_text())

    assert len(records) == 1
    assert summary["fixture_ids"] == ["recovery_empty_result"]
    assert manifest["rollout_input_count"] == 1
    assert manifest["output_sha256"]
    assert manifest["summary_sha256"]
    assert json.loads(manifest_path.read_text()) == manifest
