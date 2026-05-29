from __future__ import annotations

import json

from data.value_choice_consistency_rollout_inputs import (
    build_value_choice_consistency_rollout_inputs,
    summarize_value_choice_consistency_rollout_inputs,
    write_value_choice_consistency_rollout_input_artifacts,
)


def test_value_choice_consistency_input_exposes_choice_without_reference_sql() -> None:
    [row] = build_value_choice_consistency_rollout_inputs()
    prompt = "\n".join(message["content"] for message in row["messages"][:-1])

    assert row["artifact_type"] == "value_choice_consistency_rollout_inputs"
    assert row["repair_reference_sql_visible_to_model"] is False
    assert row["matched_value_choices_visible_to_model"] is True
    assert row["expected_value_choice"]["storage_value"] == "FR"
    assert row["expected_value_choice"]["visible_to_model"] is False
    assert '"candidate_storage_values": [\n      "FR"\n    ]' in prompt
    assert '"distractor_storage_values": [\n      "US"\n    ]' in prompt
    assert row["repair_reference_sql"] not in prompt


def test_value_choice_consistency_summary_records_scoring_gate() -> None:
    summary = summarize_value_choice_consistency_rollout_inputs(
        build_value_choice_consistency_rollout_inputs()
    )

    assert summary["rollout_input_count"] == 1
    assert summary["value_choice_scoring_gate"] == "eval.value_choice_consistency"
    assert summary["matched_value_choice_source"] == "database_contents_plus_visible_user_text"


def test_write_value_choice_consistency_artifacts(tmp_path) -> None:
    output_path = tmp_path / "value_choice_consistency_rollout_inputs.jsonl"
    summary_path = tmp_path / "value_choice_consistency_rollout_inputs_summary.json"
    manifest_path = tmp_path / "value_choice_consistency_rollout_inputs.manifest.json"

    manifest = write_value_choice_consistency_rollout_input_artifacts(
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
    assert manifest["artifact_type"] == "value_choice_consistency_rollout_inputs"
    assert manifest["output_sha256"]
