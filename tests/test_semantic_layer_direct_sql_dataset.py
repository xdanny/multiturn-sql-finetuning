from __future__ import annotations

import json
from pathlib import Path

from data.semantic_layer_direct_sql_dataset import (
    DEFAULT_MANIFEST_OUTPUT,
    DEFAULT_OUTPUT,
    DEFAULT_SUMMARY_OUTPUT,
    build_semantic_layer_direct_sql_training_rows,
    summarize_semantic_layer_direct_sql_training_rows,
    write_semantic_layer_direct_sql_training_artifacts,
)
from data.synthetic_method_fixtures import build_synthetic_method_fixtures


def test_build_semantic_layer_direct_sql_training_rows_derives_same_row_control() -> None:
    rows = build_semantic_layer_direct_sql_training_rows(build_synthetic_method_fixtures())

    assert [row["fixture_id"] for row in rows] == [
        "value_normalization_france",
        "entity_resolution_followup",
        "grain_fanout_bridge",
        "measure_preservation_metric",
        "recovery_empty_result",
    ]
    assert all(row["training_target"] == "direct_sql_control" for row in rows)
    assert all(row["evaluation_mode"] == "non_oracle_generation" for row in rows)
    assert all(row["benchmark"] == "synthetic_semantic_layer_direct_sql" for row in rows)
    assert all(row["oracle_policy"] == "non_oracle_inputs_only" for row in rows)
    assert all(row["messages"][-1]["content"] == row["reference_sql"] for row in rows)

    first_user_prompt = rows[0]["messages"][1]["content"]
    assert "Respond with only SQL." in first_user_prompt
    assert "Semantic model:" not in first_user_prompt
    assert "Show revenue by country." in first_user_prompt
    assert "reference_sql" not in first_user_prompt
    assert "expected_rows" not in first_user_prompt
    assert "storage_value" not in first_user_prompt


def test_summarize_semantic_layer_direct_sql_training_rows_reports_fixture_coverage() -> None:
    summary = summarize_semantic_layer_direct_sql_training_rows(
        build_semantic_layer_direct_sql_training_rows(build_synthetic_method_fixtures())
    )

    assert summary["row_count"] == 5
    assert summary["training_target"] == "direct_sql_control"
    assert summary["benchmark"] == "synthetic_semantic_layer_direct_sql"
    assert summary["fixture_ids"] == [
        "value_normalization_france",
        "entity_resolution_followup",
        "grain_fanout_bridge",
        "measure_preservation_metric",
        "recovery_empty_result",
    ]
    assert summary["failure_mode_counts"]["recovery"] == 1
    assert summary["failure_mode_counts"]["measure_preservation"] == 1


def test_write_semantic_layer_direct_sql_training_artifacts_writes_jsonl_summary_and_manifest(
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "semantic_layer_direct_sql_train.jsonl"
    summary_path = tmp_path / "semantic_layer_direct_sql_train_summary.json"
    manifest_path = tmp_path / "semantic_layer_direct_sql_train.manifest.json"

    manifest = write_semantic_layer_direct_sql_training_artifacts(
        output_path=output_path,
        summary_path=summary_path,
        manifest_path=manifest_path,
        command=["uv", "run", "python", "-m", "data.semantic_layer_direct_sql_dataset"],
    )

    rows = [json.loads(line) for line in output_path.read_text().splitlines()]
    summary = json.loads(summary_path.read_text())
    written_manifest = json.loads(manifest_path.read_text())

    assert len(rows) == 5
    assert summary["row_count"] == 5
    assert summary["training_target"] == "direct_sql_control"
    assert summary["benchmark"] == "synthetic_semantic_layer_direct_sql"
    assert manifest["artifact_type"] == "semantic_layer_direct_sql_training_rows"
    assert manifest["row_count"] == 5
    assert manifest["command"] == [
        "uv",
        "run",
        "python",
        "-m",
        "data.semantic_layer_direct_sql_dataset",
    ]
    assert written_manifest == manifest


def test_checked_in_semantic_layer_direct_sql_training_artifacts_are_current(
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "semantic_layer_direct_sql_training_rows.jsonl"
    summary_path = tmp_path / "semantic_layer_direct_sql_training_rows_summary.json"
    manifest_path = tmp_path / "semantic_layer_direct_sql_training_rows.manifest.json"

    manifest = write_semantic_layer_direct_sql_training_artifacts(
        output_path=output_path,
        summary_path=summary_path,
        manifest_path=manifest_path,
    )

    checked_in_manifest_payload = json.loads(DEFAULT_MANIFEST_OUTPUT.read_text())
    for field in [
        "artifact_type",
        "training_target",
        "benchmark",
        "row_count",
        "fixture_source_path",
        "fixture_source_sha256",
    ]:
        assert checked_in_manifest_payload[field] == manifest[field]
    assert DEFAULT_OUTPUT.read_text() == output_path.read_text()
    assert DEFAULT_SUMMARY_OUTPUT.read_text() == summary_path.read_text()
