from __future__ import annotations

import json
from pathlib import Path

from data.metric_dsl_dataset import (
    DEFAULT_MANIFEST_OUTPUT,
    DEFAULT_OUTPUT,
    DEFAULT_SUMMARY_OUTPUT,
    build_metric_dsl_training_rows,
    summarize_metric_dsl_training_rows,
    write_metric_dsl_training_artifacts,
)
from data.synthetic_method_fixtures import build_synthetic_method_fixtures


def test_build_metric_dsl_training_rows_derives_non_oracle_rows_from_fixtures() -> None:
    rows = build_metric_dsl_training_rows(build_synthetic_method_fixtures())

    assert [row["fixture_id"] for row in rows] == [
        "grain_fanout_bridge",
        "measure_preservation_metric",
    ]
    assert all(row["training_target"] == "metric_dsl" for row in rows)
    assert all(row["evaluation_mode"] == "metric_dsl" for row in rows)
    assert all(row["benchmark"] == "synthetic_metric_dsl_bootstrap" for row in rows)
    assert all(row["oracle_policy"] == "non_oracle_inputs_only" for row in rows)
    assert all(row["messages"][-1]["role"] == "assistant" for row in rows)
    assert all(row["messages"][-1]["content"] == row["gold_dsl"] for row in rows)
    assert all(row["semantic_model"]["semantic_model_id"] == "synthetic_revenue_v1" for row in rows)

    first_user_prompt = rows[0]["messages"][1]["content"]
    assert "Respond with only the metric DSL query." in first_user_prompt
    assert "MEASURE(" not in first_user_prompt
    assert "reference_sql" not in first_user_prompt
    assert "Which campaigns generated the most revenue?" in first_user_prompt
    assert "duplicate_row_policy" in first_user_prompt


def test_summarize_metric_dsl_training_rows_reports_fixture_coverage() -> None:
    summary = summarize_metric_dsl_training_rows(
        build_metric_dsl_training_rows(build_synthetic_method_fixtures())
    )

    assert summary["row_count"] == 2
    assert summary["training_target"] == "metric_dsl"
    assert summary["semantic_model_ids"] == ["synthetic_revenue_v1"]
    assert summary["fixture_ids"] == [
        "grain_fanout_bridge",
        "measure_preservation_metric",
    ]
    assert summary["failure_mode_counts"]["measure_preservation"] == 1
    assert summary["failure_mode_counts"]["grain_fanout"] == 1


def test_write_metric_dsl_training_artifacts_writes_jsonl_summary_and_manifest(
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "metric_dsl_train.jsonl"
    summary_path = tmp_path / "metric_dsl_train_summary.json"
    manifest_path = tmp_path / "metric_dsl_train.manifest.json"

    manifest = write_metric_dsl_training_artifacts(
        output_path=output_path,
        summary_path=summary_path,
        manifest_path=manifest_path,
        command=["uv", "run", "python", "-m", "data.metric_dsl_dataset"],
    )

    rows = [json.loads(line) for line in output_path.read_text().splitlines()]
    summary = json.loads(summary_path.read_text())
    written_manifest = json.loads(manifest_path.read_text())

    assert [row["fixture_id"] for row in rows] == [
        "grain_fanout_bridge",
        "measure_preservation_metric",
    ]
    assert summary["row_count"] == 2
    assert summary["training_target"] == "metric_dsl"
    assert manifest["artifact_type"] == "metric_dsl_training_rows"
    assert manifest["row_count"] == 2
    assert manifest["output_sha256"]
    assert manifest["summary_sha256"]
    assert manifest["command"] == ["uv", "run", "python", "-m", "data.metric_dsl_dataset"]
    assert written_manifest == manifest


def test_checked_in_metric_dsl_training_artifacts_are_current(tmp_path: Path) -> None:
    output_path = tmp_path / "metric_dsl_training_rows.jsonl"
    summary_path = tmp_path / "metric_dsl_training_rows_summary.json"
    manifest_path = tmp_path / "metric_dsl_training_rows.manifest.json"

    manifest = write_metric_dsl_training_artifacts(
        output_path=output_path,
        summary_path=summary_path,
        manifest_path=manifest_path,
    )

    checked_in_output = DEFAULT_OUTPUT
    checked_in_summary = DEFAULT_SUMMARY_OUTPUT
    checked_in_manifest = DEFAULT_MANIFEST_OUTPUT

    checked_in_manifest_payload = json.loads(checked_in_manifest.read_text())
    for field in [
        "artifact_type",
        "training_target",
        "row_count",
        "fixture_source_path",
        "fixture_source_sha256",
    ]:
        assert checked_in_manifest_payload[field] == manifest[field]
    assert checked_in_output.read_text() == output_path.read_text()
    assert checked_in_summary.read_text() == summary_path.read_text()
