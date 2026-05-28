from __future__ import annotations

import json

from data.metric_dsl_training_rows import (
    build_metric_dsl_finetuning_rows,
    summarize_metric_dsl_finetuning_rows,
    write_metric_dsl_finetuning_artifacts,
)


def test_metric_dsl_training_rows_have_same_fixture_direct_sql_controls() -> None:
    dsl_rows, direct_sql_rows = build_metric_dsl_finetuning_rows()

    assert [row["fixture_id"] for row in dsl_rows] == [
        "grain_fanout_bridge",
        "measure_preservation_metric",
    ]
    assert [row["fixture_id"] for row in direct_sql_rows] == [
        row["fixture_id"] for row in dsl_rows
    ]
    assert all(row["training_target"] == "metric_dsl" for row in dsl_rows)
    assert all(row["training_target"] == "direct_sql_control" for row in direct_sql_rows)
    assert all("reference_sql" not in row["messages"][1]["content"] for row in dsl_rows)
    assert all(row["messages"][-1]["content"].startswith("MEASURE(") for row in dsl_rows)


def test_write_metric_dsl_finetuning_artifacts(tmp_path) -> None:
    dsl_output = tmp_path / "metric_dsl_training_rows.jsonl"
    direct_sql_output = tmp_path / "metric_dsl_direct_sql_training_rows.jsonl"
    summary_output = tmp_path / "metric_dsl_training_rows_summary.json"
    manifest_output = tmp_path / "metric_dsl_training_rows.manifest.json"

    manifest = write_metric_dsl_finetuning_artifacts(
        dsl_output_path=dsl_output,
        direct_sql_output_path=direct_sql_output,
        summary_path=summary_output,
        manifest_path=manifest_output,
    )

    dsl_rows = [json.loads(line) for line in dsl_output.read_text().splitlines()]
    direct_sql_rows = [json.loads(line) for line in direct_sql_output.read_text().splitlines()]
    summary = summarize_metric_dsl_finetuning_rows(dsl_rows, direct_sql_rows)

    assert manifest["metric_dsl_row_count"] == 2
    assert manifest["direct_sql_control_row_count"] == 2
    assert manifest["metric_dsl_output_sha256"]
    assert manifest["direct_sql_output_sha256"]
    assert json.loads(summary_output.read_text()) == summary
    assert json.loads(manifest_output.read_text()) == manifest
