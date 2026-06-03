from __future__ import annotations

import json

from data.metric_dsl_prediction_inputs import (
    build_metric_dsl_prediction_inputs,
    summarize_metric_dsl_prediction_inputs,
    write_metric_dsl_prediction_input_artifacts,
)


def _prompt_text(row: dict) -> str:
    return json.dumps(row["messages"], sort_keys=True)


def test_metric_dsl_prediction_inputs_pair_same_fixture_ids_without_prompt_leakage() -> None:
    metric_rows, direct_rows = build_metric_dsl_prediction_inputs()

    assert [row["fixture_id"] for row in metric_rows] == [
        "grain_fanout_bridge",
        "measure_preservation_metric",
    ]
    assert [row["fixture_id"] for row in direct_rows] == [
        row["fixture_id"] for row in metric_rows
    ]
    assert all(row["generation_target"] == "metric_dsl" for row in metric_rows)
    assert all(row["generation_target"] == "direct_sql" for row in direct_rows)
    assert all(row["messages"][-1]["role"] == "user" for row in metric_rows + direct_rows)
    assert all("assistant" not in {message["role"] for message in row["messages"]} for row in metric_rows + direct_rows)

    for row in metric_rows + direct_rows:
        prompt = _prompt_text(row)
        assert row["reference_sql"] not in prompt
        assert "gold_dsl" not in prompt
        if row.get("gold_dsl"):
            assert row["gold_dsl"] not in prompt
        assert row["semantic_model"]
        assert row["reference_sql_visible_to_model"] is False
        assert row["scoring_fields_visible_to_model"] is False


def test_metric_dsl_prediction_input_summary_records_eval_command() -> None:
    metric_rows, direct_rows = build_metric_dsl_prediction_inputs()
    summary = summarize_metric_dsl_prediction_inputs(metric_rows, direct_rows)

    assert summary["metric_dsl_prediction_input_count"] == 2
    assert summary["direct_sql_prediction_input_count"] == 2
    assert summary["comparison_contract"] == "same_fixture_metric_dsl_vs_direct_sql_predictions"
    assert "eval.run_metric_dsl_comparison" in summary["evaluation_command"]


def test_write_metric_dsl_prediction_input_artifacts(tmp_path) -> None:
    metric_output = tmp_path / "metric_predictions.input.jsonl"
    direct_output = tmp_path / "direct_predictions.input.jsonl"
    summary_output = tmp_path / "metric_prediction_inputs_summary.json"
    manifest_output = tmp_path / "metric_prediction_inputs.manifest.json"

    manifest = write_metric_dsl_prediction_input_artifacts(
        metric_output_path=metric_output,
        direct_output_path=direct_output,
        summary_path=summary_output,
        manifest_path=manifest_output,
    )

    metric_rows = [json.loads(line) for line in metric_output.read_text().splitlines()]
    direct_rows = [json.loads(line) for line in direct_output.read_text().splitlines()]
    summary = json.loads(summary_output.read_text())

    assert len(metric_rows) == 2
    assert len(direct_rows) == 2
    assert summary["fixture_ids"] == [row["fixture_id"] for row in metric_rows]
    assert manifest["metric_output_sha256"]
    assert manifest["direct_output_sha256"]
    assert json.loads(manifest_output.read_text()) == manifest
