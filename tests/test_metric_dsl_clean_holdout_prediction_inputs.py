from __future__ import annotations

import json
from pathlib import Path

from data.metric_dsl_clean_holdout_prediction_inputs import (
    build_metric_dsl_clean_holdout_prediction_inputs,
    paired_prediction_rows_from_gold_label,
    summarize_metric_dsl_clean_holdout_prediction_inputs,
    write_metric_dsl_clean_holdout_prediction_input_artifacts,
)


def _gold_label_row() -> dict:
    return {
        "schema_version": 1,
        "artifact_type": "metric_dsl_gold_label_rows",
        "id": "prepared-1:0",
        "dialog_id": "prepared-1",
        "turn_index": 0,
        "turn_count": 1,
        "database_id": "store",
        "split_id": "cosql_dev_clean_holdout_v1",
        "split_role": "clean_local_holdout",
        "split_row_id": "cosql_dev:0100:store",
        "evaluation_mode": "metric_dsl",
        "generation_target": "metric_dsl",
        "messages": [
            {
                "role": "system",
                "content": "You translate analytics questions into the project metric DSL.",
            },
            {
                "role": "user",
                "content": (
                    "Database: store\n\n"
                    "Schema/context:\norders(id number, amount number)\n\n"
                    "Semantic model:\n- Cube orders\n\n"
                    "Question:\nShow revenue.\n\n"
                    "Return only the metric DSL."
                ),
            },
        ],
        "reference_sql": "SELECT SUM(amount) FROM orders",
        "gold_dsl": "MEASURE(sum_amount)",
        "reference_metric_dsl": "MEASURE(sum_amount)",
        "semantic_model": {
            "base_table": "orders",
            "measures": {"sum_amount": {"sql": "SUM(orders.amount)"}},
            "dimensions": {},
            "joins": [],
        },
        "semantic_model_source": "prepared_prompt_context_non_oracle",
        "gold_metric_dsl_available": True,
        "gold_plan": {"query_skeleton": {"select": True}},
        "readiness_blockers": [],
        "reference_sql_visible_to_model_prompt": False,
        "scorer_fields_visible_to_model_prompt": False,
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )


def _prompt_text(row: dict) -> str:
    return json.dumps(row["messages"], sort_keys=True)


def test_paired_prediction_rows_keep_same_identity_without_prompt_leakage() -> None:
    metric_row, direct_row = paired_prediction_rows_from_gold_label(
        _gold_label_row(),
        database_root=Path("data/raw/cosql_dataset/database"),
    )

    assert metric_row["id"] == direct_row["id"] == "prepared-1:0"
    assert metric_row["generation_target"] == "metric_dsl"
    assert metric_row["evaluation_mode"] == "metric_dsl"
    assert direct_row["generation_target"] == "direct_sql"
    assert direct_row["evaluation_mode"] == "non_oracle_generation"
    assert metric_row["database_path"] == "data/raw/cosql_dataset/database/store/store.sqlite"
    assert direct_row["database_path"] == metric_row["database_path"]
    assert "gold_plan" not in metric_row
    assert "gold_plan" not in direct_row

    for row in (metric_row, direct_row):
        prompt = _prompt_text(row)
        assert row["reference_sql"] not in prompt
        assert row["gold_dsl"] not in prompt
        assert row["reference_sql_visible_to_model"] is False
        assert row["scoring_fields_visible_to_model"] is False
        assert row["reference_sql_visible_to_model_prompt"] is False
        assert row["scorer_fields_visible_to_model_prompt"] is False

    assert "Return only SQL" in direct_row["messages"][-1]["content"]
    assert "metric DSL" not in direct_row["messages"][0]["content"]


def test_build_and_summarize_prediction_inputs(tmp_path: Path) -> None:
    input_path = tmp_path / "gold_labels.jsonl"
    leaky_row = _gold_label_row()
    leaky_row["id"] = "prepared-2:1"
    leaky_row["messages"] = [
        *leaky_row["messages"][:-1],
        {"role": "assistant", "content": leaky_row["reference_sql"]},
        leaky_row["messages"][-1],
    ]
    _write_jsonl(input_path, [_gold_label_row(), leaky_row])

    metric_rows, direct_rows = build_metric_dsl_clean_holdout_prediction_inputs(
        input_path,
        database_root=Path("data/raw/cosql_dataset/database"),
    )
    summary = summarize_metric_dsl_clean_holdout_prediction_inputs(
        metric_rows,
        direct_rows,
        input_path=input_path,
        database_root=Path("data/raw/cosql_dataset/database"),
    )

    assert [row["id"] for row in metric_rows] == [row["id"] for row in direct_rows]
    assert [row["id"] for row in metric_rows] == ["prepared-1:0"]
    assert summary["artifact_type"] == "metric_dsl_clean_holdout_prediction_input_summary"
    assert summary["paired_row_count"] == 1
    assert summary["metric_dsl_prediction_input_count"] == 1
    assert summary["direct_sql_prediction_input_count"] == 1
    assert summary["input_labelled_count"] == 1
    assert summary["split_roles"] == {"clean_local_holdout": 1}
    assert summary["promotion_status"] == "not_ready"
    assert "generate same-row Metric DSL" in summary["next_step"]


def test_write_prediction_input_artifacts(tmp_path: Path) -> None:
    input_path = tmp_path / "gold_labels.jsonl"
    metric_output = tmp_path / "metric_inputs.jsonl"
    direct_output = tmp_path / "direct_inputs.jsonl"
    summary_path = tmp_path / "summary.json"
    manifest_path = tmp_path / "manifest.json"
    leaky_row = _gold_label_row()
    leaky_row["id"] = "prepared-2:1"
    leaky_row["messages"] = [
        *leaky_row["messages"][:-1],
        {"role": "assistant", "content": leaky_row["reference_sql"]},
        leaky_row["messages"][-1],
    ]
    _write_jsonl(input_path, [_gold_label_row(), leaky_row])

    manifest = write_metric_dsl_clean_holdout_prediction_input_artifacts(
        input_path=input_path,
        metric_output_path=metric_output,
        direct_output_path=direct_output,
        summary_path=summary_path,
        manifest_path=manifest_path,
        database_root=Path("data/raw/cosql_dataset/database"),
        command=["write-prediction-inputs"],
    )

    metric_rows = [json.loads(line) for line in metric_output.read_text().splitlines()]
    direct_rows = [json.loads(line) for line in direct_output.read_text().splitlines()]
    summary = json.loads(summary_path.read_text())

    assert len(metric_rows) == 1
    assert len(direct_rows) == 1
    assert summary["paired_row_count"] == 1
    assert summary["input_labelled_count"] == 2
    assert summary["excluded_prompt_leakage_count"] == 1
    assert summary["excluded_prompt_leakage_rows"][0]["id"] == "prepared-2:1"
    assert manifest["artifact_type"] == "metric_dsl_clean_holdout_prediction_input_manifest"
    assert manifest["paired_row_count"] == 1
    assert manifest["input_labelled_count"] == 2
    assert manifest["excluded_prompt_leakage_count"] == 1
    assert manifest["input_sha256"]
    assert manifest["metric_output_sha256"]
    assert manifest["direct_output_sha256"]
    assert manifest["summary_sha256"]
    assert json.loads(manifest_path.read_text()) == manifest
