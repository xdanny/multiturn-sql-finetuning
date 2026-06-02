from __future__ import annotations

import json
from pathlib import Path

import pytest

from data.metric_dsl_clean_holdout_readiness import (
    build_metric_dsl_clean_holdout_candidates,
    summarize_metric_dsl_clean_holdout_candidates,
    write_metric_dsl_clean_holdout_candidate_artifacts,
)


def _prepared_dialog(*, split_role: str = "clean_local_holdout") -> dict:
    return {
        "dialog_id": "dialog-store",
        "source": "cosql_dev_clean_holdout_v1",
        "history_policy": "gold_sql_teacher_forced",
        "evaluation_mode": "non_oracle_generation",
        "database_id": "store",
        "split_id": "cosql_dev_clean_holdout_v1",
        "split_role": split_role,
        "split_row_id": "cosql_dev:0001:store",
        "messages": [
            {"role": "system", "content": "Return only SQL."},
            {
                "role": "user",
                "content": (
                    "Database: store\n\n"
                    "Schema/context:\n"
                    "customers(id int, country text)\n"
                    "orders(id int, customer_id int, amount real)\n\n"
                    "Semantic model:\n"
                    "- Cube orders (grain: one row per orders)\n"
                    "  Measures: count, sum_amount=sum(amount)\n\n"
                    "Question:\nShow revenue by country."
                ),
            },
            {
                "role": "assistant",
                "content": (
                    "SELECT customers.country, SUM(orders.amount) FROM orders "
                    "JOIN customers ON orders.customer_id = customers.id "
                    "GROUP BY customers.country;"
                ),
            },
            {"role": "user", "content": "Question:\nOnly count the orders."},
            {"role": "assistant", "content": "SELECT COUNT(*) FROM orders;"},
        ],
        "gold_plans": [
            {
                "query_skeleton": {"select": True, "join": True, "group_by": True},
                "projection_shape": {
                    "selected_expressions": [
                        "customers.country",
                        "SUM(orders.amount)",
                    ],
                    "aggregations": ["sum"],
                    "group_by": ["customers.country"],
                    "selected_count": 2,
                },
            },
            {
                "query_skeleton": {"select": True},
                "projection_shape": {
                    "selected_expressions": ["COUNT(*)"],
                    "aggregations": ["count"],
                    "group_by": [],
                    "selected_count": 1,
                },
            },
        ],
        "schema_link_labels": [
            {
                "query_skeleton": {"select": True, "join": True, "group_by": True},
                "projection_shape": {
                    "selected_expressions": [
                        "customers.country",
                        "SUM(orders.amount)",
                    ],
                    "aggregations": ["sum"],
                    "group_by": ["customers.country"],
                    "selected_count": 2,
                },
            },
            {
                "query_skeleton": {"select": True},
                "projection_shape": {
                    "selected_expressions": ["COUNT(*)"],
                    "aggregations": ["count"],
                    "group_by": [],
                    "selected_count": 1,
                },
            },
        ],
        "uses_oracle_planning_hints": False,
        "semantic_context_pruned_by_oracle_labels": False,
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _prompt_text(row: dict) -> str:
    return json.dumps(row["messages"], sort_keys=True)


def test_build_metric_dsl_clean_holdout_candidates_keeps_labels_scorer_side(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "clean_holdout.jsonl"
    _write_jsonl(input_path, [_prepared_dialog()])

    rows = build_metric_dsl_clean_holdout_candidates(input_path)

    assert [row["id"] for row in rows] == ["dialog-store:0", "dialog-store:1"]
    assert all(row["split_role"] == "clean_local_holdout" for row in rows)
    assert all(row["generation_target"] == "metric_dsl" for row in rows)
    assert rows[0]["metric_signals"] == ["aggregation", "group_by"]
    assert rows[1]["metric_signals"] == ["aggregation"]
    assert rows[0]["reference_sql"].startswith("SELECT customers.country")
    assert rows[0]["reference_sql_visible_to_model_prompt"] is False
    assert rows[0]["scorer_fields_visible_to_model_prompt"] is False
    assert rows[0]["gold_metric_dsl_available"] is False
    assert rows[0]["readiness_blockers"] == ["structured gold Metric DSL labels missing"]
    assert "Return only the metric DSL" in rows[0]["messages"][0]["content"]
    assert "Return only SQL" not in rows[0]["messages"][0]["content"]

    for row in rows:
        prompt = _prompt_text(row)
        assert row["reference_sql"] not in prompt
        assert "schema_link_labels" not in prompt
        assert "gold_plan" not in prompt


def test_build_metric_dsl_clean_holdout_candidates_rejects_non_holdout_split(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "proxy.jsonl"
    _write_jsonl(input_path, [_prepared_dialog(split_role="proxy_dev_seen")])

    with pytest.raises(ValueError, match="split_role=clean_local_holdout"):
        build_metric_dsl_clean_holdout_candidates(input_path)


def test_metric_dsl_clean_holdout_candidates_allow_repeated_prior_sql_history(
    tmp_path: Path,
) -> None:
    dialog = _prepared_dialog()
    dialog["messages"][2]["content"] = "SELECT COUNT(*) FROM orders;"
    input_path = tmp_path / "clean_holdout.jsonl"
    _write_jsonl(input_path, [dialog])

    rows = build_metric_dsl_clean_holdout_candidates(input_path)

    assert rows[1]["reference_sql"] == "SELECT COUNT(*) FROM orders;"
    assert "SELECT COUNT(*) FROM orders;" in _prompt_text(rows[1])
    assert rows[1]["messages"][-1]["role"] == "user"


def test_metric_dsl_clean_holdout_candidate_summary_records_blockers(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "clean_holdout.jsonl"
    _write_jsonl(input_path, [_prepared_dialog()])
    rows = build_metric_dsl_clean_holdout_candidates(input_path)

    summary = summarize_metric_dsl_clean_holdout_candidates(rows, input_path=input_path)

    assert summary["artifact_type"] == "metric_dsl_clean_holdout_candidate_summary"
    assert summary["candidate_count"] == 2
    assert summary["split_roles"] == {"clean_local_holdout": 2}
    assert summary["metric_signal_counts"] == {"aggregation": 2, "group_by": 1}
    assert summary["readiness_blockers"] == {
        "structured gold Metric DSL labels missing": 2
    }
    assert summary["promotion_status"] == "not_ready"
    assert summary["next_step"] == (
        "derive or author scorer-side gold Metric DSL labels, then generate same-row "
        "Metric DSL and direct-SQL predictions for eval.run_metric_dsl_comparison"
    )


def test_write_metric_dsl_clean_holdout_candidate_artifacts(tmp_path: Path) -> None:
    input_path = tmp_path / "clean_holdout.jsonl"
    output_path = tmp_path / "metric_dsl_clean_holdout_candidates.jsonl"
    summary_path = tmp_path / "metric_dsl_clean_holdout_readiness.json"
    manifest_path = tmp_path / "metric_dsl_clean_holdout_candidates.manifest.json"
    _write_jsonl(input_path, [_prepared_dialog()])

    manifest = write_metric_dsl_clean_holdout_candidate_artifacts(
        input_path=input_path,
        output_path=output_path,
        summary_path=summary_path,
        manifest_path=manifest_path,
        command=["write-candidates"],
    )

    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert len(rows) == 2
    assert summary["candidate_count"] == 2
    assert manifest["artifact_type"] == "metric_dsl_clean_holdout_candidate_manifest"
    assert manifest["candidate_count"] == 2
    assert manifest["input_sha256"]
    assert manifest["output_sha256"]
    assert manifest["summary_sha256"]
    assert json.loads(manifest_path.read_text(encoding="utf-8")) == manifest
