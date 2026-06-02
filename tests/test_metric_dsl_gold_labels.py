from __future__ import annotations

import json
from pathlib import Path

from data.metric_dsl_gold_labels import (
    derive_metric_dsl_gold_label,
    label_metric_dsl_candidates,
    summarize_metric_dsl_gold_labels,
    write_metric_dsl_gold_label_artifacts,
)


def _candidate(*, reference_sql: str | None = None, gold_plan: dict | None = None) -> dict:
    return {
        "id": "dialog-store:0",
        "dialog_id": "dialog-store",
        "turn_index": 0,
        "turn_count": 1,
        "database_id": "store",
        "split_id": "cosql_dev_clean_holdout_v1",
        "split_role": "clean_local_holdout",
        "messages": [
            {
                "role": "system",
                "content": "Return only the metric DSL.",
            },
            {
                "role": "user",
                "content": (
                    "Database: store\n\n"
                    "Schema/context:\n"
                    "customers(id number, country text)\n"
                    "orders(id number, customer_id number, amount number)\n\n"
                    "Semantic model:\n"
                    "- Cube customers (grain: one row per customers; primary key: id)\n"
                    "  Dimensions: id [number, primary_key], country [string]\n"
                    "  Measures: count\n"
                    "- Cube orders (grain: one row per orders; primary key: id)\n"
                    "  Dimensions: id [number, primary_key], customer_id [number], amount [number]\n"
                    "  Measures: count, sum_amount=sum(amount), avg_amount=avg(amount)\n"
                    "  Joins: orders.customer_id -> customers.id (many_to_one)\n\n"
                    "Question:\nShow revenue by country.\n\n"
                    "Return only the metric DSL."
                ),
            },
        ],
        "reference_sql": reference_sql
        or (
            "SELECT customers.country, SUM(orders.amount) FROM orders "
            "JOIN customers ON orders.customer_id = customers.id "
            "GROUP BY customers.country"
        ),
        "gold_plan": gold_plan
        or {
            "relevant_tables": ["orders", "customers"],
            "relevant_columns": ["customers.country", "orders.amount"],
            "join_path": ["orders.customer_id = customers.id"],
            "query_skeleton": {"select": True, "join": True, "group_by": True},
            "projection_shape": {
                "selected_expressions": ["customers.country", "SUM(orders.amount)"],
                "aggregations": ["sum(orders.amount)"],
                "group_by": ["customers.country"],
                "selected_count": 2,
            },
        },
        "readiness_blockers": ["structured gold Metric DSL labels missing"],
        "reference_sql_visible_to_model_prompt": False,
        "scorer_fields_visible_to_model_prompt": False,
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_derive_metric_dsl_gold_label_maps_aggregate_dimension_and_join() -> None:
    row = derive_metric_dsl_gold_label(_candidate())

    assert row["gold_metric_dsl_available"] is True
    assert row["gold_dsl"] == "MEASURE(sum_amount) BY country"
    assert row["reference_metric_dsl"] == row["gold_dsl"]
    assert row["semantic_model"]["base_table"] == "orders"
    assert row["semantic_model"]["measures"]["sum_amount"]["sql"] == "SUM(orders.amount)"
    assert row["semantic_model"]["dimensions"]["country"]["sql"] == "customers.country"
    assert row["semantic_model"]["joins"] == [
        {
            "table": "customers",
            "sql_on": "orders.customer_id = customers.id",
            "required_by": ["country"],
        }
    ]
    assert row["semantic_model_source"] == "prepared_prompt_context_non_oracle"
    assert row["readiness_blockers"] == []
    prompt = json.dumps(row["messages"], sort_keys=True)
    assert row["gold_dsl"] not in prompt
    assert "gold_plan" not in prompt


def test_derive_metric_dsl_gold_label_keeps_unsupported_shape_blocked() -> None:
    row = derive_metric_dsl_gold_label(
        _candidate(
            reference_sql=(
                "SELECT customers.country FROM customers UNION "
                "SELECT suppliers.country FROM suppliers"
            ),
            gold_plan={
                "query_skeleton": {"union": True},
                "projection_shape": {"selected_expressions": ["customers.country"]},
            },
        )
    )

    assert row["gold_metric_dsl_available"] is False
    assert "gold_dsl" not in row
    assert row["metric_dsl_label_error"] == "unsupported SQL shape: set operation"
    assert row["readiness_blockers"] == ["metric DSL gold label derivation incomplete"]


def test_derive_metric_dsl_gold_label_resolves_sql_alias_and_table_case() -> None:
    row = derive_metric_dsl_gold_label(
        _candidate(
            reference_sql=(
                "SELECT T1.Country, SUM(T2.Amount) FROM Customers AS T1 "
                "JOIN Orders AS T2 ON T2.Customer_ID = T1.ID GROUP BY T1.Country"
            ),
            gold_plan={
                "relevant_tables": ["Orders", "Customers"],
                "query_skeleton": {"select": True, "join": True, "group_by": True},
                "projection_shape": {
                    "selected_expressions": ["T1.Country", "SUM(T2.Amount)"],
                    "aggregations": ["SUM(T2.Amount)"],
                    "group_by": ["T1.Country"],
                    "selected_count": 2,
                },
            },
        )
    )

    assert row["gold_metric_dsl_available"] is True
    assert row["gold_dsl"] == "MEASURE(sum_amount) BY country"
    assert row["semantic_model"]["base_table"] == "orders"
    assert row["semantic_model"]["dimensions"]["country"]["sql"] == "customers.country"


def test_label_metric_dsl_candidates_and_summary_count_labelable_rows(tmp_path: Path) -> None:
    input_path = tmp_path / "candidates.jsonl"
    _write_jsonl(
        input_path,
        [
            _candidate(),
            _candidate(
                reference_sql="SELECT COUNT(*) FROM orders",
                gold_plan={
                    "relevant_tables": ["orders"],
                    "query_skeleton": {"select": True},
                    "projection_shape": {
                        "selected_expressions": ["COUNT(*)"],
                        "aggregations": ["count(*)"],
                        "selected_count": 1,
                    },
                },
            ),
        ],
    )

    rows = label_metric_dsl_candidates(input_path)
    summary = summarize_metric_dsl_gold_labels(rows, input_path=input_path)

    assert [row["gold_dsl"] for row in rows] == [
        "MEASURE(sum_amount) BY country",
        "MEASURE(count)",
    ]
    assert summary["artifact_type"] == "metric_dsl_gold_label_summary"
    assert summary["candidate_count"] == 2
    assert summary["labelled_count"] == 2
    assert summary["unlabelled_count"] == 0
    assert summary["split_roles"] == {"clean_local_holdout": 2}
    assert summary["promotion_status"] == "not_ready"
    assert "generate same-row Metric DSL predictions" in summary["next_step"]


def test_write_metric_dsl_gold_label_artifacts(tmp_path: Path) -> None:
    input_path = tmp_path / "candidates.jsonl"
    output_path = tmp_path / "metric_dsl_gold_labels.jsonl"
    summary_path = tmp_path / "metric_dsl_gold_label_summary.json"
    manifest_path = tmp_path / "metric_dsl_gold_labels.manifest.json"
    _write_jsonl(
        input_path,
        [
            _candidate(),
            _candidate(
                reference_sql=(
                    "SELECT customers.country FROM customers UNION "
                    "SELECT suppliers.country FROM suppliers"
                ),
                gold_plan={
                    "query_skeleton": {"union": True},
                    "projection_shape": {"selected_expressions": ["customers.country"]},
                },
            ),
        ],
    )

    manifest = write_metric_dsl_gold_label_artifacts(
        input_path=input_path,
        output_path=output_path,
        summary_path=summary_path,
        manifest_path=manifest_path,
        command=["write-labels"],
    )

    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert len(rows) == 1
    assert rows[0]["gold_metric_dsl_available"] is True
    assert "metric_dsl_label_error" not in rows[0]
    assert summary["candidate_count"] == 2
    assert summary["labelled_count"] == 1
    assert summary["unlabelled_count"] == 1
    assert manifest["artifact_type"] == "metric_dsl_gold_label_manifest"
    assert manifest["candidate_count"] == 2
    assert manifest["labelled_count"] == 1
    assert manifest["unlabelled_count"] == 1
    assert manifest["output_row_count"] == 1
    assert manifest["input_sha256"]
    assert manifest["output_sha256"]
    assert manifest["summary_sha256"]
    assert json.loads(manifest_path.read_text(encoding="utf-8")) == manifest
