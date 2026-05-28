from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from eval.run_metric_dsl_comparison import (
    evaluate_direct_sql_prediction_rows,
    run_metric_dsl_comparison,
)

SEMANTIC_MODEL = {
    "base_table": "orders",
    "measures": {"revenue": {"sql": "SUM(orders.amount)"}},
    "dimensions": {"customer_country": {"sql": "customers.country"}},
    "joins": [
        {
            "table": "customers",
            "sql_on": "orders.customer_id = customers.id",
            "required_by": ["customer_country"],
        }
    ],
}


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")


def _make_database(path: Path) -> None:
    with sqlite3.connect(path) as conn:
        conn.executescript(
            """
            CREATE TABLE customers (id INTEGER PRIMARY KEY, country TEXT);
            CREATE TABLE orders (id INTEGER PRIMARY KEY, customer_id INTEGER, amount REAL);
            INSERT INTO customers VALUES (1, 'FR'), (2, 'US');
            INSERT INTO orders VALUES (10, 1, 12.5), (11, 1, 7.5), (12, 2, 3.0);
            """
        )


def test_evaluate_direct_sql_prediction_rows_scores_generated_sql(tmp_path) -> None:
    database_path = tmp_path / "store.sqlite"
    _make_database(database_path)
    rows = [
        {
            "id": "metric-1",
            "raw_generation": "```sql\nSELECT 0 AS revenue\n```",
            "reference_sql": "SELECT SUM(orders.amount) AS revenue FROM orders",
            "database_path": str(database_path),
        }
    ]

    [result] = evaluate_direct_sql_prediction_rows(rows, model_name="direct-sql-model")

    assert result["evaluation_mode"] == "non_oracle_generation"
    assert result["generated_sql"] == "SELECT 0 AS revenue"
    assert result["model_name"] == "direct-sql-model"
    assert result["value_execution_score"] == 0.0
    assert result["strict_execution_score"] == 0.0
    assert result["syntax_valid"] is True


def test_run_metric_dsl_comparison_writes_metric_direct_and_compared_manifests(
    tmp_path,
) -> None:
    database_path = tmp_path / "store.sqlite"
    _make_database(database_path)
    reference_sql = (
        "SELECT customers.country AS customer_country, SUM(orders.amount) AS revenue "
        "FROM orders JOIN customers ON orders.customer_id = customers.id "
        "GROUP BY customers.country"
    )
    metric_input = tmp_path / "metric_predictions.jsonl"
    direct_input = tmp_path / "direct_predictions.jsonl"
    _write_jsonl(
        metric_input,
        [
            {
                "id": "metric-1",
                "predicted_dsl": "MEASURE(revenue) BY customer_country",
                "gold_dsl": "MEASURE(revenue) BY customer_country",
                "semantic_model": SEMANTIC_MODEL,
                "reference_sql": reference_sql,
                "database_path": str(database_path),
            }
        ],
    )
    _write_jsonl(
        direct_input,
        [
            {
                "id": "metric-1",
                "generated_sql": "SELECT 0 AS revenue",
                "reference_sql": reference_sql,
                "database_path": str(database_path),
            }
        ],
    )

    compared = run_metric_dsl_comparison(
        metric_dsl_predictions=metric_input,
        direct_sql_predictions=direct_input,
        output_dir=tmp_path / "results" / "metric_dsl",
        run_id="paired-smoke",
        metric_dsl_model_name="metric-dsl-model",
        direct_sql_model_name="direct-sql-model",
        command=["python", "-m", "eval.run_metric_dsl_comparison"],
    )

    assert (tmp_path / "results" / "metric_dsl" / "paired-smoke.metric_dsl.jsonl").exists()
    assert (tmp_path / "results" / "metric_dsl" / "paired-smoke.direct_sql.jsonl").exists()
    assert (
        tmp_path / "results" / "metric_dsl" / "paired-smoke.comparison.manifest.json"
    ).exists()
    assert compared["benchmark"] == "metric_dsl"
    assert compared["metrics"]["direct_sql_model_name"] == "direct-sql-model"
    assert compared["metrics"]["metric_dsl_value_delta_vs_direct_sql"] == 1.0
    assert compared["metrics"]["metric_dsl_comparable_row_count"] == 1
