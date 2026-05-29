from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from eval.metric_dsl_eval import (
    evaluate_metric_dsl_rows,
    run_metric_dsl_eval,
    summarize_metric_dsl_results,
)

SEMANTIC_MODEL = {
    "base_table": "orders",
    "measures": {
        "revenue": {"sql": "SUM(orders.amount)"},
        "orders_count": {"sql": "COUNT(DISTINCT orders.id)"},
    },
    "dimensions": {
        "customer_country": {"sql": "customers.country"},
        "order_month": {"sql": "strftime('%Y-%m', orders.created_at)"},
    },
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


def test_evaluate_metric_dsl_rows_scores_intent_compile_and_execution(tmp_path) -> None:
    database_path = tmp_path / "store.sqlite"
    _make_database(database_path)
    rows = [
        {
            "id": "metric-1",
            "predicted_dsl": "MEASURE(revenue) BY customer_country WHERE customer_country = 'FR'",
            "gold_dsl": "MEASURE(revenue) BY customer_country WHERE customer_country = 'FR'",
            "semantic_model": SEMANTIC_MODEL,
            "reference_sql": (
                "SELECT customers.country AS customer_country, SUM(orders.amount) AS revenue "
                "FROM orders JOIN customers ON orders.customer_id = customers.id "
                "WHERE customers.country = 'FR' GROUP BY customers.country"
            ),
            "database_path": str(database_path),
        }
    ]

    [result] = evaluate_metric_dsl_rows(rows)

    assert result["id"] == "metric-1"
    assert result["parsed_predicted_metric_query"] == {
        "measures": ["revenue"],
        "dimensions": ["customer_country"],
        "filters": ["customer_country = 'FR'"],
        "order_by": None,
        "limit": None,
    }
    assert result["parsed_gold_metric_query"]["measures"] == ["revenue"]
    assert result["metric_dsl_parse_success"] is True
    assert result["semantic_model_sha256"]
    assert result["metric_scores"] == {
        "measure_f1": 1.0,
        "dimension_f1": 1.0,
        "filter_f1": 1.0,
        "measure_preservation": 1.0,
    }
    assert result["compile_success"] is True
    assert result["compiled_sql"].startswith("SELECT customers.country AS customer_country")
    assert result["value_execution_score"] == 1.0
    assert result["strict_execution_score"] == 1.0
    assert result["syntax_valid"] is True


def test_evaluate_metric_dsl_rows_keeps_measure_preservation_separate_from_sql_shape() -> None:
    rows = [
        {
            "id": "metric-raw-sql-like",
            "predicted_dsl": "SUM(orders.amount) BY customer_country",
            "gold_dsl": "MEASURE(revenue) BY customer_country",
            "semantic_model": SEMANTIC_MODEL,
        }
    ]

    [result] = evaluate_metric_dsl_rows(rows)

    assert result["metric_scores"]["measure_f1"] == 0.0
    assert result["metric_scores"]["dimension_f1"] == 0.0
    assert result["metric_scores"]["measure_preservation"] == 0.0
    assert result["metric_dsl_parse_success"] is False
    assert result["compile_success"] is False
    assert result["compiled_sql"] is None
    assert "MEASURE" in result["parse_error"]


def test_evaluate_metric_dsl_rows_rejects_sql_tail_after_measure_token() -> None:
    rows = [
        {
            "id": "metric-sql-tail",
            "predicted_dsl": "MEASURE(revenue) FROM orders",
            "gold_dsl": "MEASURE(revenue)",
            "semantic_model": SEMANTIC_MODEL,
        }
    ]

    [result] = evaluate_metric_dsl_rows(rows)

    assert result["metric_dsl_parse_success"] is False
    assert result["compile_success"] is False
    assert result["compiled_sql"] is None
    assert "SQL keyword" in result["parse_error"]


def test_evaluate_metric_dsl_rows_rejects_sql_tails_inside_clause_bodies() -> None:
    for predicted_dsl in [
        "MEASURE(revenue) WHERE 1=1 FROM orders",
        "MEASURE(revenue) ORDER BY revenue FROM orders",
        "MEASURE(revenue) BY customer_country WHERE customer_country = 'FR' FROM orders",
    ]:
        [result] = evaluate_metric_dsl_rows(
            [
                {
                    "id": "metric-sql-tail",
                    "predicted_dsl": predicted_dsl,
                    "gold_dsl": "MEASURE(revenue)",
                    "semantic_model": SEMANTIC_MODEL,
                }
            ]
        )

        assert result["metric_dsl_parse_success"] is False
        assert result["compile_success"] is False
        assert result["compiled_sql"] is None
        assert "SQL keyword" in result["parse_error"]


def test_evaluate_metric_dsl_rows_accepts_generation_and_reference_field_aliases() -> None:
    rows = [
        {
            "id": "metric-aliases",
            "generated_metric_dsl": "MEASURE(revenue) BY customer_country",
            "reference_metric_dsl": "MEASURE(revenue) BY customer_country",
            "raw_generation": "MEASURE(revenue) BY customer_country",
            "semantic_model": SEMANTIC_MODEL,
        }
    ]

    [result] = evaluate_metric_dsl_rows(rows)

    assert result["predicted_dsl"] == "MEASURE(revenue) BY customer_country"
    assert result["gold_dsl"] == "MEASURE(revenue) BY customer_country"
    assert result["raw_generation"] == "MEASURE(revenue) BY customer_country"
    assert result["metric_scores"]["measure_preservation"] == 1.0


def test_evaluate_metric_dsl_rows_marks_unknown_measure_as_compile_failure() -> None:
    rows = [
        {
            "id": "metric-bad",
            "predicted_dsl": "MEASURE(gross_margin) BY customer_country",
            "gold_dsl": "MEASURE(revenue) BY customer_country",
            "semantic_model": SEMANTIC_MODEL,
        }
    ]

    [result] = evaluate_metric_dsl_rows(rows)

    assert result["compile_success"] is False
    assert result["compiled_sql"] is None
    assert result["metric_dsl_parse_success"] is True
    assert result["parse_error"] is None
    assert "unknown measure" in result["compile_error"]
    assert result["metric_scores"]["measure_preservation"] == 0.0


def test_evaluate_metric_dsl_rows_records_missing_contract_fields_as_row_failure() -> None:
    rows = [{"id": "metric-missing", "predicted_dsl": "MEASURE(revenue)"}]

    [result] = evaluate_metric_dsl_rows(rows)

    assert result["metric_dsl_parse_success"] is False
    assert result["compile_success"] is False
    assert result["semantic_model_error"] == "missing semantic_model"
    assert result["compiled_sql"] is None


def test_summarize_metric_dsl_results_aggregates_semantic_and_sql_metrics() -> None:
    results = [
        {
            "metric_scores": {
                "measure_f1": 1.0,
                "dimension_f1": 1.0,
                "filter_f1": 1.0,
                "measure_preservation": 1.0,
            },
            "metric_dsl_parse_success": True,
            "compile_success": True,
            "sql_execution_attempted": True,
            "value_execution_score": 1.0,
            "strict_execution_score": 0.0,
            "syntax_valid": True,
            "database_path": "store.sqlite",
        },
        {
            "metric_scores": {
                "measure_f1": 0.0,
                "dimension_f1": 1.0,
                "filter_f1": 0.0,
                "measure_preservation": 0.0,
            },
            "metric_dsl_parse_success": True,
            "compile_success": False,
            "sql_execution_attempted": False,
            "value_execution_score": 0.0,
            "strict_execution_score": 0.0,
            "syntax_valid": False,
            "database_path": "store.sqlite",
        },
    ]

    summary = summarize_metric_dsl_results(results)

    assert summary["rows"] == 2
    assert summary["metric_dsl_parse_rate"] == 1.0
    assert summary["metric_dsl_compile_rate"] == 0.5
    assert summary["compiled_sql_execution_attempt_rate"] == 0.5
    assert summary["compiled_sql_execution_evaluated_rows"] == 1
    assert summary["measure_f1"] == 0.5
    assert summary["dimension_f1"] == 1.0
    assert summary["filter_f1"] == 0.5
    assert summary["measure_preservation"] == 0.5
    assert summary["value_execution_accuracy"] == 0.5
    assert summary["strict_execution_accuracy"] == 0.0
    assert summary["syntax_accuracy"] == 0.5


def test_run_metric_dsl_eval_writes_results_and_manifest(tmp_path) -> None:
    database_path = tmp_path / "store.sqlite"
    _make_database(database_path)
    input_path = tmp_path / "metric_dsl_predictions.jsonl"
    output_path = tmp_path / "metric_dsl_results.jsonl"
    manifest_path = tmp_path / "metric_dsl_results.manifest.json"
    _write_jsonl(
        input_path,
        [
            {
                "id": "metric-1",
                "predicted_dsl": "MEASURE(revenue) BY customer_country",
                "gold_dsl": "MEASURE(revenue) BY customer_country",
                "semantic_model": SEMANTIC_MODEL,
                "reference_sql": (
                    "SELECT customers.country AS customer_country, SUM(orders.amount) AS revenue "
                    "FROM orders JOIN customers ON orders.customer_id = customers.id "
                    "GROUP BY customers.country"
                ),
                "database_path": str(database_path),
            }
        ],
    )

    exit_code = run_metric_dsl_eval(
        input_path=input_path,
        output_path=output_path,
        manifest_output=manifest_path,
        model_name="metric-dsl-model",
        command=["python", "-m", "eval.metric_dsl_eval"],
    )

    assert exit_code == 0
    result = json.loads(output_path.read_text().splitlines()[0])
    assert result["evaluation_mode"] == "metric_dsl"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["benchmark"] == "metric_dsl"
    assert manifest["model_name"] == "metric-dsl-model"
    assert manifest["evaluation_mode"] == "metric_dsl"
    assert manifest["row_count"] == 1
    assert manifest["input_sha256"]
    assert manifest["output_sha256"]
    assert manifest["metrics"]["measure_preservation"] == 1.0
    assert manifest["metrics"]["semantic_model_sha256s"]
    assert manifest["metrics"]["semantic_model_sources"] == {"inline_unversioned": 1}
    assert manifest["metrics"]["semantic_model_oracle_derived_rows"] == 0
    assert manifest["metrics"]["compiled_sql_execution_evaluated_rows"] == 1
    assert manifest["oracle_allowed"] is False


def test_run_metric_dsl_eval_keeps_non_oracle_fixture_source_non_oracle(tmp_path) -> None:
    input_path = tmp_path / "metric_dsl_predictions.jsonl"
    output_path = tmp_path / "metric_dsl_results.jsonl"
    manifest_path = tmp_path / "metric_dsl_results.manifest.json"
    _write_jsonl(
        input_path,
        [
            {
                "id": "metric-non-oracle",
                "predicted_dsl": "MEASURE(revenue)",
                "gold_dsl": "MEASURE(revenue)",
                "semantic_model": SEMANTIC_MODEL,
                "semantic_model_source": "synthetic_non_oracle_fixture",
            }
        ],
    )

    exit_code = run_metric_dsl_eval(
        input_path=input_path,
        output_path=output_path,
        manifest_output=manifest_path,
        model_name="metric-dsl-model",
        command=["python", "-m", "eval.metric_dsl_eval"],
    )

    assert exit_code == 0
    manifest = json.loads(manifest_path.read_text())
    assert manifest["oracle_allowed"] is False
    assert manifest["metrics"]["semantic_model_oracle_derived_rows"] == 0


def test_run_metric_dsl_eval_marks_oracle_derived_semantic_model_manifest(tmp_path) -> None:
    input_path = tmp_path / "metric_dsl_predictions.jsonl"
    output_path = tmp_path / "metric_dsl_results.jsonl"
    manifest_path = tmp_path / "metric_dsl_results.manifest.json"
    _write_jsonl(
        input_path,
        [
            {
                "id": "metric-oracle",
                "predicted_dsl": "MEASURE(revenue)",
                "gold_dsl": "MEASURE(revenue)",
                "semantic_model": SEMANTIC_MODEL,
                "semantic_model_source": "gold_reference_sql_pruned_semantic_model",
                "semantic_context_pruned_by_oracle_labels": True,
            }
        ],
    )

    exit_code = run_metric_dsl_eval(
        input_path=input_path,
        output_path=output_path,
        manifest_output=manifest_path,
        model_name="metric-dsl-model",
        command=["python", "-m", "eval.metric_dsl_eval"],
    )

    assert exit_code == 0
    manifest = json.loads(manifest_path.read_text())
    assert manifest["oracle_allowed"] is True
    assert manifest["metrics"]["semantic_model_oracle_derived_rows"] == 1
