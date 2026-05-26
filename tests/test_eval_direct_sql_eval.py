from __future__ import annotations

import json
from pathlib import Path

from eval.direct_sql_eval import evaluate_direct_sql_rows, run_direct_sql_eval


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")


def test_evaluate_direct_sql_rows_scores_synthetic_fixture_execution(tmp_path: Path) -> None:
    rows = [
        {
            "id": "metric-1",
            "fixture_id": "measure_preservation_metric",
            "generated_sql": (
                "SELECT customers.country_code, SUM(orders.amount) AS revenue "
                "FROM orders JOIN customers ON orders.customer_id = customers.id "
                "GROUP BY customers.country_code ORDER BY revenue DESC"
            ),
            "reference_sql": (
                "SELECT customers.country_code, SUM(orders.amount) AS revenue "
                "FROM orders JOIN customers ON orders.customer_id = customers.id "
                "GROUP BY customers.country_code ORDER BY revenue DESC"
            ),
        }
    ]

    [result] = evaluate_direct_sql_rows(rows, fixtures_path=None, working_dir=tmp_path)

    assert result["fixture_id"] == "measure_preservation_metric"
    assert result["sql_execution_attempted"] is True
    assert result["value_execution_score"] == 1.0
    assert result["strict_execution_score"] == 1.0
    assert result["syntax_valid"] is True
    assert result["database_path"]


def test_run_direct_sql_eval_writes_results_and_manifest(tmp_path: Path) -> None:
    input_path = tmp_path / "direct_sql_predictions.jsonl"
    output_path = tmp_path / "direct_sql_results.jsonl"
    manifest_path = tmp_path / "direct_sql_results.manifest.json"
    _write_jsonl(
        input_path,
        [
            {
                "id": "metric-1",
                "fixture_id": "grain_fanout_bridge",
                "generated_sql": (
                    "SELECT campaigns.name, SUM(deduped.amount) AS revenue "
                    "FROM campaigns "
                    "JOIN ("
                    "  SELECT DISTINCT order_promotions.order_id, order_promotions.campaign_id, orders.amount "
                    "  FROM order_promotions JOIN orders ON order_promotions.order_id = orders.id"
                    ") AS deduped ON campaigns.id = deduped.campaign_id "
                    "GROUP BY campaigns.name ORDER BY revenue DESC"
                ),
                "reference_sql": (
                    "SELECT campaigns.name, SUM(deduped.amount) AS revenue "
                    "FROM campaigns "
                    "JOIN ("
                    "  SELECT DISTINCT order_promotions.order_id, order_promotions.campaign_id, orders.amount "
                    "  FROM order_promotions JOIN orders ON order_promotions.order_id = orders.id"
                    ") AS deduped ON campaigns.id = deduped.campaign_id "
                    "GROUP BY campaigns.name ORDER BY revenue DESC"
                ),
                "evaluation_mode": "non_oracle_generation",
            }
        ],
    )

    exit_code = run_direct_sql_eval(
        input_path=input_path,
        output_path=output_path,
        manifest_output=manifest_path,
        model_name="local-9b",
        fixtures_path=None,
        working_dir=tmp_path / "scratch",
        command=["python", "-m", "eval.direct_sql_eval"],
    )

    manifest = json.loads(manifest_path.read_text())
    written_rows = [json.loads(line) for line in output_path.read_text().splitlines()]

    assert exit_code == 0
    assert len(written_rows) == 1
    assert manifest["benchmark"] == "metric_dsl_direct_sql"
    assert manifest["evaluation_mode"] == "non_oracle_generation"
    assert manifest["metrics"]["value_execution_accuracy"] == 1.0
    assert manifest["metrics"]["strict_execution_accuracy"] == 1.0
    assert manifest["metrics"]["execution_evaluated_rows"] == 1


def test_evaluate_direct_sql_rows_scores_recovery_success_for_repair_fixture(tmp_path: Path) -> None:
    reference_sql = (
        "SELECT customers.name, SUM(orders.amount) AS revenue "
        "FROM orders JOIN customers ON orders.customer_id = customers.id "
        "WHERE customers.country_code = 'FR' GROUP BY customers.name "
        "ORDER BY revenue DESC LIMIT 1"
    )
    rows = [
        {
            "id": "recovery-1",
            "fixture_id": "recovery_empty_result",
            "generated_sql": reference_sql,
            "reference_sql": reference_sql,
        }
    ]

    [result] = evaluate_direct_sql_rows(rows, fixtures_path=None, working_dir=tmp_path)

    assert result["recovery_required"] is True
    assert result["recovery_success"] is True


def test_run_direct_sql_eval_writes_recovery_metrics_for_behavior_recovery_benchmark(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "recovery_predictions.jsonl"
    output_path = tmp_path / "recovery_results.jsonl"
    manifest_path = tmp_path / "recovery_results.manifest.json"
    reference_sql = (
        "SELECT customers.name, SUM(orders.amount) AS revenue "
        "FROM orders JOIN customers ON orders.customer_id = customers.id "
        "WHERE customers.country_code = 'FR' GROUP BY customers.name "
        "ORDER BY revenue DESC LIMIT 1"
    )
    _write_jsonl(
        input_path,
        [
            {
                "id": "recovery-1",
                "fixture_id": "recovery_empty_result",
                "generated_sql": reference_sql,
                "reference_sql": reference_sql,
                "evaluation_mode": "non_oracle_generation",
            }
        ],
    )

    exit_code = run_direct_sql_eval(
        input_path=input_path,
        output_path=output_path,
        manifest_output=manifest_path,
        model_name="local-9b",
        fixtures_path=None,
        working_dir=tmp_path / "scratch",
        benchmark="behavior_recovery",
        command=["python", "-m", "eval.direct_sql_eval"],
    )

    manifest = json.loads(manifest_path.read_text())

    assert exit_code == 0
    assert manifest["benchmark"] == "behavior_recovery"
    assert manifest["metrics"]["recovery_evaluated_rows"] == 1
    assert manifest["metrics"]["recovery_success_rate"] == 1.0
