from __future__ import annotations

import csv
import json

from eval.classify_errors import classify_failure, classify_file, extract_sql_features


def test_extract_sql_features_collects_tables_columns_and_clauses() -> None:
    features = extract_sql_features(
        "SELECT c.name, COUNT(*) FROM customers c JOIN orders o ON c.id = o.customer_id "
        "WHERE o.status = 'paid' GROUP BY c.name ORDER BY COUNT(*) DESC LIMIT 1"
    )

    assert features.parseable
    assert features.tables == {"customers", "orders"}
    assert "customers.name" in features.columns
    assert "orders.status" in features.columns
    assert features.aggregations == {"count(*)"}
    assert features.order_sql is not None
    assert features.limit_sql is not None
    assert features.literals == {"'paid'", "1"}
    assert not features.distinct


def test_extract_sql_features_repairs_cosql_spaced_comparison_operator() -> None:
    features = extract_sql_features(
        "SELECT name FROM singer GROUP BY name HAVING count(*) > = 3"
    )

    assert features.parseable
    assert features.having_sql == "having count(*) >= 3"


def test_classify_failure_marks_correct_rows() -> None:
    primary, secondary = classify_failure({"value_execution_score": 1.0})

    assert primary == "correct"
    assert secondary == []


def test_classify_failure_detects_schema_link_mismatch() -> None:
    primary, secondary = classify_failure(
        {
            "value_execution_score": 0.0,
            "syntax_valid": True,
            "turn_index": 0,
            "reference_sql": "SELECT name FROM singer;",
            "generated_sql": "SELECT name FROM concert;",
        }
    )

    assert primary == "schema_link"
    assert secondary == []


def test_classify_failure_detects_join_path_mismatch() -> None:
    primary, secondary = classify_failure(
        {
            "value_execution_score": 0.0,
            "syntax_valid": True,
            "turn_index": 0,
            "reference_sql": "SELECT c.name FROM customers c JOIN orders o ON c.id = o.customer_id;",
            "generated_sql": "SELECT c.name FROM customers c JOIN orders o ON c.id = o.id;",
        }
    )

    assert primary == "join_path"
    assert secondary == []


def test_classify_failure_detects_ordering_limit_mismatch() -> None:
    primary, secondary = classify_failure(
        {
            "value_execution_score": 0.0,
            "syntax_valid": True,
            "turn_index": 0,
            "reference_sql": "SELECT name FROM singer ORDER BY age DESC LIMIT 1;",
            "generated_sql": "SELECT name FROM singer LIMIT 1;",
        }
    )

    assert primary == "ordering_limit"
    assert secondary == []


def test_classify_failure_detects_projection_order_mismatch() -> None:
    primary, secondary = classify_failure(
        {
            "value_execution_score": 0.0,
            "syntax_valid": True,
            "turn_index": 0,
            "reference_sql": "SELECT COUNT(*), district FROM city GROUP BY district;",
            "generated_sql": "SELECT district, COUNT(*) FROM city GROUP BY district;",
        }
    )

    assert primary == "projection"
    assert secondary == []


def test_classify_failure_detects_distinct_grain_mismatch() -> None:
    primary, secondary = classify_failure(
        {
            "value_execution_score": 0.0,
            "syntax_valid": True,
            "turn_index": 0,
            "reference_sql": "SELECT district FROM city;",
            "generated_sql": "SELECT DISTINCT district FROM city;",
        }
    )

    assert primary == "grain_fanout"
    assert secondary == []


def test_classify_failure_detects_nested_distinct_grain_mismatch() -> None:
    primary, secondary = classify_failure(
        {
            "value_execution_score": 0.0,
            "syntax_valid": True,
            "turn_index": 0,
            "reference_sql": "SELECT count(*) FROM (SELECT DISTINCT treatment_type_code FROM treatments);",
            "generated_sql": "SELECT count(*) FROM treatments;",
        }
    )

    assert primary == "grain_fanout"
    assert secondary == []


def test_classify_failure_detects_distinct_count_on_wrong_column() -> None:
    primary, secondary = classify_failure(
        {
            "value_execution_score": 0.0,
            "syntax_valid": True,
            "turn_index": 0,
            "reference_sql": "SELECT count(*) FROM (SELECT DISTINCT treatment_type_code FROM treatments);",
            "generated_sql": "SELECT DISTINCT count(*) FROM treatments;",
        }
    )

    assert primary == "grain_fanout"
    assert secondary == []


def test_classify_failure_adds_history_resolution_for_followup_mismatch() -> None:
    primary, secondary = classify_failure(
        {
            "value_execution_score": 0.0,
            "syntax_valid": True,
            "turn_index": 2,
            "reference_sql": "SELECT COUNT(*) FROM singer WHERE age > 30;",
            "generated_sql": "SELECT COUNT(*) FROM singer;",
        }
    )

    assert primary == "value_grounding"
    assert "history_resolution" in secondary


def test_classify_failure_detects_case_sensitive_literal_mismatch() -> None:
    primary, secondary = classify_failure(
        {
            "value_execution_score": 0.0,
            "syntax_valid": True,
            "turn_index": 0,
            "reference_sql": 'SELECT name FROM country WHERE GovernmentForm = "US Territory";',
            "generated_sql": 'SELECT name FROM country WHERE GovernmentForm = "US territory";',
        }
    )

    assert primary == "value_grounding"
    assert secondary == []


def test_classify_file_writes_jsonl_and_summary(tmp_path) -> None:
    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "classified.jsonl"
    summary_path = tmp_path / "summary.csv"
    input_path.write_text(
        json.dumps(
            {
                "model_name": "model",
                "prompt_variant": "variant",
                "value_execution_score": 0.0,
                "strict_execution_score": 0.0,
                "syntax_valid": True,
                "turn_index": 0,
                "reference_sql": "SELECT name FROM singer;",
                "generated_sql": "SELECT name FROM concert;",
            }
        )
        + "\n"
    )

    assert classify_file(input_path, output_path, summary_path) == 1
    row = json.loads(output_path.read_text())
    summary = list(csv.DictReader(summary_path.open()))

    assert row["error_primary"] == "schema_link"
    assert row["missing_tables"] == ["singer"]
    assert summary[0]["error_primary"] == "schema_link"
