from __future__ import annotations

import json

from eval.alias_column_validity import score_alias_column_validity


def _write_jsonl(path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")


def _input_row() -> dict:
    return {
        "dialog_id": "d1",
        "expected_column_validity": {
            "invalid_patterns_to_avoid": ["customers.customer_id"],
        },
        "messages": [
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "allowed_columns": {
                            "customers": ["id", "name", "country_code"],
                            "orders": ["id", "customer_id", "amount"],
                        },
                        "artifact_type": "synthetic_column_role_constraints",
                    }
                ),
            }
        ],
    }


def _bullet_input_row() -> dict:
    return {
        "dialog_id": "d1",
        "expected_column_validity": {
            "invalid_patterns_to_avoid": ["customers.customer_id"],
        },
        "messages": [
            {
                "role": "user",
                "content": (
                    "Column-role constraints (non-oracle; schema introspection only):\n"
                    "Allowed columns:\n"
                    "- customers: id, name, country_code\n"
                    "- orders: id, customer_id, amount\n"
                    "Join keys:\n"
                    "- orders.customer_id = customers.id"
                ),
            }
        ],
    }


def test_score_alias_column_validity_detects_invalid_customer_column(tmp_path) -> None:
    input_path = tmp_path / "input.jsonl"
    rollout_path = tmp_path / "rollout.jsonl"
    _write_jsonl(input_path, [_input_row()])
    _write_jsonl(
        rollout_path,
        [
            {
                "id": "d1:1",
                "dialog_id": "d1",
                "generated_sql": (
                    "SELECT customers.name FROM customers "
                    "GROUP BY customers.customer_id, customers.name"
                ),
            }
        ],
    )

    payload = score_alias_column_validity(
        input_path=input_path,
        rollout_output_path=rollout_path,
    )

    assert payload["scored_row_count"] == 1
    assert payload["column_validity_accuracy"] == 0.0
    assert payload["schema_valid_sql_rate"] == 0.0
    assert payload["rows"][0]["invalid_table_column_refs"] == [
        {
            "sql": "customers.customer_id",
            "qualifier": "customers",
            "table": "customers",
            "column": "customer_id",
            "select_alias_ref": False,
        }
    ]
    assert payload["rows"][0]["matched_invalid_patterns"] == ["customers.customer_id"]


def test_score_alias_column_validity_accepts_allowed_columns(tmp_path) -> None:
    input_path = tmp_path / "input.jsonl"
    rollout_path = tmp_path / "rollout.jsonl"
    _write_jsonl(input_path, [_input_row()])
    _write_jsonl(
        rollout_path,
        [
            {
                "id": "d1:1",
                "dialog_id": "d1",
                "generated_sql": (
                    "SELECT customers.name FROM orders "
                    "JOIN customers ON orders.customer_id = customers.id "
                    "GROUP BY customers.name"
                ),
            }
        ],
    )

    payload = score_alias_column_validity(
        input_path=input_path,
        rollout_output_path=rollout_path,
    )

    assert payload["column_validity_accuracy"] == 1.0
    assert payload["schema_valid_sql_rate"] == 1.0
    assert payload["alias_resolution_success_rate"] == 1.0
    assert payload["rows"][0]["column_validity_match"] is True


def test_score_alias_column_validity_resolves_table_aliases(tmp_path) -> None:
    input_path = tmp_path / "input.jsonl"
    rollout_path = tmp_path / "rollout.jsonl"
    _write_jsonl(input_path, [_input_row()])
    _write_jsonl(
        rollout_path,
        [
            {
                "id": "d1:1",
                "dialog_id": "d1",
                "generated_sql": (
                    "SELECT c.name FROM orders AS o "
                    "JOIN customers AS c ON o.customer_id = c.id "
                    "GROUP BY c.name"
                ),
            }
        ],
    )

    payload = score_alias_column_validity(
        input_path=input_path,
        rollout_output_path=rollout_path,
    )

    assert payload["column_validity_accuracy"] == 1.0
    assert payload["rows"][0]["table_column_refs"][0] == {
        "sql": "c.name",
        "qualifier": "c",
        "table": "customers",
        "column": "name",
        "select_alias_ref": False,
    }


def test_score_alias_column_validity_allows_order_by_select_alias(tmp_path) -> None:
    input_path = tmp_path / "input.jsonl"
    rollout_path = tmp_path / "rollout.jsonl"
    _write_jsonl(input_path, [_input_row()])
    _write_jsonl(
        rollout_path,
        [
            {
                "id": "d1:1",
                "dialog_id": "d1",
                "generated_sql": (
                    "SELECT customers.name, SUM(orders.amount) AS revenue "
                    "FROM orders JOIN customers ON orders.customer_id = customers.id "
                    "GROUP BY customers.name ORDER BY revenue DESC"
                ),
            }
        ],
    )

    payload = score_alias_column_validity(
        input_path=input_path,
        rollout_output_path=rollout_path,
    )

    assert payload["column_validity_accuracy"] == 1.0
    assert payload["rows"][0]["table_column_refs"][-1] == {
        "sql": "revenue",
        "qualifier": "",
        "table": "",
        "column": "revenue",
        "select_alias_ref": True,
    }


def test_score_alias_column_validity_reads_bullet_allowed_columns(tmp_path) -> None:
    input_path = tmp_path / "input.jsonl"
    rollout_path = tmp_path / "rollout.jsonl"
    _write_jsonl(input_path, [_bullet_input_row()])
    _write_jsonl(
        rollout_path,
        [
            {
                "id": "d1:1",
                "dialog_id": "d1",
                "generated_sql": (
                    "SELECT c.name FROM orders o "
                    "JOIN customers c ON o.customer_id = c.id "
                    "GROUP BY c.name"
                ),
            }
        ],
    )

    payload = score_alias_column_validity(
        input_path=input_path,
        rollout_output_path=rollout_path,
    )

    assert payload["column_validity_accuracy"] == 1.0


def test_score_alias_column_validity_accepts_unambiguous_unqualified_column(tmp_path) -> None:
    input_path = tmp_path / "input.jsonl"
    rollout_path = tmp_path / "rollout.jsonl"
    _write_jsonl(input_path, [_bullet_input_row()])
    _write_jsonl(
        rollout_path,
        [
            {
                "id": "d1:1",
                "dialog_id": "d1",
                "generated_sql": "SELECT name FROM customers GROUP BY name",
            }
        ],
    )

    payload = score_alias_column_validity(
        input_path=input_path,
        rollout_output_path=rollout_path,
    )

    assert payload["column_validity_accuracy"] == 1.0


def test_score_alias_column_validity_rejects_ambiguous_unqualified_column(tmp_path) -> None:
    input_path = tmp_path / "input.jsonl"
    rollout_path = tmp_path / "rollout.jsonl"
    row = _bullet_input_row()
    row["messages"][0]["content"] += "\n- archived_orders: id, customer_id, amount"
    _write_jsonl(input_path, [row])
    _write_jsonl(
        rollout_path,
        [
            {
                "id": "d1:1",
                "dialog_id": "d1",
                "generated_sql": (
                    "SELECT customer_id FROM orders "
                    "JOIN archived_orders ON orders.id = archived_orders.id"
                ),
            }
        ],
    )

    payload = score_alias_column_validity(
        input_path=input_path,
        rollout_output_path=rollout_path,
    )

    assert payload["column_validity_accuracy"] == 0.0
