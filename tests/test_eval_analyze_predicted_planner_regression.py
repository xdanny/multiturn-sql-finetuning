from __future__ import annotations

import sqlite3

from eval.analyze_predicted_planner_regression import analyze_pair


def _plan(selected: list[str]) -> dict:
    return {
        "parseable": True,
        "relevant_tables": ["users"],
        "relevant_columns": selected,
        "join_path": [],
        "query_skeleton": {"select": True},
        "projection_shape": {
            "selected_count": len(selected),
            "selected_expressions": selected,
            "preserve_duplicates": True,
        },
    }


def _row(
    *,
    raw_generation: str,
    reference_sql: str,
    database_path: str,
    mode: str,
    predicted_plan: dict | None = None,
) -> dict:
    row = {
        "database_id": "unit",
        "database_path": database_path,
        "evaluation_mode": mode,
        "gold_plan": _plan(["id", "name"]),
        "id": f"{mode}-0",
        "messages": [{"role": "user", "content": "Question:\nList users."}],
        "raw_generation": raw_generation,
        "reference_sql": reference_sql,
    }
    if predicted_plan is not None:
        row["predicted_plan"] = predicted_plan
    return row


def test_analyze_pair_rescores_chat_continuations_and_flags_projection_flip(tmp_path) -> None:
    database = tmp_path / "unit.sqlite"
    with sqlite3.connect(database) as conn:
        conn.execute("CREATE TABLE users (id INTEGER, name TEXT)")
        conn.execute("INSERT INTO users VALUES (1, 'Ada')")

    direct_rows = [
        _row(
            raw_generation="SELECT id, name FROM users\nuser\nQuestion:\nnext",
            reference_sql="SELECT id, name FROM users",
            database_path=str(database),
            mode="non_oracle_generation",
        )
    ]
    predicted_rows = [
        _row(
            raw_generation="SELECT name, id FROM users\nassistant\nSELECT id FROM users",
            reference_sql="SELECT id, name FROM users",
            database_path=str(database),
            mode="predicted_planner",
            predicted_plan=_plan(["name", "id"]),
        )
    ]

    analysis = analyze_pair(direct_rows=direct_rows, predicted_rows=predicted_rows)

    assert analysis["metrics"]["bucket_counts"] == {"direct_only_regression": 1}
    assert analysis["metrics"]["direct_sql_value_execution_accuracy"] == 1.0
    assert analysis["metrics"]["predicted_planner_value_execution_accuracy"] == 0.0
    assert analysis["metrics"]["regression_cause_counts"] == {"projection_order_flip": 1}
    assert analysis["examples"][0]["predicted_sql"] == "SELECT name, id FROM users"
