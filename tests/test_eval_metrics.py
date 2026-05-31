from __future__ import annotations

import sqlite3

from eval.ragas_metrics import (
    clean_sql,
    extract_sql,
    normalize_sql,
    score_multi_turn,
    score_single_turn,
    syntax_valid,
)


def test_extract_sql_from_prose_and_fence() -> None:
    assert extract_sql("Here is the query:\n```sql\nSELECT id FROM users;\n```") == (
        "SELECT id FROM users;"
    )
    assert extract_sql("Reasoning first. SELECT name FROM users; extra") == "SELECT name FROM users;"


def test_extract_sql_stops_before_chat_role_continuation() -> None:
    assert (
        extract_sql(
            'SELECT name FROM users WHERE role = "assistant"\n'
            "user\n"
            "Question:\n"
            "List user ids\n"
            "assistant\n"
            "SELECT id FROM users"
        )
        == 'SELECT name FROM users WHERE role = "assistant"'
    )


def test_clean_sql_repairs_split_operators() -> None:
    assert clean_sql("HAVING count ( * ) > = 3 AND x < = 5") == (
        "HAVING count ( * ) >= 3 AND x <= 5"
    )


def test_normalize_sql_matches_case_and_whitespace_variants() -> None:
    assert normalize_sql("select id from users") == normalize_sql("SELECT id FROM users;")


def test_syntax_valid_rejects_invalid_sql() -> None:
    assert syntax_valid("SELECT 1")
    assert not syntax_valid("SELECT FROM")


def test_score_single_turn_without_database_uses_normalized_match() -> None:
    result = score_single_turn("SELECT id FROM users;", "select id from users")

    assert result.syntax_valid
    assert result.normalized_match
    assert result.execution_score == 1.0


def test_score_single_turn_compares_sqlite_results(tmp_path) -> None:
    database = tmp_path / "test.sqlite"
    with sqlite3.connect(database) as conn:
        conn.execute("CREATE TABLE users (id INTEGER, name TEXT)")
        conn.execute("INSERT INTO users VALUES (1, 'Ada'), (2, 'Grace')")

    result = score_single_turn(
        "SELECT id FROM users ORDER BY id;",
        "SELECT id FROM users ORDER BY id;",
        database_path=database,
    )

    assert result.execution_score == 1.0
    assert result.error is None
    assert result.strict_execution_score == 1.0
    assert result.value_execution_score == 1.0


def test_score_single_turn_ignores_harmless_alias_differences(tmp_path) -> None:
    database = tmp_path / "test.sqlite"
    with sqlite3.connect(database) as conn:
        conn.execute("CREATE TABLE users (id INTEGER)")
        conn.execute("INSERT INTO users VALUES (1), (2)")

    result = score_single_turn(
        "SELECT count(*) FROM users;",
        "SELECT count(*) AS total FROM users;",
        database_path=database,
    )

    assert result.execution_score == 1.0
    assert result.value_execution_score == 1.0
    assert result.strict_execution_score == 0.0


def test_score_single_turn_preserves_order_for_topk(tmp_path) -> None:
    database = tmp_path / "test.sqlite"
    with sqlite3.connect(database) as conn:
        conn.execute("CREATE TABLE users (id INTEGER, name TEXT)")
        conn.execute("INSERT INTO users VALUES (1, 'Ada'), (2, 'Grace')")

    result = score_single_turn(
        "SELECT name FROM users ORDER BY id LIMIT 2;",
        "SELECT name FROM users ORDER BY id DESC LIMIT 2;",
        database_path=database,
    )

    assert result.execution_score == 0.0
    assert result.order_sensitive


def test_score_multi_turn_reports_interaction_match() -> None:
    result = score_multi_turn(
        [
            {"reference_sql": "SELECT 1;", "generated_sql": "SELECT 1;"},
            {"reference_sql": "SELECT 2;", "generated_sql": "SELECT 3;"},
        ]
    )

    assert result["turn_count"] == 2
    assert result["execution_accuracy"] == 0.5
    assert result["interaction_match"] is False
