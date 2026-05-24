from __future__ import annotations

import json
import sqlite3

from eval.rescore_results import rescore_file


def test_rescore_file_preserves_original_score_and_adds_value_score(tmp_path) -> None:
    database = tmp_path / "test.sqlite"
    with sqlite3.connect(database) as conn:
        conn.execute("CREATE TABLE users (id INTEGER)")
        conn.execute("INSERT INTO users VALUES (1), (2)")

    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "output.jsonl"
    input_path.write_text(
        json.dumps(
            {
                "reference_sql": "SELECT count(*) FROM users;",
                "generated_sql": "SELECT count(*) AS total FROM users;",
                "database_path": str(database),
                "execution_score": 0.0,
            }
        )
        + "\n"
    )

    assert rescore_file(input_path, output_path) == 1
    row = json.loads(output_path.read_text())

    assert row["original_execution_score"] == 0.0
    assert row["execution_score"] == 1.0
    assert row["value_execution_score"] == 1.0
    assert row["strict_execution_score"] == 0.0
