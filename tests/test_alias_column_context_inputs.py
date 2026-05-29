from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from data.alias_column_context_inputs import (
    add_alias_column_context,
    build_alias_column_context_inputs,
    introspect_column_role_context,
    write_alias_column_context_input_artifacts,
)
from eval.result_manifest import sha256_file


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")


def _make_database(root: Path) -> Path:
    db_dir = root / "store"
    db_dir.mkdir(parents=True)
    db_path = db_dir / "store.sqlite"
    connection = sqlite3.connect(db_path)
    connection.executescript(
        """
        CREATE TABLE customers (
          id INTEGER PRIMARY KEY,
          name TEXT,
          country_code TEXT
        );
        CREATE TABLE orders (
          id INTEGER PRIMARY KEY,
          customer_id INTEGER,
          amount REAL,
          FOREIGN KEY(customer_id) REFERENCES customers(id)
        );
        """
    )
    connection.close()
    return db_path


def _prepared_record() -> dict:
    return {
        "id": "dialog-a",
        "database_id": "store",
        "source": "unit",
        "evaluation_mode": "non_oracle_generation",
        "uses_oracle_planning_hints": False,
        "semantic_context_pruned_by_oracle_labels": False,
        "gold_plans": [{"parseable": True}],
        "messages": [
            {"role": "system", "content": "sql"},
            {"role": "user", "content": "Show revenue by customer."},
            {"role": "assistant", "content": "SELECT customers.name FROM customers"},
        ],
    }


def test_introspect_column_role_context_reads_columns_and_foreign_keys(tmp_path) -> None:
    db_path = _make_database(tmp_path)

    context = introspect_column_role_context(db_path)

    assert context["source"] == "sqlite_schema_introspection"
    assert context["allowed_columns"]["customers"] == ["id", "name", "country_code"]
    assert context["allowed_columns"]["orders"] == ["id", "customer_id", "amount"]
    assert context["join_keys"] == ["orders.customer_id = customers.id"]


def test_add_alias_column_context_uses_schema_not_gold_labels(tmp_path) -> None:
    db_path = _make_database(tmp_path)
    context = introspect_column_role_context(db_path)

    updated, summary = add_alias_column_context(
        _prepared_record(),
        column_context_by_database={"store": context},
    )

    prompt = updated["messages"][1]["content"]
    assert "Column-role constraints" in prompt
    assert "orders.customer_id = customers.id" in prompt
    assert "gold" not in updated["alias_column_context"]["leakage_boundary"]
    assert summary["column_count"] == 6


def test_build_alias_column_context_inputs_rejects_oracle_rows(tmp_path) -> None:
    _make_database(tmp_path)
    record = _prepared_record()
    record["uses_oracle_planning_hints"] = True

    with pytest.raises(ValueError, match="non-oracle"):
        build_alias_column_context_inputs(
            prepared_rows=[record],
            database_root=tmp_path,
        )


def test_write_alias_column_context_input_artifacts(tmp_path) -> None:
    _make_database(tmp_path)
    input_path = tmp_path / "prepared.jsonl"
    output_path = tmp_path / "alias_column.jsonl"
    summary_path = tmp_path / "summary.json"
    manifest_path = tmp_path / "manifest.json"
    _write_jsonl(input_path, [_prepared_record()])

    manifest = write_alias_column_context_input_artifacts(
        input_path=input_path,
        database_root=tmp_path,
        output_path=output_path,
        summary_path=summary_path,
        manifest_path=manifest_path,
        command=["unit"],
    )

    output_rows = [json.loads(line) for line in output_path.read_text().splitlines()]
    summary = json.loads(summary_path.read_text())
    assert output_rows[0]["alias_column_context"]["source"] == "sqlite_schema_introspection"
    assert summary["oracle_policy"] == "non_oracle_schema_introspection_only"
    assert manifest["output_sha256"] == sha256_file(output_path)
    assert manifest["summary_sha256"] == sha256_file(summary_path)
