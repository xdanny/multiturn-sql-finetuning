from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from data.value_index import (
    build_value_index_for_database,
    evaluate_value_index_coverage,
    normalize_value_token,
    run_value_index_export,
    summarize_value_index,
)
from eval.result_manifest import sha256_file


def _sqlite_db(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            CREATE TABLE customers (
              id INTEGER PRIMARY KEY,
              country_code TEXT,
              customer_name TEXT,
              notes TEXT
            );
            INSERT INTO customers(country_code, customer_name, notes)
            VALUES
              ('FR', 'Alice Smith', 'prefers email'),
              ('US', 'Bob Jones', 'prefers phone'),
              ('FR', 'Alice Smith', 'repeat buyer'),
              ('', 'No Country', NULL);
            """
        )
    return path


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")


def test_normalize_value_token_is_alias_safe() -> None:
    assert normalize_value_token("  Alice-Smith, Jr. ") == "alice smith jr"
    assert normalize_value_token("São Paulo") == "sao paulo"


def test_build_value_index_for_database_scans_text_values_without_gold_sql(tmp_path) -> None:
    db_path = _sqlite_db(tmp_path / "store.sqlite")

    rows = build_value_index_for_database(
        database_id="store",
        db_path=db_path,
        max_values_per_column=2,
    )

    by_value = {(row["table"], row["column"], row["raw_value"]): row for row in rows}
    country = by_value[("customers", "country_code", "FR")]
    assert country["artifact_type"] == "non_oracle_value_index_entry"
    assert country["index_source"] == "database_contents"
    assert country["normalized_value"] == "fr"
    assert country["aliases"] == ["FR", "fr"]
    assert country["source_frequency"] == 2

    assert ("customers", "country_code", "") not in by_value
    assert len([row for row in rows if row["column"] == "country_code"]) == 2


def test_summarize_value_index_counts_scope() -> None:
    rows = [
        {
            "database_id": "store",
            "table": "customers",
            "column": "country_code",
            "raw_value": "FR",
            "aliases": ["FR", "fr"],
        },
        {
            "database_id": "store",
            "table": "customers",
            "column": "customer_name",
            "raw_value": "Alice Smith",
            "aliases": ["Alice Smith", "alice smith"],
        },
    ]

    summary = summarize_value_index(rows)

    assert summary["artifact_type"] == "non_oracle_value_index_summary"
    assert summary["index_source"] == "database_contents"
    assert summary["entry_count"] == 2
    assert summary["database_count"] == 1
    assert summary["column_count"] == 2
    assert summary["alias_count"] == 4


def test_evaluate_value_index_coverage_separates_resolved_values_from_user_aliases() -> None:
    index_rows = [
        {
            "database_id": "store",
            "table": "customers",
            "column": "country_code",
            "raw_value": "FR",
            "aliases": ["FR", "fr"],
        },
        {
            "database_id": "store",
            "table": "customers",
            "column": "customer_name",
            "raw_value": "Alice Smith",
            "aliases": ["Alice Smith", "alice smith"],
        },
    ]
    label_rows = [
        {
            "id": "turn:0",
            "database_id": "store",
            "resolved_table": "customers",
            "resolved_column": "customers.country_code",
            "resolved_value": "FR",
            "mention_text": "France",
        },
        {
            "id": "turn:1",
            "database_id": "store",
            "resolved_table": "customers",
            "resolved_column": "customer_name",
            "resolved_value": "Alice Smith",
            "mention_text": "Alice Smith",
        },
    ]

    summary = evaluate_value_index_coverage(index_rows, label_rows)

    assert summary["label_count"] == 2
    assert summary["resolved_value_indexed_count"] == 2
    assert summary["resolved_value_indexed_rate"] == 1.0
    assert summary["mention_alias_indexed_count"] == 1
    assert summary["mention_alias_indexed_rate"] == 0.5
    assert summary["missing_mention_alias_examples"] == [
        {
            "id": "turn:0",
            "database_id": "store",
            "resolved_table": "customers",
            "resolved_column": "customers.country_code",
            "resolved_value": "FR",
            "mention_text": "France",
        }
    ]


def test_run_value_index_export_writes_index_summary_and_manifest(tmp_path) -> None:
    database_root = tmp_path / "database"
    db_dir = database_root / "store"
    db_path = _sqlite_db(db_dir / "store.sqlite")
    assert db_path.exists()
    input_path = tmp_path / "prepared.jsonl"
    labels_path = tmp_path / "labels.jsonl"
    output_path = tmp_path / "value_index.jsonl"
    summary_path = tmp_path / "value_index_summary.json"
    manifest_path = tmp_path / "value_index.manifest.json"
    _write_jsonl(input_path, [{"database_id": "store", "messages": []}])
    _write_jsonl(
        labels_path,
        [
            {
                "id": "turn:0",
                "database_id": "store",
                "resolved_table": "customers",
                "resolved_column": "customers.country_code",
                "resolved_value": "FR",
                "mention_text": "France",
            }
        ],
    )
    labels_manifest_path = labels_path.with_suffix(".manifest.json")
    labels_manifest_path.write_text(
        json.dumps(
            {
                "input_path": str(input_path),
                "input_sha256": sha256_file(input_path),
            }
        )
        + "\n"
    )

    exit_code = run_value_index_export(
        input_path=input_path,
        database_root=database_root,
        output_path=output_path,
        summary_path=summary_path,
        manifest_path=manifest_path,
        labels_path=labels_path,
        max_values_per_column=10,
    )

    assert exit_code == 0
    rows = [json.loads(line) for line in output_path.read_text().splitlines()]
    summary = json.loads(summary_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    assert {row["database_id"] for row in rows} == {"store"}
    assert summary["entry_count"] > 0
    assert summary["coverage"]["resolved_value_indexed_count"] == 1
    assert summary["coverage"]["mention_alias_indexed_count"] == 0
    assert manifest["artifact_type"] == "non_oracle_value_index_v1"
    assert manifest["index_source"] == "database_contents"
    assert manifest["label_source"] == "optional_gold_sql_coverage_eval"


def test_value_index_coverage_requires_table_identity() -> None:
    index_rows = [
        {
            "database_id": "store",
            "table": "billing_addresses",
            "column": "country_code",
            "raw_value": "FR",
            "aliases": ["FR", "fr"],
        }
    ]
    label_rows = [
        {
            "id": "turn:0",
            "database_id": "store",
            "resolved_table": "shipping_addresses",
            "resolved_column": "country_code",
            "resolved_value": "FR",
            "mention_text": "FR",
        }
    ]

    summary = evaluate_value_index_coverage(index_rows, label_rows)

    assert summary["resolved_value_indexed_count"] == 0
    assert summary["mention_alias_indexed_count"] == 0
