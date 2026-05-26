"""Build a non-oracle value index from database contents.

The index is created from SQLite values that are visible in the database, not
from reference SQL. Optional label coverage can be computed afterwards to show
where the database value exists but the user-facing mention still needs an alias
or entity-resolution artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
import unicodedata
from pathlib import Path
from typing import Any

from eval.result_manifest import sha256_file


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def normalize_value_token(value: Any) -> str:
    """Normalize a database value or user mention for alias matching."""

    text = unicodedata.normalize("NFKD", str(value))
    text = text.encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^A-Za-z0-9]+", " ", text.lower()).strip()
    return re.sub(r"\s+", " ", text)


def _quote_identifier(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _is_indexable_column(column_type: str) -> bool:
    declared_type = (column_type or "").upper()
    return "BLOB" not in declared_type


def _aliases(raw_value: Any) -> list[str]:
    raw = str(raw_value).strip()
    normalized = normalize_value_token(raw)
    aliases = [raw]
    if normalized and normalized != raw:
        aliases.append(normalized)
    return aliases


def _entry_id(row: dict[str, Any]) -> str:
    payload = "|".join(
        str(row[key])
        for key in ("database_id", "table", "column", "raw_value", "normalized_value")
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _tables(connection: sqlite3.Connection) -> list[str]:
    rows = connection.execute(
        """
        SELECT name
        FROM sqlite_master
        WHERE type = 'table' AND name NOT LIKE 'sqlite_%'
        ORDER BY name
        """
    ).fetchall()
    return [str(row[0]) for row in rows]


def _table_columns(connection: sqlite3.Connection, table: str) -> list[dict[str, Any]]:
    return [
        {
            "name": str(row[1]),
            "type": str(row[2] or ""),
        }
        for row in connection.execute(f"PRAGMA table_info({_quote_identifier(table)})")
    ]


def build_value_index_for_database(
    *,
    database_id: str,
    db_path: Path,
    max_values_per_column: int = 200,
    max_value_length: int = 120,
) -> list[dict[str, Any]]:
    """Scan one SQLite database and return non-oracle value-index rows."""

    rows: list[dict[str, Any]] = []
    with sqlite3.connect(db_path) as connection:
        for table in _tables(connection):
            for column in _table_columns(connection, table):
                column_name = column["name"]
                if not _is_indexable_column(column["type"]):
                    continue
                query = f"""
                    SELECT CAST({_quote_identifier(column_name)} AS TEXT) AS raw_value,
                           COUNT(*) AS source_frequency
                    FROM {_quote_identifier(table)}
                    WHERE {_quote_identifier(column_name)} IS NOT NULL
                      AND TRIM(CAST({_quote_identifier(column_name)} AS TEXT)) != ''
                      AND LENGTH(CAST({_quote_identifier(column_name)} AS TEXT)) <= ?
                    GROUP BY CAST({_quote_identifier(column_name)} AS TEXT)
                    ORDER BY source_frequency DESC, raw_value ASC
                    LIMIT ?
                """
                for raw_value, source_frequency in connection.execute(
                    query,
                    (max_value_length, max_values_per_column),
                ):
                    value = str(raw_value).strip()
                    normalized = normalize_value_token(value)
                    if not normalized:
                        continue
                    row = {
                        "artifact_type": "non_oracle_value_index_entry",
                        "schema_version": 1,
                        "index_source": "database_contents",
                        "database_id": database_id,
                        "table": table,
                        "column": column_name,
                        "raw_value": value,
                        "normalized_value": normalized,
                        "aliases": _aliases(value),
                        "source_frequency": int(source_frequency),
                    }
                    row["id"] = _entry_id(row)
                    rows.append(row)
    return sorted(rows, key=lambda row: (row["database_id"], row["table"], row["column"], row["raw_value"]))


def summarize_value_index(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "artifact_type": "non_oracle_value_index_summary",
        "index_source": "database_contents",
        "entry_count": len(rows),
        "database_count": len({row["database_id"] for row in rows}),
        "table_count": len({(row["database_id"], row["table"]) for row in rows}),
        "column_count": len(
            {(row["database_id"], row["table"], row["column"]) for row in rows}
        ),
        "alias_count": sum(len(row.get("aliases") or []) for row in rows),
    }


def _column_tail(column: Any) -> str:
    text = str(column or "").strip().strip('"`[]').lower()
    return text.split(".")[-1].strip('"`[]')


def _coverage_keys(index_rows: list[dict[str, Any]]) -> set[tuple[str, str, str]]:
    keys: set[tuple[str, str, str]] = set()
    for row in index_rows:
        database_id = str(row.get("database_id") or "")
        column = _column_tail(row.get("column"))
        for alias in row.get("aliases") or []:
            normalized = normalize_value_token(alias)
            if normalized:
                keys.add((database_id, column, normalized))
    return keys


def _label_payload(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": row.get("id"),
        "database_id": row.get("database_id"),
        "resolved_column": row.get("resolved_column") or row.get("column"),
        "resolved_value": row.get("resolved_value") or row.get("literal_value"),
        "mention_text": row.get("mention_text"),
    }


def evaluate_value_index_coverage(
    index_rows: list[dict[str, Any]],
    label_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Evaluate database-derived index coverage against optional gold labels."""

    keys = _coverage_keys(index_rows)
    resolved_hits = 0
    mention_hits = 0
    missing_resolved_examples: list[dict[str, Any]] = []
    missing_mention_examples: list[dict[str, Any]] = []

    for label in label_rows:
        database_id = str(label.get("database_id") or "")
        column = _column_tail(label.get("resolved_column") or label.get("column"))
        resolved_value = label.get("resolved_value") or label.get("literal_value")
        mention_text = label.get("mention_text")
        resolved_key = (database_id, column, normalize_value_token(resolved_value))
        mention_key = (database_id, column, normalize_value_token(mention_text))

        if resolved_key in keys:
            resolved_hits += 1
        elif len(missing_resolved_examples) < 5:
            missing_resolved_examples.append(_label_payload(label))

        if mention_text and mention_key in keys:
            mention_hits += 1
        elif mention_text and len(missing_mention_examples) < 5:
            missing_mention_examples.append(_label_payload(label))

    label_count = len(label_rows)
    return {
        "label_source": "optional_gold_sql_coverage_eval",
        "label_count": label_count,
        "resolved_value_indexed_count": resolved_hits,
        "resolved_value_indexed_rate": resolved_hits / label_count if label_count else 0.0,
        "mention_alias_indexed_count": mention_hits,
        "mention_alias_indexed_rate": mention_hits / label_count if label_count else 0.0,
        "missing_resolved_value_examples": missing_resolved_examples,
        "missing_mention_alias_examples": missing_mention_examples,
    }


def _database_ids_from_input(records: list[dict[str, Any]]) -> list[str]:
    return sorted(
        {
            str(record.get("database_id") or record.get("db_id"))
            for record in records
            if record.get("database_id") or record.get("db_id")
        }
    )


def _resolve_sqlite_path(database_root: Path, database_id: str) -> Path:
    candidates = [
        database_root / database_id / f"{database_id}.sqlite",
        database_root / f"{database_id}.sqlite",
    ]
    candidates.extend(sorted((database_root / database_id).glob("*.sqlite")))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find SQLite database for {database_id} under {database_root}")


def build_value_index_manifest(
    *,
    input_path: Path,
    database_root: Path,
    output_path: Path,
    summary_path: Path,
    summary: dict[str, Any],
    labels_path: Path | None = None,
) -> dict[str, Any]:
    manifest = {
        "schema_version": 1,
        "artifact_type": "non_oracle_value_index_v1",
        "index_source": "database_contents",
        "input_path": str(input_path),
        "input_sha256": sha256_file(input_path),
        "database_root": str(database_root),
        "output_path": str(output_path),
        "output_sha256": sha256_file(output_path),
        "summary_path": str(summary_path),
        "summary_sha256": sha256_file(summary_path),
        "row_count": summary["entry_count"],
        "database_count": summary["database_count"],
        "table_count": summary["table_count"],
        "column_count": summary["column_count"],
    }
    if labels_path:
        manifest["label_source"] = "optional_gold_sql_coverage_eval"
        manifest["labels_path"] = str(labels_path)
        manifest["labels_sha256"] = sha256_file(labels_path)
    return manifest


def run_value_index_export(
    *,
    input_path: Path,
    database_root: Path,
    output_path: Path,
    summary_path: Path,
    manifest_path: Path | None = None,
    labels_path: Path | None = None,
    max_values_per_column: int = 200,
    max_value_length: int = 120,
) -> int:
    records = _load_jsonl(input_path)
    rows: list[dict[str, Any]] = []
    for database_id in _database_ids_from_input(records):
        rows.extend(
            build_value_index_for_database(
                database_id=database_id,
                db_path=_resolve_sqlite_path(database_root, database_id),
                max_values_per_column=max_values_per_column,
                max_value_length=max_value_length,
            )
        )
    summary = summarize_value_index(rows)
    if labels_path:
        summary["coverage"] = evaluate_value_index_coverage(rows, _load_jsonl(labels_path))
    _write_jsonl(output_path, rows)
    _write_json(summary_path, summary)
    if manifest_path is None:
        manifest_path = output_path.with_suffix(".manifest.json")
    _write_json(
        manifest_path,
        build_value_index_manifest(
            input_path=input_path,
            database_root=database_root,
            output_path=output_path,
            summary_path=summary_path,
            summary=summary,
            labels_path=labels_path,
        ),
    )
    print(f"Wrote {len(rows)} non-oracle value-index rows to {output_path}")
    print(f"Wrote value-index summary to {summary_path}")
    print(f"Wrote value-index manifest to {manifest_path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/processed/eval_100_each.jsonl"))
    parser.add_argument(
        "--database-root",
        type=Path,
        default=Path("data/raw/cosql_dataset/database"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/data_artifacts/value_index_cosql_dev_100.jsonl"),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("docs/data_artifacts/value_index_cosql_dev_100_summary.json"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("docs/data_artifacts/value_index_cosql_dev_100.manifest.json"),
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=Path("docs/data_artifacts/value_grounding_labels_cosql_dev_100.jsonl"),
    )
    parser.add_argument("--max-values-per-column", type=int, default=200)
    parser.add_argument("--max-value-length", type=int, default=120)
    args = parser.parse_args()
    labels_path = args.labels if args.labels and args.labels.exists() else None
    return run_value_index_export(
        input_path=args.input,
        database_root=args.database_root,
        output_path=args.output,
        summary_path=args.summary,
        manifest_path=args.manifest,
        labels_path=labels_path,
        max_values_per_column=args.max_values_per_column,
        max_value_length=args.max_value_length,
    )


if __name__ == "__main__":
    raise SystemExit(main())
