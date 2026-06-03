"""Build CoSQL prepared inputs with schema-derived column-role context."""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from eval.result_manifest import sha256_file

SCHEMA_VERSION = 1
ARTIFACT_TYPE = "alias_column_context_prepared_inputs"
DEFAULT_INPUT = Path("data/processed/eval_cosql_dev_100.jsonl")
DEFAULT_DATABASE_ROOT = Path("data/raw/cosql_dataset/database")
DEFAULT_OUTPUT = Path("data/processed/eval_cosql_dev_100_alias_column_context.jsonl")
DEFAULT_SUMMARY_OUTPUT = Path("docs/data_artifacts/alias_column_context_inputs_summary.json")
DEFAULT_MANIFEST_OUTPUT = Path("docs/data_artifacts/alias_column_context_inputs.manifest.json")


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _database_path(database_root: Path, database_id: str) -> Path:
    path = database_root / database_id / f"{database_id}.sqlite"
    if not path.exists():
        raise FileNotFoundError(f"database not found for {database_id}: {path}")
    return path


def introspect_column_role_context(database_path: Path) -> dict[str, Any]:
    """Return schema-derived allowed columns and join keys for one SQLite database."""

    connection = sqlite3.connect(database_path)
    try:
        table_rows = connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        ).fetchall()
        table_names = sorted(str(row[0]) for row in table_rows)
        allowed_columns = {}
        join_keys = []
        primary_keys = {}
        for table in table_names:
            columns = connection.execute(f"PRAGMA table_info({json.dumps(table)})").fetchall()
            allowed_columns[table] = [str(column[1]) for column in columns]
            primary_keys[table] = [str(column[1]) for column in columns if int(column[5] or 0)]
            foreign_keys = connection.execute(f"PRAGMA foreign_key_list({json.dumps(table)})").fetchall()
            for foreign_key in foreign_keys:
                join_keys.append(
                    f"{table}.{foreign_key[3]} = {foreign_key[2]}.{foreign_key[4]}"
                )
    finally:
        connection.close()

    return {
        "artifact_type": "column_role_context",
        "source": "sqlite_schema_introspection",
        "allowed_columns": allowed_columns,
        "primary_keys": primary_keys,
        "join_keys": sorted(join_keys),
    }


def _context_block(context: dict[str, Any]) -> str:
    lines = [
        "Column-role constraints (non-oracle; schema introspection only):",
        "Allowed columns:",
    ]
    for table, columns in context["allowed_columns"].items():
        lines.append(f"- {table}: {', '.join(columns)}")
    if context["join_keys"]:
        lines.append("Join keys:")
        for join_key in context["join_keys"]:
            lines.append(f"- {join_key}")
    lines.append(
        "Use only listed physical table.column references; SELECT aliases are allowed only as aliases."
    )
    return "\n".join(lines)


def add_alias_column_context(
    record: dict[str, Any],
    *,
    column_context_by_database: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return one prepared record with schema-derived column-role context."""

    messages = [dict(message) for message in record.get("messages") or []]
    database_id = str(record.get("database_id") or "")
    context = column_context_by_database[database_id]
    block = _context_block(context)
    added = False
    for message in messages:
        if message.get("role") == "user":
            message["content"] = f"{message.get('content')}\n\n{block}"
            added = True
            break
    if not added:
        raise ValueError("prepared record has no user message for alias/column context")

    updated = {
        **record,
        "messages": messages,
        "evaluation_mode": "non_oracle_generation",
        "alias_column_context": {
            "artifact_type": "alias_column_context",
            "source": context["source"],
            "table_count": len(context["allowed_columns"]),
            "column_count": sum(len(columns) for columns in context["allowed_columns"].values()),
            "join_key_count": len(context["join_keys"]),
            "leakage_boundary": "context comes from database schema introspection only",
        },
    }
    summary = {
        "assistant_turn_count": sum(1 for message in messages if message.get("role") == "assistant"),
        "database_id": database_id,
        "table_count": updated["alias_column_context"]["table_count"],
        "column_count": updated["alias_column_context"]["column_count"],
        "join_key_count": updated["alias_column_context"]["join_key_count"],
    }
    return updated, summary


def build_alias_column_context_inputs(
    *,
    prepared_rows: list[dict[str, Any]],
    database_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build prepared rows with schema-derived column-role context."""

    database_ids = sorted({str(row.get("database_id") or "") for row in prepared_rows})
    column_context_by_database = {
        database_id: introspect_column_role_context(_database_path(database_root, database_id))
        for database_id in database_ids
    }
    output_rows = []
    row_summaries = []
    for record in prepared_rows:
        updated, summary = add_alias_column_context(
            record,
            column_context_by_database=column_context_by_database,
        )
        output_rows.append(updated)
        row_summaries.append(summary)

    database_counts = Counter(summary["database_id"] for summary in row_summaries)
    return output_rows, {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "row_count": len(output_rows),
        "assistant_turn_count": sum(summary["assistant_turn_count"] for summary in row_summaries),
        "database_count": len(database_counts),
        "database_row_counts": dict(sorted(database_counts.items())),
        "table_count": sum(summary["table_count"] for summary in row_summaries),
        "column_count": sum(summary["column_count"] for summary in row_summaries),
        "join_key_count": sum(summary["join_key_count"] for summary in row_summaries),
        "leakage_policy": "schema_introspection_only",
        "leakage_boundary": (
            "no reference SQL, gold plans, expected rows, assistant SQL, or future user turns "
            "are used to build column-role context"
        ),
        "evaluation_command": "eval.run_eval or eval.rollout_eval plus eval.alias_column_validity",
    }


def write_alias_column_context_input_artifacts(
    *,
    input_path: Path = DEFAULT_INPUT,
    database_root: Path = DEFAULT_DATABASE_ROOT,
    output_path: Path = DEFAULT_OUTPUT,
    summary_path: Path = DEFAULT_SUMMARY_OUTPUT,
    manifest_path: Path = DEFAULT_MANIFEST_OUTPUT,
    command: list[str] | None = None,
) -> dict[str, Any]:
    """Write alias/column context prepared inputs, summary, and manifest."""

    output_rows, summary = build_alias_column_context_inputs(
        prepared_rows=_load_jsonl(input_path),
        database_root=database_root,
    )
    _write_jsonl(output_path, output_rows)
    _write_json(summary_path, summary)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "input_path": str(input_path),
        "input_sha256": sha256_file(input_path),
        "database_root": str(database_root),
        "output_path": str(output_path),
        "output_sha256": sha256_file(output_path),
        "summary_path": str(summary_path),
        "summary_sha256": sha256_file(summary_path),
        "row_count": summary["row_count"],
        "assistant_turn_count": summary["assistant_turn_count"],
        "database_count": summary["database_count"],
        "leakage_policy": summary["leakage_policy"],
        "leakage_boundary": summary["leakage_boundary"],
        "evaluation_command": summary["evaluation_command"],
        "command": command or sys.argv,
    }
    _write_json(manifest_path, manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--database-root", type=Path, default=DEFAULT_DATABASE_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY_OUTPUT)
    parser.add_argument("--manifest-output", type=Path, default=DEFAULT_MANIFEST_OUTPUT)
    args = parser.parse_args()

    command = [
        "python",
        "-m",
        "data.alias_column_context_inputs",
        "--input",
        str(args.input),
        "--database-root",
        str(args.database_root),
        "--output",
        str(args.output),
        "--summary-output",
        str(args.summary_output),
        "--manifest-output",
        str(args.manifest_output),
    ]
    manifest = write_alias_column_context_input_artifacts(
        input_path=args.input,
        database_root=args.database_root,
        output_path=args.output,
        summary_path=args.summary_output,
        manifest_path=args.manifest_output,
        command=command,
    )
    print(
        "Wrote alias/column context prepared inputs: "
        f"{manifest['row_count']} rows across {manifest['database_count']} databases"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
