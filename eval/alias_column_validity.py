"""Score invalid table-column references in generated SQL."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import sqlglot
from sqlglot import exp

from eval.result_manifest import sha256_file


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _allowed_columns_from_prompt(messages: list[dict[str, str]]) -> dict[str, set[str]]:
    prompt = "\n".join(message.get("content", "") for message in messages)
    match = re.search(r'"allowed_columns"\s*:\s*(?P<object>\{.*?\})\s*,\s*"artifact_type"', prompt, re.DOTALL)
    if not match:
        return _allowed_columns_from_bullets(prompt)
    try:
        payload = json.loads(match.group("object"))
    except json.JSONDecodeError:
        return _allowed_columns_from_bullets(prompt)
    return {
        str(table).lower(): {str(column).lower() for column in columns}
        for table, columns in payload.items()
    }


def _allowed_columns_from_bullets(prompt: str) -> dict[str, set[str]]:
    allowed: dict[str, set[str]] = {}
    in_allowed = False
    for raw_line in prompt.splitlines():
        line = raw_line.strip()
        if line == "Allowed columns:":
            in_allowed = True
            continue
        if in_allowed and line in {"Join keys:", ""}:
            break
        if not in_allowed or not line.startswith("- ") or ":" not in line:
            continue
        table, columns = line[2:].split(":", 1)
        allowed[table.strip().lower()] = {
            column.strip().lower()
            for column in columns.split(",")
            if column.strip()
        }
    return allowed


def _parse_column_refs(
    sql: str,
) -> tuple[list[dict[str, str | bool]], list[dict[str, str | bool]], str | None, set[str]]:
    try:
        expression = sqlglot.parse_one(sql, dialect="sqlite")
    except sqlglot.errors.SqlglotError as exc:
        return [], [], str(exc), set()

    alias_to_table = {
        table.alias_or_name.lower(): table.name.lower()
        for table in expression.find_all(exp.Table)
        if table.alias_or_name and table.name
    }
    table_scope = set(alias_to_table.values())
    output_aliases = {
        alias.alias.lower()
        for alias in expression.find_all(exp.Alias)
        if alias.alias
    }
    refs = []
    unresolved_aliases = []
    for column in expression.find_all(exp.Column):
        qualifier = str(column.table or "")
        table = alias_to_table.get(qualifier.lower(), qualifier.lower()) if qualifier else ""
        is_select_alias_ref = not qualifier and column.name.lower() in output_aliases
        ref = {
            "sql": column.sql(dialect="sqlite"),
            "qualifier": qualifier,
            "table": table,
            "column": column.name,
            "select_alias_ref": is_select_alias_ref,
        }
        refs.append(ref)
        if qualifier and qualifier.lower() not in alias_to_table:
            unresolved_aliases.append(ref)
    return refs, unresolved_aliases, None, table_scope


def _invalid_refs(
    refs: list[dict[str, str | bool]],
    *,
    allowed: dict[str, set[str]],
    table_scope: set[str],
) -> list[dict[str, str | bool]]:
    invalid = []
    for ref in refs:
        column = str(ref["column"]).lower()
        table = str(ref["table"]).lower()
        if ref["select_alias_ref"] is True:
            continue
        if table:
            if table not in allowed or column not in allowed[table]:
                invalid.append(ref)
            continue
        candidate_tables = [
            scoped_table
            for scoped_table in table_scope
            if scoped_table in allowed and column in allowed[scoped_table]
        ]
        if len(candidate_tables) != 1:
            invalid.append(ref)
    return invalid


def score_alias_column_validity(
    *,
    input_path: Path,
    rollout_output_path: Path,
) -> dict[str, Any]:
    inputs_by_dialog = {
        str(row.get("dialog_id") or row.get("id") or f"prepared-{index}"): row
        for index, row in enumerate(_load_jsonl(input_path))
    }
    output_rows = _load_jsonl(rollout_output_path)
    scored_rows = []
    for output in output_rows:
        dialog_id = str(output.get("dialog_id"))
        input_row = inputs_by_dialog.get(dialog_id)
        if not input_row:
            continue
        allowed = _allowed_columns_from_prompt(input_row.get("messages", []))
        refs, unresolved_aliases, parse_error, table_scope = _parse_column_refs(
            str(output.get("generated_sql") or "")
        )
        invalid_refs = _invalid_refs(refs, allowed=allowed, table_scope=table_scope)
        expected = input_row.get("expected_column_validity", {})
        invalid_patterns = [str(pattern) for pattern in expected.get("invalid_patterns_to_avoid", [])]
        generated_sql = str(output.get("generated_sql") or "")
        matched_invalid_patterns = [
            pattern for pattern in invalid_patterns if pattern.lower() in generated_sql.lower()
        ]
        scored_rows.append(
            {
                "id": output.get("id"),
                "dialog_id": dialog_id,
                "parse_error": parse_error,
                "table_column_refs": refs,
                "unresolved_alias_refs": unresolved_aliases,
                "invalid_table_column_refs": invalid_refs,
                "matched_invalid_patterns": matched_invalid_patterns,
                "alias_resolution_success": not unresolved_aliases and parse_error is None,
                "schema_valid_sql": not invalid_refs and parse_error is None,
                "column_validity_match": (
                    parse_error is None and not invalid_refs and not matched_invalid_patterns
                ),
            }
        )

    correct = sum(1 for row in scored_rows if row["column_validity_match"])
    total = len(scored_rows)
    return {
        "schema_version": 1,
        "artifact_type": "alias_column_validity_score",
        "input_path": str(input_path),
        "input_sha256": sha256_file(input_path),
        "rollout_output_path": str(rollout_output_path),
        "rollout_output_sha256": sha256_file(rollout_output_path),
        "scored_row_count": total,
        "column_validity_accuracy": correct / total if total else 0.0,
        "schema_valid_sql_rate": (
            sum(1 for row in scored_rows if row["schema_valid_sql"]) / total if total else 0.0
        ),
        "alias_resolution_success_rate": (
            sum(1 for row in scored_rows if row["alias_resolution_success"]) / total
            if total
            else 0.0
        ),
        "rows": scored_rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--rollout-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    payload = score_alias_column_validity(
        input_path=args.input,
        rollout_output_path=args.rollout_output,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(
        "Column-validity accuracy: "
        f"{payload['column_validity_accuracy']:.3f} over {payload['scored_row_count']} rows"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
