"""Build value-grounding labels from prepared multi-turn SQL dialogs.

The labels are extracted from reference SQL in prepared records. They are
supervision/evaluation artifacts, not production-time hints.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import sqlglot
from sqlglot import exp

from data.sql_labels import repair_cosql_spacing
from eval.result_manifest import sha256_file

COMPARISON_OPERATORS: tuple[type[exp.Expression], ...] = (
    exp.EQ,
    exp.NEQ,
    exp.GT,
    exp.GTE,
    exp.LT,
    exp.LTE,
)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _parse_sql(sql: str) -> exp.Expression | None:
    try:
        return sqlglot.parse_one(repair_cosql_spacing(sql), dialect="sqlite")
    except Exception:
        return None


def _literal_payload(literal: exp.Literal) -> tuple[str, str]:
    value = str(literal.this)
    literal_type = "string" if literal.is_string else "number"
    return value, literal_type


def _column_name(column: exp.Column) -> str:
    return column.sql(dialect="sqlite").strip("`\"[]").lower()


def _operator(node: exp.Expression) -> str:
    return {
        exp.EQ: "=",
        exp.NEQ: "!=",
        exp.GT: ">",
        exp.GTE: ">=",
        exp.LT: "<",
        exp.LTE: "<=",
    }[type(node)]


def _table_aliases(expression: exp.Expression) -> tuple[dict[str, str], list[str]]:
    aliases: dict[str, str] = {}
    table_names: list[str] = []
    for table in expression.find_all(exp.Table):
        table_name = str(table.name).strip("`\"[]").lower()
        if not table_name:
            continue
        table_names.append(table_name)
        aliases[table_name] = table_name
        alias = str(table.alias_or_name or "").strip("`\"[]").lower()
        if alias:
            aliases[alias] = table_name
    return aliases, sorted(set(table_names))


def _resolved_table(
    column: exp.Column,
    aliases: dict[str, str],
    table_names: list[str],
) -> str | None:
    qualifier = str(column.table or "").strip("`\"[]").lower()
    if qualifier:
        return aliases.get(qualifier, qualifier)
    if len(table_names) == 1:
        return table_names[0]
    return None


def _comparison_reference(
    node: exp.Expression,
    aliases: dict[str, str],
    table_names: list[str],
) -> dict[str, Any] | None:
    left = getattr(node, "left", None)
    right = getattr(node, "right", None)
    if isinstance(left, exp.Column) and isinstance(right, exp.Literal):
        value, literal_type = _literal_payload(right)
        return {
            "table": _resolved_table(left, aliases, table_names),
            "column": _column_name(left),
            "operator": _operator(node),
            "literal_value": value,
            "literal_type": literal_type,
        }
    if isinstance(left, exp.Literal) and isinstance(right, exp.Column):
        value, literal_type = _literal_payload(left)
        return {
            "table": _resolved_table(right, aliases, table_names),
            "column": _column_name(right),
            "operator": _operator(node),
            "literal_value": value,
            "literal_type": literal_type,
        }
    return None


def extract_sql_value_references(sql: str) -> list[dict[str, Any]]:
    """Extract column/literal predicates from one SQL query."""

    expression = _parse_sql(sql)
    if expression is None:
        return []

    aliases, table_names = _table_aliases(expression)
    references: list[dict[str, Any]] = []
    for node_type in COMPARISON_OPERATORS:
        for node in expression.find_all(node_type):
            reference = _comparison_reference(node, aliases, table_names)
            if reference:
                references.append(reference)
    for node in expression.find_all(exp.In):
        if not isinstance(node.this, exp.Column):
            continue
        column = _column_name(node.this)
        table = _resolved_table(node.this, aliases, table_names)
        for item in node.expressions:
            if isinstance(item, exp.Literal):
                value, literal_type = _literal_payload(item)
                references.append(
                    {
                        "table": table,
                        "column": column,
                        "operator": "in",
                        "literal_value": value,
                        "literal_type": literal_type,
                    }
                )
    return references


def _find_exact_mention(value: str, text: str) -> str | None:
    pattern = re.compile(rf"(?<![A-Za-z0-9]){re.escape(value)}(?![A-Za-z0-9])", re.I)
    match = pattern.search(text)
    return match.group(0) if match else None


def _mention_status(
    value: str,
    current_text: str,
    history_text: str,
    prior_sql_text: str,
) -> tuple[str, str | None]:
    current = _find_exact_mention(value, current_text)
    if current is not None:
        return "exact_in_current_turn", current
    history = _find_exact_mention(value, history_text)
    if history is not None:
        return "exact_in_history", history
    prior_sql = _find_exact_mention(value, prior_sql_text)
    if prior_sql is not None:
        return "carried_from_prior_sql", prior_sql
    return "missing_from_user_text", None


def _prior_turn_reference(status: str) -> str | None:
    return {
        "exact_in_history": "user_history",
        "carried_from_prior_sql": "assistant_sql",
    }.get(status)


def build_value_grounding_artifacts(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build one value-grounding row per SQL literal predicate."""

    rows: list[dict[str, Any]] = []
    for record_index, record in enumerate(records):
        database_id = str(record.get("database_id") or record.get("db_id") or "unknown")
        dialog_id = str(record.get("dialog_id") or f"{database_id}:{record_index}")
        source = record.get("source")
        user_messages: list[str] = []
        assistant_sql_history: list[str] = []
        turn_index = 0
        for message in record.get("messages") or []:
            role = message.get("role")
            content = str(message.get("content") or "")
            if role == "user":
                user_messages.append(content)
                continue
            if role != "assistant":
                continue
            current_text = user_messages[-1] if user_messages else ""
            history_text = "\n".join(user_messages[:-1])
            prior_sql_text = "\n".join(assistant_sql_history)
            for value_index, reference in enumerate(extract_sql_value_references(content)):
                status, span = _mention_status(
                    str(reference["literal_value"]),
                    current_text,
                    history_text,
                    prior_sql_text,
                )
                rows.append(
                    {
                        "id": f"{dialog_id}:{turn_index}:{value_index}",
                        "artifact_type": "value_grounding_label",
                        "schema_version": 1,
                        "label_source": "gold_reference_sql",
                        "dialog_id": dialog_id,
                        "turn_id": f"{dialog_id}:{turn_index}",
                        "turn_index": turn_index,
                        "database_id": database_id,
                        "source": source,
                        "column": reference["column"],
                        "resolved_table": reference["table"],
                        "resolved_column": reference["column"],
                        "operator": reference["operator"],
                        "literal_value": reference["literal_value"],
                        "resolved_value": reference["literal_value"],
                        "literal_type": reference["literal_type"],
                        "mention_status": status,
                        "evidence_span": span,
                        "mention_text": span,
                        "prior_turn_reference": _prior_turn_reference(status),
                        "requires_context_carryover": status
                        in {"exact_in_history", "carried_from_prior_sql"},
                        "requires_value_normalization": status
                        in {"missing_from_user_text", "carried_from_prior_sql"},
                        "question_text": current_text,
                        "reference_sql": content,
                    }
                )
            assistant_sql_history.append(content)
            turn_index += 1
    return rows


def summarize_value_grounding_artifacts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    statuses = Counter(str(row.get("mention_status") or "") for row in rows)
    return {
        "value_reference_count": len(rows),
        "dialog_turn_count": len({(row.get("dialog_id"), row.get("turn_index")) for row in rows}),
        "database_count": len({row.get("database_id") for row in rows}),
        "exact_in_current_turn_count": statuses.get("exact_in_current_turn", 0),
        "exact_in_history_count": statuses.get("exact_in_history", 0),
        "carried_from_prior_sql_count": statuses.get("carried_from_prior_sql", 0),
        "missing_from_user_text_count": statuses.get("missing_from_user_text", 0),
        "requires_context_carryover_count": sum(
            1 for row in rows if row.get("requires_context_carryover")
        ),
        "requires_value_normalization_count": sum(
            1 for row in rows if row.get("requires_value_normalization")
        ),
    }


def build_value_grounding_manifest(
    *,
    input_path: Path,
    output_path: Path,
    summary: dict[str, Any],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build the reproducibility manifest for a value/entity artifact export."""

    return {
        "schema_version": 1,
        "artifact_type": "value_entity_artifact_v1",
        "label_source": "gold_reference_sql",
        "input_path": str(input_path),
        "input_sha256": sha256_file(input_path),
        "output_path": str(output_path),
        "output_sha256": sha256_file(output_path),
        "row_count": len(rows),
        "database_count": len({row.get("database_id") for row in rows}),
        "dialog_turn_count": len({(row.get("dialog_id"), row.get("turn_index")) for row in rows}),
        "metrics": dict(summary),
        "row_schema": {
            "artifact_type": "value_grounding_label",
            "schema_version": 1,
            "required_fields": [
                "id",
                "dialog_id",
                "turn_id",
                "turn_index",
                "database_id",
                "resolved_column",
                "operator",
                "resolved_value",
                "literal_type",
                "resolved_table",
                "mention_status",
                "mention_text",
                "prior_turn_reference",
                "requires_context_carryover",
                "requires_value_normalization",
                "label_source",
            ],
        },
    }


def run_value_grounding_export(
    *,
    input_path: Path,
    output_path: Path,
    summary_path: Path,
    manifest_path: Path | None = None,
) -> int:
    if manifest_path is None:
        manifest_path = output_path.with_suffix(".manifest.json")
    records = _load_jsonl(input_path)
    rows = build_value_grounding_artifacts(records)
    summary = summarize_value_grounding_artifacts(rows)
    _write_jsonl(output_path, rows)
    _write_json(summary_path, summary)
    _write_json(
        manifest_path,
        build_value_grounding_manifest(
            input_path=input_path,
            output_path=output_path,
            summary=summary,
            rows=rows,
        ),
    )
    print(f"Wrote {len(rows)} value-grounding label rows to {output_path}")
    print(f"Wrote value-grounding summary to {summary_path}")
    print(f"Wrote value-grounding manifest to {manifest_path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/eval_cosql_dev_100.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/data_artifacts/value_grounding_labels_cosql_dev_100.jsonl"),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("docs/data_artifacts/value_grounding_labels_cosql_dev_100_summary.json"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("docs/data_artifacts/value_grounding_labels_cosql_dev_100.manifest.json"),
    )
    args = parser.parse_args()
    return run_value_grounding_export(
        input_path=args.input,
        output_path=args.output,
        summary_path=args.summary,
        manifest_path=args.manifest,
    )


if __name__ == "__main__":
    raise SystemExit(main())
