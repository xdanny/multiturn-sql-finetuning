"""
Classify SQL benchmark failures into actionable error categories.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import sqlglot
from sqlglot import exp

ERROR_LABELS = [
    "correct",
    "invalid_sql",
    "execution_error",
    "schema_link",
    "join_path",
    "aggregation",
    "grain_fanout",
    "history_resolution",
    "ordering_limit",
    "value_grounding",
    "projection",
    "other",
]


@dataclass(frozen=True)
class SqlFeatures:
    parseable: bool
    tables: set[str]
    columns: set[str]
    join_tables: set[str]
    join_conditions: set[str]
    aggregations: set[str]
    group_by: set[str]
    where_sql: str | None
    order_sql: str | None
    limit_sql: str | None
    having_sql: str | None
    projections: set[str]
    projection_order: tuple[str, ...]
    literals: set[str]
    distinct: bool


def _repair_cosql_spacing(sql: str) -> str:
    """Repair token spacing emitted by CoSQL annotations before parser analysis."""

    repaired = re.sub(r">\s+=", ">=", sql)
    repaired = re.sub(r"<\s+=", "<=", repaired)
    repaired = re.sub(r"!\s+=", "!=", repaired)
    return repaired


def _norm_sql(value: Any) -> str:
    return " ".join(str(value).lower().split())


def _norm_literal(value: Any) -> str:
    return " ".join(str(value).split())


def _quoted_literals(sql: str) -> set[str]:
    return {
        _norm_literal(match.group(0))
        for pattern in (r"'[^']*'", r'"[^"]*"')
        for match in re.finditer(pattern, sql)
    }


def _expression_set(expressions: list[exp.Expression] | None) -> set[str]:
    if not expressions:
        return set()
    return {_norm_sql(expression.sql(dialect="sqlite")) for expression in expressions}


def _has_distinct(expression: exp.Expression) -> bool:
    return any(select.args.get("distinct") is not None for select in expression.find_all(exp.Select))


def _alias_map(expression: exp.Expression) -> dict[str, str]:
    aliases = {}
    for table in expression.find_all(exp.Table):
        name = table.name.lower()
        alias = table.alias.lower() if table.alias else name
        aliases[alias] = name
    return aliases


def _column_name(column: exp.Column, aliases: dict[str, str]) -> str:
    table = column.table.lower() if column.table else ""
    table = aliases.get(table, table)
    name = column.name.lower()
    return f"{table}.{name}" if table else name


def extract_sql_features(sql: str) -> SqlFeatures:
    repaired_sql = _repair_cosql_spacing(sql)
    try:
        expression = sqlglot.parse_one(repaired_sql, dialect="sqlite")
    except Exception:
        return SqlFeatures(
            parseable=False,
            tables=set(),
            columns=set(),
            join_tables=set(),
            join_conditions=set(),
            aggregations=set(),
            group_by=set(),
            where_sql=None,
            order_sql=None,
            limit_sql=None,
            having_sql=None,
            projections=set(),
            projection_order=(),
            literals=set(),
            distinct=False,
        )

    aliases = _alias_map(expression)
    tables = {table.name.lower() for table in expression.find_all(exp.Table)}
    columns = {_column_name(column, aliases) for column in expression.find_all(exp.Column)}
    joins = list(expression.find_all(exp.Join))
    join_tables = {
        join.this.name.lower()
        for join in joins
        if isinstance(join.this, exp.Table) and join.this.name
    }
    join_conditions = {
        _norm_sql(join.args["on"].sql(dialect="sqlite"))
        for join in joins
        if join.args.get("on") is not None
    }
    group = expression.args.get("group")
    select_expressions = expression.args.get("expressions") or []
    projection_order = tuple(_norm_sql(item.sql(dialect="sqlite")) for item in select_expressions)
    return SqlFeatures(
        parseable=True,
        tables=tables,
        columns=columns,
        join_tables=join_tables,
        join_conditions=join_conditions,
        aggregations={_norm_sql(agg.sql(dialect="sqlite")) for agg in expression.find_all(exp.AggFunc)},
        group_by=_expression_set(group.expressions if group else None),
        where_sql=_norm_sql(expression.args["where"].sql(dialect="sqlite"))
        if expression.args.get("where")
        else None,
        order_sql=_norm_sql(expression.args["order"].sql(dialect="sqlite"))
        if expression.args.get("order")
        else None,
        limit_sql=_norm_sql(expression.args["limit"].sql(dialect="sqlite"))
        if expression.args.get("limit")
        else None,
        having_sql=_norm_sql(expression.args["having"].sql(dialect="sqlite"))
        if expression.args.get("having")
        else None,
        projections=_expression_set(select_expressions),
        projection_order=projection_order,
        literals={
            _norm_literal(literal.sql(dialect="sqlite"))
            for literal in expression.find_all(exp.Literal)
        }
        | _quoted_literals(repaired_sql),
        distinct=_has_distinct(expression),
    )


def _symmetric_difference(left: set[str], right: set[str]) -> set[str]:
    return (left - right) | (right - left)


def classify_failure(row: dict[str, Any]) -> tuple[str, list[str]]:
    """Return primary and secondary error labels for one result row."""

    if float(row.get("value_execution_score") if row.get("value_execution_score") is not None else row.get("execution_score", 0.0)) == 1.0:
        return "correct", []
    if not row.get("syntax_valid", True):
        return "invalid_sql", []
    if row.get("score_error"):
        return "execution_error", []

    reference = extract_sql_features(row.get("reference_sql", ""))
    generated = extract_sql_features(row.get("generated_sql", ""))
    if not generated.parseable:
        return "invalid_sql", []
    if not reference.parseable:
        return "other", []

    labels = []
    if reference.tables != generated.tables:
        labels.append("schema_link")
    elif reference.join_tables != generated.join_tables or reference.join_conditions != generated.join_conditions:
        labels.append("join_path")

    if reference.aggregations != generated.aggregations:
        labels.append("aggregation")
    if (
        reference.group_by != generated.group_by
        or reference.distinct != generated.distinct
        or ((reference.distinct or generated.distinct) and reference.columns != generated.columns)
    ):
        labels.append("grain_fanout")
    if reference.order_sql != generated.order_sql or reference.limit_sql != generated.limit_sql:
        labels.append("ordering_limit")
    if (
        reference.where_sql != generated.where_sql
        or reference.having_sql != generated.having_sql
        or reference.literals != generated.literals
    ):
        labels.append("value_grounding")
    if (
        reference.projections != generated.projections
        or reference.projection_order != generated.projection_order
    ) and "aggregation" not in labels:
        labels.append("projection")

    turn_index = int(row.get("turn_index") or 0)
    if turn_index > 0 and any(
        label in labels
        for label in {"schema_link", "join_path", "aggregation", "grain_fanout", "value_grounding"}
    ):
        labels.append("history_resolution")

    if not labels:
        labels.append("other")

    priority = [
        "invalid_sql",
        "execution_error",
        "schema_link",
        "join_path",
        "aggregation",
        "grain_fanout",
        "ordering_limit",
        "value_grounding",
        "history_resolution",
        "projection",
        "other",
    ]
    primary = next(label for label in priority if label in labels)
    secondary = [label for label in labels if label != primary]
    return primary, secondary


def classify_row(row: dict[str, Any]) -> dict[str, Any]:
    primary, secondary = classify_failure(row)
    classified = dict(row)
    classified["error_primary"] = primary
    classified["error_secondary"] = secondary
    classified["error_secondary_text"] = ",".join(secondary)

    reference = extract_sql_features(row.get("reference_sql", ""))
    generated = extract_sql_features(row.get("generated_sql", ""))
    classified["reference_tables"] = sorted(reference.tables)
    classified["generated_tables"] = sorted(generated.tables)
    classified["missing_tables"] = sorted(reference.tables - generated.tables)
    classified["extra_tables"] = sorted(generated.tables - reference.tables)
    classified["missing_columns"] = sorted(reference.columns - generated.columns)
    classified["extra_columns"] = sorted(generated.columns - reference.columns)
    classified["table_diff_count"] = len(_symmetric_difference(reference.tables, generated.tables))
    classified["column_diff_count"] = len(_symmetric_difference(reference.columns, generated.columns))
    return classified


def write_classified_rows(rows: list[dict[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_error_summary(rows: list[dict[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row.get("model_name", ""), row.get("prompt_variant") or "")].append(row)

    with output.open("w", newline="") as f:
        fieldnames = [
            "model_name",
            "prompt_variant",
            "error_primary",
            "count",
            "rate",
            "value_accuracy",
            "strict_accuracy",
            "syntax_accuracy",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for (model_name, prompt_variant), group in sorted(grouped.items()):
            counts = Counter(row["error_primary"] for row in group)
            total = len(group)
            for label in ERROR_LABELS:
                count = counts.get(label, 0)
                if count == 0:
                    continue
                label_rows = [row for row in group if row["error_primary"] == label]
                writer.writerow(
                    {
                        "model_name": model_name,
                        "prompt_variant": prompt_variant,
                        "error_primary": label,
                        "count": count,
                        "rate": count / total,
                        "value_accuracy": sum(
                            float(row.get("value_execution_score") or 0.0) for row in label_rows
                        )
                        / count,
                        "strict_accuracy": sum(
                            float(row.get("strict_execution_score") or 0.0) for row in label_rows
                        )
                        / count,
                        "syntax_accuracy": sum(bool(row.get("syntax_valid")) for row in label_rows)
                        / count,
                    }
                )


def classify_file(input_path: Path, output_path: Path, summary_path: Path) -> int:
    rows = []
    with input_path.open() as f:
        for line in f:
            if line.strip():
                rows.append(classify_row(json.loads(line)))
    write_classified_rows(rows, output_path)
    write_error_summary(rows, summary_path)
    return len(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()

    count = classify_file(args.input, args.output, args.summary)
    print(f"Wrote {count} classified rows to {args.output}")
    print(f"Wrote summary to {args.summary}")
    return 0 if count else 1


if __name__ == "__main__":
    raise SystemExit(main())
