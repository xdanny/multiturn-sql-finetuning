"""
Oracle SQL-derived labels for schema linking and prompt pruning.

These labels are extracted from reference SQL. They are useful supervised
targets and upper-bound diagnostics, but they are not available to a production
system at inference time.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

import sqlglot
from sqlglot import exp


def repair_cosql_spacing(sql: str) -> str:
    """Repair token spacing emitted by CoSQL annotations before parser analysis."""

    repaired = re.sub(r">\s+=", ">=", sql)
    repaired = re.sub(r"<\s+=", "<=", repaired)
    repaired = re.sub(r"!\s+=", "!=", repaired)
    repaired = re.sub(r"\bWHERE(?=[A-Za-z_])", "WHERE ", repaired, flags=re.IGNORECASE)
    return repaired


def _repair_misplaced_alias_after_where(sql: str) -> str:
    match = re.match(
        r"(?P<prefix>.+?\bFROM\s+[A-Za-z_][A-Za-z0-9_]*)\s+WHERE\s+"
        r"(?P<condition>.+?)\s+AS\s+(?P<alias>[A-Za-z_][A-Za-z0-9_]*)\s+"
        r"(?P<join>JOIN\s+.+?)(?P<tail>\s+GROUP\s+BY\s+.+|\s+ORDER\s+BY\s+.+|\s+LIMIT\s+.+)?$",
        sql,
        flags=re.IGNORECASE,
    )
    if not match:
        return sql
    tail = match.group("tail") or ""
    return (
        f"{match.group('prefix')} AS {match.group('alias')} "
        f"{match.group('join')} WHERE {match.group('condition')}{tail}"
    )


def _strip_unmatched_trailing_parens(sql: str) -> str:
    repaired = sql.rstrip()
    while repaired.endswith(")") and repaired.count(")") > repaired.count("("):
        repaired = repaired[:-1].rstrip()
    return repaired


def _parse_sql(sql: str) -> exp.Expression:
    repaired = repair_cosql_spacing(sql)
    candidates = [
        repaired,
        _strip_unmatched_trailing_parens(repaired),
        _repair_misplaced_alias_after_where(repaired),
    ]
    seen = set()
    last_error: Exception | None = None
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        try:
            return sqlglot.parse_one(candidate, dialect="sqlite")
        except Exception as exc:
            last_error = exc
    if last_error:
        raise last_error
    raise ValueError("no SQL candidate to parse")


def _norm_sql(value: Any) -> str:
    return " ".join(str(value).lower().split())


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


def _normalize_identifier(value: str) -> str:
    return value.strip().strip("`\"[]").lower()


def schema_columns_from_context(schema: str | None) -> set[str]:
    """Extract normalized table/column names from compact `table(col type)` schema text."""

    if not schema:
        return set()

    sql_types = {
        "boolean",
        "bool",
        "date",
        "datetime",
        "decimal",
        "double",
        "float",
        "int",
        "integer",
        "number",
        "others",
        "real",
        "text",
        "time",
        "timestamp",
        "varchar",
    }
    columns: set[str] = set()
    for line in schema.splitlines():
        match = re.match(r"^\s*([^\s(]+)\((.*)\)\s*$", line)
        if not match:
            continue
        table = _normalize_identifier(match.group(1))
        for raw_column in match.group(2).split(","):
            column = raw_column.strip()
            if not column:
                continue
            name, _, maybe_type = column.rpartition(" ")
            if maybe_type.lower() in sql_types and name:
                column = name
            column = _normalize_identifier(column)
            if column and column != "*":
                columns.add(column)
                columns.add(f"{table}.{column}")
    return columns


def _looks_like_schema_column(name: str) -> bool:
    parts = name.split(".")
    return all(re.fullmatch(r"[a-z_][a-z0-9_]*", part) for part in parts)


def _allowed_column_names(schema_columns: Iterable[str] | None) -> set[str]:
    if not schema_columns:
        return set()
    allowed: set[str] = set()
    for column in schema_columns:
        normalized = _normalize_identifier(str(column))
        if normalized:
            allowed.add(normalized)
            if "." in normalized:
                allowed.add(normalized.rsplit(".", maxsplit=1)[-1])
    return allowed


def _is_schema_column(column: exp.Column, aliases: dict[str, str], allowed_columns: set[str]) -> bool:
    name = _column_name(column, aliases)
    if allowed_columns:
        unqualified = name.rsplit(".", maxsplit=1)[-1]
        return name in allowed_columns or unqualified in allowed_columns

    # CoSQL annotations frequently use double quotes for string literals. Sqlglot
    # parses those as quoted identifiers in SQLite mode, so unqualified quoted
    # identifier-like values such as "Wang" or "null" must not become columns.
    if column.this.args.get("quoted") and not column.table:
        return False
    return _looks_like_schema_column(name)


def _expression_list(expressions: list[exp.Expression] | None) -> list[str]:
    if not expressions:
        return []
    return [_norm_sql(expression.sql(dialect="sqlite")) for expression in expressions]


def _selects(expression: exp.Expression) -> list[exp.Select]:
    return list(expression.find_all(exp.Select))


def _primary_select(expression: exp.Expression) -> exp.Select | exp.Expression:
    if isinstance(expression, exp.Select):
        return expression
    selects = _selects(expression)
    return selects[0] if selects else expression


def _query_skeleton(expression: exp.Expression) -> dict[str, bool]:
    distinct = _has_distinct(expression)
    selects = _selects(expression)

    def has_select_arg(name: str) -> bool:
        return any(select.args.get(name) is not None for select in selects) or expression.args.get(name) is not None

    return {
        "select": True,
        "join": any(expression.find_all(exp.Join)),
        "where": has_select_arg("where"),
        "group_by": has_select_arg("group"),
        "having": has_select_arg("having"),
        "order_by": has_select_arg("order"),
        "limit": has_select_arg("limit"),
        "nested": any(expression.find_all(exp.Subquery)),
        "distinct": distinct,
    }


def _has_distinct(expression: exp.Expression) -> bool:
    return any(select.args.get("distinct") is not None for select in expression.find_all(exp.Select))


def labels_from_sql(sql: str, *, schema_columns: Iterable[str] | None = None) -> dict[str, Any]:
    """Extract schema-link, skeleton, and projection labels from one gold SQL query."""

    try:
        expression = _parse_sql(sql)
    except Exception:
        return {
            "parseable": False,
            "relevant_tables": [],
            "relevant_columns": [],
            "join_path": [],
            "query_skeleton": {},
            "projection_shape": {},
        }

    aliases = _alias_map(expression)
    allowed_columns = _allowed_column_names(schema_columns)
    primary_select = _primary_select(expression)
    tables = sorted({table.name.lower() for table in expression.find_all(exp.Table)})
    columns = sorted(
        {
            _column_name(column, aliases)
            for column in expression.find_all(exp.Column)
            if _is_schema_column(column, aliases, allowed_columns)
        }
    )
    joins = list(expression.find_all(exp.Join))
    join_path = [
        _norm_sql(join.args["on"].sql(dialect="sqlite"))
        for join in joins
        if join.args.get("on") is not None
    ]
    group = primary_select.args.get("group")
    order = primary_select.args.get("order") or expression.args.get("order")
    select_expressions = primary_select.args.get("expressions") or []
    projection_order = _expression_list(select_expressions)
    aggregations = [
        _norm_sql(aggregation.sql(dialect="sqlite"))
        for aggregation in expression.find_all(exp.AggFunc)
    ]
    distinct = _has_distinct(expression)

    return {
        "parseable": True,
        "relevant_tables": tables,
        "relevant_columns": columns,
        "join_path": join_path,
        "query_skeleton": _query_skeleton(expression),
        "projection_shape": {
            "selected_expressions": projection_order,
            "selected_count": len(projection_order),
            "aggregations": sorted(set(aggregations)),
            "group_by": _expression_list(group.expressions if group else None),
            "order_by": _norm_sql(order.sql(dialect="sqlite")) if order else None,
            "limit": _norm_sql((primary_select.args.get("limit") or expression.args["limit"]).sql(dialect="sqlite"))
            if primary_select.args.get("limit") or expression.args.get("limit")
            else None,
            "distinct": distinct,
            "preserve_duplicates": not distinct,
        },
    }

