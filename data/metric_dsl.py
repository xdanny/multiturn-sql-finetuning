"""Metric DSL contract for semantic-layer-first text-to-SQL experiments."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class MetricQuery:
    measures: tuple[str, ...]
    dimensions: tuple[str, ...] = ()
    filters: tuple[str, ...] = ()
    duplicate_row_policy: str | None = None
    order_by: str | None = None
    limit: int | None = None


def _normalize_name(value: str) -> str:
    return re.sub(r"\s+", "_", value.strip().strip("`\"[]").lower())


def _split_csv(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split(",") if part.strip())


def _parse_measures(value: str) -> tuple[str, ...]:
    return tuple(_normalize_name(match) for match in re.findall(r"MEASURE\(([^)]+)\)", value, re.I))


def _parse_duplicate_row_policy(value: str) -> str | None:
    match = re.search(
        r"duplicate_row_policy\s*=\s*['\"]([^'\"]+)['\"]",
        value,
        re.I,
    )
    return _normalize_name(match.group(1)) if match else None


def parse_metric_query(text: str) -> MetricQuery:
    """Parse a compact metric DSL while preserving governed MEASURE() tokens."""

    remaining = " ".join(text.strip().split())
    limit = None
    limit_match = re.search(r"\s+LIMIT\s+(\d+)\s*$", remaining, re.I)
    if limit_match:
        limit = int(limit_match.group(1))
        remaining = remaining[: limit_match.start()].strip()

    order_by = None
    order_match = re.search(r"\s+ORDER\s+BY\s+(.+)$", remaining, re.I)
    if order_match:
        order_by = order_match.group(1).strip()
        remaining = remaining[: order_match.start()].strip()

    filters: tuple[str, ...] = ()
    where_match = re.search(r"\s+WHERE\s+(.+)$", remaining, re.I)
    if where_match:
        filters = (where_match.group(1).strip(),)
        remaining = remaining[: where_match.start()].strip()

    by_match = re.search(r"\s+BY\s+(.+)$", remaining, re.I)
    dimension_text = ""
    measure_text = remaining
    duplicate_row_policy = None
    if by_match:
        dimension_text = by_match.group(1).strip()
        measure_text = remaining[: by_match.start()].strip()
        using_match = re.search(r"\s+USING\s+(.+)$", dimension_text, re.I)
        if using_match:
            duplicate_row_policy = _parse_duplicate_row_policy(using_match.group(1).strip())
            dimension_text = dimension_text[: using_match.start()].strip()

    return MetricQuery(
        measures=_parse_measures(measure_text),
        dimensions=tuple(_normalize_name(value) for value in _split_csv(dimension_text)),
        filters=filters,
        duplicate_row_policy=duplicate_row_policy,
        order_by=order_by,
        limit=limit,
    )


def _semantic_lookup(semantic_model: Mapping[str, Any], section: str, key: str) -> Mapping[str, Any]:
    values = semantic_model.get(section, {})
    if key not in values:
        raise KeyError(f"unknown {section[:-1]}: {key}")
    return values[key]


def _required_joins(query: MetricQuery, semantic_model: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    used_dimensions = set(query.dimensions)
    for filter_expression in query.filters:
        used_dimensions.update(
            name
            for name in semantic_model.get("dimensions", {})
            if re.search(rf"\b{re.escape(name)}\b", filter_expression)
        )

    joins = []
    for join in semantic_model.get("joins", []):
        if used_dimensions.intersection(join.get("required_by", [])):
            joins.append(join)
    return joins


def _replace_dimension_references(expression: str, semantic_model: Mapping[str, Any]) -> str:
    replaced = expression
    for name in sorted(semantic_model.get("dimensions", {}), key=len, reverse=True):
        sql = _semantic_lookup(semantic_model, "dimensions", name)["sql"]
        replaced = re.sub(rf"\b{re.escape(name)}\b", str(sql), replaced)
    return replaced


def _compile_order_by(order_by: str | None, semantic_model: Mapping[str, Any]) -> str | None:
    if not order_by:
        return None

    def replace_measure(match: re.Match[str]) -> str:
        return _normalize_name(match.group(1))

    compiled = re.sub(r"MEASURE\(([^)]+)\)", replace_measure, order_by, flags=re.I)
    return _replace_dimension_references(compiled, semantic_model)


def compile_metric_query(query: MetricQuery, semantic_model: Mapping[str, Any]) -> str:
    """Compile a governed metric query to SQL after semantic validation."""

    base_table = str(semantic_model["base_table"])
    select_parts = []
    group_by_parts = []
    for dimension in query.dimensions:
        dimension_sql = str(_semantic_lookup(semantic_model, "dimensions", dimension)["sql"])
        select_parts.append(f"{dimension_sql} AS {dimension}")
        group_by_parts.append(dimension_sql)
    for measure in query.measures:
        measure_spec = _semantic_lookup(semantic_model, "measures", measure)
        measure_sql = str(measure_spec["sql"])
        policy_sql = measure_spec.get("sql_by_policy", {})
        if query.duplicate_row_policy and query.duplicate_row_policy in policy_sql:
            measure_sql = str(policy_sql[query.duplicate_row_policy])
        select_parts.append(f"{measure_sql} AS {measure}")
    if not select_parts:
        raise ValueError("metric query must select at least one measure or dimension")

    parts = [f"SELECT {', '.join(select_parts)}", f"FROM {base_table}"]
    for join in _required_joins(query, semantic_model):
        parts.append(f"JOIN {join['table']} ON {join['sql_on']}")
    if query.filters:
        parts.append(
            "WHERE "
            + " AND ".join(
                _replace_dimension_references(filter_expression, semantic_model)
                for filter_expression in query.filters
            )
        )
    if query.measures and group_by_parts:
        parts.append(f"GROUP BY {', '.join(group_by_parts)}")
    order_by = _compile_order_by(query.order_by, semantic_model)
    if order_by:
        parts.append(f"ORDER BY {order_by}")
    if query.limit is not None:
        parts.append(f"LIMIT {query.limit}")
    return " ".join(parts)


def _f1(predicted: tuple[str, ...], gold: tuple[str, ...]) -> float:
    predicted_set = set(predicted)
    gold_set = set(gold)
    if not predicted_set and not gold_set:
        return 1.0
    if not predicted_set or not gold_set:
        return 0.0
    true_positive = len(predicted_set.intersection(gold_set))
    if true_positive == 0:
        return 0.0
    precision = true_positive / len(predicted_set)
    recall = true_positive / len(gold_set)
    return 2 * precision * recall / (precision + recall)


def score_metric_query(predicted: MetricQuery, gold: MetricQuery) -> dict[str, float]:
    """Score semantic intent separately from final SQL execution."""

    measure_f1 = _f1(predicted.measures, gold.measures)
    dimension_f1 = _f1(predicted.dimensions, gold.dimensions)
    filter_f1 = _f1(predicted.filters, gold.filters)
    return {
        "measure_f1": measure_f1,
        "dimension_f1": dimension_f1,
        "filter_f1": filter_f1,
        "measure_preservation": 1.0 if gold.measures and measure_f1 == 1.0 else 0.0,
    }
