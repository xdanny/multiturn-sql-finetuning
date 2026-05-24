"""
Compare SQL execution results with explicit strict and value-only modes.

Strict comparison keeps output column labels. Value comparison ignores harmless
alias differences but still preserves selected column order and duplicate rows.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import pandas as pd
import sqlglot


@dataclass(frozen=True)
class ResultComparison:
    strict_match: bool
    value_match: bool
    order_sensitive: bool


def is_order_sensitive_sql(sql: str) -> bool:
    """Return true when SQL result order should be preserved for comparison."""

    try:
        expression = sqlglot.parse_one(sql, dialect="sqlite")
    except Exception:
        return False
    return bool(expression.args.get("order") or expression.args.get("limit"))


def _normalize_cell(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, float):
        return value
    if isinstance(value, int):
        return float(value)
    return value


def _cells_equal(left: Any, right: Any, *, float_tolerance: float) -> bool:
    left = _normalize_cell(left)
    right = _normalize_cell(right)
    if left is None or right is None:
        return left is None and right is None
    if isinstance(left, float) and isinstance(right, float):
        return math.isclose(left, right, rel_tol=float_tolerance, abs_tol=float_tolerance)
    return left == right


def _row_sort_key(row: tuple[Any, ...]) -> tuple[str, ...]:
    return tuple(f"{type(value).__name__}:{value!r}" for value in row)


def _rows(df: pd.DataFrame) -> list[tuple[Any, ...]]:
    return [tuple(_normalize_cell(value) for value in row) for row in df.itertuples(index=False, name=None)]


def _rows_equal(
    left_rows: list[tuple[Any, ...]],
    right_rows: list[tuple[Any, ...]],
    *,
    float_tolerance: float,
) -> bool:
    if len(left_rows) != len(right_rows):
        return False
    for left_row, right_row in zip(left_rows, right_rows, strict=True):
        if len(left_row) != len(right_row):
            return False
        if not all(
            _cells_equal(left_cell, right_cell, float_tolerance=float_tolerance)
            for left_cell, right_cell in zip(left_row, right_row, strict=True)
        ):
            return False
    return True


def compare_dataframes(
    reference: pd.DataFrame,
    generated: pd.DataFrame,
    *,
    order_sensitive: bool,
    float_tolerance: float = 1e-6,
) -> ResultComparison:
    """Compare execution DataFrames in strict and alias-tolerant modes."""

    if len(reference.columns) != len(generated.columns):
        return ResultComparison(
            strict_match=False,
            value_match=False,
            order_sensitive=order_sensitive,
        )

    reference_rows = _rows(reference)
    generated_rows = _rows(generated)
    if not order_sensitive:
        reference_rows = sorted(reference_rows, key=_row_sort_key)
        generated_rows = sorted(generated_rows, key=_row_sort_key)

    value_match = _rows_equal(
        reference_rows,
        generated_rows,
        float_tolerance=float_tolerance,
    )
    strict_match = list(reference.columns) == list(generated.columns) and value_match
    return ResultComparison(
        strict_match=strict_match,
        value_match=value_match,
        order_sensitive=order_sensitive,
    )
