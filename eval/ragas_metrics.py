"""
Lightweight SQL evaluation metrics for single-turn and dialog benchmarks.

RAGAS agent metrics can be layered on later, but these deterministic metrics
are the first gate for local Qwen base-vs-finetuned benchmarking.
"""

from __future__ import annotations

import re
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import sqlglot

from eval.result_compare import compare_dataframes, is_order_sensitive_sql

SQL_START_RE = re.compile(r"\b(select|with|insert|update|delete)\b", re.IGNORECASE)


@dataclass
class SqlEvalResult:
    execution_score: float
    normalized_match: bool
    syntax_valid: bool
    latency_ms: float
    error: str | None = None
    strict_execution_score: float | None = None
    value_execution_score: float | None = None
    order_sensitive: bool = False


def normalize_sql(sql: str) -> str | None:
    """Return a canonical SQL string, or None when parsing fails."""

    sql = clean_sql(extract_sql(sql))
    try:
        return sqlglot.parse_one(sql).sql(dialect="sqlite", pretty=False).strip().lower()
    except Exception:
        return None


def extract_sql(text: str) -> str:
    """Extract the most likely SQL statement from model output."""

    stripped = text.strip()
    fenced = re.search(r"```(?:sql)?\s*(.*?)```", stripped, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        stripped = fenced.group(1).strip()
    match = SQL_START_RE.search(stripped)
    if not match:
        return stripped
    candidate = stripped[match.start() :].strip()
    semicolon = candidate.find(";")
    if semicolon >= 0:
        return candidate[: semicolon + 1]
    return candidate


def clean_sql(sql: str) -> str:
    """Normalize common dataset tokenization artifacts before parsing/execution."""

    return (
        sql.replace("> =", ">=")
        .replace("< =", "<=")
        .replace("! =", "!=")
        .replace("= =", "==")
    )


def syntax_valid(sql: str) -> bool:
    return normalize_sql(sql) is not None


def execute_sql(sql: str, database_path: str | Path) -> pd.DataFrame:
    sql = clean_sql(extract_sql(sql))
    with sqlite3.connect(database_path) as conn:
        return pd.read_sql_query(sql, conn)


def dataframes_equal(left: pd.DataFrame, right: pd.DataFrame) -> bool:
    """Backward-compatible strict DataFrame comparison."""

    return compare_dataframes(left, right, order_sensitive=False).strict_match


def score_single_turn(
    reference_sql: str,
    generated_sql: str,
    database_path: str | Path | None = None,
) -> SqlEvalResult:
    """Score generated SQL against a reference query.

    If `database_path` is supplied, execution-result equality is primary. When
    no database is available, normalized SQL string equality is used instead.
    """

    started = time.perf_counter()
    reference_norm = normalize_sql(reference_sql)
    generated_norm = normalize_sql(generated_sql)
    is_valid = generated_norm is not None
    normalized_match = reference_norm is not None and generated_norm == reference_norm

    if database_path is None:
        return SqlEvalResult(
            execution_score=1.0 if normalized_match else 0.0,
            normalized_match=normalized_match,
            syntax_valid=is_valid,
            latency_ms=(time.perf_counter() - started) * 1000,
            strict_execution_score=None,
            value_execution_score=None,
        )

    if not is_valid:
        return SqlEvalResult(
            execution_score=0.0,
            normalized_match=normalized_match,
            syntax_valid=False,
            latency_ms=(time.perf_counter() - started) * 1000,
            error="generated SQL is not parseable",
            strict_execution_score=0.0,
            value_execution_score=0.0,
        )

    try:
        reference_df = execute_sql(reference_sql, database_path)
        generated_df = execute_sql(generated_sql, database_path)
        order_sensitive = is_order_sensitive_sql(reference_sql)
        comparison = compare_dataframes(
            reference_df,
            generated_df,
            order_sensitive=order_sensitive,
        )
        return SqlEvalResult(
            execution_score=1.0 if comparison.value_match else 0.0,
            normalized_match=normalized_match,
            syntax_valid=True,
            latency_ms=(time.perf_counter() - started) * 1000,
            strict_execution_score=1.0 if comparison.strict_match else 0.0,
            value_execution_score=1.0 if comparison.value_match else 0.0,
            order_sensitive=order_sensitive,
        )
    except Exception as exc:
        return SqlEvalResult(
            execution_score=0.0,
            normalized_match=normalized_match,
            syntax_valid=True,
            latency_ms=(time.perf_counter() - started) * 1000,
            error=str(exc),
            strict_execution_score=0.0,
            value_execution_score=0.0,
        )


def score_multi_turn(turns: list[dict], database_path: str | Path | None = None) -> dict:
    """Score a dialog represented as per-turn reference/generated SQL pairs."""

    per_turn = [
        score_single_turn(
            reference_sql=turn["reference_sql"],
            generated_sql=turn["generated_sql"],
            database_path=database_path,
        )
        for turn in turns
    ]
    if not per_turn:
        return {"turn_count": 0, "execution_accuracy": 0.0, "interaction_match": False}

    execution_accuracy = sum(result.execution_score for result in per_turn) / len(per_turn)
    return {
        "turn_count": len(per_turn),
        "execution_accuracy": execution_accuracy,
        "syntax_accuracy": sum(result.syntax_valid for result in per_turn) / len(per_turn),
        "normalized_accuracy": sum(result.normalized_match for result in per_turn) / len(per_turn),
        "interaction_match": all(result.execution_score == 1.0 for result in per_turn),
        "per_turn": [result.__dict__ for result in per_turn],
    }
