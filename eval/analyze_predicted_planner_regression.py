"""Analyze corrected row-level deltas for a predicted-planner SQL pair."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from eval.planner_eval import score_plans
from eval.ragas_metrics import execute_sql, extract_sql, score_single_turn
from eval.result_compare import compare_dataframes, is_order_sensitive_sql


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _value_ok(row: dict[str, Any], sql: str) -> tuple[bool, float, float, bool, str | None]:
    score = score_single_turn(row["reference_sql"], sql, database_path=row.get("database_path"))
    return (
        bool(score.value_execution_score),
        float(score.value_execution_score or 0.0),
        float(score.strict_execution_score or 0.0),
        bool(score.syntax_valid),
        score.error,
    )


def _bucket(*, direct_value_ok: bool, predicted_value_ok: bool) -> str:
    if direct_value_ok and predicted_value_ok:
        return "both_correct"
    if direct_value_ok and not predicted_value_ok:
        return "direct_only_regression"
    if predicted_value_ok and not direct_value_ok:
        return "planner_only_fix"
    return "both_wrong"


def _projection_order_flip(reference_sql: str, generated_sql: str, database_path: str | None) -> bool:
    if database_path is None:
        return False
    try:
        reference = execute_sql(reference_sql, database_path)
        generated = execute_sql(generated_sql, database_path)
    except Exception:
        return False
    if list(reference.columns) == list(generated.columns):
        return False
    if set(reference.columns) != set(generated.columns):
        return False
    reordered = generated[list(reference.columns)]
    comparison = compare_dataframes(
        reference,
        reordered,
        order_sensitive=is_order_sensitive_sql(reference_sql),
    )
    return comparison.value_match


def _regression_cause(row: dict[str, Any], predicted_sql: str) -> str:
    if _projection_order_flip(row["reference_sql"], predicted_sql, row.get("database_path")):
        return "projection_order_flip"
    try:
        execute_sql(predicted_sql, row.get("database_path"))
    except Exception:
        return "predicted_sql_execution_error"
    planner_score = score_plans(row.get("gold_plan"), row.get("predicted_plan"))
    if planner_score["macro_planner_score"] < 0.9:
        return "planner_state_error"
    return "sql_generation_error_after_good_plan"


def _current_question(row: dict[str, Any]) -> str:
    for message in reversed(row.get("messages") or []):
        if message.get("role") != "user":
            continue
        content = str(message.get("content") or "")
        matches = re.findall(r"(?is)\bQuestion:\s*(.*)", content)
        if matches:
            return " ".join(matches[-1].split())
        return " ".join(content.split())
    return ""


def analyze_pair(
    *,
    direct_rows: list[dict[str, Any]],
    predicted_rows: list[dict[str, Any]],
    max_examples: int = 12,
) -> dict[str, Any]:
    """Return corrected row-level analysis for a same-row planner comparison."""

    if len(direct_rows) != len(predicted_rows):
        raise ValueError("direct and predicted rows must have equal length")

    bucket_counts: Counter[str] = Counter()
    regression_causes: Counter[str] = Counter()
    planner_scores = []
    examples: list[dict[str, Any]] = []
    direct_value_total = 0.0
    direct_strict_total = 0.0
    predicted_value_total = 0.0
    predicted_strict_total = 0.0
    direct_syntax_valid = 0
    predicted_syntax_valid = 0

    for direct, predicted in zip(direct_rows, predicted_rows, strict=True):
        direct_sql = extract_sql(str(direct.get("raw_generation") or direct.get("generated_sql") or ""))
        predicted_sql = extract_sql(
            str(predicted.get("raw_generation") or predicted.get("generated_sql") or "")
        )
        direct_ok, direct_value, direct_strict, direct_syntax, direct_error = _value_ok(
            direct, direct_sql
        )
        predicted_ok, predicted_value, predicted_strict, predicted_syntax, predicted_error = (
            _value_ok(predicted, predicted_sql)
        )
        bucket = _bucket(direct_value_ok=direct_ok, predicted_value_ok=predicted_ok)
        bucket_counts[bucket] += 1
        direct_value_total += direct_value
        direct_strict_total += direct_strict
        predicted_value_total += predicted_value
        predicted_strict_total += predicted_strict
        direct_syntax_valid += int(direct_syntax)
        predicted_syntax_valid += int(predicted_syntax)

        planner_score = score_plans(predicted.get("gold_plan"), predicted.get("predicted_plan"))
        planner_scores.append(planner_score)
        cause = None
        if bucket == "direct_only_regression":
            cause = _regression_cause(predicted, predicted_sql)
            regression_causes[cause] += 1

        if bucket != "both_correct" and len(examples) < max_examples:
            examples.append(
                {
                    "bucket": bucket,
                    "database_id": direct.get("database_id"),
                    "direct_error": direct_error,
                    "direct_sql": direct_sql,
                    "id": direct.get("id"),
                    "predicted_error": predicted_error,
                    "predicted_sql": predicted_sql,
                    "question": _current_question(direct),
                    "reference_sql": direct.get("reference_sql"),
                    "regression_cause": cause,
                    "planner_macro_score": planner_score["macro_planner_score"],
                }
            )

    row_count = len(direct_rows)
    direct_value_accuracy = direct_value_total / row_count if row_count else 0.0
    predicted_value_accuracy = predicted_value_total / row_count if row_count else 0.0
    direct_strict_accuracy = direct_strict_total / row_count if row_count else 0.0
    predicted_strict_accuracy = predicted_strict_total / row_count if row_count else 0.0
    planner_macro = (
        sum(score["macro_planner_score"] for score in planner_scores) / len(planner_scores)
        if planner_scores
        else 0.0
    )
    return {
        "schema_version": 1,
        "artifact_type": "predicted_planner_sql_regression_analysis",
        "row_count": row_count,
        "metrics": {
            "bucket_counts": dict(sorted(bucket_counts.items())),
            "direct_sql_strict_execution_accuracy": direct_strict_accuracy,
            "direct_sql_syntax_valid_count": direct_syntax_valid,
            "direct_sql_value_execution_accuracy": direct_value_accuracy,
            "mean_planner_macro_score": planner_macro,
            "predicted_planner_strict_delta_vs_direct_sql": (
                predicted_strict_accuracy - direct_strict_accuracy
            ),
            "predicted_planner_strict_execution_accuracy": predicted_strict_accuracy,
            "predicted_planner_syntax_valid_count": predicted_syntax_valid,
            "predicted_planner_value_delta_vs_direct_sql": (
                predicted_value_accuracy - direct_value_accuracy
            ),
            "predicted_planner_value_execution_accuracy": predicted_value_accuracy,
            "regression_cause_counts": dict(sorted(regression_causes.items())),
        },
        "examples": examples,
        "interpretation": [
            "Corrected SQL extraction removes the local chat-continuation syntax artifact.",
            "Most direct-only regressions are projection-order flips after injecting predicted planner state.",
            "A positive value-accuracy delta is still required before Checkpoint 5 can promote the planner path.",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direct-output", type=Path, required=True)
    parser.add_argument("--predicted-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-examples", type=int, default=12)
    args = parser.parse_args(argv)

    payload = analyze_pair(
        direct_rows=_load_jsonl(args.direct_output),
        predicted_rows=_load_jsonl(args.predicted_output),
        max_examples=args.max_examples,
    )
    payload["inputs"] = {
        "direct_output": str(args.direct_output),
        "predicted_output": str(args.predicted_output),
    }
    _write_json(args.output, payload)
    print(f"Wrote predicted-planner regression analysis to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
