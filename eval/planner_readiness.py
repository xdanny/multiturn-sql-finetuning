"""Summarize planner risks before running predicted-planner SQL evaluation."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

PLANNER_PROMOTION_POLICY = {
    "minimum_row_count": 24,
    "maximum_parse_error_rate": 0.0,
    "maximum_macro_below_0_50_rate": 0.2,
    "minimum_mean_scores": {
        "macro_planner_score": 0.65,
        "table_f1": 0.70,
        "column_f1": 0.60,
        "skeleton_f1": 0.70,
        "selected_count_match": 0.70,
        "selected_expression_order_match": 0.90,
        "output_slot_order_match": 0.90,
        "duplicate_policy_match": 0.90,
    },
    "maximum_conditional_zero_rates": {
        "join_zero_when_gold_join_rate": 0.40,
        "group_by_zero_when_gold_group_by_rate": 0.40,
    },
}


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _score(row: dict[str, Any], key: str) -> float:
    return float((row.get("planner_scores") or {}).get(key) or 0.0)


def _gold_join_required(row: dict[str, Any]) -> bool:
    return bool((row.get("gold_plan") or {}).get("join_path") or [])


def _gold_group_by_required(row: dict[str, Any]) -> bool:
    projection = (row.get("gold_plan") or {}).get("projection_shape") or {}
    return bool(projection.get("group_by") or [])


def _predicted_projection_empty(row: dict[str, Any]) -> bool:
    projection = (row.get("predicted_plan") or {}).get("projection_shape") or {}
    return not bool(projection.get("selected_expressions") or projection.get("output_slots") or [])


def _rate(count: int, total: int) -> float:
    return count / total if total else 0.0


def _mean_score(rows: list[dict[str, Any]], key: str) -> float:
    return sum(_score(row, key) for row in rows) / len(rows) if rows else 0.0


def _risk_names(summary: dict[str, Any]) -> list[str]:
    risks = {
        "column_linking": summary["column_zero_count"],
        "projection_shape": summary["selected_count_mismatch_count"],
        "projection_order": summary["selected_expression_order_mismatch_count"],
        "output_slot_order": summary["output_slot_order_mismatch_count"],
        "empty_projection_expression": summary["empty_projection_expression_count"],
        "join_path": summary["join_zero_when_gold_join_count"],
        "group_by": summary["group_by_zero_when_gold_group_by_count"],
        "low_macro_score": summary["macro_below_0_50_count"],
    }
    return [
        name
        for name, _ in sorted(risks.items(), key=lambda item: (-int(item[1]), item[0]))
        if int(_) > 0
    ][:4]


def _promotion_blockers(summary: dict[str, Any]) -> list[str]:
    policy = summary["promotion_policy"]
    blockers = []
    if summary["row_count"] < policy["minimum_row_count"]:
        blockers.append(
            f"row_count below minimum {policy['minimum_row_count']}"
        )
    if summary["parse_error_rate"] > policy["maximum_parse_error_rate"]:
        blockers.append("planner parse errors present")
    if summary["macro_below_0_50_rate"] > policy["maximum_macro_below_0_50_rate"]:
        blockers.append("too many rows below macro planner score 0.50")
    for field, minimum in policy["minimum_mean_scores"].items():
        if summary["mean_scores"][field] < minimum:
            blockers.append(f"{field} below minimum {minimum:.2f}")
    for field, maximum in policy["maximum_conditional_zero_rates"].items():
        if summary[field] > maximum:
            blockers.append(f"{field} above maximum {maximum:.2f}")
    if not summary["endpoint_pair_ready"]:
        blockers.append("endpoint pair preflight is not ready")
    return blockers


def summarize_planner_readiness(
    rows: list[dict[str, Any]],
    *,
    preflight: dict[str, Any] | None = None,
) -> dict[str, Any]:
    total = len(rows)
    parse_error = sum(1 for row in rows if row.get("planner_parse_error"))
    macro_below = sum(1 for row in rows if _score(row, "macro_planner_score") < 0.5)
    column_zero = sum(1 for row in rows if _score(row, "column_f1") == 0.0)
    selected_mismatch = sum(1 for row in rows if _score(row, "selected_count_match") == 0.0)
    selected_order_mismatch = sum(
        1 for row in rows if _score(row, "selected_expression_order_match") == 0.0
    )
    output_slot_order_mismatch = sum(
        1 for row in rows if _score(row, "output_slot_order_match") == 0.0
    )
    empty_projection = sum(1 for row in rows if _predicted_projection_empty(row))
    join_zero = sum(
        1 for row in rows if _gold_join_required(row) and _score(row, "join_f1") == 0.0
    )
    group_zero = sum(
        1 for row in rows if _gold_group_by_required(row) and _score(row, "group_by_f1") == 0.0
    )
    source_counts = Counter(str(row.get("predicted_plan_source") or "unknown") for row in rows)
    summary = {
        "schema_version": 1,
        "artifact_type": "predicted_planner_risk_summary",
        "claim_boundary": "risk summary only; no SQL execution claim",
        "row_count": total,
        "dialog_count": len({row.get("dialog_id") for row in rows}),
        "database_count": len({row.get("database_id") for row in rows}),
        "prediction_sources": dict(source_counts),
        "preflight_status": (preflight or {}).get("status"),
        "endpoint_pair_ready": (preflight or {}).get("status") == "ready_for_endpoint_pair",
        "preflight_row_count": (preflight or {}).get("row_count"),
        "promotion_policy": PLANNER_PROMOTION_POLICY,
        "parse_error_count": parse_error,
        "parse_error_rate": _rate(parse_error, total),
        "macro_below_0_50_count": macro_below,
        "macro_below_0_50_rate": _rate(macro_below, total),
        "column_zero_count": column_zero,
        "column_zero_rate": _rate(column_zero, total),
        "selected_count_mismatch_count": selected_mismatch,
        "selected_count_mismatch_rate": _rate(selected_mismatch, total),
        "selected_expression_order_mismatch_count": selected_order_mismatch,
        "selected_expression_order_mismatch_rate": _rate(selected_order_mismatch, total),
        "output_slot_order_mismatch_count": output_slot_order_mismatch,
        "output_slot_order_mismatch_rate": _rate(output_slot_order_mismatch, total),
        "empty_projection_expression_count": empty_projection,
        "empty_projection_expression_rate": _rate(empty_projection, total),
        "join_zero_when_gold_join_count": join_zero,
        "join_zero_when_gold_join_rate": _rate(join_zero, total),
        "group_by_zero_when_gold_group_by_count": group_zero,
        "group_by_zero_when_gold_group_by_rate": _rate(group_zero, total),
        "mean_scores": {
            field: _mean_score(rows, field)
            for field in PLANNER_PROMOTION_POLICY["minimum_mean_scores"]
        },
    }
    summary["top_risks"] = _risk_names(summary)
    summary["promotion_blockers"] = _promotion_blockers(summary)
    summary["promotion_ready"] = not summary["promotion_blockers"]
    summary["recommendation"] = (
        "run_endpoint_pair" if summary["promotion_ready"] else "improve_planner_before_comparison"
    )
    return summary


def run_planner_readiness_report(
    *,
    planner_input: Path,
    preflight_input: Path | None,
    output: Path,
) -> int:
    rows = _load_jsonl(planner_input)
    preflight = _load_json(preflight_input) if preflight_input else None
    summary = summarize_planner_readiness(rows, preflight=preflight)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"Wrote planner risk summary to {output}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--planner-input",
        type=Path,
        default=Path("results/planner_eval_cosql_dev_100.jsonl"),
    )
    parser.add_argument(
        "--preflight-input",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/predicted_planner/planner_risk.json"),
    )
    args = parser.parse_args()
    return run_planner_readiness_report(
        planner_input=args.planner_input,
        preflight_input=args.preflight_input,
        output=args.output,
    )


if __name__ == "__main__":
    raise SystemExit(main())
