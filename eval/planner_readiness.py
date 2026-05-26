"""Summarize planner risks before running predicted-planner SQL evaluation."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


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
    return not bool(projection.get("selected_expressions") or [])


def _rate(count: int, total: int) -> float:
    return count / total if total else 0.0


def _risk_names(summary: dict[str, Any]) -> list[str]:
    risks = {
        "column_linking": summary["column_zero_count"],
        "projection_shape": summary["selected_count_mismatch_count"],
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


def summarize_planner_readiness(
    rows: list[dict[str, Any]],
    *,
    preflight: dict[str, Any] | None = None,
) -> dict[str, Any]:
    total = len(rows)
    macro_below = sum(1 for row in rows if _score(row, "macro_planner_score") < 0.5)
    column_zero = sum(1 for row in rows if _score(row, "column_f1") == 0.0)
    selected_mismatch = sum(1 for row in rows if _score(row, "selected_count_match") == 0.0)
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
        "artifact_type": "predicted_planner_readiness_report",
        "claim_boundary": "readiness only; no SQL execution claim",
        "row_count": total,
        "dialog_count": len({row.get("dialog_id") for row in rows}),
        "database_count": len({row.get("database_id") for row in rows}),
        "prediction_sources": dict(source_counts),
        "preflight_status": (preflight or {}).get("status"),
        "endpoint_pair_ready": (preflight or {}).get("status") == "ready_for_endpoint_pair",
        "preflight_row_count": (preflight or {}).get("row_count"),
        "macro_below_0_50_count": macro_below,
        "macro_below_0_50_rate": _rate(macro_below, total),
        "column_zero_count": column_zero,
        "column_zero_rate": _rate(column_zero, total),
        "selected_count_mismatch_count": selected_mismatch,
        "selected_count_mismatch_rate": _rate(selected_mismatch, total),
        "empty_projection_expression_count": empty_projection,
        "empty_projection_expression_rate": _rate(empty_projection, total),
        "join_zero_when_gold_join_count": join_zero,
        "join_zero_when_gold_join_rate": _rate(join_zero, total),
        "group_by_zero_when_gold_group_by_count": group_zero,
        "group_by_zero_when_gold_group_by_rate": _rate(group_zero, total),
    }
    summary["top_risks"] = _risk_names(summary)
    summary["recommendation"] = (
        "run_endpoint_pair"
        if summary["endpoint_pair_ready"] and summary["macro_below_0_50_rate"] <= 0.2
        else "improve_planner_before_claim"
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
    print(f"Wrote planner readiness report to {output}")
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
        default=Path("docs/predicted_planner_comparison_preflight.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/planner_readiness_cosql_dev_100.json"),
    )
    args = parser.parse_args()
    return run_planner_readiness_report(
        planner_input=args.planner_input,
        preflight_input=args.preflight_input,
        output=args.output,
    )


if __name__ == "__main__":
    raise SystemExit(main())
