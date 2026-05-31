from __future__ import annotations

import json

from eval.planner_readiness import (
    run_planner_readiness_report,
    summarize_planner_readiness,
)


def _row(
    *,
    macro: float,
    table: float = 1.0,
    column: float,
    skeleton: float = 1.0,
    selected_count: float,
    join: float,
    group_by: float,
    duplicate_policy: float = 1.0,
    gold_join: bool = True,
    gold_group_by: bool = True,
    planner_parse_error: str | None = None,
) -> dict:
    return {
        "dialog_id": "dialog-a",
        "database_id": "store",
        "predicted_plan_source": "lexical_schema_baseline",
        "gold_plan": {
            "join_path": ["orders.customer_id = customers.id"] if gold_join else [],
            "projection_shape": {
                "group_by": ["customers.country"] if gold_group_by else [],
            },
        },
        "predicted_plan": {
            "projection_shape": {
                "selected_expressions": [] if selected_count == 0 else ["customers.country"],
            },
        },
        "planner_parse_error": planner_parse_error,
        "planner_scores": {
            "macro_planner_score": macro,
            "table_f1": table,
            "column_f1": column,
            "skeleton_f1": skeleton,
            "selected_count_match": selected_count,
            "join_f1": join,
            "group_by_f1": group_by,
            "duplicate_policy_match": duplicate_policy,
        },
    }


def test_summarize_planner_readiness_counts_endpoint_risks() -> None:
    summary = summarize_planner_readiness(
        [
            _row(macro=0.80, column=0.75, selected_count=1.0, join=1.0, group_by=1.0),
            _row(macro=0.40, column=0.0, selected_count=0.0, join=0.0, group_by=0.0),
        ],
        preflight={
            "status": "ready_for_endpoint_pair",
            "row_count": 2,
            "dialog_count": 1,
            "database_count": 1,
        },
    )

    assert summary["artifact_type"] == "predicted_planner_risk_summary"
    assert summary["row_count"] == 2
    assert summary["preflight_status"] == "ready_for_endpoint_pair"
    assert summary["endpoint_pair_ready"] is True
    assert summary["promotion_ready"] is False
    assert summary["macro_below_0_50_count"] == 1
    assert summary["column_zero_count"] == 1
    assert summary["selected_count_mismatch_count"] == 1
    assert summary["empty_projection_expression_count"] == 1
    assert summary["join_zero_when_gold_join_count"] == 1
    assert summary["group_by_zero_when_gold_group_by_count"] == 1
    assert summary["recommendation"] == "improve_planner_before_comparison"
    assert summary["promotion_policy"]["minimum_row_count"] == 24
    assert "row_count below minimum 24" in summary["promotion_blockers"]
    assert "column_linking" in summary["top_risks"]


def test_summarize_planner_readiness_promotes_large_clean_planner_slice() -> None:
    summary = summarize_planner_readiness(
        [
            _row(
                macro=0.90,
                table=0.95,
                column=0.85,
                skeleton=0.90,
                selected_count=1.0,
                join=0.80,
                group_by=0.80,
                duplicate_policy=1.0,
            )
            for _ in range(24)
        ],
        preflight={
            "status": "ready_for_endpoint_pair",
            "row_count": 24,
            "dialog_count": 8,
            "database_count": 4,
        },
    )

    assert summary["promotion_ready"] is True
    assert summary["promotion_blockers"] == []
    assert summary["recommendation"] == "run_endpoint_pair"
    assert summary["mean_scores"]["macro_planner_score"] == 0.90


def test_summarize_planner_readiness_blocks_parse_errors() -> None:
    rows = [
        _row(
            macro=0.90,
            table=0.95,
            column=0.85,
            skeleton=0.90,
            selected_count=1.0,
            join=0.80,
            group_by=0.80,
            duplicate_policy=1.0,
        )
        for _ in range(23)
    ]
    rows.append(
        _row(
            macro=0.90,
            table=0.95,
            column=0.85,
            skeleton=0.90,
            selected_count=1.0,
            join=0.80,
            group_by=0.80,
            duplicate_policy=1.0,
            planner_parse_error="invalid JSON",
        )
    )

    summary = summarize_planner_readiness(
        rows,
        preflight={
            "status": "ready_for_endpoint_pair",
            "row_count": 24,
            "dialog_count": 8,
            "database_count": 4,
        },
    )

    assert summary["parse_error_count"] == 1
    assert summary["promotion_ready"] is False
    assert "planner parse errors present" in summary["promotion_blockers"]


def test_run_planner_readiness_report_writes_summary(tmp_path) -> None:
    planner_path = tmp_path / "planner.jsonl"
    preflight_path = tmp_path / "preflight.json"
    output_path = tmp_path / "readiness.json"
    planner_path.write_text(
        "\n".join(
            json.dumps(row)
            for row in [
                _row(macro=0.80, column=0.75, selected_count=1.0, join=1.0, group_by=1.0),
                _row(macro=0.40, column=0.0, selected_count=0.0, join=0.0, group_by=0.0),
            ]
        )
        + "\n"
    )
    preflight_path.write_text(
        json.dumps(
            {
                "status": "ready_for_endpoint_pair",
                "row_count": 2,
                "dialog_count": 1,
                "database_count": 1,
            }
        )
        + "\n"
    )

    assert (
        run_planner_readiness_report(
            planner_input=planner_path,
            preflight_input=preflight_path,
            output=output_path,
        )
        == 0
    )

    written = json.loads(output_path.read_text())
    assert written["row_count"] == 2
    assert written["claim_boundary"] == "risk summary only; no SQL execution claim"
